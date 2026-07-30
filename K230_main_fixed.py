"""
K230 写死版主程序
识别上半区碎片 → 按边数+边长匹配模板 → 计算目标位姿 → UART 发送
"""

import gc
import math
import time

import cv2
import image
from machine import FPIOA, Pin
from media.display import *
from media.media import *
from media.sensor import *

from config_pieces import PIECES, TARGETS, PIXELS_PER_CM
from uart_protocol import Protocol

# -------------------- 硬件参数 --------------------
CAMERA_W, CAMERA_H = 800, 480
IMG_W, IMG_H = 480, 800
VISION_SCALE = 2
VISION_W, VISION_H = IMG_W // VISION_SCALE, IMG_H // VISION_SCALE

DIVIDER_Y = VISION_H // 2
DIVIDER_LCD_Y = DIVIDER_Y * VISION_SCALE
BUTTON_PIN = 53

# -------------------- 提取参数 --------------------
MIN_AREA = 150
MAX_AREA = 30000
APPROX_EPS_LIST = (0.008, 0.012, 0.018, 0.025, 0.035)
COLLINEAR_TOL_PX = 2.0
COLLINEAR_TOL_RATIO = 0.04
MIN_EDGE_ABS_PX = 4
MAX_VERTICES = 5
SNAPSHOT_TRIALS = 5
MATCH_TOL_PX = 20


# -------------------- 工具函数 --------------------
def dist(a, b):
    return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)


def parse_poly_points(approx):
    pts = []
    try:
        flat = approx.flatten()
        for i in range(0, len(flat) - 1, 2):
            pts.append((int(flat[i]), int(flat[i + 1])))
        if len(pts) >= 3:
            return pts
    except Exception:
        pass
    try:
        for pt in approx:
            while hasattr(pt, "__len__") and len(pt) == 1:
                pt = pt[0]
            if hasattr(pt, "__len__") and len(pt) >= 2:
                pts.append((int(pt[0]), int(pt[1])))
    except Exception:
        pass
    return pts


def point_line_dist(p, a, c):
    dx, dy = c[0]-a[0], c[1]-a[1]
    length = math.sqrt(dx*dx + dy*dy)
    if length < 1e-6:
        return dist(p, a)
    return abs((p[0]-a[0])*dy - (p[1]-a[1])*dx) / length


def merge_collinear(pts, tol_px, tol_ratio):
    pts = list(pts)
    while len(pts) > 3:
        n = len(pts)
        best_i, best_d, best_ac = -1, float("inf"), 1.0
        for i in range(n):
            a, b, c = pts[(i-1)%n], pts[i], pts[(i+1)%n]
            ac = dist(a, c)
            if ac < 1e-6:
                continue
            d = point_line_dist(b, a, c)
            if d < best_d:
                best_d, best_i, best_ac = d, i, ac
        if best_i < 0:
            break
        if best_d < max(tol_px, best_ac * tol_ratio):
            del pts[best_i]
        else:
            break
    return pts


def merge_short_vertices(pts, min_px):
    pts = list(pts)
    while len(pts) > 3:
        n = len(pts)
        best_i, best_d = -1, float("inf")
        for i in range(n):
            d = dist(pts[i], pts[(i+1)%n])
            if d < best_d:
                best_d, best_i = d, i
        if best_d >= min_px:
            break
        a, b = pts[best_i], pts[(best_i+1)%n]
        pts[best_i] = ((a[0]+b[0])//2, (a[1]+b[1])//2)
        del pts[(best_i+1)%n]
    return pts


# -------------------- 碎片提取 --------------------
def extract_pieces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if cy >= DIVIDER_Y:
            continue
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        pts = []
        for eps_r in APPROX_EPS_LIST:
            approx = cv2.approxPolyDP(hull, peri * eps_r, True)
            cand = parse_poly_points(approx)
            if len(cand) < 3:
                continue
            cand = merge_collinear(cand, COLLINEAR_TOL_PX, COLLINEAR_TOL_RATIO)
            cand = merge_short_vertices(cand, MIN_EDGE_ABS_PX)
            if 3 <= len(cand) <= MAX_VERTICES:
                pts = cand
                break
        if len(pts) < 3:
            approx = cv2.approxPolyDP(hull, peri * 0.03, True)
            cand = parse_poly_points(approx)
            cand = merge_collinear(cand, COLLINEAR_TOL_PX, COLLINEAR_TOL_RATIO)
            cand = merge_short_vertices(cand, MIN_EDGE_ABS_PX)
            if 3 <= len(cand) <= MAX_VERTICES:
                pts = cand
        if len(pts) < 3:
            continue
        n = len(pts)
        edges = sorted(
            [dist(pts[i], pts[(i+1)%n]) for i in range(n)], reverse=True)
        pieces.append({
            "cx": cx, "cy": cy,
            "proc_pts": pts, "edges_sorted": edges,
            "n_verts": n, "area": area,
        })
    return pieces


# -------------------- 模板匹配 --------------------
def match_piece(detected, templates):
    """用边数+边长向量匹配。返回最佳模板id或None"""
    n = detected["n_verts"]
    edges = detected["edges_sorted"]
    best_id, best_err = None, float("inf")
    for t in templates:
        if t["n_verts"] != n:
            continue
        t_edges = t["edges_sorted"]
        if len(edges) != len(t_edges):
            continue
        err = sum(abs(a - b) for a, b in zip(edges, t_edges))
        if err < best_err:
            best_err = err
            best_id = t["id"]
    if best_err < MATCH_TOL_PX * n:
        return best_id
    return None


def compute_angle(pts):
    """用最长边的方向作为碎片朝向(deg, 0~360)"""
    n = len(pts)
    best_i, best_d = 0, 0
    for i in range(n):
        d = dist(pts[i], pts[(i+1)%n])
        if d > best_d:
            best_d = d
            best_i = i
    a, b = pts[best_i], pts[(best_i+1)%n]
    ang = math.atan2(b[1]-a[1], b[0]-a[0])
    return math.degrees(ang) % 360


def match_all(detected_list):
    """对检测到的碎片逐个匹配，返回 {id: detected_piece}"""
    used_ids = set()
    result = {}
    for det in detected_list:
        mid = match_piece(det, PIECES)
        if mid is not None and mid not in used_ids:
            used_ids.add(mid)
            result[mid] = det
    return result


# -------------------- 构建移动指令 --------------------
def build_moves(matched):
    """matched: {id: detected_piece}。生成UART移动指令列表"""
    moves = []
    for tgt in TARGETS:
        pid = tgt["id"]
        if pid not in matched:
            continue
        det = matched[pid]
        cur_ang = compute_angle(det["proc_pts"])
        cur_x_mm = int(det["cx"] * 10 / (PIXELS_PER_CM or 1))
        cur_y_mm = int(det["cy"] * 10 / (PIXELS_PER_CM or 1))
        tgt_x_mm = int(tgt["cx"] * 10)
        tgt_y_mm = int(tgt["cy"] * 10)
        moves.append({
            "id": pid,
            "cur_x": cur_x_mm, "cur_y": cur_y_mm,
            "cur_ang": int(cur_ang * 10) % 3600,
            "tgt_x": tgt_x_mm, "tgt_y": tgt_y_mm,
            "tgt_ang": int(tgt["angle"] * 10) % 3600,
        })
    return moves


# -------------------- 调试绘制 --------------------
def draw_result(frame, matched):
    cv2.line(frame, (0, DIVIDER_LCD_Y), (IMG_W, DIVIDER_LCD_Y), (255, 0, 0), 2)
    for pid, det in matched.items():
        pts = [(x*VISION_SCALE, y*VISION_SCALE) for x, y in det["proc_pts"]]
        n = len(pts)
        for i in range(n):
            cv2.line(frame, pts[i], pts[(i+1)%n], (0, 255, 0), 2)
        lcx, lcy = det["cx"]*VISION_SCALE, det["cy"]*VISION_SCALE
        cv2.putText(frame, "ID%d" % pid, (lcx-10, lcy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


# -------------------- 主程序 --------------------
def main():
    fpioa = FPIOA()
    fpioa.set_function(BUTTON_PIN, getattr(FPIOA, "GPIO%d" % BUTTON_PIN))
    button = Pin(BUTTON_PIN, Pin.IN, Pin.PULL_DOWN)

    sensor = Sensor(id=2, width=CAMERA_W, height=CAMERA_H, fps=15)
    sensor.reset()
    sensor.set_framesize(width=CAMERA_W, height=CAMERA_H)
    sensor.set_pixformat(Sensor.RGB888)
    Display.init(Display.ST7701, width=IMG_W, height=IMG_H, to_ide=False)
    MediaManager.init()
    sensor.run()

    proto = Protocol()
    frozen = False
    sent = False

    try:
        while True:
            gc.collect()
            if button.value() == 1:
                time.sleep_ms(200)
                frozen = not frozen
                sent = False

            if not frozen:
                img = sensor.snapshot()
                raw = img.to_rgb888()
                frame = raw.to_numpy_ref()
                small = cv2.resize(frame, (VISION_W, VISION_H))
                pieces = extract_pieces(small)
                rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                info = "LIVE pieces=%d" % len(pieces)
                cv2.putText(rotated, info, (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                Display.show_image(image.Image.from_numpy_ref(rotated))
            else:
                if not sent:
                    frames = []
                    for _ in range(SNAPSHOT_TRIALS):
                        img = sensor.snapshot()
                        raw = img.to_rgb888()
                        frame = raw.to_numpy_ref()
                        small = cv2.resize(frame, (VISION_W, VISION_H))
                        frames.append(extract_pieces(small))
                        time.sleep_ms(50)
                    best = max(frames, key=len)
                    matched = match_all(best)
                    moves = build_moves(matched)
                    if moves:
                        proto.send_all(moves)
                    rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    draw_result(rotated, matched)
                    n_ok = len(matched)
                    cv2.putText(rotated,
                                "MATCH %d/4 SENT" % n_ok, (8, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
                    Display.show_image(image.Image.from_numpy_ref(rotated))
                    sent = True
    except KeyboardInterrupt:
        pass
    finally:
        sensor.stop()
        Display.deinit()
        MediaManager.deinit()


if __name__ == "__main__":
    main()