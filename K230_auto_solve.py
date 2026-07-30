"""
K230 任意版 — 自动识别碎片 + 求解矩形拼法
按键一次：检测4片 → 求解拼成什么矩形 → 输出每片目标位姿

求解思路：
  4片凸多边形拼成矩形，关键约束：
  1. 总面积 = 矩形面积
  2. 相邻片的接触边等长
  3. 矩形4条外边由各片的外边组成
  4. 所有内角要么是片的顶点角，要么拼合后=180度(共线)或90度(矩形角)

  简化策略（适合比赛4片场景）：
  - 收集所有边长，找成对出现的(内部边)
  - 不成对的边组成矩形外周
  - 用面积验证
"""

import gc
import math
import os
import time

import cv2
import image
from machine import FPIOA, Pin
from media.display import *
from media.media import *
from media.sensor import *

# -------------------- 参数 --------------------
CAMERA_W, CAMERA_H = 800, 480
IMG_W, IMG_H = 480, 800
VISION_SCALE = 2
VISION_W, VISION_H = IMG_W // VISION_SCALE, IMG_H // VISION_SCALE

DIVIDER_Y = VISION_H // 2
DIVIDER_LCD_Y = DIVIDER_Y * VISION_SCALE
BUTTON_PIN = 53

MIN_AREA = 150
MAX_AREA = 30000
APPROX_EPS_LIST = (0.008, 0.012, 0.018, 0.025, 0.035)
COLLINEAR_TOL_PX = 2.0
COLLINEAR_TOL_RATIO = 0.04
MIN_EDGE_ABS_PX = 4
MAX_VERTICES = 5
SNAPSHOT_TRIALS = 5

EDGE_MATCH_TOL = 0.08  # 边长匹配容差(比例)


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
        edges = [dist(pts[i], pts[(i+1)%n]) for i in range(n)]
        pieces.append({
            "cx": cx, "cy": cy, "area": area,
            "proc_pts": pts, "edges": edges,
            "lcd_cx": cx * VISION_SCALE, "lcd_cy": cy * VISION_SCALE,
            "lcd_pts": [(x*VISION_SCALE, y*VISION_SCALE) for x, y in pts],
        })
    return pieces


# -------------------- 求解器 --------------------
def edges_match(a, b):
    """两条边长是否匹配（相对误差 < EDGE_MATCH_TOL）"""
    avg = (a + b) / 2.0
    if avg < 1:
        return False
    return abs(a - b) / avg < EDGE_MATCH_TOL


def find_rect_dimensions(pieces):
    """从4片碎片推断目标矩形的宽和高(px)。
    方法：总面积 = W * H，外边组成矩形周长 = 2*(W+H)。
    外边 = 所有边中去掉内部配对边后剩余的。"""
    all_edges = []
    for idx, p in enumerate(pieces):
        for ei, e in enumerate(p["edges"]):
            all_edges.append((e, idx, ei))
    all_edges.sort(key=lambda x: x[0])

    # 找内部配对边：不同片之间边长相等的配对
    paired = set()
    for i in range(len(all_edges)):
        if i in paired:
            continue
        for j in range(i+1, len(all_edges)):
            if j in paired:
                continue
            if all_edges[i][1] == all_edges[j][1]:
                continue  # 同一片的边不配对
            if edges_match(all_edges[i][0], all_edges[j][0]):
                paired.add(i)
                paired.add(j)
                break

    # 外边 = 没被配对的
    outer_edges = [all_edges[i][0] for i in range(len(all_edges)) if i not in paired]

    total_area = sum(p["area"] for p in pieces)

    # 矩形外周 = 2*(W+H)，外边之和 ≈ 周长
    perimeter = sum(outer_edges)
    half_p = perimeter / 2.0

    # W + H = half_p, W * H = total_area
    # 解二次方程: x^2 - half_p*x + total_area = 0
    disc = half_p * half_p - 4 * total_area
    if disc < 0:
        # 配对可能不完美，用面积和最长外边估算
        if outer_edges:
            W = max(outer_edges)
            H = total_area / W if W > 0 else 0
        else:
            W = math.sqrt(total_area)
            H = W
    else:
        sqrt_disc = math.sqrt(disc)
        W = (half_p + sqrt_disc) / 2.0
        H = (half_p - sqrt_disc) / 2.0

    return W, H, outer_edges, paired, all_edges


def solve_puzzle(pieces):
    """求解4片拼成矩形的方案。
    返回: {"W": px, "H": px, "total_area": px2, "info": str}"""
    if len(pieces) != 4:
        return None

    W, H, outer, paired, all_e = find_rect_dimensions(pieces)
    total_area = sum(p["area"] for p in pieces)
    rect_area = W * H

    # 面积校验
    area_err = abs(rect_area - total_area) / total_area if total_area > 0 else 1
    info = "矩形: %.0f x %.0f px, 面积误差: %.1f%%" % (W, H, area_err*100)
    info += "\n内部配对边: %d对, 外边: %d条" % (len(paired)//2, len(outer))

    return {
        "W": W, "H": H,
        "total_area": total_area,
        "rect_area": rect_area,
        "area_err": area_err,
        "outer_edges": outer,
        "n_pairs": len(paired) // 2,
        "info": info,
    }


# -------------------- 选帧 --------------------
def choose_best_frame(frames):
    valid = [f for f in frames if f]
    if not valid:
        return []
    return max(valid, key=len)


# -------------------- 绘制 --------------------
def draw_pieces(frame, pieces):
    cv2.line(frame, (0, DIVIDER_LCD_Y), (IMG_W, DIVIDER_LCD_Y), (255,0,0), 2)
    for idx, p in enumerate(pieces):
        pts = p["lcd_pts"]
        n = len(pts)
        for i in range(n):
            cv2.line(frame, pts[i], pts[(i+1)%n], (0,255,0), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0,0,255), -1)
        cv2.putText(frame, "P%d n=%d" % (idx+1, n),
                    (p["lcd_cx"]-20, p["lcd_cy"]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)


def draw_rect_result(frame, result):
    """在屏幕下方显示求解出的矩形信息"""
    if result is None:
        cv2.putText(frame, "SOLVE FAILED", (8, DIVIDER_LCD_Y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        return
    W, H = result["W"], result["H"]
    # 在下半区中央画目标矩形轮廓
    rx = IMG_W // 2 - int(W) // 2
    ry = DIVIDER_LCD_Y + 20
    rw, rh = int(W), int(H)
    if rw > 0 and rh > 0 and rw < IMG_W and rh < (IMG_H - DIVIDER_LCD_Y - 40):
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0,255,255), 2)
        cv2.putText(frame, "%.0fx%.0f" % (W, H), (rx+5, ry+rh+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
    # 状态文字
    txt = "RECT %.0fx%.0f err=%.1f%% pairs=%d" % (
        W, H, result["area_err"]*100, result["n_pairs"])
    cv2.putText(frame, txt, (8, IMG_H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)


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

    is_frozen = False
    last_btn_val = 0
    frame_cnt = 0
    frozen_done = False

    print("=== K230 拼图(任意版) 就绪 ===")
    print("按键 = 检测 + 自动求解矩形")

    try:
        while True:
            os.exitpoint()
            frame_cnt += 1
            btn_val = button.value()
            if btn_val == 1 and last_btn_val == 0:
                time.sleep_ms(50)
                if button.value() == 1:
                    is_frozen = not is_frozen
                    if is_frozen:
                        frozen_done = False
                    else:
                        gc.collect()
                    print("MODE -> %s" % ("SOLVE" if is_frozen else "LIVE"))
            last_btn_val = btn_val

            if is_frozen:
                if not frozen_done:
                    gc.collect()
                    frames = []
                    for _ in range(SNAPSHOT_TRIALS):
                        raw = sensor.snapshot()
                        np_ref = raw.to_numpy_ref()
                        rot = cv2.rotate(np_ref, cv2.ROTATE_90_CLOCKWISE)
                        small = cv2.resize(rot, (VISION_W, VISION_H))
                        frames.append(extract_pieces(small))
                        del small, rot, np_ref, raw
                        time.sleep_ms(50)

                    pieces = choose_best_frame(frames)
                    print("DETECTED %d pieces" % len(pieces))
                    for idx, p in enumerate(pieces):
                        el = ["%dpx" % int(e) for e in p["edges"]]
                        print("  P%d n=%d edges=%s" % (
                            idx+1, len(p["proc_pts"]), el))

                    result = solve_puzzle(pieces)
                    if result:
                        print("=== SOLVE ===")
                        print(result["info"])
                    else:
                        print("求解失败（检测到 %d 片）" % len(pieces))

                    # 显示
                    raw = sensor.snapshot()
                    np_ref = raw.to_numpy_ref()
                    display_frame = cv2.rotate(np_ref, cv2.ROTATE_90_CLOCKWISE)
                    draw_pieces(display_frame, pieces)
                    draw_rect_result(display_frame, result)
                    disp_img = image.Image(
                        IMG_W, IMG_H, image.RGB888,
                        alloc=image.ALLOC_REF, data=display_frame)
                    Display.show_image(disp_img)
                    frozen_done = True
                    del np_ref, raw
                    gc.collect()
                else:
                    time.sleep_ms(120)
            else:
                raw = sensor.snapshot()
                np_ref = raw.to_numpy_ref()
                rot = cv2.rotate(np_ref, cv2.ROTATE_90_CLOCKWISE)
                img = image.Image(IMG_W, IMG_H, image.RGB888,
                                  alloc=image.ALLOC_REF, data=rot)
                img.draw_line(0, DIVIDER_LCD_Y, IMG_W, DIVIDER_LCD_Y,
                              color=(255,0,0), thickness=2)
                img.draw_string_advanced(10, 10, 20,
                    "AUTO SOLVE (press key)", color=(255,255,255))
                Display.show_image(img)
                del img, rot, np_ref, raw
                if frame_cnt % 5 == 0:
                    gc.collect()
    except KeyboardInterrupt:
        print("停止")
    finally:
        try:
            sensor.stop()
        except Exception:
            pass
        try:
            Display.deinit()
        except Exception:
            pass
        try:
            MediaManager.deinit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
