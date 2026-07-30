"""
K230 任意版 — 碎片形状未知，自动识别+求解拼成矩形
按键一次：检测4片 → 边匹配求解 → LCD显示拼接结果
算法参考 puzzle-vision-simulator 的 align_edge + 组合搜索
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

EDGE_TOL = 0.20
MAX_CANDIDATES = 20
MAX_COMBOS = 150


# -------------------- 工具函数 --------------------
def dist(a, b):
    return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

def parse_poly_points(approx):
    pts = []
    try:
        flat = approx.flatten()
        for i in range(0, len(flat)-1, 2):
            pts.append((int(flat[i]), int(flat[i+1])))
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

def poly_area(pts):
    n = len(pts)
    a = 0
    for i in range(n):
        j = (i+1) % n
        a += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(a) / 2.0


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
            continue
        n = len(pts)
        edges = [dist(pts[i], pts[(i+1)%n]) for i in range(n)]
        pieces.append({
            "cx": cx, "cy": cy, "area": area,
            "pts": pts, "edges": edges,
        })
    return pieces


# -------------------- 求解核心 --------------------
def align_edge(src_pts, src_ei, dst_pts, dst_ei):
    """刚体变换：src的边src_ei反向对齐到dst的边dst_ei"""
    ns, nd = len(src_pts), len(dst_pts)
    sa, sb = src_pts[src_ei], src_pts[(src_ei+1)%ns]
    da, db = dst_pts[dst_ei], dst_pts[(dst_ei+1)%nd]
    src_ang = math.atan2(sb[1]-sa[1], sb[0]-sa[0])
    dst_ang = math.atan2(da[1]-db[1], da[0]-db[0])
    rot = dst_ang - src_ang
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    rotated = []
    for x, y in src_pts:
        dx, dy = x - sa[0], y - sa[1]
        rotated.append((dx*cos_r - dy*sin_r + sa[0],
                        dx*sin_r + dy*cos_r + sa[1]))
    ra = rotated[src_ei]
    tx, ty = db[0] - ra[0], db[1] - ra[1]
    return [(x+tx, y+ty) for x, y in rotated]


def find_candidates(pieces):
    """找所有边长相近的不同片边配对"""
    cands = []
    n = len(pieces)
    for i in range(n):
        ei = pieces[i]["edges"]
        for j in range(i+1, n):
            ej = pieces[j]["edges"]
            for ai in range(len(ei)):
                for bi in range(len(ej)):
                    avg = (ei[ai] + ej[bi]) / 2
                    if avg < 8:
                        continue
                    if abs(ei[ai] - ej[bi]) / avg < EDGE_TOL:
                        cands.append((i, j, ai, bi, avg))
    cands.sort(key=lambda x: -x[4])
    return cands[:MAX_CANDIDATES]


def is_connected(edges, n_pieces):
    """检查边集合是否连通所有片"""
    adj = [[] for _ in range(n_pieces)]
    for i, j, _, _, _ in edges:
        adj[i].append(j)
        adj[j].append(i)
    visited = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for nb in adj[node]:
            if nb not in visited:
                stack.append(nb)
    return len(visited) == n_pieces


def assemble(pieces, matchings):
    """用一组匹配拼合所有片，返回 placed 列表"""
    n = len(pieces)
    placed = [None] * n
    placed[0] = pieces[0]["pts"]
    done = {0}
    changed = True
    while changed:
        changed = False
        for pi, pj, ei, ej, _ in matchings:
            if pi in done and pj not in done:
                placed[pj] = align_edge(
                    pieces[pj]["pts"], ej, placed[pi], ei)
                done.add(pj)
                changed = True
            elif pj in done and pi not in done:
                placed[pi] = align_edge(
                    pieces[pi]["pts"], ei, placed[pj], ej)
                done.add(pi)
                changed = True
    if len(done) < n:
        return None
    return placed


def score(placed):
    """评分：fill=碎片总面积/外框面积，越接近1.0越好"""
    all_pts = []
    for pts in placed:
        all_pts.extend(pts)
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)
    if bw < 1 or bh < 1:
        return 0
    bbox_area = bw * bh
    piece_area = sum(poly_area(pts) for pts in placed)
    fill = piece_area / bbox_area
    # 完美拼合 fill=1.0，重叠>1或间隙<1都扣分
    s = 1.0 - abs(1.0 - fill)
    # 宽高比不合理也扣分
    ratio = max(bw, bh) / min(bw, bh)
    if ratio > 3.0:
        s *= 0.5
    return s


def converge(placed):
    """收束：将碎片向组中心等比缩放，提升填充率"""
    n = len(placed)
    # 计算各片质心
    centers = []
    for pts in placed:
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        centers.append((cx, cy))
    gcx = sum(c[0] for c in centers) / n
    gcy = sum(c[1] for c in centers) / n
    best_placed = placed
    best_s = score(placed)
    # 逐步收缩 0.95, 0.90, ..., 0.50
    for shrink_pct in range(95, 45, -5):
        shrink = shrink_pct / 100.0
        new_placed = []
        for i in range(n):
            cx, cy = centers[i]
            ncx = gcx + (cx - gcx) * shrink
            ncy = gcy + (cy - gcy) * shrink
            dx, dy = ncx - cx, ncy - cy
            new_placed.append([(x+dx, y+dy) for x, y in placed[i]])
        s = score(new_placed)
        if s > best_s:
            best_s = s
            best_placed = new_placed
    return best_placed, best_s


def solve(pieces):
    """求解4片拼成矩形的最佳方案"""
    if len(pieces) != 4:
        return None
    cands = find_candidates(pieces)
    n = len(pieces)
    best_placed = None
    best_score = 0
    tried = 0
    nc = len(cands)
    # 需要3条边连通4片，枚举C(nc,3)中有效的
    for i in range(nc):
        for j in range(i+1, nc):
            for k in range(j+1, nc):
                combo = [cands[i], cands[j], cands[k]]
                if not is_connected(combo, n):
                    continue
                # 检查同一片的同一条边不能用两次
                used_edges = set()
                valid = True
                for pi, pj, ei, ej, _ in combo:
                    key_a = (pi, ei)
                    key_b = (pj, ej)
                    if key_a in used_edges or key_b in used_edges:
                        valid = False
                        break
                    used_edges.add(key_a)
                    used_edges.add(key_b)
                if not valid:
                    continue
                placed = assemble(pieces, combo)
                if placed is None:
                    continue
                s = score(placed)
                if s > best_score:
                    best_score = s
                    best_placed = placed
                tried += 1
                if tried >= MAX_COMBOS:
                    return best_placed, best_score
    return best_placed, best_score


# -------------------- 显示 --------------------
def draw_solution(frame, placed):
    all_pts = []
    for pts in placed:
        all_pts.extend(pts)
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    sw, sh = max_x - min_x, max_y - min_y
    if sw < 1 or sh < 1:
        return
    disp_y = DIVIDER_LCD_Y + 25
    disp_w = IMG_W - 60
    disp_h = IMG_H - DIVIDER_LCD_Y - 55
    scale = min(disp_w / sw, disp_h / sh) * 0.85
    off_x = 30 + (disp_w - sw * scale) / 2
    off_y = disp_y + (disp_h - sh * scale) / 2
    def to_lcd(px, py):
        return (int((px-min_x)*scale+off_x), int((py-min_y)*scale+off_y))
    for idx, pts in enumerate(placed):
        lcd_pts = [to_lcd(x, y) for x, y in pts]
        n = len(lcd_pts)
        for i in range(n):
            cv2.line(frame, lcd_pts[i], lcd_pts[(i+1)%n], (0,128,255), 2)
        cx = sum(p[0] for p in lcd_pts) // n
        cy = sum(p[1] for p in lcd_pts) // n
        cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)
    all_lcd = [to_lcd(x, y) for pts in placed for x, y in pts]
    lx = [p[0] for p in all_lcd]
    ly = [p[1] for p in all_lcd]
    cv2.rectangle(frame, (min(lx)-4, min(ly)-4),
                  (max(lx)+4, max(ly)+4), (0,255,255), 2)


def draw_pieces_upper(frame, pieces):
    cv2.line(frame, (0, DIVIDER_LCD_Y), (IMG_W, DIVIDER_LCD_Y), (255,0,0), 2)
    for idx, p in enumerate(pieces):
        pts = [(x*VISION_SCALE, y*VISION_SCALE) for x, y in p["pts"]]
        n = len(pts)
        for i in range(n):
            cv2.line(frame, pts[i], pts[(i+1)%n], (0,255,0), 2)
        cx = p["cx"] * VISION_SCALE
        cy = p["cy"] * VISION_SCALE
        cv2.putText(frame, "P%d" % (idx+1), (cx-10, cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)


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
    frozen_done = False
    frame_cnt = 0
    print("=== K230 拼图(任意版) 就绪 ===")
    print("按键 = 检测 + 自动求解矩形拼法")
    try:
        while True:
            os.exitpoint()
            frame_cnt += 1
            btn = button.value()
            if btn == 1:
                time.sleep_ms(50)
                if button.value() == 1:
                    is_frozen = not is_frozen
                    frozen_done = False
                    if not is_frozen:
                        gc.collect()
                    while button.value() == 1:
                        time.sleep_ms(10)
            if is_frozen and not frozen_done:
                gc.collect()
                best_pieces = []
                for _ in range(SNAPSHOT_TRIALS):
                    raw = sensor.snapshot()
                    np_ref = raw.to_numpy_ref()
                    rot = cv2.rotate(np_ref, cv2.ROTATE_90_CLOCKWISE)
                    small = cv2.resize(rot, (VISION_W, VISION_H))
                    p = extract_pieces(small)
                    if len(p) > len(best_pieces):
                        best_pieces = p
                    del small, rot, np_ref, raw
                    time.sleep_ms(30)
                pieces = best_pieces
                print("DETECTED %d pieces" % len(pieces))
                for i, p in enumerate(pieces):
                    el = ["%dpx" % int(e) for e in p["edges"]]
                    print("  P%d n=%d edges=%s" % (i+1, len(p["pts"]), el))
                result = None
                if len(pieces) == 4:
                    t0 = time.ticks_ms()
                    result = solve(pieces)
                    dt = time.ticks_diff(time.ticks_ms(), t0)
                    if result and result[0]:
                        print("SOLVED! score=%.0f%% time=%dms" % (
                            result[1]*100, dt))
                    else:
                        print("SOLVE FAILED (%dms)" % dt)
                        result = None
                raw = sensor.snapshot()
                np_ref = raw.to_numpy_ref()
                disp = cv2.rotate(np_ref, cv2.ROTATE_90_CLOCKWISE)
                draw_pieces_upper(disp, pieces)
                if result and result[0]:
                    try:
                        draw_solution(disp, result[0])
                        cv2.putText(disp, "SCORE %.0f%%" % (result[1]*100),
                                    (8, IMG_H-12), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0,255,0), 1)
                    except Exception as e:
                        print("draw err:", e)
                else:
                    cv2.putText(disp, "SOLVE FAILED", (8, IMG_H-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                img = image.Image(IMG_W, IMG_H, image.RGB888,
                                  alloc=image.ALLOC_REF, data=disp)
                Display.show_image(img)
                frozen_done = True
                del np_ref, raw
                gc.collect()
            elif is_frozen:
                time.sleep_ms(100)
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
        pass
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
