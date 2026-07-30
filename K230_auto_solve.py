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

DIVIDER_Y = IMG_H // 2  # 400: 上下分界
BUTTON_PIN = 53

MIN_AREA = 600
MAX_AREA = 120000
APPROX_EPS_LIST = (0.008, 0.012, 0.018, 0.025, 0.035)
COLLINEAR_TOL_PX = 2.0
COLLINEAR_TOL_RATIO = 0.04
MIN_EDGE_ABS_PX = 10
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
        # 统一为逆时针方向
        n = len(pts)
        signed_area = 0
        for i in range(n):
            j = (i + 1) % n
            signed_area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
        if signed_area > 0:  # 顺时针则反转
            pts = pts[::-1]
        n = len(pts)
        edges = [dist(pts[i], pts[(i+1)%n]) for i in range(n)]
        pieces.append({
            "cx": cx, "cy": cy, "area": area,
            "pts": pts, "edges": edges,
        })
    return pieces


# -------------------- 求解核心 --------------------
def align_edge(src_pts, src_ei, dst_pts, dst_ei):
    """刚体变换：src边反向平行于dst边，中点对齐"""
    ns, nd = len(src_pts), len(dst_pts)
    sa, sb = src_pts[src_ei], src_pts[(src_ei+1)%ns]
    da, db = dst_pts[dst_ei], dst_pts[(dst_ei+1)%nd]
    # 旋转：src边方向 → dst边反方向
    src_ang = math.atan2(sb[1]-sa[1], sb[0]-sa[0])
    dst_ang = math.atan2(da[1]-db[1], da[0]-db[0])
    rot = dst_ang - src_ang
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    # 以src边中点为旋转中心
    smx, smy = (sa[0]+sb[0])/2, (sa[1]+sb[1])/2
    rotated = []
    for x, y in src_pts:
        dx, dy = x - smx, y - smy
        rotated.append((dx*cos_r - dy*sin_r + smx,
                        dx*sin_r + dy*cos_r + smy))
    # 平移：旋转后的src中点 → dst边中点
    r_smx = (rotated[src_ei][0] + rotated[(src_ei+1)%ns][0]) / 2
    r_smy = (rotated[src_ei][1] + rotated[(src_ei+1)%ns][1]) / 2
    dmx, dmy = (da[0]+db[0])/2, (da[1]+db[1])/2
    tx, ty = dmx - r_smx, dmy - r_smy
    return [(x+tx, y+ty) for x, y in rotated]


def find_candidates(pieces):
    """找所有边长相近的不同片边配对，按匹配精度排序"""
    cands = []
    n = len(pieces)
    for i in range(n):
        ei = pieces[i]["edges"]
        for j in range(i+1, n):
            ej = pieces[j]["edges"]
            for ai in range(len(ei)):
                for bi in range(len(ej)):
                    avg = (ei[ai] + ej[bi]) / 2
                    if avg < 15:
                        continue
                    err = abs(ei[ai] - ej[bi]) / avg
                    if err < EDGE_TOL:
                        cands.append((i, j, ai, bi, err))
    # 误差最小的排前面（最可能是真正的内部切割边）
    cands.sort(key=lambda x: x[4])
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



def convex_hull(points):
    """Jarvis march求凸包"""
    pts = []
    seen = set()
    for p in points:
        key = (int(p[0]*10), int(p[1]*10))
        if key not in seen:
            seen.add(key)
            pts.append(p)
    n = len(pts)
    if n < 3:
        return pts
    start = min(range(n), key=lambda i: (pts[i][0], pts[i][1]))
    hull = []
    current = start
    for _ in range(n + 1):
        hull.append(pts[current])
        nxt = 0
        for i in range(n):
            if i == current:
                continue
            ox, oy = pts[current]
            ax, ay = pts[nxt]
            bx, by = pts[i]
            cross = (ax-ox)*(by-oy) - (ay-oy)*(bx-ox)
            if nxt == current or cross < 0:
                nxt = i
            elif cross == 0:
                da = (ax-ox)**2 + (ay-oy)**2
                db = (bx-ox)**2 + (by-oy)**2
                if db > da:
                    nxt = i
        current = nxt
        if current == start:
            break
    return hull


def score(placed):
    """评分：fill接近1 + 凸包必须是矩形"""
    all_pts = []
    for pts in placed:
        all_pts.extend(pts)
    if len(all_pts) < 3:
        return 0
    piece_area = sum(poly_area(pts) for pts in placed)
    hull = convex_hull(all_pts)
    hull_area = poly_area(hull)
    if hull_area < 1:
        return 0
    fill = piece_area / hull_area
    if fill > 1.0:
        fill_score = max(0, 1.0 - (fill - 1.0) * 3)
    else:
        fill_score = fill

    # 去共线点简化凸包(容忍检测噪声)
    n = len(hull)
    simple = []
    for i in range(n):
        a = hull[(i-1) % n]
        b = hull[i]
        c = hull[(i+1) % n]
        ac = dist(a, c)
        if ac < 1:
            continue
        # b到ac连线的距离 / ac长度 > 8%才保留
        d = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) / ac
        if d < ac * 0.08:
            continue
        simple.append(b)
    ns = len(simple)
    if ns == 4:
        # 检查4个角是否接近90度
        angle_err = 0
        for i in range(4):
            a = simple[(i-1) % 4]
            b = simple[i]
            c = simple[(i+1) % 4]
            ba = (a[0]-b[0], a[1]-b[1])
            bc = (c[0]-b[0], c[1]-b[1])
            dot = ba[0]*bc[0] + ba[1]*bc[1]
            mag = math.sqrt(ba[0]**2+ba[1]**2) * math.sqrt(bc[0]**2+bc[1]**2)
            if mag < 1:
                angle_err += 90
                continue
            cos_a = max(-1, min(1, dot / mag))
            ang = math.degrees(math.acos(cos_a))
            angle_err += abs(ang - 90)
        # 更宽松：平均偏差30°以内都可接受
        rect_score = max(0, 1.0 - angle_err / 200)
        # 宽高比必须接近5:3(=1.67)
        edges_s = []
        for i in range(4):
            edges_s.append(dist(simple[i], simple[(i+1)%4]))
        edges_s.sort()
        short = (edges_s[0] + edges_s[1]) / 2
        long = (edges_s[2] + edges_s[3]) / 2
        if short > 0:
            aspect = long / short
            # 目标1.67，容忍±0.4
            aspect_err = abs(aspect - 1.67)
            aspect_score = max(0, 1.0 - aspect_err / 0.6)
        else:
            aspect_score = 0
        rect_score *= aspect_score
    elif ns == 5:
        rect_score = 0.4
    elif ns == 3:
        rect_score = 0.3
    else:
        rect_score = 0.15
    return fill_score * rect_score


def solve(pieces):
    """边配对枚举求解"""
    if len(pieces) != 4:
        return None, 0
    cands = find_candidates(pieces)
    n = len(pieces)
    best_placed = None
    best_score = 0
    tried = 0
    nc = len(cands)
    for i in range(nc):
        for j in range(i+1, nc):
            for k in range(j+1, nc):
                combo = [cands[i], cands[j], cands[k]]
                if not is_connected(combo, n):
                    continue
                used_edges = set()
                valid = True
                for pi, pj, ei, ej, _ in combo:
                    ka = (pi, ei)
                    kb = (pj, ej)
                    if ka in used_edges or kb in used_edges:
                        valid = False
                        break
                    used_edges.add(ka)
                    used_edges.add(kb)
                if not valid:
                    continue
                placed = assemble(pieces, combo)
                if placed is None:
                    continue
                s = score(placed)
                if s > best_score:
                    best_score = s
                    best_placed = placed
                    print("  combo[%d]: P%d-e%d~P%d-e%d, P%d-e%d~P%d-e%d, P%d-e%d~P%d-e%d => %.0f%%" % (
                        tried,
                        combo[0][0]+1, combo[0][2], combo[0][1]+1, combo[0][3],
                        combo[1][0]+1, combo[1][2], combo[1][1]+1, combo[1][3],
                        combo[2][0]+1, combo[2][2], combo[2][1]+1, combo[2][3],
                        s*100))
                tried += 1
                if tried >= MAX_COMBOS:
                    break
            if tried >= MAX_COMBOS:
                break
        if tried >= MAX_COMBOS:
            break
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
    disp_y = DIVIDER_Y + 25
    disp_w = IMG_W - 60
    disp_h = IMG_H - DIVIDER_Y - 55
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
    cv2.line(frame, (0, DIVIDER_Y), (IMG_W, DIVIDER_Y), (255,0,0), 2)
    for idx, p in enumerate(pieces):
        pts = p["pts"]
        n = len(pts)
        for i in range(n):
            cv2.line(frame, pts[i], pts[(i+1)%n], (0,255,0), 2)
        cx = p["cx"]
        cy = p["cy"]
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
                best_result = None
                best_score = 0
                best_pieces = []
                for attempt in range(1, 61):
                    raw = sensor.snapshot()
                    np_ref = raw.to_numpy_ref()
                    rot = cv2.rotate(np_ref, cv2.ROTATE_90_CLOCKWISE)
                    pieces = extract_pieces(rot)
                    del rot, np_ref, raw
                    if len(pieces) != 4:
                        time.sleep_ms(50)
                        continue
                    result = solve(pieces)
                    if result and result[1] > best_score:
                        best_score = result[1]
                        best_result = result[0]
                        best_pieces = pieces
                        print("#%d NEW BEST=%.0f%%" % (attempt, best_score*100))
                        if best_score >= 0.95:
                            break
                    if attempt % 10 == 0:
                        gc.collect()

                pieces = best_pieces
                print("FINAL score=%.0f%% (%d tries)" % (best_score*100, attempt))

                # 显示结果
                raw = sensor.snapshot()
                np_ref = raw.to_numpy_ref()
                disp = cv2.rotate(np_ref, cv2.ROTATE_90_CLOCKWISE)
                draw_pieces_upper(disp, pieces)
                if best_result:
                    try:
                        draw_solution(disp, best_result)
                        cv2.putText(disp, "BEST %.0f%%" % (best_score*100),
                                    (8, IMG_H-12), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0,255,0), 1)
                    except Exception as e:
                        print("draw err:", e)
                else:
                    cv2.putText(disp, "NO SOLUTION",
                                (8, IMG_H-12), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,0,255), 1)
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
                img.draw_line(0, DIVIDER_Y, IMG_W, DIVIDER_Y,
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
