"""
K230 拼图 · 第1步：可靠碎片提取 + 调试可视化
================================================================
本步只做一件事：把上半区的白色碎片，可靠地提取成"顶点准确的凸多边形"，
并在 LCD 上把每个顶点、每条边长、边数都标出来，方便你肉眼判断提取对不对。

不做任何拼接。等你在实机上看到"每块碎片的多边形都贴合真实形状、
边数正确、没有假顶点"，再进入第2步（约束求解）。

用法：
  上电后 LIVE 预览；按按键(引脚53)冻结/解冻。
  冻结时上半区显示提取结果：绿色多边形、红色顶点(v0..vn)、黄色边长、片编号+边数。

需要你实机调的参数（见下方"参数"区，我都标了 TODO）：
  PIXELS_PER_CM  —— 标定后填入，用于把边长换算成 cm 并用"每边>=2cm"过滤
  MIN_AREA/MAX_AREA —— 按你的碎片大小微调
  APPROX_EPS_RATIO —— 顶点数不对时调这个
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
VISION_W, VISION_H = IMG_W // VISION_SCALE, IMG_H // VISION_SCALE  # 240 x 400

DIVIDER_Y = VISION_H // 2                 # 200：上半区下边界（处理坐标）
DIVIDER_LCD_Y = DIVIDER_Y * VISION_SCALE  # 400：LCD 上的分界线 y

BUTTON_PIN = 53

# 碎片面积过滤（240x400 处理坐标下）。TODO: 按你的碎片实际大小微调
MIN_AREA = 150
MAX_AREA = 30000

# 赛题要求每条边 >= 2cm。标定后填 PIXELS_PER_CM，程序就能：
#   1) 把边长显示成 cm；2) 用 2cm 下限过滤假顶点。
# 标定方法见文件末尾说明。未标定先填 None，程序用像素阈值兜底。
PIXELS_PER_CM = None        # TODO: 例如 8.5
MIN_EDGE_CM = 2.0

# 未标定时的最小边像素阈值：只用来砍"明显是噪声的极短边(<几像素)"。
# 真实边的下限由赛题"每边>=2cm"决定，所以一旦标定 PIXELS_PER_CM，
# 程序就改用 2cm 像素做下限，这里只是未标定时的兜底。
MIN_EDGE_ABS_PX = 4

# approxPolyDP 的 epsilon 倍率（相对凸包周长）。起点放小，优先保留真实顶点。
APPROX_EPS_LIST = (0.008, 0.012, 0.018, 0.025, 0.035)

# 共线合并：删除长直边上被误插的假顶点。B 偏离 AC 连线 < tol 视为共线。
COLLINEAR_TOL_PX = 2.0
COLLINEAR_TOL_RATIO = 0.04
# 冻结取样：不把不同帧的顶点混合，只从多帧中挑一帧质量最好的结果。
# 这样既能避开偶发漏检，也不会破坏顶点顺序。
SNAPSHOT_TRIALS = 5

MAX_VERTICES = 5            # 赛题约束：每片 <=5 边


# -------------------- 工具函数 --------------------
def parse_poly_points(approx):
    """把 cv2 的 approxPolyDP 结果统一解析成 [(x,y), ...]。"""
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


def dist(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.sqrt(dx * dx + dy * dy)


def min_edge_px_for(pts):
    """本片允许的最短边像素。
    标定后用 2cm 对应像素（物理正确）；未标定只用一个小绝对值兜底砍噪声。
    不再用"最长边的比例"，避免大碎片上误杀真实短边。"""
    if PIXELS_PER_CM:
        return MIN_EDGE_CM * PIXELS_PER_CM
    return MIN_EDGE_ABS_PX


def point_line_dist(p, a, c):
    """点 p 到直线 ac 的距离（用叉积 / |ac|）。"""
    dx = c[0] - a[0]
    dy = c[1] - a[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return dist(p, a)
    return abs((p[0] - a[0]) * dy - (p[1] - a[1]) * dx) / length


def merge_collinear(pts, tol_px, tol_ratio):
    """删除"几乎在长直线上的中间顶点"，即长直边被 approxPolyDP 切出的假顶点。
    每轮找偏离邻居连线最近的顶点，若偏离 < max(tol_px, 邻边长*tol_ratio) 就删。"""
    pts = list(pts)
    while len(pts) > 3:
        n = len(pts)
        best_i, best_d = -1, float("inf")
        best_ac = 1.0
        for i in range(n):
            a = pts[(i - 1) % n]
            b = pts[i]
            c = pts[(i + 1) % n]
            ac = dist(a, c)
            if ac < 1e-6:
                continue
            d = point_line_dist(b, a, c)
            if d < best_d:
                best_d, best_i, best_ac = d, i, ac
        if best_i < 0:
            break
        tol = max(tol_px, best_ac * tol_ratio)
        if best_d < tol:
            del pts[best_i]
        else:
            break
    return pts


def merge_short_vertices(pts, min_px):
    """反复合并最短的那条边（用中点替代两个端点），直到所有边 >= min_px。"""
    pts = list(pts)
    while len(pts) > 3:
        n = len(pts)
        best_i, best_d = -1, float("inf")
        for i in range(n):
            d = dist(pts[i], pts[(i + 1) % n])
            if d < best_d:
                best_d, best_i = d, i
        if best_d >= min_px:
            break
        a = pts[best_i]
        b = pts[(best_i + 1) % n]
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        pts[best_i] = mid
        del pts[(best_i + 1) % n]
    return pts


# -------------------- 碎片提取 --------------------
def extract_pieces(frame):
    """frame: 240x400 RGB。返回上半区的碎片列表。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 黑底白片：Otsu 自动阈值
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)  # 填小洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)   # 去小白点

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        if cy >= DIVIDER_Y:           # 只要上半区
            continue

        # 凸包去掉反光造成的浅凹口，再做多边形近似
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)

        pts = []
        for eps_r in APPROX_EPS_LIST:
            approx = cv2.approxPolyDP(hull, peri * eps_r, True)
            cand = parse_poly_points(approx)
            if len(cand) < 3:
                continue
            cand = merge_collinear(cand, COLLINEAR_TOL_PX, COLLINEAR_TOL_RATIO)
            cand = merge_short_vertices(cand, min_edge_px_for(cand))
            if 3 <= len(cand) <= MAX_VERTICES:
                pts = cand
                break
        if len(pts) < 3:
            # 兜底：中等 eps 再试一次
            approx = cv2.approxPolyDP(hull, peri * 0.03, True)
            cand = parse_poly_points(approx)
            cand = merge_collinear(cand, COLLINEAR_TOL_PX, COLLINEAR_TOL_RATIO)
            cand = merge_short_vertices(cand, min_edge_px_for(cand))
            if 3 <= len(cand) <= MAX_VERTICES:
                pts = cand
        if len(pts) < 3:
            continue  # 这片提取失败，跳过

        n = len(pts)
        edges = [dist(pts[i], pts[(i + 1) % n]) for i in range(n)]
        pieces.append({
            "area": area,
            "cx": cx, "cy": cy,                                  # 处理坐标
            "lcd_cx": cx * VISION_SCALE, "lcd_cy": cy * VISION_SCALE,
            "proc_pts": pts,                                     # 处理坐标
            "lcd_pts": [(x * VISION_SCALE, y * VISION_SCALE) for x, y in pts],
            "edges": edges,                                      # 处理坐标像素长度
        })
    return pieces


# -------------------- 多帧投票稳定 --------------------
def cluster_points(pts, radius, min_count):
    """把跨帧收集到的所有顶点做贪心聚类，返回"出现次数>=min_count"的簇质心。
    真顶点每帧都在附近 → 簇大；随机噪声位置散 → 簇小被滤掉。"""
    unique = sorted(set((int(p[0]), int(p[1])) for p in pts))
    clusters = []  # 每项 [sum_x, sum_y, count]
    for x, y in unique:
        placed = False
        for c in clusters:
            mx = c[0] / c[2]
            my = c[1] / c[2]
            if (x - mx) * (x - mx) + (y - my) * (y - my) < radius * radius:
                c[0] += x
                c[1] += y
                c[2] += 1
                placed = True
                break
        if not placed:
            clusters.append([x, y, 1])
    out = []
    for c in clusters:
        if c[2] >= min_count:
            out.append((int(c[0] / c[2] + 0.5), int(c[1] / c[2] + 0.5)))
    return out


def merge_stable_pieces(frames):
    """frames: 多帧的 pieces 列表。按中心匹配同一片，对其顶点聚类投票。
    不能固定使用第1帧：第1帧偶尔漏检时，会错误得到0片；改用本轮检测数量最多的帧作基准。"""
    if not frames:
        return []
    base = max(frames, key=lambda item: len(item))
    if not base:
        return []
    merged = []
    for bp in base:
        all_verts = list(bp["proc_pts"])
        cx_sum, cy_sum, area_sum, cnt = bp["cx"], bp["cy"], bp["area"], 1
        for other in frames[1:]:
            best, best_d = None, 1e9
            for op in other:
                d = dist((bp["cx"], bp["cy"]), (op["cx"], op["cy"]))
                if d < best_d:
                    best_d, best = d, op
            if best is not None and best_d < 30:   # 中心很近才算同一片
                all_verts.extend(best["proc_pts"])
                cx_sum += best["cx"]
                cy_sum += best["cy"]
                area_sum += best["area"]
                cnt += 1
        cents = cluster_points(all_verts, CLUSTER_RADIUS, VOTE_M)
        if len(cents) < 3:
            continue
        cx = cx_sum // cnt
        cy = cy_sum // cnt
        n = len(cents)
        edges = [dist(cents[i], cents[(i + 1) % n]) for i in range(n)]
        merged.append({
            "area": area_sum / cnt,
            "cx": cx, "cy": cy,
            "lcd_cx": cx * VISION_SCALE, "lcd_cy": cy * VISION_SCALE,
            "proc_pts": cents,
            "lcd_pts": [(x * VISION_SCALE, y * VISION_SCALE) for x, y in cents],
            "edges": edges,
        })
    return merged


def choose_snapshot_frame(frames):
    """从多次独立检测中选一帧，不混合任何顶点坐标。

    规则：
    - 碎片数最多优先；
    - 同样数量时，按中心位置排序后，边数更接近多数帧的优先；
    - 最后用总面积作轻微的稳定性参考。
    """
    valid = [frame for frame in frames if frame]
    if not valid:
        return []
    max_count = max(len(frame) for frame in valid)
    candidates = [frame for frame in valid if len(frame) == max_count]
    if len(candidates) == 1:
        return candidates[0]

    ordered = []
    for frame in valid:
        ordered.append(sorted(frame, key=lambda p: (p["cy"], p["cx"])))

    target_counts = []
    for index in range(max_count):
        counts = []
        for frame in ordered:
            if len(frame) == max_count and index < len(frame):
                counts.append(len(frame[index]["proc_pts"]))
        if counts:
            counts.sort()
            target_counts.append(counts[len(counts) // 2])
        else:
            target_counts.append(4)

    best = candidates[0]
    best_score = None
    for frame in candidates:
        frame_ordered = sorted(frame, key=lambda p: (p["cy"], p["cx"]))
        shape_error = 0
        total_area = 0.0
        for index, piece in enumerate(frame_ordered):
            if index < len(target_counts):
                shape_error += abs(len(piece["proc_pts"]) - target_counts[index])
            total_area += piece["area"]
        # 主要按边数一致性选，面积只作极弱的平局参考。
        score = shape_error * 1000000 - total_area
        if best_score is None or score < best_score:
            best_score = score
            best = frame
    return best


# -------------------- 调试绘制 --------------------
def draw_debug(rotated_frame, pieces):
    # 蓝色水平分界线（保留之前正确的效果）
    cv2.line(rotated_frame, (0, DIVIDER_LCD_Y), (IMG_W, DIVIDER_LCD_Y), (255, 0, 0), 2)

    for idx, p in enumerate(pieces):
        pts = p["lcd_pts"]
        n = len(pts)
        # 绿色多边形
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            cv2.line(rotated_frame, a, b, (0, 255, 0), 2)
        # 红色顶点 + 编号
        for i, (x, y) in enumerate(pts):
            cv2.circle(rotated_frame, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(rotated_frame, "v%d" % i, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        # 黄色边长（边中点）
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
            L = p["edges"][i]
            if PIXELS_PER_CM:
                label = "%.1fcm" % (L / PIXELS_PER_CM)
            else:
                label = "%dpx" % int(L)
            cv2.putText(rotated_frame, label, (mx - 14, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        # 片编号 + 边数
        cv2.putText(rotated_frame, "P%d n=%d" % (idx + 1, n),
                    (p["lcd_cx"] - 24, p["lcd_cy"] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 顶部状态栏
    summary = " ".join("P%d:%d" % (i + 1, len(p["lcd_pts"])) for i, p in enumerate(pieces))
    cv2.putText(rotated_frame, "PIECES=%d  %s" % (len(pieces), summary),
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


def draw_targets(rotated_frame, target_positions, matched):
    """在LCD下半区画目标位置（黄色圆+ID），上半区画匹配箭头"""
    for mid, tgt in target_positions.items():
        # 目标位置画黄色圆圈 + ID标签
        tx = tgt["cx"] * VISION_SCALE
        ty = tgt["cy"] * VISION_SCALE
        cv2.circle(rotated_frame, (tx, ty), 12, (0, 255, 255), 2)
        cv2.putText(rotated_frame, "T%d" % mid, (tx + 14, ty + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    for mid, det in matched.items():
        if mid in target_positions:
            # 从当前位置画箭头指向目标
            cx = det["cx"] * VISION_SCALE
            cy = det["cy"] * VISION_SCALE
            tx = target_positions[mid]["cx"] * VISION_SCALE
            ty = target_positions[mid]["cy"] * VISION_SCALE
            cv2.arrowedLine(rotated_frame, (cx, cy), (tx, ty),
                            (255, 0, 255), 2, tipLength=0.15)


# -------------------- 写死版：碎片模板 --------------------
# 用边长比例匹配（最长边=1.0），不受距离缩放影响
TEMPLATES = [
    {"id": 1, "name": "大四边形", "n": 4, "ratios": [1.0, 0.79, 0.38, 0.32]},
    {"id": 2, "name": "窄四边形", "n": 4, "ratios": [1.0, 0.68, 0.48, 0.11]},
    {"id": 3, "name": "大三角形", "n": 3, "ratios": [1.0, 0.71, 0.63]},
    {"id": 4, "name": "小四边形", "n": 4, "ratios": [1.0, 0.64, 0.60, 0.55]},
]
MATCH_TOL_RATIO = 0.15  # 比例误差容差


def get_edge_ratios(piece):
    """返回边长比例列表（从大到小，最长=1.0）"""
    edges = sorted(piece["edges"], reverse=True)
    longest = edges[0] if edges[0] > 0 else 1
    return [e / longest for e in edges]


def _try_match_ratios(ratios, n):
    """尝试用给定比例和边数匹配模板"""
    best_id, best_err = None, float("inf")
    for t in TEMPLATES:
        if t["n"] == n and len(t["ratios"]) == len(ratios):
            err = sum(abs(a-b) for a, b in zip(ratios, t["ratios"]))
            if err < best_err:
                best_err = err
                best_id = t["id"]
    return best_id, best_err


def match_piece_to_template(piece):
    """用边长比例匹配，自动处理多检测1-2个假顶点的情况"""
    n = len(piece["proc_pts"])
    ratios = get_edge_ratios(piece)

    # 直接匹配
    best_id, best_err = _try_match_ratios(ratios, n)

    # 如果有短边(比例<0.12)，尝试去掉短边后当 n-1 匹配
    while len(ratios) > 3 and ratios[-1] < 0.12:
        ratios = ratios[:-1]
        reduced_n = len(ratios)
        longest = ratios[0] if ratios[0] > 0 else 1
        ratios_norm = [r / longest for r in ratios]
        tid, terr = _try_match_ratios(ratios_norm, reduced_n)
        if tid is not None and terr < best_err:
            best_err = terr
            best_id = tid

    if best_err < MATCH_TOL_RATIO * 4:
        return best_id
    return None


def compute_angle(pts):
    """用最长边方向作为碎片朝向(deg)"""
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


def extract_lower_half(frame):
    """提取下半区碎片（用于标定目标位置）"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        if cy < DIVIDER_Y:
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
            "cx": cx, "cy": cy, "proc_pts": pts,
            "edges": edges, "area": area,
        })
    return pieces


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
    frozen_snapshot_done = False
    frozen_frame = None
    frozen_display_img = None
    # 写死的目标位置（标定结果直接内嵌，不需要运行时标定）
    target_positions = {
        1: {"cx": 152, "cy": 347, "angle": 179.6},
        2: {"cx": 152, "cy": 292, "angle": 169.7},
        3: {"cx": 70,  "cy": 271, "angle": 134.1},
        4: {"cx": 184, "cy": 241, "angle": 192.8},
    }

    print("=== K230 拼图(写死版) 就绪 ===")
    print("按键 = 检测上半区 + 匹配 + 输出移动指令")

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
                        frozen_snapshot_done = False
                    else:
                        frozen_snapshot_done = False
                        frozen_frame = None
                        frozen_display_img = None
                        gc.collect()
                    print("MODE -> %s" % ("SNAPSHOT" if is_frozen else "LIVE"))

            last_btn_val = btn_val

            if is_frozen:
                # 冻结态只取一次：拍 SNAPSHOT_TRIALS 帧，挑一帧最佳结果，之后保持不变。
                if not frozen_snapshot_done:
                    gc.collect()
                    trial_frames = []
                    for _ in range(SNAPSHOT_TRIALS):
                        raw_img = sensor.snapshot()
                        np_ref = raw_img.to_numpy_ref()
                        trial_rotated = cv2.rotate(np_ref, cv2.ROTATE_90_CLOCKWISE)
                        trial_small = cv2.resize(trial_rotated, (VISION_W, VISION_H))
                        trial_frames.append(extract_pieces(trial_small))
                        del trial_small, trial_rotated, np_ref, raw_img

                    pieces = choose_snapshot_frame(trial_frames)
                    # 用最后一帧作为显示背景；识别结果仍来自挑选出的单帧。
                    raw_img = sensor.snapshot()
                    np_ref = raw_img.to_numpy_ref()
                    rotated_frame = cv2.rotate(np_ref, cv2.ROTATE_90_CLOCKWISE)

                    print("DETECTED %d pieces, verts=%s" % (
                        len(pieces), [len(p["proc_pts"]) for p in pieces]))
                    for idx, p in enumerate(pieces):
                        if PIXELS_PER_CM:
                            el = ["%.1fcm" % (e / PIXELS_PER_CM) for e in p["edges"]]
                        else:
                            el = ["%dpx" % int(e) for e in p["edges"]]
                        print("  P%d n=%d edges=%s" % (
                            idx + 1, len(p["proc_pts"]), el))

                    # 写死版匹配
                    print("--- MATCH ---")
                    matched = {}
                    for idx, p in enumerate(pieces):
                        mid = match_piece_to_template(p)
                        if mid is not None:
                            name = TEMPLATES[mid - 1]["name"]
                            print("  P%d -> ID%d (%s)" % (idx+1, mid, name))
                            matched[mid] = p
                        else:
                            print("  P%d -> ???" % (idx+1))

                    # 输出移动指令
                    if target_positions:
                        print("--- MOVE ---")
                        for mid, det in matched.items():
                            if mid in target_positions:
                                tgt = target_positions[mid]
                                cur_ang = compute_angle(det["proc_pts"])
                                delta_ang = tgt["angle"] - cur_ang
                                print("  ID%d: (%d,%d,%.0f) -> (%d,%d,%.0f) d_ang=%.0f" % (
                                    mid, det["cx"], det["cy"], cur_ang,
                                    tgt["cx"], tgt["cy"], tgt["angle"], delta_ang))
                    else:
                        print("(未标定目标，长按按键先标定)")

                    draw_debug(rotated_frame, pieces)
                    if target_positions:
                        draw_targets(rotated_frame, target_positions, matched)
                    cv2.putText(rotated_frame, "[SNAPSHOT] press key to resume",
                                (8, IMG_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)
                    frozen_frame = rotated_frame
                    frozen_display_img = image.Image(
                        IMG_W, IMG_H, image.RGB888,
                        alloc=image.ALLOC_REF, data=frozen_frame,
                    )
                    Display.show_image(frozen_display_img)
                    frozen_snapshot_done = True
                    del np_ref, raw_img
                    gc.collect()
                else:
                    # 不重新采集、不重新计算，保持已经选中的结果。
                    time.sleep_ms(120)
            else:
                # 实时预览
                raw_img = sensor.snapshot()
                np_ref = raw_img.to_numpy_ref()
                rotated_frame = cv2.rotate(np_ref, cv2.ROTATE_90_CLOCKWISE)

                img = image.Image(IMG_W, IMG_H, image.RGB888,
                                  alloc=image.ALLOC_REF, data=rotated_frame)
                img.draw_line(0, DIVIDER_LCD_Y, IMG_W, DIVIDER_LCD_Y,
                              color=(255, 0, 0), thickness=2)
                img.draw_string_advanced(10, 10, 20, "LIVE  (press key to freeze)",
                                         color=(255, 255, 255))
                Display.show_image(img)
                del img, rotated_frame, np_ref, raw_img
                if frame_cnt % 5 == 0:
                    gc.collect()
    except KeyboardInterrupt:
        print("用户停止")
    except BaseException as e:
        if "IDE interrupt" in str(e):
            print("IDE 停止")
        else:
            print("异常:", e)
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


# -------------------- 标定 PIXELS_PER_CM 的小说明 --------------------
# 1. 找一张白纸，画一条已知长度的线（比如 10cm）放在 A4 上半区。
# 2. 运行本程序冻结，读出这条线两端在 LCD 上的像素距离 L_px。
#    （或直接在串口 print 一个已知边长的像素值）
# 3. PIXELS_PER_CM = L_px / 10。
# 4. 填到上方 PIXELS_PER_CM，重新运行。此后边长显示 cm，且 <2cm 的假边会被自动合并。
#
# 注意：处理坐标(240x400)下的像素，换算到 cm 用的是同一个 PIXELS_PER_CM，
#      因为 edges 存的就是处理坐标像素，标定也应在处理坐标或同等比例下做。
#      最稳妥：标定时也用 240x400 的图量像素。

if __name__ == "__main__":
    main()
