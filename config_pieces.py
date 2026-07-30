"""
碎片模板配置（写死版）
3个四边形 + 1个三角形，拼成目标矩形。
边长从实机识别结果获取，单位像素（标定后可转cm）。
"""

PIXELS_PER_CM = None  # TODO: 实机标定后填入

# 目标矩形尺寸 (cm)  TODO: 量好后填入
TARGET_RECT_W = 15.0
TARGET_RECT_H = 10.0

# 4片碎片模板：边长列表（从长到短排序，单位px）
# 实机检测值
PIECES = [
    {
        "id": 1,
        "name": "四边形A",
        "n_verts": 4,
        "edges_sorted": [146, 115, 55, 47],
    },
    {
        "id": 2,
        "name": "四边形B",
        "n_verts": 4,
        "edges_sorted": [114, 77, 55, 13],
    },
    {
        "id": 3,
        "name": "三角形",
        "n_verts": 3,
        "edges_sorted": [160, 114, 100],
    },
    {
        "id": 4,
        "name": "四边形C(小)",
        "n_verts": 4,
        "edges_sorted": [53, 34, 32, 29],
    },
]

# 目标位置：每片在下半区的放置（质心坐标cm + 旋转角度deg）
# 坐标原点 = 下半区左上角
# TODO: 根据拼接方案实测后填入
TARGETS = [
    {"id": 1, "cx": 5.0,  "cy": 5.0, "angle": 0},
    {"id": 2, "cx": 10.0, "cy": 5.0, "angle": 180},
    {"id": 3, "cx": 12.0, "cy": 3.0, "angle": 90},
    {"id": 4, "cx": 7.5,  "cy": 5.0, "angle": 0},
]
