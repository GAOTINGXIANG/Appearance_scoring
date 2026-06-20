"""
人脸颜值评分系统 - 配置与常量模块

集中管理评分维度、特征标签、模型路径等常量。
"""

import os
import sys


# ===================== 特征值标签 =====================
# (显示名称, 说明文字)
FEATURE_LABELS = {
    # 对称性
    "symmetry_error_px":   ("对称误差(px)", "越小越对称，<5px 为优秀"),
    "symmetry_score_raw":  ("对称性得分", "0~1，越接近1越对称"),
    # 皮肤
    "skin_evenness":       ("肤色均匀度", "0~1，CrCb色度方差越小越好"),
    "skin_coverage":       ("皮肤覆盖率", "皮肤区域占人脸比例，越高越好"),
    # 面部比例
    "face_width_height_ratio": ("面型宽高比", "理想值≈0.62（黄金比例倒数）"),
    "upper_ratio":         ("上庭比例", "额→眉：占三庭理想 1/3"),
    "middle_ratio":        ("中庭比例", "眉→鼻：占三庭理想 1/3"),
    "lower_ratio":         ("下庭比例", "鼻→下巴：占三庭理想 1/3"),
    "eye_distance_ratio":  ("眼距比例", "两眼距/眼宽，理想 2.5~3.5"),
    "mouth_width_ratio":   ("嘴宽比例", "嘴宽/脸宽，理想 0.30~0.42"),
    # 眼睛
    "eye_openness":        ("眼睛有神度", "眼高/眼宽，理想 0.28~0.48"),
    # 嘴唇
    "lip_fullness":        ("嘴唇饱满度", "唇高/唇宽，理想 0.18~0.38"),
    # 下巴
    "chin_ratio":          ("下巴比例", "下巴/下庭，理想 0.35~0.60"),
    # 眉毛
    "eyebrow_score_raw":   ("眉眼协调度", "0~1，眉眼距离与高度比例"),
    # 尺寸
    "face_width_px":        ("人脸宽度(px)", "像素尺寸"),
    "face_height_px":       ("人脸高度(px)", "像素尺寸"),
}


# ===================== 评分维度标签 =====================
# (显示名称, 满分)
SCORE_LABELS = {
    "symmetry":    ("对称性", 25),
    "skin":       ("皮肤质量", 25),
    "proportion": ("面部比例", 20),
    "eye":        ("眼睛有神", 10),
    "lip":        ("嘴唇饱满", 10),
    "chin":       ("下巴轮廓", 5),
    "eyebrow":    ("眉眼协调", 5),
}


# ===================== 模型与分类器 =====================
# MediaPipe FaceLandmarker 模型文件（约 3.7 MB）
MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MIN_MODEL_SIZE = 3_500_000


# OpenCV Haar 级联分类器候选文件（按优先顺序尝试）
CASCADE_FILENAMES = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt_tree.xml",
]


# ===================== 版本信息 =====================
CURRENT_VERSION = "1.0.0"
RELEASE_PAGE_URL = "https://gitee.com/GAOTINGXIANG/appearance_scoring/releases"
GITEE_API_RELEASES = "https://gitee.com/api/v5/repos/GAOTINGXIANG/appearance_scoring/releases/latest"

# ===================== 应用参数 =====================
WINDOW_TITLE = "人脸颜值评分系统"
WINDOW_GEOMETRY = "1320x720"
WINDOW_MIN_SIZE = (900, 540)


# 默认检测参数
DEFAULT_DETECTION_PARAMS = {
    "scale_factor": 1.1,
    "min_neighbors": 5,
    "min_size": 30,
}


# ===================== 辅助函数 =====================
def get_project_dir():
    """返回项目根目录（兼容 PyInstaller 打包后的环境）。"""
    if hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def get_model_path():
    """返回 face_landmarker.task 模型文件的绝对路径。"""
    return os.path.join(get_project_dir(), MODEL_FILENAME)
