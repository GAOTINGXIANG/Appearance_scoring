"""
人脸颜值评分系统 - 公共工具模块

提供分数标签/颜色映射、图像加载保存等通用工具。
"""

import os
import numpy as np
import cv2


# ===================== 分数 → 标签 =====================
def score_tag(score):
    """根据总分返回文字标签。"""
    if score >= 85:
        return "S级 神仙颜值"
    if score >= 75:
        return "A级 非常出众"
    if score >= 65:
        return "B级 较好"
    if score >= 55:
        return "C级 一般"
    if score >= 45:
        return "D级 较弱"
    return "E级 有待提升"


# ===================== 分数 → BGR 颜色 =====================
def score_color(score):
    """根据总分返回 OpenCV BGR 颜色。"""
    if score >= 85:
        return (0, 100, 255)
    if score >= 75:
        return (0, 215, 255)
    if score >= 65:
        return (0, 255, 100)
    if score >= 55:
        return (80, 220, 80)
    if score >= 45:
        return (50, 180, 255)
    return (0, 80, 255)


# ===================== 图像 I/O（支持中文路径）=====================
def load_image(path):
    """读取图片，自动支持中文路径。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            raise FileNotFoundError(f"图片文件为空或无法读取: {path}")
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"无法解码图片，请确认文件格式: {path}")
        return image
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise FileNotFoundError(f"读取图片失败: {path} ({exc})")


def save_image(path, image):
    """保存图片，自动支持中文路径。"""
    try:
        ext = os.path.splitext(path)[1] or ".jpg"
        success, buf = cv2.imencode(ext, image)
        if not success:
            raise RuntimeError(f"无法编码图片: {path}")
        buf.tofile(path)
        return path
    except Exception as exc:
        raise RuntimeError(f"保存图片失败: {path} ({exc})")
