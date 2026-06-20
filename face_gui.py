"""
人脸颜值评分系统 - 主入口

启动图形界面，功能包括：
  · 图片 / 摄像头人脸检测
  · 基于 MediaPipe 的 7 维度颜值评分（对称性、皮肤、比例、
    眼睛、嘴唇、下巴、眉眼）
  · 结果保存为图片
  · 调用 AI 评分窗口（基于 OpenAI 兼容 SDK）

模块化结构：
  config.py        - 常量 / 默认参数 / 标签定义
  face_detector.py - 人脸检测、特征提取、评分算法
  ai_client.py     - AI 客户端封装（文本 / 图片）
  utils.py         - 公共工具（评分标签、颜色、图像 I/O）
  gui_main.py      - 主窗口 GUI
  gui_ai_window.py - AI 评分子窗口 GUI
"""

import tkinter as tk
from gui_main import FaceDetectionGUI


def main():
    root = tk.Tk()
    gui = FaceDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", gui.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
