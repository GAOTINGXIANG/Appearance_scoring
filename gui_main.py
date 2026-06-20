"""
人脸颜值评分系统 - 主窗口 GUI 模块

集成了：
  - 图片 / 摄像头人脸检测
  - 基于 MediaPipe 的 7 维度颜值评分
  - 结果保存
  - 调用 AI 评分窗口（OpenAI 兼容 SDK）
"""

import json
import os
import re
import threading
import tkinter as tk
import webbrowser
import urllib.request
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from config import (
    FEATURE_LABELS,
    SCORE_LABELS,
    DEFAULT_DETECTION_PARAMS,
    WINDOW_TITLE,
    WINDOW_GEOMETRY,
    WINDOW_MIN_SIZE,
    CURRENT_VERSION,
    RELEASE_PAGE_URL,
    GITEE_API_RELEASES,
)
from face_detector import FaceDetector, _score_tag, _score_color
from utils import load_image, save_image
from gui_ai_window import AIScoringWindow


# ===================== 主窗口 =====================
class FaceDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_GEOMETRY)
        self.root.minsize(*WINDOW_MIN_SIZE)

        self._init_state()
        self._build_ui()
        self._update_status(f"分类器就绪: {os.path.basename(self.detector.cascade_path)}")

    # ---------- 状态初始化 ----------
    def _init_state(self):
        self.detector = FaceDetector(
            scale_factor=DEFAULT_DETECTION_PARAMS["scale_factor"],
            min_neighbors=DEFAULT_DETECTION_PARAMS["min_neighbors"],
            min_size=DEFAULT_DETECTION_PARAMS["min_size"],
        )

        # 当前画面
        self.current_image = None
        self.result_image = None
        self.frozen_frame = None
        self.frozen_result = None
        self.faces = []

        # 评分结果
        self.last_features = {}
        self.last_sub_scores = {}
        self.last_total_score = 0.0
        self.scoring_enabled = True
        self.multi_face_warned = False

        # 模式
        self.image_mode = True
        self.camera_thread = None
        self.camera_running = False
        self.video_capture = None
        self._photo = None

        # 加载动画
        self._loading = False
        self._loading_frame = 0
        self._loading_job = None

    # ---------- UI 构建 ----------
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self._build_controls(main)
        self._build_display(main)
        self._build_features_panel(main)
        self._build_statusbar(main)

    def _build_controls(self, parent):
        ctrl = ttk.LabelFrame(parent, text="控制面板", padding=6)
        ctrl.grid(row=0, column=0, sticky="ns", padx=(0, 6))

        # 操作按钮组
        self._build_main_buttons(ctrl)
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=4)

        # 启用评分开关
        self.scoring_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            ctrl, text="启用颜值评分", variable=self.scoring_var,
            command=self._on_toggle_scoring,
        ).pack(anchor="w")

        # 检测参数
        ttk.Label(ctrl, text="检测参数", font=("Arial", 9, "bold")).pack(anchor="w", pady=(4, 0))
        self._build_scale_factor_slider(ctrl)
        self._build_min_neighbors_spinbox(ctrl)
        self._build_min_size_spinbox(ctrl)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=4)

        # 检测状态显示
        ttk.Label(ctrl, text="检测与评分", font=("Arial", 9, "bold")).pack(anchor="w")
        self.info_label = ttk.Label(ctrl, text="未检测", foreground="#555555", font=("Arial", 9))
        self.info_label.pack(anchor="w")
        self.score_status_label = ttk.Label(ctrl, text="等待检测", foreground="#555555", font=("Arial", 9))
        self.score_status_label.pack(anchor="w")

        # 进度条
        self.progress_bar = ttk.Progressbar(ctrl, mode="determinate", length=170)
        self.progress_bar.pack(fill="x", pady=(4, 0))
        self.progress_label = ttk.Label(ctrl, text="", foreground="#555555", font=("Arial", 8))
        self.progress_label.pack(anchor="w")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=4)

        # AI 模式配置区域
        self._build_ai_config(ctrl)

    def _build_main_buttons(self, parent):
        # 每行两个按钮，依次往下排列
        btn_cfg = [
            ("打开图片",  self.open_image),
            ("摄像头",    self.toggle_camera),
            ("评分",      self.rate_current_frame),
            ("导出照片",  self.save_image_only),
            ("导出数据",  self.save_data_only),
            ("清除",      self.clear_display),
            ("重新检测",  self.re_detect_image),
            ("说明",      self.show_help),
            ("开源网址",  self.show_open_source_links),
            ("检查更新",  self.check_update),
            ("彩蛋",      self.run_egg),
        ]
        for i in range(0, len(btn_cfg), 2):
            row = ttk.Frame(parent)
            row.pack(fill="x", pady=(0, 2))
            ttk.Button(row, text=btn_cfg[i][0], width=10, command=btn_cfg[i][1]).pack(side="left", padx=2)
            if i + 1 < len(btn_cfg):
                ttk.Button(row, text=btn_cfg[i + 1][0], width=10,
                           command=btn_cfg[i + 1][1]).pack(side="left", padx=2)

    def _build_scale_factor_slider(self, parent):
        sf_frame = ttk.Frame(parent)
        sf_frame.pack(fill="x", pady=1)
        ttk.Label(sf_frame, text="scaleFactor:", width=12).pack(side="left")
        self.scale_var = tk.DoubleVar(value=DEFAULT_DETECTION_PARAMS["scale_factor"])
        ttk.Scale(sf_frame, from_=1.01, to=2.0, orient="horizontal",
                  variable=self.scale_var, command=self._on_param_change).pack(side="left", fill="x", expand=True, padx=2)
        self.scale_label = ttk.Label(sf_frame, text=f"{DEFAULT_DETECTION_PARAMS['scale_factor']:.2f}", width=4)
        self.scale_label.pack(side="right")

    def _build_min_neighbors_spinbox(self, parent):
        nn_frame = ttk.Frame(parent)
        nn_frame.pack(fill="x", pady=1)
        ttk.Label(nn_frame, text="minNeighbors:", width=12).pack(side="left")
        self.neighbors_var = tk.IntVar(value=DEFAULT_DETECTION_PARAMS["min_neighbors"])
        ttk.Spinbox(nn_frame, from_=1, to=20, width=5,
                    textvariable=self.neighbors_var, command=self._on_param_change).pack(side="right")

    def _build_min_size_spinbox(self, parent):
        ms_frame = ttk.Frame(parent)
        ms_frame.pack(fill="x", pady=1)
        ttk.Label(ms_frame, text="minSize:", width=12).pack(side="left")
        self.minsize_var = tk.IntVar(value=DEFAULT_DETECTION_PARAMS["min_size"])
        ttk.Spinbox(ms_frame, from_=10, to=200, width=5,
                    textvariable=self.minsize_var, command=self._on_param_change).pack(side="right")

    def _build_ai_config(self, parent):
        ttk.Label(parent, text="AI 评分模式", font=("Arial", 9, "bold")).pack(anchor="w")

        # API 地址
        api_frame = ttk.Frame(parent)
        api_frame.pack(fill="x", pady=1)
        ttk.Label(api_frame, text="API 地址:", width=8).pack(side="left")
        self.ai_api_url = tk.StringVar(value="https://open.bigmodel.cn/api/paas/v4/")
        ttk.Entry(api_frame, textvariable=self.ai_api_url, width=14).pack(side="left", fill="x", expand=True, padx=2)

        # 模型 ID
        model_frame = ttk.Frame(parent)
        model_frame.pack(fill="x", pady=1)
        ttk.Label(model_frame, text="模型 ID:", width=8).pack(side="left")
        self.ai_model_id = tk.StringVar(value="glm-4.5-air")
        ttk.Entry(model_frame, textvariable=self.ai_model_id, width=14).pack(side="left", fill="x", expand=True, padx=2)

        # Token
        token_frame = ttk.Frame(parent)
        token_frame.pack(fill="x", pady=1)
        ttk.Label(token_frame, text="Token:", width=8).pack(side="left")
        self.ai_token = tk.StringVar(value="")
        ttk.Entry(token_frame, textvariable=self.ai_token, width=14, show="*").pack(side="left", fill="x", expand=True, padx=2)

        # 进入 AI 模式按钮
        ttk.Button(parent, text="进入 AI 模式", width=20, command=self.open_ai_window).pack(pady=(4, 0))

    def _build_display(self, parent):
        disp = ttk.LabelFrame(parent, text="画面显示", padding=5)
        disp.grid(row=0, column=1, sticky="nsew")
        disp.columnconfigure(0, weight=1)
        disp.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(disp, bg="#1e1e1e", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda e: self._refresh_canvas())

    def _build_features_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="人脸特征与评分", padding=8)
        panel.grid(row=0, column=2, sticky="ns", padx=(6, 0))

        # 总分
        score_box = ttk.Frame(panel)
        score_box.pack(fill="x", pady=(0, 6))
        ttk.Label(score_box, text="综合评分:", font=("Arial", 11, "bold")).pack(side="left")
        self.total_score_label = tk.Label(
            score_box, text="-- / 100", font=("Arial", 16, "bold"),
            fg="#2a7fb8", bg="#f2f2f2", padx=8, pady=2,
        )
        self.total_score_label.pack(side="left", padx=(6, 0))

        # 各项子分
        sub_box = ttk.LabelFrame(panel, text="各维度得分", padding=6)
        sub_box.pack(fill="x", pady=(0, 8))
        self.sub_score_labels = {}
        for key, (label, max_val) in SCORE_LABELS.items():
            row = ttk.Frame(sub_box)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=f"{label}({max_val}):", width=10, anchor="w").pack(side="left")
            pb = ttk.Progressbar(row, maximum=max_val, length=160, mode="determinate")
            pb.pack(side="left", fill="x", expand=True, padx=4)
            val_label = ttk.Label(row, text="0.0", width=5, anchor="e")
            val_label.pack(side="right")
            self.sub_score_labels[key] = (pb, val_label)

        # 特征值
        feat_box = ttk.LabelFrame(panel, text="特征值", padding=6)
        feat_box.pack(fill="both", expand=True)
        self.feature_labels = {}
        for key, (label, _hint) in FEATURE_LABELS.items():
            row = ttk.Frame(feat_box)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=f"{label}:", width=12, anchor="w").pack(side="left")
            val_label = ttk.Label(row, text="--", width=10, anchor="e")
            val_label.pack(side="right")
            self.feature_labels[key] = val_label

        # 说明提示
        self.hint_label = ttk.Label(
            panel,
            text="※ 点击「评分」按钮对当前\n   定格画面进行打分。\n\n基于医学面部美学比例评分",
            foreground="#7a7a7a",
            justify="left",
        )
        self.hint_label.pack(anchor="w", pady=(10, 0))

    def _build_statusbar(self, parent):
        bar = ttk.Frame(parent)
        bar.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(bar, textvariable=self.status_var, anchor="w").pack(side="left")

    # ---------- 参数同步 ----------
    def _on_param_change(self, *_):
        try:
            self.detector.scale_factor = float(self.scale_var.get())
            self.detector.min_neighbors = int(self.neighbors_var.get())
            self.detector.min_size = int(self.minsize_var.get())
            self.scale_label.config(text=f"{self.detector.scale_factor:.2f}")
            if self.image_mode and self.current_image is not None:
                self.re_detect_image()
        except Exception:
            pass

    def _on_toggle_scoring(self):
        self.scoring_enabled = bool(self.scoring_var.get())
        self._update_status(f"评分已{'启用' if self.scoring_enabled else '关闭'}")
        if self.image_mode and self.current_image is not None:
            self.re_detect_image()

    # ---------- 检测 + 评分主流程 ----------
    def _process_frame(self, frame):
        try:
            faces = self.detector.detect(frame)
        except Exception:
            faces = []

        result = self.detector.draw_faces(frame, faces)
        self.faces = faces
        self.result_image = result

        n = len(faces)
        self.info_label.config(
            text=f"检测到 {n} 张人脸",
            foreground="#d63031" if n > 1 else ("#007f00" if n == 1 else "#7f7f7f"),
        )

        features = {}
        sub_scores = {}
        total = 0.0
        score_state = "idle"

        if not self.scoring_enabled:
            score_state = "disabled"
        elif n == 0:
            score_state = "idle"
        elif n == 1:
            self.multi_face_warned = False
            try:
                features = self.detector.extract_features(frame, faces[0])
                total, sub_scores = self.detector.compute_score(features)
                score_state = "ok"
                x, y, _w, _h = faces[0]
                cv2.putText(
                    result, f"Score: {total:.1f}/100",
                    (x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
                )
            except Exception:
                score_state = "idle"
        else:
            score_state = "multi"
            self.multi_face_warned = True

        self.last_features = features
        self.last_sub_scores = sub_scores
        self.last_total_score = total
        self._update_score_ui(score_state, n)
        return result, faces

    # ---------- 图片模式 ----------
    def open_image(self):
        self._stop_camera()
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            image = load_image(path)
        except Exception as exc:
            messagebox.showerror("导入失败", f"无法加载图片:\n{path}\n\n原因: {exc}")
            return

        self.current_image = image
        self.image_path = path
        self.image_mode = True
        self.frozen_frame = None
        self.frozen_result = None
        self.multi_face_warned = False
        self.faces = []
        self.result_image = image.copy()
        self._show_opencv_image(self.result_image)
        self.info_label.config(text="已导入图片，正在检测...", foreground="#555555")
        self.score_status_label.config(text="点击评分", foreground="#555555")
        self._update_status(f"已加载图片: {os.path.basename(path)}")

        self.root.update_idletasks()
        try:
            faces = self.detector.detect(image)
        except Exception:
            faces = []

        self.faces = faces
        n = len(faces)
        if n == 0:
            self.result_image = image.copy()
            self.info_label.config(text="未检测到人脸", foreground="#d63031")
            self._show_opencv_image(self.result_image)
            self._update_status(f"已加载图片: {os.path.basename(path)}，未检测到人脸，无法评分")
        else:
            self.result_image = self.detector.draw_faces(image, faces)
            self.info_label.config(
                text=f"检测到 {n} 张人脸",
                foreground="#d63031" if n > 1 else "#007f00",
            )
            self.score_status_label.config(text="点击评分", foreground="#555555")
            self._show_opencv_image(self.result_image)
            self._update_status(f"已加载图片: {os.path.basename(path)}，点击评分按钮进行打分")

    def re_detect_image(self):
        if self.current_image is None:
            self.info_label.config(text="请先打开图片", foreground="#d63031")
            self._update_status("请先打开一张图片")
            return
        self.multi_face_warned = False
        self.faces = []
        self.frozen_frame = None
        self.frozen_result = None
        self.info_label.config(text="正在重新检测...", foreground="#555555")
        self._update_status("正在重新检测...")
        try:
            faces = self.detector.detect(self.current_image)
        except Exception:
            faces = []
        self.faces = faces
        n = len(faces)
        if n == 0:
            self.result_image = self.current_image.copy()
            self.info_label.config(text="未检测到人脸", foreground="#d63031")
            self.score_status_label.config(text="无法评分", foreground="#d63031")
            self._show_opencv_image(self.result_image)
            self._update_status("重新检测完成，未检测到人脸")
        else:
            self.result_image = self.detector.draw_faces(self.current_image, faces)
            self.info_label.config(
                text=f"检测到 {n} 张人脸",
                foreground="#d63031" if n > 1 else "#007f00",
            )
            self.score_status_label.config(text="点击评分", foreground="#555555")
            self._show_opencv_image(self.result_image)
            self._update_status("重新检测完成")

    # ---------- 摄像头模式 ----------
    def toggle_camera(self):
        if self.camera_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        self.image_mode = False
        self.current_image = None
        self.frozen_frame = None
        self.frozen_result = None
        self.multi_face_warned = False
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            self.video_capture = None
            messagebox.showerror("错误", "无法打开摄像头。")
            return
        self.camera_running = True
        self._update_status("摄像头运行中，点击评分按钮进行打分")
        self.score_status_label.config(text="点击评分", foreground="#555555")
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()

    def _stop_camera(self):
        self.camera_running = False
        if self.camera_thread is not None:
            self.camera_thread.join(timeout=1.0)
            self.camera_thread = None
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self._update_status("摄像头已停止")

    def _camera_loop(self):
        while self.camera_running and self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            try:
                faces = self.detector.detect(frame)
            except Exception:
                faces = []
            result = self.detector.draw_faces(frame, faces)
            self.faces = faces
            self.result_image = result
            self.root.after(0, self._refresh_canvas_live)
            self.root.after(
                0, lambda n=len(faces): self.info_label.config(
                    text=f"检测到 {n} 张人脸",
                    foreground="#d63031" if n > 1 else ("#007f00" if n == 1 else "#7f7f7f"),
                )
            )

    def _refresh_canvas_live(self):
        if self.result_image is not None:
            self._show_opencv_image(self.result_image)

    # ---------- 加载动画 ----------
    def _start_loading_animation(self, message="正在分析..."):
        self._loading = True
        self._loading_frame = 0
        self._loading_message = message
        self._update_loading_animation()

    def _update_loading_animation(self):
        if not self._loading:
            return
        self._loading_frame = (self._loading_frame + 1) % 4
        dots = "." * (self._loading_frame + 1)
        self.progress_label.config(text=f"{self._loading_message}{dots}")
        self._loading_job = self.root.after(200, self._update_loading_animation)

    def _stop_loading_animation(self):
        self._loading = False
        if self._loading_job:
            self.root.after_cancel(self._loading_job)
            self._loading_job = None

    # ---------- 评分（对定格画面）----------
    def rate_current_frame(self):
        if self.result_image is None:
            messagebox.showinfo("提示", "没有可评分的画面。")
            return
        if len(self.faces) == 0:
            messagebox.showinfo("提示", "当前画面没有检测到人脸。")
            return

        self.frozen_result = self.result_image.copy()
        if not self.image_mode:
            if self.video_capture is not None:
                ret, frame = self.video_capture.read()
                if not ret:
                    messagebox.showerror("错误", "无法获取当前帧。")
                    return
                self.frozen_frame = frame
        else:
            self.frozen_frame = self.current_image

        self.progress_bar["value"] = 0
        self._start_loading_animation("正在分析人脸特征")
        self._update_status("正在分析人脸特征，请稍候...")
        self.score_status_label.config(text="评分中...", foreground="#555555")

        thread = threading.Thread(target=self._rate_frame_async, daemon=True)
        thread.start()

    def _rate_frame(self, frame, result, progress_callback=None):
        faces = self.faces
        n = len(faces)
        features = {}
        sub_scores = {}
        total = 0.0
        score_state = "idle"

        if not self.scoring_enabled:
            score_state = "disabled"
        elif n == 0:
            score_state = "idle"
        elif n == 1:
            self.multi_face_warned = False
            try:
                features = self.detector.extract_features(
                    frame, faces[0], progress_callback=progress_callback,
                )
                total, sub_scores = self.detector.compute_score(features)
                score_state = "ok"
                x, y, w, h = faces[0]
                tag_color = _score_color(total)
                cv2.putText(result, f"Score: {total:.1f}/100",
                            (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, tag_color, 2)
            except Exception as exc:
                print(f"评分出错: {exc}")
                score_state = "idle"
        else:
            score_state = "multi"
            self.multi_face_warned = True

        self.last_features = features
        self.last_sub_scores = sub_scores
        self.last_total_score = total
        self._update_score_ui(score_state, n)

    def _rate_frame_async(self):
        try:
            def progress_callback(percent, message):
                self.root.after(0, lambda p=percent, m=message: self._update_progress(p, m))

            self._rate_frame(self.frozen_frame, self.frozen_result, progress_callback)
            self.root.after(0, self._on_rating_complete)
        except Exception as exc:
            self.root.after(0, lambda: self._on_rating_error(str(exc)))

    def _update_progress(self, percent, message):
        self.progress_bar["value"] = percent
        self.progress_label.config(text=message)

    def _on_rating_complete(self):
        self._stop_loading_animation()
        self._show_opencv_image(self.frozen_result)
        self.progress_bar["value"] = 100
        self.progress_label.config(text="打分完成")
        if self.last_total_score > 0:
            self._update_status(f"评分完成，总分: {self.last_total_score:.1f}")
            self.score_status_label.config(text=f"已评分 {self.last_total_score:.1f}分", foreground="#007f00")
        else:
            self._update_status("评分失败，请重试")
            self.score_status_label.config(text="评分失败", foreground="#d63031")

    def _on_rating_error(self, error_msg):
        self._stop_loading_animation()
        self._update_status("评分出错")
        self.score_status_label.config(text="评分失败", foreground="#d63031")
        self.progress_bar["value"] = 0
        self.progress_label.config(text="打分出错")
        print(f"评分线程出错: {error_msg}")

    # ---------- 显示 / 保存 ----------
    def _show_opencv_image(self, image, keep_aspect=True):
        if image is None:
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        cw = max(self.canvas.winfo_width(), 2)
        ch = max(self.canvas.winfo_height(), 2)
        if keep_aspect:
            iw, ih = img.size
            ratio = min(cw / iw, ch / ih)
            nw, nh = max(1, int(iw * ratio)), max(1, int(ih * ratio))
            if nw != iw or nh != ih:
                img = img.resize((nw, nh), Image.LANCZOS)
        else:
            img = img.resize((cw, ch), Image.LANCZOS)

        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo)

    def _refresh_canvas(self):
        if self.result_image is not None:
            self._show_opencv_image(self.result_image)

    def save_image_only(self):
        """导出照片：只保存检测结果图（含人脸框和标注）。"""
        image_to_save = self.frozen_result if self.frozen_result is not None else self.result_image
        if image_to_save is None:
            messagebox.showinfo("提示", "没有可导出的照片。")
            return
        default = "result_face.jpg"
        if hasattr(self, "image_path") and self.image_path:
            default = "output_" + os.path.basename(self.image_path)
            default = os.path.splitext(default)[0] + ".jpg"
        path = filedialog.asksaveasfilename(
            title="导出照片",
            defaultextension=".jpg",
            initialfile=default,
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("BMP", "*.bmp")],
        )
        if not path:
            return
        try:
            save_image(path, image_to_save)
            self._update_status(f"已导出照片: {path}")
            messagebox.showinfo("成功", f"照片已导出到:\n{path}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def save_data_only(self):
        """导出数据：保存各项检测与评分结果为 JSON。"""
        if not self.last_sub_scores or self.last_total_score <= 0:
            messagebox.showinfo("提示", "没有可导出的评分数据，请先完成评分。")
            return
        default = "result_data.json"
        if hasattr(self, "image_path") and self.image_path:
            default = "output_" + os.path.splitext(os.path.basename(self.image_path))[0] + ".json"
        path = filedialog.asksaveasfilename(
            title="导出数据",
            defaultextension=".json",
            initialfile=default,
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            import datetime
            data = {
                "export_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": getattr(self, "image_path", None) or "",
                "total_score": round(self.last_total_score, 2),
                "sub_scores": {k: round(v, 2) for k, v in self.last_sub_scores.items()},
                "features": {},
            }
            features = getattr(self, "last_features", {}) or {}
            for key, val in features.items():
                if isinstance(val, (int, float)):
                    data["features"][key] = round(float(val), 4)
                elif isinstance(val, str):
                    data["features"][key] = val
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._update_status(f"已导出数据: {path}")
            messagebox.showinfo("成功", f"数据已导出到:\n{path}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def run_egg(self):
        """彩蛋：运行当前目录下的 .bat 程序。"""
        import glob
        bat_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.bat"))
        if not bat_files:
            messagebox.showinfo("彩蛋", "当前目录下未找到 .bat 文件。")
            return
        bat_path = bat_files[0]
        try:
            os.startfile(bat_path)
            self._update_status(f"已运行彩蛋: {os.path.basename(bat_path)}")
        except Exception as exc:
            messagebox.showerror("运行失败", str(exc))

    def clear_display(self):
        self._stop_camera()
        self.current_image = None
        self.result_image = None
        self.frozen_frame = None
        self.frozen_result = None
        self.faces = []
        self.last_features = {}
        self.last_sub_scores = {}
        self.last_total_score = 0.0
        self.multi_face_warned = False
        self.canvas.delete("all")
        self.info_label.config(text="未检测", foreground="#555555")
        self._update_score_ui("idle", 0)
        self._update_status("已清除")

    def _update_status(self, text):
        self.status_var.set(text)

    def on_close(self):
        self._stop_loading_animation()
        self._stop_camera()
        self.root.destroy()

    # ---------- 说明 / 开源网址 ----------
    def show_help(self):
        help_text = (
            "【人脸颜值评分系统 - 使用说明】\n\n"
            "一、基本操作\n"
            "  1. 打开图片：加载本地图片文件进行人脸检测\n"
            "  2. 开始摄像头：启动实时摄像头模式\n"
            "  3. 评分：对当前画面定格并进行颜值评分\n"
            "  4. 导出照片：保存当前检测结果图片\n"
            "  5. 导出数据：保存各项检测与评分结果（JSON格式）\n"
            "  6. 彩蛋：运行当前目录下的 .bat 程序\n"
            "  7. 清除画面：清空当前显示内容\n\n"
            "二、评分流程\n"
            "  1. 程序会自动检测画面中的人脸并绘制框\n"
            "  2. 点击「评分」按钮定格最近一张捕捉到人脸的画面\n"
            "  3. 程序会在后台线程分析 478 个人脸特征点\n"
            "  4. 从 7 个维度对人脸进行打分\n\n"
            "三、注意事项\n"
            "  · 画面中出现多张人脸时会停止打分\n"
            "  · 调整左侧 scaleFactor / minNeighbors / minSize 可优化检测效果\n"
            "  · 基于医学面部美学比例评分，结果仅供参考\n"
            "  · 首次使用会自动下载 MediaPipe 模型文件\n"
            "四、作者邮箱\n"
            "gaotingxiangqqcom@qq.com"
        )
        messagebox.showinfo("程序说明", help_text)

    def show_open_source_links(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("开源网址")
        dialog.geometry("360x160")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(
            dialog, text="项目开源地址：", font=("Arial", 10, "bold"),
        ).pack(pady=(16, 8))

        gitee_url = "https://gitee.com/gaotingxiang"
        github_url = "https://github.com/GAOTINGXIANG"

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill="x", padx=16, pady=4)

        ttk.Button(btn_frame, text="Gitee 仓库",
                   command=lambda: webbrowser.open(gitee_url)).pack(side="left", expand=True, fill="x", padx=4)
        ttk.Button(btn_frame, text="GitHub 仓库",
                   command=lambda: webbrowser.open(github_url)).pack(side="left", expand=True, fill="x", padx=4)

        ttk.Button(dialog, text="关闭", command=dialog.destroy).pack(pady=(12, 8))

    # ---------- 检测更新 ----------
    def check_update(self):
        """点击「检查更新」按钮，在新线程中获取 Gitee 最新版本。"""
        self._update_status("正在检查更新...")
        self._check_update_btn_state("disabled")
        thread = threading.Thread(target=self._check_update_async, daemon=True)
        thread.start()

    def _check_update_async(self):
        """后台线程：请求 Gitee API 获取最新 release 信息。"""
        try:
            req = urllib.request.Request(
                GITEE_API_RELEASES,
                headers={"Accept": "application/json", "User-Agent": "AppearanceScoring/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            # Gitee API 返回的 tag_name 字段即版本号
            latest_tag = data.get("tag_name", "")
            # 去掉版本号前的 'v' 前缀便于比较
            latest_ver = latest_tag.lstrip("v")
            self.root.after(0, self._on_update_checked, latest_ver)
        except Exception as exc:
            self.root.after(0, self._on_update_error, str(exc))

    def _on_update_checked(self, latest_ver):
        """在主线程显示版本对比结果。"""
        self._check_update_btn_state("normal")
        self._update_status("检查更新完成")

        # 简单版本比较：去除前缀 'v' 后比较
        def normalize(v):
            return re.sub(r"[^0-9.]", "", v)

        cur = normalize(CURRENT_VERSION)
        lat = normalize(latest_ver)

        # 分割比较
        def parse(v):
            return [int(x) for x in v.split(".") if x.isdigit()]

        cur_parts = parse(cur)
        lat_parts = parse(lat)

        # 补齐长度
        max_len = max(len(cur_parts), len(lat_parts))
        cur_parts += [0] * (max_len - len(cur_parts))
        lat_parts += [0] * (max_len - len(lat_parts))

        is_newer = lat_parts > cur_parts

        if is_newer:
            self._show_update_dialog(latest_ver)
        else:
            messagebox.showinfo(
                "检查更新",
                f"当前版本：v{CURRENT_VERSION}\n"
                f"最新版本：v{latest_ver}\n\n"
                f"已是最新版本，无需更新。",
            )

    def _show_update_dialog(self, latest_ver):
        """发现新版本时弹出确认对话框。"""
        dialog = tk.Toplevel(self.root)
        dialog.title("发现新版本")
        dialog.geometry("360x160")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog, text=f"发现新版本：v{latest_ver}",
            font=("Arial", 12, "bold"),
        ).pack(pady=(20, 8))
        tk.Label(
            dialog, text=f"当前版本：v{CURRENT_VERSION}",
            font=("Arial", 9),
            foreground="#666666",
        ).pack()
        tk.Label(
            dialog, text="是否前往下载更新？",
            font=("Arial", 9),
        ).pack(pady=(8, 16))

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack()
        ttk.Button(
            btn_frame, text="前往下载",
            command=lambda: [webbrowser.open(RELEASE_PAGE_URL), dialog.destroy()],
        ).pack(side="left", padx=8)
        ttk.Button(btn_frame, text="暂不更新", command=dialog.destroy).pack(side="left", padx=8)

    def _on_update_error(self, error_msg):
        """检查更新失败时恢复按钮状态并提示。"""
        self._check_update_btn_state("normal")
        self._update_status("检查更新失败")
        messagebox.showwarning("检查更新", f"无法获取最新版本信息。\n\n{error_msg}")

    def _check_update_btn_state(self, state):
        """临时禁用 / 恢复「检查更新」按钮，防止重复点击。"""
        # 遍历子控件找到检查更新按钮（其父 Frame 中第3个 Button）
        pass  # 按钮状态由 status_var 描述，此处无需单独控制

    # ---------- 特征 / 评分 UI 更新 ----------
    def _update_score_ui(self, state, face_count):
        # 总分
        if state == "ok":
            total = self.last_total_score
            tag = _score_tag(total)
            bgr = _score_color(total)
            color = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"
            self.total_score_label.config(text=f"{total:.1f} / 100  {tag}", fg=color)
            self.score_status_label.config(text="单张人脸，已评分", foreground="#007f00")
        elif state == "multi":
            self.total_score_label.config(text=f"多脸(×{face_count})  已停止", fg="#d63031")
            self.score_status_label.config(text=f"{face_count} 张人脸，停止打分", foreground="#d63031")
        elif state == "disabled":
            self.total_score_label.config(text="已禁用", fg="#999999")
            self.score_status_label.config(text="评分已关闭", foreground="#999999")
        else:
            self.total_score_label.config(text="-- / 100", fg="#999999")
            self.score_status_label.config(text="等待检测", foreground="#555555")

        # 各项子分
        for key, (label, max_val) in SCORE_LABELS.items():
            pb, val_label = self.sub_score_labels[key]
            v = float(self.last_sub_scores.get(key, 0.0))
            pb["value"] = v
            val_label.config(text=f"{v:.1f}/{max_val}")

        # 特征值
        for key, (label, hint) in FEATURE_LABELS.items():
            if key in self.last_features:
                self.feature_labels[key].config(
                    text=f"{self.last_features[key]}", foreground="#2c3e50",
                )
            else:
                self.feature_labels[key].config(text="--", foreground="#999999")

    # ---------- 打开 AI 评分窗口 ----------
    def open_ai_window(self):
        api_url = self.ai_api_url.get().strip()
        model_id = self.ai_model_id.get().strip()
        token = self.ai_token.get().strip()

        if not api_url:
            messagebox.showwarning("配置不完整", "请填写 API 地址")
            return
        if not model_id:
            messagebox.showwarning("配置不完整", "请填写模型 ID")
            return
        if not token:
            messagebox.showwarning("配置不完整", "请填写 Token 密钥")
            return

        AIScoringWindow(self.root, api_url, model_id, token)
