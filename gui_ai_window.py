"""
人脸颜值评分系统 - AI 评分窗口 GUI 模块

封装 AI 对话窗口界面，用户可与 AI 进行文本对话，
也可上传图片进行颜值评分。
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from ai_client import AIClient, is_openai_available
from utils import load_image


# ===================== AI 评分窗口 =====================
class AIScoringWindow:
    """AI 对话框窗口类。"""

    # ---------- 初始化 ----------
    def __init__(self, parent, api_url, model_id, token):
        self.api_url = api_url
        self.model_id = model_id
        self.token = token
        self.current_image_path = None
        self.current_image = None
        self._photo = None
        self.ai_client = None
        self.messages = []  # 对话历史

        if not is_openai_available():
            messagebox.showerror(
                "缺少依赖",
                "请先安装 openai 库：\npip install openai>=1.0.0",
            )
            return

        # 创建 Toplevel 子窗口
        self.window = tk.Toplevel(parent)
        self.window.title("AI 颜值评分助手")
        self.window.geometry("900x650")
        self.window.minsize(700, 500)

        self._build_ui()
        self._init_client()

    # ---------- 客户端 ----------
    def _init_client(self):
        """初始化 AI 客户端并更新状态。"""
        try:
            self.ai_client = AIClient(
                api_url=self.api_url,
                model_id=self.model_id,
                token=self.token,
                temperature=float(self.temp_var.get()),
                top_p=float(self.top_p_var.get()),
            )
            self.status_label.config(text="已连接到 AI 服务", foreground="#007f00")
        except Exception as exc:
            self.status_label.config(text=f"连接失败: {exc}", foreground="#d63031")

    # ---------- UI 构建 ----------
    def _build_ui(self):
        main = ttk.Frame(self.window, padding=8)
        main.pack(fill="both", expand=True)

        self._build_config_bar(main)
        self._build_content(main)
        self._build_bottom_bar(main)

        self._add_message("system", "已连接到 AI 服务，请输入消息或发送图片进行评分")

    def _build_config_bar(self, parent):
        top = ttk.LabelFrame(parent, text="当前配置", padding=6)
        top.pack(fill="x", pady=(0, 6))
        api_text = self.api_url[:50] + ("..." if len(self.api_url) > 50 else "")
        ttk.Label(top, text=f"API 地址: {api_text}", font=("Arial", 8)).pack(anchor="w")
        ttk.Label(top, text=f"模型 ID: {self.model_id}", font=("Arial", 8)).pack(anchor="w")

    def _build_content(self, parent):
        content = ttk.Frame(parent)
        content.pack(fill="both", expand=True, pady=(0, 6))

        self._build_image_preview(content)
        self._build_chat_panel(content)

    def _build_image_preview(self, parent):
        left = ttk.LabelFrame(parent, text="图片预览", padding=5)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.canvas = tk.Canvas(left, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

    def _build_chat_panel(self, parent):
        right = ttk.LabelFrame(parent, text="AI 对话", padding=5)
        right.pack(side="right", fill="both", expand=True, padx=(5, 0))

        # 消息显示区
        msg_frame = ttk.Frame(right)
        msg_frame.pack(fill="both", expand=True, pady=(0, 5))
        self.msg_text = tk.Text(
            msg_frame, wrap="word", font=("Arial", 10),
            state="disabled", bg="#f8f8f8",
        )
        self.msg_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(msg_frame, command=self.msg_text.yview)
        scroll.pack(side="right", fill="y")
        self.msg_text.config(yscrollcommand=scroll.set)

        # tag 配置
        self.msg_text.tag_configure("user", foreground="#0066cc", justify="right")
        self.msg_text.tag_configure("ai", foreground="#333333", justify="left")
        self.msg_text.tag_configure("system", foreground="#888888", justify="center")
        self.msg_text.tag_configure("error", foreground="#cc0000", justify="left")

        # 输入区
        input_frame = ttk.Frame(right)
        input_frame.pack(fill="x")
        self.input_entry = ttk.Entry(input_frame, font=("Arial", 10))
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.input_entry.bind("<Return>", lambda e: self.send_message())
        self.send_btn = ttk.Button(input_frame, text="发送", width=8, command=self.send_message)
        self.send_btn.pack(side="right")

    def _build_bottom_bar(self, parent):
        bottom = ttk.Frame(parent)
        bottom.pack(fill="x")

        # 按钮
        btn_frame = ttk.Frame(bottom)
        btn_frame.pack(side="left")
        ttk.Button(btn_frame, text="选择图片", width=10, command=self.select_image).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="发送图片评分", width=12, command=self.send_image_for_rating).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="清空对话", width=10, command=self.clear_chat).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="说明", width=8, command=self.show_help).pack(side="left", padx=2)

        # 参数调节
        param_frame = ttk.Frame(bottom)
        param_frame.pack(side="left", padx=20)

        ttk.Label(param_frame, text="temperature:", font=("Arial", 8)).pack(side="left")
        self.temp_var = tk.DoubleVar(value=0.8)
        ttk.Spinbox(param_frame, from_=0.0, to=2.0, width=4,
                    textvariable=self.temp_var, format="%.1f").pack(side="left", padx=2)

        ttk.Label(param_frame, text="top_p:", font=("Arial", 8)).pack(side="left")
        self.top_p_var = tk.DoubleVar(value=0.7)
        ttk.Spinbox(param_frame, from_=0.0, to=1.0, width=4,
                    textvariable=self.top_p_var, format="%.1f").pack(side="left", padx=2)

        # 状态显示
        self.status_label = ttk.Label(bottom, text="就绪", foreground="#555555")
        self.status_label.pack(side="right")

    # ---------- 消息文本操作 ----------
    def _add_message(self, tag, text):
        def _do():
            self.msg_text.config(state="normal")
            self.msg_text.insert("end", text + "\n\n", tag)
            self.msg_text.see("end")
            self.msg_text.config(state="disabled")
        self.window.after(0, _do)

    # ---------- 图片加载/显示 ----------
    def select_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            self.current_image_path = path
            self.current_image = load_image(path)
            self._show_image(self.current_image)
            self.status_label.config(
                text=f"已加载: {os.path.basename(path)}", foreground="#007f00"
            )
            self._add_message("system", f"已加载图片: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("错误", f"加载图片失败: {exc}")

    def _show_image(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        iw, ih = img.size
        ratio = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * ratio)), max(1, int(ih * ratio))
        if nw != iw or nh != ih:
            img = img.resize((nw, nh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo)

    # ---------- 发送文本消息 ----------
    def send_message(self):
        if self.ai_client is None:
            messagebox.showerror("错误", "AI 客户端未初始化")
            return

        user_input = self.input_entry.get().strip()
        if not user_input:
            return
        self.input_entry.delete(0, "end")

        self.messages.append({"role": "user", "content": user_input})
        self._add_message("user", f"用户: {user_input}")

        self._lock_ui("AI 正在思考...")
        threading.Thread(target=self._call_chat, daemon=True).start()

    def _call_chat(self):
        ok, content = self.ai_client.chat(self.messages)
        if ok:
            self.messages.append({"role": "assistant", "content": content})
            self._add_message("ai", f"AI: {content}")
            self.window.after(0, lambda: self._unlock_ui(True))
        else:
            self._add_message("error", content)
            self.window.after(0, lambda: self._unlock_ui(False))

    # ---------- 发送图片评分 ----------
    def send_image_for_rating(self):
        if self.current_image is None:
            messagebox.showinfo("提示", "请先选择图片")
            return
        if self.ai_client is None:
            messagebox.showerror("错误", "AI 客户端未初始化")
            return

        self._lock_ui("AI 正在评分...")
        threading.Thread(target=self._call_image_rating, daemon=True).start()

    def _call_image_rating(self):
        ok, content = self.ai_client.chat_with_image(self.messages, self.current_image)
        if ok:
            # 注意：chat_with_image 不会自动追加 user 消息到 history，
            # 这里手动保持上下文连贯性。
            self.messages.append({
                "role": "user",
                "content": "[图片]",
            })
            self.messages.append({"role": "assistant", "content": content})
            self._add_message("system", "已发送图片，请等待 AI 评分...")
            self._add_message("ai", f"AI 评分结果:\n{content}")
            self.window.after(0, lambda: self._unlock_ui(True))
        else:
            self._add_message("error", content)
            self.window.after(0, lambda: self._unlock_ui(False))

    # ---------- UI 锁定 / 解锁 ----------
    def _lock_ui(self, status_text):
        self.send_btn.config(state="disabled", text="...")
        self.status_label.config(text=status_text, foreground="#555555")

    def _unlock_ui(self, success):
        self.send_btn.config(state="normal", text="发送")
        if success:
            self.status_label.config(text="就绪", foreground="#007f00")
        else:
            self.status_label.config(text="请求失败", foreground="#d63031")

    # ---------- 其他操作 ----------
    def clear_chat(self):
        self.messages = []
        self.msg_text.config(state="normal")
        self.msg_text.delete("1.0", "end")
        self.msg_text.config(state="disabled")
        self._add_message("system", "对话已清空")

    def show_help(self):
        help_text = (
            "【AI 颜值评分助手 - 使用说明】\n\n"
            "一、界面布局\n"
            "  · 左侧：图片预览区域，显示当前选中的图片\n"
            "  · 右侧：AI 对话区域，显示对话历史和输入框\n"
            "  · 底部：操作按钮、参数调节和状态信息\n\n"
            "二、基本操作\n"
            "  1. 选择图片：打开本地图片文件，用于 AI 评分\n"
            "  2. 发送图片评分：将图片发送给 AI 进行颜值打分\n"
            "  3. 清空对话：清空当前对话历史，重新开始\n"
            "  4. 文本对话：在输入框中输入内容后按回车键发送\n\n"
            "三、参数说明\n"
            "  · temperature (0.0 - 2.0)：控制 AI 回答的创造性\n"
            "  · top_p (0.0 - 1.0)：控制采样范围\n\n"
            "四、调用方式\n"
            "  采用 OpenAI 兼容的 SDK 调用：\n"
            "  client.chat.completions.create(...)\n\n"
            "五、注意事项\n"
            "  · 图片评分需要模型支持多模态输入（Vision）\n"
            "  · 对话历史会随每次调用发送，用于多轮交互"
        )
        messagebox.showinfo("AI 模式说明", help_text)
