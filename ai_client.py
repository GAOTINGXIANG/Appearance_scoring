"""
人脸颜值评分系统 - AI 客户端模块

封装 OpenAI 兼容 SDK 的 API 调用逻辑，支持：
  - 文本对话（单轮 / 多轮上下文）
  - 图片评分（Vision 多模态输入）
"""

import base64
import io

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ===================== AI 服务连接检查 =====================
def is_openai_available():
    """判断 openai 库是否已安装。"""
    return OpenAI is not None


# ===================== OpenAI 兼容客户端 =====================
class AIClient:
    """封装 OpenAI 兼容 SDK 调用。"""

    def __init__(self, api_url, model_id, token,
                 temperature=0.8, top_p=0.7, system_prompt=None):
        if not is_openai_available():
            raise RuntimeError(
                "缺少 openai 库，请先运行：pip install openai>=1.0.0"
            )
        self.api_url = api_url
        self.model_id = model_id
        self.token = token
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt or "你是一个专业的颜值评分助手。"

        # 创建底层客户端
        self._client = OpenAI(api_key=token, base_url=api_url)

    # ---------- 核心调用 ----------
    def _build_messages(self, history):
        """按系统提示 + 历史消息组装 messages。"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)
        return messages

    def chat(self, history):
        """文本对话调用，返回 (is_ok, response_text_or_error)。"""
        try:
            completion = self._client.chat.completions.create(
                model=self.model_id,
                messages=self._build_messages(history),
                temperature=self.temperature,
                top_p=self.top_p,
            )
            if not completion.choices:
                return False, "AI 未返回内容（choices 为空）"
            content = completion.choices[0].message.content
            if not content:
                return False, "AI 未返回文本内容"
            return True, content
        except Exception as exc:
            return False, f"请求出错: {exc}"

    def chat_with_image(self, history, image_bgr, prompt=None):
        """图片评分调用，传入 OpenCV BGR 图像，返回 (is_ok, response_text_or_error)。"""
        try:
            # 将 BGR 转为 PIL JPEG 字节流 -> base64
            try:
                from PIL import Image as _PILImage
            except ImportError:
                return False, "缺少 Pillow 库，无法处理图片"

            # 颜色空间转换
            import numpy as _np
            if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
                return False, "图像为空"

            rgb = image_bgr[:, :, ::-1]  # BGR -> RGB
            pil_img = _PILImage.fromarray(rgb)

            buf = io.BytesIO()
            try:
                pil_img.save(buf, format="JPEG")
            except Exception:
                # 若存在 RGBA / 异常，尝试降级保存
                pil_img.convert("RGB").save(buf, format="JPEG")
            img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # 构造多模态消息
            user_msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt or "请为这张图片中的人脸颜值进行打分，给出评分结果（0-100分）和简要评价。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ],
            }

            completion = self._client.chat.completions.create(
                model=self.model_id,
                messages=self._build_messages(history) + [user_msg],
                temperature=self.temperature,
                top_p=self.top_p,
            )
            if not completion.choices:
                return False, "AI 未返回内容"
            content = completion.choices[0].message.content
            if not content:
                return False, "AI 未返回文本内容"
            return True, content
        except Exception as exc:
            return False, f"图片评分出错: {exc}"
