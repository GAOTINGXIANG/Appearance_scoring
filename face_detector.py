import os
import sys
import cv2
import numpy as np

from config import (
    MODEL_FILENAME,
    MODEL_URL,
    MIN_MODEL_SIZE,
    CASCADE_FILENAMES,
    get_model_path,
)
from utils import score_tag, score_color, load_image, save_image

# 保持对旧版调用的兼容（face_gui.py 中以旧名使用）
_score_tag = score_tag
_score_color = score_color


# 日志回调函数
_log_callback = None


def set_log_callback(callback):
    """设置日志回调函数，用于接收打分进度信息。"""
    global _log_callback
    _log_callback = callback


def _log(msg):
    """输出日志信息。"""
    print(msg)
    if _log_callback:
        _log_callback(msg)


def _mirror_idx(idx):
    """FaceLandmarker 478 landmarks 左右镜像映射。"""
    MIRROR = {
        33: 263, 263: 33, 133: 362, 362: 133, 160: 385, 385: 160,
        159: 386, 386: 159, 158: 387, 387: 158, 161: 388, 388: 161,
        246: 466, 466: 246, 153: 397, 397: 153, 154: 398, 398: 154,
        155: 399, 399: 155, 145: 374, 374: 145, 233: 463, 463: 233,
        7: 296, 296: 7, 163: 383, 383: 163,
        70: 300, 300: 70, 71: 251, 251: 71, 107: 336, 336: 107,
        108: 337, 337: 108, 109: 338, 338: 109, 46: 276, 276: 46,
        53: 281, 281: 53, 52: 282, 282: 52,
        55: 285, 285: 55, 61: 291, 291: 61, 63: 289, 289: 63,
        13: 14, 14: 13,
    }
    return MIRROR.get(idx, idx)


def _get_model_path():
    """返回模型文件的绝对路径（脚本所在目录）。"""
    return get_model_path()


def _download_model(target_path=None, show_progress=True):
    """下载 face_landmarker.task 模型文件（约 3.7 MB）。"""
    import urllib.request

    if target_path is None:
        target_path = _get_model_path()

    if os.path.exists(target_path) and os.path.getsize(target_path) > MIN_MODEL_SIZE:
        if show_progress:
            print(f"模型已存在: {target_path} ({os.path.getsize(target_path)} bytes)")
        return target_path

    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)

    if show_progress:
        print(f"正在下载模型文件: {MODEL_FILENAME}")
        print(f"  来源: {MODEL_URL}")
        print(f"  保存到: {target_path}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, int(downloaded / total_size * 100)) if total_size > 0 else 0
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
        bar_len = 40
        filled = int(bar_len * percent / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {percent:3d}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)

    reporthook = _progress if show_progress else None
    urllib.request.urlretrieve(MODEL_URL, target_path, reporthook=reporthook)

    if show_progress:
        print()
        print(f"下载完成: {os.path.getsize(target_path)} bytes")

    actual_size = os.path.getsize(target_path)
    if actual_size < MIN_MODEL_SIZE:
        os.remove(target_path)
        raise RuntimeError(
            f"下载的文件异常过小 ({actual_size} bytes)，请检查网络或手动下载。\n"
            f"下载地址: {MODEL_URL}"
        )
    return target_path


class FaceDetector:
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=30):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.cascade = None
        self.cascade_path = None
        self._load_cascade()

    def _load_cascade(self):
        search_paths = []
        if hasattr(sys, '_MEIPASS'):
            search_paths.append(sys._MEIPASS)
        search_paths.append(os.path.dirname(os.path.abspath(__file__)))
        search_paths.append(cv2.data.haarcascades)
        for filename in CASCADE_FILENAMES:
            for base_dir in search_paths:
                path = os.path.join(base_dir, filename)
                if os.path.exists(path):
                    cascade = cv2.CascadeClassifier(path)
                    if not cascade.empty():
                        self.cascade = cascade
                        self.cascade_path = path
                        return
        raise RuntimeError("无法加载 Haar 分类器，请检查 OpenCV 安装。")

    # ---------- 检测 ----------
    def detect(self, frame):
        if self.cascade is None:
            raise RuntimeError("分类器未初始化。")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
        )
        return faces

    def draw_faces(self, frame, faces, show_score=False, score=None):
        result = frame.copy()
        for (x, y, w, h) in faces:
            color = (0, 255, 0) if len(faces) == 1 else (0, 165, 255)
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            label = f"{len(faces)} Faces" if len(faces) != 1 else "Face"
            cv2.putText(
                result, label, (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
        if show_score and len(faces) == 1 and score is not None:
            x, y, w, h = faces[0]
            tag = _score_tag(score)
            tag_color = _score_color(score)
            text = f"{score:.1f} / 100  {tag}"
            cv2.putText(result, text, (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, tag_color, 2)
        return result

    def detect_and_draw(self, frame):
        faces = self.detect(frame)
        return self.draw_faces(frame, faces), faces

    # ---------- 颜值特征提取 ----------
    def extract_features(self, frame, face_box, landmark_indices=None, progress_callback=None):
        """提取人脸特征，支持进度回调。
        
        Args:
            frame: BGR格式图像
            face_box: 人脸框 (x, y, w, h)
            landmark_indices: 可选，特征点索引列表
            progress_callback: 进度回调函数，签名为 callback(percent: int, message: str)
        """
        import mediapipe as _mp
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
        from mediapipe.tasks.python.core import base_options as mp_base_options
        from mediapipe import ImageFormat, Image

        def _report(percent, msg):
            _log(f"[{percent}%] {msg}")
            if progress_callback:
                progress_callback(percent, msg)

        _report(5, "检查模型文件...")

        model_path = _get_model_path()
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 3_500_000:
            _report(5, "未找到模型文件，正在下载...")
            try:
                _download_model(model_path, show_progress=True)
            except Exception as exc:
                _report(5, f"自动下载失败: {exc}，请手动下载：" + MODEL_URL)
                return {}
        else:
            _report(10, f"模型文件就绪: {os.path.basename(model_path)}")

        _report(15, "初始化人脸特征点检测器...")

        options = FaceLandmarkerOptions(
            base_options=mp_base_options.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        with FaceLandmarker.create_from_options(options) as detector:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
            _report(25, "检测人脸特征点...")
            result = detector.detect(mp_img)

        if not result.face_landmarks:
            _report(25, "未检测到人脸特征点")
            return {}

        _report(35, "人脸特征点检测完成")

        lm = result.face_landmarks[0]
        fx, fy, fw, fh = face_box
        frame_h, frame_w = frame.shape[:2]

        def to_pixel(idx):
            """将归一化坐标转换为相对于人脸 ROI 的像素坐标。"""
            pt = lm[idx]
            px = int(pt.x * frame_w)
            py = int(pt.y * frame_h)
            return (px - fx, py - fy)

        # 常用特征点（像素坐标，相对于 face_box）
        l_eye_outer = to_pixel(33)    # 左眼外角
        r_eye_outer = to_pixel(263)   # 右眼外角
        l_eye_inner = to_pixel(133)   # 左眼内角
        r_eye_inner = to_pixel(362)   # 右眼内角
        l_eye_top   = to_pixel(159)   # 左眼上缘
        l_eye_bot   = to_pixel(145)   # 左眼下缘
        r_eye_top   = to_pixel(386)   # 右眼上缘
        r_eye_bot   = to_pixel(374)   # 右眼下缘
        l_eyebrow_l = to_pixel(336)   # 左眉外侧
        l_eyebrow_r = to_pixel(296)   # 左眉内侧
        r_eyebrow_l = to_pixel(107)   # 右眉内侧
        r_eyebrow_r = to_pixel(70)    # 右眉外侧
        nose_tip    = to_pixel(4)     # 鼻尖
        nose_left   = to_pixel(275)   # 鼻翼左侧
        nose_right  = to_pixel(45)   # 鼻翼右侧
        mouth_l     = to_pixel(61)    # 嘴角左侧
        mouth_r     = to_pixel(291)   # 嘴角右侧
        mouth_top   = to_pixel(13)    # 上唇中心
        mouth_bot   = to_pixel(14)    # 下唇中心
        chin        = to_pixel(152)   # 下巴中心
        forehead    = to_pixel(10)    # 额头上部
        face_l      = to_pixel(234)   # 左侧颚边缘
        face_r      = to_pixel(454)   # 右侧颚边缘

        def dist(p1, p2):
            return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

        def mid(p1, p2):
            return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        # 辅助尺寸
        face_width  = dist(face_l, face_r)
        face_height = dist(forehead, chin)
        eye_distance = dist(l_eye_outer, r_eye_outer)  # 两眼中心距
        mouth_width  = dist(mouth_l, mouth_r)
        nose_width   = dist(nose_left, nose_right)

        l_eye_center = mid(l_eye_top, l_eye_bot)
        r_eye_center = mid(r_eye_top, r_eye_bot)
        l_eye_w = dist(l_eye_outer, l_eye_inner)
        r_eye_w = dist(r_eye_outer, r_eye_inner)
        l_eye_h = dist(l_eye_top, l_eye_bot)
        r_eye_h = dist(r_eye_top, r_eye_bot)
        avg_eye_h = (l_eye_h + r_eye_h) / 2.0

        # 人脸三庭（鼻尖为界）
        upper = dist(forehead, nose_tip)     # 上庭：额→鼻尖
        middle = dist(nose_tip, mouth_top)   # 中庭：鼻尖→上唇
        lower  = dist(mouth_bot, chin)       # 下庭：下唇→下巴

        nose_to_mouth = dist(nose_tip, mouth_top)
        mouth_to_chin = dist(mouth_bot, chin)

        # ===================== 1. 面部对称性（25分）=====================
        _report(40, "分析面部对称性...")

        # 计算所有 landmark 点与其镜像点的欧氏距离偏差
        symmetry_errors = []
        key_indices = [
            33, 133, 160, 159, 158, 161, 246,
            70, 71, 107, 108, 109, 46, 53, 52,
            55, 61, 63, 13, 14,
        ]
        for idx in key_indices:
            p = lm[idx]
            mir = _mirror_idx(idx)
            pm = lm[mir]
            err = np.hypot((p.x - pm.x) * frame_w, (p.y - pm.y) * frame_h)
            symmetry_errors.append(err)
        symmetry_error_avg = np.mean(symmetry_errors)

        # —— 新评分曲线：用指数衰减替代分段线性 ——
        # score = base + (1 - base) * exp(-error / scale)
        # - 误差 0 → 满分 1.0
        # - 误差 3 px → ≈ 0.90（接近满分）
        # - 误差 8 px → ≈ 0.80（良好）
        # - 误差 15 px → ≈ 0.66（可接受）
        # - 误差 30 px → ≈ 0.51（轻度不对称）
        # - 误差 50 px → ≈ 0.43（明显不对称，保底约 4+ 分）
        # base=0.35：下限约 8.75 分（25 分制），确保不会接近 0
        sym_scale = 22.0
        sym_base = 0.7
        symmetry_score = sym_base + (1.0 - sym_base) * float(
            np.exp(-symmetry_error_avg / sym_scale)
        )
        symmetry_score = max(0.35, min(1.0, symmetry_score))
        _report(45, f"对称性分析完成，误差: {symmetry_error_avg:.1f}px, 得分: {symmetry_score:.1%}")

        # ===================== 2. 皮肤质量（25分）=====================
        _report(50, "分析皮肤质量...")

        # 使用 YCrCb 色彩空间检测皮肤区域，并分析肤色均匀度
        face_roi = frame[fy:fy + fh, fx:fx + fw]
        if face_roi.size > 0:
            ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
            # 经验皮肤范围：Cr ∈ [133,173], Cb ∈ [77,127]
            mask = cv2.inRange(
                ycrcb,
                np.array([0, 133, 77], dtype=np.uint8),
                np.array([255, 173, 127], dtype=np.uint8),
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            skin_pixels = ycrcb[mask > 0]
            if skin_pixels.size > 10:
                cr_std = float(skin_pixels[:, 1].std())
                cb_std = float(skin_pixels[:, 2].std())
                color_variance = cr_std + cb_std
                # 优化：方差越小肤色越均匀，使用平滑曲线，避免0分
                # variance < 3 → 完美, variance > 12 → 保底
                if color_variance <= 3.0:
                    skin_evenness = 1.0
                elif color_variance <= 8.0:
                    skin_evenness = 0.7 + 0.3 * (8.0 - color_variance) / 5.0
                elif color_variance <= 15.0:
                    skin_evenness = 0.3 + 0.4 * (15.0 - color_variance) / 7.0
                else:
                    skin_evenness = 0.25  # 保底25%
                skin_coverage = mask.sum() / float(fw * fh)
                # 覆盖率 + 均匀度综合
                skin_quality = 0.6 * skin_evenness + 0.4 * min(skin_coverage / 0.5, 1.0)
            else:
                skin_evenness = 0.5
                skin_coverage = 0.3
                skin_quality = 0.5
        else:
            skin_evenness = 0.0
            skin_coverage = 0.0
            skin_quality = 0.0

        _report(60, f"皮肤质量分析完成，均匀度: {skin_evenness:.1%}")

        # ===================== 3. 面部比例/黄金比例（20分）=====================
        _report(65, "分析面部比例与五官分布...")
        # 黄金比例 φ ≈ 1.618
        PHI = 1.618
        phi_scores = []

        # 3a. 整体宽高比（理想 1:1.618）
        if face_height > 0:
            fwhr = face_width / face_height
            deviation = abs(fwhr - PHI)
            if deviation <= 0.5:
                phi_scores.append(1.0)
            else:
                phi_scores.append(max(0.25, 1.0 - (deviation - 0.5) / 0.8))

        # 3b. 三庭比例（上:中:下 ≈ 1:1:1）
        total_three = upper + middle + lower
        if total_three > 0:
            u_ratio = upper / total_three
            m_ratio = middle / total_three
            l_ratio = lower / total_three
            deviation = abs(u_ratio - 1/3) + abs(m_ratio - 1/3) + abs(l_ratio - 1/3)
            # 三庭偏离理想越远分数越低，但不会为0
            phi_scores.append(max(0.3, 1.0 - deviation * 1.2))

        # 3c. 鼻梁宽度/鼻翼宽度（理想鼻翼约为鼻梁的 1.5 倍）
        if nose_width > 0 and l_eye_inner[0] > 0:
            nose_bridge = dist(nose_tip, mid(to_pixel(51), to_pixel(281)))
            nb_to_nw = nose_bridge / nose_width if nose_width > 0 else 1.0
            deviation = abs(nb_to_nw - 0.6)
            if deviation <= 0.2:
                phi_scores.append(1.0)
            else:
                phi_scores.append(max(0.25, 1.0 - (deviation - 0.2) / 0.3))

        # 3d. 眼距/眼宽比例（两眼间距约为一只眼的宽度）
        avg_eye_w = (l_eye_w + r_eye_w) / 2.0
        if avg_eye_w > 0:
            eye_ratio = eye_distance / avg_eye_w
            # 优化：扩大理想范围 2.0~4.0，保底分数
            deviation = abs(eye_ratio - 3.0)
            if deviation <= 1.0:
                phi_scores.append(1.0)
            else:
                phi_scores.append(max(0.25, 1.0 - (deviation - 1.0) / 2.0))

        # 3e. 嘴宽/脸宽比例（理想 0.3~0.4）
        if face_width > 0:
            mw_to_fw = mouth_width / face_width
            # 优化：扩大理想范围，保底分数
            if 0.25 <= mw_to_fw <= 0.50:
                phi_scores.append(1.0)
            else:
                deviation = abs(mw_to_fw - 0.36)
                phi_scores.append(max(0.25, 1.0 - deviation / 0.2))

        proportion_score = float(np.mean(phi_scores)) if phi_scores else 0.5

        # ===================== 4. 眼睛有神度（10分）=====================
        # 眼高/眼宽比（眼高反映眼睛睁开程度，越大越有神）
        if l_eye_w > 0 and r_eye_w > 0:
            l_eye_ratio = l_eye_h / l_eye_w
            r_eye_ratio = r_eye_h / r_eye_w
            avg_eye_open = (l_eye_ratio + r_eye_ratio) / 2.0
            # 优化：扩大理想范围 0.25~0.55，保底分数
            if 0.22 <= avg_eye_open <= 0.55:
                eye_score = 1.0
            elif avg_eye_open < 0.22:
                eye_score = max(0.25, 0.25 + (0.22 - avg_eye_open) / 0.22 * 0.75)
            else:  # avg_eye_open > 0.55
                eye_score = max(0.25, 0.25 - (avg_eye_open - 0.55) / 0.5)
        else:
            eye_score = 0.4

        # ===================== 5. 嘴唇饱满度（10分）=====================
        # 嘴高/嘴宽（嘴唇高宽比反映饱满程度）
        lip_height = dist(mouth_top, mouth_bot)
        if mouth_width > 0:
            lip_ratio = lip_height / mouth_width
            # —— 新评分曲线：用高斯（指数衰减）平滑替代硬阈值 ——
            # 理想值：0.22（常见自然嘴唇高宽比）
            # σ=0.18：曲线较宽，在 0.05 ~ 0.55 之间都能拿到不低于 0.55 的基础分
            # 同时做保底：最小值 0.45，避免闭合嘴 / 张太大直接 0 分
            ideal_lip = 0.22
            sigma_lip = 0.18
            lip_raw = float(np.exp(-0.5 * ((lip_ratio - ideal_lip) / sigma_lip) ** 2))
            # 对过薄（<0.05）和过厚（>0.60）做轻微二次惩罚（曲线已足够宽容，这里只做保底）
            lip_score = max(0.45, lip_raw)
        else:
            lip_score = 0.5

        # ===================== 6. 下巴轮廓（5分）=====================
        # 下庭占比：下巴高度 / 下庭总高（下巴太小或太长都不好）
        lower_total = mouth_to_chin + nose_to_mouth
        if lower_total > 0:
            chin_ratio = mouth_to_chin / lower_total
            # 优化：扩大理想范围 0.30~0.70，保底分数
            if 0.28 <= chin_ratio <= 0.72:
                chin_score = 1.0
            elif chin_ratio < 0.28:
                chin_score = max(0.3, 0.3 + (0.28 - chin_ratio) / 0.28 * 0.7)
            else:  # chin_ratio > 0.72
                chin_score = max(0.3, 0.3 - (chin_ratio - 0.72) / 0.28)
        else:
            chin_score = 0.4

        # ===================== 7. 眉眼协调（5分）=====================
        # 左右眉高度差 + 眉眼距离与眼高的比例
        l_eyebrow_center = mid(l_eyebrow_l, l_eyebrow_r)
        r_eyebrow_center = mid(r_eyebrow_l, r_eyebrow_r)
        brow_height_diff = abs(l_eyebrow_center[1] - r_eyebrow_center[1])
        brow_to_eye = (dist(l_eyebrow_center, l_eye_center) +
                       dist(r_eyebrow_center, r_eye_center)) / 2.0
        if avg_eye_h > 0:
            brow_eye_ratio = brow_to_eye / avg_eye_h
            # —— 新评分曲线：两次平滑评分 + 加权合成，避免乘法导致0分 ——
            # ① 眉眼比例：理想 1.15（常见自然值），σ=0.55，极宽容错
            ideal_ratio = 1.15
            sigma_ratio = 0.55
            ratio_raw = float(np.exp(-0.5 * ((brow_eye_ratio - ideal_ratio) / sigma_ratio) ** 2))
            ratio_score = max(0.45, min(1.0, ratio_raw))

            # ② 眉高度差：用指数衰减，差值 0 → 1.0，30px → ≈0.60，60px → ≈0.40
            diff_scale = 40.0
            diff_base = 0.40
            diff_score = diff_base + (1.0 - diff_base) * float(
                np.exp(-brow_height_diff / diff_scale)
            )
            diff_score = max(0.40, min(1.0, diff_score))

            # ③ 加权合成（70% 比例 + 30% 高度差）—— 避免两个低分相乘 → 0
            brow_score = 0.7 * ratio_score + 0.3 * diff_score
            brow_score = max(0.45, min(1.0, brow_score))
        else:
            brow_score = 0.5

        # ===================== 汇总特征值 =====================
        _report(85, "汇总特征值...")

        features = {
            # 对称性
            "symmetry_error_px":  round(symmetry_error_avg, 2),
            "symmetry_score_raw": round(symmetry_score, 4),
            # 皮肤
            "skin_evenness":      round(skin_evenness, 4),
            "skin_coverage":       round(skin_coverage, 4),
            # 比例
            "face_width_height_ratio": round(face_width / face_height if face_height > 0 else 0, 3),
            "upper_ratio":        round(u_ratio if total_three > 0 else 0, 4),
            "middle_ratio":       round(m_ratio if total_three > 0 else 0, 4),
            "lower_ratio":        round(l_ratio if total_three > 0 else 0, 4),
            "eye_distance_ratio": round(eye_ratio if avg_eye_w > 0 else 0, 3),
            "mouth_width_ratio":  round(mw_to_fw if face_width > 0 else 0, 4),
            # 眼睛
            "eye_openness":       round(avg_eye_open if avg_eye_w > 0 else 0, 3),
            # 嘴唇
            "lip_fullness":       round(lip_ratio if mouth_width > 0 else 0, 3),
            # 下巴
            "chin_ratio":         round(chin_ratio if lower_total > 0 else 0, 3),
            # 眉毛
            "eyebrow_score_raw":  round(brow_score, 4),
            # 尺寸
            "face_width_px":      int(face_width),
            "face_height_px":     int(face_height),
        }

        _report(100, "特征提取完成！")
        return features

    # ---------- 颜值评分 ----------
    def compute_score(self, features):
        if not features:
            return 0.0, {}

        scores = {}

        # 1. 对称性 25分
        s1 = float(features.get("symmetry_score_raw", 0))
        scores["symmetry"] = round(s1 * 25.0, 1)

        # 2. 皮肤质量 25分
        s2 = float(features.get("skin_evenness", 0))
        scores["skin"] = round(s2 * 25.0, 1)

        # 3. 面部比例 20分
        # 多子项平均
        phi_items = []
        fwhr = features.get("face_width_height_ratio", 1.0)
        phi_items.append(max(0.0, 1.0 - abs(fwhr - 1.618) / 1.618))
        for key in ("upper_ratio", "middle_ratio", "lower_ratio"):
            v = features.get(key, 0.333)
            phi_items.append(max(0.0, 1.0 - abs(v - 0.333) * 3.0))
        mw_ratio = features.get("mouth_width_ratio", 0.36)
        phi_items.append(1.0 if 0.30 <= mw_ratio <= 0.42
                          else max(0.0, 1.0 - abs(mw_ratio - 0.36) / 0.3))
        ed_ratio = features.get("eye_distance_ratio", 3.0)
        phi_items.append(max(0.0, 1.0 - abs(ed_ratio - 3.0) / 2.0))
        s3 = float(np.mean(phi_items))
        scores["proportion"] = round(s3 * 20.0, 1)

        # 4. 眼睛有神度 10分
        eo = features.get("eye_openness", 0.35)
        s4 = 1.0 if 0.28 <= eo <= 0.48 else max(0.4, 1.0 - abs(eo - 0.38) / 0.3)
        scores["eye"] = round(s4 * 10.0, 1)

        # 5. 嘴唇饱满度 10分 —— 使用高斯平滑曲线，避免嘴巴闭合时 0 分
        lf = features.get("lip_fullness", 0.22)
        ideal_lf = 0.22
        sigma_lf = 0.20
        s5_raw = float(np.exp(-0.5 * ((lf - ideal_lf) / sigma_lf) ** 2))
        s5 = max(0.45, min(1.0, s5_raw))
        scores["lip"] = round(s5 * 10.0, 1)

        # 6. 下巴轮廓 5分
        cr = features.get("chin_ratio", 0.48)
        s6 = 1.0 if 0.35 <= cr <= 0.60 else max(0.4, 1.0 - abs(cr - 0.48) / 0.3)
        scores["chin"] = round(s6 * 5.0, 1)

        # 7. 眉眼协调 5分 —— 使用眉眼比例平滑曲线 + 保底，避免两个惩罚相乘为 0
        eb_ratio = features.get("eyebrow_score_raw", 0.6)
        s7_raw = float(np.exp(-0.5 * ((eb_ratio - 0.6) / 0.25) ** 2))
        s7 = max(0.45, min(1.0, s7_raw))
        scores["eyebrow"] = round(s7 * 5.0, 1)

        total = round(sum(scores.values()), 1)
        return total, scores

    # ---------- I/O ----------
    @staticmethod
    def load_image(path):
        """读取图片（支持中文路径）。"""
        return load_image(path)

    @staticmethod
    def save_image(path, image):
        """保存图片（支持中文路径）。"""
        return save_image(path, image)




