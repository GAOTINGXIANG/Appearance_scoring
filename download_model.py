"""
人脸颜值评分系统 - 模型文件下载工具
================================================
下载 MediaPipe FaceLandmarker 模型文件 (face_landmarker.task, 约 3.7 MB)

使用方法:
    python download_model.py              # 下载到脚本所在目录
    python download_model.py C:\\path\\   # 下载到指定目录
"""

import os
import sys
import urllib.request


MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MIN_SIZE = 3_500_000  # 约 3.5 MB


MIRROR_URLS = [
    MODEL_URL,
    "https://github.com/google-ai-edge/mediapipe/raw/refs/heads/master/mediapipe/tasks/python/vision/face_landmarker/face_landmarker.task",
]


def download(url, target_path):
    """带进度条下载单个 URL"""
    block_size = 8192
    downloaded = 0

    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 face-scoring-app"}
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        total_mb = total / (1024 * 1024) if total else 0

        with open(target_path, "wb") as f:
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    percent = int(downloaded / total * 100)
                    mb = downloaded / (1024 * 1024)
                    bar = "#" * int(40 * percent / 100) + "-" * int(40 * (1 - percent / 100))
                    print(
                        f"\r  [{bar}] {percent:3d}%  {mb:.1f}/{total_mb:.1f} MB",
                        end="",
                        flush=True,
                    )
                else:
                    mb = downloaded / (1024 * 1024)
                    print(f"\r  已下载: {mb:.1f} MB", end="", flush=True)

    print()
    return os.path.getsize(target_path)


def main():
    if len(sys.argv) >= 2:
        target_dir = sys.argv[1]
    else:
        target_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, MODEL_FILENAME)

    print("=" * 60)
    print("  人脸颜值评分系统 - 模型文件下载工具")
    print("=" * 60)

    if os.path.exists(target_path) and os.path.getsize(target_path) >= MIN_SIZE:
        print(f"\n模型文件已存在: {target_path}")
        print(f"大小: {os.path.getsize(target_path):,} bytes")
        print("如需重新下载，请先删除现有文件。")
        return 0

    print(f"\n保存位置: {target_path}")
    print(f"尝试下载（共 {len(MIRROR_URLS)} 个镜像源）...\n")

    last_err = None
    for i, url in enumerate(MIRROR_URLS, 1):
        print(f"[{i}/{len(MIRROR_URLS)}] 尝试: {url[:70]}{'...' if len(url) > 70 else ''}")
        try:
            size = download(url, target_path)
            if size >= MIN_SIZE:
                print(f"\n✓ 下载成功！ ({size:,} bytes)")
                print(f"\n模型已保存到: {target_path}")
                print("现在可以运行: python face_gui.py")
                return 0
            else:
                print(f"  文件过小 ({size} bytes)，尝试下一个镜像源...")
                if os.path.exists(target_path):
                    os.remove(target_path)
        except Exception as exc:
            last_err = exc
            print(f"  ✗ 失败: {exc}")
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except Exception:
                    pass
            continue

    print()
    print("=" * 60)
    print("  所有自动下载方式均失败。")
    print(f"  请手动下载: {MODEL_URL}")
    print(f"  保存为: {target_path}")
    print()
    print("  其他手动下载方式:")
    print("    1) 使用浏览器打开上面的 URL")
    print("    2) 使用命令: curl -o face_landmarker.task \"" + MODEL_URL + "\"")
    print("    3) 使用命令: powershell Invoke-WebRequest -Uri \"" + MODEL_URL + "\" -OutFile face_landmarker.task")
    print("    4) 从其他镜像源（如 huggingface）搜索 mediapipe face_landmarker")
    if last_err:
        print(f"\n最近的错误: {last_err}")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
