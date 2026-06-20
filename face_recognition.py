import sys
import os
import cv2


CASCADE_FILENAMES = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt_tree.xml",
]


def load_cascade():
    for filename in CASCADE_FILENAMES:
        path = cv2.data.haarcascades + filename
        if os.path.exists(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade, path
    return None, None


def detect_in_image(image_path):
    cascade, cascade_path = load_cascade()
    if cascade is None:
        print("错误: 无法加载人脸分类器，请检查 OpenCV 安装。")
        sys.exit(1)
    print(f"使用分类器: {cascade_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图片 '{image_path}'。")
        sys.exit(1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    print(f"检测到 {len(faces)} 张人脸。")

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    output_path = "output_" + os.path.basename(image_path)
    cv2.imwrite(output_path, image)
    print(f"结果已保存到: {output_path}")

    cv2.imshow("Face Detection - 按任意键退出", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_in_video():
    cascade, cascade_path = load_cascade()
    if cascade is None:
        print("错误: 无法加载人脸分类器，请检查 OpenCV 安装。")
        sys.exit(1)
    print(f"使用分类器: {cascade_path}")
    print("打开摄像头... 按 q 键退出。")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头。")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            f"Faces: {len(faces)}  (press q to quit)",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Face Detection (Camera)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python face_recognition.py camera          # 使用摄像头实时检测")
        print("  python face_recognition.py <图片路径>      # 检测单张图片中的人脸")
        sys.exit(1)

    arg = sys.argv[1]
    if arg.lower() in ("camera", "cam", "0", "webcam"):
        detect_in_video()
    else:
        detect_in_image(arg)


if __name__ == "__main__":
    main()
