"""
实时美颜程序 Mock 示例
与 README.md 开头的示例一致。
"""
import time

import numpy as np

from func2stream import DataSource, Pipeline


class MockCamera:
    def __init__(self, width=640, height=480, sleep=0.033, seed=None):
        self.width = width
        self.height = height
        self.sleep = sleep
        self.frame_count = 0
        self.rng = np.random.default_rng(seed)

    def read(self):
        self.frame_count += 1
        if self.sleep:
            time.sleep(self.sleep)
        return self.rng.integers(0, 255, (self.height, self.width, 3), dtype=np.uint8)


def normalize(img):
    """归一化图像"""
    return img.astype(np.float32) / 255.0


def get_eye_positions(x, y, w, h):
    """计算眼睛位置"""
    return [(x + w // 3, y + h // 3), (x + 2 * w // 3, y + h // 3)]


def apply_brightness(crop, factor=0.9):
    """调整亮度"""
    return crop * factor + (1 - factor)


def some_preprocess(frame) -> "tensor":
    """预处理：归一化"""
    return normalize(frame)


def detect(tensor) -> "face_boxes":
    """人脸检测（mock：返回固定框）"""
    h, w = tensor.shape[:2]
    return [(w // 4, h // 4, w // 2, h // 2)]


def get_face_crop(tensor, face_boxes) -> ("landmarks", "crops"):
    """获取人脸区域和关键点"""
    crops = []
    landmarks = []
    for (x, y, w, h) in face_boxes:
        crops.append(tensor[y:y + h, x:x + w])
        landmarks.append(get_eye_positions(x, y, w, h))
    return landmarks, crops


def beautify_model(crops) -> "beautified":
    """美颜模型（mock：简单亮度调整）"""
    return [apply_brightness(crop) for crop in crops]


def display(frame, landmarks, beautified) -> "displayed":
    """显示结果"""
    print(
        f"[display] frame shape: {frame.shape}, "
        f"landmarks: {len(landmarks)}, beautified: {len(beautified)}"
    )
    return True


if __name__ == "__main__":
    camera = MockCamera()
    pipeline = Pipeline([
        DataSource(camera.read),
        some_preprocess,
        detect,
        get_face_crop,
        beautify_model,
        display,
    ])
    print("启动美颜流水线...")
    pipeline.start()
    time.sleep(3)
    print(f"\n处理帧数: {camera.frame_count}")
    pipeline.exec_time_summary_lite()
    pipeline.stop()
    print("流水线已停止")
