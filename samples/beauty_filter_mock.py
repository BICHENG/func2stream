"""
实时美颜程序 Mock 示例
与 README.md 开头的示例一致
"""
import time
import numpy as np
from func2stream import Pipeline
from func2stream.core import DataSource


# ─── Mock 摄像头 ───────────────────────────────────────────────
class MockCamera:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.frame_count = 0
    
    def read(self):
        self.frame_count += 1
        time.sleep(0.033)  # ~30 FPS
        return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)


camera = MockCamera()


# ─── 工具函数（不进流水线，不加 -> "数据名"）─────────────────

def normalize(img):
    """归一化图像"""
    return img.astype(np.float32) / 255.0


def get_eye_positions(x, y, w, h):
    """计算眼睛位置"""
    return [(x + w//3, y + h//3), (x + 2*w//3, y + h//3)]


def apply_brightness(crop, factor=0.9):
    """调整亮度"""
    return crop * factor + (1 - factor)


# ─── 流水线函数（与文档一致）─────────────────────────────────────

def some_preprocess(frame) -> "tensor":
    """预处理：归一化"""
    return normalize(frame)  # 调用工具函数


def detect(tensor) -> "face_boxes":
    """人脸检测（mock：返回固定框）"""
    h, w = tensor.shape[:2]
    return [(w//4, h//4, w//2, h//2)]  # (x, y, w, h)


def get_face_crop(tensor, face_boxes) -> ("landmarks", "crops"):
    """获取人脸区域和关键点"""
    crops = []
    landmarks = []
    for (x, y, w, h) in face_boxes:
        crop = tensor[y:y+h, x:x+w]
        crops.append(crop)
        landmarks.append(get_eye_positions(x, y, w, h))  # 调用工具函数
    return landmarks, crops


def beautify_model(crops) -> "beautified":
    """美颜模型（mock：简单模糊）"""
    return [apply_brightness(crop) for crop in crops]  # 调用工具函数


def display(frame, landmarks, beautified) -> "displayed":
    """显示结果"""
    print(f"[display] frame shape: {frame.shape}, "
          f"landmarks: {len(landmarks)}, beautified: {len(beautified)}")
    return True


# ─── 组装流水线（与文档一致）───────────────────────────────────

pipeline = Pipeline([
    DataSource(camera.read),
    some_preprocess,
    detect,
    get_face_crop,
    beautify_model,
    display,
])


if __name__ == "__main__":
    print("启动美颜流水线...")
    pipeline.start()
    
    # 运行 3 秒
    time.sleep(3)
    
    print(f"\n处理帧数: {camera.frame_count}")
    pipeline.exec_time_summary_lite()
    pipeline.stop()
    print("流水线已停止")
