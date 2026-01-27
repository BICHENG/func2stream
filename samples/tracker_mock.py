"""
多目标追踪器 Mock 示例
与 README.md 中 @init_ctx 例子一致
"""
import time
import numpy as np
from func2stream import Pipeline, init_ctx
from func2stream.core import DataSource


# ─── Mock 摄像头 ───────────────────────────────────────────────

class MockCamera:
    def __init__(self, name, width=640, height=480):
        self.name = name
        self.width = width
        self.height = height
        self.frame_count = 0
    
    def read(self):
        self.frame_count += 1
        time.sleep(0.05)  # ~20 FPS
        return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)


front_camera = MockCamera("front")
rear_camera = MockCamera("rear")


# ─── Mock 模型和数据结构 ───────────────────────────────────────

class MockBox:
    def __init__(self, x, y, w, h, conf):
        self.bbox = (x, y, w, h)
        self.conf = conf


def load_model(model_path):
    """Mock 模型加载"""
    print(f"[mock] 加载模型: {model_path}")
    def model(frame):
        h, w = frame.shape[:2]
        return [
            MockBox(w//4, h//4, w//2, h//2, 0.8),
            MockBox(w//3, h//3, w//4, h//4, 0.4),
        ]
    return model


# ─── @init_ctx 创建有状态的追踪器（与 README 一致）─────────────

@init_ctx
def create_tracker(model_path, threshold=0.5):
    # ─── 状态：模型、计数器、追踪历史 ───────────────────────────
    model = load_model(model_path)
    frame_count = 0
    track_history = {}
    
    # ─── 流水线函数 ────────────────────────────────────────────
    def detect(frame) -> "boxes":
        return [b for b in model(frame) if b.conf > threshold]
    
    def track(frame, boxes) -> "tracks":
        nonlocal frame_count
        frame_count += 1
        # 简单追踪：为每个 box 分配 ID
        tracks = []
        for i, b in enumerate(boxes):
            track_id = f"T{i}"
            track_history[track_id] = track_history.get(track_id, 0) + 1
            tracks.append({"id": track_id, "bbox": b.bbox})
        return tracks
    
    def draw(frame, tracks) -> "frame":
        # Mock 绘制：只打印信息
        for t in tracks:
            pass  # cv2.rectangle(frame, t["bbox"], (0, 255, 0), 2)
        return frame
    
    # ─── 工具函数（不进流水线）─────────────────────────────────
    def get_frame_count() -> int:
        return frame_count
    
    def get_track_history() -> dict:
        return track_history
    
    return locals()


# ─── 显示函数 ──────────────────────────────────────────────────

def display(frame) -> "displayed":
    """显示结果"""
    return True


# ─── 主程序（与 README 一致）──────────────────────────────────

if __name__ == "__main__":
    # 创建两个独立的追踪器实例（各自有独立的模型和状态）
    tracker_front = create_tracker("yolo.pt", threshold=0.7)
    tracker_rear = create_tracker("yolo.pt", threshold=0.5)
    
    print(f"创建追踪器: front(threshold=0.7), rear(threshold=0.5)")

    p1 = Pipeline([
        DataSource(front_camera.read),
        tracker_front.detect,
        tracker_front.track,
        tracker_front.draw,
        display,
    ])

    p2 = Pipeline([
        DataSource(rear_camera.read),
        tracker_rear.detect,
        tracker_rear.track,
        tracker_rear.draw,
        display,
    ])

    p1.start()
    p2.start()
    
    print("流水线已启动，查看状态...")

    # 查看状态（与 README 一致）
    target_frames = 20
    while min(tracker_front.get_frame_count(), tracker_rear.get_frame_count()) < target_frames:
        print(f"  front: {tracker_front.get_frame_count():3d} frames, "
              f"rear: {tracker_rear.get_frame_count():3d} frames")
        time.sleep(0.5)
    
    p1.stop()
    p2.stop()
    
    print("\n最终统计（状态隔离验证）：")
    print(f"  front: {tracker_front.get_frame_count()} frames, history={tracker_front.get_track_history()}")
    print(f"  rear:  {tracker_rear.get_frame_count()} frames, history={tracker_rear.get_track_history()}")
    print("\n流水线已停止")
