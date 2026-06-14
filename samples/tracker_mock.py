"""
多目标追踪器 Mock 示例
与 README.md 中 @init_ctx 例子一致。
"""
import time

import numpy as np

from func2stream import DataSource, Pipeline, init_ctx


class MockCamera:
    def __init__(self, name, width=640, height=480, sleep=0.05, seed=None):
        self.name = name
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
            MockBox(w // 4, h // 4, w // 2, h // 2, 0.8),
            MockBox(w // 3, h // 3, w // 4, h // 4, 0.4),
        ]

    return model


@init_ctx
def create_tracker(model_path, threshold=0.5):
    model = load_model(model_path)
    frame_count = 0
    track_history = {}

    def detect(frame) -> "boxes":
        return [b for b in model(frame) if b.conf > threshold]

    def track(frame, boxes) -> "tracks":
        nonlocal frame_count
        frame_count += 1
        tracks = []
        for i, box in enumerate(boxes):
            track_id = f"T{i}"
            track_history[track_id] = track_history.get(track_id, 0) + 1
            tracks.append({"id": track_id, "bbox": box.bbox})
        return tracks

    def draw(frame, tracks) -> "frame":
        return frame

    def get_frame_count() -> int:
        return frame_count

    def get_track_history() -> dict:
        return dict(track_history)

    return locals()


def display(frame) -> "displayed":
    """显示结果"""
    return True


if __name__ == "__main__":
    front_camera = MockCamera("front", seed=1)
    rear_camera = MockCamera("rear", seed=2)
    tracker_front = create_tracker("yolo.pt", threshold=0.7)
    tracker_rear = create_tracker("yolo.pt", threshold=0.5)

    print("创建追踪器: front(threshold=0.7), rear(threshold=0.5)")

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

    target_frames = 20
    while min(tracker_front.get_frame_count(), tracker_rear.get_frame_count()) < target_frames:
        print(
            f"  front: {tracker_front.get_frame_count():3d} frames, "
            f"rear: {tracker_rear.get_frame_count():3d} frames"
        )
        time.sleep(0.5)

    p1.stop()
    p2.stop()

    print("\n最终统计（状态隔离验证）：")
    print(f"  front: {tracker_front.get_frame_count()} frames, history={tracker_front.get_track_history()}")
    print(f"  rear:  {tracker_rear.get_frame_count()} frames, history={tracker_rear.get_track_history()}")
    print("\n流水线已停止")
