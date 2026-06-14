import threading

from func2stream import DataSource, Pipeline
from samples.beauty_filter_mock import MockCamera as BeautyCamera
from samples.beauty_filter_mock import beautify_model, detect, get_face_crop, some_preprocess
from samples.tracker_mock import MockCamera as TrackerCamera
from samples.tracker_mock import create_tracker


def test_beauty_filter_sample_processes_realistic_frames():
    camera = BeautyCamera(width=64, height=48, sleep=0, seed=1)
    complete = threading.Event()
    frames = []

    def collect(frame, landmarks, beautified) -> "displayed":
        frames.append((frame, landmarks, beautified))
        if len(frames) >= 3:
            complete.set()
        return True

    pipeline = Pipeline([
        DataSource(camera.read),
        some_preprocess,
        detect,
        get_face_crop,
        beautify_model,
        collect,
    ])

    try:
        pipeline.start()
        assert complete.wait(3), f"beauty sample produced only {len(frames)} frames"
    finally:
        pipeline.stop()

    frame, landmarks, beautified = frames[0]
    assert frame.shape == (48, 64, 3)
    assert len(landmarks) == 1
    assert beautified[0].shape == (24, 32, 3)


def test_tracker_sample_runs_two_isolated_stateful_pipelines():
    front_camera = TrackerCamera("front", width=64, height=48, sleep=0, seed=1)
    rear_camera = TrackerCamera("rear", width=64, height=48, sleep=0, seed=2)
    tracker_front = create_tracker("yolo.pt", threshold=0.7)
    tracker_rear = create_tracker("yolo.pt", threshold=0.3)
    complete = threading.Event()
    displayed = []

    def collect(label):
        def display(frame) -> "displayed":
            displayed.append({"camera": label, "frame_shape": tuple(frame.shape)})
            if len(displayed) >= 6:
                complete.set()
            return True

        return display

    p1 = Pipeline([
        DataSource(front_camera.read),
        tracker_front.detect,
        tracker_front.track,
        tracker_front.draw,
        collect("front"),
    ])
    p2 = Pipeline([
        DataSource(rear_camera.read),
        tracker_rear.detect,
        tracker_rear.track,
        tracker_rear.draw,
        collect("rear"),
    ])

    try:
        p1.start()
        p2.start()
        assert complete.wait(3), f"tracker sample displayed only {len(displayed)} frames"
    finally:
        p1.stop()
        p2.stop()

    assert tracker_front.get_frame_count() > 0
    assert tracker_rear.get_frame_count() > 0
    assert tracker_front.get_track_history()["T0"] > 0
    assert tracker_rear.get_track_history()["T0"] > 0
    assert tracker_rear.get_track_history()["T1"] > 0
    assert {item["camera"] for item in displayed} == {"front", "rear"}
