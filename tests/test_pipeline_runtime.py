import threading
import time

from func2stream import DataSource, Pipeline


class SequenceReader:
    def __init__(self, values):
        self.values = list(values)

    def read(self):
        if self.values:
            return self.values.pop(0)
        time.sleep(0.005)
        return None


def test_pipeline_processes_annotated_multistage_context():
    reader = SequenceReader([1, 2, 3])
    results = []
    complete = threading.Event()

    def preprocess(frame) -> "tensor":
        if frame is None:
            return None
        return frame + 1

    def fanout(tensor) -> ("double", "triple"):
        if tensor is None:
            return None, None
        return tensor * 2, tensor * 3

    def collect(frame, double, triple) -> "displayed":
        if frame is not None:
            results.append((frame, double, triple))
            if len(results) == 3:
                complete.set()
        return True

    pipeline = Pipeline(
        [
            DataSource(reader.read),
            preprocess,
            fanout,
            collect,
        ],
        friendly_name="test-pipeline",
    )

    try:
        pipeline.start()
        assert complete.wait(2), f"pipeline produced only {results}"
    finally:
        pipeline.stop()

    assert results == [
        (1, 4, 6),
        (2, 6, 9),
        (3, 8, 12),
    ]


def test_pipeline_supports_none_annotated_side_effect_sink():
    reader = SequenceReader([4])
    complete = threading.Event()
    observed = []

    def preprocess(frame) -> "tensor":
        if frame is None:
            return None
        return frame + 1

    def side_effect_sink(frame, tensor) -> None:
        if frame is not None:
            observed.append((frame, tensor))
            complete.set()

    pipeline = Pipeline(
        [
            DataSource(reader.read),
            preprocess,
            side_effect_sink,
        ],
        friendly_name="side-effect-pipeline",
    )

    try:
        pipeline.start()
        assert complete.wait(2), f"side effect sink did not observe data: {observed}"
    finally:
        pipeline.stop()

    assert observed == [(4, 5)]
