import pytest

from func2stream import init_ctx
from func2stream.implicit_ctx import _should_wrap, auto_ctx


def test_auto_ctx_uses_parameter_names_and_string_return_keys():
    def preprocess(frame) -> "tensor":
        return frame + 1

    wrapped = auto_ctx(preprocess)
    ctx = {"frame": 2}

    assert wrapped(ctx) is ctx
    assert ctx["tensor"] == 3
    assert wrapped.get == ["frame"]
    assert wrapped.ret == ["tensor"]


def test_auto_ctx_supports_multiple_outputs():
    def split(tensor) -> ("double", "triple"):
        return tensor * 2, tensor * 3

    ctx = {"tensor": 4}

    assert auto_ctx(split)(ctx) is ctx
    assert ctx["double"] == 8
    assert ctx["triple"] == 12


def test_auto_ctx_keeps_type_annotations_out_of_pipeline_wrapping():
    def helper(frame) -> int:
        return frame

    assert _should_wrap(helper) is False
    assert auto_ctx(helper) is helper


def test_auto_ctx_wraps_none_return_annotation_as_side_effect_stage():
    seen = []

    def record(frame, tensor) -> None:
        seen.append((frame, tensor))

    ctx = {"frame": 3, "tensor": 4}
    wrapped = auto_ctx(record)

    assert wrapped(ctx) is ctx
    assert seen == [(3, 4)]
    assert wrapped.get == ["frame", "tensor"]
    assert wrapped.ret == []


def test_auto_ctx_wraps_future_annotations_none_as_side_effect_stage():
    namespace = {}
    exec(
        "from __future__ import annotations\n"
        "def record(frame, tensor) -> None:\n"
        "    seen.append((frame, tensor))\n",
        {"seen": [], "__builtins__": __builtins__},
        namespace,
    )
    record = namespace["record"]
    seen = record.__globals__["seen"]
    ctx = {"frame": 5, "tensor": 6}
    wrapped = auto_ctx(record)

    assert wrapped(ctx) is ctx
    assert seen == [(5, 6)]
    assert wrapped.ret == []


def test_auto_ctx_under_future_annotations_keeps_string_keys_and_ignores_type_annotations():
    namespace = {}
    exec(
        "from __future__ import annotations\n"
        "def preprocess(frame) -> 'tensor':\n"
        "    return frame + 1\n"
        "def split(tensor) -> ('double', 'triple'):\n"
        "    return tensor * 2, tensor * 3\n"
        "def helper(frame) -> int:\n"
        "    return frame\n",
        {"__builtins__": __builtins__},
        namespace,
    )

    ctx = {"frame": 7}
    preprocess = auto_ctx(namespace["preprocess"])
    split = auto_ctx(namespace["split"])
    helper = namespace["helper"]

    assert preprocess(ctx) is ctx
    assert split(ctx) is ctx
    assert ctx["tensor"] == 8
    assert ctx["double"] == 16
    assert ctx["triple"] == 24
    assert _should_wrap(helper) is False
    assert auto_ctx(helper) is helper


def test_auto_ctx_reports_missing_context_keys():
    def detect(tensor) -> "boxes":
        return [tensor]

    with pytest.raises(KeyError, match="tensor"):
        auto_ctx(detect)({"frame": 1})


def test_init_ctx_wraps_pipeline_functions_and_keeps_state_isolated():
    @init_ctx
    def create_counter():
        total = 0

        def add(frame) -> "total":
            nonlocal total
            total += frame
            return total

        def get_total() -> int:
            return total

        return locals()

    left = create_counter()
    right = create_counter()

    left.add({"frame": 2})
    left.add({"frame": 3})
    right.add({"frame": 10})

    assert left.get_total() == 5
    assert right.get_total() == 10
