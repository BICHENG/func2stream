from .core import Pipeline, DataSource, Element, init_ctx, gpu_model
from .implicit_ctx import auto_ctx

try:
    from .video import VideoSource
except ImportError as _video_error:
    if getattr(_video_error, "name", None) != "cv2" and "cv2" not in str(_video_error).lower():
        raise

    def VideoSource(*args, **kwargs):
        raise ModuleNotFoundError("Install func2stream[video] to use VideoSource.")

__all__ = [
    'Pipeline',
    'DataSource',
    'VideoSource',
    'Element',
    'init_ctx',
    'auto_ctx',
    'gpu_model',
]
