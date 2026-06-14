"""
gpu_model() 示例：把 GPU 模型延迟到 Pipeline 工作线程里创建。

运行: python samples/gpu_model_trt.py
"""
import time

import numpy as np
import torch
import torchvision.models as models

from func2stream import DataSource, Pipeline, gpu_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def preprocess(frame) -> "tensor":
    import cv2

    img = cv2.resize(frame, (224, 224))
    img = img[:, :, ::-1].copy()
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return tensor.to(DEVICE)


class TRTModel:
    def __init__(self, device=DEVICE):
        self.device = device
        self.model = models.resnet18(weights=None).to(device).eval()
        try:
            import torch_tensorrt  # noqa: F401

            self.model = torch.compile(self.model, backend="tensorrt")
            self.backend = "tensorrt"
        except ImportError:
            self.backend = "torch"
            pass

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            self.model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
        print(f"[gpu_model] backend={self.backend}, device={device}")

    def __call__(self, tensor):
        with torch.no_grad():
            return self.model(tensor)


model = gpu_model(lambda: TRTModel(device=DEVICE))


def classify(tensor) -> "top3":
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    values, indices = torch.topk(probs, 3, dim=1)
    return [(i.item(), p.item()) for i, p in zip(indices[0], values[0])]


def display(top3) -> "displayed":
    print(top3)
    return True


class MockCamera:
    def read(self):
        time.sleep(0.01)
        return load_image()


if __name__ == "__main__":
    if DEVICE != "cuda":
        print("未检测到 CUDA，本示例会在 CPU 上运行，无法体现 TensorRT 线程亲和性收益。")
    print("模型会在 Pipeline 工作线程首次访问时创建。")

    pipeline = Pipeline([
        DataSource(MockCamera().read),
        preprocess,
        classify,
        display,
    ])

    pipeline.start()
    time.sleep(3)
    pipeline.stop()
