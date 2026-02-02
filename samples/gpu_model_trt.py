"""
gpu_model() 用法示例 - 使用 torch_tensorrt 编译的真实 TRT 模型

背景: TRT 模型在跨线程执行时可能触发 CUDA context 同步
运行: python samples/gpu_model_trt.py

本示例使用 torch.compile(backend="tensorrt") 编译真实模型，
测试单模型和多模型场景下的线程亲和性问题。
"""
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

try:
    import torch_tensorrt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False
    print("[警告] torch_tensorrt 未安装，将使用 torch.compile(backend='inductor')")

from func2stream import Pipeline, init_ctx, gpu_model
from func2stream.core import DataSource


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_FRAMES = 50  # 增加帧数以获得足够的推理样本
BACKEND = "tensorrt" if HAS_TRT else "inductor"


# ─── 预处理函数 ───────────────────────────────────────────────────

def load_image():
    """模拟摄像头读取"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def preprocess(img):
    """BGR -> normalized tensor"""
    import cv2
    img = cv2.resize(img, (224, 224))
    img = img[:, :, ::-1].copy()  # BGR -> RGB
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return tensor.to(DEVICE)


def postprocess(logits):
    """logits -> top3 结果"""
    probs = torch.softmax(logits, dim=1)
    top3 = torch.topk(probs, 3, dim=1)
    return [(i.item(), p.item()) for i, p in zip(top3.indices[0], top3.values[0])]


# ─── TRT 模型封装 ─────────────────────────────────────────────────

class TRTModel:
    """
    使用 torch.compile(backend="tensorrt") 编译的模型
    """
    
    def __init__(self, model_name="resnet18", device=DEVICE):
        self.device = device
        self.created_thread = threading.current_thread().ident
        print(f"[TRT] 创建 {model_name}, 线程={self.created_thread}, backend={BACKEND}")
        
        # 加载基础模型
        if model_name == "resnet18":
            base_model = models.resnet18(weights=None)
        elif model_name == "resnet34":
            base_model = models.resnet34(weights=None)
        elif model_name == "vgg16":
            base_model = models.vgg16(weights=None)
        else:
            base_model = models.resnet18(weights=None)
        
        base_model = base_model.to(device).eval()
        
        # TRT 编译
        self.model = torch.compile(base_model, backend=BACKEND)
        
        # warmup（触发实际编译）
        print(f"[TRT] 编译中...")
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            for _ in range(5):
                _ = self.model(dummy)
        torch.cuda.synchronize()
        print(f"[TRT] {model_name} 就绪")
    
    def __call__(self, tensor):
        with torch.no_grad():
            return self.model(tensor)


# ─── Mock 数据源 ──────────────────────────────────────────────────

class MockCamera:
    def __init__(self, max_frames=NUM_FRAMES):
        self.frame_count = 0
        self.max_frames = max_frames
    
    def read(self):
        if self.frame_count >= self.max_frames:
            time.sleep(0.5)
            return None
        self.frame_count += 1
        time.sleep(0.01)
        return load_image()


# ─── 测试 1: 单模型对照组（主线程创建）────────────────────────────

def test_single_model_main_thread():
    """对照组：单模型，主线程创建"""
    print("\n" + "=" * 60)
    print("测试 1: 单模型，主线程创建")
    print("=" * 60)
    
    camera = MockCamera()
    model = TRTModel("resnet18", device=DEVICE)
    
    inference_times = []
    
    def do_preprocess(frame) -> "tensor":
        if frame is None:
            return None
        return preprocess(frame)
    
    def do_classify(tensor) -> "logits":
        if tensor is None:
            return None
        t0 = time.perf_counter()
        logits = model(tensor)
        torch.cuda.synchronize()
        inference_times.append(time.perf_counter() - t0)
        return logits
    
    def do_postprocess(logits) -> "result":
        if logits is None:
            return None
        return postprocess(logits)
    
    def sink(result) -> None:
        pass
    
    pipeline = Pipeline([
        DataSource(camera.read),
        do_preprocess,
        do_classify,
        do_postprocess,
        sink,
    ])
    pipeline.start()
    
    while camera.frame_count < camera.max_frames:
        time.sleep(0.1)
    time.sleep(2)  # 增加等待时间
    pipeline.stop()
    
    if len(inference_times) > 2:
        avg = sum(inference_times[2:]) / len(inference_times[2:])  # 跳过前2帧
        print(f"推理帧数: {len(inference_times)}")
        print(f"平均推理时间: {avg*1000:.2f}ms")
        return avg
    print(f"推理帧数不足: {len(inference_times)}")
    return None


# ─── 测试 2: 单模型实验组（gpu_model 延迟创建）───────────────────

def test_single_model_gpu_model():
    """实验组：单模型，gpu_model 延迟创建"""
    print("\n" + "=" * 60)
    print("测试 2: 单模型，gpu_model() 延迟创建")
    print("=" * 60)
    
    camera = MockCamera()
    model = gpu_model(lambda: TRTModel("resnet18", device=DEVICE))
    
    inference_times = []
    
    def do_preprocess(frame) -> "tensor":
        if frame is None:
            return None
        return preprocess(frame)
    
    def do_classify(tensor) -> "logits":
        if tensor is None:
            return None
        t0 = time.perf_counter()
        logits = model(tensor)
        torch.cuda.synchronize()
        inference_times.append(time.perf_counter() - t0)
        return logits
    
    def do_postprocess(logits) -> "result":
        if logits is None:
            return None
        return postprocess(logits)
    
    def sink(result) -> None:
        pass
    
    pipeline = Pipeline([
        DataSource(camera.read),
        do_preprocess,
        do_classify,
        do_postprocess,
        sink,
    ])
    pipeline.start()
    
    while camera.frame_count < camera.max_frames:
        time.sleep(0.1)
    time.sleep(2)
    pipeline.stop()
    
    if len(inference_times) > 2:
        avg = sum(inference_times[2:]) / len(inference_times[2:])
        print(f"推理帧数: {len(inference_times)}")
        print(f"平均推理时间: {avg*1000:.2f}ms")
        return avg
    print(f"推理帧数不足: {len(inference_times)}")
    return None


# ─── 测试 3: 多模型对照组（主线程创建）────────────────────────────

def test_multi_model_main_thread():
    """对照组：多模型，主线程创建"""
    print("\n" + "=" * 60)
    print("测试 3: 多模型，主线程创建")
    print("=" * 60)
    
    camera = MockCamera()
    
    # 主线程创建多个模型
    detector = TRTModel("resnet18", device=DEVICE)
    classifier = TRTModel("resnet34", device=DEVICE)
    
    inference_times = []
    
    def do_preprocess(frame) -> "tensor":
        if frame is None:
            return None
        return preprocess(frame)
    
    def do_detect(tensor) -> "features":
        if tensor is None:
            return None
        t0 = time.perf_counter()
        features = detector(tensor)
        torch.cuda.synchronize()
        inference_times.append(('detect', time.perf_counter() - t0))
        return features
    
    def do_classify(features) -> "logits":
        if features is None:
            return None
        # 使用 features 的形状创建新输入（模拟级联）
        tensor = torch.randn(1, 3, 224, 224, device=DEVICE)
        t0 = time.perf_counter()
        logits = classifier(tensor)
        torch.cuda.synchronize()
        inference_times.append(('classify', time.perf_counter() - t0))
        return logits
    
    def do_postprocess(logits) -> "result":
        if logits is None:
            return None
        return postprocess(logits)
    
    def sink(result) -> None:
        pass
    
    pipeline = Pipeline([
        DataSource(camera.read),
        do_preprocess,
        do_detect,
        do_classify,
        do_postprocess,
        sink,
    ])
    pipeline.start()
    
    while camera.frame_count < camera.max_frames:
        time.sleep(0.1)
    time.sleep(3)
    pipeline.stop()
    
    detect_times = [t for n, t in inference_times if n == 'detect']
    classify_times = [t for n, t in inference_times if n == 'classify']
    if len(detect_times) > 2 and len(classify_times) > 2:
        avg_detect = sum(detect_times[2:]) / len(detect_times[2:])
        avg_classify = sum(classify_times[2:]) / len(classify_times[2:])
        total = avg_detect + avg_classify
        print(f"检测平均: {avg_detect*1000:.2f}ms")
        print(f"分类平均: {avg_classify*1000:.2f}ms")
        print(f"总计: {total*1000:.2f}ms")
        return total
    print(f"推理帧数不足: detect={len(detect_times)}, classify={len(classify_times)}")
    return None


# ─── 测试 4: 多模型实验组（gpu_model 延迟创建）───────────────────

def test_multi_model_gpu_model():
    """实验组：多模型，gpu_model 延迟创建"""
    print("\n" + "=" * 60)
    print("测试 4: 多模型，gpu_model() 延迟创建")
    print("=" * 60)
    
    camera = MockCamera()
    
    # gpu_model 延迟创建
    detector = gpu_model(lambda: TRTModel("resnet18", device=DEVICE))
    classifier = gpu_model(lambda: TRTModel("resnet34", device=DEVICE))
    
    inference_times = []
    
    def do_preprocess(frame) -> "tensor":
        if frame is None:
            return None
        return preprocess(frame)
    
    def do_detect(tensor) -> "features":
        if tensor is None:
            return None
        t0 = time.perf_counter()
        features = detector(tensor)
        torch.cuda.synchronize()
        inference_times.append(('detect', time.perf_counter() - t0))
        return features
    
    def do_classify(features) -> "logits":
        if features is None:
            return None
        tensor = torch.randn(1, 3, 224, 224, device=DEVICE)
        t0 = time.perf_counter()
        logits = classifier(tensor)
        torch.cuda.synchronize()
        inference_times.append(('classify', time.perf_counter() - t0))
        return logits
    
    def do_postprocess(logits) -> "result":
        if logits is None:
            return None
        return postprocess(logits)
    
    def sink(result) -> None:
        pass
    
    pipeline = Pipeline([
        DataSource(camera.read),
        do_preprocess,
        do_detect,
        do_classify,
        do_postprocess,
        sink,
    ])
    pipeline.start()
    
    while camera.frame_count < camera.max_frames:
        time.sleep(0.1)
    time.sleep(3)
    pipeline.stop()
    
    detect_times = [t for n, t in inference_times if n == 'detect']
    classify_times = [t for n, t in inference_times if n == 'classify']
    if len(detect_times) > 2 and len(classify_times) > 2:
        avg_detect = sum(detect_times[2:]) / len(detect_times[2:])
        avg_classify = sum(classify_times[2:]) / len(classify_times[2:])
        total = avg_detect + avg_classify
        print(f"检测平均: {avg_detect*1000:.2f}ms")
        print(f"分类平均: {avg_classify*1000:.2f}ms")
        print(f"总计: {total*1000:.2f}ms")
        return total
    print(f"推理帧数不足: detect={len(detect_times)}, classify={len(classify_times)}")
    return None


# ─── 测试 5: init_ctx + gpu_model 组合 ────────────────────────────

def test_init_ctx_gpu_model():
    """组合: @init_ctx + gpu_model()"""
    print("\n" + "=" * 60)
    print("测试 5: @init_ctx + gpu_model() 组合")
    print("=" * 60)
    
    @init_ctx
    def create_pipeline_ctx(threshold=0.5):
        frame_count = 0
        
        # 多个 GPU 模型 - 延迟创建
        detector = gpu_model(lambda: TRTModel("resnet18", device=DEVICE))
        classifier = gpu_model(lambda: TRTModel("resnet34", device=DEVICE))
        
        def do_preprocess(frame) -> "tensor":
            if frame is None:
                return None
            return preprocess(frame)
        
        def do_detect(tensor) -> "features":
            nonlocal frame_count
            if tensor is None:
                return None
            frame_count += 1
            return detector(tensor)
        
        def do_classify(features) -> "logits":
            if features is None:
                return None
            tensor = torch.randn(1, 3, 224, 224, device=DEVICE)
            return classifier(tensor)
        
        def do_postprocess(logits) -> "result":
            if logits is None:
                return None
            return postprocess(logits)
        
        def get_count():
            return frame_count
        
        return locals()
    
    camera = MockCamera(max_frames=15)
    ctx = create_pipeline_ctx(threshold=0.7)
    
    def sink(result) -> None:
        pass
    
    pipeline = Pipeline([
        DataSource(camera.read),
        ctx.do_preprocess,
        ctx.do_detect,
        ctx.do_classify,
        ctx.do_postprocess,
        sink,
    ])
    pipeline.start()
    
    for _ in range(5):
        time.sleep(0.5)
        print(f"  状态查询: frames={ctx.get_count()}")
    
    pipeline.stop()
    print(f"  最终: frames={ctx.get_count()}")
    return ctx.get_count()


# ─── 主程序 ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"gpu_model() 验证 - torch_tensorrt")
    print(f"Device: {DEVICE}, Backend: {BACKEND}")
    print("=" * 60)
    
    if DEVICE == 'cpu':
        print("\n[注意] 未检测到 CUDA，跳过测试\n")
        return
    
    # 单模型测试
    avg_single_main = test_single_model_main_thread()
    avg_single_gpu = test_single_model_gpu_model()
    
    # 多模型测试
    avg_multi_main = test_multi_model_main_thread()
    avg_multi_gpu = test_multi_model_gpu_model()
    
    # 组合测试
    test_init_ctx_gpu_model()
    
    # 结果
    print("\n" + "=" * 60)
    print("结果")
    print("=" * 60)
    
    if avg_single_main and avg_single_gpu:
        ratio_single = avg_single_main / avg_single_gpu if avg_single_gpu > 0 else 0
        print(f"单模型:")
        print(f"  主线程创建: {avg_single_main*1000:.2f}ms")
        print(f"  gpu_model(): {avg_single_gpu*1000:.2f}ms")
        print(f"  比值: {ratio_single:.2f}x")
    
    if avg_multi_main and avg_multi_gpu:
        ratio_multi = avg_multi_main / avg_multi_gpu if avg_multi_gpu > 0 else 0
        print(f"多模型:")
        print(f"  主线程创建: {avg_multi_main*1000:.2f}ms")
        print(f"  gpu_model(): {avg_multi_gpu*1000:.2f}ms")
        print(f"  比值: {ratio_multi:.2f}x")


if __name__ == "__main__":
    main()
