# func2stream v1.0

[中文](README.md) | [English](README_en.md)

> [!TIP]
> 把函数串成异步流水线，工作量和写注释一样，但性能天差地别

---

**你的原生代码：**
```python
def detect(tensor):
    return model(tensor)
```

**只需加个返回注解：**
```python
def detect(tensor) -> "boxes":
    return model(tensor)
```

**然后放进 Pipeline，自动异步：**
```python
Pipeline([... preprocess, detect, display]).start()
```

这不是魔法，只是我认为没人会注解字符串，所以在这个前提下，完成了很多工作。🤷‍♂️

## 为什么设计了 func2stream

目的：最小侵入串行代码，让阶段N的处理与阶段N+1的处理在时间上重叠。

对于实时推理，或希望处理序列数据（如视频）的场景，你有一堆处理函数，顺序调用时，吞吐量很低。

但在开发时，如果过早考虑并行，会到处管理变量和状态，增加代码复杂度。

```python
# 例如一个实时美颜程序
while True:
    frame = camera.read()
    tensor = some_preprocess(frame)
    face_boxes = detect(tensor)
    landmarks,crops = get_face_crop(tensor, face_boxes)
    beautified = beautify_model(crops)
    display(frame, landmarks, beautified)
```

**需求**：让它们并行运行、不增加代码复杂度、专注于业务逻辑。

**传统做法**：写线程、队列，代码变乱，无法专注业务逻辑，再加上全局变量到处飞，改一处动全身。

**速度可能起来了，但掉头发的速度也起来了。(bushi**

**现在**：

```python
Pipeline([
    DataSource(camera.read),
    some_preprocess,
    detect,
    get_face_crop,
    beautify_model,
    display,
]).start()
```

每个函数自动在独立线程运行，数据自动传递，无需在全局暴露 queue、thread 等基础设施。

> 完整可运行示例：[samples/beauty_filter_mock.py](samples/beauty_filter_mock.py)

---

## 安装

```bash
pip install git+https://github.com/bicheng/func2stream.git
```

---

## 如何使用？

在现有代码的基础上，只需要针对可参与异步流水线的函数做修改，加 `-> "数据名"` 即可。

```python
from func2stream import Pipeline
from func2stream.core import DataSource

# ─── 工具函数（不进流水线，不加 -> "数据名"）─────────────────

def normalize(img):
    return img.astype(np.float32) / 255.0

def get_eye_positions(x, y, w, h):
    return [(x + w//3, y + h//3), (x + 2*w//3, y + h//3)]

def apply_brightness(crop, factor=0.9):
    return crop * factor + (1 - factor)


# ─── 流水线函数（加 -> "数据名"）──────────────────────────────

def some_preprocess(frame) -> "tensor":
    return normalize(frame)                                         # 调用工具函数

def detect(tensor) -> "face_boxes":
    return model.detect(tensor)

def get_face_crop(tensor, face_boxes) -> ("landmarks", "crops"):   # 多输出
    landmarks, crops = [], []
    for (x, y, w, h) in face_boxes:
        landmarks.append(get_eye_positions(x, y, w, h))            # 调用工具函数
        crops.append(tensor[y:y+h, x:x+w])
    return landmarks, crops

def beautify_model(crops) -> "beautified":
    return [apply_brightness(crop) for crop in crops]              # 调用工具函数

def display(frame, landmarks, beautified) -> "displayed":          # 多输入
    cv2.imshow("win", frame)
    return True


# ─── 组装 ──────────────────────────────────────────────────────

Pipeline([
    DataSource(camera.read),    # → frame
    some_preprocess,            # → tensor
    detect,                     # → face_boxes
    get_face_crop,              # → landmarks, crops
    beautify_model,             # → beautified
    display,                    # → displayed
]).start()
```

**数据流**：`frame → tensor → face_boxes → (landmarks, crops) → beautified → displayed`

---

## 用 `@init_ctx` 持有状态，避免全局变量

你可能会遇到这样的开发问题:
- 全局变量满天飞
- 流水线内需要持有状态（模型、计数器、配置等）
- 写一个类属于过度设计

用 `@init_ctx` 把它们打包在一起。

> 完整可运行示例：[samples/tracker_mock.py](samples/tracker_mock.py)

### 例子：多目标追踪器

```python
from func2stream import Pipeline, init_ctx
from func2stream.core import DataSource

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
        # ... 追踪逻辑，更新 track_history ...
        return tracks
    
    def draw(frame, tracks) -> "frame":
        for t in tracks:
            cv2.rectangle(frame, t.bbox, (0, 255, 0), 2)
        return frame
    
    # ─── 工具函数（不进流水线）─────────────────────────────────
    def get_frame_count() -> int:
        return frame_count
    
    def get_track_history() -> dict:
        return track_history
    
    return locals()


# 创建两个独立的追踪器实例（各自有独立的模型和状态）
tracker_front = create_tracker("yolo.pt", threshold=0.7)
tracker_rear = create_tracker("yolo.pt", threshold=0.5)

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

# 查看状态
while (min(tracker_front.get_frame_count(), tracker_rear.get_frame_count()) < 100):
    print(f"front: {tracker_front.get_frame_count()}")
    print(f"rear: {tracker_rear.get_frame_count()}")
    time.sleep(1)

```

---

## 用 `gpu_model()` 处理 GPU 资源

`gpu_model()` 可以避免 GPU 模型跨线程执行时的性能问题。

```python
from func2stream import Pipeline, init_ctx, gpu_model
from func2stream.core import DataSource

@init_ctx
def create_detector(threshold=0.5):
    # 主线程变量
    frame_count = 0
    
    # GPU 模型 - 延迟到工作线程创建
    model = gpu_model(lambda: TRTModel(device='cuda'))
    
    def detect(frame) -> "boxes":
        nonlocal frame_count
        frame_count += 1
        return [b for b in model(frame) if b.conf > threshold]
    
    def get_count():
        return frame_count
    
    return locals()

ctx = create_detector(threshold=0.7)
Pipeline([DataSource(camera.read), ctx.detect, display]).start()
```

`gpu_model()` 接受一个无参 lambda，在工作线程首次访问时执行。调用方式与原模型一致。

> 示例：[samples/gpu_model_trt.py](samples/gpu_model_trt.py)

---

## 常见误区

### 误区 1：流水线函数不再能嵌套调用
解释：如果你有多个流水线函数，不要在某一个流水线函数里直接调用另一个流水线函数。

PS: 后期会想办法支持嵌套


```python
def to_gray(frame) -> "gray":
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 💥 流水线函数会被自动包装，行为不再是直接调用
def step1(frame) -> "edges":
    gray = to_gray(frame)  # 不能在这里调用 to_gray
    return cv2.Canny(gray, 50, 150)

# -------------------------------------------------------------
# ✅ 方案一：如果overlap的性能收益比较小，合并处理步骤到一个函数
def step1(frame) -> "edges":
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 50, 150)

Pipeline([
    DataSource(...),
    step1,
])

# -------------------------------------------------------------
# ✅ 方案二：拆成独立步骤，让 Pipeline 自动串联
def canny(gray) -> "edges":
    return cv2.Canny(gray, 50, 150)

Pipeline([
    DataSource(...),
    to_gray,   # 先执行，产出 gray
    canny,     # 后执行，自动读取 gray
    ...
])

```

### 误区 2：流水线中使用的函数忘了写 `-> "数据名"`

PS: 这是唯一工作量，好好写吧

```python
def process(frame):
    return frame * 2

Pipeline([
    DataSource(...),
    process,  # 💥 Pipeline 会警告，且函数运行时收到的并非 frame
])

# ✅ 放进 Pipeline 的函数，必须声明输出
def process(frame) -> "result":
    return frame * 2
```

### 误区 3：类型注解

```python
# 👀 -> int 是类型注解，Pipeline 不会识别它
def process(frame) -> int:
    return frame * 2

# ✅ 用引号包裹的字符串才会被识别
def process(frame) -> "result":
    return frame * 2
```


---

## v1.0 Breaking Changes

相比之下，早期版本更像是 geek 的玩具，现已升级到 v1.0，以下 API 已移除，**使用就像注释一样简单**：

| 移除项 | 替代方案 |
|--------|----------|
| `@from_ctx(get=[], ret=[])` | 使用 `-> "key"` 返回注解 |
| `build_ctx()` | `DataSource` 自动构建 ctx |
| `MapReduce` | 实验性功能，已弃用（见 Roadmap） |
| `nodrop()` | 无替代，该功能无使用场景 |

**v1.0 的核心改进**：
- **零装饰器数据流**：只需 `-> "key"` 注解，Pipeline 自动识别并包装
- **隐式 Context**：参数名即读取键，无需手动声明
- **状态隔离**：`@init_ctx` 封装模型、计数器等，避免全局变量

---

## Roadmap

### 并行分支与同步（规划中）

```python
Pipeline([
    preprocess,
    (                        # ← 元组 = 并行分支
        [branch_a1, branch_a2],   # 分支1：串行子流水线
        [branch_b1],              # 分支2
        branch_c,                 # 分支3：单函数
    ),                       # ← 元组结束 = 隐式同步点
    merge_results,           # 收到 (a2_out, b1_out, c_out) 打包
    postprocess,
]).start()
```

**设计原则**：同步点语法隐式，元组结束即同步，无需显式 Barrier。

---

## 许可证

MPL-2.0
