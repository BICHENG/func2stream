# func2stream v1.0

[中文](README.md) | [English](README_en.md)

> [!TIP]
> Chain functions into async pipelines. Same effort as writing a comment. Throughput: yes.

---

**Your code:**
```python
def detect(tensor):
    return model(tensor)
```

**Just add a return annotation:**
```python
def detect(tensor) -> "boxes":
    return model(tensor)
```

**Drop it into a Pipeline, async works:**
```python
Pipeline([... preprocess, detect, display]).start()
```

String return annotations become pipeline keys.

## Why func2stream?

Purpose: Minimal intrusion on serial code, let stage N processing overlap with stage N+1 processing in time.

Real-time inference. Video streams. A chain of ordinary functions can overlap stage by stage.

Parallelizing too early usually spreads threads, queues, and shared state through business code.

```python
# e.g., a real-time beauty filter
while True:
    frame = camera.read()
    tensor = some_preprocess(frame)
    face_boxes = detect(tensor)
    landmarks, crops = get_face_crop(tensor, face_boxes)
    beautified = beautify_model(crops)
    display(frame, landmarks, beautified)
```

**What you want**: Run them in parallel, keep the code clean, stay focused on business logic.

**The old way**: Threads, queues, code becomes a mess, can't focus on the actual work. Globals flying everywhere — touch one thing, break three others. 🔥 Ask me how I know.

**Speed goes up. So does your hair loss.** 👨‍🦲

**Now**:

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

Each function runs in its own thread. Data flows automatically. No queues, threads, or infrastructure polluting your global scope.

> Full runnable example: [samples/beauty_filter_mock.py](samples/beauty_filter_mock.py)

---

## Installation

```bash
pip install git+https://github.com/bicheng/func2stream.git
```

---

## How to Use

Add `-> "data_name"` to functions that go into the pipeline. That's it.

```python
from func2stream import Pipeline, DataSource

# ─── Helper functions (not in pipeline, no annotation) ──────────────

def normalize(img):
    return img.astype(np.float32) / 255.0

def get_eye_positions(x, y, w, h):
    return [(x + w//3, y + h//3), (x + 2*w//3, y + h//3)]

def apply_brightness(crop, factor=0.9):
    return crop * factor + (1 - factor)


# ─── Pipeline functions (just add -> "key") ─────────────────────────

def some_preprocess(frame) -> "tensor":
    return normalize(frame)

def detect(tensor) -> "face_boxes":
    return model.detect(tensor)

def get_face_crop(tensor, face_boxes) -> ("landmarks", "crops"):   # multiple outputs
    landmarks, crops = [], []
    for (x, y, w, h) in face_boxes:
        landmarks.append(get_eye_positions(x, y, w, h))
        crops.append(tensor[y:y+h, x:x+w])
    return landmarks, crops

def beautify_model(crops) -> "beautified":
    return [apply_brightness(crop) for crop in crops]

def display(frame, landmarks, beautified) -> "displayed":          # multiple inputs
    cv2.imshow("win", frame)
    return True


# ─── Assembly ───────────────────────────────────────────────────────

Pipeline([
    DataSource(camera.read),    # → frame
    some_preprocess,            # → tensor
    detect,                     # → face_boxes
    get_face_crop,              # → landmarks, crops
    beautify_model,             # → beautified
    display,                    # → displayed
]).start()
```

**Data flow**: `frame → tensor → face_boxes → (landmarks, crops) → beautified → displayed`

---

## `@init_ctx`: State Without Globals

Sound familiar?
- Globals everywhere
- Pipeline needs to hold state (models, counters, configs)
- Writing a full class feels like overkill

Bundle them with `@init_ctx`.

> Full runnable example: [samples/tracker_mock.py](samples/tracker_mock.py)

### Example: Multi-Object Tracker

```python
from func2stream import Pipeline, DataSource, init_ctx

@init_ctx
def create_tracker(model_path, threshold=0.5):
    # ─── State ──────────────────────────────────────────────────────
    model = load_model(model_path)
    frame_count = 0
    track_history = {}
    
    # ─── Pipeline functions ─────────────────────────────────────────
    def detect(frame) -> "boxes":
        return [b for b in model(frame) if b.conf > threshold]
    
    def track(frame, boxes) -> "tracks":
        nonlocal frame_count
        frame_count += 1
        # ... tracking logic ...
        return tracks
    
    def draw(frame, tracks) -> "frame":
        for t in tracks:
            cv2.rectangle(frame, t.bbox, (0, 255, 0), 2)
        return frame
    
    # ─── Helpers ────────────────────────────────────────────────────
    def get_frame_count() -> int:
        return frame_count
    
    def get_track_history() -> dict:
        return track_history
    
    return locals()


# Same factory, two isolated tracker instances
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

# Check state
while min(tracker_front.get_frame_count(), tracker_rear.get_frame_count()) < 100:
    print(f"front: {tracker_front.get_frame_count()}")
    print(f"rear: {tracker_rear.get_frame_count()}")
    time.sleep(1)

```

---

## `gpu_model()`: GPU Resources

`gpu_model()` avoids performance issues when GPU models execute across threads.

```python
from func2stream import Pipeline, DataSource, init_ctx, gpu_model

@init_ctx
def create_detector(threshold=0.5):
    # Main thread variables
    frame_count = 0
    
    # GPU model - deferred to worker thread
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

`gpu_model()` takes a zero-argument lambda. Execution is deferred to first access in the worker thread. Usage is identical to the original model.

> Example: [samples/gpu_model_trt.py](samples/gpu_model_trt.py)

---

## Gotchas ⚠️

### Gotcha 1: Pipeline functions can't call each other

Don't call one pipeline function from inside another — they get auto-wrapped and it will break. If it breaks, you were warned.

```python
def to_gray(frame) -> "gray":
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 💥 This breaks — wrapped functions expect ctx dict, not raw args
def step1(frame) -> "edges":
    gray = to_gray(frame)  # nope
    return cv2.Canny(gray, 50, 150)

# -------------------------------------------------------------
# ✅ Option 1: Merge if the overlap perf gain is negligible
def step1(frame) -> "edges":
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 50, 150)

Pipeline([
    DataSource(...),
    step1,
])

# -------------------------------------------------------------
# ✅ Option 2: Keep them separate, let Pipeline chain them
def canny(gray) -> "edges":
    return cv2.Canny(gray, 50, 150)

Pipeline([
    DataSource(...),
    to_gray,   # outputs gray
    canny,     # reads gray automatically
    ...
])

```

### Gotcha 2: Forgot `-> "data_name"`

Pipeline stages declare their output with one string return annotation.

```python
def process(frame):
    return frame * 2

Pipeline([
    DataSource(...),
    process,  # 💥 Pipeline warns; function receives ctx dict, not frame
])

# ✅ Always declare output
def process(frame) -> "result":
    return frame * 2
```

### Gotcha 3: Type annotations don't count

```python
# 👀 -> int is a type hint — ignored
def process(frame) -> int:
    return frame * 2

# ✅ Quoted strings only
def process(frame) -> "result":
    return frame * 2
```


---

## v1.0 Breaking Changes

Earlier versions were... let's say "rough around the edges". Now v1.0 — **usage is as simple as writing a comment**:

| Removed | Replacement |
|---------|-------------|
| `@from_ctx(get=[], ret=[])` | Use `-> "key"` return annotation |
| `build_ctx()` | `DataSource` auto-builds ctx |
| `MapReduce` | Experimental, deprecated (see Roadmap) |
| `nodrop()` | No replacement, no use case |

**What's new in v1.0**:
- **Zero decorators**: Just `-> "key"`. That's the whole API.
- **Implicit context**: Param names = ctx keys. Zero boilerplate.
- **Clean state**: `@init_ctx` replaces scattered globals.

---

## Roadmap

### Parallel Branches & Sync (Planned)

```python
Pipeline([
    preprocess,
    (                        # ← Tuple = parallel branches
        [branch_a1, branch_a2],   # Branch 1: serial sub-pipeline
        [branch_b1],              # Branch 2
        branch_c,                 # Branch 3: single function
    ),                       # ← Tuple end = implicit sync point
    merge_results,           # Receives (a2_out, b1_out, c_out) packed
    postprocess,
]).start()
```

**Design**: Sync is implicit. Tuple closes = all branches join. No explicit barrier.

---

## License

MPL-2.0
