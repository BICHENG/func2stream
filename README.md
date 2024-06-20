# func2stream

`func2stream` is a Python library that simplifies the construction of data processing pipelines, particularly for computationally intensive tasks such as real-time video processing and computer vision applications.

Inspired by Gstreamer's pipeline architecture, `func2stream` provides a clean and minimally invasive way to integrate your existing code into an efficient and scalable processing workflow.

## ✅ Latest Updates
  
- Better VideoIO performance with Gstreamer:
  - Moore Threads's VAAPI decoding is now enabled by default.
  - Intel's iHD VAAPI decoding is also enabled by default.
  - For more details, refer to [func2stream/utils.py](https://github.com/BICHENG/func2stream/blob/main/func2stream/utils.py#L39).

- "No-Fuss" Context Management, now with `ContextContainer` and `init_ctx`:
  - This enhancement makes it easier to manage and pass around state within your pipeline functions by encapsulating the state within closures.
  - To use `ContextContainer` feature:
    - JUST simply place some global initialization code in a function
    - decorate it with `@init_ctx`
    - use `return locals()` at the end of the function.
    - Call `your_ctx_name = your_init_func()` to get the `ContextContainer` object.
    - Use like a class instance to access all the variables by using `your_ctx_name.var_name`.

For example, you can **migrate your code to clean up global variables and class instances** by simply using `init_ctx`:

<details>
  <summary>Expand the full code example for migrating single, messy dashcam POC code to multiple dashcams</summary>

```python
import torch, cv2, numpy as np, time
from collections import deque
from func2stream import Pipeline, init_ctx, from_ctx
from VehicleOBDInterface import create_obd_interface # a mockup class for OBD data source

obd_datasorce = create_obd_interface(subscribe=['steering_angle,brake,throttle,speed'])

@init_ctx
def redlight_recognizer_dashcam(camera_id=0):
    # Those variables and functions may comes from a early test code for single dashcam
    # and now you want to apply them to multiple dashcams
    # without func2stream, you may:
    #   - use class instance to manage them, but got a class like a cthulhu(overkill)
    #   - use global variables, but got a global hell, and lock hell
    # func2stream provides a better way to manage them:
    #   - use `@init_ctx` to encapsulate them in a closure, and 'ContextContainer' will manage explicitly
    #   - use `return locals()` at the end, and all the variables will be managed by 'ContextContainer'

    # some global constants form your single dashcam POC code
    THRESHOLD = 0.5
    MODEL_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CAMERA_ID = camera_id

    # some debug varaibles can be modified somehow
    redlight_break_counter = 0
    frame_snap_shot = None

    # init a DNN model, or pass it as a parameter(recommended, but for simplicity, we init it here)
    # model is a classifer model, and we want to use it to recognize redlight
    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    model.eval().to(MODEL_DEVICE)

    snd_buffer = deque(maxlen=10)

    vout = cv2.VideoWriter(f'dashcam{i}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))

    @from_ctx(get=['frame'], set=['tensor'])
    def process_frame(frame):
        # Do some preprocessing
        frame = cv2.resize(frame, (224, 224))
        return torch.tensor(frame)

    @from_ctx(get=['tensor'])
    def infer(tensor):
        # Do some inference
        ret = model(tensor)
        snd_buffer.append(ret)

    @from_ctx(get=['frame'])
    def dbg_draw_and_save(frame):
        caption = f'Green Light' if snd_buffer[-1].argmax() == 0 else 'Red Light'
        cv2.putText(frame, f'{caption}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if snd_buffer[-1].argmax() == 0 else (0, 0, 255), 2)

        if frame_snap_shot is None:   # ugly debug code, just demo how to modify the varaibles from outside
          frame_snap_shot = frame.copy()
          cv2.imwrite(f'debug_{camera_id}.jpg', frame_snap_shot)
        cv2.imshow(f'Dashcam {camera_id}', frame)
        cv2.waitKey(1)
        vout.write(frame)

    # Add all the variables and functions to ContextContainer
    # ...

    # locals() EXPLAINED:
    # locals is just a hint for the ContextContainer to collect the variables and functions
    # All variables in the function scope are stored in the locals() dictionary,
    # but the locals() dictionary is a read-only snapshot of the interpreter's state.
    
    # The ContextContainer just uses keynames and rebuilds the attributes can be modified.
    # This approach allows for more secure and flexible handling of data, based on everyones' experience only.
    # You can still modify the contents of the deque object, do whatever you want with the model, etc.
    # However, nobody thinks it's a good idea to swap, delete, or monkey patch the members of a class instance,
    # the same applies to the ContextContainer. Initialize it, use it, but better not to fuck with it.
    return locals() 

# 4 dashcams instance can be created in one line
redlight_recognizer_dashcams = [redlight_recognizer_dashcam(i) for _ in range(4)] 

# build a pipeline for each dashcam, and start it
# use like a class instance to access all the variables by using `rrd.var_name`

uri_formatter = "rtsp://localhost:8554/dash/{}"
dashcam_pipelines = [Pipeline([
    VideoSource(uri_formatter.format(i),use_umat=False),
    build_ctx(key="frame"),
    rrd.process_frame, 
    rrd.infer,
    rrd.dbg_draw_and_save
]).start() for i,rrd in enumerate(redlight_recognizer_dashcams)]

# Now, you can access all the variables by using `rrd.var_name` in the pipeline functions.
# For example, `rrd.snd_buffer` will be accessible and you can aggregate the results from multiple dashcams.
while True:
    # Read data from OBD
    steering_angle, brake, throttle, speed = obd_datasorce.read()

    # Aggregate the results from multiple dashcams
    rets = [rrd.snd_buffer[-1] for rrd in redlight_recognizer_dashcams]
    # Do some aggregation
    status = sum(rets) > 2
    if status:
        print('AUTOHOLD WILL BE ACTIVATED')
        # for example, you can increase the counter
        for rrd in redlight_recognizer_dashcams:
            rrd.redlight_break_counter += 1

    elif status <= 1 and steering_angle <-10:
        print('UNPROTECTED LEFT TURN ACTION DETECTED')
        # for example, you can clear the buffer
        # modify the data inside the instance, not the instance itself
        for rrd in redlight_recognizer_dashcams:
            rrd.snd_buffer.clear()
        # and add some padding
        for rr in dashcam_pipelines:
            for i in range(10): rr.snd_buffer.append(0)

    if brake > 0.5 and throttle > 0.5 and speed <10:
        print('PANIC DEBUGGING')
        time.sleep(1) # for debugging
        for rrd in redlight_recognizer_dashcams:
            rrd.frame_snap_shot = None
        break
```

</details>

## 🚧 Current Status and Roadmap

`func2stream` is currently in early development and is being actively worked on. While it shows promise, it is not yet recommended for production use. The following areas are being prioritized for improvement:

- [ ] 🎯 Strive towards a decorator-free solution

  - [x] Implement automatic conversion of SISO functions to Elements and connect them in `Pipeline([fn1, fn2...])`

    *Decorators are aesthetically unpleasing and may not provide a foolproof solution. As other projects have attempted and abandoned.*

  - [ ] Eliminate the need for decorators in MIMO functions
    - [ ] Introduce implicit Context construction using AST parsing
    - [x] Simplify the process for users while maintaining necessary control and flexibility(see [ContextContainer](https://github.com/BICHENG/func2stream/blob/main/func2stream/core.py#L372))

- [ ] 🧩 Implement implicit Context

  - [ ] Extend the automation to MIMO functions
    - [ ] Use AST to parse function input and output parameters
    - [ ] Automatically convert parameters into variable keys
    - [ ] Enable users to build Context mode by ensuring consistent naming in function parameter lists and return statements
  - [ ] Simplify the process for users while maintaining necessary control and flexibility

## Key Features

### 1. Easy Integration with Existing Code

`func2stream` allows you to use your existing functions without modifying their internal logic. You can focus on your project's requirements while `func2stream` handles the underlying asynchronous `Pipeline` orchestration.

### 2. Intuitive Pipeline Construction

Building pipelines with `func2stream` is straightforward thanks to its `Element` abstraction and automatic `Pipeline` assembly.

- Functions can be easily transformed into `Element`s, promoting modularity and reusability in `Pipeline` design.
- Simply define the sequence of functions or Elements in your `Pipeline`, and `func2stream` will efficiently link them together.
- `Pipeline`s can be treated as `Element`s, allowing for the creation of complex, nested structures.

### 3. MapReduce Processing

`func2stream` incorporates the Map-Reduce paradigm, enabling easy parallelization of processing tasks for improved performance.

- Define concurrent `Element`s using the `MapReduce([fn1,fn2,fn3...])` syntax, and `func2stream` will handle parallel execution and result aggregation.
- The `MapReduce` functionality itself is an `Element`, seamlessly integrating parallel processing into your `Pipeline`.

### 4. Managed Context

`func2stream` introduces a managed `context` mechanism to streamline data flow through the pipeline, reducing the need for manual data management between Elements.

- The managed context handles data passing between Elements, making data usage across functions more intuitive and convenient.
- Decorators allow you to specify which variables an `Element` reads from and writes to the context.
- The centralized data store ensures that variables accessed and modified by different asynchronous steps remain independent, preventing unintended interactions and maintaining data integrity.
