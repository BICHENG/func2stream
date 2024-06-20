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
      - decorate it with `@init_ctx`.
      - use `return locals()` at the end of the function.
    - Call `your_ctx_name = your_init_func()` to get the `ContextContainer` object.
    - Use like a class instance to access all the variables by using `your_ctx_name.var_name`.
    - For example, you can migrate your single dashcam POC code to multiple dashcams with ease:
      ```python
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

          # init a DNN model, or pass it as a parameter(recommended, but for simplicity, we init it here)
          # model is a classifer model, and we want to use it to recognize redlight
          model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
          model.eval()

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
          def draw_and_save(frame):
              # Do some postprocessing
              caption = f'Inference result: {snd_buffer[-1].argmax()}'
              cv2.putText(frame, f'{caption}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
              vout.write(frame)

          # Add all the variables and functions to ContextContainer
          # ...

          return locals() # Here, all the variables will be managed by ContextContainer

      redlight_recognizer_dashcams = [redlight_recognizer_dashcam(i) for _ in range(4)] # make 4 dashcams
      dashcam_pipelines = []
      for rrd in redlight_recognizer_dashcams:
          pipeline = Pipeline([rrd.process_frame, rrd.infer, rrd.draw_and_save]) # use like a class instance
          dashcam_pipelines.append(pipeline).start()

      # Now, you can access all the variables by using `rrd.var_name` in the pipeline functions.
      # For example, `rrd.snd_buffer` will be accessible and you can aggregate the results from multiple dashcams.
      while True:
          # Aggregate the results from multiple dashcams
          rets = [rrd.snd_buffer[-1] for rrd in redlight_recognizer_dashcams]
          # Do some aggregation
          status = sum(rets) > 2
          if status:
              # Do some actions
              pass
          else:
              # Do some actions
              pass
      ```
    
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
