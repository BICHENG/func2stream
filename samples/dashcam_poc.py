"""
THIS IS A PSEUDO-CODE FOR A DASHCAM POC, YOU CANNOT RUN IT DIRECTLY.

This pseudocode demonstrates how to manage variables and functions across multiple dashcams using the `func2stream` library.
It is designed as a proof-of-concept (POC) to detect and respond to red lights using dashcam footage.

Overview:
============
The pseudocode initializes a pipeline for each dashcam that processes video frames, infers red light signals, and draws debug information.
Additionally, it simulates integration with an OBD (On-Board Diagnostics) data source to enhance situational awareness and to enable more advanced decision-making based on vehicle status.

Key Concepts:
-----------------
- `ContextContainer`: Manages the variables and functions using `locals()` to encapsulate them in a structured, easily manageable manner.
- `func2stream`: A library designed to simplify the creation and handling of data processing pipelines for stream data.
- `OBD Data Source`: This simulates real-time vehicle data inputs which can be used to refine and improve the situational awareness of the system.

Implementation Details:
-------------------------------
1. **Initialization and setup**:
    - The OBD interface is created and subscribed to various vehicle parameters like steering angle, brake status, throttle position, and speed.
    - `redlight_recognizer_dashcam` function initializes the variables and functions necessary for processing dashcam data and utilizes `@init_ctx` to encapsulate these variables in a closure.

2. **Pipeline Construction**:
    - For each dashcam, a pipeline is established which consists of video frame processing, inference using a pre-trained model, and debugging/drawing results on the frames.
    - The variables and functions used in these processes are managed by `ContextContainer`, ensuring they remain organized and modifiable without leading to code complexity or global state issues.

3. **Data Processing and Inference**:
    - Video frames are processed and resized before being fed into a pre-trained deep neural network model (e.g., ResNet18) for inference.
    - The inference results are buffered and used to determine the presence of a red light, with corresponding debug information drawn onto the video frames.

4. **Handling Multiple Dashcams**:
    - The system is designed to scale to multiple dashcams seamlessly, with each dashcam initialized and managed through a well-defined pipeline.
    - The use of `locals()` helps encapsulate all relevant variables within the function scope, but the `ContextContainer` makes those variables accessible and modifiable across the entire pipeline.

5. **Integration with OBD Data**:
    - The pseudocode incorporates mock OBD data for enhanced decision-making. By reading data such as steering angle, brake, throttle, and speed, the system can aggregate results from multiple dashcams to make more informed decisions.
    - This integration allows for advanced features like detecting unsafe driving maneuvers (e.g., unprotected left turns) or emergency situations (e.g., panic debugging).

Note:
---------
This pseudocode serves as a conceptual example and require adjustments for real-world applications. It demonstrates the potential of using the `func2stream` library and `ContextContainer` to manage and process stream data from multiple dashcams effectively.
"""



import torch, cv2, numpy as np, time
from collections import deque
from func2stream import Pipeline, init_ctx, from_ctx
from VehicleOBDInterface import create_obd_interface # a mockup class for OBD data source

obd_datasorce = create_obd_interface(subscribe=['steering_angle,brake,throttle,speed'])

@init_ctx
def redlight_recognizer_dashcam(camera_id=0):
    # The variables and functions you have might originate from initial test code for a single dashcam.
    # For demonstration, suppose you want to extend this code to manage multiple dashcams.
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

    # ----------------------
    # Understanding `locals()`
    # ----------------------
    # The `locals()` function serves as a hint for the `ContextContainer` to gather variables and functions within its scope.
    # Here’s a deeper explanation:

    # In the function scope, all local variables are stored in a dictionary returned by `locals()`.
    # The dictionary provided by `locals()` offers a read-only snapshot of the interpreter's current state.

    # The `ContextContainer` leverages the keys from this dictionary to reconstruct its attributes, allowing them to be modified.
    # This approach enhances both the security and flexibility in managing data, based on collective best practices.

    # While you are free to manipulate the contents of objects such as deque, or interact with the model, it is generally advised
    # against swapping, deleting, or monkey patching instance members of a class. The same principle applies to the `ContextContainer`:
    # it should be initialized and utilized as intended, but modifications beyond typical use are discouraged to maintain stability
    # and predictability.
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