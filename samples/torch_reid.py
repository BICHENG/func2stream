
# This example demonstrates a Pipeline for multi-object tracking and recognition.
# The process is as follows:
# 1. Image Preprocessing and Tensor Loading:
#    - Convert the video frame into a tensor suitable for deep learning model inputs.
# 2. Object Detection:
#    - Use a detector to identify all persons in the frame and return bounding box information.
# 3. Object Tracking:
#    - Based on detection results, update the tracker to maintain a dictionary that stores the bounding box positions and timestamps for all current IDs.
# 4. MapReduce Example:
#    - Execute in parallel the following tasks for each detected bounding box:
#      - Pose Detection: Estimate the skeletal pose of each person.
#      - Attribute Detection: Identify various attributes (e.g., clothing color, accessories) of each person.
#      - Face Descriptor Extraction: Extract facial features as descriptors for each person.
#      - ReID Descriptor Extraction: Extract ReID (Re-identification) features to uniquely identify each person across frames.
# 5. Result Update:
#    - Based on detection and tracking results, add each piece of information (pose, attributes, face descriptor, and ReID descriptor) to the corresponding track ID's attributes.
#
# This approach, using Pipeline and MapReduce, is well-suited for scenarios where you need to process video frames or images through multiple independent processing steps,
# parallelize certain operations to improve performance, and combine or merge results from different processing paths.
# The pipeline design pattern helps in organizing complex AI workflows by breaking the task into smaller, manageable, zero-copy, and reusable components.
# - VideoSource: Ingests the video file or stream, providing frames to the pipeline.
# - Parallel Processing:
#   - The `MapReduce` element allows multiple processing tasks (pose detection, attribute detection, etc.) to be executed in parallel, enhancing performance and efficiency.
#   - `ctx_mode` ensures that the correct context (input tensors and bounding boxes) is passed to each processing function.
# - Modularity:
#   - Each element of the pipeline (e.g., `to_tensor`, `detect_persons`) is your custom processing function, encapsulated within a pipeline element.
#   - This modularity allows for easy replacement or modification of specific steps (e.g., switching the person detector or tracker).
# - Contextual Data Sharing:
#   - `build_ctx` and `from_ctx` facilitate efficient context creation and data sharing across different stages of the pipeline.
#   - The `ctx` object allows you to access shared data across different processing steps, ensuring that the correct information is available at each stage.
#   - No need declare global variables or pass data through function arguments.
# - Sequential and Parallel Mix:
#   - Pipeline effectively combines sequential processing steps (e.g., frame ingestion, detection) with parallel processing (e.g., attribute and descriptor extraction), providing a balanced approach to optimize performance.
# It is particularly useful for real-time video processing, video analytics, and any application where structured, incremental data processing is required.
#
# THIS CODE IS A SIMPLIFIED EXAMPLE AND MAY NOT WORK AS-IS. PLEASE ADAPT IT TO YOUR SPECIFIC USE CASE.
#
# About pipeline depth: The pipeline depth is the number of processing steps in the pipeline. 
# A deeper pipeline may introduce additional latency, but it can also improve performance by parallelizing tasks and optimizing resource utilization.

from some_detection_library import PersonDetector
from some_tracking_library import Tracker
from some_pose_library import PoseEstimator
from some_attribute_library import AttributeDetector
from some_face_library import FaceDescriptor
from some_reid_library import ReIDDescriptor
from pipeline import Pipeline, VideoSource, Element, MapReduce, build_ctx, from_ctx

# 实例化所有处理对象
person_detector = PersonDetector()
tracker = Tracker()
pose_estimator = PoseEstimator()
attribute_detector = AttributeDetector()
face_descriptor = FaceDescriptor()
reid_descriptor = ReIDDescriptor()

# 预处理和tensor加载函数
@from_ctx(get=["frame"], ret=["tensor"])
def to_tensor(ctx):
    frame = ctx
    tensor = ...  # 实现你的图片转换为tensor的逻辑
    return tensor

# 图像检测器，用于检测所有人的bbox
@from_ctx(get=["tensor"], ret=["person_bboxes"])
def detect_persons(ctx):
    tensor = ctx
    person_bboxes = person_detector.detect(tensor)
    return person_bboxes

# 进行bbox的跟踪，并得到trk_id
@from_ctx(get=["person_bboxes", "frame"], ret=["trk_info"])
def update_trk(ctx):
    person_bboxes, frame = ctx
    trk_info = tracker.update(person_bboxes, frame)
    return trk_info

# 骨骼检测
@from_ctx(get=["tensor", "person_bboxes"], ret=["poses"])
def infer_pose(ctx):
    tensor, person_bboxes = ctx
    poses = [pose_estimator.estimate(tensor, bbox) for bbox in person_bboxes]
    return poses

# 属性检测
@from_ctx(get=["tensor", "person_bboxes"], ret=["attributes"])
def infer_attributes(ctx):
    tensor, person_bboxes = ctx
    attributes = [attribute_detector.detect(tensor, bbox) for bbox in person_bboxes]
    return attributes

# 面部特征描述子
@from_ctx(get=["tensor", "person_bboxes"], ret=["face_descriptors"])
def infer_face_descriptor(ctx):
    tensor, person_bboxes = ctx
    face_descriptors = [face_descriptor.descriptor(tensor, bbox) for bbox in person_bboxes]
    return face_descriptors

# ReID特征描述子
@from_ctx(get=["tensor", "person_bboxes"], ret=["reid_descriptors"])
def infer_reid_descriptor(ctx):
    tensor, person_bboxes = ctx
    reid_descriptors = [reid_descriptor.descriptor(tensor, bbox) for bbox in person_bboxes]
    return reid_descriptors

# 更新结果
@from_ctx(get=["trk_info", "poses", "attributes", "face_descriptors", "reid_descriptors"], ret=["trk_info"])
def update_result(ctx):
    trk_info, poses, attributes, face_descriptors, reid_descriptors = ctx
    for i, trk_id in enumerate(trk_info.keys()):
        trk_info[trk_id]["pose"] = poses[i]
        trk_info[trk_id]["attributes"] = attributes[i]
        trk_info[trk_id]["face_descriptor"] = face_descriptors[i]
        trk_info[trk_id]["reid_descriptor"] = reid_descriptors[i]
    return trk_info

# Pipeline 设计
pipeline = Pipeline([
    VideoSource('video.mp4'),      # 视频源
    build_ctx("frame"),            # 建立上下文"frame"
    (to_tensor),                   # 预处理和tensor加载
    (detect_persons),              # 图片检测器
    (update_trk),                  # 更新跟踪
    MapReduce([                    # 并行执行检测与描述子
        (infer_pose),
        (infer_attributes),
        (infer_face_descriptor),
        (infer_reid_descriptor)
    ], "Infer").ctx_mode(
        get=[["tensor", "person_bboxes"], ["tensor", "person_bboxes"], ["tensor", "person_bboxes"], ["tensor", "person_bboxes"]],
        ret=[["poses"], ["attributes"], ["face_descriptors"], ["reid_descriptors"]]
    ),
    (update_result)                # 更新结果
]).start()