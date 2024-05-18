
# This example demonstrates how to build a video processing pipeline using OpenCV and a custom pipeline library. 
#
# The goal is to read video frames, process them in parallel through different filters, and then combine the 
# results into a single image for display. Specifically, it performs edge detection, Gaussian blurring, and 
# grayscale conversion using these steps:
# 1. Grayscale Conversion: Convert each frame to a grayscale image.
# 2. Parallel Processing:
#    - Canny Edge Detection: Detect edges in the grayscale image.
#    - Gaussian Blurring: Apply Gaussian blur to the grayscale image.
# 3. Merging Results: Combine the outputs of the edge detection, Gaussian blurring, and original grayscale image into a single
#    color image where each result is a different channel.
# 4. Display: Show the combined image in a window.
#
# The pipeline design pattern helps in organizing complex image processing workflows by breaking the task into smaller, manageable, and reusable components:
# - Contextual Data Sharing:
#   - `build_ctx` and `from_ctx` facilitate efficient context creation and data sharing across different stages of the pipeline.
#   - The `ctx` object allows you to access shared data across different processing steps, ensuring that the correct information is available at each stage.
#   - No need declare global variables or pass data through function arguments.
# - Sequential and Parallel Mix:
#   - Pipeline effectively combines sequential processing steps (e.g., frame ingestion, detection) with parallel processing (e.g., attribute and descriptor extraction), providing a balanced approach to optimize performance.
#
# This approach, using Pipeline and MapReduce, is well-suited for scenarios where:
# - You need to process video frames or images through multiple independent processing steps.
# - You want to parallelize certain operations to improve performance.
# - You need to combine or merge results from different processing paths.
#
# The pipeline design pattern helps in organizing complex image processing workflows by breaking the task into smaller, manageable,
# and reusable components. It is particularly useful for real-time video processing, video analytics, and any application where
# structured, incremental data processing is required.
#
# THIS CODE IS A SIMPLIFIED EXAMPLE AND MAY NOT WORK AS-IS. PLEASE ADAPT IT TO YOUR SPECIFIC USE CASE.
#
# About pipeline depth: The pipeline depth is the number of processing steps in the pipeline. 
# A deeper pipeline may introduce additional latency, but it can also improve performance by parallelizing tasks and optimizing resource utilization.

import cv2
import numpy as np
from func2stream import Pipeline, VideoSource, MapReduce, build_ctx, from_ctx

# Define processing functions and convert them into pipeline elements
@from_ctx(get=["frame"], ret=["gray"])
def grayscale_conversion(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

@from_ctx(get=["gray"], ret=["edges"])
def canny_edge_detection(gray):
    return cv2.Canny(gray, 100, 200)

@from_ctx(get=["gray"], ret=["blurred"])
def gaussian_blur(gray):
    return cv2.GaussianBlur(gray, (15, 15), 0)

@from_ctx(get=["edges", "blurred", "gray"], ret=["combined"])
def merge_results(edges, blurred, gray):
    return cv2.merge((edges, blurred, gray))

@from_ctx(get=["combined"])
def display_result(combined):
    cv2.imshow("Processed Frame", combined)
    cv2.waitKey(1)

# Initialize the pipeline
video_source = 0  # Can be a camera index or a video file path
pipeline = Pipeline([
    VideoSource(video_source),
    build_ctx("frame"),
    (grayscale_conversion),
    MapReduce([
        (canny_edge_detection),
        (gaussian_blur),
    ], "ImgProc").ctx_mode(
        get=[["gray"], ["gray"]],
        ret=[["edges"], ["blurred"]]
    ),
    (merge_results),
    (display_result)
]).start()

if __name__ == "__main__":
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Processing stopped by user")
        cv2.destroyAllWindows()