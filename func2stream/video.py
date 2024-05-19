"""
video.py

This file is part of func2stream: DataSources for video stream processing.

Author: BI CHENG
GitHub: https://github.com/BICHENG/func2stream
License: MPL2.0
Created: 2024/5/1

For Usage, please refer to https://github.com/BICHENG/func2stream/samples or README.md
"""

__author__ = "BI CHENG"
__version__ = "0.0.0"
__license__ = "MPL2.0"


import os,time,threading,traceback,queue
import cv2

from .core import DataSource
from .utils import find_gstreamer

class _VideoCapture:
    def __init__(self, uri, cap_options={}, use_umat=False):
        self.uri = uri
        self.cap_options = cap_options if len(cap_options) > 0 else self.get_capture_params(uri)
        self._swap = queue.Queue(1)
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._worker, name="VideoCapture", daemon=True)
        self.thread.start()
        self.use_umat = use_umat
    
    def get_capture_params(self, video_uri):
        import sys
        """
        Automatically recognize the mode based on the input video_uri and return the parameter list required for video capture.
        
        :param video_uri: Identifier of the resource, such as the path of the video file or the URL of the network stream
        :return: Parameter list or configuration for initializing cv2.VideoCapture
        """     
        
        mode = ""
        if sys.platform == "win32" and video_uri.isdigit(): 
            mode = "uvc"
        elif sys.platform == "linux" and video_uri.startswith("/dev/video"): 
            mode = "uvc"
        elif video_uri.startswith("rtsp://"): 
            mode = "rtsp"
        elif video_uri.startswith("rtmp://"): 
            mode = "rtmp"
        elif video_uri.startswith("gst-launch-1.0 "):
            mode = "gst"
        else:
            # 检查是否是视频文件URI
            uri_mode_map = {
                ".mp4": "video", ".avi": "video", ".mkv": "video"
            }
            for ext, possible_mode in uri_mode_map.items():
                if video_uri.endswith(ext): 
                    mode = possible_mode
                    video_uri=os.path.abspath(video_uri)
                    break
        print(mode)
        assert mode, f"Unrecognized video resource: {video_uri}, available modes include: uvc, rtsp, rtmp, video file path"
        
        # 依据模式返回不同的参数
        if mode == "uvc":
            if sys.platform == "win32":
                return [int(video_uri)]
            elif sys.platform == "linux":
                return [video_uri, cv2.CAP_V4L]

        elif mode in ["rtsp", "rtmp"]:
            pipeline_base = {
                "rtsp": f"rtspsrc location={video_uri} latency=50 ! queue ! parsebin ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true sync=false",
                "rtmp": f"rtmpsrc location={video_uri} ! queue ! parsebin ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true sync=false"
            }
            gst_found, gst_version = find_gstreamer()
            if not gst_found:
                print(f"Warning: OpenCV is built without GStreamer support, {mode} will try to use FFMPEG backend")
                print(f"\tYOUR {mode.upper()} MAY SUFFER LATENCY ISSUES!")
                return [video_uri]
            return [pipeline_base[mode], cv2.CAP_GSTREAMER]

        elif mode == "video":
            appsink_config = "appsink max-buffers=1 drop=false"
            pipeline = f"filesrc location={video_uri} ! decodebin ! videoconvert ! {appsink_config} sync=false"

            return [
                video_uri,cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
                ]
        
    def _worker(self):
        while not self.stop_flag.is_set():
            try:
                if isinstance(self.cap_options, dict):
                    cap = cv2.VideoCapture(self.uri, **self.cap_options)
                elif isinstance(self.cap_options, list):
                    cap = cv2.VideoCapture(*self.cap_options)
                else:
                    raise Exception(f"Unrecognized cap_options type: {type(self.cap_options)}")
                   
                if not cap.isOpened():
                    raise Exception(f"cap.isOpened() returns False")
                if self.use_umat: buf = cv2.UMat(cap.read()[1])
                else: buf = cap.read()[1]
                self._swap.put(buf)
                print(f"{self.uri} opened")
                while buf is not None and not self.stop_flag.is_set():
                    if self._swap.full():
                        time.sleep(0.0001)
                        continue
                    #     self._swap.get()
                    self._swap.put(buf)
                    good = cap.grab()
                    good, buf = cap.retrieve(buf)
                cap.release()
            except Exception as e:
                traceback_info = '\t'.join(traceback.format_exception(None, e, e.__traceback__))
                print(f"VideoCapture@{self.uri} will try to reopen, reason：{e}, traceback: {traceback_info}")
                time.sleep(1)
        print(f"{self.uri} closed")
    def read(self):
        return self._swap.get().copy()
    
    def stop(self):
        self.stop_flag.set()
        self.thread.join()
        return self
 
class VideoSource(DataSource):
    def __init__(self, uri, cap_options={}, use_umat=False,friendly_name=""):
        self.video_capture = _VideoCapture(uri, cap_options, use_umat)
        super().__init__(reader_call=self.video_capture.read,
                         friendly_name=uri if friendly_name == "" else friendly_name)

    def stop(self):
        super().stop()
        self.video_capture.stop()
        return self
