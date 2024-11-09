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
__version__ = "0.1.0"
__license__ = "MPL2.0"

import os,sys,time,threading,traceback,queue
import cv2

from .core import DataSource
from .utils import find_gstreamer, init_gstreamer_hwaccel_env

class _VideoCapture:
    def __init__(self, uri, cap_options={}, use_umat=False, gst_hwaccel_vendor="", reopen = True):
        self.uri = uri
        self.cap_options = cap_options if len(cap_options) > 0 else self.get_capture_params(uri)
        self._swap = queue.Queue(1)
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._worker, name="VideoCapture", daemon=True)
        self.thread.start()
        self.use_umat = use_umat
        self.cap = None
        self.reopen = reopen
        
        if gst_hwaccel_vendor:
            init_gstreamer_hwaccel_env(gst_hwaccel_vendor)
            print(f"NOTE: Gstreamer will only use in rtsp/rtmp mode, and will try to use HW acceleration for {gst_hwaccel_vendor}")
        
        self._need_restart = False     
           
    def get_capture_params(self, video_uri):        
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
                "rtsp": f"rtspsrc location={video_uri} latency=0 ! queue ! parsebin ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true sync=false",
                "rtmp": f"rtmpsrc location={video_uri} ! queue ! parsebin ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true sync=false"
            }
            gst_found, gst_version = find_gstreamer()
            if not gst_found:
                print(f"Warning: OpenCV is built without GStreamer support, {mode} will try to use FFMPEG backend")
                print(f"\tYOUR {mode.upper()} MAY SUFFER LATENCY ISSUES!")
                return [video_uri]
            return [pipeline_base[mode], cv2.CAP_GSTREAMER]

        elif mode == "video":
            # appsink_config = "appsink max-buffers=1 drop=false"
            # pipeline = f"filesrc location={video_uri} ! decodebin ! videoconvert ! {appsink_config} sync=false"

            return [
                video_uri,cv2.CAP_FFMPEG,
                # [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
                ]
        
    def _worker(self):
        frame_cnt = 0
        while not self.stop_flag.is_set():
            try:
                if isinstance(self.cap_options, dict):
                    self.cap = cv2.VideoCapture(self.uri, **self.cap_options)
                elif isinstance(self.cap_options, list):
                    self.cap = cv2.VideoCapture(*self.cap_options)
                else:
                    raise Exception(f"Unrecognized cap_options type: {type(self.cap_options)}")
                   
                if not self.cap.isOpened():
                    raise Exception(f"VideoCapture.isOpened() returns False")
                if self.use_umat: buf = cv2.UMat(self.cap.read()[1])
                else: buf = self.cap.read()[1]
                self._swap.put(buf)
                print(f"{self.uri} opened")
                while buf is not None and not self.stop_flag.is_set():
                    if self._swap.full():
                        time.sleep(0.0001)
                        continue
                    #     self._swap.get()
                    self._swap.put(buf)
                    good = self.cap.grab()
                    good, buf = self.cap.retrieve(buf)
                    frame_cnt += 1
                self.cap.release()
            except Exception as e:
                traceback_info = '\t'.join(traceback.format_exception(None, e, e.__traceback__))
                if not self.reopen: 
                    for i in range(frame_cnt*10): 
                        self._swap.put(None)
                        print(f"{self.uri} closed, frame_cnt: {frame_cnt}")
                
                print(f"VideoCapture@{self.uri} will try to reopen, reason：{e}, traceback: {traceback_info}")
                
                time.sleep(1)
        print(f"{self.uri} closed")
    def read(self):
        return self._swap.get().copy()
    
    def stop(self):
        self.stop_flag.set()
        self.thread.join()
        return self
    
    def kill(self):
        try:
            self.cap.release()
            print(f"{self.uri} closed, killed")
        except:
            del self.cap
            self.cap = None
            print(f"{self.uri} can't be closed, forced killed")
        return self
 
 
class UVC_VideoCapture(_VideoCapture):
    def __init__(self, device_id=0, width=1920, height=1080, fourcc="MJPG", backend=cv2.CAP_DSHOW,
                 cap_prop_settings={},
                 use_umat=False):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fourcc = fourcc
        self.backend = backend
        self.cap_prop_settings = cap_prop_settings
        super().__init__(uri=str(device_id), use_umat=use_umat)

    def _worker(self):
        while not self.stop_flag.is_set():
            try:
                cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
                for prop_id, prop_value in self.cap_prop_settings.items():
                    cap.set(prop_id, prop_value)

                if not cap.isOpened():
                    raise Exception(f"Failed to open camera {self.device_id}")

                if self.use_umat:
                    buf = cv2.UMat(cap.read()[1])
                else:
                    buf = cap.read()[1]
                self._swap.put(buf)
                print(f"Camera {self.device_id} opened")

                while buf is not None and not self.stop_flag.is_set():
                    if self._swap.full():
                        time.sleep(0.0001)
                        continue
                    self._swap.put(buf)
                    good = cap.grab()
                    good, buf = cap.retrieve(buf)
                cap.release()
            except Exception as e:
                traceback_info = '\t'.join(traceback.format_exception(None, e, e.__traceback__))
                print(f"UVCVideoCapture@{self.device_id} will try to reopen, reason：{e}, traceback: {traceback_info}")
                time.sleep(1)
        print(f"Camera {self.device_id} closed")

class VideoSource(DataSource):
    def __init__(self, uri, cap_options={}, use_umat=False,friendly_name="",reopen=True):
        self.video_capture = _VideoCapture(uri, cap_options, use_umat, reopen=reopen)
        super().__init__(reader_call=self.video_capture.read,
                         friendly_name=uri if friendly_name == "" else friendly_name)

    def stop(self):
        super().stop()
        self.video_capture.stop()
        return self
    
    def reopen(self):
        self.video_capture.kill()
        return self
    
class VideoSource_UVC(DataSource):
    def __init__(self, device_id=0, width=1920, height=1080, fourcc="MJPG", backend=cv2.CAP_DSHOW,
                 cap_prop_settings={},
                 use_umat=False,friendly_name=""):
        self.video_capture = UVC_VideoCapture(device_id, width, height, fourcc, backend, cap_prop_settings, use_umat)
        super().__init__(reader_call=self.video_capture.read,
                         friendly_name=f"UVC Camera {device_id}" if friendly_name == "" else friendly_name)

    def stop(self):
        super().stop()
        self.video_capture.stop()
        return self