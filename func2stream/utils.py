"""
utils.py

This file is part of func2stream: Utilities for user environment.

Author: BI CHENG
GitHub: https://github.com/BICHENG/func2stream
License: MPL2.0
Created: 2024/5/1

For Usage, please refer to https://github.com/BICHENG/func2stream/samples or README.md
"""

__author__ = "BI CHENG"
__version__ = "1.0.0-pre"
__license__ = "MPL2.0"


def find_gstreamer():
    import cv2
    build_info = cv2.getBuildInformation().split('\n')
    gstreamer_info = None
    for line in build_info:
        if 'GStreamer' in line:
            gstreamer_info = line
            break
    if gstreamer_info is None:
        print("GStreamer information not found in OpenCV build information.")
        return False, "Unknown"

    tokens = gstreamer_info.split()
    # Typically, tokens[1] is 'YES' and the version follows.
    # The structure might look like: ['GStreamer:', 'YES', '(1.16.2)']        
    gstreamer_found = True if tokens[1] == "YES" else False
    gstreamer_version = "Unknown" if len(tokens) < 3 else tokens[2]
    return gstreamer_found,gstreamer_version


def init_gstreamer_hwaccel_env(vendor):
    """vendor: intel/mt/nvidia"""
    import os
    os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapih264dec:1024,vaapih265dec:1024,nvh264sldec:1024,nvh265sldec:1024,nvh264dec:1024,nvh265dec:1024" # HW acceleration for Intel, Moore Threads and NVIDIA
    if not vendor in ["intel","mt","nvidia"]:
        print(f"WARNING: Vendor {vendor} is not supported for HW acceleration yet.")
        print("You can open an issue on GitHub to request support. The Chinese vendor will be first-priority supported.")
        print("\thttps://github.com/BICHENG/func2stream/issues")
        return


    if vendor == "intel": # Intel media driver(iHD)
        # You could enable guc/huc by modifying the kernel parameters for better performance.
        # see: https://github.com/intel/media-driver#known-issues-and-limitations
        
        os.environ["LIBVA_DRIVER_NAME"] = "iHD"
        os.environ["LIBVA_DRIVERS_PATH"] = "/usr/lib/x86_64-linux-gnu/dri" # /usr/lib/x86_64-linux-gnu/dri/iHD_drv_video.so
    
    elif vendor == "mt": # Moore Threads musa version 2.7+
        # PLEASE ENSURE YOU CAN CONTACT MOORE THREADS SUPPORT BEFORE USING THIS.
        # Fragile: VAAPI CAN BE BROKEN IN MANY CASES, BUT FFmpeg HWACCEL CAN STILL WORK SOMEHOW.
        #   e.g. Resolution not 1080p, no monitor connected, etc.
        #   e.g. DKMS not installed, kernel version higher than 5.4, etc.
        #   e.g. Above 4G decoding not enabled, resizeble BAR not enabled, etc.
        #   e.g. Ubuntu 20.04 may be the only supported version
        #   e.g. VSCode may cause VAAPI to fail???
        
        os.environ["GST_VAAPI_ALL_DRIVERS"] = "2"
        os.environ["GST_VAAPI_DRM_DEVICE"] = "/dev/dri/renderD128"
        os.environ["LIBVA_DRIVER_NAME"] = "mtgpu"
        os.environ["LIBVA_DRIVERS_PATH"] = "/usr/lib/x86_64-linux-gnu/dri" # /usr/lib/x86_64-linux-gnu/dri/mtgpu_drv_video.so
   
    
    # NOTE: NVIDIA should use nvcodec, no need to set VAAPI environment variables

    # ABANDONED: AMD GPU, HYGON DPU(Still a AMD GPU), etc.
    # PLEASE SET VDPAU_DRIVER, LIBVA_DRIVER_NAME, LIBVA_DRIVERS_PATH, etc. manually.
    # And set GST_PLUGIN_FEATURE_RANK for AMD GPU