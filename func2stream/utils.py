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
__version__ = "0.0.0"
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