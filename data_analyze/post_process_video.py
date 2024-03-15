#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 28 10:47:58 2023

@author: LJ
"""

import cv2
from tqdm import tqdm


def extract_trajectory(video_path: str):
    """
    Extract trajectory of object in Video.

    :param video_path: The Video path.
    :return: The trajectory.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = cap.read()
    tracker = cv2.TrackerMIL_create()
    bbox = cv2.selectROI("Object Tracking", frame, False)
    cv2.destroyWindow("Object Tracking")
    _ = tracker.init(frame, bbox)

    locations = []
    print("Tracking Object...")
    for fno in tqdm(range(1, total_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        _, frame = cap.read()
        ret, bbox = tracker.update(frame)
        if ret:
            middle = int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2)
            locations.append(middle)
    else:
        cv2.destroyAllWindows()
    return locations
