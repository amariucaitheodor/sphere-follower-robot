#!/usr/bin/env python

import cv2
import numpy as np

sphere_template = cv2.imread("/home/theo/catkin_ws/src/sphere-follower-robot/src/templates/sphere.png", 0)


# Detecting the centre of a colored circle
def detect_blob_center(mask):  # mask isolates the color in the image as a binary image
    # This applies a dilate that makes the binary region larger (the more iterations the larger it becomes)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Obtain the moments of the binary image
    m = cv2.moments(mask)

    # Blob is hidden
    if m['m00'] == 0:
        return None

    # Calculate pixel coordinates for the centre of the blob
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])
    return np.array([cx, cy])


# Calculate the conversion from pixels to meters
def pixel2meter(yellow_mask, blue_mask):
    yellow_joint = detect_blob_center(yellow_mask)
    blue_joint = detect_blob_center(blue_mask)

    # find the distance between two circles
    dist = np.sum((yellow_joint - blue_joint) ** 2)
    return 2 / np.sqrt(dist)  # link between yellow and blue is 2 meters
