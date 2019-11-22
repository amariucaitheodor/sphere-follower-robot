#!/usr/bin/env python

import cv2
import numpy as np

to_meters_ratio_img1 = 0.04080782932503862  # Precomputed with vis.pixel2meter(yellow_mask, blue_mask) in image 1
to_meters_ratio_img2 = 0.04311306135592269  # Precomputed with vis.pixel2meter(yellow_mask, blue_mask) in image 2
# Precomputed with detect_blob_center(cv2.inRange(self.cv_image2, (0, 100, 100), (80, 255, 255))) in image 2, when orange sphere was not interfering
yellow_blob_center_img2 = np.array([399, 533])
yellow_blob_center_img1 = np.array([399, 533])
sphere_template = cv2.imread("src/ivr_assignment/src/templates/sphere.png", 0)


# Detecting the centre of a colored circle
def detect_blob_center(mask):  # mask isolates the color in the image as a binary image
    # This applies a dilate that makes the binary region larger (the more iterations the larger it becomes)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # Obtain the moments of the binary image
    M = cv2.moments(mask)
    # Calculate pixel coordinates for the centre of the blob
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array([cx, cy])


# Calculate the conversion from pixels to meters
def pixel2meter(yellow_mask, blue_mask):
    yellow_joint = detect_blob_center(yellow_mask)
    blue_joint = detect_blob_center(blue_mask)
    # find the distance between two circles
    dist = np.sum((yellow_joint - blue_joint) ** 2)
    return 2 / np.sqrt(dist)  # link between yellow and blue is 2 meters
