#!/usr/bin/env python
from __future__ import print_function

import cv2
import rospy
import numpy as np
import visionlib as vis
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError


def find_target(mask, template, target_history, zy=False):
    # Build template
    w, h = template.shape[::-1]

    # Apply template matching
    res = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # max_loc is the top left point of the match
    center = (max_loc[0] + w / 2, max_loc[1] + h / 2)

    # To detect the box:
    # box_template = cv2.imread("src/ivr_assignment/src/box.png", 0)
    if max_val < 6800000:
        if zy:
            return np.array([target_history[1], target_history[2]])
        else:
            return np.array([target_history[0], target_history[2]])

    if zy:
        target_history[1] = center[0]
    else:
        target_history[0] = center[0]
    target_history[2] = center[1]
    return np.array([center[0], center[1]])


class TargetEstimator:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('target_estimation', anonymous=True)

        # initialize a subscriber to receive messages from a topic named /robot/camera1/image_raw and use callback
        # function to receive data
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.image1_callback, queue_size=10)
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.image2_callback, queue_size=10)

        # initialize a subscriber to receive messages from topics named pix_to_m_ratio and use callback function to
        # receive data
        self.cam1_ratio_sub = rospy.Subscriber("/camera1/pix_to_m_ratio", Float64, self.get_camera1_ratio, queue_size=10)
        self.cam2_ratio_sub = rospy.Subscriber("/camera2/pix_to_m_ratio", Float64, self.get_camera2_ratio, queue_size=10)
        self.pix_to_m_ratio_img1 = 1.0
        self.pix_to_m_ratio_img2 = 1.0

        # initialize publisher to publish target position estimate [x, y, z]
        self.target_position_pub = rospy.Publisher("/target_position_estimate", Float64MultiArray, queue_size=10)
        self.target_position = Float64MultiArray()
        self.target_position.data = [0.0, 0.0, 0.0]

        self.target_history = [0.0, 0.0, 0.0]

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

    def get_camera1_ratio(self, ratio):
        self.pix_to_m_ratio_img1 = ratio.data
        # print(self.pix_to_m_ratio_img1, " ratio 1")

    def get_camera2_ratio(self, ratio):
        self.pix_to_m_ratio_img2 = ratio.data
        # print(self.pix_to_m_ratio_img2, " ratio 2")

    def image1_callback(self, data):
        try:
            cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")

            # IMAGE 1
            # Color masks (BGR)
            orange_mask = cv2.inRange(cv_image1, (75, 100, 125), (90, 180, 220))
            kernel = np.ones((3, 3), np.uint8)
            orange_mask = cv2.erode(orange_mask, kernel, iterations=1)
            orange_mask = cv2.dilate(orange_mask, kernel, iterations=1)
            sphere_position = find_target(orange_mask, vis.sphere_template, self.target_history, True)

            yellow_mask = cv2.inRange(cv_image1, vis.yellow_mask_low, vis.yellow_mask_high)
            base_frame = vis.detect_blob_center(yellow_mask)
            if base_frame is None:
                rospy.logwarn("Cannot detect yellow blob (base frame) in camera 1 for target estimation")
                return
            sphere_relative_distance = np.absolute(sphere_position - base_frame)

            # Visualize movement of target
            # y_line = cv2.line(orange_mask, (base_frame[0], base_frame[1]), (sphere_position[0], base_frame[1]), color=(255, 255, 255))
            # z_line = cv2.line(orange_mask, (base_frame[0], base_frame[1]), (base_frame[0], sphere_position[1]), color=(255, 255, 255))
            # center_line = cv2.line(orange_mask, (base_frame[0], base_frame[1]), (sphere_position[0], sphere_position[1]), color=(255, 255, 255))
            # cv2.imshow('Visualization Image 1, Target ZY', orange_mask)
            # cv2.imshow('Original Image 1, Target ZY', self.cv_image1)
            # cv2.waitKey(3)

            # Publish the results
            self.target_position.data[1] = self.pix_to_m_ratio_img1 * sphere_relative_distance[0]
            self.target_position.data[2] = self.pix_to_m_ratio_img1 * (
                        (np.absolute(self.target_history[2] - base_frame[1]) + sphere_relative_distance[1]) / 2)
            # print("Target position: X={0:.2f}, Y={1:.2f}, Z={2:.2f}".format(self.target_position.data[0],
            #                                                                 self.target_position.data[1],
            #                                                                 self.target_position.data[2]), end='\r')
            self.target_position_pub.publish(self.target_position)

        except CvBridgeError as e:
            print(e)

    def image2_callback(self, data):
        try:
            cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")

            orange_mask = cv2.inRange(cv_image2, (75, 100, 125), (90, 180, 220))
            kernel = np.ones((3, 3), np.uint8)
            orange_mask = cv2.erode(orange_mask, kernel, iterations=1)
            orange_mask = cv2.dilate(orange_mask, kernel, iterations=1)
            sphere_position = find_target(orange_mask, vis.sphere_template, self.target_history, False)

            yellow_mask = cv2.inRange(cv_image2, vis.yellow_mask_low, vis.yellow_mask_high)
            base_frame = vis.detect_blob_center(yellow_mask)
            if base_frame is None:
                rospy.logwarn("Cannot detect yellow blob (base frame) in camera 2 for target estimation")
                return
            sphere_relative_distance = np.absolute(sphere_position - base_frame)

            # Visualize movement of target
            # x_line = cv2.line(orange_mask, (base_frame[0], base_frame[1]), (sphere_position[0], base_frame[1]), color=(255, 255, 255))
            # # z_line = cv2.line(orange_mask, (base_frame[0], base_frame[1]), (base_frame[0], sphere_position[1]), color=(255, 255, 255))
            # center_line = cv2.line(orange_mask, (base_frame[0], base_frame[1]), (sphere_position[0], sphere_position[1]), color=(255, 255, 255))
            # cv2.imshow('Visualization Image 2, Target ZX', orange_mask)
            # # cv2.imshow('Visualization Image 2, Yellow Blob ZX', yellow_mask)
            # cv2.waitKey(3)

            # Publish the results
            self.target_position.data[0] = self.pix_to_m_ratio_img2 * sphere_relative_distance[0]
            self.target_position.data[2] = self.pix_to_m_ratio_img2 * (
                        (np.absolute(self.target_history[2] - base_frame[1]) + sphere_relative_distance[1]) / 2)
            self.target_position_pub.publish(self.target_position)
        except CvBridgeError as e:
            print(e)


# call the class
def main():
    TargetEstimator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


# run the code if the node is called
if __name__ == '__main__':
    main()
