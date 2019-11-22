#!/usr/bin/env python
from __future__ import print_function

import sys
import cv2
import rospy
import numpy as np
import visionlib as vis
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError


class BlobsEstimator:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('blob_estimation', anonymous=True)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.image1_callback)
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.image2_callback)
        # initialize a publisher to publish position of blobs
        self.blob_pub = rospy.Publisher("/blobs_pos", Float64MultiArray, queue_size=10)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        self.blobs = Float64MultiArray()
        self.blobs_history = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def image1_callback(self, data):
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Color masks (BGR)
        # blue_mask = cv2.inRange(self.cv_image1, (100, 0, 0), (255, 80, 80))
        green_mask = cv2.inRange(self.cv_image1, (0, 100, 0), (80, 255, 80))
        red_mask = cv2.inRange(self.cv_image1, (0, 0, 100), (80, 80, 255))

        # update y and z of blobs
        if len(self.blobs.data) == 0:
            new_blobs = self.blobs_history
        else:
            new_blobs = self.blobs.data
            self.blobs_history = new_blobs

        base_frame = vis.yellow_blob_center_img1
        green_detected = vis.detect_blob_center(green_mask)
        relative_green = base_frame - green_detected
        new_blobs[7] = vis.to_meters_ratio_img1 * relative_green[0]
        new_blobs[8] = vis.to_meters_ratio_img1 * relative_green[1]

        red_detected = vis.detect_blob_center(red_mask)
        relative_red = base_frame - red_detected
        new_blobs[10] = vis.to_meters_ratio_img1 * relative_red[0]
        new_blobs[11] = vis.to_meters_ratio_img1 * relative_red[1]

        self.blobs.data = new_blobs
        self.blob_pub.publish(self.blobs)

        # Visualize green blob
        # y_line = cv2.line(green_mask, (base_frame[0], base_frame[1]), (green_detected[0], base_frame[1]), color=(255, 255, 255))
        # z_line = cv2.line(green_mask, (base_frame[0], base_frame[1]), (base_frame[0], green_detected[1]), color=(255, 255, 255))
        # center_line = cv2.line(green_mask, (base_frame[0], base_frame[1]), (green_detected[0], green_detected[1]), color=(255, 255, 255))
        # cv2.imshow('Visualization Image 1, Target ZY, Green Blob', green_mask)
        # # cv2.imshow('Original Image 1, Target ZY', self.cv_image1)
        # cv2.waitKey(3)

        # Visualize red blob
        # y_line = cv2.line(red_mask, (base_frame[0], base_frame[1]), (red_detected[0], base_frame[1]), color=(255, 255, 255))
        # z_line = cv2.line(red_mask, (base_frame[0], base_frame[1]), (base_frame[0], red_detected[1]), color=(255, 255, 255))
        # center_line = cv2.line(red_mask, (base_frame[0], base_frame[1]), (red_detected[0], red_detected[1]), color=(255, 255, 255))
        # cv2.imshow('Visualization Image 1, Target ZY, Red Blob', red_mask)
        # # cv2.imshow('Original Image 1, Target ZY', self.cv_image1)
        # cv2.waitKey(3)

        print("YE:({0:.1f}, {1:0.2f}, {2:.2f}), BL:({3:.2f}, {4:.2f}, {5:.2f}), GR:({6:.2f}, {7:.2f}, {8:.2f}), "
              "RE:({9:.2f}, {10:.2f}, {11:.2f})".format(new_blobs[0], new_blobs[1], new_blobs[2],
                                                        new_blobs[3], new_blobs[4], new_blobs[5],
                                                        new_blobs[6], new_blobs[7], new_blobs[8],
                                                        new_blobs[9], new_blobs[10], new_blobs[11]),
              end='\r')

    def image2_callback(self, data):
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Color masks (BGR)
        # blue_mask = cv2.inRange(self.cv_image1, (100, 0, 0), (255, 80, 80))
        green_mask = cv2.inRange(self.cv_image2, (0, 100, 0), (80, 255, 80))
        red_mask = cv2.inRange(self.cv_image2, (0, 0, 100), (80, 80, 255))

        # update x of blobs
        # get the blob positions relative to the blue blob (blue at (0,0,0))
        if len(self.blobs.data) == 0:
            new_blobs = self.blobs_history
        else:
            new_blobs = self.blobs.data
            self.blobs_history = new_blobs

        base_frame = vis.yellow_blob_center_img2
        green_detected = vis.detect_blob_center(green_mask)
        relative_green = base_frame - green_detected
        new_blobs[6] = vis.to_meters_ratio_img2 * relative_green[0]

        red_detected = vis.detect_blob_center(red_mask)
        relative_red = base_frame - red_detected
        new_blobs[9] = vis.to_meters_ratio_img2 * relative_red[0]

        # Visualize green blob
        # x_line = cv2.line(green_mask, (base_frame[0], base_frame[1]), (green_detected[0], base_frame[1]), color=(255, 255, 255))
        # z_line = cv2.line(green_mask, (base_frame[0], base_frame[1]), (base_frame[0], green_detected[1]), color=(255, 255, 255))
        # center_line = cv2.line(green_mask, (base_frame[0], base_frame[1]), (green_detected[0], green_detected[1]), color=(255, 255, 255))
        # cv2.imshow('Visualization Image 1, Target ZX, Green Blob', green_mask)
        # # cv2.imshow('Original Image 1, Target ZY', self.cv_image2)
        # cv2.waitKey(3)

        # Visualize red blob
        # x_line = cv2.line(red_mask, (base_frame[0], base_frame[1]), (red_detected[0], base_frame[1]), color=(255, 255, 255))
        # z_line = cv2.line(red_mask, (base_frame[0], base_frame[1]), (base_frame[0], red_detected[1]), color=(255, 255, 255))
        # center_line = cv2.line(red_mask, (base_frame[0], base_frame[1]), (red_detected[0], red_detected[1]), color=(255, 255, 255))
        # cv2.imshow('Visualization Image 1, Target ZY, Red Blob', red_mask)
        # # cv2.imshow('Original Image 1, Target ZX', self.cv_image2)
        # cv2.waitKey(3)

        self.blobs.data = new_blobs
        self.blob_pub.publish(self.blobs)


# call the class
def main(args):
    be = BlobsEstimator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
