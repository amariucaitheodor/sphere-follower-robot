#!/usr/bin/env python
from __future__ import print_function

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
        # initialize a publisher to publish pixel to meter ratios
        self.cam1_ratio_pub = rospy.Publisher("/camera1/pix_to_m_ratio", Float64, queue_size=10)
        self.cam2_ratio_pub = rospy.Publisher("/camera2/pix_to_m_ratio", Float64, queue_size=10)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        self.blobs = Float64MultiArray()
        self.blobs_history = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # update y and z of blobs (z is the average between the two images)
    def image1_callback(self, data):
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Color masks (BGR)
        yellow_mask = cv2.inRange(self.cv_image1, (0, 100, 100), (80, 255, 255))
        blue_mask = cv2.inRange(self.cv_image1, (100, 0, 0), (255, 80, 80))
        green_mask = cv2.inRange(self.cv_image1, (0, 100, 0), (80, 255, 80))
        red_mask = cv2.inRange(self.cv_image1, (0, 0, 100), (80, 80, 255))

        pix_to_m_ratio_img1 = Float64()
        pix_to_m_ratio_img1.data = vis.pixel2meter(yellow_mask, blue_mask)
        self.cam1_ratio_pub.publish(pix_to_m_ratio_img1)
        base_frame = vis.detect_blob_center(yellow_mask)

        green_detected = vis.detect_blob_center(green_mask)
        if green_detected is not None:
            relative_green = base_frame - green_detected
            self.blobs_history[7] = pix_to_m_ratio_img1.data * relative_green[0]
            self.blobs_history[8] = (self.blobs_history[8] + pix_to_m_ratio_img1.data * relative_green[1]) / 2

            # Visualize green blob through camera 1
            # y_line = cv2.line(green_mask, (base_frame[0], base_frame[1]), (int(green_detected[0]), base_frame[1]),
            #                   color=(255, 255, 255))
            # z_line = cv2.line(green_mask, (base_frame[0], base_frame[1]), (base_frame[0], int(green_detected[1])),
            #                   color=(255, 255, 255))
            # center_line = cv2.line(green_mask, (base_frame[0], base_frame[1]),
            #                        (int(green_detected[0]), int(green_detected[1])), color=(255, 255, 255))
            # cv2.imshow('Visualization Image 1, Target ZY, Green Blob', green_mask)
            # cv2.imshow('Original Image 1, Target ZY', self.cv_image1)
            # cv2.waitKey(3)

        red_detected = vis.detect_blob_center(red_mask)
        if red_detected is not None:
            relative_red = base_frame - red_detected
            self.blobs_history[10] = pix_to_m_ratio_img1.data * relative_red[0]
            self.blobs_history[11] = (self.blobs_history[11] + pix_to_m_ratio_img1.data * relative_red[1]) / 2

            # Visualize red blob through camera 1
            # y_line = cv2.line(red_mask, (base_frame[0], base_frame[1]), (int(red_detected[0]), base_frame[1]),
            #                   color=(255, 255, 255))
            # z_line = cv2.line(red_mask, (base_frame[0], base_frame[1]), (base_frame[0], int(red_detected[1])),
            #                   color=(255, 255, 255))
            # center_line = cv2.line(red_mask, (base_frame[0], base_frame[1]),
            #                        (int(red_detected[0]), int(red_detected[1])), color=(255, 255, 255))
            # cv2.imshow('Visualization Image 1, Target ZY, Red Blob', red_mask)
            # cv2.imshow('Original Image 1, Target ZY', self.cv_image1)
            # cv2.imshow('Yellow Mask Image 1, Target ZY', yellow_mask)
            # cv2.waitKey(3)

        self.blobs.data = self.blobs_history
        self.blob_pub.publish(self.blobs)

        print("YE:({0:.1f}, {1:0.2f}, {2:.2f}), BL:({3:.2f}, {4:.2f}, {5:.2f}), GR:({6:.2f}, {7:.2f}, {8:.2f}), "
              "RE:({9:.2f}, {10:.2f}, {11:.2f})".format(self.blobs_history[0], self.blobs_history[1],
                                                        self.blobs_history[2],
                                                        self.blobs_history[3], self.blobs_history[4],
                                                        self.blobs_history[5],
                                                        self.blobs_history[6], self.blobs_history[7],
                                                        self.blobs_history[8],
                                                        self.blobs_history[9], self.blobs_history[10],
                                                        self.blobs_history[11]),
              end='\r')

    # update x and z of blobs (z is the average between the two images)
    def image2_callback(self, data):
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Color masks (BGR)
        yellow_mask = cv2.inRange(self.cv_image2, (0, 100, 100), (80, 255, 255))
        blue_mask = cv2.inRange(self.cv_image2, (100, 0, 0), (255, 80, 80))
        green_mask = cv2.inRange(self.cv_image2, (0, 100, 0), (80, 255, 80))
        red_mask = cv2.inRange(self.cv_image2, (0, 0, 100), (80, 80, 255))

        # cv2.imshow('Visualization Image 2, Target ZX, Yellow Blob', yellow_mask)
        # cv2.imshow('Visualization Image 2, Target ZX, Blue Blob', blue_mask)            # cv2.imshow('Visualization Image 2, Target ZX, Green Blob', green_mask)
        # cv2.waitKey(3)
        pix_to_m_ratio_img2 = Float64()
        pix_to_m_ratio_img2.data = vis.pixel2meter(yellow_mask, blue_mask)
        self.cam2_ratio_pub.publish(pix_to_m_ratio_img2)
        base_frame = vis.detect_blob_center(yellow_mask)

        green_detected = vis.detect_blob_center(green_mask)
        if green_detected is not None:
            relative_green = base_frame - green_detected
            self.blobs_history[6] = pix_to_m_ratio_img2.data * relative_green[0]
            self.blobs_history[8] = (self.blobs_history[8] + pix_to_m_ratio_img2.data * relative_green[1]) / 2

            # Visualize green blob through camera 2
            # x_line = cv2.line(green_mask, (base_frame[0], base_frame[1]), (int(green_detected[0]), base_frame[1]),
            #                   color=(255, 255, 255))
            # z_line = cv2.line(green_mask, (base_frame[0], base_frame[1]), (base_frame[0], int(green_detected[1])),
            #                   color=(255, 255, 255))
            # center_line = cv2.line(green_mask, (base_frame[0], base_frame[1]),
            #                        (int(green_detected[0]), int(green_detected[1])), color=(255, 255, 255))
            # cv2.imshow('Visualization Image 2, Target ZX, Green Blob', green_mask)
            # cv2.imshow('Original Image 2, Target ZX', self.cv_image2)
            # cv2.waitKey(3)

        red_detected = vis.detect_blob_center(red_mask)
        if red_detected is not None:
            relative_red = base_frame - red_detected
            self.blobs_history[9] = pix_to_m_ratio_img2.data * relative_red[0]
            self.blobs_history[11] = (self.blobs_history[11] + pix_to_m_ratio_img2.data * relative_red[1]) / 2

            # Visualize red blob through camera 2
            # x_line = cv2.line(red_mask, (base_frame[0], base_frame[1]), (int(red_detected[0]), base_frame[1]),
            #                   color=(255, 255, 255))
            # z_line = cv2.line(red_mask, (base_frame[0], base_frame[1]), (base_frame[0], int(red_detected[1])),
            #                   color=(255, 255, 255))
            # center_line = cv2.line(red_mask, (base_frame[0], base_frame[1]),
            #                        (int(red_detected[0]), int(red_detected[1])), color=(255, 255, 255))
            # cv2.imshow('Visualization Image 2, Target ZX, Red Blob', red_mask)
            # cv2.imshow('Original Image 2, Target ZX', self.cv_image2)
            # cv2.waitKey(3)

        self.blobs.data = self.blobs_history
        self.blob_pub.publish(self.blobs)


# call the class
def main():
    BlobsEstimator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


# run the code if the node is called
if __name__ == '__main__':
    main()
