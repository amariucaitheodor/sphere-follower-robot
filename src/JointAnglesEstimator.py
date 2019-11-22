#!/usr/bin/env python
from __future__ import print_function

import sys
import rospy
import numpy as np
from std_msgs.msg import Float64, Float64MultiArray
from scipy.optimize import least_squares


# Defined elementary rotations
def rot_around_x(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])


def rot_around_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])


def rot_around_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


class JointAnglesEstimator:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('joint_angles_estimation', anonymous=True)
        # initialize a subscriber to get position of blobs
        self.blob_sub = rospy.Subscriber("/blobs_pos", Float64MultiArray, self.callback_estimate_angles)
        self.blob_positions = np.array([0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0])
        # initialize a publisher to publish measured joint angles
        self.joint_angles_pub = rospy.Publisher("/joint_angles", Float64MultiArray, queue_size=10)
        self.joint_angles = Float64MultiArray()

    # Receive target data from blob estimator, calculate joint angles and publish them
    def callback_estimate_angles(self, blobs):
        if len(blobs.data) != 0:
            self.blob_positions = blobs.data

        joint_angles = self.measure_angles()

        print("THETA 1:{0:.2f}, THETA 2:{1:.2f}, THETA 3:{2:.2f}, THETA 4:{3:.2f}".format(joint_angles[0],
                                                                                          joint_angles[1],
                                                                                          joint_angles[2],
                                                                                          joint_angles[3]), end='\r')

        self.joint_angles.data = joint_angles
        self.joint_angles_pub.publish(self.joint_angles)

    def error_fct(self, theta):
        s1 = np.sin(theta[0])
        c1 = np.cos(theta[0])
        s2 = np.sin(theta[1])
        c2 = np.cos(theta[1])
        s3 = np.sin(theta[2])
        c3 = np.cos(theta[2])

        blue = np.array([self.blob_positions[3], self.blob_positions[4], self.blob_positions[5]])
        green = np.array([self.blob_positions[6], self.blob_positions[7], self.blob_positions[8]])

        blue_to_green_normalized = green - blue
        # print("blue: {},  green: {}, blue_to_green: {}".format(blue, green, blue_to_green), end='\r')
        blue_to_green_normalized = blue_to_green_normalized / np.linalg.norm(blue_to_green_normalized)

        # return error which is to be minimized
        # i.e. the difference between our (blue to green vector) and the (blue to green vector) that we are trying to obtain from the FK of A1A2A3
        return abs(-c1 * s3 + s1 * s2 * c3 - blue_to_green_normalized[0]) + abs(
            -s1 * s3 - c1 * s2 * c3 - blue_to_green_normalized[1]) + abs(
            c2 * c3 - blue_to_green_normalized[2])

    def measure_angles(self):
        # Descends in the angle space towards the minimum error, minimizing the error function and finding theta 1, 2 and 3
        measured_angles = least_squares(self.error_fct, np.array([0.5, 0.5, 0.5]), bounds=(0, 1))
        measured_angles = measured_angles.x
        theta1 = measured_angles[0]
        theta2 = measured_angles[1]
        theta3 = measured_angles[2]

        green = np.array([self.blob_positions[6], self.blob_positions[7], self.blob_positions[8]])
        red = np.array([self.blob_positions[9], self.blob_positions[10], self.blob_positions[11]])

        green_to_red_normalized = red - green
        green_to_red_normalized = green_to_red_normalized / np.linalg.norm(green_to_red_normalized)

        # Perform inverse rotations to obtain x3 (gr to red normalized) in reference frame
        green_to_red_normalized_in_ref_frame = rot_around_y(-theta3).dot(rot_around_x(-theta2).dot(
            rot_around_z(-theta1).dot(green_to_red_normalized)))

        # Theta 4 rotates around z, therefore we calculate the arctangent w.r.t the Y and X displacement
        # Consult robot configuration for a visualization of this
        y_displacement = green_to_red_normalized_in_ref_frame[1]
        x_displacement = green_to_red_normalized_in_ref_frame[0]
        theta4 = np.arctan2(y_displacement, x_displacement)
        measured_angles = np.append(measured_angles, theta4)
        return measured_angles


# call the class
def main(args):
    ja = JointAnglesEstimator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
