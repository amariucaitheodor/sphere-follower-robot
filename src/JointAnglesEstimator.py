#!/usr/bin/env python
from __future__ import print_function

import rospy
import numpy as np
from std_msgs.msg import Float64, Float64MultiArray
from scipy.optimize import least_squares, leastsq


# estimate joint angles via an optimization problem knowing joint positions
# descends in the angle space towards the minimum error, minimizing the error function and finding the thetas
def estimate_joint_angles_v1(green_blob, red_blob):
    # all differences are to be minimized
    # i.e. the difference between our yellow to green vector and the yellow to green vector that we know from the
    # FK of A1A2A3 (top three rows of last column) will give us THETA 1, 2, 3
    # to get THETA 4, we also minimize distance between Z of red and FK-obtained Z of red
    def cost_function(x):
        s1 = np.sin(x[0])  # We don't add (np.pi / 2) here because FK was manually computed with the addition
        c1 = np.cos(x[0])  # We don't add (np.pi / 2) here because FK was manually computed with the addition
        s2 = np.sin(x[1])  # We don't add (np.pi / 2) here because FK was manually computed with the addition
        c2 = np.cos(x[1])  # We don't add (np.pi / 2) here because FK was manually computed with the addition
        s3 = np.sin(x[2])
        c3 = np.cos(x[2])
        s4 = np.sin(x[3])
        c4 = np.cos(x[3])
        return np.abs(3 * s1 * s2 * c3 + 3 * c1 * s3 - green_blob[0]
                      - 3 * c1 * s2 * c3 - 3 * s1 * s3 - green_blob[1] +
                      3 * c2 * c3 + 2 - green_blob[2] +
                      2 * c2 * c3 * c4 - 2 * s2 * s4 + 3 * c2 * c3 + 2 - red_blob[2])

    solution = least_squares(
        cost_function,
        # np.array([np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2]),  # initial joint angle guess as recommended by lecturer
        np.zeros(4),
        jac="3-point",  # as suggested by classmate
        loss="cauchy",
        bounds=(-np.pi / 2, np.pi / 2)  # joint angle bounds as recommended by lecturer
    )

    return np.round(solution.x, 2)


# estimate the joint angles by the position of joints
# descends in the angle space towards the minimum error, minimizing the error function and finding the thetas
def estimate_joint_angles_v2(green_blob, red_blob):
    def cost_function(x, data):
        s1 = np.sin(x[0])  # We don't add (np.pi / 2) here because FK was manually computed with the addition
        c1 = np.cos(x[0])  # We don't add (np.pi / 2) here because FK was manually computed with the addition
        s2 = np.sin(x[1])  # We don't add (np.pi / 2) here because FK was manually computed with the addition
        c2 = np.cos(x[1])  # We don't add (np.pi / 2) here because FK was manually computed with the addition
        s3 = np.sin(x[2])
        c3 = np.cos(x[2])
        s4 = np.sin(x[3])
        c4 = np.cos(x[3])
        return ((3 * s1 * s2 * c3 + 3 * c1 * s3 - data[0]),
                (-3 * c1 * s2 * c3 - 3 * s1 * s3 - data[1]),
                (3 * c2 * c3 + 2 - data[2]),
                (2 * (c2 * c3 * c4 - s2 * s4) + data[2] - data[3]))

    solution = leastsq(cost_function, np.zeros(4), args=[green_blob[0], green_blob[1], green_blob[2], red_blob[2]])

    # discard the result beyond joints config space and using the same joint angle as last time
    if reduce(lambda x, y: x or y > np.pi or y < -np.pi, solution[0], False):
        return None

    return np.round(solution[0], 2)


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
        if len(blobs.data) == 0:
            rospy.logwarn("Joint angle estimation cannot take place because blob positions received are empty")
            return

        self.blob_positions = blobs.data

        joint_angles = estimate_joint_angles_v2(
            np.array([self.blob_positions[6], self.blob_positions[7], self.blob_positions[8]]),
            np.array([self.blob_positions[9], self.blob_positions[10], self.blob_positions[11]])
        )

        if joint_angles is None:
            return

        # print("THETA 1:{0:.2f}, THETA 2:{1:.2f}, THETA 3:{2:.2f}, THETA 4:{3:.2f}".format(joint_angles[0],
        #                                                                                   joint_angles[1],
        #                                                                                   joint_angles[2],
        #                                                                                   joint_angles[3]), end='\r')

        self.joint_angles.data = joint_angles
        self.joint_angles_pub.publish(self.joint_angles)


# call the class
def main():
    JointAnglesEstimator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


# run the code if the node is called
if __name__ == '__main__':
    main()
