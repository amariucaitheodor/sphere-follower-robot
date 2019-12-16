#!/usr/bin/env python
from __future__ import print_function

from functools import reduce
import rospy
import numpy as np
from std_msgs.msg import Float64, Float64MultiArray
from scipy.optimize import leastsq


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

        joint_angles = self.estimate_joint_angles()

        if joint_angles is None:
            return

        # print("THETA 1:{0:.2f}, THETA 2:{1:.2f}, THETA 3:{2:.2f}, THETA 4:{3:.2f}".format(joint_angles[0],
        #                                                                                   joint_angles[1],
        #                                                                                   joint_angles[2],
        #                                                                                   joint_angles[3]), end='\r')

        self.joint_angles.data = joint_angles
        self.joint_angles_pub.publish(self.joint_angles)

    # estimate joint angles via an optimization problem knowing joint positions
    # descends in the angle space towards the minimum error, minimizing the error function and finding the thetas
    def estimate_joint_angles(self):
        # all differences are to be minimized
        # i.e. the difference between our yellow to green vector and the yellow to green vector that we know from the
        # FK of A1A2A3 (top three rows of last column)
        # to get theta 4, we also minimize distance between Z of red and FK-obtained Z of red
        def cost_function(x, data):
            s1 = np.sin(x[0])
            c1 = np.cos(x[0])
            s2 = np.sin(x[1])
            c2 = np.cos(x[1])
            s3 = np.sin(x[2])
            c3 = np.cos(x[2])
            s4 = np.sin(x[3])
            c4 = np.cos(x[3])
            return ((3 * s1 * s2 * c3 + 3 * c1 * s3 - data[0]),
                    (-3 * c1 * s2 * c3 - 3 * s1 * s3 - data[1]),
                    (3 * c2 * c3 + 2 - data[2]),
                    (2 * c2 * c3 * c4 - 2 * s2 * s4 + 3 * c2 * c3 + 2 - data[3]))  # minimize distance between

        green = np.array([self.blob_positions[6], self.blob_positions[7], self.blob_positions[8]])
        red = np.array([self.blob_positions[9], self.blob_positions[10], self.blob_positions[11]])

        solution = leastsq(cost_function, np.zeros(4), args=[green[0], green[1], green[2], red[2]])

        # discard the result beyond joints configuration space (basically use the same joint angle as last time)
        if reduce(lambda x, y: x or y > np.pi or y < -np.pi, solution[0], False):
            return None

        return np.round(solution[0], 2)


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
