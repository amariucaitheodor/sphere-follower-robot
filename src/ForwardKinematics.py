#!/usr/bin/env python2.7
from __future__ import print_function

import rospy
import cv2
import numpy as np
from std_msgs.msg import Float64MultiArray


# calculate a homogeneous transformation using DH parameters
def transform_matrix(alpha, a, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])


# calculate the transformation matrix T40 which takes a coordinate in the end effector frame to the base frame
# this is also documented in the Details folder of my project
def automatic_forward_kinematics(joints):
    t10 = transform_matrix(np.pi / 2, 0, 2, joints[0] + np.pi / 2)  # we add pi/2 to make 0, 0, 0, 0 an
    # upright configuration
    t21 = transform_matrix(np.pi / 2, 0, 0, joints[1] + np.pi / 2)  # we add pi/2 to make 0, 0, 0, 0 an
    # upright configuration
    t32 = transform_matrix(-np.pi / 2, 3, 0, joints[2])
    t43 = transform_matrix(0, 2, 0, joints[3])
    end_effector = t10.dot(t21).dot(t32).dot(t43)[0:3, 3]  # first three rows of last column of T40
    return end_effector


def manual_forward_kinematics(joints):
    s1 = np.sin(joints[0])
    c1 = np.cos(joints[0])
    s2 = np.sin(joints[1])
    c2 = np.cos(joints[1])
    s3 = np.sin(joints[2])
    c3 = np.cos(joints[2])
    s4 = np.sin(joints[3])
    c4 = np.cos(joints[3])
    x_e = 2 * s1 * s2 * c3 * c4 + 2 * c1 * s3 * c4 + 2 * s1 * c2 * s4 + 3 * s1 * s2 * c3 + 3 * c1 * s3
    y_e = - 2 * c1 * s2 * c3 * c4 + 2 * s1 * s3 * c4 - 2 * c1 * c2 * s4 - 3 * c1 * s2 * c3 + 3 * s1 * s3
    z_e = 2 * c2 * c3 * c4 - 2 * s2 * s4 + 3 * c2 * c3 + 2
    end_effector = np.array([x_e, y_e, z_e])
    return end_effector  # still first three rows of last column of T40


class ForwardKinematics:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('forward_kinematics', anonymous=True)
        # initialize a subscriber to get position of blobs
        self.blob_sub = rospy.Subscriber("/blobs_pos", Float64MultiArray, self.get_blob_positions_and_publish_fk)
        # initialize a subscriber to get measured joint angles
        self.blob_sub = rospy.Subscriber("/joint_angles", Float64MultiArray, self.get_joint_angles)
        # initialize a publisher to publish the end effector position calculated by FK
        self.end_effector_pub = rospy.Publisher("/end_effector_position", Float64MultiArray, queue_size=10)
        self.end_effector_position = Float64MultiArray()
        self.blob_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # initialize the joint angles here for which you want to calculate FK
        self.joints = np.ones(4)

    def get_blob_positions_and_publish_fk(self, blobs):
        self.blob_positions = blobs.data

        # print("Vision\tx: {}, y:{}, z:{}".format(self.blob_positions[9], self.blob_positions[10],
        # self.blob_positions[11]), end='\r')
        end_effector = automatic_forward_kinematics(self.joints)
        # print("FK\tx: {0:.2f}, y: {1:.2f}, z: {2:.2f}".format(end_effector[0], end_effector[1], end_effector[2]),
        #       end='\r')
        self.end_effector_position.data = end_effector
        self.end_effector_pub.publish(self.end_effector_position)

    def get_joint_angles(self, joints):
        self.joints = joints.data


# call the class
def main():
    ForwardKinematics()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main()
