#!/usr/bin/env python
from __future__ import print_function

import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import Float64MultiArray


class ForwardKinematics:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('forward_kinematics', anonymous=True)
        # initialize a subscriber to get position of blobs
        self.blob_sub = rospy.Subscriber("/blobs_pos", Float64MultiArray, self.callback_update_blob_positions)
        # initialize a publisher to publish the end effector position calculated by FK
        self.end_effector_pub = rospy.Publisher("/end_effector_position", Float64MultiArray, queue_size=10)
        self.end_effector_position = Float64MultiArray()
        self.blob_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # input the joint angles here for which you want to calculate FK
        self.joints = np.array([1.0, 1.0, 1.0, 1.0])

    def callback_update_blob_positions(self, blobs):
        if len(blobs.data) == 0:
            received_blobs = self.blob_positions
        else:
            received_blobs = blobs.data
            self.blob_positions = received_blobs

        # print("Vision\tx: {}, y:{}, z:{}".format(received_blobs[9], received_blobs[10], received_blobs[11]), end='\r')
        end_effector = self.calculate_fk(self.joints)
        print("FK\tx: {0:.2f}, y: {1:.2f}, z: {2:.2f}".format(end_effector[0], end_effector[1], end_effector[2]), end='\r')
        self.end_effector_position.data = end_effector
        self.end_effector_pub.publish(self.end_effector_position)

    def calculate_fk(self, joints):
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
        return end_effector


# call the class
def main(args):
    fk = ForwardKinematics()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
