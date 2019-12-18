#!/usr/bin/env python2.7
from __future__ import print_function

import numpy as np
import rospy
from std_msgs.msg import Float64, Float64MultiArray

import JointAnglesEstimator


def precomputed_jacobian(joints):
    s1 = np.sin(joints[0])
    c1 = np.cos(joints[0])
    s2 = np.sin(joints[1])
    c2 = np.cos(joints[1])
    s3 = np.sin(joints[2])
    c3 = np.cos(joints[2])
    s4 = np.sin(joints[3])
    c4 = np.cos(joints[3])
    jacobian = np.zeros((3, 4))

    jacobian[0, 0] = 2 * c1 * s2 * c3 * c4 - 2 * s1 * s3 * c4 + 2 * c1 * c2 * s4 + 3 * c1 * s2 * c3 - 3 * s1 * s3
    jacobian[0, 1] = 2 * s1 * c2 * c3 * c4 - 2 * s1 * s2 * s4 + 3 * s1 * c2 * c3
    jacobian[0, 2] = - 2 * s1 * s2 * s3 * c4 + 2 * c1 * c3 * c4 - 3 * s1 * s2 * s3 + 3 * c1 * c3
    jacobian[0, 3] = - 2 * s1 * s2 * c3 * s4 - 2 * c1 * s3 * s4 + 2 * s1 * c2 * c4

    jacobian[1, 0] = 2 * s1 * s2 * c3 * c4 + 2 * c1 * s3 * c4 + 2 * s1 * c2 * s4 + 3 * s1 * s2 * c3 + 3 * c1 * s3
    jacobian[1, 1] = - 2 * c1 * c2 * c3 * c4 + 2 * c1 * s2 * s4 - 3 * c1 * c2 * c3
    jacobian[1, 2] = 2 * c1 * s2 * s3 * c4 + 2 * s1 * c3 * c4 + 3 * c1 * s2 * s3 + 3 * s1 * c3
    jacobian[1, 3] = 2 * c1 * s2 * c3 * s4 - 2 * s1 * s3 * s4 - 2 * c1 * c2 * c4

    jacobian[2, 0] = 0
    jacobian[2, 1] = - 2 * s2 * c3 * c4 - 2 * c2 * s4 - 3 * s2 * c3
    jacobian[2, 2] = - 2 * c2 * s3 * c4 - 3 * c2 * s3
    jacobian[2, 3] = - 2 * c2 * c3 * s4 - 2 * s2 * c4

    return jacobian


class Controller:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named controller
        rospy.init_node('controller', anonymous=True)

        # initialize a subscriber to get position of blobs
        self.blob_sub = rospy.Subscriber("/blobs_pos", Float64MultiArray, self.get_blob_positions_and_move_robot)

        # initialize a subscriber to get position of target
        self.target_sub = rospy.Subscriber("/target_position_estimate", Float64MultiArray, self.get_target_position)

        # initialize a subscriber to obtain real target position (for DEBUGGING only)
        # self.target1_x_sub = rospy.Subscriber("/target/x_position_controller/command", Float64,
        #                                       self.get_target_real_position_x)
        # self.target1_y_sub = rospy.Subscriber("/target/y_position_controller/command", Float64,
        #                                       self.get_target_real_position_y)
        # self.target1_z_sub = rospy.Subscriber("/target/z_position_controller/command", Float64,
        #                                       self.get_target_real_position_z)
        # self.target_real_position = np.array([0.0, 0.0, 7.0])

        # initialize a publisher to publish new joint angles to the robot
        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

        # initialize variables
        self.blob_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 5.0, 0.0, 0.0, 7.0])
        self.target_position = np.array([0.0, 0.0, 7.0])
        self.joint_angles = np.zeros(4)
        self.joint1 = Float64()
        self.joint2 = Float64()
        self.joint3 = Float64()
        self.joint4 = Float64()

        # initialize time
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')

        # initialize error
        self.error = np.array([0.0, 0.0, 0.0], dtype='float64')

        # initialize derivative of error
        self.error_d = np.array([0.0, 0.0, 0.0], dtype='float64')

    def get_blob_positions_and_move_robot(self, blobs):
        if len(blobs.data) == 0:
            rospy.logwarn("Joint angle estimation (and therefore control) cannot take place because blob positions "
                          "received are empty")
            return

        # update the current blob positions
        self.blob_positions = blobs.data

        # print("BL:({0:.2f}, {1:.2f}, {2:.2f}), GR:({3:.2f}, {4:.2f}, {5:.2f}), RE:({6:.2f}, {7:.2f}, {8:.2f}), "
        #       "TAR:({9:.1f}, {10:0.2f}, {11:.2f}), ".format(self.blob_positions[3], self.blob_positions[4],
        #                                                     self.blob_positions[5],
        #                                                     self.blob_positions[6], self.blob_positions[7],
        #                                                     self.blob_positions[8],
        #                                                     self.blob_positions[9], self.blob_positions[10],
        #                                                     self.blob_positions[11],
        #                                                     self.target_position[0], self.target_position[1],
        #                                                     self.target_position[2]))
        # end='\r')

        # calculate the new joint angles using closed-loop control
        new_joint_angles = self.control_closed()

        # move the robot to the new joint angles
        self.move_robot(new_joint_angles)

    # def get_target_real_position_x(self, target_real_x):
    #     self.target_real_position[0] = target_real_x.data
    #
    # def get_target_real_position_y(self, target_real_y):
    #     self.target_real_position[1] = target_real_y.data
    #
    # def get_target_real_position_z(self, target_real_z):
    #     self.target_real_position[2] = target_real_z.data

    def get_target_position(self, target):
        self.target_position = target.data

    def move_robot(self, desired_joint_angles):
        self.joint_angles = desired_joint_angles
        self.joint1.data = desired_joint_angles[0]
        self.joint2.data = desired_joint_angles[1]
        self.joint3.data = desired_joint_angles[2]
        self.joint4.data = desired_joint_angles[3]
        self.robot_joint1_pub.publish(self.joint1)
        self.robot_joint2_pub.publish(self.joint2)
        self.robot_joint3_pub.publish(self.joint3)
        self.robot_joint4_pub.publish(self.joint4)

    # Closed loop control aka feedback control
    def control_closed(self):
        # P gain
        k_p = np.eye(3) * 10

        # D gain
        k_d = np.eye(3) * 0.1

        # estimate time step
        cur_time = np.array([rospy.get_time()])
        time_delta = cur_time - self.time_previous_step
        if time_delta == 0:
            return self.joint_angles
        self.time_previous_step = cur_time

        # robot end-effector position
        pos = np.array([self.blob_positions[9], self.blob_positions[10], self.blob_positions[11]])

        # desired position (target position)
        pos_d = self.target_position
        # DEBUGGING only: gets the real target position
        # pos_d = self.target_real_position

        # estimate derivative of (previous) error for the D (Derivative) part in the PD controller
        self.error_d = ((pos_d - pos) - self.error) / time_delta

        # estimate new error for the P (Proportional) part in the PD controller
        self.error = pos_d - pos

        # Note: Gravity is unaccounted for as per the simulation conditions! The I part in the PID controller is
        # missing.

        # don't estimate the joint angles, just select angles based on assumption that robot moved correctly
        # q_initial = self.joint_angles
        # OR estimate the joint angles according to the position obtained from image
        q_initial = JointAnglesEstimator.estimate_joint_angles_v2(
            np.array([self.blob_positions[6], self.blob_positions[7], self.blob_positions[8]]),
            np.array([self.blob_positions[9], self.blob_positions[10], self.blob_positions[11]])
        )
        if q_initial is None:
            q_initial = self.joint_angles

        # print("THETA 1:{0:.2f}, THETA 2:{1:.2f}, THETA 3:{2:.2f}, THETA 4:{3:.2f}".format(joint_angles[0],
        #                                                                                   joint_angles[1],
        #                                                                                   joint_angles[2],
        #                                                                                   joint_angles[3]), end='\r')

        # calculate the Moore-Penrose psuedo-inverse of Jacobian to obtain angle velocity
        jacobian_inverse = np.linalg.pinv(precomputed_jacobian(q_initial))

        # angular displacement of joints
        # delta_Q = J^(+) * delta_X
        # delta_X  = Kp*error + Kd*error_derivative
        q_delta = np.dot(jacobian_inverse,
                         (np.dot(
                             k_d,
                             self.error_d.transpose()) +
                          np.dot(
                              k_p,
                              self.error.transpose())))

        # control output (angular position of joints)
        return q_initial + (time_delta * q_delta)


# call the class
def main():
    rospy.sleep(7)  # Wait for initialization of environment before moving robot
    Controller()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


# run the code if the node is called
if __name__ == '__main__':
    main()
