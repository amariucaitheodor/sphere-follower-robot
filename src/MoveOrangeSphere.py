#!/usr/bin/env python2.7

import rospy
import numpy as np
from std_msgs.msg import Float64


# Publish data
def move():
    rospy.init_node('target_pos_cmd', anonymous=True)
    rate = rospy.Rate(30)  # 30hz
    # initialize a publisher to send joints' angular position to the robot
    target1_x_pub = rospy.Publisher("/target/x_position_controller/command", Float64, queue_size=10)
    target1_y_pub = rospy.Publisher("/target/y_position_controller/command", Float64, queue_size=10)
    target1_z_pub = rospy.Publisher("/target/z_position_controller/command", Float64, queue_size=10)
    target2_x_pub = rospy.Publisher("/target2/x2_position_controller/command", Float64, queue_size=10)
    target2_y_pub = rospy.Publisher("/target2/y2_position_controller/command", Float64, queue_size=10)
    target2_z_pub = rospy.Publisher("/target2/z2_position_controller/command", Float64, queue_size=10)
    t0 = rospy.get_time()
    while not rospy.is_shutdown():
        cur_time = np.array([rospy.get_time()]) - t0
        # y_d = float(6 + np.absolute(1.5* np.sin(cur_time * np.pi/100)))
        x_d = 1.5 * np.cos(cur_time * np.pi / 5)
        y_d = 1.5 * np.sin(cur_time * np.pi / 5)
        z_d = 1 * np.sin(cur_time * np.pi / 5)

        target_x = Float64()
        target_y = Float64()
        target_z = Float64()
        target_x.data = + 2 + x_d
        target_y.data = + 2.5 + y_d
        target_z.data = 5.5 + z_d
        target1_x_pub.publish(target_x)
        target1_y_pub.publish(target_y)
        target1_z_pub.publish(target_z)

        x_d = 2 * np.cos(cur_time * np.pi / 5)
        y_d = 2 * np.sin(cur_time * np.pi / 5)

        target2_x = Float64()
        target2_y = Float64()
        target2_z = Float64()
        target2_x.data = x_d
        target2_y.data = y_d
        target2_z.data = 5

        target2_x_pub.publish(target2_x)
        target2_y_pub.publish(target2_y)
        target2_z_pub.publish(target2_z)
        rate.sleep()


# run the code if the node is called
if __name__ == '__main__':
    try:
        move()
    except rospy.ROSInterruptException:
        pass
