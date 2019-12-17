import numpy as np
import sys

sys.path.append('../')
import JointAnglesEstimator

np.set_printoptions(suppress=True)


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
def green_blob_fk(joints):
    t10 = transform_matrix(np.pi / 2, 0, 2, joints[0] + np.pi / 2)  # we add pi/2 to make 0, 0, 0, 0 an
    # upright configuration
    t21 = transform_matrix(np.pi / 2, 0, 0, joints[1] + np.pi / 2)  # we add pi/2 to make 0, 0, 0, 0 an
    # upright configuration
    t32 = transform_matrix(-np.pi / 2, 3, 0, joints[2])
    return t10.dot(t21).dot(t32)[0:3, 3]


# calculate the transformation matrix T40 which takes a coordinate in the end effector frame to the base frame
# this is also documented in the Details folder of my project
def red_blob_fk(joints):
    t10 = transform_matrix(np.pi / 2, 0, 2, joints[0] + np.pi / 2)  # we add pi/2 to make 0, 0, 0, 0 an
    # upright configuration
    t21 = transform_matrix(np.pi / 2, 0, 0, joints[1] + np.pi / 2)  # we add pi/2 to make 0, 0, 0, 0 an
    # upright configuration
    t32 = transform_matrix(-np.pi / 2, 3, 0, joints[2])
    t43 = transform_matrix(0, 2, 0, joints[3])
    return t10.dot(t21).dot(t32).dot(t43)[0:3, 3]


test_joints_array = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4],
    [np.pi / 8, np.pi / 2, np.pi / 3, np.pi / 3],
    [np.pi / 3, np.pi / 5, np.pi / 6, np.pi / 7],
    [-np.pi / 4, np.pi / 4, -np.pi / 4, np.pi / 6],
    [-np.pi / 9, -np.pi / 8, -np.pi / 5, np.pi / 3],
    [-np.pi / 10, np.pi / 6, np.pi / 6, np.pi / 2],
    [np.pi / 4, np.pi / 7, -np.pi / 5, -np.pi / 7],
    [np.pi / 5, -np.pi / 4, -np.pi / 8, np.pi / 8]
])

for i in range(test_joints_array.shape[0]):
    true_angles = test_joints_array[i]
    true_angles = np.round(true_angles, 2)

    green_blob = green_blob_fk(true_angles)
    red_blob = red_blob_fk(true_angles)
    estimated_angles = JointAnglesEstimator.estimate_joint_angles(green_blob, red_blob)

    if np.allclose(estimated_angles, true_angles, atol=np.pi/4):
        print("Estimated joint angles fairly (?) accurately in this case: {}. True Angles were: {}.".format(
            estimated_angles,
            true_angles)
        )
