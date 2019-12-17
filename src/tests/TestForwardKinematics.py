import numpy as np
import sys

sys.path.append('../')
import ForwardKinematics

np.set_printoptions(suppress=True)
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
    result1 = ForwardKinematics.automatic_forward_kinematics(test_joints_array[i])
    result2 = ForwardKinematics.manual_forward_kinematics(test_joints_array[i])

    if np.allclose(result1, result2) is False:
        print("Automatic FK is different from Manual FK in the following case:")
        print("Joint Angles: {}. Automatic FK: {}. Manual FK: {}.".format(
            test_joints_array[i],
            ForwardKinematics.automatic_forward_kinematics(test_joints_array[i]),
            ForwardKinematics.manual_forward_kinematics(test_joints_array[i])))