# Ball Tracker Robot
## Theodor Amariucai
### I started the project with Sonia Marshall as part of the coursework for INFR09019 Introduction to Vision and Robotics
### Robotic Arm Kinematic GUI (part of MRPT) was very useful in helping me understand the Denavit-Hartenberg parameters

#### BEFORE ANYTHING:
source devel/setup.sh

#### LAUNCH PROJECT:
roslaunch ivr_assignment spawn.launch

#### RUN CODE:
##### Activate temporary python environment with Python 2.7 needed for ROS
source temp-python/bin/activate
##### For basic functionality please run:
rosrun ivr_assignment BlobsEstimator.py \
rosrun ivr_assignment TargetEstimator.py \
rosrun ivr_assignment JointAnglesEstimator.py 
###### then, to get the results of forward kinematics please run:
rosrun ivr_assignment ForwardKinematics.py 
###### and finally, to move the robot please run:
rosrun ivr_assignment Controller.py

#### MOVE ROBOT MANUALLY:
rostopic pub -1 /robot/joint1_position_controller/command std_msgs/Float64 “data: 1.0”

#### PRODUCE GRAPHS FOR TARGET ESTIMATES:
- for each one, run the command
- click on the arrow button to set axes (choose around 50 seconds)
- save as an image
rqt_plot /target_position_estimate/data[0] /target/x_position_controller/command/data
rqt_plot /target_position_estimate/data[1] /target/y_position_controller/command/data
rqt_plot /target_position_estimate/data[2] /target/z_position_controller/command/data

#### PRODUCE GRAPHS FOR CONTROL ACCURACY:
rqt_plot /target_position_estimate/data[0] /blobs_pos/data[9]
rqt_plot /target_position_estimate/data[1] /blobs_pos/data[10]
rqt_plot /target_position_estimate/data[2] /blobs_pos/data[11]

