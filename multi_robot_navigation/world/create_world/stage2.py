#!/usr/bin/env python3
import os
import math
import numpy as np
world_dir = "collision_avoidance/world"
launch_dir = "collision_avoidance/launch"
goal_dir = "collision_avoidance/goal_point"
pose_dir = "collision_avoidance/init_pose"
map_name = "stage2"
scale = 10
map_size = [100.0, 80.0]
window_size = [1050.0, 850.0]
num_of_robot = 20
robot_width = 0.38
robot_length = 0.44
robot_height = 0.22
laser_max_range = 6.0
laser_fov = 180
laser_samples = 512
laser_pose = [0.0, 0.0, 0.0, 0.0]

goal_point = [[-18.0, 11.5], [-18.0, 9.5], [-7.0, 11.5], [-7.0, 9.5], [-12.5, 4.0], [-12.5, 17.0],
                [-2.0, 3.0], [0.0, 3.0], [3.0, 3.0], [5.0, 3.0], [10.0, 10.0], [12.0, 10.0],
                [14.0, 10.0], [16.0, 10.0], [18.0, 10.0], [3.5, -2.5], [5.5, -2.5], [-2.5, -2.5],
                [-0.5, -2.5], [-2.5, -5.5], [-0.5, -5.5], [1.5, -5.5], [3.5, -5.5], [5.5, -5.5],
                [-18.0, -10.0], [-16.85, -13.53], [-13.85, -15.71], [-10.15, -15.71], [-7.15, -13.53], [-6.00, -10.00],
                [-7.15, -6.47], [-10.15, -4.29], [-13.85, -4.29], [-16.85, -6.47]]

init_pose = [[-7.00, 11.50, np.pi], [-7.00, 9.50, np.pi], [-18.00, 11.50, 0.00], [-18.00, 9.50, 0.00],
            [-12.50, 17.00, np.pi*3/2], [-12.50, 4.00, np.pi/2], [-2.00, 16.00, -np.pi/2], [0.00, 16.00, -np.pi/2],
            [3.00, 16.00, -np.pi/2], [5.00, 16.00, -np.pi/2], [10.00, 4.00, np.pi/2], [12.00, 4.00, np.pi/2],
            [14.00, 4.00, np.pi/2], [16.00, 4.00, np.pi/2], [18.00, 4.00, np.pi/2], [-2.5, -2.5, 0.00],
            [-0.5, -2.5, 0.00], [3.5, -2.5, np.pi], [5.5, -2.5, np.pi], [-2.5, -18.5, np.pi/2],
            [-0.5, -18.5, np.pi/2], [1.5, -18.5, np.pi/2], [3.5, -18.5, np.pi/2], [5.5, -18.5, np.pi/2],
            [-6.00, -10.00, np.pi], [-7.15, -6.47, np.pi*6/5], [-10.15, -4.29, np.pi*7/5], [-13.85, -4.29, np.pi*8/5],
            [-16.85, -6.47, np.pi*9/5], [-18.00, -10.00, np.pi*2], [-16.85, -13.53, np.pi*11/5], [-13.85, -15.71, np.pi*12/5],
            [-10.15, -15.71, np.pi*13/5], [-7.15, -13.53, np.pi*14/5], [10.00, -17.00, np.pi/2], [12.00, -17.00, np.pi/2],
            [14.00, -17.00, np.pi/2], [16.00, -17.00, np.pi/2], [18.00, -17.00, np.pi/2], [10.00, -2.00, -np.pi/2],
            [12.00, -2.00, -np.pi/2], [14.00, -2.00, -np.pi/2], [16.00, -2.00, -np.pi/2], [18.00, -2.00, -np.pi/2]]

np.savetxt(os.path.join(goal_dir, "{}.txt".format(map_name)), np.array(goal_point), fmt= '%.5f')
np.savetxt(os.path.join(pose_dir, "{}.txt".format(map_name)), np.array(init_pose), fmt= '%.5f')

launch_file =open(os.path.join(launch_dir, "{}.launch".format(map_name)), 'w')
launch_file.write(
"""
<launch>

    <!--  ************** Global Parameters ***************  -->
    <param name="/use_sim_time" value="true"/>
    <arg name="world_file" default="$(find collision_avoidance)/world/{}.world" />
    <node name="stageros" type="stageros" pkg="stage_ros_add_pose_and_crash" args=" $(arg world_file)"/>
""".format(map_name))
for i in range(24):
    launch_file.write('\t<node pkg="collision_avoidance" type="test_policy.py" name="robot_{}" output="screen"/>\n'.format(i))
launch_file.write("</launch>")
launch_file.close()
