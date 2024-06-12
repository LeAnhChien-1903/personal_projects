#!/usr/bin/env python3
import os
import math
import numpy as np
world_dir = "collision_avoidance/world"
launch_dir = "collision_avoidance/launch"
goal_dir = "collision_avoidance/goal_point"
pose_dir = "collision_avoidance/init_pose"
map_name = "empty"
scale = 15
map_size = [30.0, 30.0]
num_of_robot = 20
radius = 13
robot_width = 0.38
robot_length = 0.44
robot_height = 0.22
laser_max_range = 6.0
laser_fov = 180
laser_samples = 512
laser_pose = [0.0, 0.0, 0.0, 0.0]

world_file = open(os.path.join(world_dir, "circle_{}.world".format(num_of_robot)), 'w')

world_file.write(
"""
show_clock 0
show_clock_interval 10000
resolution 0.025
threads 4
speedup 1

define laser ranger
(
    sensor(
        pose [ 0 0 0.1 0 ]
        fov {}
        range [ 0.0 {} ]
        samples {}
    )
    color "random"
    block( 
        points 4
        point[0] [0 0]
        point[1] [0 1]
        point[2] [1 1]
        point[3] [1 0]
        z [0 0.21]
    )
)


define floor model
(
    color "gray30"
    boundary 1

    gui_nose 0
    gui_grid 0
    gui_move 0
    gui_outline 0
    gripper_return 0
    fiducial_return 0
    ranger_return 1
    obstacle_return 1
)

floor
(
    name "{}"
    bitmap "../maps/{}.png"
    size [{} {} 2.00]
    pose [0.000 0.000 0.000 0.000]
)

window
(
    size [500.0 500.0]
    center [0.000000 0.000000] # Camera options 
    rotate [0.000000 0.000000] # Camera options 
    scale {}
    show_data 1
    show_grid 1
    show_trailarrows 1
)
define agent position
(
    # actual size
    size [{} {} {}] # sizes from MobileRobots' web site

    # the pioneer's center of rotation is offset from its center of area
    origin [0 0 0 0]

    # draw a nose on the robot so we can see which way it points
    gui_nose 1

    color "random"
    drive "diff"		 	# Differential steering model.
    obstacle_return 1           	# Can hit things.
    ranger_return 0.5            	# reflects sonar beams
    blob_return 1               	# Seen by blobfinders  
    fiducial_return 1           	# Seen as "1" fiducial finders
    laser
    (
        pose [ {} {} {} {} ] 
    )
)
""".format(laser_fov, laser_max_range, laser_samples, map_name, map_name, 
            map_size[0], map_size[1], scale , robot_length, robot_width, robot_height,
            laser_pose[0], laser_pose[1], laser_pose[2], laser_pose[3]))

step = 2 * math.pi / num_of_robot
goal_point = []
init_pose = []
for i in range(num_of_robot):
    theta = step * i
    world_file.write("agent( pose [{} {} 0.0 {}])\n".format(round(radius*math.cos(theta), 2), round(radius * math.sin(theta), 2), round((-math.pi + theta) * 180 / math.pi, 2)))
    goal_point.append([-round(radius*math.cos(theta), 2), -round(radius * math.sin(theta), 2)])
    init_pose.append([round(radius*math.cos(theta), 2), round(radius * math.sin(theta), 2), (-math.pi + theta)])
world_file.close()

np.savetxt(os.path.join(goal_dir, "circle_{}.txt".format(num_of_robot)), np.array(goal_point), fmt= '%.2f')
np.savetxt(os.path.join(pose_dir, "circle_{}.txt".format(num_of_robot)), np.array(init_pose), fmt= '%.5f')


launch_file =open(os.path.join(launch_dir, "circle_{}.launch".format(num_of_robot)), 'w')
launch_file.write(
"""
<launch>

    <!--  ************** Global Parameters ***************  -->
    <param name="/use_sim_time" value="true"/>
    <arg name="world_file" default="$(find collision_avoidance)/world/circle_{}.world" />
    <node name="stageros" type="stageros" pkg="stage_ros_add_pose_and_crash" args=" $(arg world_file)"/>
""".format(num_of_robot))
for i in range(num_of_robot):
    launch_file.write('\t<node pkg="collision_avoidance" type="circle_test_{}.py" name="robot_{}" output="screen"/>\n'.format(num_of_robot, i))
launch_file.write("</launch>")
launch_file.close()
