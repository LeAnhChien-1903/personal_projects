#!/usr/bin/env python3
import os
import math
import numpy as np
import cv2
world_dir = "collision_avoidance/world"
launch_dir = "collision_avoidance/launch"
goal_dir = "collision_avoidance/goal_point"
pose_dir = "collision_avoidance/init_pose"
map_name = "world_1"

image = cv2.imread("collision_avoidance/maps/{}.png".format(map_name))
scale = 20
map_size = [image.shape[1] * 0.025, image.shape[0]* 0.025]
window_size = [image.shape[1]/2 + 50, image.shape[0]/2 + 50]
num_of_robot = 20
robot_width = 0.38
robot_length = 0.44
robot_height = 0.22
laser_max_range = 6.0
laser_fov = 180
laser_samples = 512
laser_pose = [0.0, 0.0, 0.0, 0.0]

world_file = open(os.path.join(world_dir, "{}_{}.world".format(map_name, num_of_robot)), 'w')

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
    size [{} {}]
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
            map_size[0], map_size[1], window_size[0], window_size[1], scale, 
            robot_length, robot_width, robot_height, laser_pose[0], laser_pose[1], 
            laser_pose[2], laser_pose[3]))

step = 2 * math.pi / num_of_robot
goal_point = []
init_pose = []

# 5 robot
pose = [map_size[0]/2 - 2.0, 2.0, 180]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([-pose[0],  pose[1]])

pose = [map_size[0]/2 - 2.0, -2.0, 180]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([-pose[0],  pose[1]])

pose = [map_size[0]/2 - 4.0, 2.0, 180]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([-pose[0],  pose[1]])

pose = [map_size[0]/2 - 4.0, -2.0, 180]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([-pose[0],  pose[1]])

pose = [map_size[0]/2 - 3, 0.0, 180]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([-pose[0],  pose[1]])

# 5 robot
pose = [-map_size[0]/2 + 2.0, 2.0, 0.0]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([-pose[0], pose[1]])

pose = [-map_size[0]/2 + 2.0, -2.0, 0.0]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([-pose[0], pose[1]])

pose = [-map_size[0]/2 + 4.0, 2.0, 0.0]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([-pose[0],  pose[1]])

pose = [-map_size[0]/2 + 4.0, -2.0, 0.0]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([-pose[0],  pose[1]])

pose = [-map_size[0]/2 + 3, 0.0, 0.0]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([-pose[0],  pose[1]])

# 5 robot
pose = [2.0, map_size[0]/2 - 2.0, -90]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([pose[0],  -pose[1]])

pose = [-2.0, map_size[0]/2 - 2.0, -90]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([pose[0],  -pose[1]])

pose = [2.0, map_size[0]/2 - 4.0, -90]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([pose[0],  -pose[1]])

pose = [-2.0, map_size[0]/2 - 4.0, -90]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([pose[0],  -pose[1]])

pose = [0.0, map_size[0]/2 - 3, -90]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([pose[0],  -pose[1]])

# 5 robot
pose = [2.0, -map_size[0]/2 + 2.0, 90]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([pose[0],  -pose[1]])

pose = [-2.0, -map_size[0]/2 + 2.0, 90]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([pose[0],  -pose[1]])

pose = [2.0, -map_size[0]/2 + 4.0, 90]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([pose[0],  -pose[1]])

pose = [-2.0, -map_size[0]/2 + 4.0, 90]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([pose[0],  -pose[1]])

pose = [0.0, -map_size[0]/2 + 3, 90]
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(pose[0], pose[1], pose[2]))
init_pose.append([pose[0], pose[1], pose[2]*math.pi/180])
goal_point.append([pose[0],  -pose[1]])

world_file.close()

np.savetxt(os.path.join(goal_dir, "{}_{}.txt".format(map_name, num_of_robot)), np.array(goal_point), fmt= '%.5f')
np.savetxt(os.path.join(pose_dir, "{}_{}.txt".format(map_name, num_of_robot)), np.array(init_pose), fmt= '%.5f')

launch_file =open(os.path.join(launch_dir, "world_1_{}.launch".format(num_of_robot)), 'w')
launch_file.write(
"""
<launch>

    <!--  ************** Global Parameters ***************  -->
    <param name="/use_sim_time" value="true"/>
    <arg name="world_file" default="$(find collision_avoidance)/world/{}_{}.world" />
    <node name="stageros" type="stageros" pkg="stage_ros_add_pose_and_crash" args=" $(arg world_file)"/>
""".format(map_name, num_of_robot))
for i in range(num_of_robot):
    launch_file.write('\t<node pkg="collision_avoidance" type="test_policy.py" name="robot_{}" output="screen"/>\n'.format(i))
launch_file.write("</launch>")
launch_file.close()
