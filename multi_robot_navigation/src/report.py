#!/usr/bin/env python3
import numpy as np
import os
import math
def calculatedDistance(point1: np.ndarray, point2: np.ndarray):
    '''
        Calculates the Euclidean distance between two points
        ### Parameters
        - point1: the coordinate of first point
        - point2: the coordinate of second point
    '''
    return math.sqrt(np.sum(np.square(point1 - point2)))


evaluate_path = "collision_avoidance/evaluate"
init_pose_path = "collision_avoidance/init_pose"
goal_point_path = "collision_avoidance/goal_point"
num_of_robot = 20
num_of_test = 20
max_speed = 1.0
environment_name = "circle_{}".format(num_of_robot)

data = []
init_pose = np.loadtxt(os.path.join(init_pose_path, environment_name + ".txt"))
init_pose = init_pose[:, 0:-1]
goal_pose = np.loadtxt(os.path.join(goal_point_path, environment_name + ".txt"))
for i in range(num_of_robot):
    robot_data = np.loadtxt(os.path.join(evaluate_path, environment_name, "robot_{}.txt".format(i)))
    data.append(robot_data[2:])

# Calculate the distance standard
standard_distance = 0.0
standard_time = 0.0
for i in range(num_of_robot):
    standard_distance += calculatedDistance(init_pose[i], goal_pose[i])
    standard_time += calculatedDistance(init_pose[i], goal_pose[i]) / max_speed
standard_distance = round(standard_distance / num_of_robot, 2)
standard_time = round(standard_time / num_of_robot, 2)

success = 0.0
average_speed = 0.0
traveled_distance = 0.0
traveled_time = 0.0
for test in range(num_of_test):
    for i in range(num_of_robot):
        success+= data[i][test][0]
        traveled_time += data[i][test][1]
        traveled_distance += data[i][test][2]
        average_speed += data[i][test][3]

print("Success: {}%".format(success/(num_of_robot * num_of_test) * 100))
print("Average speed: {} m/s".format(round(average_speed/(num_of_robot * num_of_test), 3)))
print("Extra distance: {} m".format(round(traveled_distance/(num_of_robot * num_of_test) - standard_distance, 3)))
print("Extra time: {} s".format(round(traveled_time/(num_of_robot * num_of_test) - standard_time, 3)))