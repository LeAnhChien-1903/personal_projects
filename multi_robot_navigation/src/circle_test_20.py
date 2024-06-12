#!/usr/bin/env python3

import rospy
from collision_avoidance.agent import Agent
from torch.autograd import Variable
import torch
import numpy as np
import math
from collision_avoidance.network import CNNPolicy
import os
def calculatedDistance(point1: np.ndarray, point2: np.ndarray):
    '''
        Calculates the Euclidean distance between two points
        ### Parameters
        - point1: the coordinate of first point
        - point2: the coordinate of second point
    '''
    return math.sqrt(np.sum(np.square(point1 - point2)))


evaluate_path = "/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/collision_avoidance/evaluate"
environment_name = "circle_20"
rospy.init_node("deep_rl_node", anonymous=False)
index = int(rospy.get_name().split("_")[1])
controller = Agent(512, index, environment_name)

if not os.path.exists(os.path.join(evaluate_path, environment_name, "robot_{}.txt".format(index))):
    np.savetxt(os.path.join(evaluate_path, environment_name, "robot_{}.txt".format(index)), np.zeros((2, 4)), fmt="%.2f")

action_bound = [[0, -1], [1, 1]]


policy_path = "/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/collision_avoidance/parameters"
# policy = MLPPolicy(obs_size, act_size)
policy = CNNPolicy(frames=3, action_space=2)
policy.cuda()

file = policy_path + '/stage2.pth'
print ('############Loading Model###########')
state_dict = torch.load(file)
policy.load_state_dict(state_dict)
action_bound = [[0, -1], [1, 1]]
rate = rospy.Rate(10)



traveled_distance = 0.0
traveled_time = 0.0
average_speed = 0.0
collision = False
success = False
prev_pos = np.array(controller.state_GT[0:-1])
for i in range(500):
    ex = controller.goal_point[0] - controller.state_GT[0]
    ey = controller.goal_point[1] - controller.state_GT[1]
    if math.hypot(ex, ey) < 0.1:
        controller.control_vel([[0.0, 0.0]])
        if collision: success = False
        else: success = True
        rate.sleep()
        break
    else:
        laser = np.asarray(controller.state_obs[0])
        goal = np.asarray(controller.state_obs[1])
        speed = np.asarray(controller.state_obs[2])
        laser_obs = Variable(torch.from_numpy(laser.reshape(1, 3, controller.beam_num))).float().cuda()
        goal_obs = Variable(torch.from_numpy(goal.reshape(1, 2))).float().cuda()
        speed_obs = Variable(torch.from_numpy(speed.reshape(1, 2))).float().cuda()
        
        _, _, _, mean = policy(laser_obs, goal_obs, speed_obs)
        mean = mean.data.cpu().numpy()
        scaled_action = np.clip(mean, a_min=action_bound[0], a_max=action_bound[1])

        controller.control_vel(scaled_action)
        rate.sleep()
        s_next = controller.get_laser_observation()
        left = controller.obs_stack.popleft()
        controller.obs_stack.append(s_next)
        goal_next = np.asarray(controller.get_local_goal())
        speed_next = np.asarray(controller.get_self_speed())
        state_next = [controller.obs_stack, goal_next, speed_next]

        controller.state_obs = state_next
        traveled_distance += calculatedDistance(np.array(controller.state_GT[0:-1]), prev_pos)
        traveled_time+= 0.1
        average_speed += speed_next[0]
        if controller.is_crashed == True:
            collision = True
        prev_pos = np.array(controller.state_GT[0:-1])

prev_evaluate = np.loadtxt(os.path.join(evaluate_path, environment_name, "robot_{}.txt".format(index)))
evaluate = np.array([success, round(traveled_time, 2), round(traveled_distance, 2), round(average_speed/int(traveled_time/0.1), 2)], np.float32).reshape((1, 4))
total_evaluate = np.concatenate((prev_evaluate, evaluate), axis=0)
print("Number of test of robot {} : {}".format(index, total_evaluate.shape[0]-2))
np.savetxt(os.path.join(evaluate_path, environment_name, "robot_{}.txt".format(index)), total_evaluate, fmt="%.2f")
# rospy.spin()