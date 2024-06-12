#!/usr/bin/env python3

import rospy
from collision_avoidance.agent import Agent

rospy.init_node("deep_rl_node", anonymous=False)

index = int(rospy.get_name().split("_")[1])
controller = Agent(512, index, "stage2")
rospy.spin()