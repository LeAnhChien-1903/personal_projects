from turtle import home
from utlis.utlis import *
from env.factory_map_chessboard import FactoryMapChessboard
from env.robot import Robot
class EnvironmentChessboard:
    def __init__(self, data_folder:str, num_of_robot: int, robot_max_speed: float, robot_max_payload: float,
                figure: figure.Figure, map_visual: axs.Axes):
        self.dt = 0.1
        self.single_line_width = 1.0
        self.double_line_width = 2.0
        self.point_line_length = 1.2
        self.data_folder = data_folder
        self.figure = figure 
        self.map_visual:axs.Axes = map_visual
        self.factory_map = FactoryMapChessboard(data_folder, self.single_line_width, self.double_line_width, self.point_line_length)
        self.graph = self.factory_map.graph.copy()
        self.home_vertices: List[Vertex] = [] 
        for v_list in self.factory_map.waiting_vertices:
            for v in v_list:
                self.home_vertices.append(v)
        self.factory_map.visualizeMap(self.map_visual)
        assert num_of_robot <= len(self.home_vertices)
        self.num_of_robot = num_of_robot
        self.robot_max_speed = robot_max_speed
        self.robot_max_payload = robot_max_payload
        
        self.robots: List[Robot] = []
        index = 0
        for id in range(self.num_of_robot):
            home_id = index
            if id % 2 == 1:
                home_id = index + int(len(self.home_vertices)/2)
                index += 1
            if self.home_vertices[id].getCenterY() > self.factory_map.map_width/2:
                self.robots.append(Robot(self.dt, id, self.home_vertices[home_id], -math.pi/2, 
                                self.robot_max_speed, self.robot_max_payload, self.map_visual))
            else:
                self.robots.append(Robot(self.dt, id, self.home_vertices[home_id], math.pi/2, 
                                self.robot_max_speed, self.robot_max_payload, self.map_visual))
        d_safe_junction: float = 0.01
        d_safe_closest: float = 0.2
        robot_length: float = self.robots[0].robot_length
        self.wait_point_junction_dist: float = robot_length + d_safe_junction
        self.wait_point_closest_dist: float = robot_length + d_safe_closest
        
    def FSMControl(self, waiting_time: int):
        for robot in self.robots:
            robot.setWaitingTime(waiting_time)
            if self.taskAssignmentFSMState(robot) == True:
                robot.setFSMState(STOP)
            else:
                self.calculateBetweenAndSameList(robot)
                
    def AStarFSMControl(self, waiting_time: int):
        traffic_control: List[bool] = []
        for robot in self.robots:
            robot.setWaitingTime(waiting_time)
            if self.AStarTaskAssignmentFSMState(robot) == True:
                robot.setFSMState(STOP)
                robot.setNumBetween(self.num_of_robot)
                traffic_control.append(False)
            else:
                self.calculateBetweenAndSameList(robot)
                traffic_control.append(True)
        id_list = [i for i in range(len(self.robots))]
        sorted_id_list = sorted(id_list, key= lambda id: self.robots[id].getNumBetween())
        # Calculate velocity
        for id in sorted_id_list:
            if traffic_control[id] == True:
                self.trafficStateAssignment(self.robots[id])
        
    def taskAssignmentFSMState(self, robot: Robot):
        if robot.hasTask() == False:
            return True
        if robot.taskIsDone() == True:
            robot.clearRoute()
            robot.clearTask()
            return True
        if robot.hasRoute() == False:
            return True
        if robot.goalReached() == False:
            return False
        else:
            if robot.waitForPicking(robot.getWaitingTime()) == False:
                return True
            else:
                robot.startIsDone()
                robot.clearRoute()
                return False
    
    def AStarTaskAssignmentFSMState(self, robot: Robot):
        if robot.hasTask() == False:
            return True
        if robot.route.getType() == TO_TARGET:
            if robot.taskIsDone():
                if robot.waitForPicking(robot.getWaitingTime()) == False:
                    return True
                else:
                    robot.target_done = True
                    robot.setState(FREE)
                    robot.resetNextPoint()
                    robot.clearRoute()
                    robot.clearTask()
                    return True
        if robot.hasRoute() == False:
            return True
        if robot.getRouteDone() == False:
            if robot.start_reward_count == True:
                robot.plusRewardAllocation()
            if robot.route.getType() == TO_START:
                robot.setState(ON_WAY_TO_START)
            if robot.route.getType() == TO_TARGET:
                robot.setState(ON_WAY_TO_TARGET)
            if robot.route.getType() == TO_WAITING:
                robot.setState(ON_WAY_TO_WAITING)
            if robot.route.getType() == TO_CHARGING:
                robot.setState(ON_WAY_TO_CHARGING)
            return False
        else:
            if robot.waitForPicking(robot.getWaitingTime()) == False:
                return True
            else:
                robot.start_reward_count = False
                robot.done_count_reward = True
                robot.startIsDone()
                robot.clearRoute()
                robot.setRoute(robot.getTask().getRoute())
                return False
    
    def trafficStateAssignment(self, robot: Robot):
        if robot.getNumBetween() == 0:
            if robot.getNextPointIndex() == robot.getRouteLength() - 2:
                if robot.getNumSame() == 0:
                    wp = self.calculateWaitPoint(robot, robot.getRouteCoord(-2), self.wait_point_junction_dist)
                    if self.lineIsSafe(robot, robot.getRouteCoord(-2), robot.getGoal()) == False:
                        if isSamePoint(robot.getPosition(), wp):
                            robot.setFSMState(STOP_BY_GOAL_OCCUPIED)
                        else:
                            robot.setFSMState(GO_TO_WAIT_POINT_JUNCTION)
                            robot.setFSMNextPoint(wp)
                            
                    else:
                        if self.vertexIsSafe(robot, robot.getRouteCoord(-2)):
                            robot.setFSMState(GO_TO_NEXT_POINT)
                            robot.setFSMNextPoint(robot.getRouteCoord(-2))
                            
                        else:
                            if isSamePoint(robot.getPosition(), wp):
                                robot.setFSMState(STOP)
                                
                            else:
                                robot.setFSMState(GO_TO_WAIT_POINT_JUNCTION)
                                robot.setFSMNextPoint(wp)
                                
                else:
                    self.setStatePassSamePoint(robot)
            else:
                if robot.getNumSame() == 0:
                    if self.vertexIsSafe(robot, robot.getNextPoint()):
                        robot.setFSMState(GO_TO_NEXT_POINT)
                        robot.setFSMNextPoint(robot.getNextPoint())
                        
                    else:
                        wp2 = self.calculateWaitPoint(robot, robot.getNextPoint(), self.wait_point_junction_dist) 
                        if isSamePoint(robot.getPosition(), wp2):
                            robot.setFSMState(STOP)
                            
                        else:
                            robot.setFSMState(GO_TO_WAIT_POINT_JUNCTION)
                            robot.setFSMNextPoint(wp2)
                            
                else:
                    self.setStatePassSamePoint(robot)
        else:
            closest_robot: Robot = self.robots[robot.getClosestID()]
            if robot.getNextPointIndex() == robot.getRouteLength() - 2:
                if robot.getNumSame() == 0:
                    if pointIsBetweenALine(robot.getPosition(), closest_robot.getPosition(), robot.getRouteCoord(-2)):
                        if self.lineIsSafe(robot, robot.getRouteCoord(-2), robot.getGoal()):
                            if self.vertexIsSafe(robot, robot.getRouteCoord(-2)):
                                robot.setFSMState(GO_TO_NEXT_POINT)
                                robot.setFSMNextPoint(robot.getRouteCoord(-2))
                                
                            else:
                                wp3 = self.calculateWaitPoint(robot, robot.getRouteCoord(-2), self.wait_point_junction_dist)
                                if isSamePoint(robot.getPosition(), wp3):
                                    robot.setFSMState(STOP)
                                    
                                else:
                                    robot.setFSMState(GO_TO_WAIT_POINT_JUNCTION)
                                    robot.setFSMNextPoint(wp3)
                                    
                        else:
                            wp4 = self.calculateWaitPoint(robot, robot.getRouteCoord(-2), self.wait_point_junction_dist)
                            if isSamePoint(robot.getPosition(), wp4):
                                robot.setFSMState(STOP_BY_GOAL_OCCUPIED)
                                
                            else:
                                robot.setFSMState(GO_TO_WAIT_POINT_JUNCTION)
                                robot.setFSMNextPoint(wp4)
                                
                    else:
                        wp = self.calculateWaitPoint(robot, self.robots[robot.getClosestID()].getPosition(), self.wait_point_closest_dist)
                        if isSamePoint(robot.getPosition(), wp):
                            robot.setFSMState(STOP)
                            
                        else:
                            robot.setFSMState(GO_TO_WAIT_POINT_CLOSEST)
                            robot.setFSMNextPoint(wp) 
                else:
                    if pointIsBetweenALine(robot.getPosition(), closest_robot.getPosition(), robot.getSamePoint()):
                        self.setStatePassSamePoint(robot)
                    else:
                        wp = self.calculateWaitPoint(robot, self.robots[robot.getClosestID()].getPosition(), self.wait_point_closest_dist)
                        if isSamePoint(robot.getPosition(), wp):
                            robot.setFSMState(STOP)
                            
                        else:
                            robot.setFSMState(GO_TO_WAIT_POINT_CLOSEST)
                            robot.setFSMNextPoint(wp) 
            else:
                if robot.getNumSame() == 0:
                    wp = self.calculateWaitPoint(robot, self.robots[robot.getClosestID()].getPosition(), self.wait_point_closest_dist)
                    if isSamePoint(robot.getPosition(), wp):
                        robot.setFSMState(STOP)
                    else:
                        robot.setFSMState(GO_TO_WAIT_POINT_CLOSEST)
                        robot.setFSMNextPoint(wp) 
                else:
                    if pointIsBetweenALine(robot.getPosition(), closest_robot.getPosition(), robot.getSamePoint()):
                        self.setStatePassSamePoint(robot)
                    else:
                        wp = self.calculateWaitPoint(robot, self.robots[robot.getClosestID()].getPosition(), self.wait_point_closest_dist)
                        if isSamePoint(robot.getPosition(), wp):
                            robot.setFSMState(STOP)
                            
                        else:
                            robot.setFSMState(GO_TO_WAIT_POINT_CLOSEST)
                            robot.setFSMNextPoint(wp) 
        robot.run()
        
    def setStatePassSamePoint(self, robot: Robot):
        if robot.getNextPointIndex() == robot.getRouteLength() - 2:
            wp = self.calculateWaitPoint(robot, robot.getRouteCoord(-2), self.wait_point_junction_dist)
            if self.lineIsSafe(robot, robot.getRouteCoord(-2), robot.getGoal()) == False:
                if isSamePoint(robot.getPosition(), wp):
                    robot.setFSMState(STOP_BY_GOAL_OCCUPIED)
                else:
                    robot.setFSMState(GO_TO_WAIT_POINT_JUNCTION)
                    robot.setFSMNextPoint(wp)
            else:
                if self.vertexIsSafe(robot, robot.getRouteCoord(-2)):
                    robot.setFSMState(GO_TO_NEXT_POINT)
                    robot.setFSMNextPoint(robot.getRouteCoord(-2))
                else:
                    if isSamePoint(robot.getPosition(), wp):
                        robot.setFSMState(STOP)
                    else:
                        robot.setFSMState(GO_TO_WAIT_POINT_JUNCTION)
                        robot.setFSMNextPoint(wp)
        else:
            if EuclidDistance(robot.getPosition(), robot.getSamePoint()) < self.wait_point_junction_dist:
                robot.setFSMState(GO_TO_NEXT_POINT)
                robot.setFSMNextPoint(robot.getSamePoint())
            else:
                wait_time_list = robot.getWaitTimeList()
                dist_list = robot.getDistToSameList()
                if wait_time_list.index(max(wait_time_list)) == 0:
                    if wait_time_list.count(max(wait_time_list)) == 1:
                        if self.vertexIsSafe(robot, robot.getSamePoint()):
                            robot.setFSMState(GO_TO_NEXT_POINT)
                            robot.setFSMNextPoint(robot.getSamePoint())
                            
                        else:
                            wp = self.calculateWaitPoint(robot, robot.getSamePoint(), self.wait_point_junction_dist)
                            if isSamePoint(robot.getPosition(), wp):
                                robot.setFSMState(STOP)
                                
                            else:
                                robot.setFSMState(GO_TO_WAIT_POINT_JUNCTION)
                                robot.setFSMNextPoint(wp)
                                
                    else:
                        if dist_list.index(min(dist_list)) == 0:
                            if self.vertexIsSafe(robot, robot.getSamePoint()):
                                robot.setFSMState(GO_TO_NEXT_POINT)
                                robot.setFSMNextPoint(robot.getSamePoint())
                                
                            else:
                                wp = self.calculateWaitPoint(robot, robot.getSamePoint(), self.wait_point_junction_dist)
                                if isSamePoint(robot.getPosition(), wp):
                                    robot.setFSMState(STOP)
                                    
                                else:
                                    robot.setFSMState(GO_TO_WAIT_POINT_JUNCTION)
                                    robot.setFSMNextPoint(wp)
                                    
                        else:
                            wp = self.calculateWaitPoint(robot, robot.getSamePoint(), self.wait_point_junction_dist)
                            if isSamePoint(robot.getPosition(), wp):
                                robot.setFSMState(STOP)
                                
                            else:
                                robot.setFSMState(GO_TO_WAIT_POINT_JUNCTION)
                                robot.setFSMNextPoint(wp)
                                
                else:
                    wp = self.calculateWaitPoint(robot, robot.getSamePoint(), self.wait_point_junction_dist)
                    if isSamePoint(robot.getPosition(), wp):
                        robot.setFSMState(STOP)
                        
                    else:
                        robot.setFSMState(GO_TO_WAIT_POINT_JUNCTION)
                        robot.setFSMNextPoint(wp)

    def calculateBetweenAndSameList(self, robot: Robot):
        same_line_list: List[int] = [robot.getID()]
        same_vertex_list: List[int] = []
        min_same_dist: float = float('inf')
        
        for other in self.robots:
            if other.getID() == robot.getID():
                continue
            intersect_cond, same_point = check_line_segments_intersection_2d(robot.getPosition(), robot.getNextPoint(), other.getPosition(), other.getNextPoint())
            angle_cond = angleByTwoPoint(other.getPosition(), other.getNextPoint()) == angleByTwoPoint(robot.getPosition(), robot.getNextPoint())
            if intersect_cond:
                cond1 = min_same_dist >= EuclidDistance(robot.getPosition(), same_point)
                cond2 = pointIsBetweenALine(robot.getPosition(), robot.getNextPoint(), same_point)
                cond3 = pointIsBetweenALine(other.getPosition(), other.getNextPoint(), same_point)
                if cond1 and cond2 and cond3 and angle_cond == False:
                    min_same_dist = EuclidDistance(robot.getPosition(), same_point)
                    robot.setSamePoint(same_point)
                    same_vertex_list.append(other.getID())
            if angle_cond and are_points_collinear(robot.getPosition(), robot.getNextPoint(), other.getNextPoint()):
                same_line_list.append(other.getID())
        angle = angleByTwoPoint(robot.getPosition(), robot.getNextPoint())
        same_line_sorted = sorted(same_line_list, key=lambda id: self.robots[id].getX())
        if -math.pi/2 - 0.01 <= angle <= -math.pi/2 + 0.01: # Up wait point
            same_line_sorted = sorted(same_line_list, key=lambda id: self.robots[id].getY(), reverse=True)
        if math.pi/2 - 0.01 <= angle <= math.pi/2 + 0.01: # Down wait point
            same_line_sorted = sorted(same_line_list, key=lambda id: self.robots[id].getY(), reverse=False)
        if math.pi - 0.01 <= abs(normalizeAngle(angle)) <= math.pi + 0.01: # Right wait point
            same_line_sorted = sorted(same_line_list, key=lambda id: self.robots[id].getX(), reverse=True)
        if -0.01 <= angle <= 0.01: # Left wait point
            same_line_sorted = sorted(same_line_list, key=lambda id: self.robots[id].getX(), reverse=False)

        between_id_filtered: List[int] = same_line_sorted[same_line_sorted.index(robot.getID())+1: ]
        robot.setNumBetween(len(between_id_filtered))
        if len(between_id_filtered) > 0:
            robot.setClosestID(between_id_filtered[0])
        
        if len(same_vertex_list) > 0:
            wait_time_list: List[float] = [robot.getFSMWaitTime()]
            dist_to_same_list: List[float] = [EuclidDistance(robot.getPosition(), robot.getSamePoint())]
            rest_to_goal_list: List[float] = [robot.getRestRouteCost()]
            same_id_filtered = self.findSameVertexList(robot, same_vertex_list)
            for id in same_id_filtered:
                wait_time_list.append(self.robots[id].getFSMWaitTime())
                dist_to_same_list.append(EuclidDistance(self.robots[id].getPosition(), robot.getSamePoint()))
                rest_to_goal_list.append(self.robots[id].getRestRouteCost())
            robot.setWaitTimeList(wait_time_list)
            robot.setRestCostList(rest_to_goal_list)
            robot.setDistToSameList(dist_to_same_list)
            robot.setSameIDList(same_id_filtered) 
            robot.setNumSame(len(same_id_filtered))
        else:
            robot.setNumSame(len(same_vertex_list))
            robot.setSameIDList(same_vertex_list)  
    
    def findSameVertexList(self, robot: Robot, same_vertex_list: List[int]):
        if len(same_vertex_list) == 1: return same_vertex_list
        same_id_filtered: List[int] = []
        list_1, list_2, list_3 = [], [], []
        robot_same_point = robot.getSamePoint()
        for id in same_vertex_list:
            _, same_point = check_line_segments_intersection_2d(robot.getPosition(), robot.getNextPoint(), self.robots[id].getPosition(), self.robots[id].getNextPoint())
            if isSamePoint(robot_same_point, same_point):
                if len(list_1) == 0:
                    list_1.append([id, EuclidDistance(self.robots[id].getPosition(), 
                                                        robot_same_point), 
                                    angleByTwoPoint(self.robots[id].getPosition(), robot_same_point)])
                else:
                    angle = angleByTwoPoint(self.robots[id].getPosition(), robot_same_point)
                    if abs(calculateDifferenceOrientation(angle, list_1[0][-1])) < math.pi/12:
                        list_1.append([id, EuclidDistance(self.robots[id].getPosition(), 
                                                        robot_same_point), 
                                    angleByTwoPoint(self.robots[id].getPosition(), robot_same_point)])
                    else:
                        if len(list_2) == 0:
                            list_2.append([id, EuclidDistance(self.robots[id].getPosition(), 
                                                                robot_same_point), 
                                            angleByTwoPoint(self.robots[id].getPosition(), robot_same_point)])
                        else:
                            angle = angleByTwoPoint(self.robots[id].getPosition(), robot_same_point)
                            if abs(calculateDifferenceOrientation(angle, list_2[0][-1])) < math.pi/12:
                                list_2.append([id, EuclidDistance(self.robots[id].getPosition(), 
                                                                robot_same_point), 
                                            angleByTwoPoint(self.robots[id].getPosition(), robot_same_point)])
                            else:
                                if len(list_3) == 0:
                                    list_3.append([id, EuclidDistance(self.robots[id].getPosition(), 
                                                                        robot_same_point), 
                                                    angleByTwoPoint(self.robots[id].getPosition(), robot_same_point)])
                                else:
                                    angle = angleByTwoPoint(self.robots[id].getPosition(), robot_same_point)
                                    if abs(calculateDifferenceOrientation(angle, list_3[0][-1])) < math.pi/12:
                                        list_3.append([id, EuclidDistance(self.robots[id].getPosition(), 
                                                                        robot_same_point), 
                                                    angleByTwoPoint(self.robots[id].getPosition(), robot_same_point)])
        if len(list_1) != 0:
            list_1 = sorted(list_1, key= lambda x: x[1])
            same_id_filtered.append(list_1[0][0])
        if len(list_2) != 0:
            list_2 = sorted(list_2, key= lambda x: x[1])
            same_id_filtered.append(list_2[0][0])
        if len(list_3) != 0:
            list_3 = sorted(list_3, key= lambda x: x[1])
            same_id_filtered.append(list_3[0][0])
        return same_id_filtered
    
    def calculateWaitPoint(self, robot: Robot, target_point: np.ndarray, distance: float):
        angle = angleByTwoPoint(robot.getPosition(), target_point)
        if -math.pi/2 - 0.01 <= angle <= -math.pi/2 + 0.01: # Up wait point
            return np.array([target_point[0], target_point[1] + distance])
        if math.pi/2 - 0.01 <= angle <= math.pi/2 + 0.01: # Down wait point
            return np.array([target_point[0], target_point[1] - distance])
        if math.pi - 0.01 <= abs(normalizeAngle(angle)) <= math.pi + 0.01: # Right wait point
            return np.array([target_point[0] + distance, target_point[1]])
        if -0.01 <= angle <= 0.01: # Left wait point
            return np.array([target_point[0] - distance, target_point[1]])
        return robot.getPosition()
    
    def lineIsSafe(self, robot: Robot, start_point: np.ndarray, end_point: np.ndarray):
        for other in self.robots:
            if other.getID() == robot.getID(): continue
            if EuclidDistance(start_point, other.getPosition()) < self.wait_point_junction_dist: return False
            if EuclidDistance(end_point, other.getPosition()) < self.wait_point_junction_dist: return False
            if pointIsBetweenALine(start_point, end_point, other.getPosition()): return False
            if distanceBetweenPointAndLine(start_point, end_point, other.getPosition()) < self.wait_point_junction_dist: return False
        return True
            
    def vertexIsSafe(self, robot: Robot, next_point: np.ndarray):
        for other in self.robots:
            if other.getID() == robot.getID(): continue
            if EuclidDistance(next_point, other.getPosition()) < self.wait_point_junction_dist:
                return False
        return True
        
    def stateControl(self):
        id_list = [i for i in range(len(self.robots))]
        sorted_id_list = sorted(id_list, key= lambda id: self.robots[id].getNumBetween())
        # Calculate velocity
        for id in sorted_id_list:
            if self.robots[id].getFSMState() == GO_TO_WAIT_POINT_CLOSEST:
                wp = self.calculateWaitPoint(self.robots[id], self.robots[self.robots[id].getClosestID()].getPosition(), 
                                            self.wait_point_closest_dist)
                if isSamePoint(self.robots[id].getPosition(), wp):
                    self.robots[id].setFSMState(STOP)
                else:
                    self.robots[id].setFSMNextPoint(wp)               
            self.robots[id].run()
        
    def visualize(self):
        for robot in self.robots:
            robot.robotVisualization()
            