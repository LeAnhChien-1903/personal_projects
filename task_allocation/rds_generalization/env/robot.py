from utlis.utlis import *

class Robot:
    def __init__(self, dt: float, id: int, home: Vertex, home_theta: float, max_speed: float, max_payload: float, 
                map_visual: axs.Axes, robot_length: float = 0.98, robot_width: float = 0.62):
        self.dt = dt
        self.id = id
        self.state = FREE
        self.fsm_state: int = STOP
        self.fsm_wait_time: float = 0.0
        self.fsm_next_point: np.ndarray = home.getCenter()
        self.num_between: int = 0 # number of robot between robot and next point
        self.num_same: int = 0 # number of robot has same next point with robot
        self.closest_id: int = 0 # id of closest robot
        self.same_id_list: List[int] = [] # id of same robot   
        self.same_point: np.ndarray = np.zeros(2)
        self.wait_time_list: List[float] = [] #
        self.dist_to_same_list: List[float] = [] # list of distance from same id robot to same point 
        self.rest_cost_list: List[float] = [] # list of rest cost from robot to its goal
        self.home: Vertex = home
        self.graph_id: int = home.getID()
        self.pose: np.ndarray = np.append(home.getCenter(), home_theta)
        self.vel: np.ndarray = np.zeros(2)
        self.payload: float = 0.0
        self.max_speed = max_speed
        self.max_payload = max_payload
        self.task: Task = Task()
        self.task_list: List[Task] = []
        self.route: Route = Route()
        self.route_id: int = 1
        self.route_done: bool = False
        self.goal: np.ndarray = home.getCenter()
        self.robot_length = robot_length
        self.robot_width = robot_width
        robot_coords = np.array(calculateRectangleCoordinate(home.getCenterX(), home.getCenterY(), home_theta, 
                                                            robot_length, robot_width))
        self.map_visual = map_visual
        self.robot_visual: Polygon = map_visual.fill(robot_coords[:, 0], robot_coords[:, 1], c= state_color[self.getState()])[0]
        self.robot_text: Text = map_visual.text(self.getX(), self.getY(), str(self.id), size=5)
        self.route_visual: Line2D = map_visual.plot([], [], '-')[0]
        self.waiting_counter: int = 1
        self.waiting_time: int = 30
        self.reward_allocation: float = 0.0
        self.reward_idx: int = -1
        self.start_reward_count = False
        self.done_count_reward = False
        self.target_done: bool = False
        self.task_for_test: bool = False

    def waitForPicking(self, waiting_time: int):
        if self.waiting_counter < waiting_time:
            self.setState(PICKING_UP)
            self.waiting_counter += 1
        if self.waiting_counter >= waiting_time:
            self.setState(BUSY)
            return True
        return False
    
    def routeDone(self):
        if EuclidDistance(self.getPosition(), self.route.getCoord(-1)) < MAX_SAME_DIST:
            return True
        return False
        
    def limitVelocity(self, vel:np.ndarray):
        speed = math.hypot(vel[0], vel[1])
        if speed > self.max_speed:
            limited_speed = (vel / speed) * self.max_speed
            return limited_speed
        return vel
    
    def run(self):
        cond1 = self.getFSMState() == GO_TO_NEXT_POINT
        cond2 = self.getFSMState() == GO_TO_WAIT_POINT_JUNCTION
        cond3 = self.getFSMState() == GO_TO_WAIT_POINT_CLOSEST
        if cond1 or cond2 or cond3:
            self.resetFSMWaitTime()
        if self.getFSMState() == STOP_BY_GOAL_OCCUPIED:
            self.fsm_wait_time = -0.1
        if self.getFSMState() == STOP and self.getState() != PICKING_UP:
            self.plusFSMWaitTime()
        
        if self.getFSMState() == STOP or self.getFSMState() == STOP_BY_GOAL_OCCUPIED:
            self.vel = np.zeros(2)
        else:
            self.vel = self.limitVelocity((self.getFSMNextPoint() - self.getPosition())/self.dt)
            self.pose[0]+= self.vel[0] * self.dt
            self.pose[1]+= self.vel[1] * self.dt
            
        if self.getNextPointIndex() < self.route.getLength() - 1:
            self.pose[2] = angleByTwoPoint(self.getRouteCoord(self.getNextPointIndex() - 1), self.getNextPoint())
        else:
            self.pose[2] = angleByTwoPoint(self.getNextPoint(), self.getRouteCoord(self.getNextPointIndex() - 1))
        
        if isSamePoint(self.getPosition(), self.getNextPoint()):
            self.plusNextPointIndex()

    def goalReached(self):
        if EuclidDistance(self.getGoal(), self.getPosition()) < MAX_SAME_DIST:
            return True
        return False
    
    def taskIsDone(self):
        if self.task.isTask() == True:
            if EuclidDistance(self.task.getTarget().getCenter(), self.getPosition()) < MAX_SAME_DIST:
                self.setState(FREE)
                return True
        return False
    
    def startIsDone(self):
        if self.task.isTask() == True:
            if EuclidDistance(self.task.getStart().getCenter(), self.getPosition()) < MAX_SAME_DIST:
                self.goal = self.task.getTarget().getCenter()
    
    def getRouteType(self): return self.route.getType()
    def getRouteLength(self): return self.route.getLength()
    def getRouteVertices(self): return self.route.getVertices()
    def getRouteVertex(self, id: int): return self.route.getVertex(id)
    def getRouteCoords(self): return self.route.getCoords()
    def getRouteCoord(self, id: int): return self.route.getCoord(id)
    def getRouteX(self, id: int): return self.route.getCoordX(id)
    def getRouteY(self, id: int): return self.route.getCoordY(id)
    def getRouteXList(self): return self.route.getCoordXList()
    def getRouteYList(self): return self.route.getCoordYList()
    def getRouteCost(self): return self.route.getRouteCost()
    
    def hasRoute(self): return self.route.isRoute()
    def hasTask(self): return self.task.isTask()
    def getRouteDone(self):
        if self.hasRoute():
            if isSamePoint(self.getPosition(), self.getRouteCoord(-1)):
                return True
        return False
    def clearRoute(self): self.route.clearRoute()
    def clearTask(self): self.task.clearTask()
    
    def getStartCenter(self): return self.task.getStartCenter()
    def getTargetCenter(self): return self.task.getTargetCenter()
    
    def updateRoute(self, next_vertex: int, next_coord: np.ndarray):
        self.route.update(next_vertex, next_coord)
        
    def setRoute(self, route: Route):
        # self.route_visual.set_xdata(route.getCoordXList())
        # self.route_visual.set_ydata(route.getCoordYList())
        self.route = route.copy()
        self.route_id = 1
        self.route_done = False
        self.waiting_counter = 1

    def getNextPoint(self)-> np.ndarray: return self.route.getCoord(self.route_id)
    def getNextPointIndex(self)-> int: return self.route_id
    def getNextVertex(self)-> int: return self.route.getVertex(self.route_id)
    def plusNextPointIndex(self): 
        if self.goalReached() == False:
            self.route_id += 1
        else:
            self.route_id = self.getRouteLength() - 1
    def resetNextPoint(self): self.route_id = 1
    def getRestRouteCost(self):
        rest_cost = EuclidDistance(self.getPosition(), self.getNextPoint())
        for i in range(self.getNextPointIndex(), self.getRouteLength() - 1):
            rest_cost += EuclidDistance(self.getRouteCoord(i), self.getRouteCoord(i+1))
        return rest_cost
    def getSamePoint(self): return self.same_point
    def setSamePoint(self, same_point: np.ndarray): self.same_point = same_point
    
    def getWaitingTime(self): return self.waiting_time
    def setWaitingTime(self, waiting_time: int): self.waiting_time = waiting_time
    
    def getID(self): return self.id
    def getHome(self): return self.home.copy()
    def getGraphID(self): return self.graph_id
    
    def getX(self): return self.pose[0]
    def getY(self): return self.pose[1]
    def getPosition(self): return self.pose[0:2].copy()
    def getTheta(self): return self.pose[2]
    def getPose(self): return self.pose.copy()
    
    def getGoalX(self): return self.goal[0]
    def getGoalY(self): return self.goal[1]
    def getGoal(self): return self.goal.copy()
    
    def getVelX(self): return self.vel[0]
    def getVelY(self): return self.vel[1]
    def getVel(self): return self.vel.copy()
    def setVel(self, vel: np.ndarray): self.vel = vel.copy()
    
    def getPayload(self): return self.payload
    def getRestPayload(self): return self.max_payload - self.payload
    
    def getState(self): return self.state
    def getFSMState(self): return self.fsm_state
    def getFSMWaitTime(self): return self.fsm_wait_time
    def getFSMNextPoint(self): return self.fsm_next_point
    def setFSMState(self, state: int): self.fsm_state = state
    def resetFSMWaitTime(self): self.fsm_wait_time = 0.0
    def plusFSMWaitTime(self): self.fsm_wait_time += self.dt
    
    def setFSMNextPoint(self, point: np.ndarray): self.fsm_next_point = point
    def getNumBetween(self): return self.num_between
    def getNumSame(self): return self.num_same
    def getClosestID(self): return self.closest_id
    def getSameIDList(self): return self.same_id_list.copy()
    def getWaitTimeList(self): return self.wait_time_list.copy()
    def getDistToSameList(self): return self.dist_to_same_list.copy()
    def getRestCostList(self): return self.rest_cost_list.copy()
    
    def setNumBetween(self, num_between: int): self.num_between = num_between
    def setNumSame(self, num_same: int): self.num_same = num_same
    def setClosestID(self, closest_id: int): self.closest_id = closest_id
    def setSameIDList(self, same_id_list: List[int]): self.same_id_list = same_id_list.copy()
    def setWaitTimeList(self, wait_time_list: List[float]): self.wait_time_list = wait_time_list.copy()
    def setDistToSameList(self, dist_to_same: List[float]): self.dist_to_same_list = dist_to_same.copy()
    def setRestCostList(self, rest_cost_list: List[float]): self.rest_cost_list = rest_cost_list.copy()
    
    def getTask(self): return self.task
    def setTask(self, task: Task): 
        self.task = task.copy()
        self.payload += task.getMass()
        self.goal = task.getStartCenter()
        self.target_done = False
    
    def setGraphID(self, id: int): self.graph_id = id
    def setState(self, state: int): self.state = state
    
    def getRewardIndex(self): return self.reward_idx
    def getRewardAllocation(self): return self.reward_allocation
    def plusRewardAllocation(self): self.reward_allocation = self.reward_allocation + 0.1
    def resetRewardAllocation(self): self.reward_allocation = 0.0
    def setRewardIndex(self, reward_index: int):
        self.reward_allocation = 0.0
        self.start_reward_count = True
        self.reward_idx: int = reward_index
        self.done_count_reward = False
    
    def setPose(self, x: float, y: float, theta: float):
        self.pose[0] = x
        self.pose[1] = y
        self.pose[2] = theta
    
    def robotVisualization(self):
        self.robot_visual.set_xy(np.array(calculateRectangleCoordinate(self.pose[0], self.pose[1],
                                                                        self.pose[2], self.robot_length,
                                                                        self.robot_width)))
        self.robot_visual.set_color(state_color[self.getState()])
        self.robot_text.set_x(self.getX())
        self.robot_text.set_y(self.getY())