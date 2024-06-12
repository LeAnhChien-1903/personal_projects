from matplotlib.dates import FR
from numpy import deg2rad
from env.robot import Robot
from utlis.utlis import *
from models.planning_model import PlanningNetwork
import copy

class MiniBatch:
    def __init__(self, mini_batch_size: int, num_neighbor: int, num_graph_edge: int = 5, num_graph_feat: int = 8,
                num_robot_feat: int = 5, num_next_point: int = 4, num_next_point_feat: int = 4):
        self.mini_batch_size: int = mini_batch_size
        self.num_neighbor: int = num_neighbor + 1
        self.num_graph_edge: int = num_graph_edge
        self.num_graph_feat: int = num_graph_feat
        self.num_robot_feat: int = num_robot_feat
        self.num_next_point: int = num_next_point
        self.num_next_point_feat: int = num_next_point_feat
        self.mini_batch_id: int = 0
        with torch.no_grad():
            self.robot_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size, self.num_robot_feat).to(device)
            self.next_point_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size, self.num_next_point, self.num_next_point_feat).to(device)
            self.graph_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size, self.num_neighbor, self.num_graph_edge, self.num_graph_feat).to(device)
            self.mask_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size, self.num_next_point).to(device)
            self.action_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size).to(device)
            self.reward_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size).to(device)
            self.done_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size).to(device)
            self.value_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size).to(device)
            self.log_prob_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size).to(device)
            self.entropy_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size).to(device)
            self.advantage_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size).to(device)
            self.return_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size).to(device)
        
    def update(self, robot_data: torch.Tensor, next_point_data: torch.Tensor, graph_data: torch.Tensor, mask_data: torch.Tensor,
                action: torch.Tensor, value: torch.Tensor, log_prob: torch.Tensor, entropy: torch.Tensor):
        with torch.no_grad():
            self.robot_mini_batch[self.mini_batch_id] = robot_data
            self.next_point_mini_batch[self.mini_batch_id] = next_point_data
            self.graph_mini_batch[self.mini_batch_id] = graph_data
            self.mask_mini_batch[self.mini_batch_id] = mask_data
            self.action_mini_batch[self.mini_batch_id] = action
            self.value_mini_batch[self.mini_batch_id] = value
            self.log_prob_mini_batch[self.mini_batch_id] = log_prob
            self.entropy_mini_batch[self.mini_batch_id] = entropy
    
    def updateRewardAndDoneState(self, reward: torch.Tensor, done_state: torch.Tensor):
        with torch.no_grad():
            self.done_mini_batch[self.mini_batch_id] = done_state
            self.reward_mini_batch[self.mini_batch_id] = reward
            self.mini_batch_id += 1
    
    def advantageEstimator(self, last_value:torch.Tensor, lam: float = 0.95, gamma: float = 0.99):
        with torch.no_grad():
            last_gae_lam = torch.zeros(1).to(device)
            for t in reversed(range(self.mini_batch_size)):
                if t == self.mini_batch_size - 1:
                    next_value = last_value
                    terminal = 1 - self.done_mini_batch[-1]
                else:
                    next_value = self.value_mini_batch[t+1]
                    terminal = 1 - self.done_mini_batch[t+1]
                delta = self.reward_mini_batch[t] + gamma * next_value * terminal - self.value_mini_batch[t]
                self.advantage_mini_batch[t] = delta + gamma * lam * terminal * last_gae_lam
                last_gae_lam = delta + gamma * lam * last_gae_lam
            
            self.return_mini_batch = self.advantage_mini_batch + self.value_mini_batch
    def clear(self):
        self.mini_batch_id: int = 0
        with torch.no_grad():
            self.robot_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size, self.num_robot_feat).to(device)
            self.next_point_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size, self.num_next_point, self.num_next_point_feat).to(device)
            self.graph_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size, self.num_neighbor, self.num_graph_edge, self.num_graph_feat).to(device)
            self.mask_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size, self.num_next_point).to(device)
            self.action_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size).to(device)
            self.reward_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size).to(device)
            self.done_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size).to(device)
            self.value_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size).to(device)
            self.log_prob_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size).to(device)
            self.entropy_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size).to(device)
            self.advantage_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size).to(device)
            self.return_mini_batch: torch.Tensor = torch.zeros(self.mini_batch_size).to(device)

class PathPlanning:
    def __init__(self, robots: List[Robot], graph: Graph, comm_range: float, model_folder: str, 
                num_neighbor: int = 10):
        self.robots: List[Robot] = robots
        self.factory_graph: Graph = graph
        self.planning_graph: Graph = graph.copy()
        self.comm_range: float = comm_range
        self.model_folder: str = model_folder
        self.num_neighbor: int = min(len(self.robots) -1, num_neighbor)
        self.model: PlanningNetwork = PlanningNetwork().to(device= device)
        self.loadModel()
        self.best_all_rewards: float = -float('inf')
        self.best_local_reward: float = -float('inf')
        self.best_loss: float = float('inf')
        self.clip_range: float = 0.1
        self.lr: float = 1e-4 # Learning rate
        self.batch_size: int = 8000
        self.mini_batch_size: int = 512 # Update Interval while training
        self.num_epoch: int  = 16 # Epoch per iteration of PPO
        self.gamma: float = 0.99 # Discount factor
        self.lam: float = 0.95 # Lambda used in GAE (General Advantage Estimation)
        self.entropy_coef: float = 0.01 # Entropy coefficient
        self.value_coef: float = 0.0002 # Value function coefficient
        self.policy_coef: float = 0.02 # Policy function coefficient
        self.has_last_value: List[bool] = [False for _ in range(len(self.robots))]
        self.mini_batch_list: List[MiniBatch] = [MiniBatch(mini_batch_size= self.mini_batch_size, 
                                                            num_neighbor= self.num_neighbor) for _ in range(len(self.robots))]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.test_reward: float = 0.0
        
    def testing(self, stop: bool = False):
        if stop == False:
            self.pathTesting()
            return False, 0.0
        else: 
            test_reward = round(self.test_reward, 2)
            self.test_reward = 0.0
            return True, test_reward
    
    def training(self, iter: int, save_interval: int= 10):
        if False in self.has_last_value:
            self.pathTraining()
            min_mini_batch_id = self.mini_batch_list[0].mini_batch_id
            for mini_batch in self.mini_batch_list:
                if mini_batch.mini_batch_id < min_mini_batch_id:
                    min_mini_batch_id = mini_batch.mini_batch_id
            # if min_mini_batch_id % 200 == 199 or min_mini_batch_id == self.mini_batch_size - 1:
                # print("Min mini batch id: ", min_mini_batch_id + 1)
            return False
        else:
            all_rewards: float = 0.0
            for mini_batch in self.mini_batch_list:
                all_rewards += torch.sum(mini_batch.reward_mini_batch).item()
            # print("All Reward in iteration {}:".format(iter+1),round(all_rewards, 2))
            if self.best_all_rewards < all_rewards:
                print("\nBest all rewards: {}".format(round(all_rewards, 2)))
                self.best_all_rewards = all_rewards
                self.best_all_reward_model = copy.deepcopy(self.model)
            for _ in range(self.num_epoch):
                random.shuffle(self.mini_batch_list)
                for mini_batch in self.mini_batch_list:
                    loss, kl_div = self.calculateLoss(mini_batch)
                    # Zero out the previously calculated gradients
                    self.optimizer.zero_grad()
                    # Calculate gradients
                    loss.backward()
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    # Update parameters based on gradients
                    self.optimizer.step()
                    if abs(self.best_loss) > abs(loss.item()):
                        self.best_loss = loss.item()
                        self.best_loss_model = copy.deepcopy(self.model)
            if iter % save_interval == save_interval - 1:
                self.saveModel()
                self.saveBestLossModel()
                self.saveBestAllRewardModel()
            self.planning_graph = self.factory_graph.copy()
            self.has_last_value = [False for _ in range(len(self.robots))]
            for mini_batch in self.mini_batch_list:
                mini_batch.clear()
                
            return True
    
    def calculateLoss(self, mini_batch: MiniBatch):
        normalized_advantage = self.normalize(mini_batch.advantage_mini_batch)
        value, log_prob, entropy = self.model.evaluateAction(mini_batch.robot_mini_batch, mini_batch.next_point_mini_batch,
                                                                mini_batch.graph_mini_batch, mini_batch.mask_mini_batch, 
                                                                mini_batch.action_mini_batch)
        policy_loss: torch.Tensor = self.calculatePolicyLoss(mini_batch, log_prob, normalized_advantage)
        entropy_bonus = entropy.mean()
        value_loss: torch.Tensor = self.calculateValueLoss(mini_batch, value)

        loss = (policy_loss + value_loss * self.value_coef - self.entropy_coef * entropy_bonus)
        approx_kl_divergence = 0.5 * ( (mini_batch.log_prob_mini_batch - log_prob) ** 2).mean()

        return loss, approx_kl_divergence
    
    def calculatePolicyLoss(self, mini_batch: MiniBatch, log_prob: torch.Tensor, advantage: torch.Tensor):
        ratio = torch.exp(log_prob - mini_batch.log_prob_mini_batch)
        clipped_ratio = ratio.clamp(min = 1.0 - self.clip_range,
                                    max = 1.0 + self.clip_range)
        policy_reward = torch.min(ratio * advantage, clipped_ratio * advantage)
        
        return -policy_reward.mean()
    
    def calculateValueLoss(self, mini_batch: MiniBatch, value: torch.Tensor):
        clipped_value = mini_batch.value_mini_batch + (value - mini_batch.value_mini_batch).clamp(min=-self.clip_range, max=self.clip_range)
        vf_loss = torch.max((value - mini_batch.value_mini_batch) ** 2, (clipped_value - mini_batch.value_mini_batch) ** 2)
        return 0.5 * vf_loss.mean()
    
    @staticmethod
    def normalize(adv: torch.Tensor):
        return (adv - adv.mean()) / (adv.std() - 1e-8)
    
    def pathTesting(self):
        for robot in self.robots:
            if isSamePoint(robot.getGoal(), robot.getPosition()):
                for other in self.robots:
                    if robot.getID() != other.getID():
                        other.route.clearRoute()
        graph_list = self.getRobotsGraph()
        need_planning: List[bool] = [False for _ in range(len(self.robots))]
        # Collect data for all robots
        for robot in self.robots:
            if self.has_last_value[robot.getID()] == True:
                continue
            if self.needToPlanning(robot) == True:
                need_planning[robot.getID()] = True
                point_list, robot_data, next_point_data, graph_data, mask_data = self.getRobotState(robot, graph_list)
                action = self.model.getActionForTest(robot_data, next_point_data, graph_data, mask_data)
                next_point = point_list[int(action.detach()[0])]
                self.setRobotRoute(robot, next_point)

    def pathTraining(self):
        graph_list = self.getRobotsGraph()
        need_planning: List[bool] = [False for _ in range(len(self.robots))]
        # Collect data for all robots
        for robot in self.robots:
            # if self.has_last_value[robot.getID()] == True:
            #     print("Robot {} finished collecting data".format(robot.getID()))
            #     continue
            if self.needToPlanning(robot) == True:
                need_planning[robot.getID()] = True
                point_list, robot_data, next_point_data, graph_data, mask_data = self.getRobotState(robot, graph_list)
                action, value, log_prob, entropy = self.model.getAction(robot_data, next_point_data, graph_data, mask_data)
                if self.mini_batch_list[robot.getID()].mini_batch_id == self.mini_batch_size and self.has_last_value[robot.getID()] == False:
                    self.mini_batch_list[robot.getID()].advantageEstimator(value, self.lam, self.gamma)
                    self.has_last_value[robot.getID()] = True
                else:
                    next_point = point_list[int(action.detach()[0])]
                    if self.has_last_value[robot.getID()] == False:
                        self.mini_batch_list[robot.getID()].update(robot_data, next_point_data, graph_data, mask_data, 
                                                                    action, value, log_prob, entropy)
                    self.setRobotRoute(robot, next_point)
        # Calculate reward for all robot
        collision_list: List[bool] = [False for _ in range(len(self.robots))]
        for robot in self.robots:
            # print("--------------------------------")
            if need_planning[robot.getID()] == True and self.has_last_value[robot.getID()] == False:
                reward, done, collision = self.calculateReward(robot)
                collision_list[robot.getID()] = collision
                self.mini_batch_list[robot.getID()].updateRewardAndDoneState(reward, done)
        # for robot in self.robots:
        #     if collision_list[robot.getID()]:
        #         robot.setPose(robot.home.getCenterX(), robot.home.getCenterY(), math.pi/2)
        #         robot.route.clearRoute()
        #         robot.task.clearTask()
        #         robot.setState(FREE)
    
    def setRobotRoute(self, robot: Robot, next_point: List[float]):
        if robot.hasRoute() == False:
            if isSamePoint(robot.getGoal(), robot.getStartCenter()):
                robot.setRoute(Route(TO_START, [robot.getGraphID(), int(next_point[0])], 
                                    np.array([[robot.getX(), robot.getY()], [next_point[1], next_point[2]]])))
            if isSamePoint(robot.getGoal(), robot.getTargetCenter()):
                robot.setRoute(Route(TO_TARGET, [robot.getGraphID(), int(next_point[0])], 
                                    np.array([[robot.getX(), robot.getY()], [next_point[1], next_point[2]]])))
        else:
            robot.updateRoute(int(next_point[0]), np.array(next_point[1:]))
    
    def calculateReward(self, robot: Robot):
        reward: float = 0.0
        done: bool = False
        next_point = robot.getRouteCoord(-1)
        prev_point = robot.getRouteCoord(-2)
        # Goal reaching reward
        if isSamePoint(next_point, robot.getGoal()): 
            # print("Robot {} is reached goal with goal is ({}, {})".format(robot.getID(), robot.getGoalX(), robot.getGoalY()))
            done = True
            reward += 2.0
        # Step penalty
        if isSamePoint(next_point, prev_point):
            reward -= 0.05
        else:
            reward -= EuclidDistance(prev_point, next_point)/100
        # Collision penalty
        collision = False
        for other in self.robots:
            if other.getID() == robot.getID(): continue
            if other.hasRoute() == False:
                if isSamePoint(next_point, other.getPosition()):
                    done = True
                    # collision = True
                    reward -= 0.01
                    break
            else:
                other_angle = math.atan2(other.getRouteY(-1) - other.getRouteY(-2), other.getRouteX(-1) - other.getRouteX(-2))
                robot_angle = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
                if abs(calculateDifferenceOrientation(other_angle, robot_angle)) >= math.pi - deg2rad(5):
                    if isSamePoint(next_point, other.getRouteCoord(-1)):
                        # print("Robot {} and {} is in opposite directions!".format(robot.getID(), other.getID()))
                        done = True
                        collision = True
                        reward -= 1.0
                        break
        # Oscillation penalty
        # for i in range(len(robot.getRouteCoords())-1, -1, -1):
        #     if isSamePoint(next_point, robot.getRouteCoord(i)):
        #         reward -= 0.05
        #         break
        return torch.tensor(reward, dtype= torch.float32).to(device), torch.tensor(done, dtype= torch.float32).to(device), collision
        
    def getRobotState(self, robot: Robot, graph_list: List[List[List[float]]]):
        # Local graph data from neighbors of robot
        graph_data = self.getLocalGraphFromNeighbors(robot, graph_list)
        # Next point data of robot
        point_list, next_point_data, mask_data = self.getNextPointDataOfRobot(robot)
        # Robot data
        robot_data = self.getRobotData(robot)
        return point_list, robot_data.to(device), next_point_data.to(device), graph_data.to(device), mask_data.to(device)
    
    def loadModel(self):
        if not os.path.exists(os.path.join(self.model_folder, "model.pth")):
            torch.save(self.model.state_dict(), os.path.join(self.model_folder, "model.pth"))
            # print("Save initial planning model!")
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.model_folder, "model.pth"), map_location= device))
            # print("Load planning model!")
        
        self.best_all_reward_model = copy.deepcopy(self.model)
        self.best_local_reward_model = copy.deepcopy(self.model)
        self.best_loss_model = copy.deepcopy(self.model)
    
    def setModel(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location= device))
    def saveModel(self):
        torch.save(self.model.state_dict(), os.path.join(self.model_folder, "model.pth"))
    def saveBestLossModel(self):
        torch.save(self.best_loss_model.state_dict(), os.path.join(self.model_folder, "best_loss_model.pth"))
    def saveBestAllRewardModel(self):
        torch.save(self.best_all_reward_model.state_dict(), os.path.join(self.model_folder, "best_all_reward_model.pth"))
    def saveBestLocalRewardModel(self):
        torch.save(self.best_local_reward_model.state_dict(), os.path.join(self.model_folder, "best_local_reward_model.pth"))
    
    def getRobotData(self, robot: Robot):
        robot_data: List[float] = []
        if robot.hasRoute() == False:
            angle = robot.getTheta()
            robot_data: List[float] = [robot.getX(), robot.getY(), angle, robot.getGoalX(), robot.getGoalY()]
        else:
            length = robot.getRouteLength()
            angle = math.atan2(robot.getRouteY(-1) - robot.getRouteY(length-2), robot.getRouteX(-1) - robot.getRouteX(length-2))
            robot_data: List[float] = [robot.getRouteX(-1), robot.getRouteY(-1), angle, robot.getGoalX(), robot.getGoalY()]
        return torch.tensor(robot_data, dtype=torch.float32).reshape(1, -1)
    
    def getNextPointDataOfRobot(self, robot: Robot):
        center_id: int = robot.getGraphID()
        if robot.hasRoute() == True: center_id = robot.getRouteVertex(-1)
        
        return self.getNextPointOfVertex(center_id, self.planning_graph.getVertex(center_id).getCenter(), robot.getGoal())
        
    def getNextPointOfVertex(self, center_id: int, center_pos: np.ndarray, goal_pos: np.ndarray):
        up_point: List[float] = []
        down_point: List[float] = []
        right_point: List[float] = []
        left_point: List[float] = []
        for neighbor_id in self.planning_graph.getNeighbor(center_id):
            if len(self.planning_graph.getVertex(neighbor_id).getNeighbors()) == 0:
                continue
            x_neighbor = self.planning_graph.getVertex(neighbor_id).getCenterX()
            y_neighbor =  self.planning_graph.getVertex(neighbor_id).getCenterY()
            dist_x = x_neighbor - center_pos[0]
            dist_y = y_neighbor - center_pos[1]
            dist_from_center = math.hypot(dist_x, dist_y)
            dist_to_goal = ManhattanDistance(self.planning_graph.getVertex(neighbor_id).getCenter(), goal_pos)
            
            if abs(dist_x) > MAX_SAME_DIST and abs(dist_y) < MAX_SAME_DIST :
                if dist_x > 0:
                    up_point = [x_neighbor, y_neighbor, dist_from_center, dist_to_goal, neighbor_id]
                if dist_x < 0:
                    down_point = [x_neighbor, y_neighbor, dist_from_center, dist_to_goal, neighbor_id]
            elif abs(dist_x) < MAX_SAME_DIST and abs(dist_y) > MAX_SAME_DIST:
                if dist_y > 0:
                    right_point = [x_neighbor, y_neighbor, dist_from_center, dist_to_goal, neighbor_id]
                if dist_y < 0:
                    left_point = [x_neighbor, y_neighbor, dist_from_center, dist_to_goal, neighbor_id]
        next_point_data: List[List[float]] = []
        point_list:  List[List[float]] = []
        mask_list: List[bool] = []
        dist_to_goal = ManhattanDistance(center_pos, goal_pos)
        self.addPointToNextPointOfVertex(up_point, next_point_data, point_list, mask_list, center_id, center_pos, dist_to_goal)
        self.addPointToNextPointOfVertex(down_point, next_point_data, point_list, mask_list, center_id, center_pos, dist_to_goal)
        self.addPointToNextPointOfVertex(right_point, next_point_data, point_list, mask_list, center_id, center_pos, dist_to_goal)
        self.addPointToNextPointOfVertex(left_point, next_point_data, point_list, mask_list, center_id, center_pos, dist_to_goal)
        
        return point_list, torch.tensor(next_point_data, dtype=torch.float32).reshape(1, len(next_point_data), len(next_point_data[0])), torch.tensor(mask_list, dtype=torch.float32).reshape(1, len(mask_list))
    
    def addPointToNextPointOfVertex(self, point_data: List[float], next_point_data: List[List[float]],
                                    point_list: List[List[float]], mask_list: List[bool], 
                                    center_id: int, center_pos: np.ndarray, dist_to_goal: float):
        if len(point_data) == 0:
            next_point_data.append([center_pos[0], center_pos[1], 0.0, dist_to_goal])
            point_list.append([center_id, center_pos[0], center_pos[1]])
            mask_list.append(False)
        else:
            next_point_data.append(point_data[0:-1])
            point_list.append([point_data[-1], point_data[0], point_data[1]])
            mask_list.append(True)
            
    def getLocalGraphFromNeighbors(self, robot: Robot, graph_list: List[List[List[float]]]) -> torch.Tensor:
        # Calculate distance list
        distance_list = []
        for other in self.robots:
            if other.getID() == robot.getID():
                continue
            distance_list.append([other.getID(), EuclidDistance(other.getPosition(), robot.getPosition())])
        sorted_distance_list = sorted(distance_list, key=lambda distance: distance[1])
        # Get num_neighbor graph of robot
        graph_data: List[List[List[float]]] = [graph_list[robot.getID()]]
        for i in range(self.num_neighbor):
            graph_data.append(graph_list[sorted_distance_list[i][0]])
        graph_data_torch = torch.tensor(graph_data, dtype=torch.float32).reshape(1, len(graph_data), len(graph_data[0]), len(graph_data[0][0]))
        
        return graph_data_torch
        
    def getRobotsGraph(self):
        graph_list: List[List[List[float]]] = []
        for robot in self.robots:
            graph_list.append(self.getRobotGraph(robot))
        
        return graph_list

    def needToPlanning(self, robot: Robot):
        if robot.hasTask() == True and robot.hasRoute() == False:
            return True
        if robot.hasRoute() == True and EuclidDistance(robot.getRouteCoord(-1), robot.getGoal()) > MAX_SAME_DIST:
            return True
        return False
    
    def getRobotGraph(self, robot: Robot):
        graph: List[List[float]] = []
        if robot.hasRoute() == False and robot.hasTask() == True:
            self.addRobotToGraph(robot)
            dist_to_goal = EuclidDistance(robot.getPosition(), robot.getGoal())
            # Add edge involved in goal
            graph.append([CENTER_GRAPH, robot.getX(), robot.getY(), GOAL_NEIGHBOR, 
                        robot.getGoalX(), robot.getGoalY(), dist_to_goal, robot.max_speed])
            # Add edge involved in neighbors node
            self.addNeighborToRobotGraph(robot.getGraphID(), robot.getX(), robot.getY(), graph)
        if robot.hasRoute() == False and robot.hasTask() == False:
            self.addRobotToGraph(robot)
            # Add edge involved in goal
            graph.append([CENTER_GRAPH, robot.getX(), robot.getY(), GOAL_NEIGHBOR, 
                        robot.getX(), robot.getY(), 0.0, 0.0])
            # Add edge involved in neighbors node
            self.addNeighborToRobotGraph(robot.getGraphID(), robot.getX(), robot.getY(), graph)
        if robot.hasRoute() == True:
            center_id = robot.getRouteVertex(-1)
            dist_to_goal = EuclidDistance(robot.getRouteCoord(-1), robot.getGoal())
            # Add edge involved in goal
            graph.append([CENTER_GRAPH, robot.getRouteX(-1), robot.getRouteY(-1), GOAL_NEIGHBOR, 
                        robot.getGoalX(), robot.getGoalY(), dist_to_goal, robot.max_speed])
            # Add edge involved in neighbors node
            self.addNeighborToRobotGraph(center_id,  robot.getRouteX(-1), robot.getRouteY(-1), graph)
        return graph
    
    def addNeighborToRobotGraph(self, global_graph_id: int, x: float, y: float, graph: List[List[float]]):
        # Add right neighbor
        up_neighbor: List[float] = []
        down_neighbor: List[float] = []
        right_neighbor: List[float] = []
        left_neighbor: List[float] = []
        for neighbor_id in self.planning_graph.getNeighbor(global_graph_id):
            dist_x = self.planning_graph.getVertex(neighbor_id).getCenterX() - x
            dist_y = self.planning_graph.getVertex(neighbor_id).getCenterY() - y
            dist = EuclidDistance(np.array([x, y]), self.planning_graph.getVertex(neighbor_id).getCenter())
            speed = self.robots[0].max_speed
            if dist <= MAX_SAME_DIST:
                speed = 0.0
            if abs(dist_x) > MAX_SAME_DIST and abs(dist_y) < MAX_SAME_DIST:
                if dist_x > 0:
                    up_neighbor = [CENTER_GRAPH, x, y, FRONT_NEIGHBOR, 
                                    self.planning_graph.getVertex(neighbor_id).getCenterX(), 
                                    self.planning_graph.getVertex(neighbor_id).getCenterY(), dist, speed]
                if dist_x < 0:
                    down_neighbor = [CENTER_GRAPH, x, y, BACK_NEIGHBOR, 
                                    self.planning_graph.getVertex(neighbor_id).getCenterX(), 
                                    self.planning_graph.getVertex(neighbor_id).getCenterY(), dist, speed]
            elif abs(dist_x) < MAX_SAME_DIST and abs(dist_y) > MAX_SAME_DIST:
                if dist_y > 0:
                    right_neighbor = [CENTER_GRAPH, x, y, RIGHT_NEIGHBOR, 
                                    self.planning_graph.getVertex(neighbor_id).getCenterX(), 
                                    self.planning_graph.getVertex(neighbor_id).getCenterY(), dist, speed]
                if dist_y < 0:
                    left_neighbor = [CENTER_GRAPH, x, y, LEFT_NEIGHBOR, 
                                    self.planning_graph.getVertex(neighbor_id).getCenterX(), 
                                    self.planning_graph.getVertex(neighbor_id).getCenterY(), dist, speed]
        if len(up_neighbor) == 0:
            graph.append([CENTER_GRAPH, x, y, FRONT_NEIGHBOR, x, y, 0.0, 0.0])
        else:
            graph.append(up_neighbor)
        if len(down_neighbor) == 0:
            graph.append([CENTER_GRAPH, x, y, BACK_NEIGHBOR, x, y, 0.0, 0.0])
        else:
            graph.append(down_neighbor)
        if len(right_neighbor) == 0:
            graph.append([CENTER_GRAPH, x, y, RIGHT_NEIGHBOR, x, y, 0.0, 0.0])
        else:
            graph.append(right_neighbor)
        if len(left_neighbor) == 0:
            graph.append([CENTER_GRAPH, x, y, LEFT_NEIGHBOR, x, y, 0.0, 0.0])
        else:
            graph.append(left_neighbor)
        
    def addRobotToGraph(self, robot: Robot):
        self.planning_graph.addVertex(ROBOT_VERTEX, robot.getPose()[0:2])
        robot.setGraphID(self.planning_graph.getVertex(-1).getID())
        for zone in self.planning_graph.getZones():
            condition1 = zone.getCenterX() - zone.length/2 < robot.getX() < zone.getCenterX() + zone.length/2 
            condition2 = zone.getCenterY() - zone.width/2 <  robot.getY() < zone.getCenterY() + zone.width/2 
            if condition1 and condition2:
                self.findNeighborsOfRobot(robot, zone)
                break
            
    def findNeighborOfZone(self, zone: GraphZone):
        zone_list: List[GraphZone] = []
        zone_list.append(zone)
        row_ids = [zone.getRowID() - 1, zone.getRowID(), zone.getRowID(), zone.getRowID() + 1]
        col_ids = [zone.getColID(), zone.getColID() - 1, zone.getColID() + 1, zone.getColID()]
        for row, col in zip(row_ids, col_ids):
            if 0 <= row < self.planning_graph.getNumRow() and 0 <= col < self.planning_graph.getNumCol():
                zone_list.append(self.planning_graph.getZone(row + col * self.planning_graph.getNumCol()))
        return zone_list
    
    def findNeighborsOfRobot(self, robot: Robot, zone: GraphZone):        
        up_id:int = -1
        min_up_dist: float = 100
        down_id: int = -1
        max_down_dist: float = -100
        right_id:int = -1
        min_right_dist: float = 100
        left_id: int = -1
        max_left_dist: float = -100
        # Find neighbors
        zone_list: List[GraphZone] = self.findNeighborOfZone(zone)
        for zone in zone_list:
            for id in zone.getVertices():
                dist_x = self.planning_graph.getVertex(id).getCenterX() - robot.getX()
                dist_y = self.planning_graph.getVertex(id).getCenterY() - robot.getY()
                if abs(dist_x) < MAX_SAME_DIST and abs(dist_y) < MAX_SAME_DIST:
                    for neighbor in self.planning_graph.getVertex(id).getNeighbors():
                        self.planning_graph.addEdge(robot.getGraphID(), neighbor)
                    return
                if self.planning_graph.getVertex(id).getType() == LINE_VERTEX:
                    if abs(dist_x) > MAX_SAME_DIST and abs(dist_y) < MAX_SAME_DIST:
                        if dist_x > 0 and dist_x < min_up_dist:
                            up_id = id
                            min_up_dist = dist_x
                        if dist_x < 0 and dist_x > max_down_dist:
                            down_id = id
                            max_down_dist = dist_x
                    elif abs(dist_x) < MAX_SAME_DIST and abs(dist_y) > MAX_SAME_DIST:
                        if dist_y > 0 and dist_y < min_right_dist:
                            right_id = id
                            min_right_dist = dist_y
                        if dist_y < 0 and dist_y > max_left_dist:
                            left_id = id
                            max_left_dist = dist_y
        if up_id != -1:
            if self.planning_graph.getVertex(up_id).getHorizontalDirect() == POSITIVE_DIRECTED:
                self.planning_graph.addEdge(robot.getGraphID(), up_id)
            elif self.planning_graph.getVertex(up_id).getHorizontalDirect() == UNDIRECTED:
                self.planning_graph.addEdge(robot.getGraphID(), up_id)
                
        if down_id != -1:
            if self.planning_graph.getVertex(down_id).getHorizontalDirect() == NEGATIVE_DIRECTED:
                self.planning_graph.addEdge(robot.getGraphID(), down_id)
            elif self.planning_graph.getVertex(down_id).getHorizontalDirect() == UNDIRECTED:
                self.planning_graph.addEdge(robot.getGraphID(), down_id)
        
        if right_id != -1:
            if self.planning_graph.getVertex(right_id).getVerticalDirect() == POSITIVE_DIRECTED:
                self.planning_graph.addEdge(robot.getGraphID(), right_id)
            elif self.planning_graph.getVertex(right_id).getVerticalDirect() == UNDIRECTED:
                self.planning_graph.addEdge(robot.getGraphID(), right_id)
                
        if left_id != -1:
            if self.planning_graph.getVertex(left_id).getVerticalDirect() == NEGATIVE_DIRECTED:
                self.planning_graph.addEdge(robot.getGraphID(), left_id)
            elif self.planning_graph.getVertex(left_id).getVerticalDirect() == UNDIRECTED:
                self.planning_graph.addEdge(robot.getGraphID(), left_id)