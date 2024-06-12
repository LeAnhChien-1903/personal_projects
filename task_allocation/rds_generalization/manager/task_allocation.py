import copy
from typing import Callable
from env.env_chessboard import EnvironmentChessboard
from env.env import Environment
from utlis.utlis import *
from env.robot import Robot
from manager.task_generate import TaskGenerator
from models.attention_mechanism import AttentionNetwork
class BatchData:
    def __init__(self, batch_size: int, num_of_robot: int, num_of_task_queue: int, 
                selected_vec_size: int = 3, robot_vec_size: int = 5, task_vec_size: int = 9):
        self.batch_size: int = batch_size
        self.num_of_robot: int = num_of_robot
        self.num_of_task_queue: int = num_of_task_queue
        self.robot_vec_size: int = robot_vec_size
        self.task_vec_size: int = task_vec_size
        self.selected_vec_size: int = selected_vec_size
        self.batch_id: int = 0
        with torch.no_grad():
            self.selected_batch: torch.Tensor = torch.zeros(batch_size, selected_vec_size).to(device)
            self.robot_batch: torch.Tensor = torch.zeros(batch_size, num_of_robot, robot_vec_size).to(device)
            self.task_batch: torch.Tensor = torch.zeros(batch_size, num_of_task_queue, task_vec_size).to(device)
            self.action_batch: torch.Tensor = torch.zeros(batch_size).to(device)
            self.reward_batch: torch.Tensor = torch.zeros(batch_size).to(device)
            self.value_batch: torch.Tensor = torch.zeros(batch_size).to(device)
            self.log_prob_batch: torch.Tensor = torch.zeros(batch_size).to(device)
            self.entropy_batch: torch.Tensor = torch.zeros(batch_size).to(device)
            self.advantage_batch: torch.Tensor = torch.zeros(batch_size).to(device)
            self.return_batch: torch.Tensor = torch.zeros(batch_size).to(device)
            self.reward_state = [False for _ in range(batch_size)]
        
    def update(self, selected: torch.Tensor, robot: torch.Tensor, task: torch.Tensor, action: torch.Tensor,
                value: torch.Tensor, log_prob: torch.Tensor, entropy: torch.Tensor):
        with torch.no_grad():
            self.selected_batch[self.batch_id] = selected
            self.robot_batch[self.batch_id] = robot
            self.task_batch[self.batch_id] = task
            self.action_batch[self.batch_id] = action
            self.value_batch[self.batch_id] = value
            self.log_prob_batch[self.batch_id] = log_prob
            self.entropy_batch[self.batch_id] = entropy
        self.batch_id += 1
        
    def updateReward(self, reward: torch.Tensor, idx: int):
        with torch.no_grad():
            self.reward_batch[idx] = reward
            self.reward_state[idx] = True
        
    def advantageEstimator(self, last_value:torch.Tensor, lam: float = 0.95, gamma: float = 0.99):
        with torch.no_grad():
            last_gae_lam = torch.zeros(1).to(device)
            for t in reversed(range(self.batch_size)):
                if t == self.batch_size - 1:
                    next_value = last_value
                else:
                    next_value = self.value_batch[t+1]
                delta = self.reward_batch[t] + gamma * next_value - self.value_batch[t]
                self.advantage_batch[t] = delta + gamma * lam * last_gae_lam
                last_gae_lam = delta + gamma * lam * last_gae_lam
            
            self.return_batch = self.advantage_batch + self.value_batch
        
    def clear(self):
        self.batch_id: int = 0
        with torch.no_grad():
            self.selected_batch: torch.Tensor = torch.zeros(self.batch_size, self.selected_vec_size).to(device)
            self.robot_batch: torch.Tensor = torch.zeros(self.batch_size, self.num_of_robot, self.robot_vec_size).to(device)
            self.task_batch: torch.Tensor = torch.zeros(self.batch_size, self.num_of_task_queue, self.task_vec_size).to(device)
            self.action_batch: torch.Tensor = torch.zeros(self.batch_size).to(device)
            self.reward_batch: torch.Tensor = torch.zeros(self.batch_size).to(device)
            self.value_batch: torch.Tensor = torch.zeros(self.batch_size).to(device)
            self.log_prob_batch: torch.Tensor = torch.zeros(self.batch_size).to(device)
            self.entropy_batch: torch.Tensor = torch.zeros(self.batch_size).to(device)
            self.advantage_batch: torch.Tensor = torch.zeros(self.batch_size).to(device)
            self.return_batch: torch.Tensor = torch.zeros(self.batch_size).to(device)
            self.reward_state = [False for _ in range(self.batch_size)]
        
class TaskAllocation:
    def __init__(self, robots: List[Robot], task_generator: TaskGenerator, original_graph: Graph, 
                model_folder: str, env: EnvironmentChessboard, num_task_test: int = 100):
        self.robots: List[Robot] = robots
        self.task_generator: TaskGenerator = task_generator
        self.original_graph = original_graph
        self.allocation_graph = original_graph.copy()
        self.model: AttentionNetwork = AttentionNetwork(selected_size=3, robot_size= 5, task_size=6).to(device= device)
        self.model_folder: str = model_folder
        self.loadModel()
        self.best_loss: float = float('inf')
        self.best_reward: float = -float('inf')
        self.has_last_value: bool = False
        self.last_task_assigned: bool = False
        self.clip_range: float = 0.1
        self.lr: float = 3e-4 # Learning rate
        self.update_interval: int = 128 # Update Interval while training
        self.num_epoch: int  = 16 # Epoch per iteration of PPO
        self.gamma: float = 0.99 # Discount factor
        self.lam: float = 0.95 # Lambda used in GAE (General Advantage Estimation)
        self.entropy_coef: float = 0.01 # Entropy coefficient
        self.value_coef: float = 0.0002 # Value function coefficient
        self.policy_coef: float = 0.02 # Policy function coefficient 
        self.last_value: torch.Tensor = torch.zeros(self.update_interval).to(device)
        self.batch: BatchData = BatchData(batch_size= self.update_interval, num_of_robot= len(robots), 
                                        num_of_task_queue= task_generator.num_task_in_queue,
                                        robot_vec_size= 5, task_vec_size= 6)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.num_task_test = num_task_test
        self.test_done = False
        self.test_times: List[float] = [0 for _ in range(self.num_task_test)]
        self.test_priorities: List[int] = [0 for _ in range(self.num_task_test)]
        self.test_state: List[bool] = [False for _ in range(self.num_task_test)]
        self.test_task_state: List[bool] = [False for _ in range(self.num_task_test)]
        self.test_counter: int = 0
        self.map_length = env.factory_map.map_length
        self.map_width = env.factory_map.map_width
        self.max_load = task_generator.max_load
        self.max_priority = task_generator.num_priority
        self.max_route_cost = self.map_length + self.map_width
        self.max_rest_time = (self.map_length + self.map_width)*0.1 * 1.25
        
    
    def nearestAStarTesting(self):
        if self.test_done == False:
            self.allocationNearest(self.calculateNearestAStarInput)
            return False, [], []
        else:
            test_times = self.test_times.copy()
            test_priorities = self.test_priorities.copy()
            self.test_times = [0 for _ in range(self.num_task_test)]
            self.test_priorities = [0 for _ in range(self.num_task_test)]
            self.test_state = [False for _ in range(self.num_task_test)]
            self.test_task_state = [False for _ in range(self.num_task_test)]
            self.test_counter = 0
            self.test_done = False
            self.last_task_assigned = False
            return True, test_times, test_priorities
    
    def nearestManhattanTesting(self):
        if self.test_done == False:
            self.allocationNearest(self.calculateNearestManhattanInput)
            return False, [], []
        else:
            test_times = self.test_times.copy()
            test_priorities = self.test_priorities.copy()
            self.test_times = [0 for _ in range(self.num_task_test)]
            self.test_priorities = [0 for _ in range(self.num_task_test)]
            self.test_state = [False for _ in range(self.num_task_test)]
            self.test_task_state = [False for _ in range(self.num_task_test)]
            self.test_counter = 0
            self.test_done = False
            self.last_task_assigned = False
            return True, test_times, test_priorities
    
    def AStarTesting(self):
        if self.test_done == False:
            self.allocationAStarTesting()
            return False, [], []
        else:
            test_times = self.test_times.copy()
            test_priorities = self.test_priorities.copy()
            self.test_times = [0 for _ in range(self.num_task_test)]
            self.test_priorities = [0 for _ in range(self.num_task_test)]
            self.test_state = [False for _ in range(self.num_task_test)]
            self.test_task_state = [False for _ in range(self.num_task_test)]
            self.test_counter = 0
            self.test_done = False
            self.last_task_assigned = False
            return True, test_times, test_priorities
        
    def training(self, iter: int, save_interval: int = 5) -> bool:
        if self.has_last_value == False:
            self.allocationTraining()
            return False
        else:
            sum_reward = torch.sum(self.batch.reward_batch).item()
            if self.best_reward < sum_reward:
                    print("\nBest reward at {} iteration: ".format(iter), round(sum_reward, 2))
                    self.best_reward = sum_reward
                    self.best_reward_model = copy.deepcopy(self.model)
                    self.saveBestRewardModel()
            for _ in range(self.num_epoch):
                loss, _ = self.calculateLoss()
                # Zero out the previously calculated gradients
                self.optimizer.zero_grad()
                # Calculate gradients
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                # Update parameters based on gradients
                self.optimizer.step()
            
            self.saveModel()
                
            self.has_last_value = False
            self.last_task_assigned = False
            self.batch.clear()
            return True

    def loadModel(self):
        if os.path.exists(os.path.join(self.model_folder, "model_seed_{}.pth".format(seed_value))):
        #     torch.save(self.model.state_dict(), os.path.join(self.model_folder, "model_seed_{}.pth".format(seed_value)))
        #     print("Save initial allocation model!")
        # else:
            self.model.load_state_dict(torch.load(os.path.join(self.model_folder, "model_seed_{}.pth".format(seed_value)), map_location= device))
            print("Load allocation model!")
        self.best_loss_model = copy.deepcopy(self.model)
        self.best_reward_model = copy.deepcopy(self.model)
    
    def setModel(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location= device))
    
    def saveModel(self):
        torch.save(self.model.state_dict(), os.path.join(self.model_folder, "model_seed_{}.pth".format(seed_value)))
    
    def saveBestLossModel(self):
        torch.save(self.best_loss_model.state_dict(), os.path.join(self.model_folder, "best_loss_model_seed_{}.pth".format(seed_value)))
    
    def saveBestRewardModel(self):
        torch.save(self.best_reward_model.state_dict(), os.path.join(self.model_folder, "best_reward_model_seed_{}.pth".format(seed_value)))
        
    def calculateLoss(self):
        normalized_advantage = self.normalize(self.batch.advantage_batch)
        value, log_prob, entropy = self.model.evaluateAction(self.batch.selected_batch, self.batch.robot_batch, 
                                                                self.batch.task_batch, self.batch.action_batch)
        policy_loss: torch.Tensor = self.calculatePolicyLoss(log_prob, normalized_advantage)
        entropy_bonus = entropy.mean()
        value_loss: torch.Tensor = self.calculateValueLoss(value)

        loss = (policy_loss + value_loss * self.value_coef - self.entropy_coef * entropy_bonus)
        approx_kl_divergence = 0.5 * ((self.batch.log_prob_batch - log_prob) ** 2).mean()

        return loss, approx_kl_divergence
    
    def calculatePolicyLoss(self, log_prob: torch.Tensor, advantage: torch.Tensor):
        ratio = torch.exp(log_prob - self.batch.log_prob_batch)
        clipped_ratio = ratio.clamp(min = 1.0 - self.clip_range,
                                    max = 1.0 + self.clip_range)
        policy_reward = torch.min(ratio * advantage, clipped_ratio * advantage)
        
        return -policy_reward.mean()
    
    def calculateValueLoss(self, value: torch.Tensor):
        clipped_value = self.batch.value_batch + (value - self.batch.value_batch).clamp(min=-self.clip_range, max=self.clip_range)
        vf_loss = torch.max((value - self.batch.value_batch) ** 2, (clipped_value - self.batch.value_batch) ** 2)
        return 0.5 * vf_loss.mean()
    
    @staticmethod
    def normalize(adv: torch.Tensor):
        return (adv - adv.mean()) / (adv.std() - 1e-8)
    
    def allocationTraining(self):
        for robot in self.robots:
            if robot.done_count_reward and self.batch.reward_state[robot.getRewardIndex()] == False:
                reward = - robot.getRewardAllocation() # + 2 * float(robot.getTask().getPriority())
                reward_tensor = torch.tensor(reward, dtype= torch.float32).to(device)
                self.batch.updateReward(reward_tensor, robot.getRewardIndex())
                robot.done_count_reward = False
        if (False in self.batch.reward_state) == False:
            self.batch.advantageEstimator(self.last_value, self.lam, self.gamma)
            self.has_last_value = True
            return 
        for robot in self.robots:
            if robot.getState() == FREE:
                selected_data, robot_data, task_data = self.calculateAllocationState(robot)
                action, value, log_prob, entropy = self.model.getAction(selected_data, robot_data, task_data)
                selected_task = self.task_generator.getTask(int(action.detach()[0]), robot.getID())
                robot.setTask(selected_task)
                robot.setRoute(AStarPlanning(self.allocation_graph, TO_START, robot.getGraphID(), 
                                            selected_task.getStartID()))
                if self.last_task_assigned == False:
                    if self.batch.batch_id == self.batch.batch_size:
                        self.last_value = value
                        self.last_task_assigned = True
                        return
                    else:
                        robot.setRewardIndex(self.batch.batch_id)
                        self.batch.update(selected_data, robot_data, task_data, action, value, log_prob, entropy)
            
    def allocationAStarTesting(self):
        for robot in self.robots:
            if robot.done_count_reward and self.test_state[robot.getRewardIndex()] == False:
                self.test_times[robot.getRewardIndex()] = round(-robot.getRewardAllocation(), 2)
                self.test_priorities[robot.getRewardIndex()] = robot.getTask().getPriority()
                self.test_state[robot.getRewardIndex()] = True
                robot.done_count_reward = False
                robot.task_for_test = True
            if robot.target_done and robot.task_for_test and self.test_task_state[robot.getRewardIndex()] == False:
                robot.target_done = False
                robot.task_for_test = False
                self.test_task_state[robot.getRewardIndex()] = True
        if (False in self.test_task_state) == False:
            self.test_done = True
            return  
        for robot in self.robots:
            if robot.getState() == FREE:
                selected_data, robot_data, task_data = self.calculateAllocationState(robot)
                action = self.model.getActionForTest(selected_data, robot_data, task_data)
                selected_task = self.task_generator.getTask(int(action.detach()[0]), robot.getID())
                robot.setTask(selected_task)
                robot.setRoute(AStarPlanning(self.allocation_graph, TO_START, robot.getGraphID(), 
                                                        selected_task.getStartID()))
                if self.last_task_assigned == False:
                    if self.test_counter == self.num_task_test:
                        self.last_task_assigned = True
                        return
                    else:
                        robot.setRewardIndex(self.test_counter)
                        self.test_counter += 1
                        
    def allocationNearest(self, func: Callable):
        for robot in self.robots:
            if robot.done_count_reward and self.test_state[robot.getRewardIndex()] == False:
                self.test_times[robot.getRewardIndex()] = round(-robot.getRewardAllocation(), 2)
                self.test_priorities[robot.getRewardIndex()] = robot.getTask().getPriority()
                self.test_state[robot.getRewardIndex()] = True
                robot.done_count_reward = False
                robot.task_for_test = True
            if robot.target_done and robot.task_for_test and self.test_task_state[robot.getRewardIndex()] == False:
                robot.target_done = False
                robot.task_for_test = False
                self.test_task_state[robot.getRewardIndex()] = True
        if (False in self.test_task_state) == False:
            self.test_done = True
            return 
        for robot in self.robots:
            if robot.getState() == FREE:
                task_distances = func(robot)
                action = task_distances.index(min(task_distances))
                selected_task = self.task_generator.getTask(action, robot.getID())
                robot.setTask(selected_task)
                robot.setRoute(AStarPlanning(self.allocation_graph, TO_START, robot.getGraphID(), 
                                                        selected_task.getStartID()))
                if self.last_task_assigned == False:
                    if self.test_counter == self.num_task_test:
                        self.last_task_assigned = True
                        return
                    else:
                        robot.setRewardIndex(self.test_counter)
                        self.test_counter += 1
        
    def calculateAllocationState(self, selected_robot: Robot):
        task_queue: torch.Tensor = self.calculateTaskInput(selected_robot).to(device)
        selected_data: torch.Tensor = torch.tensor([selected_robot.getX()/self.map_length, 
                                                    selected_robot.getY()/self.map_width, 0.0], 
                                                    dtype= torch.float32).to(device)
        robot_data: torch.Tensor = self.calculateRobotInput().to(device)
        return selected_data.reshape(1, -1), robot_data, task_queue 
    
    def calculateNearestAStarInput(self, selected_robot: Robot):
        task_queue = self.task_generator.getTaskQueue()
        self.addRobotToGraph(selected_robot)
        task_distances: List[float] = []
        for task in task_queue:
            task_distances.append(AStarPlanningCost(self.allocation_graph, selected_robot.getGraphID(), 
                                                    task.getStartID()))
        
        return task_distances

    def calculateNearestManhattanInput(self, selected_robot: Robot):
        task_queue = self.task_generator.getTaskQueue()
        self.addRobotToGraph(selected_robot)
        task_distances: List[float] = []
        for task in task_queue:
            task_distances.append(ManhattanDistance(selected_robot.getPosition(), task.getStartCenter()))
        
        return task_distances
        
    def calculateTaskInput(self, selected_robot: Robot) -> torch.Tensor:
        task_queue = self.task_generator.getTaskQueue()
        self.addRobotToGraph(selected_robot)
        task_queue_mat: List[List[float]] = []
        for task in task_queue:
            task_queue_mat.append([task.getStartX()/self.map_length, task.getStartY()/self.map_width])
            task_queue_mat[-1].append(task.getTargetX()/self.map_length)
            task_queue_mat[-1].append(task.getTargetY()/self.map_width)
            # task_queue_mat[-1].append(task.getPriority()/self.max_priority)
            # task_queue_mat[-1].append(task.getType())
            # task_queue_mat[-1].append(task.getMass()/self.max_load)
            task_queue_mat[-1].append(AStarPlanningCost(self.allocation_graph, selected_robot.getGraphID(), 
                                                        task.getStartID())/self.max_route_cost)
            task_queue_mat[-1].append(task.getRouteCost()/self.max_route_cost)
        task_queue_torch: torch.Tensor = torch.tensor(task_queue_mat, dtype= torch.float32)
        return task_queue_torch.reshape(1, task_queue_torch.shape[0], task_queue_torch.shape[1])
    
    def calculateRobotInput(self) -> torch.Tensor:
        robot_mat: List[List[float]] = []
        for robot in self.robots:
            if robot.hasTask() == True:
                robot_mat.append([robot.getX()/self.map_length, robot.getY()/self.map_width, 
                                robot.getGoalX()/self.map_length, robot.getGoalY()/self.map_width,
                                self.calculateRestTime(robot)/self.max_rest_time])
            else:
                robot_mat.append([robot.getX()/self.map_length, robot.getY()/self.map_width, 
                                robot.getX()/self.map_length, robot.getY()/self.map_width,
                                0.0])
        robot_torch: torch.Tensor = torch.tensor(robot_mat, dtype= torch.float32)
        return robot_torch.reshape(1, robot_torch.shape[0], robot_torch.shape[1])
    
    def calculateRestTime(self, robot: Robot) -> float:
        rest_time: float = 0.0
        if robot.hasRoute() == False:
            return rest_time
        else:
            if robot.getRouteType() == TO_START:
                current_id = robot.getNextPointIndex()
                route = robot.getRouteCoords()
                rest_time += EuclidDistance(robot.getPose(), route[current_id])
                for i in range(current_id, robot.getRouteCoords().shape[0] - 1):
                    rest_time += EuclidDistance(route[i], route[i+1])
                rest_time += robot.getWaitingTime()*2 + robot.getRouteCost()
            if robot.getRouteType() == TO_TARGET:
                current_id = robot.getNextPointIndex()
                route = robot.getRouteCoords()
                rest_time += EuclidDistance(robot.getPose(), route[current_id])
                for i in range(current_id, robot.getRouteCoords().shape[0] - 1):
                    rest_time += EuclidDistance(route[i], route[i+1])
                rest_time += robot.getWaitingTime()
            return rest_time + min(rest_time, abs(rest_time * np.random.normal(loc= 0.1, scale= 0.5)))
    
    def addRobotsToGraph(self):
        for robot in self.robots:
            self.addRobotToGraph(robot)

    def addRobotToGraph(self, robot: Robot):
        self.allocation_graph.addVertex(ROBOT_VERTEX, robot.getPose()[0:2])
        robot.setGraphID(self.allocation_graph.getVertex(-1).getID())
        for zone in self.original_graph.getZones():
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
            if 0 <= row < self.original_graph.getNumRow() and 0 <= col < self.original_graph.getNumCol():
                zone_list.append(self.original_graph.getZone(row * self.original_graph.getNumCol() + col))
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
                dist1 = self.original_graph.getVertex(id).getCenterX() - robot.getX()
                dist2 = self.original_graph.getVertex(id).getCenterY() - robot.getY()
                if abs(dist1) < MAX_SAME_DIST and abs(dist2) < MAX_SAME_DIST:
                    for neighbor in self.original_graph.getVertex(id).getNeighbors():
                        self.allocation_graph.addEdge(robot.getGraphID(), neighbor)
                    return
                if self.original_graph.getVertex(id).getType() == LINE_VERTEX:
                    if abs(dist1) > MAX_SAME_DIST and abs(dist2) < MAX_SAME_DIST:
                        if dist1 > 0 and dist1 < min_up_dist:
                            up_id = id
                            min_up_dist = dist1
                        if dist1 < 0 and dist1 > max_down_dist:
                            down_id = id
                            max_down_dist = dist1
                    elif abs(dist1) < MAX_SAME_DIST and abs(dist2) > MAX_SAME_DIST:
                        if dist2 > 0 and dist2 < min_right_dist:
                            right_id = id
                            min_right_dist = dist2
                        if dist2 < 0 and dist2 > max_left_dist:
                            left_id = id
                            max_left_dist = dist2
        if up_id != -1:
            if self.allocation_graph.getVertex(up_id).getHorizontalDirect() == POSITIVE_DIRECTED:
                self.allocation_graph.addEdge(robot.getGraphID(), up_id)
            elif self.allocation_graph.getVertex(up_id).getHorizontalDirect() == UNDIRECTED:
                self.allocation_graph.addEdge(robot.getGraphID(), up_id)
                
        if down_id != -1:
            if self.allocation_graph.getVertex(down_id).getHorizontalDirect() == NEGATIVE_DIRECTED:
                self.allocation_graph.addEdge(robot.getGraphID(), down_id)
            elif self.allocation_graph.getVertex(down_id).getHorizontalDirect() == UNDIRECTED:
                self.allocation_graph.addEdge(robot.getGraphID(), down_id)
        
        if right_id != -1:
            if self.allocation_graph.getVertex(right_id).getVerticalDirect() == POSITIVE_DIRECTED:
                self.allocation_graph.addEdge(robot.getGraphID(), right_id)
            elif self.allocation_graph.getVertex(right_id).getVerticalDirect() == UNDIRECTED:
                self.allocation_graph.addEdge(robot.getGraphID(), right_id)
                
        if left_id != -1:
            if self.allocation_graph.getVertex(left_id).getVerticalDirect() == NEGATIVE_DIRECTED:
                self.allocation_graph.addEdge(robot.getGraphID(), left_id)
            elif self.allocation_graph.getVertex(left_id).getVerticalDirect() == UNDIRECTED:
                self.allocation_graph.addEdge(robot.getGraphID(), left_id)
        
    def resetAllocationGraph(self):
        self.allocation_graph = self.original_graph.copy()