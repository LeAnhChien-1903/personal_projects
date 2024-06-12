from utlis.utlis import *
from env.robot import Robot
class TaskGenerator:
    def __init__(self, max_task_gen: int, num_task_in_queue: int, num_priority: int, num_type: int, 
                min_load: float, max_load: float, graph:Graph):
        self.max_task_gen = max_task_gen
        self.graph = graph.copy()
        self.num_task_in_queue: int = num_task_in_queue
        self.num_priority: int = num_priority
        self.num_type: int = num_type
        self.min_load: float = min_load
        self.max_load: float = max_load
        self.working_task_ids: Dict = {}
        self.extractZonePoints(graph)
        self.task_queue: List[Task] = []
        
        self.generateTaskQueue()
        
    
    def getTaskQueue(self):
        return self.task_queue.copy()
    
    def getTask(self, task_id: int, robot_id: int):
        task = self.task_queue[task_id].copy()
        self.working_task_ids[robot_id] = [task.getStartID(), task.getTargetID()]
        del self.task_queue[task_id]
        self.addTask()
        return task 
    
    def addTask(self):
        ids_has_used: List[int] = []
        
        for task in list(self.working_task_ids.values()):
            ids_has_used.append(task[0])
            ids_has_used.append(task[1])
        
        for task in self.task_queue:
            ids_has_used.append(task.getStartID())
            ids_has_used.append(task.getTargetID())
        
        for _ in range(100):
            epsilon = random.random()
            type = random.randint(0, self.num_type - 1)
            priority = random.randint(1, self.num_priority)
            mass = random.uniform(self.min_load, self.max_load)
            if epsilon >= 0.75:
                start = self.working_vertices[random.randint(0, len(self.working_vertices) - 1)]
                target = self.storage_vertices[random.randint(0, len(self.storage_vertices) - 1)]
                if EuclidDistance(start.getCenter(),  target.getCenter()) > 10.0:
                    if len(self.task_queue) == 0:
                        self.task_queue.append(Task(start, target, type, priority, mass))
                        self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
                                                                    self.task_queue[-1].getTargetID()))
                        return
                    else:
                        if start.getID() not in ids_has_used and target.getID() not in ids_has_used:
                            self.task_queue.append(Task(start, target, type, priority, mass))
                            self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
                                                                    self.task_queue[-1].getTargetID()))
                            return
            elif 0.5 <= epsilon < 0.75:
                id_list: np.ndarray = np.arange(0, len(self.working_vertices))
                start_id = np.random.choice(id_list)
                target_id = np.random.choice(np.delete(id_list, start_id))
                if EuclidDistance(self.working_vertices[start_id].getCenter(),  self.working_vertices[target_id].getCenter()) > 10.0:
                    if len(self.task_queue) == 0:
                        self.task_queue.append(Task(self.working_vertices[start_id], self.working_vertices[target_id], type, priority, mass))
                        self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
                                                                self.task_queue[-1].getTargetID()))
                        return
                    else:
                        if self.working_vertices[start_id].getID() not in ids_has_used and self.working_vertices[target_id].getID() not in ids_has_used:
                            self.task_queue.append(Task(self.working_vertices[start_id], self.working_vertices[target_id], type, priority, mass))
                            self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
                                                                    self.task_queue[-1].getTargetID()))
                            return
            elif 0.25 <= epsilon < 0.5:
                id_list: np.ndarray = np.arange(0, len(self.storage_vertices))
                start_id = np.random.choice(id_list)
                target_id = np.random.choice(np.delete(id_list, start_id))
                if EuclidDistance(self.storage_vertices[start_id].getCenter(),  self.storage_vertices[target_id].getCenter()) > 10.0:
                    if len(self.task_queue) == 0:
                        self.task_queue.append(Task(self.storage_vertices[start_id], self.storage_vertices[target_id], type, priority, mass))
                        self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
                                                                    self.task_queue[-1].getTargetID()))
                        return
                    else:
                        if self.storage_vertices[start_id].getID() not in ids_has_used and self.storage_vertices[target_id] not in ids_has_used:
                            self.task_queue.append(Task(self.storage_vertices[start_id], self.storage_vertices[target_id], type, priority, mass))
                            self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
                                                                        self.task_queue[-1].getTargetID()))
                            return
            else:
                start = self.working_vertices[random.randint(0, len(self.working_vertices) - 1)]
                target = self.storage_vertices[random.randint(0, len(self.storage_vertices) - 1)]
                if EuclidDistance(start.getCenter(),  target.getCenter()) > 10.0:
                    if len(self.task_queue) == 0:
                        self.task_queue.append(Task(start, target, type, priority, mass))
                        self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
                                                                    self.task_queue[-1].getTargetID()))
                        return
                    else:
                        if start.getID() not in ids_has_used and target.getID() not in ids_has_used:
                            self.task_queue.append(Task(start, target, type, priority, mass))
                            self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
                                                                        self.task_queue[-1].getTargetID()))
                            return
        
        for _ in range(100):
            epsilon = random.random()
            type = random.randint(0, self.num_type - 1)
            priority = random.randint(1, self.num_priority)
            mass = random.uniform(self.min_load, self.max_load)
            if epsilon >= 0.75:
                start = self.working_vertices[random.randint(0, len(self.working_vertices) - 1)]
                target = self.storage_vertices[random.randint(0, len(self.storage_vertices) - 1)]
                if EuclidDistance(start.getCenter(),  target.getCenter()) > 10.0:
                    if len(self.task_queue) == 0:
                        self.task_queue.append(Task(start, target, type, priority, mass))
                        self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
                                                                    self.task_queue[-1].getTargetID()))
                        return
                    else:
                        self.task_queue.append(Task(start, target, type, priority, mass))
                        self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
                                                                self.task_queue[-1].getTargetID()))
                        return
            elif 0.5 <= epsilon < 0.75:
                id_list: np.ndarray = np.arange(0, len(self.working_vertices))
                start_id = np.random.choice(id_list)
                target_id = np.random.choice(np.delete(id_list, start_id))
                if EuclidDistance(self.working_vertices[start_id].getCenter(),  self.working_vertices[target_id].getCenter()) > 10.0:
                    if len(self.task_queue) == 0:
                        self.task_queue.append(Task(self.working_vertices[start_id], self.working_vertices[target_id], type, priority, mass))
                        self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
                                                                self.task_queue[-1].getTargetID()))
                        return
                    else:
                        self.task_queue.append(Task(self.working_vertices[start_id], self.working_vertices[target_id], type, priority, mass))
                        self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
                                                                self.task_queue[-1].getTargetID()))
                        return
            elif 0.25 <= epsilon < 0.5:
                id_list: np.ndarray = np.arange(0, len(self.storage_vertices))
                start_id = np.random.choice(id_list)
                target_id = np.random.choice(np.delete(id_list, start_id))
                if EuclidDistance(self.storage_vertices[start_id].getCenter(),  self.storage_vertices[target_id].getCenter()) > 10.0:
                    if len(self.task_queue) == 0:
                        self.task_queue.append(Task(self.storage_vertices[start_id], self.storage_vertices[target_id], type, priority, mass))
                        self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
                                                                    self.task_queue[-1].getTargetID()))
                        return
                    else:
                        self.task_queue.append(Task(self.storage_vertices[start_id], self.storage_vertices[target_id], type, priority, mass))
                        self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
                                                                    self.task_queue[-1].getTargetID()))
                        return
            else:
                start = self.working_vertices[random.randint(0, len(self.working_vertices) - 1)]
                target = self.storage_vertices[random.randint(0, len(self.storage_vertices) - 1)]
                if EuclidDistance(start.getCenter(),  target.getCenter()) > 10.0:
                    if len(self.task_queue) == 0:
                        self.task_queue.append(Task(start, target, type, priority, mass))
                        self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
                                                                    self.task_queue[-1].getTargetID()))
                        return
                    else:
                        self.task_queue.append(Task(start, target, type, priority, mass))
                        self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
                                                                    self.task_queue[-1].getTargetID()))
                        return
        
    
    def extractZonePoints(self, graph:Graph):
        self.working_vertices: List[Vertex] = []
        self.storage_vertices: List[Vertex] = []
        for vertex in graph.getVertices():
            if vertex.getType() == WORKING_VERTEX:
                self.working_vertices.append(vertex)
            elif vertex.getType() == STORAGE_VERTEX:
                self.storage_vertices.append(vertex)
    
    def generateTaskQueue(self):
        for _ in range(self.num_task_in_queue):
            self.addTask()

# class TaskGenFromData:
#     def __init__(self, robots: List[Robot], task_data_path: str, num_task_in_queue: int, graph: Graph, mode: bool = TEST_MODE):
#         self.task_data: np.ndarray = np.loadtxt(task_data_path)
#         self.task_data_origin: np.ndarray = np.loadtxt(task_data_path)
#         self.task_id: int = 0
#         self.num_task_in_queue = num_task_in_queue
#         self.graph: Graph = graph
#         self.task_queue: List[Task] = []
#         self.mode: bool = mode
#         self.robots: List[Robot] = robots
#         self.generateTaskQueue()
    
#     def getTaskQueue(self): return self.task_queue.copy()
    
#     def getTask(self, id: int) -> Task:
#         task = self.task_queue[id]
#         del self.task_queue[id]
#         for i in range(id, len(self.task_queue)):
#             self.task_queue[i].setID(i)
#         self.addTask()
        
#         return task 
    
#     def generateTaskQueue(self):
#         self.task_data = self.task_data_origin.copy()
#         self.task_id = 0
#         for _ in range(self.num_task_in_queue):
#             self.addTask()
#     def addTask(self):
#         start_list = []
#         for robot in self.robots:
#             if robot.hasTask() == True and robot.hasRoute() == True:
#                 if robot.getRouteType() == TO_START:
#                     start_list.append(robot.getTask().getStartID())
#                 elif robot.getRouteType() == TO_TARGET:
#                     start_list.append(robot.getTask().getTargetID())
#         if len(self.task_queue) != 0:
#             for task in self.task_queue:
#                 start_list.append(task.getStartID())
                
#         if self.mode == TEST_MODE:
#             if self.task_id >= self.task_data_origin.shape[0] - 1:
#                 return
#             if len(self.task_queue) == 0:
#                 start: Vertex = self.graph.getVertex(int(self.task_data[self.task_id, 0]))
#                 target: Vertex = self.graph.getVertex(int(self.task_data[self.task_id, 1]))
#                 self.task_queue.append(Task(start, target, int(self.task_data[self.task_id, 2]), 
#                                             int(self.task_data[self.task_id, 3]), self.task_data[self.task_id, 4]))
#                 self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
#                                                             self.task_queue[-1].getTargetID()))
#                 self.task_data = np.delete(self.task_data, 0, axis= 0)
#             else:
#                 index = 0
#                 while True:
#                     if index < self.task_data.shape[0] - 1:
#                         start: Vertex = self.graph.getVertex(int(self.task_data[index, 0]))
#                         if start.getID() in start_list: 
#                             index += 1
#                         else:
#                             target: Vertex = self.graph.getVertex(int(self.task_data[index, 1]))
#                             self.task_queue.append(Task(start, target, int(self.task_data[index, 2]), 
#                                                     int(self.task_data[index, 3]), self.task_data[index, 4]))
#                             self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
#                                                                     self.task_queue[-1].getTargetID()))
#                             self.task_data = np.delete(self.task_data, index, axis= 0)
#                             break
#                     else:
#                         start: Vertex = self.graph.getVertex(int(self.task_data[0, 0]))
#                         target: Vertex = self.graph.getVertex(int(self.task_data[0, 1]))
#                         self.task_queue.append(Task(start, target, int(self.task_data[0, 2]), 
#                                                 int(self.task_data[0, 3]), self.task_data[0, 4]))
#                         self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
#                                                                 self.task_queue[-1].getTargetID()))
#                         self.task_data = np.delete(self.task_data, index, axis= 0)
#                         break
#         else:
#             if len(self.task_queue) == 0:
#                 start: Vertex = self.graph.getVertex(int(self.task_data[self.task_id, 0]))
#                 target: Vertex = self.graph.getVertex(int(self.task_data[self.task_id, 1]))
#                 self.task_queue.append(Task(start, target, int(self.task_data[self.task_id, 2]), 
#                                             int(self.task_data[self.task_id, 3]), self.task_data[self.task_id, 4]))
#                 self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
#                                                             self.task_queue[-1].getTargetID()))
#                 self.task_data = np.delete(self.task_data, 0, axis= 0)
#             else:
#                 index = 0
#                 while True:
#                     if index < self.task_data.shape[0] - 1:
#                         start: Vertex = self.graph.getVertex(int(self.task_data[index, 0]))
#                         if start.getID() in start_list: 
#                             index += 1
#                         else:
#                             target: Vertex = self.graph.getVertex(int(self.task_data[index, 1]))
#                             self.task_queue.append(Task(start, target, int(self.task_data[index, 2]), 
#                                                     int(self.task_data[index, 3]), self.task_data[index, 4]))
#                             self.task_queue[-1].setRoute(AStarPlanning(self.graph, TO_TARGET, self.task_queue[-1].getStartID(), 
#                                                                     self.task_queue[-1].getTargetID()))
#                             self.task_data = np.delete(self.task_data, index, axis= 0)
#                             break
#                     else:
#                         self.task_data = self.task_data_origin.copy()
#         if self.task_id >= self.task_data_origin.shape[0] - 1:
#             self.task_id = 0
#         else:
#             self.task_id += 1