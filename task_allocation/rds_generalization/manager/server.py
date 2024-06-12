from utlis.utlis import *
from manager.task_generate import TaskGenerator
from manager.task_allocation import TaskAllocation
from manager.path_planning import PathPlanning
from env.env_chessboard import EnvironmentChessboard
from env.env import Environment

class RDSServer:
    def __init__(self, env: Environment, max_task_gen: int, num_task_in_queue: int, num_priority: int, num_type: int, 
                min_load: float, max_load: float, num_zone_in_cols: int, num_zone_in_rows: int, 
                allocation_model_folder: str, planning_model_folder: str, num_task_test: int = 100):
        self.env: Environment = env
        self.factory_graph = self.env.graph
        self.factory_graph.createZone(self.env.factory_map.map_center[0], self.env.factory_map.map_center[1], 
                                        self.env.factory_map.map_length, self.env.factory_map.map_width, 
                                        num_zone_in_cols, num_zone_in_rows)
        self.robots = self.env.robots
        # self.task_generator = TaskGenFromData(self.robots, task_data_path, num_task_in_queue, self.factory_graph, mode= allocation_mode)
        
        self.task_generator = TaskGenerator(max_task_gen, num_task_in_queue, num_priority, num_type, 
                                            min_load, max_load, self.factory_graph)
        self.task_allocation = TaskAllocation(self.robots, self.task_generator, self.factory_graph, allocation_model_folder, env, num_task_test) # type: ignore
    
    def allocationAStarTesting(self):
        return self.task_allocation.AStarTesting()
    
    def allocationTraining(self, iter: int, save_interval: int = 10):
        self.task_allocation.training(iter, save_interval)
    
    def setAllocationModel(self, path: str):
        self.task_allocation.setModel(path)
        

class RDSServerChessboard:
    def __init__(self, env: EnvironmentChessboard, max_task_gen: int, num_task_in_queue: int, num_priority: int, num_type: int, 
                min_load: float, max_load: float, num_zone_in_cols: int, num_zone_in_rows: int, 
                allocation_model_folder: str, planning_model_folder: str, num_task_test: int = 100):
        self.env: EnvironmentChessboard = env
        self.factory_graph = self.env.graph
        self.factory_graph.createZone(self.env.factory_map.map_center[0], self.env.factory_map.map_center[1], 
                                        self.env.factory_map.map_length, self.env.factory_map.map_width, 
                                        num_zone_in_cols, num_zone_in_rows)
        self.robots = self.env.robots
        # self.task_generator = TaskGenFromData(self.robots, task_data_path, num_task_in_queue, self.factory_graph, mode= allocation_mode)
        
        self.task_generator = TaskGenerator(max_task_gen, num_task_in_queue, num_priority, num_type, 
                                            min_load, max_load, self.factory_graph)
        self.task_allocation = TaskAllocation(self.robots, self.task_generator, self.factory_graph, allocation_model_folder, env, num_task_test)
    
    def allocationAStarTesting(self):
        return self.task_allocation.AStarTesting()
    
    def allocationTraining(self, iter: int, save_interval: int = 10):
        self.task_allocation.training(iter, save_interval)
    
    def setAllocationModel(self, path: str):
        self.task_allocation.setModel(path)