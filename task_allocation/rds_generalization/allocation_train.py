from utlis.utlis import *
from env.env_chessboard import EnvironmentChessboard
from manager.server import RDSServerChessboard
import os
from tqdm import tqdm
data_folder = "data/chessboard_layout/60x30"
allocation_model_folder = 'data/model/task_allocation/chessboard_layout/60x30/original'
planning_model_folder = 'data/model/path_planning'
num_of_robot = 32
num_task_in_queue = 16
max_task_gen: int = 1000
num_priority: int = 5
num_type: int = 2 
min_load: float = 30
max_load: float = 200    
num_zone_in_cols: int = 10
num_zone_in_rows: int = 5
figure, map_visual = plt.subplots(subplot_kw={'aspect': 'equal'})
env = EnvironmentChessboard(data_folder, num_of_robot= num_of_robot, robot_max_speed= 1.0, robot_max_payload= 200, figure=figure, map_visual= map_visual)
server = RDSServerChessboard(env, max_task_gen, num_task_in_queue, num_priority, num_type, min_load, max_load, 
                            num_zone_in_cols, num_zone_in_rows, allocation_model_folder, planning_model_folder)

num_of_iteration = 100
task_text: Text = map_visual.text(45, 32.5, "", color='red')
best_time = float('inf')
best_model = deepcopy(server.task_allocation.model)
num_of_task_test = 100
for iter in tqdm(range(num_of_iteration)):
    i = 0
    while True:
        training_state = server.task_allocation.training(iter, 5)
        server.env.AStarFSMControl(waiting_time= 30)
        server.env.visualize()
        task_text.set_text("Done: {}%".format(round(server.task_allocation.batch.reward_state.count(True)/server.task_allocation.update_interval * 100, 2)))
        if i % 500 == 499:
            figure.savefig("frame_train.png")
        if training_state == True:
            break
        i+=1
    if iter % 5 == 0:
        figure_test, map_visual_test = plt.subplots(subplot_kw={'aspect': 'equal'})
        task_text_test: Text = map_visual_test.text(45, 32.5, "", color='red')
        env_test = EnvironmentChessboard(data_folder, num_of_robot= num_of_robot, robot_max_speed= 1.0, robot_max_payload= 200, figure= figure_test, map_visual= map_visual_test)
        server_test = RDSServerChessboard(env_test, max_task_gen, num_task_in_queue, num_priority, num_type, min_load, max_load, 
                                            num_zone_in_cols, num_zone_in_rows, allocation_model_folder, 
                                            planning_model_folder, num_of_task_test)
        server_test.task_allocation.model = deepcopy(server.task_allocation.best_reward_model)
        time_counter = 0.0
        j = 0
        while True:
            state, times, priorities = server_test.task_allocation.AStarTesting()
            task_text_test.set_text("Done: {}%".format(round(server_test.task_allocation.test_task_state.count(True)/ num_of_task_test * 100, 2)))
            env_test.AStarFSMControl(waiting_time= 30)
            env_test.visualize()
            time_counter += 0.1
            if j % 500 == 499:
                figure_test.savefig("frame_test.png")
            j+=1
            if state == True:
                if round(time_counter, 2) < best_time:
                    best_time = round(time_counter, 2)
                    best_model = deepcopy(server.task_allocation.best_reward_model)
                    torch.save(best_model.state_dict(), os.path.join(allocation_model_folder, "best_time_model_seed_{}.pth".format(seed_value)))
                    print("Best time test: {} s".format(best_time))
                break  