from utlis.utlis import *
from env.env import Environment
from manager.server import RDSServer
from matplotlib.animation import FuncAnimation
import os
from tqdm import tqdm

data_folder = "data/30x20"
allocation_model_folder = 'data/model/task_allocation'
planning_model_folder = 'data/model/path_planning'
num_of_task = 1000
num_of_robot = 20
task_data_path = os.path.join(data_folder, 'task_data', '{}_task.txt'.format(num_of_task))
figure, map_visual = plt.subplots(subplot_kw={'aspect': 'equal'})
env = Environment(data_folder, num_of_robot= num_of_robot, robot_max_speed= 1.0, robot_max_payload= 200, figure=figure, map_visual= map_visual)
server = RDSServer(env= env, task_data_path= task_data_path, num_task_in_queue= min(num_of_robot, 50), 
                    num_zone_in_cols= 5, num_zone_in_rows= 5, allocation_model_folder= allocation_model_folder, 
                    planning_model_folder= planning_model_folder, allocation_mode= TRAIN_MODE)

allocation_model_list = ['model.pth', 'best_loss_model.pth', 'best_reward_model.pth']
server.setAllocationModel(os.path.join(allocation_model_folder, allocation_model_list[1]))

num_of_iteration = 1000
for iter in tqdm(range(num_of_iteration)):
    while True:
        server.allocationTesting()
        training_state = server.path_planning.training(iter, 10)
        env.pathTrainingControl()
        if training_state == True:
            break
# def update(frame):
#     server.allocationTesting()
#     server.path_planning.training(frame, 10)
#     map_visual.set_title('Time {} s'.format(round((frame + 1) * 0.1, 1)))
#     env.pathTrainingControl()
#     env.visualize()
    
# ani = FuncAnimation(fig=env.figure, func=update, frames=1000, interval=10, repeat= True) # type: ignore
# plt.tight_layout()
# plt.xticks([])
# plt.yticks([])
# plt.show()