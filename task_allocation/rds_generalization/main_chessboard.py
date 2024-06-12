from utlis.utlis import *
from env.env_chessboard import EnvironmentChessboard
from manager.server import RDSServerChessboard
import os
data_folder = "data/chessboard_layout/60x30"
allocation_model_folder = 'data/model/task_allocation'
planning_model_folder = 'data/model/path_planning'
num_of_robot = 24
num_task_in_queue= 20
max_task_gen: int = 100
num_priority: int = 5
num_type: int = 2 
min_load: float = 30
max_load: float = 200 
num_zone_in_cols: int = 10
num_zone_in_rows: int = 5
figure, map_visual = plt.subplots(subplot_kw={'aspect': 'equal'})
env = EnvironmentChessboard(data_folder, num_of_robot= num_of_robot, robot_max_speed= 1.0, robot_max_payload= 200, figure=figure, map_visual= map_visual)
# server = RDSServerChessboard(env, max_task_gen, num_task_in_queue, num_priority, num_type, min_load, max_load,
#                     num_zone_in_cols, num_zone_in_rows, allocation_model_folder, planning_model_folder)

# allocation_model_list = ['model.pth', 'best_loss_model.pth', 'best_reward_model.pth']
# planning_model_list = ['model.pth', 'best_loss_model.pth', 'best_all_reward_model.pth']
# server.setAllocationModel(os.path.join(allocation_model_folder, allocation_model_list[2]))
# # server.setPlanningModel(os.path.join(planning_model_folder, planning_model_list[2]))

# def update(frame):
#     server.allocationAStarTesting()
#     map_visual.set_title('Time {} s'.format(round((frame + 1) * 0.1, 1)))
#     env.AStarFSMControl(waiting_time= 30)
#     env.visualize()
    
# ani = FuncAnimation(fig=env.figure, func=update, frames=1000, interval=100, repeat= True) # type: ignore
# ani.save("test_{}.gif".format(num_of_robot), writer="pillow")
plt.tight_layout()
# plt.xticks([])
# plt.yticks([])
plt.show()
