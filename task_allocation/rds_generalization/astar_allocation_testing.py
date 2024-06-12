from matplotlib.axes import Axes
from utlis.utlis import *
from env.env import Environment
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
num_of_test = 1
num_task_test = 100
figure, map_visual = plt.subplots(subplot_kw={'aspect': 'equal'})
map_visual: Axes = map_visual
env = EnvironmentChessboard(data_folder, num_of_robot= num_of_robot, robot_max_speed= 1.0, robot_max_payload= 200, figure=figure, map_visual= map_visual)
server = RDSServerChessboard(env, max_task_gen, num_task_in_queue, num_priority, num_type, min_load, max_load, 
                    num_zone_in_cols, num_zone_in_rows, allocation_model_folder, planning_model_folder, num_task_test=num_task_test)

server.setAllocationModel(os.path.join(allocation_model_folder, "best_time_model_seed_{}.pth".format(30)))
# time_counter = 0.0
# task_text: Text = map_visual.text(45, 32.5, "", color='red')
# # for iter in range(num_of_test):
# i = 0
# while True:
#     state, times, priorities = server.task_allocation.AStarTesting()
#     task_text.set_text("Done: {}%".format(round(server.task_allocation.test_task_state.count(True)/num_task_test * 100, 2)))
#     env.AStarFSMControl(waiting_time= 30)
#     env.visualize()
#     time_counter += 0.1
#     if i % 500 == 499:
#         figure.savefig("frame_test.png")
#     i+=1
#     if state == True:
#         print("Total implemented time in seed {}: {} s".format(seed_value, round(time_counter, 2)))
#         # print("Testing TTD in seed {}: ".format(seed_value), times)
#         print("Sum testing TTD in seed {}: ".format(seed_value), round(sum(times), 2))
#         print("Sum testing priority in seed {}: ".format(seed_value), sum(priorities))
#         time_counter = 0.0
#         break
time_counter = 0.0
task_text: Text = map_visual.text(45, 32.5, "", color='red')
robot_text: Text = map_visual.text(5, 32.5, "Robot: {}".format(num_of_robot), color='red')
task_text.set_fontsize(18)
robot_text.set_fontsize(18)
index = 0
def update(frame):
    global time_counter, index
    task_text.set_text("Tasks: {}/{}".format(server.task_allocation.test_task_state.count(True), num_task_test))
    if index < 30:
        index += 1
    else:
        map_visual.set_title('Time {} s'.format(round((frame + 1) * 0.1, 1)))
        state, _, _ = server.task_allocation.AStarTesting()
        env.AStarFSMControl(waiting_time= 30)
        env.visualize()
        if state == True:
            # print("Testing times: ", sum(times))
            # print("Testing priorities: ", sum(priorities))
            print("Total implemented time in test {}: {} s".format(iter, round(time_counter, 2)))
            time_counter = 0.0
        else:
            time_counter += 0.1
    
ani = FuncAnimation(fig=env.figure, func=update, frames=5000, interval=100, repeat= True) # type: ignore
# ani.save("test_{}.gif".format(num_of_robot), writer="pillow")
plt.tight_layout()
# plt.xticks([])
# plt.yticks([])
plt.show()