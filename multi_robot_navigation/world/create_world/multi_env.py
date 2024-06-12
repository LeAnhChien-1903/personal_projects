import numpy as np
import matplotlib.pyplot as plt
import os

init_pose = np.array([[-7.00, 11.50, np.pi], [-7.00, 9.50, np.pi], [-18.00, 11.50, 0.00], [-18.00, 9.50, 0.00],
            [-12.50, 17.00, np.pi*3/2], [-12.50, 4.00, np.pi/2], [-2.00, 16.00, -np.pi/2], [0.00, 16.00, -np.pi/2],
            [3.00, 16.00, -np.pi/2], [5.00, 16.00, -np.pi/2], [10.00, 4.00, np.pi/2], [12.00, 4.00, np.pi/2],
            [14.00, 4.00, np.pi/2], [16.00, 4.00, np.pi/2], [18.00, 4.00, np.pi/2], [-2.5, -2.5, 0.00],
            [-0.5, -2.5, 0.00], [3.5, -2.5, np.pi], [5.5, -2.5, np.pi], [-2.5, -18.5, np.pi/2],
            [-0.5, -18.5, np.pi/2], [1.5, -18.5, np.pi/2], [3.5, -18.5, np.pi/2], [5.5, -18.5, np.pi/2],
            [-6.00, -10.00, np.pi], [-7.15, -6.47, np.pi*6/5], [-10.15, -4.29, np.pi*7/5], [-13.85, -4.29, np.pi*8/5],
            [-16.85, -6.47, np.pi*9/5], [-18.00, -10.00, np.pi*2], [-16.85, -13.53, np.pi*11/5], [-13.85, -15.71, np.pi*12/5],
            [-10.15, -15.71, np.pi*13/5], [-7.15, -13.53, np.pi*14/5], [10.00, -17.00, np.pi/2], [12.00, -17.00, np.pi/2],
            [14.00, -17.00, np.pi/2], [16.00, -17.00, np.pi/2], [18.00, -17.00, np.pi/2], [10.00, -2.00, -np.pi/2],
            [12.00, -2.00, -np.pi/2], [14.00, -2.00, -np.pi/2], [16.00, -2.00, -np.pi/2], [18.00, -2.00, -np.pi/2]])

goal_point = np.array([[-18.0, 11.5], [-18.0, 9.5], [-7.0, 11.5], [-7.0, 9.5], [-12.5, 4.0], [-12.5, 17.0],
                        [-2.0, 3.0], [0.0, 3.0], [3.0, 3.0], [5.0, 3.0], [10.0, 10.0], [12.0, 10.0],
                        [14.0, 10.0], [16.0, 10.0], [18.0, 10.0], [3.5, -2.5], [5.5, -2.5], [-2.5, -2.5],
                        [-0.5, -2.5], [-2.5, -5.5], [-0.5, -5.5], [1.5, -5.5], [3.5, -5.5], [5.5, -5.5],
                        [-18.0, -10.0], [-16.85, -13.53], [-13.85, -15.71], [-10.15, -15.71], [-7.15, -13.53], 
                        [-6.00, -10.00], [-7.15, -6.47], [-10.15, -4.29], [-13.85, -4.29], [-16.85, -6.47], 
                        [10.00, -2.00], [12.00, -2.00], [14.00, -2.00], [16.00, -2.00], [18.00, -2.00], 
                        [10.00, -17.00], [12.00, -17.00], [14.00, -17.00], [16.00, -17.00], [18.00, -17.00]])

goal_dir = "goal_point"
pose_dir = "init_pose"
map_name = "multi_env"
np.savetxt(os.path.join(goal_dir, "{}.txt".format(map_name)), np.array(goal_point), fmt= '%.5f')
np.savetxt(os.path.join(pose_dir, "{}.txt".format(map_name)), np.array(init_pose), fmt= '%.5f')
# plt.subplot(1, 2, 1)
# plt.plot(init_pose[:,0], init_pose[:,1], 'r.')
# plt.grid()
# plt.subplot(1, 2, 2)
# plt.plot(goal_point[:,0], goal_point[:,1], 'r.')
# plt.grid()
# plt.show()

for i in range(goal_point.shape[0]):
    plt.plot([init_pose[i, 0], goal_point[i, 0]], [init_pose[i, 1], goal_point[i, 1]], marker='.')
plt.show()