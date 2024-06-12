from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("data/measure/60x30/statistic.csv", delimiter=',')
time_32_100_tasks = data[:, 0:3]
time_60_100_tasks = data[:, 3:6]
time_32_500_tasks = data[:, 6:9]
time_60_500_tasks = data[:, 9:12]
ttd_32_100_tasks = data[:, 12:15]
ttd_60_100_tasks = data[:, 15:18]
ttd_32_500_tasks = data[:, 18:21]
ttd_60_500_tasks = data[:, 21:]

colors = ['r', 'g', 'orange']

fig, ax1 = plt.subplots(1, 2)
fig.set_dpi(300)
ax1_0: Axes = ax1[0]
ax1_1: Axes = ax1[1]
box_plot0 = ax1_0.boxplot(time_32_100_tasks, patch_artist=True)
ax1_0.set_xticklabels(["RL", "A*", "Manhattan"])
ax1_0.grid(True)
ax1_0.set_ylabel('Thời gian hoàn thành nhiệm vụ')
# ax1_0.set_ylabel('Tổng thời gian đến điểm bắt đầu của các nhiệm vụ')
ax1_0.set_xlabel("Phương pháp phân nhiệm")
ax1_0.set_title('32 robot')
box_plot1 = ax1_1.boxplot(time_60_100_tasks, patch_artist=True)
ax1_1.set_xticklabels(["RL", "A*", "Manhattan"])
ax1_1.grid(True)
ax1_1.set_xlabel("Phương pháp phân nhiệm")
ax1_1.set_title('60 robot')
# fill with colors
for patch, color in zip(box_plot0['boxes'], colors):
    patch.set_facecolor(color)
for median in box_plot0['medians']:
    median.set_color('black')
for patch, color in zip(box_plot1['boxes'], colors):
    patch.set_facecolor(color)
for median in box_plot1['medians']:
    median.set_color('black')
# fig2, ax2 = plt.subplots()
# box_plot = ax2.boxplot(time_500_tasks, patch_artist=True)
# ax2.set_xticklabels(["RL(32)", "A*(32)", "Manhattan(32)", "RL(60)", "A*(60)", "Manhattan(60)"])
# ax2.grid(True)
# ax2.set_ylabel('Thời gian hoàn thành nhiệm vụ')
# ax2.set_xlabel("Phương pháp phân nhiệm")
# ax2.set_title("Thời gian hoàn thành 500 nhiệm vụ của 60 robot")
# fill with colors
# for patch, color in zip(box_plot['boxes'], colors):
#     patch.set_facecolor(color)
# for median in box_plot['medians']:
#     median.set_color('black')

# fig3, ax3 = plt.subplots()
# box_plot = ax3.boxplot(ttd_100_tasks, patch_artist=True)
# ax3.set_xticklabels(["RL(32)", "A*(32)", "Manhattan(32)", "RL(60)", "A*(60)", "Manhattan(60)"])
# ax3.grid(True)
# ax3.set_ylabel('Tổng thời gian đến điểm bắt đầu nhiệm vụ')
# ax3.set_xlabel("Phương pháp phân nhiệm")
# # ax3.set_title("Tổng thời gian đến điểm bắt đầu 100 nhiệm vụ của 60 robot")
# # fill with colors
# for patch, color in zip(box_plot['boxes'], colors):
#     patch.set_facecolor(color)
# for median in box_plot['medians']:
#     median.set_color('black')

# fig4, ax4 = plt.subplots()
# box_plot = ax4.boxplot(ttd_500_tasks, patch_artist=True)
# ax4.set_xticklabels(["RL(32)", "A*(32)", "Manhattan(32)", "RL(60)", "A*(60)", "Manhattan(60)"])
# ax4.grid(True)
# ax4.set_ylabel('Tổng thời gian đến điểm bắt đầu nhiệm vụ')
# ax4.set_xlabel("Phương pháp phân nhiệm")
# # ax4.set_title("Tổng thời gian đến điểm bắt đầu 500 nhiệm vụ của 60 robot")
# # fill with colors
# for patch, color in zip(box_plot['boxes'], colors):
#     patch.set_facecolor(color)
# for median in box_plot['medians']:
#     median.set_color('black')
# plt.tight_layout()
plt.show()