#!/usr/bin/env python3
import glob
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from libraries import *
import random
import torch
from torch.autograd import grad

model = YOLO('best.pt')
training_path = glob.glob('training/training_center/*.jpg')
random.shuffle(training_path)
p_matrix, img_points, r_matrix, t_vector, k_matrix, dist_coef = cameraCalibration()
# Set position of object and the bounding box
x_a_list = list()
y_a_list = list()
x_p_list = list()
y_p_list = list()

for i in range(15):
    path = training_path[i]
    path_split = path.split('_')
    x_obj = float(path_split[1].split('/')[1])
    y_obj = float(path_split[2])
    x_a_list.append(x_obj)
    y_a_list.append(y_obj)
    image = cv2.imread(path)
    point1, point2 = getBoundingBox(model, image)
    bounding_center = np.array([0.5*(point1[0] + point2[0]), 0.5*(point1[1] + point2[1])])
    point3d = convert2Dto3DPoint(bounding_center, r_matrix, t_vector, k_matrix)
    x_p_list.append(round(point3d[0], 2))
    y_p_list.append(round(point3d[1], 2))

x_a_np = np.array(x_a_list)
y_a_np = np.array(y_a_list)
x_p_np = np.array(x_p_list)
y_p_np = np.array(y_p_list)

x_a = torch.from_numpy(x_a_np.astype(np.float32))
y_a = torch.from_numpy(y_a_np.astype(np.float32))
x_p = torch.from_numpy(x_p_np.astype(np.float32))
y_p = torch.from_numpy(y_p_np.astype(np.float32))
# x_c = torch.randn(1, requires_grad=True)
# y_c = torch.randn(1, requires_grad=True)
x_c = torch.tensor(0.0, requires_grad=True)
y_c = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.01

for i in range(10000):
    if i % 100 == 0:
        print("Step:", i+1)
    cost1 = calculateCostFunction(x_c, y_c, x_a, y_a, x_p, y_p)
    df_x_c = grad(cost1, x_c)[0]
    cost2 = calculateCostFunction(x_c, y_c, x_a, y_a, x_p, y_p)
    df_y_c = grad(cost2, y_c)[0]
    # print("Df/dx_c:",df_x_c)
    # print("Df/dy_c:",df_y_c)
    x_c = x_c - learning_rate * df_x_c
    y_c = y_c - learning_rate * df_y_c
    # print("x_c:",x_c)
    # print("y_c:",y_c)
    if (abs(df_x_c.item()) < 0.0001 and abs(df_y_c.item()) < 0.0001):
        break
    
alpha = calculateAlpha(x_c, y_c, x_a, y_a, x_p, y_p).item()
x_c_result = x_c.item()
y_c_result = y_c.item()

x_a_test = list()
y_a_test = list()
x_p_test = list()
y_p_test = list()

for i in range(20):
    path = training_path[i]
    path_split = path.split('_')
    x_obj = float(path_split[1].split('/')[1])
    y_obj = float(path_split[2])
    x_a_test.append(x_obj)
    y_a_test.append(y_obj)
    image = cv2.imread(path)
    point1, point2 = getBoundingBox(model, image)
    bounding_center = np.array([0.5*(point1[0] + point2[0]), 0.5*(point1[1] + point2[1])])
    point3d = convert2Dto3DPoint(bounding_center, r_matrix, t_vector, k_matrix)
    x_p_test.append(round(point3d[0], 2))
    y_p_test.append(round(point3d[1], 2))

x_a_test = np.array(x_a_test)
y_a_test = np.array(y_a_test)
x_p_test = np.array(x_p_test)
y_p_test = np.array(y_p_test)

x_a = (x_c_result + alpha * x_p_test) / (1+alpha)
y_a = (y_c_result + alpha * y_p_test) / (1+alpha)

x_error = np.abs(10 * (x_a - x_a_test))
y_error = np.abs(10 * (y_a - y_a_test))
total_error = np.hypot(x_error, y_error)

print("x_errors:", np.round(x_error, 2))
print("y_errors:", np.round(y_error, 2))
print("errors:", np.round(total_error, 2))
np.savetxt("report/x_errors.txt", x_error, fmt="%.2f")
np.savetxt("report/y_errors.txt", y_error, fmt = "%.2f")
np.savetxt("report/errors.txt", total_error, fmt= "%.2f")
print("Average of x_error: ", np.round(x_error.mean(), 2))
print("Average of y_error: ", np.round(y_error.mean(), 2))
print("Average of errors: ", np.round(total_error.mean(), 2))
_, axs = plt.subplots(1, 3)
axs[0].bar(np.arange(1, 21), x_error, color = 'r')
axs[0].grid(True)
axs[0].set_title("Sai số trên trục x")
axs[1].bar(np.arange(1, 21), y_error, color = 'g')
axs[1].grid(True)
axs[1].set_title("Sai số trên trục y")
axs[2].bar(np.arange(1, 21), total_error, color = 'b')
axs[2].grid(True)
axs[2].set_title("Sai số Euclid")
for ax in axs.flat:
    ax.set(xlabel='Ảnh kiểm tra', ylabel='Sai số(mm)')
for ax in axs.flat:
    ax.label_outer()
plt.show()