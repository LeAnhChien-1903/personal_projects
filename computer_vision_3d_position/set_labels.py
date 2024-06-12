#!/usr/bin/env python3
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from libraries import *
from  random import randint
import os
image_path = glob.glob('training/*.jpg')
print(len(image_path))
p_matrix, img_points, r_matrix, t_vector, k_matrix, dist_coef = cameraCalibration()
width = 3.5
length = 7.0
box_size = 3.3
path = image_path[-1]
for path in image_path:
    path_split = path.split('_')
    shape = path_split[0].split('/')[1]
    if shape == 's': # square
        x_bottom = box_size * int(path_split[1])
        if len(path_split) == 3:
            y_bottom = box_size * int(path_split[2].split('.')[0])
        else:
            y_bottom = box_size * int(path_split[2])
        x_top = x_bottom + width
        y_top = y_bottom + width
        x_center = round(0.5 * (x_bottom + x_top), 2)
        y_center = round(0.5 * (y_bottom + y_top), 2)
        if len(path_split) == 3:
            new_name = "{}_{}_.jpg".format(x_center, y_center)
        else:
            new_name = "{}_{}_{}_".format(x_center, y_center, path_split[-1])
        image = cv2.imread(path)
        cv2.imwrite(os.path.join("training/training_center/"+new_name), image)
    elif shape == 'r': # rectangle 
        type_ = path_split[3]
        x_bottom = box_size * int(path_split[1])
        y_bottom = box_size * int(path_split[2])
        if type_[0] == 'd':
            x_top = x_bottom + width
            y_top = y_bottom + length
        if type_[0] == 'n':
            x_top = x_bottom + length
            y_top = y_bottom + width
        x_center = round(0.5 * (x_bottom + x_top), 2)
        y_center = round(0.5 * (y_bottom + y_top), 2)
        if len(path_split) == 4:
            new_name = "{}_{}_.jpg".format(x_center, y_center)
        else:
            new_name = "{}_{}_{}_".format(x_center, y_center, path_split[-1])
        image = cv2.imread(path)
        cv2.imwrite(os.path.join("training/training_center/"+new_name), image)

image_path = glob.glob('training/training_center/*.jpg')
print(len(image_path))