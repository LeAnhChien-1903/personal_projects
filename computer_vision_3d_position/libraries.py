#!/usr/bin/env python
import cv2
import numpy as np
import glob
import torch
import matplotlib.pyplot as plt

def cameraCalibration():
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (7, 7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    obj_points = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    img_points = [] 

    # Defining the world coordinates for 3D points
    obj_p = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    obj_p[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    images = glob.glob('calibration/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        # cv2.imwrite(fname, cv2.resize(img, (300, 400)))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            obj_points.append(obj_p)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)

            img_points.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # plt.show()

    ret, k_matrix, dist_coef, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    r_matrix = cv2.Rodrigues(rvecs[0])[0]
    t_vector = tvecs[0]/10
    rt_matrix = np.column_stack((r_matrix, t_vector))
    p_matrix = np.matmul(k_matrix,rt_matrix) # A[R|t]

    return p_matrix, img_points, r_matrix, t_vector, k_matrix, dist_coef

def convert2Dto3DPoint(point: np.ndarray, r_matrix:np.ndarray, t_vector: np.ndarray, k_matrix: np.ndarray):
    camera_matrix_inv = np.linalg.inv(k_matrix)
    vec_1 = np.array([point[0]*t_vector[-1], point[1] * t_vector[-1], t_vector[-1]]).reshape((3, 1))
    camera_point = np.dot(camera_matrix_inv, vec_1)
    t_vector = t_vector.reshape((3, 1))
    vec_2 = camera_point - t_vector
    rotation_matrix_inv = np.linalg.inv(r_matrix)
    world_point = (np.dot(rotation_matrix_inv, vec_2) * 33).reshape(-1)
    world_point[-1] = 0.0
    return world_point

def convert3DTo2DPoint(point3d: np.ndarray, p_matrix: np.ndarray):
    point = np.append(point3d.copy()/33, 1.0).reshape((4, 1))

    point_not_norm = np.matmul(p_matrix, point)
    result = np.array([np.floor(point_not_norm[0]/point_not_norm[-1]), np.floor(point_not_norm[1]/point_not_norm[-1])])
    
    return result.astype(np.int16).reshape(-1)

def getBoundingBox(model, image):
    result = model.predict(image, imgsz=320, conf=0.1, verbose = False, save = False)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    return (x1, y1), (x2, y2)

def calculateAlpha(x_c, y_c, x_a, y_a, x_p, y_p):
    n = x_a.shape[0]
    gamma = torch.sqrt(torch.pow((x_p - x_a), 2) + torch.pow((y_p - y_a), 2))
    
    alpha = torch.sqrt(torch.pow(x_c - x_a, 2) + torch.pow(y_c - y_a, 2)) / (n * gamma)
    alpha = torch.sum(alpha)
    
    return alpha

def calculateCostFunction(x_c, y_c, x_a, y_a, x_p, y_p):
    n = x_a.shape[0]
    gamma = torch.sqrt(torch.pow((x_p - x_a), 2) + torch.pow((y_p - y_a), 2))
    alpha = torch.sqrt(torch.pow(x_c - x_a, 2) + torch.pow(y_c - y_a, 2)) / (n * gamma)
    alpha = torch.sum(alpha)
    
    cost = torch.pow((alpha - torch.sqrt(torch.pow(x_c - x_a, 2)) / gamma), 2)
    cost = (1 / n) * torch.sum(cost)
    
    return cost


def main():
    p_matrix, img_points, r_matrix, t_vector, k_matrix, dist_coef = cameraCalibration()
    print(r_matrix)
    print(t_vector)
    print(k_matrix)

# if __name__ == '__main__':
#     main()