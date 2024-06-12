#!/usr/bin/env python3
import glob
from ultralytics import YOLO
import matplotlib.pyplot as plt
from libraries import *
# Load a model
model = YOLO('best.pt')
calibration_path = glob.glob('calibration/*.jpg')
train_path = glob.glob('training/s_2_2.jpg')
image = cv2.imread(train_path[0])

point1, point2 = getBoundingBox(model, image)
center = [int((point1[0] + point2[0])/2), int(0.5 * (point1[1] + point2[1]))]
print(center)
cv2.circle(image, center, 2, (0, 0, 255), -1)
cv2.imwrite("test/center.png", image)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()