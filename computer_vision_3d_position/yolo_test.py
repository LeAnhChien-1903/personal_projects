#!/usr/bin/env python
import glob
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import torch

x = torch.tensor(5.0, requires_grad=True)
y = torch.tensor(10.0, requires_grad=True)
f = 2 *(x**2 + 5) + y**2 + 6*y
f.backward()

print(x.grad)
print(y.grad)

