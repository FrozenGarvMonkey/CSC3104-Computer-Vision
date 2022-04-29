import cv2
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as pt
from mpl_toolkits import mplot3d

img_1 = np.array([[100, 100, 100], [100, 200, 100], [100, 100, 50]], dtype=np.float32)
img_2 = np.array([[100, 100, 50], [50, 200, 100], [100, 100, 20]], dtype=np.float32)

kernel_hori = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32) / 2
kernel_vert = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32) / 2

grad_hori = cv2.filter2D(img_1, -1, kernel_hori)
grad_vert = cv2.filter2D(img_1, -1, kernel_vert)
grad_mag = np.sqrt(np.power(grad_hori, 2) + np.power(grad_vert, 2))
orientation_rad = np.arctan2(grad_vert, grad_hori)
orientation_deg = (orientation_rad * 180) / np.pi

print("\n\n============== Image 1 ============")

print(grad_mag)
print(orientation_deg)

grad_hori = cv2.filter2D(img_2, -1, kernel_hori)
grad_vert = cv2.filter2D(img_2, -1, kernel_vert)
grad_mag = np.sqrt(np.power(grad_hori, 2) + np.power(grad_vert, 2))
orientation_rad = np.arctan2(grad_vert, grad_hori)
orientation_deg = (orientation_rad * 180) / np.pi

print("\n\n============== Image 2 ============")
print(grad_mag)
print(orientation_deg)