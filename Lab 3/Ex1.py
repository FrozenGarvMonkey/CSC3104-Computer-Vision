import cv2
import numpy as np
from matplotlib import pyplot as pt

img = cv2.imread("D:\Programming\CSC3104 Computer Vision\Lab 3\sudoku.png", 0)
retVal, threshImg = cv2.threshold(img, 110, 1, cv2.THRESH_BINARY)

threshImg = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,0)

pt.figure()
pt.imshow(threshImg, cmap="gray")
pt.show()
