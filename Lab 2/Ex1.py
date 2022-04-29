import cv2
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as pt
from mpl_toolkits import mplot3d

b_img = np.zeros((5,5), dtype=np.uint8)

img_1 = deepcopy(b_img)
img_2 = deepcopy(b_img)
img_3 = deepcopy(b_img)

img_1[::,:2] = 255
print(img_1)

img_2[1:-1, 1:-1] = 255
print(img_2)


np.fill_diagonal(img_3, 255)
print(img_3)


[nrow,ncol] = img_1.shape
[xCoor, yCoor] = np.mgrid[0:nrow, 0:ncol]
pt.figure()
ax = pt.axes(projection='3d')
ax.plot_surface(xCoor, yCoor, img_1, cmap=pt.cm.jet)
pt.show()

[nrow,ncol] = img_2.shape
[xCoor, yCoor] = np.mgrid[0:nrow, 0:ncol]
pt.figure()
ax = pt.axes(projection='3d')
ax.plot_surface(xCoor, yCoor, img_2, cmap=pt.cm.jet)
pt.show()

[nrow,ncol] = img_3.shape
[xCoor, yCoor] = np.mgrid[0:nrow, 0:ncol]
pt.figure()
ax = pt.axes(projection='3d')
ax.plot_surface(xCoor, yCoor, img_3, cmap=pt.cm.jet)
pt.show()