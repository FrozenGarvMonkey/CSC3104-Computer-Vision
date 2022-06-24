import cv2
import numpy as np
from matplotlib import pyplot as pt

def RegionGrowing(img,seeds,thresh,p):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
    seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    
    while(len(seedList)>0):
    currentPoint = seedList.pop(0)

    seedMark[currentPoint.x,currentPoint.y] = label
    for i in range(8):
        tmpX = currentPoint.x + connects[i].x
        tmpY = currentPoint.y + connects[i].y
        
        if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
            continue
        
        grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
        
        if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
            seedMark[tmpX,tmpY] = label
            seedList.append(Point(tmpX,tmpY))
            
    return seedMark

img = cv2.imread("D:\Programming\CSC3104 Computer Vision\Lab 3\sudoku.png", 0)
retVal, threshImg = cv2.threshold(img, 110, 1, cv2.THRESH_BINARY)

threshImg = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,0)

pt.figure()
pt.imshow(threshImg, cmap="gray")
pt.show()
