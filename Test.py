import cv2
import numpy as np
from matplotlib import pyplot as pt

#INPUT = IMAGE
#SEED_MAP = ARRAY THAT HAS ALL THE SEEDS
#THRESHOLD = INTEGER (HOMOGENEITY TEST)
#CONNECTIVITY = 4 | 8

inputImg = [
            [5,5,5,2,1,1,1,7],
            [2,5,5,2,2,1,1,6],
            [2,5,4,4,4,6,6,6],
            [2,2,4,4,4,6,6,6],
            [3,2,2,4,7,7,7,7],
            [3,3,2,5,1,1,1,1],
            [3,3,2,2,4,2,2,1],
            [3,3,2,2,4,2,2,2]
        ]

seedMap = [
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,0]
        ]

def willGrow(seedPoint, neighbour, T):
    if(abs(seedPoint - neighbour) <= T):
        return True
    else:
        return False

def dilatePoint(point, C):
    squareKernel = np.ones((3,3), dtype=np.uint8)
    crossKernel = [[0,1,0],[1,1,1],[0,1,0]]

    if(C == 8):
        cv2.dilate(point, squareKernel, iterations=1)
    else if (C == 4):
        cv2.dilate(point, crossKernel, iterations=1)
    else:
        print("Invalid Connectivity!")
    
def RegionGrowing(inputImage, seedMap, threshold, connectivity):
    width, height = inputImage.shape
    label = 0
    
    regImg = []
    
    for i in range(0, width):
        label = label + 1
        for j in range (0, height):
            if seedMap[i][j] != 0:
                newMap = np.zeros((8,8), dtype=np.uint8)
                dilatePoint(newMap[i][j],connectivity)
                
                for k in range(0, width):
                    for l in range(0, height):
                        if(newMap[i][j] != 0):
                            if(willGrow(dilatePoint,threshold)):
                                continue
                            else:
                                newMap[i][j] = 0
                        
                for k in range(0,width):
                    for l in range(0, height):
                        if(newMap[i][j]==1):
                            newMap[i][j]=label
                            
                
        
                                
                            
                
                
                                
    return regImg
                
        
print(inputImg)
print(seedMap)        
print(RegionGrowing(inputImg, seedMap, 1, 4))    


# img = cv2.imread("sudoku.png", 0)
# retVal, threshImg = cv2.threshold(img, 110, 1, cv2.THRESH_BINARY)

# threshImg = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,0)

# pt.figure()
# pt.imshow(threshImg, cmap="gray")
# pt.show()
