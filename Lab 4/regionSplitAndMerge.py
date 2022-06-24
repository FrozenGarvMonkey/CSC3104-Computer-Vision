import cv2
import numpy as np
from matplotlib import pyplot as pt

#Create the test image
img = np.array([[5,5,5,2,1,1,1,7],
                [2,5,5,2,2,1,1,6],
                [2,5,4,4,4,6,6,6],
                [2,2,4,4,4,6,6,6],
                [3,2,2,4,7,7,7,7],
                [3,3,2,4,1,1,1,1],
                [3,3,2,2,4,2,2,1],
                [3,3,2,2,4,2,2,2]])

#Set the threshold for homogeneity test
threshold = 1

#Initialization for Splitting
toProcessList = []
doneProcessList = []

#Get the size of the image and add this as a region into toProcessList
nrow,ncol = img.shape
originalLength = nrow

#All regions (or quadrants) are stored in the form of [row coordinate, column coordinate, length]
initialQuadrant = [0,0,originalLength]
toProcessList.append(initialQuadrant)

#==============================Splitting=======================================
#Continue to loop until toProcessList is empty
while len(toProcessList) > 0:
    
    #Extract an element from the list to get the row coordinate, column coordinate, and length og the region
    startX,startY,length = toProcessList[0]
    
    #Extract the region to perform homegeneity test
    checkRegion = img[startX:startX+length,startY:startY+length]
    dif = checkRegion.max() - checkRegion.min()
    if dif <= threshold:
        #If the region is homogenous, move the element to doneProcessList
        doneProcessList.append(toProcessList[0])
        toProcessList.pop(0)
    else:
        #If the region is not homogenous, divide the length by 2, and
        #add the top left pixel of the four quadrants to the toProessList,
        #together with the new length of each quadrant
        newLength = int(length/2)
        topLeftQuadrant = [startX,startY,newLength]
        topRightQuadrant = [startX,startY+newLength,newLength]
        bottomLeftQuadrant = [startX+newLength,startY,newLength]
        bottomRightQuadrant = [startX+newLength,startY+newLength,newLength]
        toProcessList.pop(0)
        toProcessList.append(topLeftQuadrant)
        toProcessList.append(topRightQuadrant)
        toProcessList.append(bottomLeftQuadrant)
        toProcessList.append(bottomRightQuadrant)

#This part is not compulsoty, it is just to fill up the region map so that we can
#see how the entire image is divided (better visualization)
fillProcessList = doneProcessList.copy()
regionMap = np.zeros((nrow,ncol))
label = 1
while len(fillProcessList) > 0:
    startX,startY,length = fillProcessList[0]
    regionMap[startX:startX+length,startY:startY+length] = label
    label = label + 1
    fillProcessList.pop(0)

#===============================Merging========================================

#Create the merge region map for us to show the regions after merging
mergeRegionMap = np.zeros((nrow,ncol))

#Initialization
label = 0

#Same size as the doneProcessList. It is used to keep track of region that had
#already been processed (marked with '1'). For example, if doneProcessList[8] had
#already been processed, then skipPointerArrayList[8] is set to '1'.
skipPointerArrayList = np.zeros((len(doneProcessList)))
sE8 = np.array([[1,1,1],[1,1,1],[1,1,1]],dtype=np.uint8)

#Loop until every region in the doneProcessList is evaluated
for seedPointer in range(0,len(doneProcessList)):
    #skipPointerArrayList is used to keep track of region that has already been
    #processed (marked with '1'). Hence, we only proceed with it is marked as '0'.
    if skipPointerArrayList[seedPointer] == 0:
        label = label + 1
        
        #Mark with 1 to indicate that we have look into this region
        skipPointerArrayList[seedPointer] = 1
        
        #To find the max and min value  of the seed region (starting region)
        seedStartX,seedStartY,seedLength = doneProcessList[seedPointer]
        seedRegion = img[seedStartX:seedStartX+seedLength,seedStartY:seedStartY+seedLength]       
        seedMin = seedRegion.min()
        seedMax = seedRegion.max()
        
        #Set the condition map
        sameConditionListPointerMap = np.zeros((nrow,ncol))-1
        sameConditionListPointerMap[seedStartX:seedStartX+seedLength,seedStartY:seedStartY+seedLength] = 0
        
        #Set the connection map
        connectionMap = np.zeros((nrow,ncol))
        connectionMap[seedStartX:seedStartX+seedLength,seedStartY:seedStartY+seedLength] = 1

        #To process the other regions in the list. The aim is to search for
        #regions that exhibit the same characteristic, or will still make the starting
        #region remains homogenous after merging
        for pointer in range(0,len(doneProcessList)):
            #Similarly, only procss the region is it has never been processed before
            if skipPointerArrayList[pointer] == 0:
                #Get the region info (starting point and length)
                startX,startY,length = doneProcessList[pointer]
                
                #Find the max and min values of the region
                candidateRegion = img[startX:startX+length,startY:startY+length]
                candidateMin = candidateRegion.min()
                candidateMax = candidateRegion.max()
                
                #Try to calculate the max and min value after merging, and perform
                #homogeneity test.
                mergeValueCheckMatrix = np.array([[seedMin,seedMax,candidateMin,candidateMax]])
                dif = mergeValueCheckMatrix.max() - mergeValueCheckMatrix.min()
                if dif <= threshold:
                    sameConditionListPointerMap[startX:startX+length,startY:startY+length] = pointer
                     
                #Produce a binary map based on connectionMap
                sameConditionBinaryMap = np.zeros((nrow,ncol),dtype=np.uint8)
                for row in range(0,nrow):
                    for col in range(0,ncol):
                        if sameConditionListPointerMap[row,col] >= 0:
                            sameConditionBinaryMap[row,col] = 1
                
                #Check whether there is any chnages to the connection map after dilation
                #and finding the intersection between the dilated connection map with the
                #binary version of the condition map. The aim of doing this is to check
                #when we should stop finfing those pixels that are connected to the
                #starting point
                dif = 1
                while dif != 0:
                    newConnectionMap = cv2.dilate(np.uint8(connectionMap),sE8) 
                    newConnectionMap = cv2.bitwise_and(newConnectionMap,sameConditionBinaryMap)
                    dif = np.sum(np.abs(newConnectionMap - connectionMap))
                    if dif != 0:
                        connectionMap = newConnectionMap.copy()
        
        #Check those regions that we are going to merge, and marked those regions
        #as '1' in the skipPointerArrayList. This is to ensure that all these
        #regions will not be processed again in the subsequent iterations (will not
        #be merged with other regions)
        for row in range(0,nrow):
            for col in range(0,ncol):
                if connectionMap[row,col] == 1:
                    removeListPointer = int(sameConditionListPointerMap[row,col])
                    skipPointerArrayList[removeListPointer] = 1
        
        #This part is to label all the regions that should be merged with the same
        #label. We considered we have merged the regions after they are all labelled
        #with the same label (just an integer number)
        for row in range(0,nrow):
            for col in range(0,ncol):
                if connectionMap[row,col] == 1:
                    mergeRegionMap[row,col] = label
        
        
    
        
        
    

