import numpy as np
import cv2
import matplotlib.pyplot as pt

#====================================Exercise 1=================================
# Creating the test image.
binaryImage = np.zeros((13,13), dtype=np.uint8)
binaryImage[5:7,1] = 1
binaryImage[2:5,2:5] = 1
binaryImage[2,8:10] = 1
binaryImage[3:5,8:12] = 1
binaryImage[8:12,5:12] = 1

binaryContourImage = np.zeros((13,13), dtype=np.uint8)

# Perform region labelling.
noRegion, labelledBinaryImage = cv2.connectedComponents(binaryImage, connectivity=8)

#====================================Exercise 2=================================
# Find contours.
contours,_ = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
binContours = cv2.drawContours(binaryContourImage, contours, -1, (255, 0, 0), 1)

# # Example Code
# scissors = cv2.imread("scissors.png")

# scissorsGray = cv2.cvtColor(scissors,cv2.COLOR_BGR2GRAY)
# _,scissorsBinary = cv2.threshold(scissorsGray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# contours,_= cv2.findContours(scissorsBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# copy = scissors.copy()
# scissorsContours = cv2.drawContours(copy, contours, -1, (255, 0, 0), 1)

# pt.figure()
# pt.imshow(cv2.cvtColor(scissorsContours,cv2.COLOR_BGR2RGB)) 
# pt.show()

#====================================Exercise 3=================================
contour = (contours[1]).tolist()
# Points store in [col, row] format.

code = []
for pointer in range(0,len(contour)-1):
    currentCol, currentRow = contour[pointer][0]
    nextCol, nextRow = contour[pointer+1][0]
    
    # Generate chain code based on eight directions.
    if (nextCol - currentCol) == 1 and (nextRow - currentRow) == 0:
        code.append(0)
    elif (nextCol - currentCol) == 1 and (nextRow - currentRow) == -1:
        code.append(1)
    elif (nextCol - currentCol) == 0 and (nextRow - currentRow) == -1:
        code.append(2)
    elif (nextCol - currentCol) == -1 and (nextRow - currentRow) == -1:
        code.append(3)
    elif (nextCol - currentCol) == -1 and (nextRow - currentRow) == 0:
        code.append(4)
    elif (nextCol - currentCol) == -1 and (nextRow - currentRow) == 1:
        code.append(5)
    elif (nextCol - currentCol) == 0 and (nextRow - currentRow) == 1:
        code.append(6)
    elif (nextCol - currentCol) == 1 and (nextRow - currentRow) == 1:
        code.append(7)
    else:
        print("Error")

difCode = [];
# To produce code that is invariant to rotation (shape number)    
for pointer in range(0,len(code)):
    
    currentCodeNumber = code[pointer]
    # Pointer will not go out of bound.
    nextCodeNumber = code[(pointer + 1)%len(code)]
        
    if nextCodeNumber >= currentCodeNumber:
        newCodeNumber = nextCodeNumber - currentCodeNumber
    else:
        newCodeNumber = (nextCodeNumber+8) - currentCodeNumber
        
    difCode.append(newCodeNumber)

# To produce code that is invariant to starting point (rotate to find the smallest magnitude) 
# To produce the first code for comparison purpose.
smallestCodeStr = ""
for pointer in range(0,len(difCode)):
    smallestCodeStr = smallestCodeStr + str(difCode[pointer])
   
# Rotate, produce a string based on the new code, cast to integer, perform the comparison.
rotationWithSmallestCode = 0;
for numIteration in range(1,len(difCode)):
    newCode = np.roll(difCode,numIteration)
            
    newCodeStr = ""
    for pointer in range(0,len(newCode)):
        newCodeStr = newCodeStr + str(newCode[pointer])
   
    if int(newCodeStr) <= int(smallestCodeStr):
        smallestCodeStr = newCodeStr
        rotationWithSmallestCode = numIteration;

shapeNumber = np.roll(difCode,rotationWithSmallestCode)    
 
print("Chain Code:")
print(code)
print("Difference of Chain Code:")
print(difCode)
print("Shape Number")
print(shapeNumber)

##====================================Exercise 4=================================
# scissors = cv2.imread("scissors.png")

# scissorsGray = cv2.cvtColor(scissors,cv2.COLOR_BGR2GRAY)
# _, scissorsBinary = cv2.threshold(scissorsGray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# contours,_= cv2.findContours(scissorsBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# copy = scissors.copy()
# scissorsContours = cv2.drawContours(copy, contours, -1, (255, 0, 0), 1)

# pt.figure()
# pt.imshow(cv2.cvtColor(scissorsContours,cv2.COLOR_BGR2RGB)) 
# pt.show()

# contours,_ = cv2.findContours(scissorsBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# complexNumberList = []
# contourPoints = contours[0]
# maxNumPoints = len(contourPoints)
# for point in range(0,maxNumPoints):
#     complexNumberList.append(contourPoints[point,0,0] + contourPoints[point,0,1]*1j)

# FourierDescriptors = np.fft.fft(complexNumberList)

# shiftedFourierDescriptors = np.fft.fftshift(FourierDescriptors)

# magShiftedFourierDescriptors = np.abs(shiftedFourierDescriptors)
# intMagShiftedFourierDescriptors = magShiftedFourierDescriptors.astype(np.int)
# pt.figure()
# pt.plot(list(range(0,maxNumPoints)),intMagShiftedFourierDescriptors)

# numRemovePercentage = 99
# numRemoveDescriptor = int((maxNumPoints)*(numRemovePercentage/100))

# shiftedFourierDescriptors[0:int(numRemoveDescriptor/2)] = 0
# shiftedFourierDescriptors[int(-numRemoveDescriptor/2)-1:-1] = 0

# invShiftedFourierDescriptors = np.fft.ifftshift(shiftedFourierDescriptors)
# invFourierDescriptors = np.fft.ifft(invShiftedFourierDescriptors)

# invRowCoordinates = invFourierDescriptors.real.astype(np.int)
# invColCoordinates = invFourierDescriptors.imag.astype(np.int)


# invContourPoints = contourPoints.copy()
# invContourPoints[:,0,0] = invRowCoordinates
# invContourPoints[:,0,1] = invColCoordinates
# invContours = contours.copy()
# invContours[0] = invContourPoints

# [row,col,depth] = scissors.shape
# blankImg = np.zeros((row,col,depth),dtype=np.uint8)
# blankImgContours = cv2.drawContours(blankImg,invContours,-1,(255,0,0),5)
# pt.figure()
# pt.imshow(cv2.cvtColor(blankImgContours,cv2.COLOR_BGR2RGB)) 

# #====================================Exercise 5=================================

# # k usually falls within the range of 0.04 - 0.06
# # larger k value returns smaller R value (overall it produces less false corners, but might miss more real corners)
# # lower k value returns larger R value (overall it produces more corners but might have more false corners)
# # if R value of a point is more than certain threashold, then the point is considered as a corner point.
# dst = cv2.cornerHarris(scissorsGray, 3, 3, 0.05)
# scissorsCorners = scissors.copy()
# scissorsCorners[dst > 0.01*dst.max() ] = [255, 0, 0]

# pt.figure()
# pt.imshow(cv2.cvtColor(scissorsCorners, cv2.COLOR_BGR2RGB))