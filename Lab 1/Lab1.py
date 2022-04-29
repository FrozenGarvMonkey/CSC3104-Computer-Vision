import numpy as np
import cv2

##========================= A: Boundary Extraction ============================

# Create the test image
bi_img = np.zeros((9,9), dtype=np.uint8)
bi_img[1:8,1:8] = 1
bi_img[1,1] = 0
bi_img[7,1] = 0
bi_img[1,7] = 0
bi_img[7,7] = 0

se = np.array([[1,1,1],[1,1,1],[1,1,1]])
err = cv2.erode(bi_img, se, iterations=1)

rse = np.rot90(se,2)
dil = cv2.dilate(bi_img, rse, iterations=1)

inner_boundary = bi_img - err
outer_boundary = dil - bi_img

##=========================== B: Region Filling ==============================

#Create the test image
bi_img = np.zeros((9,9), dtype=np.uint8)
bi_img[1:8, 1:8] = 1
bi_img[1,1] = 0
bi_img[1,7] = 0
bi_img[7,1] = 0
bi_img[7,7] = 0
bi_img[2:7, 2:7] = 0

#Create SE
sE = array = np.ones((3,3), dtype=np.uint8)
sE[0,0] = 0
sE[0,2] = 0
sE[2,0] = 0
sE[2,2] = 0

#Create complement version of the test image
com_bi_img = 1-bi_img

#Create an image for the filling process and set point (2,2) as the starting point
fill_img = np.zeros((9,9), dtype=np.uint8)
fill_img[2,2] = 1

while True:    
     new_fill_img = cv2.dilate(fill_img, sE, iterations=1)    
     new_fill_img = cv2.bitwise_and(new_fill_img,com_bi_img)

     if (np.array_equal(new_fill_img, fill_img)):
         break
     else:
         # Continue the filling process
         fill_img = new_fill_img.copy()

#Fill the empty part
filled_img = bi_img + fill_img

##============================= C: Hit-or-Miss ================================

#Create the test image
bi_img = np.zeros((9,9), dtype=np.uint8)
bi_img[1:8, 1:8] = 1
bi_img[1:8, 4] = 0
bi_img[4, 1:8] = 0
bi_img[5,1] = 0
bi_img[7,1] = 0
bi_img[2,6] = 0
bi_img[5,3] = 0
bi_img[7,3] = 0
bi_img[6,6] = 0
bi_img[1,5] = 0
bi_img[3,5] = 0
bi_img[1,7] = 0
bi_img[3,7] = 0

#Create the first SE (to check the foreground pattern)
cSE = array = np.ones((3,3), dtype=np.uint8)
cSE[0,0] = 0
cSE[0,2] = 0
cSE[2,0] = 0
cSE[2,2] = 0

#Create the second SE (to check the background pattern)
cSEc = np.zeros((3,3), dtype=np.uint8)
cSEc[0,0] = 1
cSEc[0,2] = 1
cSEc[2,0] = 1
cSEc[2,2] = 1

#Create a complement version of the test image
#This image represents the background pixels using '1'
#So that we can use the second SE to match and find the background pattern
com_bi_img = 1-bi_img

#Find points that match with the foreground pattern defined in the first SE
foreground_candidates = cv2.erode(bi_img, cSE, iterations=1)

#Find points that match with the background pattern defined in the second SE
background_candidates = cv2.erode(com_bi_img, cSEc, iterations=1)

#Only keep those points that match with the foreground pattern and background pattern
cross_shape_location_manual = cv2.bitwise_and(foreground_candidates,background_candidates)

#If you prefer to use the built-in function
sE = np.array([[-1,1,-1],[1,1,1],[-1,1,-1]],dtype=np.int)
cross_shape_location_built_in = cv2.morphologyEx(bi_img, cv2.MORPH_HITMISS, sE)

## =============================== D. Thinning ================================

#Create all the SE
sE1 = np.array([[-1,-1,-1],[0,1,0],[1,1,1]],dtype=np.int)
sE2 = np.array([[1,0,-1],[1,1,-1],[1,0,-1]],dtype=np.int)   
sE3 = np.array([[1,1,1],[0,1,0],[-1,-1,-1]],dtype=np.int)
sE4 = np.array([[-1,0,1],[-1,1,1],[-1,0,1]],dtype=np.int)
sE5 = np.array([[0,-1,-1],[1,1,-1],[0,1,0]],dtype=np.int)
sE6 = np.array([[0,1,0],[1,1,-1],[0,-1,-1]],dtype=np.int)
sE7 = np.array([[0,1,0],[-1,1,1],[-1,-1,0]],dtype=np.int)
sE8 = np.array([[-1,-1,0],[-1,1,1],[0,1,0]],dtype=np.int)

#Create the test image
bi_img = np.zeros((9,9), dtype=np.uint8)
bi_img[2:8, 1:8] = 1
bi_img[2:8, 1:8] = 1
bi_img[2:4, 1:5] = 0
bi_img[4, 3  :5] = 0
bi_img[5:8, 7] = 0
bi_img[5:8, 1] = 0
bi_img[7, 3:6] = 0


#Manual way (hardcoded)
output01 = bi_img - cv2.morphologyEx(bi_img, cv2.MORPH_HITMISS, sE1)
output02 = output01 - cv2.morphologyEx(output01, cv2.MORPH_HITMISS, sE2)
output03 = output02 - cv2.morphologyEx(output02, cv2.MORPH_HITMISS, sE3)
output04 = output03 - cv2.morphologyEx(output03, cv2.MORPH_HITMISS, sE4)
output05 = output04 - cv2.morphologyEx(output04, cv2.MORPH_HITMISS, sE5)
output06 = output05 - cv2.morphologyEx(output05, cv2.MORPH_HITMISS, sE6)
output07 = output06 - cv2.morphologyEx(output06, cv2.MORPH_HITMISS, sE7)
output08 = output07 - cv2.morphologyEx(output07, cv2.MORPH_HITMISS, sE8)
output09 = output08 - cv2.morphologyEx(output08, cv2.MORPH_HITMISS, sE1)
output10 = output09 - cv2.morphologyEx(output09, cv2.MORPH_HITMISS, sE2)
output11 = output10 - cv2.morphologyEx(output10, cv2.MORPH_HITMISS, sE3)
output12 = output11 - cv2.morphologyEx(output11, cv2.MORPH_HITMISS, sE4)
output13 = output12 - cv2.morphologyEx(output12, cv2.MORPH_HITMISS, sE5)


#Put all the SE into a list
list_sE = [sE1, sE2, sE3, sE4, sE5, sE6, sE7, sE8]

#Just to record the number of iterations required to get the final output
iteration = 0

#Create a copy of the input image
initial_img = bi_img.copy()
intermediate_img = bi_img.copy()

while True:
    for sE in list_sE:
        #Apply thinning using sE1 to sE8 (one set)
        intermediate_img = intermediate_img - cv2.morphologyEx(intermediate_img, cv2.MORPH_HITMISS, sE)
        iteration = iteration + 1
        
    #Check if the output after applying sE1 to sE8 is still the same as the image before applying sE1 to sE8
    #If still the same, then stop
    if np.sum(intermediate_img-initial_img) == 0:
        break;
    else:
    #If not the same, then we would like to apply sE1 to sE8 (one set) to the image again
    #But before that, we need to replace what we have in initial_img with the image after we applied sE1 to sE8[]
        initial_img = intermediate_img.copy()
        
    
        
        
        
        
        
