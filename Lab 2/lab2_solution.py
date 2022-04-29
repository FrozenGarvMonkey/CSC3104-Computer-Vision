#===============================A. 3D Surface Plot============================
import cv2
import numpy as np
from matplotlib import pyplot as pt
from mpl_toolkits import mplot3d

#Automatically close all the existing figure windows
pt.close('all')

img = cv2.imread("cameraman.png", 0)
img = cv2.resize(img, (128,128))

[nrow,ncol] = img.shape
[xCoor, yCoor] = np.mgrid[0:nrow, 0:ncol]

pt.figure()
ax = pt.axes(projection='3d')
ax.plot_surface(xCoor, yCoor, img, cmap=pt.cm.jet)
pt.show()

#Exercise
#Create the test images
img_1a = np.zeros((5,5))
img_1a[:,0:2] = 255

img_1b = np.zeros((5,5))
img_1b[1:4,1:4] = 255

img_1c = np.zeros((5,5))
img_1c[0,0] = 255
img_1c[1,1] = 255
img_1c[2,2] = 255
img_1c[3,3] = 255
img_1c[4,4] = 255

[nrow,ncol] = img_1a.shape
[xCoor, yCoor] = np.mgrid[0:nrow, 0:ncol]

#In care you want to have multiple surface plot in one figure window
fig = pt.figure()
ax = fig.add_subplot(1,3,1, projection='3d')
ax.plot_surface(xCoor, yCoor, img_1a, cmap=pt.cm.jet)

ax = fig.add_subplot(1,3,2, projection='3d')
ax.plot_surface(xCoor, yCoor, img_1b, cmap=pt.cm.jet)

ax = fig.add_subplot(1,3,3, projection='3d')
ax.plot_surface(xCoor, yCoor, img_1c, cmap=pt.cm.jet)

#================================B. Image Gradient=============================
img = np.array([[100,100,100],[100,0,100],[100,100,200]],dtype=np.float32)
kernel_hori = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=np.float32)/2
kernel_vert = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=np.float32)/2

grad_hori = cv2.filter2D(img,-1,kernel_hori)
grad_vert = cv2.filter2D(img,-1,kernel_vert)

grad_mag = np.sqrt(np.power(grad_hori,2)+np.power(grad_vert,2))

orientation_rad = np.arctan2(grad_vert,grad_hori)
orientation_deg = (orientation_rad*180)/np.pi

#Exercise
#Create the test images
img_2a = np.array([[100,100,100],[100,200,100],[100,100,50]],dtype=np.float32)
img_2b = np.array([[100,100,50],[50,200,100],[100,100,20]],dtype=np.float32)

#Convolution
grad_hori_img_2a = cv2.filter2D(img_2a,-1,kernel_hori)
grad_vert_img_2a = cv2.filter2D(img_2a,-1,kernel_vert)

#Calculate the magnitude
grad_mag_img_2a = np.sqrt(np.power(grad_hori_img_2a,2)+np.power(grad_vert_img_2a,2))

#Calculate the orientation
orientation_rad_img_2a = np.arctan2(grad_vert_img_2a,grad_hori_img_2a)
orientation_deg_img_2a = (orientation_rad_img_2a*180)/np.pi

#Convolution
grad_hori_img_2b = cv2.filter2D(img_2b,-1,kernel_hori)
grad_vert_img_2b = cv2.filter2D(img_2b,-1,kernel_vert)

#Calculate the magnitude
grad_mag_img_2b = np.sqrt(np.power(grad_hori_img_2b,2)+np.power(grad_vert_img_2b,2))

#Calculate the orientation
orientation_rad_img_2b = np.arctan2(grad_vert_img_2b,grad_hori_img_2b)
orientation_deg_img_2b = (orientation_rad_img_2b*180)/np.pi

[nrow,ncol] = img_2a.shape
[xCoor, yCoor] = np.mgrid[0:nrow, 0:ncol]

fig = pt.figure()
ax = fig.add_subplot(1,2,1, projection='3d')
ax.plot_surface(xCoor, yCoor, img_2a, cmap=pt.cm.jet)

ax = fig.add_subplot(1,2,2, projection='3d')
ax.plot_surface(xCoor, yCoor, img_2b, cmap=pt.cm.jet)

#================================C. Image Gradient=============================
#Exercsie
img = cv2.imread('lena.bmp',0)

sobel_hori = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobel_vert = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

sobel_hori_mag = np.abs(sobel_hori)
sobel_vert_mag = np.abs(sobel_vert)

combined_mag = (0.5*sobel_hori_mag) + (0.5*sobel_vert_mag)

pt.figure()
pt.subplot(1,3,1)
pt.imshow(sobel_hori_mag,cmap="gray")
pt.title("Changes in Horizontal Direction (Vertical Edges)")
pt.subplot(1,3,2)
pt.imshow(sobel_vert_mag,cmap="gray")
pt.title("Changes in Vertical Direction (Horizontal Edges)")
pt.subplot(1,3,3)
pt.imshow(combined_mag,cmap="gray")
pt.title("Combined")

#Exercise
#Repeat the above but using an image that is affected by noise
img = cv2.imread('noisy_lena.bmp',0)

sobel_hori = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobel_vert = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

sobel_hori_mag = np.abs(sobel_hori)
sobel_vert_mag = np.abs(sobel_vert)

combined_mag = (0.5*sobel_hori_mag) + (0.5*sobel_vert_mag)

pt.figure()
pt.subplot(1,3,1)
pt.imshow(sobel_hori_mag,cmap="gray")
pt.title("Changes in Horizontal Direction (Vertical Edges), No Filtering")
pt.subplot(1,3,2)
pt.imshow(sobel_vert_mag,cmap="gray")
pt.title("Changes in Vertical Direction (Horizontal Edges), No Filtering")
pt.subplot(1,3,3)
pt.imshow(combined_mag,cmap="gray")
pt.title("Combined, No Filtering")

#Repeat the above but using an image that is affected by noise
#But we will filter the noise first before proceed to find the edges
img = cv2.imread('noisy_lena.bmp',0)

img = cv2.blur(img,(5,5))

sobel_hori = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobel_vert = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

sobel_hori_mag = np.abs(sobel_hori)
sobel_vert_mag = np.abs(sobel_vert)

combined_mag = (0.5*sobel_hori_mag) + (0.5*sobel_vert_mag)

pt.figure()
pt.subplot(1,3,1)
pt.imshow(sobel_hori_mag,cmap="gray")
pt.title("Changes in Horizontal Direction (Vertical Edges), After Filtering")
pt.subplot(1,3,2)
pt.imshow(sobel_vert_mag,cmap="gray")
pt.title("Changes in Vertical Direction (Horizontal Edges), After Filtering")
pt.subplot(1,3,3)
pt.imshow(combined_mag,cmap="gray")
pt.title("Combined, No Filtering")

#=============================D. Canny Edge Detection==========================
#Apply canny edge detection to an image that is not affected by noise
img = cv2.imread('lena.bmp',0)
edges = cv2.Canny(img,50,150)

pt.figure()
pt.subplot(1,3,1)
pt.imshow(edges,cmap="gray")

#Apply canny edge detection to an image that is affected by noise
img = cv2.imread('noisy_lena.bmp',0)
edges = cv2.Canny(img,50,150)

#Canny edge detection will filter the image before finding the edges
#But might not be "sufficient" to eliminate all the noise
pt.subplot(1,3,2)
pt.imshow(edges,cmap="gray")

#Perform "additional" filtering before proceed to find the edges
img = cv2.blur(img,(3,3))
edges = cv2.Canny(img,50,150)

pt.subplot(1,3,3)
pt.imshow(edges,cmap="gray")