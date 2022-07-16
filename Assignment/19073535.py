""" 
Combination of two Sky Segmentation Implementations

1. Sky Detection in Images for Solar Exposure Prediction : https://core.ac.uk/download/pdf/35462487.pdf
2. Sky Region Detection in a Single Image for Autonomous Ground Robot Navigation: https://journals.sagepub.com/doi/pdf/10.5772/56884

Algorithm inspired by Journal 2 and Channel extraction inspired by Journal 1

"""

import cv2 as cv
import numpy as np
from scipy import spatial
from matplotlib import pyplot as plt
import glob
import os

#global iteration value to track images
ITER = 0

# Optional Function in order to compare original images to their masked/sky_segmented versions
def compare_imgs(img_1,img_2):
    plt.subplots(1,2)
    plt.subplot(1,2,1)
    plt.imshow(img_1)
    plt.subplot(1,2,2)
    plt.imshow(img_2)
    plt.show()

# Iterates through image to create mask
def make_mask(b, image):
    mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
    for xx, yy in enumerate(b):
        mask[yy:, xx] = 255

    return mask

# Saves images based on order 
def save_mask(img):
    cv.imwrite(("output/image_{}.jpg".format(ITER)), img)


# Applies mask and saves image (Can optionally also uncomment compare_imgs in order to view images and their masks side-by-side)
def display_mask(b, image, color=[0, 0, 0]):    
    result = cv.bitwise_and(image, image, mask=cv.bitwise_not(make_mask(b, image))) # Mask inverted in order to save images of segmented sky
    #compare_imgs(image, result)
    save_mask(result)


# Image is first converted to grayscale and then 
# calculates the gradient magnitude by combining two gradient images (horizontal and vertical)
# generated using sobel functions.
def color_to_gradient(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return np.hypot(cv.Sobel(gray, cv.CV_64F, 1, 0), cv.Sobel(gray, cv.CV_64F, 0, 1))

# Energy Function extracted directly from Journal 2
def energy_function(b_tmp, image):
    sky_mask = make_mask(b_tmp, image)

    ground = np.ma.array(image, mask=cv.cvtColor(cv.bitwise_not(sky_mask), cv.COLOR_GRAY2BGR)).compressed()
    sky = np.ma.array(image, mask=cv.cvtColor(sky_mask, cv.COLOR_GRAY2BGR)).compressed()
    
    ground.shape = (ground.size//3, 3)
    sky.shape = (sky.size//3, 3)

    sigma_g, mu_g = cv.calcCovarMatrix(ground, None, cv.COVAR_NORMAL | cv.COVAR_ROWS | cv.COVAR_SCALE)
    sigma_s, mu_s = cv.calcCovarMatrix(sky, None, cv.COVAR_NORMAL | cv.COVAR_ROWS | cv.COVAR_SCALE)

    y = 2

    return 1 / (
        (y * np.linalg.det(sigma_s) + np.linalg.det(sigma_g)) +
        (y * np.linalg.det(np.linalg.eig(sigma_s)[1]) +
            np.linalg.det(np.linalg.eig(sigma_g)[1]))
    )

# Detect position of horizon. Taken from Journal 2.
def calculate_border(grad, t):
    sky_b = np.full(grad.shape[1], grad.shape[0])

    for x in range(grad.shape[1]):
        border_pos = np.argmax(grad[:, x] > t)

        if border_pos > 0:
            sky_b[x] = border_pos

    return sky_b

# Optimize energy function jn. 
def calculate_border_optimal(image, thresh_min=5, thresh_max=600, search_step=5):
    grad = color_to_gradient(image)

    n = ((thresh_max - thresh_min) // search_step) + 1

    b_opt = None
    jn_max = 0

    for k in range(1, n + 1):
        t = thresh_min + ((thresh_max - thresh_min) // n - 1) * (k - 1)

        b_tmp = calculate_border(grad, t)
        jn = energy_function(b_tmp, image)

        if jn > jn_max:
            jn_max = jn
            b_opt = b_tmp

    return b_opt

# Called to check if only partial sky is detected.
def PartialSky(bopt, thresh4):
    return np.any(np.diff(bopt) > thresh4)


def RefineSkyImage(bopt, image):
    sky_mask = make_mask(bopt, image)

    ground = np.ma.array(image, mask=cv.cvtColor(cv.bitwise_not(sky_mask), cv.COLOR_GRAY2BGR)).compressed()
    
    sky = np.ma.array(image, mask=cv.cvtColor(sky_mask, cv.COLOR_GRAY2BGR)).compressed()
    
    ground.shape = (ground.size//3, 3)
    sky.shape = (sky.size//3, 3)

    ret, label, center = cv.kmeans(np.float32(sky), 2, None, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv.KMEANS_RANDOM_CENTERS)

    sigma_s1, mu_s1 = cv.calcCovarMatrix(sky[label.ravel() == 0], None, cv.COVAR_NORMAL | cv.COVAR_ROWS | cv.COVAR_SCALE) 
    ic_s1 = cv.invert(sigma_s1, cv.DECOMP_SVD)[1]

    sigma_s2, mu_s2 = cv.calcCovarMatrix(sky[label.ravel() == 1], None, cv.COVAR_NORMAL | cv.COVAR_ROWS | cv.COVAR_SCALE)
    ic_s2 = cv.invert(sigma_s2, cv.DECOMP_SVD)[1]

    sigma_g, mu_g = cv.calcCovarMatrix(ground, None, cv.COVAR_NORMAL | cv.COVAR_ROWS | cv.COVAR_SCALE)
    
    icg = cv.invert(sigma_g, cv.DECOMP_SVD)[1]

    if cv.Mahalanobis(mu_s1, mu_g, ic_s1) > cv.Mahalanobis(mu_s2, mu_g, ic_s2):
        mu_s = mu_s1
        sigma_s = sigma_s1
        ics = ic_s1
        
    else:
        mu_s = mu_s2
        sigma_s = sigma_s2
        ics = ic_s2

    for x in range(image.shape[1]):
        cnt = np.sum(np.less(spatial.distance.cdist(image[0:bopt[x], x], mu_s, 'mahalanobis', VI=ics), spatial.distance.cdist(image[0:bopt[x], x], mu_g, 'mahalanobis', VI=icg)))

        if cnt < (bopt[x] / 2):
            bopt[x] = 0

    return bopt



def SkyDetect(image):
    optimal_border_img = calculate_border_optimal(image)

    if NoSkyRegion(optimal_border_img, image.shape[0]/30, image.shape[0]/4, 5):
        print("No sky detected")
        return


    elif PartialSky(optimal_border_img, image.shape[1]/3):
        refined_optimal_border = RefineSkyImage(optimal_border_img, image)
        display_mask(refined_optimal_border, image)

    else:
        display_mask(optimal_border_img, image)

        

# Called to check if the image contains no sky region based on the optimal border algorithm.
def NoSkyRegion(bopt, thresh1, thresh2, thresh3):
    border_ave = np.average(bopt)
    abs_sum_abs_diff = np.average(np.absolute(np.diff(bopt)))

    return border_ave < thresh1 or (border_ave < thresh2 and abs_sum_abs_diff > thresh3)


# Driver Function
def main():
    dir = input("Enter Folder Name (Must be in root): ")
    output_dir = "output"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ext = ["png", "jpg", "jpeg"]

    files = []
    [files.extend(glob.glob("imgs/" + dir + "/*." + e)) for e in ext]
    images = [cv.imread(file) for file in files]

    
    for i in images:
        global ITER 
        ITER = ITER + 1
        SkyDetect(i)
    
if __name__ == '__main__':
    main()
