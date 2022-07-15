import cv2 as cv
import numpy as np
from scipy import spatial
from matplotlib import pyplot as plt

def compare_imgs(img_1,img_2):
    plt.subplots(1,2)
    plt.subplot(1,2,1)
    plt.imshow(img_1)
    plt.subplot(1,2,2)
    plt.imshow(img_2)
    plt.show()

def make_mask(b, image):
    mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
    for xx, yy in enumerate(b):
        mask[yy:, xx] = 255

    return mask


def display_mask(b, image, color=[255, 255, 255]):    
    result = image.copy()
    overlay = np.full(image.shape, color, image.dtype)

    cv.addWeighted(
        cv.bitwise_or(overlay, overlay, mask=make_mask(b, image)),
        1,
        image,
        1,
        0,
        result
    )

    compare_imgs(image, result)


def color_to_gradient(image):
    gray = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    return np.hypot(
        cv.Sobel(gray, cv.CV_64F, 1, 0),
        cv.Sobel(gray, cv.CV_64F, 0, 1)
    )


def energy(b_tmp, image):
    sky_mask = make_mask(b_tmp, image)

    ground = np.ma.array(
        image,
        mask=cv.cvtColor(cv.bitwise_not(sky_mask), cv.COLOR_GRAY2BGR)
    ).compressed()
    sky = np.ma.array(
        image,
        mask=cv.cvtColor(sky_mask, cv.COLOR_GRAY2BGR)
    ).compressed()
    ground.shape = (ground.size//3, 3)
    sky.shape = (sky.size//3, 3)

    sigma_g, mu_g = cv.calcCovarMatrix(
        ground,
        None,
        cv.COVAR_NORMAL | cv.COVAR_ROWS | cv.COVAR_SCALE
    )
    sigma_s, mu_s = cv.calcCovarMatrix(
        sky,
        None,
        cv.COVAR_NORMAL | cv.COVAR_ROWS | cv.COVAR_SCALE
    )

    y = 2

    return 1 / (
        (y * np.linalg.det(sigma_s) + np.linalg.det(sigma_g)) +
        (y * np.linalg.det(np.linalg.eig(sigma_s)[1]) +
            np.linalg.det(np.linalg.eig(sigma_g)[1]))
    )

def calculate_border(grad, t):
    sky = np.full(grad.shape[1], grad.shape[0])

    for x in range(grad.shape[1]):
        border_pos = np.argmax(grad[:, x] > t)

        # argmax hax return 0 if nothing is > t
        if border_pos > 0:
            sky[x] = border_pos

    return sky

def calculate_border_optimal(image, thresh_min=5, thresh_max=600, search_step=5):
    grad = color_to_gradient(image)

    n = ((thresh_max - thresh_min) // search_step) + 1

    b_opt = None
    jn_max = 0

    for k in range(1, n + 1):
        t = thresh_min + ((thresh_max - thresh_min) // n - 1) * (k - 1)

        b_tmp = calculate_border(grad, t)
        jn = energy(b_tmp, image)

        if jn > jn_max:
            jn_max = jn
            b_opt = b_tmp

    return b_opt


def no_sky_region(bopt, thresh1, thresh2, thresh3):
    border_ave = np.average(bopt)
    asadsbp = np.average(np.absolute(np.diff(bopt)))

    return border_ave < thresh1 or (border_ave < thresh2 and asadsbp > thresh3)


def partial_sky_region(bopt, thresh4):
    return np.any(np.diff(bopt) > thresh4)


def refine_sky(bopt, image):
    sky_mask = make_mask(bopt, image)

    ground = np.ma.array(
        image,
        mask=cv.cvtColor(cv.bitwise_not(sky_mask), cv.COLOR_GRAY2BGR)
    ).compressed()
    sky = np.ma.array(
        image,
        mask=cv.cvtColor(sky_mask, cv.COLOR_GRAY2BGR)
    ).compressed()
    ground.shape = (ground.size//3, 3)
    sky.shape = (sky.size//3, 3)

    ret, label, center = cv.kmeans(
        np.float32(sky),
        2,
        None,
        (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10,
        cv.KMEANS_RANDOM_CENTERS
    )

    sigma_s1, mu_s1 = cv.calcCovarMatrix(
        sky[label.ravel() == 0],
        None,
        cv.COVAR_NORMAL | cv.COVAR_ROWS | cv.COVAR_SCALE
    )
    ic_s1 = cv.invert(sigma_s1, cv.DECOMP_SVD)[1]

    sigma_s2, mu_s2 = cv.calcCovarMatrix(
        sky[label.ravel() == 1],
        None,
        cv.COVAR_NORMAL | cv.COVAR_ROWS | cv.COVAR_SCALE
    )
    ic_s2 = cv.invert(sigma_s2, cv.DECOMP_SVD)[1]

    sigma_g, mu_g = cv.calcCovarMatrix(
        ground,
        None,
        cv.COVAR_NORMAL | cv.COVAR_ROWS | cv.COVAR_SCALE
    )
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
        cnt = np.sum(np.less(
            spatial.distance.cdist(
                image[0:bopt[x], x],
                mu_s,
                'mahalanobis',
                VI=ics
            ),
            spatial.distance.cdist(
                image[0:bopt[x], x],
                mu_g,
                'mahalanobis',
                VI=icg
            )
        ))

        if cnt < (bopt[x] / 2):
            bopt[x] = 0

    return bopt


def detect_sky(image):
    bopt = calculate_border_optimal(image)

    if no_sky_region(bopt, image.shape[0]/30, image.shape[0]/4, 5):
        print("No sky detected")
        return


    elif partial_sky_region(bopt, image.shape[1]/3):
        bnew = refine_sky(bopt, image)
        display_mask(bnew, image)

    else:
        display_mask(bopt, image)

if __name__ == '__main__':
    input_image = cv.imread("fixtures/full_sky.png")
    detect_sky(input_image)

