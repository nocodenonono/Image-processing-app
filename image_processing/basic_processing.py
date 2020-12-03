"""
Module contains functions that perform some basic image operations
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def scale(img, factor):
    """
    scale the image based on factor.
    :param img: target image to scale
    :param factor: scale factor
    :return: rescaled image
    """
    if factor <= 0:
        raise ValueError

    if factor >= 1:
        return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    else:
        return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


def blur(img, sigma):
    """
    performs gaussian filter on the target image.
    User specify kernel sigma
    :param img: image
    :param sigma: standard deviation of gaussian kernel
    """
    sigma = int(sigma)
    fil = cv2.getGaussianKernel(sigma * 3, sigma)
    fil = fil * fil.T

    return cv2.filter2D(img, -1, fil)


def cartoonize(img):
    """
    cartoonize an image, to speed up, the images are scaled down before
    applying filters.

    https://towardsdatascience.com/building-an-image-cartoonization-web-app-with-python-382c7c143b0d
    :param img: target image
    :return: cartoon image
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # scale down the image and the apply median blur, then scale up.
    gray_img = scale(cv2.medianBlur(scale(gray_img, 0.5), 5), 2)

    edges = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

    img = scale(cv2.bilateralFilter(scale(img, 0.5), 4, 20, 4), 2)

    return cv2.bitwise_and(img, img, mask=edges)


def grad(im):
    """
    find the absolute value of the sum of vertical gradient and horizontal gradient
    :param im: rbg image
    :return: gradient map of the input image in gray scale
    """
    im = im.copy()
    im = cv2.cvtColor(im * 255, cv2.COLOR_RGB2GRAY)
    im = cv2.GaussianBlur(im, (3, 3), 0) / 255
    fil1 = np.array([1, 0, -1])

    x_grad = np.abs(cv2.filter2D(im, -1, fil1))
    y_grad = np.abs(cv2.filter2D(im.T, -1, fil1))
    return x_grad + y_grad.T


def seam_finding(img):
    """
    find the best seam given cost map
    :param img: cost map
    :return: accumulating cost map and corresponding backtrack
    """
    h, w = img.shape

    backtrack = np.zeros_like(img)
    dp = img.copy()
    for i in range(1, h):
        for j in range(w):
            idx = np.argmin(dp[i - 1, max(0, j - 1): min(j + 1, w - 1)])
            if j > 0:
                idx -= 1
            backtrack[i, j] = j + idx
            dp[i, j] = dp[i, j] + min(dp[i - 1, max(0, j - 1): min(j + 1, w - 1)])

    return dp, backtrack


def seam_mask(img):
    """
    find the mask of the image that masks out the lowest cost seam
    :param img: cost map
    :return: seam mask
    """
    e_map, backtrack = seam_finding(img)

    smallest_seam_idx = np.argmin(e_map[-1])

    seam = np.ones_like(img, dtype=bool)
    cur_j = smallest_seam_idx
    for i in range(img.shape[0] - 1, -1, -1):
        cur_j = int(cur_j)
        seam[i, cur_j] = False
        cur_j = backtrack[i, cur_j]

    return seam


def remove_seam(img, mask):
    """
    given rgb image and mask, remove the pixels according to the mask
    :param img: rgb image
    :param mask: seam mask
    :return: resized image
    """
    h, w, c = img.shape
    return img.copy()[mask].reshape(h, w - 1, c)


def seam_carving(img, num_seam):
    """
    resizing the image in 1 dimension.
    Default resize along the width
    :param img: img to resize
    :param num_seam: size to reduce
    :param mode: if given as 'vertical', resize in the vertical direction
    :return: resized image
    """
    img = img.copy()

    for _ in range(num_seam):
        e = grad(img)
        mask = seam_mask(e)
        img = remove_seam(img, mask)

    return img


def contrast_enhancement(img, clip=40):
    """
    Contrast enhancement using
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    :param img: input rgb image
    :param clip: clip threshold
    :return: enhanced rgb image
    """
    img = np.array(img.copy())
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    clahe = cv2.createCLAHE(clipLimit=clip)
    L = img[:, :, 0]
    img[:, :, 0] = clahe.apply(L)

    return cv2.cvtColor(img, cv2.COLOR_LAB2RGB)


def histogram_equalization(img):
    """
    Histogram color equalization
    :param img: input rgb image
    :return: enhanced rgb image
    """
    img = np.array((img.copy()))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    L = img[:, :, 0]
    img[:, :, 0] = cv2.equalizeHist(L)

    return cv2.cvtColor(img, cv2.COLOR_LAB2RGB)


def histogram_visualization(img):
    """
    plot histogram of pixel values in L channel of LAB color space
    """
    img = np.array((img.copy()))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = img[:, :, 0]

    fig = plt.figure()
    plt.hist(L.ravel(), bins='auto')
    fig.savefig('plot.png')

