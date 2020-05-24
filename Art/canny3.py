# import the necessary packages
import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from artUtils import *


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def extractEdges(imagePath):
    # load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # blurred = cv2.GaussianBlur(blurred, (3, 3), 0)
    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    auto = despeck(auto_canny(blurred))
    # plt.imshow(despeck(np.array(auto)))
    # plt.show()
    return auto


if __name__ == '__main__':
    folder = '/Users/Ben/Desktop/Etch'
    imagePath = os.path.join(folder, 'Spongebob.jpg')

    auto = extractEdges(imagePath)
    # show the image
    # plt.imshow(despeck(np.array(auto)))
    # plt.show()