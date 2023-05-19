# import the necessary packages
import cv2
import os
import matplotlib.pyplot as plt

from artUtils import *

# https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def extractEdges(imagePath, a,b):
    # load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread(imagePath)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = image
    blurred = cv2.GaussianBlur(blurred, (3, 3), 0)
    # blurred = cv2.GaussianBlur(blurred, (3, 3), 0)
    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    auto = cv2.Canny(blurred, a,b)
    auto = (auto > 0) & (despeck(auto) > 0)
    # plt.imshow((np.array(auto)))
    # plt.show()
    return auto


if __name__ == '__main__':
    folder = '/Users/Ben/Desktop/Etch'
    imagePath = os.path.join(folder, 'Spongebob.jpg')

    auto = extractEdges(imagePath)
    # show the image
    # plt.imshow(despeck(np.array(auto)))
    # plt.show()