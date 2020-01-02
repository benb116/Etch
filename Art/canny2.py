from PIL import Image
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy import signal
from scipy.ndimage import convolve, gaussian_filter
import networkx as nx

uIg = []
uline = []
Mag = []
M = []

def updEdge(nIg, thr):
  global uIg, uline, Mag, M
  if len(uIg) == 0:
    Ig = nIg

  # plt.imshow(Ig)
  # plt.show()
  if len(uline) == 0:
    line = Ig
    line = blur(line)
    line = blur(line)
    line = blur(line)

    Mag, Magx, Magy, Ori = findDerivatives(line)
    M = nonMaxSup(Mag, Ori)

  print(thr)
  oline = (M & (Mag > thr))

  oline = thinOut(oline)
  oline = despeck(oline)

  # plt.imshow(line)
  # plt.show()
  print('Line done')
  print('aaaaa', np.sum(oline))
  # cities = getCities(line.astype(bool))
  return oline

def getEdgePoints(Ig):
  if len(Ig.shape) > 2:
    Ig = 255 - rgb2gray(Ig)

  # plt.imshow(Ig)
  # plt.show()
  line = Ig
  line = blur(line)
  line = blur(line)
  line = blur(line)
  # plt.imshow(line)
  # plt.show()
  Mag, Magx, Magy, Ori = findDerivatives(line)
  M = nonMaxSup(Mag, Ori)

  line = (M & (Mag > 80000))

  line = thinOut(line)
  line = despeck(line)

  # plt.imshow(line)
  # plt.show()
  print('Line done')

  # cities = getCities(line.astype(bool))
  return line

def findDerivatives(I_gray):
    # TODO: your code here

  dy = np.array([1,-1]).reshape(2,1)
  dx = np.array([1,-1]).reshape(1,2)

  # Convolve image with derivative of gaussian to get horz and vert gradient
  Magx = signal.convolve2d(I_gray,dx,'same')
  Magy = signal.convolve2d(I_gray,dy,'same')

  # Gradient magnitude and orientation
  Mag = np.sqrt(Magx*Magx + Magy*Magy)
  Ori = np.arctan2(-Magx,Magy)

  return Mag, Magx, Magy, Ori

def nonMaxSup(Mag, Ori):
  # TODO: your code here
  print('Start NMS')
  
  # Set up interpolation of gradient magnitude
  xmax, ymax = Mag.shape
  x = np.arange(0, xmax)
  y = np.arange(0, ymax)

  interp_spline = RectBivariateSpline(x, y, Mag)

  # Find each offset values to use for interpolation
  # Use a unit length in the direction of the gradient at each point
  Xoffset = np.cos(Ori)
  Yoffset = np.sin(Ori)

  Y, X = np.meshgrid(y,x)

  PlusPointX = X+Xoffset
  PlusPointY = Y+Yoffset
  MinuPointX = X-Xoffset
  MinuPointY = Y-Yoffset

  # Interpolate to get magnitude at those values

  PlusMag = interp_spline.ev(PlusPointX, PlusPointY)
  MinuMag = interp_spline.ev(MinuPointX, MinuPointY)

  # Compare pixel to two neighbors
  GreatPlus = ((Mag-PlusMag) > 0)
  GreatMinu = ((Mag-MinuMag) > 0)

  # Only pixels that are greater than both neighbors (1x1)
  M = np.multiply(GreatPlus, GreatMinu).astype(int)

  return M

def rgb2gray(I_rgb):
  r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
  I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return I_gray

def thinOut(line):
  kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
  c = convolve(line, kernel, mode='constant')
  return np.multiply((c < 3), line) > 0

def despeck(line):
  kernel = np.ones((5, 5))
  kernel[2,2] = 0
  c = convolve(line, kernel, mode='constant')
  return np.multiply((c > 0), line) > 0

def blur(line):
  kernel = np.ones((5, 5))
  c = convolve(line.astype(int), kernel, mode='constant')
  return c

def getCities(line):
  xmax, ymax = line.shape
  x = np.arange(0, xmax)
  y = np.arange(0, ymax)
  Y, X = np.meshgrid(y,x)
  xp = Y[line]
  yp = X[line];
  cities = (np.transpose([xp, yp]))
  return cities

if __name__ == '__main__':
  folder = '/Users/Ben/Desktop/'
  im_path = os.path.join(folder, 'test.jpg')

  I = np.array(Image.open(im_path).convert('RGB'))
  line = getEdgePoints(I)
  # cities = getCities(line.astype(bool))
  # np.set_printoptions(threshold=sys.maxsize)
  # print(np.array2string(cities, separator=','))