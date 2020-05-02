import numpy as np
from scipy.ndimage import convolve, gaussian_filter
import networkx as nx

def rgb2gray(I_rgb):
    r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
    I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return I_gray


def bin(Ig, ints):
    nI = Ig.flatten()

    A, B = np.meshgrid(ints, nI)
    d = np.argmin(np.abs(A-B), axis=1)
    sI = ints[d].reshape(np.shape(Ig))
    return sI


def blur(line):
    kernel = np.ones((5, 5))
    c = convolve(line.astype(int), kernel, mode='constant')/25
    return c


def thinOut(line):
  kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
  c = convolve(line, kernel, mode='constant')
  return np.multiply((c < 3), line) > 0


def despeck(line):
    kernel = np.ones((5, 5))
    kernel[2, 2] = 0
    c = convolve(line, kernel, mode='constant')
    return np.multiply((c > 0), line) > 0


def getCities(cmap):
    xmax, ymax = cmap.shape
    x = np.arange(0, xmax)
    y = np.arange(0, ymax)
    X, Y = np.meshgrid(y, x)
    xp = X[cmap]
    yp = Y[cmap]
    cities = tuple(zip(xp, yp))
    return cities


def pythag(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def n1deg(T):
    return [v for v, d2 in T.degree() if d2 == 1]


def nOdeg(T):
    return [v for v, d2 in T.degree() if d2 % 2 == 1]