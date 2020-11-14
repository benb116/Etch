import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from scipy.spatial import distance_matrix
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
    c = convolve(line.astype(int), kernel, mode='nearest')
    # c = c / 25
    return c / 25


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


# ori
# 1 2 3
# 4   6
# 7 8 9
def NeiEdge(G, im, ori):
    # Look for pixels with pixels above them in correct direction
    # Convolve to get pixels that match that criterion
    kernel = np.zeros(9)
    kernel[ori-1] = 1
    kernel = np.reshape(kernel, (3, 3))
    c = convolve(im.astype(int), kernel, mode='constant')

    # Get the XY coords of pixels that are in the original image and the convolved image
    cities = getCities(np.multiply(c == 1, im))
    cities = [p for p in cities if ((p[0] != 0) & (p[1] != 0))]  # Filter out zeros

    if ori % 2 == 1 and len(list(G.nodes())) > 0:
        cities = [p for p in cities if (G.degree(p) == 1)]

    # Their neighbors are below them (and left or right)
    nextCitiesX = np.array([x[0] for x in cities]) - int((ori-1) % 3 - 1)
    nextCitiesY = np.array([x[1] for x in cities]) - int(ori // 3.3 - 1)
    nextCities = list(zip(nextCitiesX, nextCitiesY))
    d = 1 + (0.414 * (ori % 2 == 1))
    dlist = np.full(nextCitiesX.shape, d)
    G.add_weighted_edges_from(tuple(zip(cities, nextCities, dlist)))

    return G


# Given a list of node numbers, return a bool map
def NodeMap(all_coord):
    maxc = np.max(all_coord).astype(int)
    A = np.zeros((maxc+1, maxc+1))
    all_x = [x[1] for x in all_coord]
    all_y = [y[0] for y in all_coord]
    A[all_x, all_y] = 1
    return A


# Find nearest component and return the edge to add
def GrowBorder(A, sg_node_coord):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    B = np.zeros(A.shape)
    sg_x = [x[1] for x in sg_node_coord]
    sg_y = [x[0] for x in sg_node_coord]
    B[sg_x, sg_y] = 1
    C = A.copy()
    C[sg_x, sg_y] = 0
    connected = False
    # Rewrite this without loop
    # Get matrix of distances from closest B
    # Argmin the closest one and connect

    # Rewrite to shift matrix instead of convolve
    while not connected:
        # print('loop')
        if np.all(B == 1):
            connected = True
            break

        B = (convolve(B, kernel, mode='constant') > 0).astype(int)
        # print('c')
        # links = np.logical_and(B, C)
        links = ~((B == 0) | (C == 0))
        # print('d')
        if np.sum(links) > 0:
            connected = True
            linkcoord = getCities(links)
            ld = distance_matrix(sg_node_coord, linkcoord)
            (z, nei) = np.unravel_index(ld.argmin(), ld.shape)
            minN = sg_node_coord[z]
            minNei = linkcoord[nei]
            return (minN, minNei, ld[z, nei])
