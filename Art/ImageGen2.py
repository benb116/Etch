from PIL import Image
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy import signal
from scipy.ndimage import convolve, gaussian_filter
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import networkx as nx

from canny2 import getEdgePoints
from postman import single_chinese_postman_path

# np.set_printoptions(threshold=sys.maxsize)

# Binning and hatching parameters
ints = np.array([30, 30, 80, 100, 130, 200])
spacing = np.array([3, 3, 9, 15, 19, 0])
orientation = np.array([-1, 1, -1, 1, -1, 1])
offset = np.array([0,0,0,0,0,0])

def main():
    folder = '/Users/Ben/Desktop/'
    im_path = os.path.join(folder, 'Steve.jpg')
    print(1)
    I = np.array(Image.open(im_path).convert('RGB'))
    Ig = rgb2gray(I)
    print(2)
    sI = Ig
    sI = blur(sI)
    sI = bin(sI)
    print(3)
    # edgeline = getEdgePoints(Ig)
    print(4)
    sumImage = np.zeros(sI.shape)
    # sumImage = edgeline
    alledges = []
    for s in range(0, len(ints)-1):
        h = buildHatch(sI.shape, orientation[s], spacing[s], offset[s])
        mh = (h & (sI <= ints[s]))
        alledges.append(createDiagEdges(mh, orientation[s]))
        sumImage = (sumImage + mh) > 0    

    sumImage = despeck(sumImage)
    # plt.imshow(1-sumImage, cmap='gray', vmin=0, vmax=1)
    # plt.show()
    print(5)
    G = genGraph(alledges)
    # print(list(G.degree()))
    # print(len(nOdeg(G)))
    # stops = np.array([u for u, v in nx.eulerian_circuit(G)])

    ec, stops = single_chinese_postman_path(G)


    np.set_printoptions(threshold=sys.maxsize)
    print('pts')
    print(np.array2string(stops, separator=','))



def rgb2gray(I_rgb):
    r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
    I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return I_gray

def bin(Ig):
    nI = Ig.flatten()

    A, B = np.meshgrid(ints, nI)
    d = np.argmin(np.abs(A-B), axis=1)
    sI = ints[d].reshape(np.shape(Ig))
    return sI

def blur(line):
    kernel = np.ones((5, 5))
    c = convolve(line.astype(int), kernel, mode='constant')/25
    return c

def despeck(line):
    kernel = np.ones((5, 5))
    kernel[2,2] = 0
    c = convolve(line, kernel, mode='constant')
    return np.multiply((c > 0), line) > 0

def getCities(cmap):
    xmax, ymax = cmap.shape
    x = np.arange(0, xmax)
    y = np.arange(0, ymax)
    X, Y = np.meshgrid(y,x)
    xp = X[cmap]
    yp = Y[cmap];
    cities = tuple(zip(xp, yp))
    return cities

def buildHatch(shape, direction, spacing, offset):
    a = np.zeros(shape)
    m = np.arange(0, shape[0])
    n = np.arange(0, shape[1])
    A, B = np.meshgrid(m, n)

    return np.transpose(abs(A + B * direction) % (spacing + 1) == offset)

def getEndPts(mask, ori):
    kernel = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
    c = convolve(mask.astype(int), kernel, mode='constant')
    endpts = (c == 1) & mask

    cities = getCities(endpts)
    lane = (cities[:,0] + cities[:,1] * ori)
    con1 = np.tile(lane, (len(cities), 1))
    con2 = np.transpose(con1)
    match = (con1 == con2)
    np.fill_diagonal(match, False)
    print(np.sum(match, axis=0))


    plt.imshow(endpts)
    plt.show()

def createDiagEdges(mask, ori):
    G = nx.Graph()

    kernel = np.array([[(.5 - .5*ori), 0, (.5 + .5*ori)], [0, 0, 0], [0, 0, 0]])
    c = convolve(mask.astype(int), kernel, mode='constant')
    startN = (c == 1) & mask
    cities = getCities(startN)
    cities = [p for p in cities if ((p[0] != 0) & (p[1] != 0))]

    nextCitiesX = np.array([x[0] for x in cities]) + ori
    nextCitiesY = np.array([x[1] for x in cities]) - 1
    nextCities = list(zip(nextCitiesX, nextCitiesY))

    G.add_nodes_from(cities)
    G.add_nodes_from(nextCities)
    G.add_edges_from(tuple(zip(cities, nextCities)))

    edges = []
    subs = (list(nx.connected_components(G)))
    for s in subs:    
        n1 = [v for v, d2 in G.degree(list(s)) if d2 == 1]
        edges.append(n1)
    return edges

def pythag(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def n1deg(T):
    return [v for v, d2 in T.degree() if d2 == 1]

def nOdeg(T):
    return [v for v, d2 in T.degree() if d2 % 2 == 1]

def CloseDegreeOneNodes(T, d):
    print('a')
    nodes_one_degree = n1deg(T)
    nnodes = {k: v for v, k in enumerate(T.nodes())}
    dnodes = {k: v for k, v in enumerate(T.nodes())}
    dOne = d[nodes_one_degree, :]
    print('a2')
    dOne = dOne[:, nodes_one_degree]
    closedOne = dOne < 5
    clipped = np.tril(closedOne)
    pairs = np.argwhere(clipped)
    for p in pairs:
        T.add_edge(nodes_one_degree[p[0]], nodes_one_degree[p[1]])
    return T

def MatchDegOne(T, d, n1):
    nodes_one_degree = n1
    dOne = d[nodes_one_degree, :]
    print('a3')
    dOne = dOne[:, nodes_one_degree]
    row_ind, col_ind = linear_sum_assignment(dOne)

    added = []
    for j in range(0,len(nodes_one_degree)):
        if j == col_ind[col_ind[j]]:
            if not (j in added or col_ind[j] in added):
                added.append(j)
                added.append(col_ind[j])
                T.add_edge(nodes_one_degree[j], nodes_one_degree[col_ind[j]])
    return T

def LinkDegOne(T, d):
    nodes_one_degree = n1deg(T)
    nnodes = {k: v for v, k in enumerate(T.nodes())}
    dnodes = {k: v for k, v in enumerate(T.nodes())}
    indOne = [nnodes[x] for x in nodes_one_degree]
    dOne = d[indOne, :]
    dOne = dOne[:, indOne]
    row_ind, col_ind = linear_sum_assignment(dOne)

    added = []
    for j in range(0,len(nodes_one_degree)):
        # if j == col_ind[col_ind[j]]:
            # if not (j in added or col_ind[j] in added):
                added.append(j)
                added.append(col_ind[j])
                # print(nodes_one_degree[j], nodes_one_degree[col_ind[j]])
                T.add_edge(nodes_one_degree[j], nodes_one_degree[col_ind[j]], weight=dOne[j, col_ind[j]])

    return T

def LinkDegOdd(T, d):
    nodes_odd_degree = nOdeg(T)
    nnodes = {k: v for v, k in enumerate(T.nodes())}
    dnodes = {k: v for k, v in enumerate(T.nodes())}
    indOne = [nnodes[x] for x in nodes_odd_degree]
    dOne = d[indOne, :]
    dOne = dOne[:, indOne]
    row_ind, col_ind = linear_sum_assignment(dOne)

    added = []
    for j in range(0,len(nodes_odd_degree)):
        # if j == col_ind[col_ind[j]]:
            # if not (j in added or col_ind[j] in added):
                added.append(j)
                added.append(col_ind[j])
                # print(nodes_odd_degree[j], nodes_odd_degree[col_ind[j]])
                T.add_edge(nodes_odd_degree[j], nodes_odd_degree[col_ind[j]], weight=dOne[j, col_ind[j]])

    return T

def ConnectSubgraphs(G, d):
    sub_graphs = nx.connected_components(G)
    nnodes = {k: v for v, k in enumerate(G.nodes())}
    dnodes = {k: v for k, v in enumerate(G.nodes())}
    # For each subgraph, connect it to the closest node in a different subgraph
    # addedges = []
    for i, sg in enumerate(sub_graphs):
        sg = list(sg)
        no = [nnodes[x] for x in sg]
        # Don't look at nodes in same subgraph
        a,b = np.meshgrid(no, no)
        # d[a,b] = 100000
        # Get distances to all nodes from sg nodes
        A = d[no,:]
        A[:, no] = 100000
        (z, nei) = np.unravel_index(A.argmin(), A.shape)
        minN = sg[z]
        if A[z, nei] == 100000:
            break
        # addedges.append((minN, dnodes[nei], A[z,nei]))
        G.add_edge(minN, dnodes[nei], weight=A[z,nei])

    # G.add_weighted_edges_from(addedges)
    # print(len(list(nx.connected_components(G))))
    return G



def genGraph(es):
    G = nx.Graph()
    alledges = [item for sublist in es for item in sublist]
    wedges = list(map(lambda e: (e[0], e[1], pythag(e[0], e[1])), alledges))
    G.add_weighted_edges_from(wedges)
    lnodes = G.nodes()
    d = distance_matrix(lnodes, lnodes)
    np.fill_diagonal(d, 100000)
    # print(len(nOdeg(G)))
    # print(len(n1deg(G)))
    # print(len(list(nx.connected_components(G))))
    # print([x[2] for x in list(G.edges.data('weight'))])
    print('step', 6)

    G = ConnectSubgraphs(G, d)
    # print(len(nOdeg(G)))
    # print(len(n1deg(G)))
    # print(len(list(nx.connected_components(G))))
    # print([x[2] for x in list(G.edges.data('weight'))])
    print('step', 7)
    # G = LinkDegOne(G, d)
    # print(len(nOdeg(G)))
    # print(len(n1deg(G)))
    # print(len(list(nx.connected_components(G))))
    # print([x[2] for x in list(G.edges.data('weight'))])
    # print('step', 8)
    # G = LinkDegOdd(G, d)
    # print(len(nOdeg(G)))
    # print(len(n1deg(G)))
    # print(len(list(nx.connected_components(G))))
    # print([x[2] for x in list(G.edges.data('weight'))])

    return G


if __name__ == '__main__':
    main()