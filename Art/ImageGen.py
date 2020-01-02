from canny2 import updEdge
from artUtils import *

import networkx as nx
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.ndimage import convolve, gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import numpy as np
import sys
import os
from PIL import Image


# Binning and hatching parameters
ints = np.array([30, 30, 80, 100, 130, 150])
spacing = np.array([5, 5, 9, 15, 19, 0])
orientation = np.array([-1, 1, -1, 1, -1, 1])
offset = np.array([0, 0, 0, 0, 0, 0])
thresh = 100000

folder = '/Users/Ben/Desktop/'
im_path = os.path.join(folder, 'Mona.jpg')
print(1)
Im = np.array(Image.open(im_path).convert('RGB'))
Ig = rgb2gray(Im)
blurred = []
binIm = []
sI = []
edIm = np.zeros(Ig.shape)
alledges = [[] for i in ints]
hatchedgim = [[] for i in ints]


def main():
    global Ig, blurred, thresh 
    Ig = rgb2gray(Im)
    blurred = blur(Ig)
    # INIT
    sumImage = Update(0, 0, 0, 0, 1)
    sumImage = Update(1, 0, 0, 0, 0)
    sumImage = Update(0, 1, 0, 0, 0)
    sumImage = Update(0, 2, 0, 0, 0)
    sumImage = Update(0, 3, 0, 0, 0)
    sumImage = Update(0, 4, 0, 0, 0)
    sumImage = Update(0, 5, 0, 0, 0)


    # fig, ax = plt.subplots()
    # plt.subplots_adjust(left=0.25, bottom=0.25)

    # axcolor = 'lightgoldenrodyellow'
    # axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    # axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    # sfreq = Slider(axfreq, 'Thresh', 1000, 200000, valinit=100000)
    # samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=5)

    # sfreq.on_changed(Update)
    # samp.on_changed(Update)

    # resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    # button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    # def reset(event):
    #     sfreq.reset()
    #     samp.reset()
    # button.on_clicked(reset)

    # plt.show()

    print(np.sum(sumImage))
    plt.imshow(1-sumImage, cmap='gray', vmin=0, vmax=1)
    plt.show()
    print(5)
    G = genGraph(alledges)
    print(list(G.degree()))
    print(len(nOdeg(G)))
    print(nx.is_eulerian(G))

def Update(nint, sp, ori, off, edge):
    if edge:
        print(1)
        global edIm
        edIm = updEdge(Ig, thresh)
    if nint:
        print(2)
        global binIm
        binIm = bin(blurred, ints)
    if sp or ori or off:
        print(sp, ori, off)
        global alledges, hatchedgim
        valid = (int(sp==0) + int(ori==0) + int(off==0))
        if valid != 2:
            print('hmmm')
        s = sp + ori + off - 1
        h = buildHatch(binIm.shape, orientation[s], spacing[s], offset[s])
        mh = (h & (binIm <= ints[s]))
        # plt.imshow(1-mh, cmap='gray', vmin=0, vmax=1)
        # plt.show()
        alledges[s] = createDiagEdges(mh, orientation[s])
        hatchedgim[s] = mh > 0

    hIm = np.zeros(Ig.shape)
    for hI in hatchedgim:
        if (len(hI) != 0):
            hIm = (hIm + hI) > 0
    sumImage = edIm + hIm
    sumImage = despeck(sumImage)
    return sumImage



def buildHatch(shape, direction, spacing, offset):
    a = np.zeros(shape)
    m = np.arange(0, shape[0])
    n = np.arange(0, shape[1])
    A, B = np.meshgrid(m, n)

    return np.transpose(abs(A + B * direction) % (spacing + 1) == offset)


def createDiagEdges(mask, ori):
    G = nx.Graph()

    kernel = np.array(
        [[(.5 - .5*ori), 0, (.5 + .5*ori)], [0, 0, 0], [0, 0, 0]])
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

def LinkDegOne(T, d):
    nodes_one_degree = n1deg(T)
    dOne = d[nodes_one_degree, :]
    dOne = dOne[:, nodes_one_degree]
    row_ind, col_ind = linear_sum_assignment(dOne)

    added = []
    for j in range(0, len(nodes_one_degree)):
        # if j == col_ind[col_ind[j]]:
        if not (j in added or col_ind[j] in added):
            added.append(j)
            added.append(col_ind[j])
            # print(nodes_one_degree[j], nodes_one_degree[col_ind[j]])
            # print(dOne[j, col_ind[j]])
            T.add_edge(
                nodes_one_degree[j], nodes_one_degree[col_ind[j]], weight=dOne[j, col_ind[j]])

    a = nx.adjacency_matrix(T)
    d = d + a * 100000
    return T, d


def LinkDegOdd(T, d):
    nodes_odd_degree = nOdeg(T)
    dOne = d[nodes_odd_degree, :]
    dOne = dOne[:, nodes_odd_degree]
    row_ind, col_ind = linear_sum_assignment(dOne)

    added = []
    for j in range(0, len(nodes_odd_degree)):
        # if j == col_ind[col_ind[j]]:
            if not (j in added or col_ind[j] in added):
                added.append(j)
                added.append(col_ind[j])
                # print(nodes_odd_degree[j], nodes_odd_degree[col_ind[j]])
                # print(dOne[j, col_ind[j]])
                T.add_edge(
                    nodes_odd_degree[j], nodes_odd_degree[col_ind[j]], weight=dOne[j, col_ind[j]])

    a = nx.adjacency_matrix(T)
    d = d + a * 100000
    return T, d


def ConnectSubgraphs(G, d):
    sub_graphs = nx.connected_components(G)
    # d = d[G.nodes(), :]
    alln = list(G.nodes())
    # d = d[:, alln]
    # For each subgraph, connect it to the closest node in a different subgraph
    # addedges = []
    for i, sg in enumerate(list(sub_graphs)):
        no = list(sg)
        # Don't look at nodes in same subgraph
        # a, b = np.meshgrid(no, no)
        # d[a,b] = 100000
        # Get distances to all nodes from sg nodes
        A = d[no, :]
        A[:, no] = 100000
        A = A[:, alln]
        (z, nei) = np.unravel_index(A.argmin(), A.shape)
        minN = no[z]
        minNei = alln[nei]
        if A[z, nei] >= 100000:
            break
        # addedges.append((minN, dnodes[nei], A[z,nei]))
        # print(A[z, nei])
        G.add_edge(minN, minNei, weight=A[z, nei])

    a = nx.adjacency_matrix(G)
    d = d + a * 100000
    return G, d


def genGraph(es):
    G = nx.MultiGraph()
    alledges = [item for sublist in es for item in sublist]
    wedges = list(map(lambda e: (e[0], e[1], pythag(e[0], e[1])), alledges))

    allnodes = list(map(lambda e: [e[0], e[1]], alledges))
    allnodes = [item for sublist in allnodes for item in sublist]
    allnodes = list(set(allnodes))
    negnodes = [(n[0], -n[1]) for n in allnodes]

    nnodes = {v: k for k, v in enumerate(allnodes)}
    dnodes = {k: v for k, v in enumerate(allnodes)}
    ndnodes = {k: v for k, v in enumerate(negnodes)}

    nwedges = list(map(lambda e: (nnodes[e[0]], nnodes[e[1]], e[2]), wedges))
    G.add_weighted_edges_from(nwedges)

    d = distance_matrix(allnodes, allnodes)
    np.fill_diagonal(d, 100000)
    a = nx.adjacency_matrix(G)
    d = d + a * 100000
    print(len(nOdeg(G)))
    print(len(n1deg(G)))
    print(len(list(nx.connected_components(G))))
    # print([x[2] for x in list(G.edges.data('weight'))])
    print('step', 6)
    # nx.draw(G, dnodes, node_size=0.2)
    # plt.show()

    G, d = LinkDegOne(G, d)
    # nx.draw(G, dnodes, node_size=0.2)
    # plt.show()
    print('step', 7)
    G, d = LinkDegOne(G, d)
    print(len(nOdeg(G)))
    print(len(n1deg(G)))
    # nx.draw(G, dnodes, node_size=0.2)
    # plt.show()
    print('step', 8)
    print(len(list(nx.connected_components(G))))
    G, d = ConnectSubgraphs(G, d)
    # a = nx.adjacency_matrix(G)
    # d = d + a * 100000
    G, d = ConnectSubgraphs(G, d)
    # a = nx.adjacency_matrix(G)
    # d = d + a * 100000
    G, d = ConnectSubgraphs(G, d)
    print(len(list(nx.connected_components(G))))
    print('step', 9)
    G, d = LinkDegOdd(G, d)
    G, d = LinkDegOdd(G, d)
    G, d = LinkDegOdd(G, d)
    print('step', 10)

    nx.draw(G, ndnodes, node_size=0.1, width=.2)
    plt.show()
    print(len(nOdeg(G)))
    print(len(n1deg(G)))
    # print(len(list(nx.connected_components(G))))
    # print([x[2] for x in list(G.edges.data('weight'))])

    stops = [list(dnodes[u]) for u, v in nx.eulerian_circuit(G)]
    # cities = dnodes[stops]
    np.set_printoptions(threshold=sys.maxsize)
    print('pts')
    print(stops)
    # print(np.array2string(stops, separator=','))

    return G


if __name__ == '__main__':
    main()
