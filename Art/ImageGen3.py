from canny2 import getEdgePoints
import networkx as nx
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.ndimage import convolve, gaussian_filter
from scipy import signal
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from PIL import Image

# np.set_printoptions(threshold=sys.maxsize)

# Binning and hatching parameters
ints = np.array([30, 30, 80, 100, 130, 200])
spacing = np.array([5, 5, 9, 15, 19, 0])
orientation = np.array([-1, 1, -1, 1, -1, 1])
offset = np.array([0, 0, 0, 0, 0, 0])


def main():
    folder = '/Users/Ben/Desktop/Etch'
    im_path = os.path.join(folder, 'Steve.jpg')
    print(1)
    Im = np.array(Image.open(im_path).convert('RGB'))
    Ig = rgb2gray(Im)
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
        print('4a')
        h = buildHatch(sI.shape, orientation[s], spacing[s], offset[s])
        mh = (h & (sI <= ints[s]))
        print('4b')
        alledges.append(createDiagEdges(mh, orientation[s]))
        sumImage = (sumImage + mh) > 0

    # buildGraph(alledges)

    sumImage = despeck(sumImage)
    print(np.sum(sumImage))
    plt.imshow(1-sumImage, cmap='gray', vmin=0, vmax=1)
    plt.show()
    print(5)
    G = genGraph(alledges)
    print(list(G.degree()))
    print(len(nOdeg(G)))
    print(nx.is_eulerian(G))
    print(nx.is_connected(G))
    print(all(d % 2 == 0 for v, d in G.degree()))
    


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
    lane = cities[:, 0] + cities[:, 1] * ori
    con1 = np.tile(lane, (len(cities), 1))
    con2 = np.transpose(con1)
    match = (con1 == con2)
    np.fill_diagonal(match, False)
    print(np.sum(match, axis=0))

    plt.imshow(endpts)
    plt.show()


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


def CloseDegreeOneNodes(T, d):
    print('a')
    nodes_one_degree = n1deg(T)
    dOne = d[nodes_one_degree, :]
    print('a2')
    dOne = dOne[:, nodes_one_degree]
    closedOne = dOne < 5
    clipped = np.tril(closedOne)
    pairs = np.argwhere(clipped)
    for p in pairs:
        # print(dOne[p[0], p[1]])
        T.add_edge(nodes_one_degree[p[0]], nodes_one_degree[p[1]], weight=dOne[p[0], p[1]])

    a = nx.adjacency_matrix(T)
    d = d + a * 100000
    return T, d


def MatchDegOne(T, d):
    nodes_one_degree = n1deg(T)
    dOne = d[nodes_one_degree, :]
    print('a3')
    dOne = dOne[:, nodes_one_degree]
    row_ind, col_ind = linear_sum_assignment(dOne)

    added = []
    for j in range(0, len(nodes_one_degree)):
        if j == col_ind[col_ind[j]]:
            if not (j in added or col_ind[j] in added):
                added.append(j)
                added.append(col_ind[j])
                T.add_edge(nodes_one_degree[j], nodes_one_degree[col_ind[j]])
    return T


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


def FixEasyOddDegree(T, d):
    print('c')
    nodes_odd_degree = nOdeg(T)
    # Any odd-degree node with an odd-degree neighbor gets a duplicate edge
    print('c2')
    for odd in nodes_odd_degree:
        if not odd in nodes_odd_degree:
            break
        bors = list(T.neighbors(odd))
        for b in bors:
            if b in nodes_odd_degree and d[odd, b] < 16:
                nodes_odd_degree.remove(b)
                nodes_odd_degree.remove(odd)
                # print(d[odd,b])
                T.add_edge(odd, b, weight=d[odd, b])
                break

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


def buildGraph(edges):
    loc2num = {}
    num2loc = {}
    allnodes = []
    for edgeset in edges:
        setnodes = list(set(list(map(lambda e: (e[0], e[1]), edgeset))))
        setnodes = list(map(lambda pair: [pair[0], pair[1]], setnodes))
        setnodes = [item for sublist in setnodes for item in sublist]
        allnodes.append(setnodes)
        loc2num.update({v: k for k, v in enumerate(setnodes)})
        num2loc.update({k: v for k, v in enumerate(setnodes)})

    print('22222')
    allnodes = [item for sublist in allnodes for item in sublist]
    d = distance_matrix(allnodes, allnodes)
    np.fill_diagonal(d, 100000)
    graphs = []
    for edgeset in edges:
        G = nx.MultiGraph()
        nwedges = list(map(lambda e: (loc2num[e[0]], loc2num[e[1]], pythag(e[0], e[1])), edgeset))
        print(len(nwedges))
        G.add_weighted_edges_from(nwedges)
        nodeloc = list(map(lambda n: num2loc[n], G.nodes()))
        # d = distance_matrix(nodeloc, nodeloc)
        # np.fill_diagonal(d, 100000)
        # d2 = d[G.nodes(), :]
        # d2 = d2[:, G.nodes()]
        G, d = ConnectSubgraphs(G, d)
        graphs.append(G)

    T = nx.MultiGraph()
    for gr in graphs:
        T = nx.compose(T, gr)
    print(898989898999)
    print(len(list(set(allnodes))))
    print(len(T.nodes()))
    print(len(nOdeg(T)))
    print(len(n1deg(T)))
    print(len(T.edges()))
    print('end')


def genGraph(es):
    G = nx.MultiGraph()
    alledges = [item for sublist in es for item in sublist]
    wedges = list(map(lambda e: (e[0], e[1], pythag(e[0], e[1])), alledges))

    allnodes = list(map(lambda e: [e[0], e[1]], alledges))
    allnodes = [item for sublist in allnodes for item in sublist]
    allnodes = list(set(allnodes))

    nnodes = {v: k for k, v in enumerate(allnodes)}
    dnodes = {k: v for k, v in enumerate(allnodes)}

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

    # G, d = ConnectSubgraphs(G, d)
    # a = nx.adjacency_matrix(G)
    # d = d + a * 100000
    # print(len(nOdeg(G)))
    # print(len(n1deg(G)))
    # print(len(list(nx.connected_components(G))))
    # weights = ([x[2] for x in list(G.edges.data('weight'))])
    # print(weights)
    print('step', 7)
    # G, d = CloseDegreeOneNodes(G, d)
    G, d = LinkDegOne(G, d)
    # nx.draw(G, dnodes, node_size=0.2)
    # plt.show()
    G, d = LinkDegOne(G, d)
    print(len(nOdeg(G)))
    print(len(n1deg(G)))
    # nx.draw(G, dnodes, node_size=0.2)
    # plt.show()
    # print(len(list(nx.connected_components(G))))
    # print([x[2] for x in list(G.edges.data('weight'))])
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
    print('step', 10)
    G, d = LinkDegOdd(G, d)
    G, d = LinkDegOdd(G, d)
    G, d = LinkDegOdd(G, d)

    nx.draw(G, dnodes, node_size=0.2, width=0.5)
    plt.show()
    # # G = LinkDegOdd(G, d)
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
