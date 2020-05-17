from canny2 import updEdge
from canny3 import extractEdges
from artUtils import *

import networkx as nx
import itertools
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button, TextBox
import numpy as np
import sys
import os
from PIL import Image


# Binning and hatching parameters
ints = np.array([30, 30, 120, 220])  # Brightness cutoffs
spacing = np.array([7, 7, 13, 15])  # Corresponding line densities
orientation = np.array([-1, 1, -1, 1])  # Direction (not all the same way)
offset = np.array([0, 0, 0, 0])  # Any offsets

folder = '/Users/Ben/Desktop/Etch/'
jpgname = 'Albert'
im_path = os.path.join(folder, jpgname+'.jpg')
Im = np.array(Image.open(im_path).convert('RGB'))
Ig = rgb2gray(Im)

blurred = []
binIm = []
sI = []
edIm = np.zeros(Ig.shape)
canedges = []
alledges = [[] for i in ints]
hatchedgim = [[] for i in ints]


def main():
    global Ig, blurred, thresh  # Find a better way than global vars
    Ig = rgb2gray(Im)
    blurred = blur(Ig)  # Smooth out any small specks
    # Build up a matrix from various bins and operations
    sumImage = Update(0, 0, 0, 0, 1)  # Edge detection
    sumImage = Update(1, 0, 0, 0, 0)  # Set up bin images
    for i in range(0, len(ints)-1):
        sumImage = Update(0, i+1, 0, 0, 0)  # Hatch the bin images

    plt.imshow(1-sumImage, cmap='gray', vmin=0, vmax=1)
    plt.show()
    print('Start graph work')
    G, stops = genGraph(alledges)

    print('Write to file')
    np.set_printoptions(threshold=sys.maxsize)
    stopstring = 'asd'
    stopstring = np.array2string(stops, separator=',')
    # print(stopstring)
    file1 = open(jpgname+".txt", "w")
    file1.write(stopstring)
    file1.close()
    # Checks
    # print(list(G.degree()))
    # print(len(nOdeg(G)))
    # print(nx.is_eulerian(G))


def Update(nint, sp, ori, off, edge):
    if edge:
        print('Run edge detection')
        global edIm, canedges
        edIm = extractEdges(im_path)
        # edIm = updEdge(Ig, 70000)
        print('Link edge detection')
        canedges = ConnectCanny(edIm)
    if nint:
        print('Bin images')
        global binIm
        binIm = bin(blurred, ints)
    if sp or ori or off:
        print('Build hatch', sp, ori, off)
        global alledges, hatchedgim
        valid = (int(sp == 0) + int(ori == 0) + int(off == 0))
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


# Link up adjacent points returned from edge detection
def ConnectCanny(cIm):
    plt.imshow(1-cIm, cmap='gray', vmin=0, vmax=1)
    # plt.show()
    cpts = getCities(cIm)
    print('ConnectCanny', len(cpts))
    G = nx.Graph()

    G = NeiEdge(G, cIm, 2)
    G = NeiEdge(G, cIm, 4)
    G = NeiEdge(G, cIm, 1)
    G = NeiEdge(G, cIm, 3)
    return list(G.edges())


# Create evenly-spaced lines in a matrix
# shape: size of matrix
# direction: (1 or -1) * 45 deg
# spacing:  # of pixels between lines
# offset: shift left or right (so hatches don't overlap each other)
def buildHatch(shape, direction, spacing, offset):
    a = np.zeros(shape)
    m = np.arange(0, shape[0])
    n = np.arange(0, shape[1])
    A, B = np.meshgrid(m, n)

    return np.transpose(abs(A + B * direction) % (spacing + 1) == offset)


# Given lines from buildHatch, turn matrix points into long edges
def createDiagEdges(mask, ori):
    G = NeiEdge(nx.Graph(), mask, (2+ori))

    # Each component is a long edge
    edges = []
    subs = (list(nx.connected_components(G)))
    for s in subs:
        # Extract the endpoints (degree == 1)
        n1 = [v for v, d2 in G.degree(list(s)) if d2 == 1]
        edges.append(n1)
    return edges


# Connect separate subgraphs into one big subgraph with optimal new edges
def ConnectSubgraphs(G, nnodes, dnodes):
    sub_graphs = list(nx.connected_components(G))
    all_coord = [dnodes[x] for x in list(G.nodes())]
    A = NodeMap(all_coord)

    # For each subgraph, connect it to the closest node in a different subgraph
    # Repeat until there's only one
    while len(sub_graphs) > 1:
        print('Subgraph loop', len(sub_graphs))

        for i, sg in enumerate(sub_graphs):
            # print(i)
            sg = nx.node_connected_component(G, list(sg)[0])
            sg_node_num = list(sg)  # Get all nodes in the subgraph
            sg_node_coord = [dnodes[x] for x in sg_node_num]

            newedge = GrowBorder(A, sg_node_coord)
            if newedge is not None:
                G.add_edge(nnodes[newedge[0]], nnodes[newedge[1]], weight=newedge[2])

        sub_graphs = list(nx.connected_components(G))

    return G


# Add a parallel edge to any degee 1 node with an odd node as a neighbor
def EasyLinkOne(G):
    n1 = n1deg(G)
    # Get the neighbor for each d1 node (and the neighbors degree)
    n1nei = list(map(lambda e: list(G.neighbors(e))[0], n1))
    neideg = list(map(lambda e: G.degree(e) % 2 == 1, n1nei))
    # Get only d1 nodes with an odd degree neighbor (and get that neighbor)
    filtern1 = [i for indx, i in enumerate(n1) if neideg[indx] is True]
    filterne = [i for indx, i in enumerate(n1nei) if neideg[indx] is True]
    # Add parallel edges
    for n1, ne in zip(filtern1, filterne):
        G.add_edge(n1, ne, weight=G.get_edge_data(n1, ne)[0]['weight'])
    return G


# Add a parallel edge to any odd node with an odd node as a neighbor
def EasyLinkOdd(G):
    nO = nOdeg(G)
    nOnei = list(map(lambda e: list(G.neighbors(e))[0], nO))
    neideg = list(map(lambda e: G.degree(e) % 2 == 1, nOnei))

    filternO = [i for indx, i in enumerate(nO) if neideg[indx] is True]
    filterne = [i for indx, i in enumerate(nOnei) if neideg[indx] is True]

    for nO, ne in zip(filternO, filterne):
        G.add_edge(nO, ne, weight=G.get_edge_data(nO, ne)[0]['weight'])

    return G


def LinkDegOne(G, nnodes, dnodes):
    nodes_one_degree = n1deg(G)
    one_coord = [dnodes[p] for p in nodes_one_degree]

    nodes_all_degree = list(G.nodes())
    all_coord = [dnodes[p] for p in nodes_all_degree]

    for i, p in enumerate(one_coord):
        if G.degree(nnodes[p]) != 1:
            continue

        # print(i)
        ld = distance_matrix([p], all_coord)
        ld[0, nnodes[p]] = 100000
        nei = ld.argmin()
        G.add_edge(nnodes[p], nnodes[all_coord[nei]], weight=ld[0, nei])
        # newedge = GrowBorder(A, [p])
        # if newedge is not None:
            # G.add_edge(nnodes[newedge[0]], nnodes[newedge[1]], weight=newedge[2])

    return G


# Link odd nodes together but only if they are each other's best odd
def LinkDegOdd(G, nnodes, dnodes):
    nodes_odd_degree = nOdeg(G)
    odd_coord = [dnodes[p] for p in nodes_odd_degree]

    for i, p in enumerate(odd_coord):
        if G.degree(nnodes[p]) % 2 != 1:
            continue

        # print(i)
        ld = distance_matrix([p], odd_coord)
        ld[0, i] = 100000
        nei = ld.argmin()
        if ld[0, nei] < 30:
            G.add_edge(nnodes[p], nnodes[odd_coord[nei]], weight=ld[0, nei])

    return G


# Link remaining odd nodes by adding parallel paths between them
# Try to find minimum weight matching
# https://brooksandrew.github.io/simpleblog/articles/intro-to-graph-optimization-solving-cpp/
def PathFinder(G, dnodes):
    nO = nOdeg(G)
    A = np.zeros((len(list(G.nodes())), len(list(G.nodes()))))
    # Get path lengths from all odd nodes to all nodes
    # Populate a matrix
    print('Get all path lengths')
    for t in nO:
        length = nx.single_source_dijkstra_path_length(G, t)
        lk = list(length.keys())
        lv = list(length.values())
        A[t, lk] = lv

    # Only look at odd to odd paths
    np.fill_diagonal(A, 1000000)  # No self loops
    A2 = A[nO, :]
    A2 = A2[:, nO]

    # Get all combinations of two odd nodes
    odd_node_pairs = list(itertools.combinations(nO, 2))
    print('Num pairs', len(odd_node_pairs))
    # Construct a temp complete graph of odd nodes
    # edge weights = path lengths (negative for max weight matching)
    OddComplete = nx.Graph()
    for op in odd_node_pairs:
        OddComplete.add_edge(op[0], op[1], weight=-A[op[0], op[1]])

    # This gets optimal matching, probably can do near-optimal matching
    print('Begin matching')
    odd_matching_dupes = nx.algorithms.max_weight_matching(OddComplete, True)
    print('End matching')
    odd_matching = list(pd.unique([tuple(sorted([k, v])) for k, v in odd_matching_dupes.items()]))

    # odd_matching = [(3858, 3906),(1464, 1667),(430, 2218),(3512, 5237),(2219, 5044),(2540, 2735),(2268, 2901),(232, 3154),(2262, 2489),(1435, 5097),(1024, 3836),(3509, 3669),(3915, 4076),(862, 1893),(18, 3312),(1784, 2438),(444, 2752),(3455, 4263),(4109, 4571),(1443, 1565),(4425, 4619),(139, 316),(3471, 4869),(1668, 4273),(2119, 2738),(1751, 1978),(2974, 4040),(436, 3846),(3808, 4014),(893, 4753),(741, 3098),(797, 1081),(1120, 2075),(2113, 4679),(779, 3403),(2696, 3911),(829, 1588),(3097, 4540),(1662, 4828),(490, 4892),(756, 3622),(1460, 3866),(583, 2504),(932, 1516),(1752, 3642),(639, 4980),(2143, 3214),(909, 5135),(2270, 3824),(3118, 4118),(4334, 4710),(49, 397),(582, 1645),(1725, 2481),(3038, 4311),(3295, 3879),(476, 3665),(1933, 4625),(1212, 1924),(201, 1634),(1758, 4797),(459, 481),(293, 5196),(937, 5228),(3010, 3417),(885, 3159),(2070, 2750),(1099, 5152),(731, 1118),(3582, 4638),(2988, 4829),(2391, 2509),(1990, 2090),(2829, 4198),(1360, 2874),(2782, 5041),(3426, 3655),(644, 3243),(1042, 1399),(4186, 4562),(1190, 3346),(409, 1523),(3361, 5280),(2388, 4353),(4156, 4499),(2437, 4362),(595, 1186),(4527, 5209),(288, 1570),(1521, 4439),(945, 2368),(840, 4443),(957, 2772),(2451, 4665),(1737, 2006),(56, 238),(2632, 4295),(3015, 5199),(4201, 5040),(1260, 2534),(1370, 2812),(1200, 3055),(90, 4050),(19, 2678),(613, 1174),(1044, 3750),(144, 4852),(2931, 4002),(224, 1969),(3444, 4529),(2332, 3811),(1590, 3392),(3019, 4582),(1112, 3224),(903, 1877),(1793, 2552),(1746, 4915),(4178, 4585),(3595, 5193),(808, 5005),(15, 3591),(999, 2785),(3377, 3525),(1060, 1371),(1278, 4655),(3339, 4494),(2281, 3325),(1385, 5052),(935, 2150),(1088, 2101),(247, 1065),(152, 3711),(570, 1555),(1988, 3050),(1142, 4380),(4996, 5254),(122, 1714),(578, 1254),(1329, 3948),(3360, 3914),(623, 3776),(728, 3108),(658, 1586),(338, 5016),(2566, 3969),(1984, 3668),(1117, 5247),(3763, 5069),(3639, 4244),(531, 1860),(3313, 4889),(2118, 2242),(612, 715),(2365, 3052),(3567, 3657),(1711, 5217),(3008, 4721),(515, 5292),(3175, 3279),(2746, 3350),(4090, 4219),(1965, 2131),(837, 1454),(324, 4845),(160, 2999),(369, 1808),(1031, 2633),(2013, 5014),(1859, 4150)]
    # For each match that is created
    print('Add parallel paths')
    for a in odd_matching:
        # Get the full path between the two
        p = nx.dijkstra_path(G, a[0], a[1])
        # Add parallel edges from the path
        for i in range(0, len(p) - 1):
            w = pythag(dnodes[p[i]], dnodes[p[i+1]])
            G.add_edge(p[i], p[i+1], weight=w)

    return G


# Do the graph theory work
def genGraph(es):
    G = nx.MultiGraph()
    # Collect all edges from all operations
    alledges = [item for sublist in es for item in sublist] + canedges
    # Add weights to them based on pythag distance
    wedges = list(map(lambda e: (e[0], e[1], pythag(e[0], e[1])), alledges))

    # Get all unique nodes in all edges
    allnodes = list(map(lambda e: [e[0], e[1]], alledges))
    allnodes = [item for sublist in allnodes for item in sublist]
    allnodes = list(set(allnodes))
    negnodes = [(n[0], -n[1]) for n in allnodes]
    # Create dicts that can translate between node  # and node coord
    nnodes = {v: k for k, v in enumerate(allnodes)}
    dnodes = {k: v for k, v in enumerate(allnodes)}
    ndnodes = {k: v for k, v in enumerate(negnodes)}
    # Add the edges with node  #'s instead of coords (makes things easier)
    nwedges = list(map(lambda e: (nnodes[e[0]], nnodes[e[1]], e[2]), wedges))
    G.add_weighted_edges_from(nwedges)

    # Need an eulerian circuit
    # One component, no odd degree nodes
    # Where do we stand
    print("Num Odd", len(nOdeg(G)))  # Number of odd degree nodes
    print("Num One", len(n1deg(G)))  # Number of degree one nodes
    print("Num components", len(list(nx.connected_components(G))))

    # Connect any separate subgraphs into one
    # Will add edges
    print('Connect Subgraphs')
    nx.draw_networkx_edges(G, ndnodes, node_size=0.01, width=.2)
    nx.draw_networkx_nodes(G, ndnodes, nodelist=nOdeg(G), node_size=0.02)
    plt.axis('scaled')
    plt.show()
    G = ConnectSubgraphs(G, nnodes, dnodes)
    print("Num Odd", len(nOdeg(G)))  # Number of odd degree nodes
    print("Num One", len(n1deg(G)))  # Number of degree one nodes
    print("Num components", len(list(nx.connected_components(G))))
    nx.draw_networkx_edges(G, ndnodes, node_size=0.01, width=.2)
    nx.draw_networkx_nodes(G, ndnodes, nodelist=nOdeg(G), node_size=0.02)
    plt.axis('scaled')
    plt.show()
    
    # Add a parallel edge to any d1 node with an odd node as a neighbor
    print('EasyLinkOne')
    G = EasyLinkOne(G)
    print("Num Odd", len(nOdeg(G)))  # Number of odd degree nodes
    print("Num One", len(n1deg(G)))  # Number of degree one nodes

    # # Link remaining d1 nodes to a good neighbor
    # # This may create new edges
    print('LinkDegOne')
    G = LinkDegOne(G, nnodes, dnodes)
    print("Num Odd", len(nOdeg(G)))  # Number of odd degree nodes
    print("Num One", len(n1deg(G)))  # Number of degree one nodes
    # nx.draw_networkx_edges(G, ndnodes, node_size=0.01, width=.2)
    # nx.draw_networkx_nodes(G, ndnodes, nodelist=nOdeg(G), node_size=0.02)
    # plt.axis('scaled')
    # plt.show()
    
    # Add a parallel edge to any odd node with an odd node as a neighbor
    print('EasyLinkOdd')
    G = EasyLinkOdd(G)
    print("Num Odd", len(nOdeg(G)))  # Number of odd degree nodes
    print("Num One", len(n1deg(G)))  # Number of degree one nodes
    # nx.draw_networkx_edges(G, ndnodes, node_size=0.01, width=.2)
    # nx.draw_networkx_nodes(G, ndnodes, nodelist=nOdeg(G), node_size=0.02)
    # plt.axis('scaled')
    # plt.show()

    # Create a distance matrix for all nodes to all other nodes
    # d = distance_matrix(allnodes, allnodes)
    # np.fill_diagonal(d, 100000)  # No self loops
    # # This will prevent certain steps from adding parallel edges
    # a = nx.adjacency_matrix(G)
    # d = d + a * 100000
    # nx.write_weighted_edgelist(G, 'aftereasy')
    # G = nx.read_weighted_edgelist('aftereasy')


    # Link odd nodes together but only if they are each other's best odd
    # This may create new edges
    print('LinkDegOdd')
    G = LinkDegOdd(G, nnodes, dnodes)
    print("Num Odd", len(nOdeg(G)))  # Number of odd degree nodes
    print("Num One", len(n1deg(G)))  # Number of degree one nodes
    print('LinkDegOdd 2')
    G = LinkDegOdd(G, nnodes, dnodes)
    print("Num Odd", len(nOdeg(G)))  # Number of odd degree nodes
    print("Num One", len(n1deg(G)))  # Number of degree one nodes
    nx.draw_networkx_edges(G, ndnodes, node_size=0.01, width=.2)
    nx.draw_networkx_nodes(G, ndnodes, nodelist=nOdeg(G), node_size=0.02)
    plt.axis('scaled')
    plt.show()

    # Match remaining odd nodes with longer parallel paths
    print('FinalOdd')
    G = PathFinder(G, dnodes)
    print("Num Odd", len(nOdeg(G)))  # Number of odd degree nodes
    print("Num One", len(n1deg(G)))  # Number of degree one nodes
    # nx.draw_networkx_edges(G, ndnodes, node_size=0.01, width=.2)
    # nx.draw_networkx_nodes(G, ndnodes, nodelist=nOdeg(G), node_size=0.02)
    # plt.axis('scaled')
    # plt.show()

    print('Begin Eulerian')
    stops = [list(dnodes[u]) for u, v in nx.eulerian_circuit(G)]
    stops = np.array(stops)
    return G, stops


if __name__ == '__main__':
    main()
