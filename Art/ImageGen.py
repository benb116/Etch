from canny2 import updEdge
from artUtils import *

import networkx as nx
import itertools
import pandas as pd
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
ints = np.array([10, 10, 20, 40, 60, 170]) # Brightness cutoffs
spacing = np.array([5, 5, 9, 15, 19, 0]) # Corresponding line densities
orientation = np.array([-1, 1, -1, 1, -1, 1]) # Direction (not all the same way)
offset = np.array([0, 0, 0, 0, 0, 0]) # Any offsets
thresh = 75000 # Canny edge threshold

folder = '/Users/Ben/Desktop/Etch/'
im_path = os.path.join(folder, 'Girl2.jpg')
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
    global Ig, blurred, thresh # Find a better way than global vars
    Ig = rgb2gray(Im)
    blurred = blur(Ig) # Smooth out any small specks
    # Build up a matrix from various bins and operations
    sumImage = Update(0, 0, 0, 0, 1) # Edge detection
    sumImage = Update(1, 0, 0, 0, 0) # Set up bin images
    sumImage = Update(0, 1, 0, 0, 0) # Hatch the bin images
    sumImage = Update(0, 2, 0, 0, 0)
    sumImage = Update(0, 3, 0, 0, 0)
    sumImage = Update(0, 4, 0, 0, 0)
    sumImage = Update(0, 5, 0, 0, 0)

    plt.imshow(1-sumImage, cmap='gray', vmin=0, vmax=1)
    plt.show()
    print('Start graph work')
    G, stops = genGraph(alledges)

    # np.set_printoptions(threshold=sys.maxsize)
    stopstring = 'asd'
    # stopstring = np.array_str(stops, separator=',')
    # print(stopstring)
    file1 = open("Stops.txt", "w")
    file1.write(stopstring)
    file1.close()

    # Checks
    print(list(G.degree()))
    print(len(nOdeg(G)))
    print(nx.is_eulerian(G))


def Update(nint, sp, ori, off, edge):
    if edge:
        print('Run edge detection')
        global edIm, canedges
        edIm = updEdge(Ig, thresh)
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
    cpts = getCities(cIm)
    cd = distance_matrix(cpts, cpts)
    np.fill_diagonal(cd, 100000)
    ca = cd < 2
    C = nx.from_numpy_matrix(np.multiply(cd, ca))
    cane = C.edges()
    ce = list(map(lambda e: [cpts[e[0]], cpts[e[1]]], cane))
    return ce

# Create evenly-spaced lines in a matrix
# shape: size of matrix
# direction: (1 or -1) * 45 deg
# spacing: # of pixels between lines
# offset: shift left or right (so hatches don't overlap each other)
def buildHatch(shape, direction, spacing, offset):
    a = np.zeros(shape)
    m = np.arange(0, shape[0])
    n = np.arange(0, shape[1])
    A, B = np.meshgrid(m, n)

    return np.transpose(abs(A + B * direction) % (spacing + 1) == offset)

# Given lines from buildHatch, turn matrix points into long edges
def createDiagEdges(mask, ori):
    G = nx.Graph()

    # Look for pixels with pixels above them in correct direction
    # Convolve to get pixels that match that criterion
    kernel = np.array(
        [[(.5 - .5*ori), 0, (.5 + .5*ori)], [0, 0, 0], [0, 0, 0]])
    c = convolve(mask.astype(int), kernel, mode='constant')
    # These points that are also in the mask are the good ones
    startN = (c == 1) & mask
    cities = getCities(startN) # Get the XY coords
    cities = [p for p in cities if ((p[0] != 0) & (p[1] != 0))] # Filter out zeros

    # Their neighbors are below them (and left or right)
    nextCitiesX = np.array([x[0] for x in cities]) + ori
    nextCitiesY = np.array([x[1] for x in cities]) - 1
    nextCities = list(zip(nextCitiesX, nextCitiesY))

    # Add to temp graph
    G.add_nodes_from(cities)
    G.add_nodes_from(nextCities)
    G.add_edges_from(tuple(zip(cities, nextCities)))

    # Each component is a long edge
    edges = []
    subs = (list(nx.connected_components(G)))
    for s in subs:
        # Extract the endpoints (degree == 1)
        n1 = [v for v, d2 in G.degree(list(s)) if d2 == 1]
        edges.append(n1)
    return edges

# Connect separate subgraphs into one big subgraph with optimal new edges
def ConnectSubgraphs(G, d):
    sub_graphs = list(nx.connected_components(G))
    all_nodes = list(G.nodes())
    # For each subgraph, connect it to the closest node in a different subgraph
    # Repeat until there's only one
    while len(list(nx.connected_components(G))) > 1:
        print('Subgraph loop')
        for i, sg in enumerate(sub_graphs):
            sg_nodes = list(sg) # Get all nodes in the subgraph
            # Get distances to other nodes not in the subgraph
            A = d[sg_nodes, :]
            A[:, sg_nodes] = 100000
            A = A[:, all_nodes]
            # Find the shortest connection between a node in sg and 
            # a node not in sg
            (z, nei) = np.unravel_index(A.argmin(), A.shape)
            minN = sg_nodes[z]
            minNei = all_nodes[nei]
            # If the distance is large, we're done
            if A[z, nei] >= 100000:
                break
            # Add the edge
            G.add_edge(minN, minNei, weight=A[z, nei])

        sub_graphs = nx.connected_components(G)

    # Mark the distance matrix with new edges
    a = nx.adjacency_matrix(G)
    d = d + a * 100000
    return G, d

# Add a parallel edge to any degee 1 node with an odd node as a neighbor
def EasyLinkOne(G, d):
    n1 = n1deg(G)
    # Get the neighbor for each d1 node (and the neighbors degree)
    n1nei = list(map(lambda e: list(G.neighbors(e))[0], n1))
    neideg = list(map(lambda e: G.degree(e) % 2 == 1, n1nei))
    # Get only d1 nodes with an odd degree neighbor (and get that neighbor)
    filtern1 = [i for indx, i in enumerate(n1) if neideg[indx] is True]
    filterne = [i for indx, i in enumerate(n1nei) if neideg[indx] is True]
    # Add parallel edges
    for n1, ne in zip(filtern1, filterne):
        G.add_edge(n1, ne, d[n1, ne])

    a = nx.adjacency_matrix(G)
    d = d + a * 100000
    return G, d

# Add a parallel edge to any odd node with an odd node as a neighbor
def EasyLinkOdd(G, d):
    nO = nOdeg(G)
    nOnei = list(map(lambda e: list(G.neighbors(e))[0], nO))
    neideg = list(map(lambda e: G.degree(e) % 2 == 1, nOnei))

    filternO = [i for indx, i in enumerate(nO) if neideg[indx] is True]
    filterne = [i for indx, i in enumerate(nOnei) if neideg[indx] is True]

    for nO, ne in zip(filternO, filterne):
        G.add_edge(nO, ne, d[nO, ne])

    a = nx.adjacency_matrix(G)
    d = d + a * 100000
    return G, d

# Link remaining d1 nodes to a good neighbor
def LinkDegOne(T, d):
    nodes_one_degree = n1deg(T)
    # Get distances from d1 nodes to all other nodes
    dOne = d[nodes_one_degree, :]
    # Find best option to add an edge
    row_ind, col_ind = linear_sum_assignment(dOne)
    print('Assignment complete')
    # added = []
    for j in range(0, len(nodes_one_degree)):
        # if not (j in added or col_ind[j] in added):
        # added.append(j)
        # added.append(col_ind[j])
        # print(nodes_one_degree[j], nodes_one_degree[col_ind[j]])
        # print(dOne[j, col_ind[j]])
        T.add_edge(
            nodes_one_degree[j], col_ind[j], weight=dOne[j, col_ind[j]])

    a = nx.adjacency_matrix(T)
    d = d + a * 100000
    return T, d

# Link odd nodes together but only if they are each other's best odd
def LinkDegOdd(T, d):
    nodes_odd_degree = nOdeg(T)
    # Get distances from odd nodes to other odd nodes
    dOne = d[nodes_odd_degree, :]
    dOne = dOne[:, nodes_odd_degree]
    # Find best match
    row_ind, col_ind = linear_sum_assignment(dOne)
    print('Assignment complete')

    # Keep track of which nodes we've connected
    added = []
    for j in range(0, len(nodes_odd_degree)):
        # If a node is it's best node's best node
        if j == col_ind[col_ind[j]]:
            # If we haven't already connected either
            if not (j in added or col_ind[j] in added):
                added.append(j)
                added.append(col_ind[j])
                T.add_edge(
                    nodes_odd_degree[j], nodes_odd_degree[col_ind[j]],
                    weight=dOne[j, col_ind[j]])

    a = nx.adjacency_matrix(T)
    d = d + a * 100000
    return T, d

# Link remaining odd nodes by adding paralle paths between them
# Try to find minimum weight matching
def PathFinder(G, d):
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
    np.fill_diagonal(A, 1000000) # No self loops
    A2 = A[nO, :]
    A2 = A2[:, nO]
    
    # Get all combinations of two odd nodes
    odd_node_pairs = list(itertools.combinations(nO, 2))
    print('Num pairs', len(odd_node_pairs))
    # Construct a temp complete graph of odd nodes
    # edge weights = path lengths (negative for max weight matching)
    O = nx.Graph()
    for op in odd_node_pairs:
        O.add_edge(op[0], op[1], weight=-A[op[0], op[1]])

    # This gets optimal matching, probably can do near-optimal matching
    print('Begin matching')
    odd_matching_dupes = nx.algorithms.max_weight_matching(O, True)
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
            G.add_edge(p[i], p[i+1], weight=d[p[i], p[i+1]])

    a = nx.adjacency_matrix(G)
    d = d + a * 100000
    return G, d

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
    # Create dicts that can translate between node # and node coord
    nnodes = {v: k for k, v in enumerate(allnodes)}
    dnodes = {k: v for k, v in enumerate(allnodes)}
    ndnodes = {k: v for k, v in enumerate(negnodes)}
    # Add the edges with node #'s instead of coords (makes things easier)
    nwedges = list(map(lambda e: (nnodes[e[0]], nnodes[e[1]], e[2]), wedges))
    G.add_weighted_edges_from(nwedges)

    # Create a distance matrix for all nodes to all other nodes
    d = distance_matrix(allnodes, allnodes)
    np.fill_diagonal(d, 100000) # No self loops
    # This will prevent certain steps from adding parallel edges
    a = nx.adjacency_matrix(G) 
    d = d + a * 100000

    # Need an eulerian circuit
    # One component, no odd degree nodes
    # Where do we stand
    print("Num Odd", len(nOdeg(G))) # Number of odd degree nodes
    print("Num One", len(n1deg(G))) # Number of degree one nodes
    print("Num components", len(list(nx.connected_components(G))))

    # Connect any separate subgraphs into one
    print('Connect Subgraphs')
    G, d = ConnectSubgraphs(G, d)
    print("Num Odd", len(nOdeg(G))) # Number of odd degree nodes
    print("Num One", len(n1deg(G))) # Number of degree one nodes
    print("Num components", len(list(nx.connected_components(G))))

    # Add a parallel edge to any d1 node with an odd node as a neighbor
    print('EasyLinkOne')
    G, d = EasyLinkOne(G, d)
    print("Num Odd", len(nOdeg(G))) # Number of odd degree nodes
    print("Num One", len(n1deg(G))) # Number of degree one nodes

    # Add a parallel edge to any odd node with an odd node as a neighbor
    print('EasyLinkOdd')
    G, d = EasyLinkOdd(G, d)
    print("Num Odd", len(nOdeg(G))) # Number of odd degree nodes
    print("Num One", len(n1deg(G))) # Number of degree one nodes
    # nx.draw_networkx_edges(G, ndnodes, node_size=0.01, width=.2)
    # nx.draw_networkx_nodes(G, ndnodes, nodelist=nOdeg(G), node_size=0.02)
    # plt.axis('scaled')
    # plt.show()

    # Link remaining d1 nodes to a good neighbor
    # This may create new edges
    print('LinkDegOne')
    G, d = LinkDegOne(G, d)
    print("Num Odd", len(nOdeg(G))) # Number of odd degree nodes
    print("Num One", len(n1deg(G))) # Number of degree one nodes
    # nx.draw_networkx_edges(G, ndnodes, node_size=0.01, width=.2)
    # nx.draw_networkx_nodes(G, ndnodes, nodelist=nOdeg(G), node_size=0.02)
    # plt.axis('scaled')
    # plt.show()

    # Link odd nodes together but only if they are each other's best odd
    # This may create new edges
    print('LinkDegOdd')
    G, d = LinkDegOdd(G, d)
    print("Num Odd", len(nOdeg(G))) # Number of odd degree nodes
    print("Num One", len(n1deg(G))) # Number of degree one nodes
    # nx.draw_networkx_edges(G, ndnodes, node_size=0.01, width=.2)
    # nx.draw_networkx_nodes(G, ndnodes, nodelist=nOdeg(G), node_size=0.02)
    # plt.axis('scaled')
    # plt.show()

    # Match remaining odd nodes with longer parallel paths
    print('FinalOdd')
    G, d = PathFinder(G, d)
    print("Num Odd", len(nOdeg(G))) # Number of odd degree nodes
    print("Num One", len(n1deg(G))) # Number of degree one nodes
    nx.draw_networkx_edges(G, ndnodes, node_size=0.01, width=.2)
    nx.draw_networkx_nodes(G, ndnodes, nodelist=nOdeg(G), node_size=0.02)
    plt.axis('scaled')
    plt.show()

    print('Begin Eulerian')
    stops = [list(dnodes[u]) for u, v in nx.eulerian_circuit(G)]

    return G, stops


if __name__ == '__main__':
    main()
