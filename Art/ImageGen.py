# Image generation script
# Used to take a jpg image, extract edges and shading,
# then chart a single line path to draw a representation

# See main() for setup
# See genGraph() for graph algorithm

from canny2 import updEdge
from canny3 import extractEdges
from artUtils import *

import networkx as nx
import itertools
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import numpy as np
import sys
import os
from PIL import Image
print('loaded')
# If interactive, show GUI for parameter tuning
# Also show progress after each step
interactive = (len(sys.argv) > 1)
# interactive = True
print(interactive)

# Binning and hatching parameters
ints = np.array([30, 30, 60, 90, 255])  # Brightness cutoffs
spacing = np.array([7, 7, 13, 15, 20])  # Corresponding line densities
orientation = np.array([-1, 1, -1, 1, -1])  # Direction (not all the same way)
offset = np.array([0, 0, 0, 0, 100000])  # Any offsets

# Edge detection parameters
edgethr = [100, 200]

# Pull the image
folder = '/Users/Ben/Desktop/Etch/'
jpgname = 'Angry2'
im_path = os.path.join(folder, jpgname+'.jpg')
Im = np.array(Image.open(im_path).convert('RGB'))
Ig = rgb2gray(Im)

# Vars for storing image, hatch, and edge info
# Used to generate the starting graph
blurred = []  # Blurred image
binIm = []  # Brightness cutoff images

hatchedgim = [[] for i in ints]  # Hatch images
edIm = np.zeros(Ig.shape)  # Edge-detected image

canedges = []  # Detected edges converted to graph edges
alledges = [[] for i in ints]  # Collection of all graph edges


def main():
    global Ig, blurred, thresh

    # First task is to collect graph edges
    # Build up detected edges and hatch edges
    # Also show edge visualization if interactive

    blurred = blur(Ig)  # Smooth out any small specks

    sumImage = Update(0, 0, 0, 0, 1)  # Edge detection
    sumImage = Update(1, 0, 0, 0, 0)  # Set up bin images
    for i in range(0, len(ints)):
        sumImage = Update(0, i+1, 0, 0, 0)  # Hatch the bin images

    SliderFigure(sumImage)  # Show visualization of current results

    # At this point we're ready to build the graph
    # and start connecting edges as necessary
    # This outputs the waypoints of the circuit
    print('Start graph work')
    G, stops = genGraph(alledges)

    print('Write to file')
    np.set_printoptions(threshold=sys.maxsize)
    stopstring = 'asd'
    stopstring = np.array2string(stops, separator=',')
    # print(stopstring)
    file1 = open('../rPi/public/art/'+jpgname+'.json', "w")
    file1.write(FormatFile(stopstring))
    file1.close()


def FormatFile(stopstring):
    pretext = '{"name":"'+jpgname+'","pxSpeed":50,"pxPerRev":200,"points":'
    return pretext+stopstring+'}'


# Show a GUI that represents current collection of edges
# And allows parameter tuning
def SliderFigure(sumImage):
    global ints, spacing, threshhold

    if not interactive:
        return

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.45, bottom=0.35)
    ax.imshow(1-sumImage, cmap='gray', vmin=0, vmax=1)

    sax1 = fig.add_axes([0.15, 0.05, 0.5, 0.03])
    sax2 = fig.add_axes([0.15, 0.10, 0.5, 0.03])
    sax3 = fig.add_axes([0.15, 0.15, 0.5, 0.03])
    sax4 = fig.add_axes([0.15, 0.20, 0.5, 0.03])
    sax5 = fig.add_axes([0.15, 0.25, 0.5, 0.03])
    s1 = Slider(sax1, 'Int1', 0, 255, valfmt='%0.0f', valinit=ints[0])
    s2 = Slider(sax2, 'Int2', 0, 255, valfmt='%0.0f', valinit=ints[1])
    s3 = Slider(sax3, 'Int3', 0, 255, valfmt='%0.0f', valinit=ints[2])
    s4 = Slider(sax4, 'Int4', 0, 255, valfmt='%0.0f', valinit=ints[3])
    s5 = Slider(sax5, 'Int5', 0, 255, valfmt='%0.0f', valinit=ints[4])
    s1.on_changed(lambda x: slid(1, x))
    s2.on_changed(lambda x: slid(2, x))
    s3.on_changed(lambda x: slid(3, x))
    s4.on_changed(lambda x: slid(4, x))
    s5.on_changed(lambda x: slid(5, x))
    tax1 = fig.add_axes([0.7, 0.05, 0.05, 0.03])
    tax2 = fig.add_axes([0.7, 0.10, 0.05, 0.03])
    tax3 = fig.add_axes([0.7, 0.15, 0.05, 0.03])
    tax4 = fig.add_axes([0.7, 0.20, 0.05, 0.03])
    tax5 = fig.add_axes([0.7, 0.25, 0.05, 0.03])
    t1 = TextBox(tax1, '', initial=str(spacing[0]))
    t2 = TextBox(tax2, '', initial=str(spacing[1]))
    t3 = TextBox(tax3, '', initial=str(spacing[2]))
    t4 = TextBox(tax4, '', initial=str(spacing[3]))
    t5 = TextBox(tax5, '', initial=str(spacing[4]))
    t1.on_submit(lambda x: space(1, x))
    t2.on_submit(lambda x: space(2, x))
    t3.on_submit(lambda x: space(3, x))
    t4.on_submit(lambda x: space(4, x))
    t5.on_submit(lambda x: space(5, x))

    eax1 = fig.add_axes([0.15, 0.5, 0.2, 0.03])
    eax2 = fig.add_axes([0.15, 0.55, 0.2, 0.03])
    e1 = Slider(eax1, 'Thr1', 0, 1000, valfmt='%0.0f', valinit=edgethr[0])
    e2 = Slider(eax2, 'Thr0', 0, 1000, valfmt='%0.0f', valinit=edgethr[1])
    e1.on_changed(lambda x: thresh(1, x))
    e2.on_changed(lambda x: thresh(2, x))

    def slid(ind, val):
        global ints, spacing, threshhold
        ints[ind-1] = int(round(val))
        sumImage = Update(1, 0, 0, 0, 0)  # Set up bin images
        sumImage = Update(0, ind, 0, 0, 0)
        ax.imshow(1-sumImage, cmap='gray', vmin=0, vmax=1)
        fig.canvas.draw()

    def space(ind, val):
        global ints, spacing, threshhold
        spacing[ind-1] = int(val)
        sumImage = Update(0, ind, 0, 0, 0)
        ax.imshow(1-sumImage, cmap='gray', vmin=0, vmax=1)
        fig.canvas.draw()

    def thresh(ind, val):
        global edgethr
        edgethr[ind-1] = val
        sumImage = Update(0, 0, 0, 0, 1)  # Set up bin images
        ax.imshow(1-sumImage, cmap='gray', vmin=0, vmax=1)
        fig.canvas.draw()

    plt.show()
    plt.close()
    print(edgethr)
    print(ints)


# Generate edges based on parameters
def Update(nint, sp, ori, off, edge):
    if edge:
        print('Run edge detection')
        global edIm, canedges
        edIm = extractEdges(im_path, edgethr[0], edgethr[1])
        # edIm = updEdge(Ig, 100000)
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
        # Which bin are we hatching?
        s = sp + ori + off - 1
        # Build the full matrix hatch
        h = buildHatch(binIm.shape, orientation[s], spacing[s], offset[s])
        # Boolean mask with bin matrix
        mh = (h & (binIm <= ints[s]))
        # Add edges to the graph
        alledges[s] = createDiagEdges(mh, orientation[s])
        hatchedgim[s] = mh > 0

    # Regen image to show user
    hIm = np.zeros(Ig.shape)
    for hI in hatchedgim:
        if (len(hI) != 0):
            hIm = (hIm + hI) > 0
    sumImage = edIm + hIm
    sumImage = despeck(sumImage)

    return sumImage


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
    PlotGraph(G, ndnodes)

    # Connect any separate subgraphs into one
    # Will add edges
    print('Connect Subgraphs')
    G = ConnectSubgraphs(G, nnodes, dnodes)
    print("Num Odd", len(nOdeg(G)))  # Number of odd degree nodes
    print("Num One", len(n1deg(G)))  # Number of degree one nodes
    print("Num components", len(list(nx.connected_components(G))))
    PlotGraph(G, ndnodes)

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
    PlotGraph(G, ndnodes)

    # Add a parallel edge to any odd node with an odd node as a neighbor
    print('EasyLinkOdd')
    G = EasyLinkOdd(G)
    print("Num Odd", len(nOdeg(G)))  # Number of odd degree nodes
    print("Num One", len(n1deg(G)))  # Number of degree one nodes
    PlotGraph(G, ndnodes)

    # Link odd nodes together but only if they are each other's best odd
    # This may create new edges
    print('LinkDegOdd')
    G = LinkDegOdd(G, nnodes, dnodes)
    print("Num Odd", len(nOdeg(G)))  # Number of odd degree nodes
    print("Num One", len(n1deg(G)))  # Number of degree one nodes
    while len(nOdeg(G)) > 400:
        print('LinkDegOdd 2')
        G = LinkDegOdd(G, nnodes, dnodes)
        print("Num Odd", len(nOdeg(G)))  # Number of odd degree nodes
        print("Num One", len(n1deg(G)))  # Number of degree one nodes

    PlotGraph(G, ndnodes)

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


# Edge detection returns a binary image of edge points
# Link up adjacent points returned from edge detection
# And return them as graph edges
def ConnectCanny(cIm):
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


# Given lines from buildHatch, turn matrix points into long graph edges
def createDiagEdges(mask, ori):
    # Connect adjacent points
    G = NeiEdge(nx.Graph(), mask, (2+ori))

    # Each component is a long edge
    edges = []
    subs = (list(nx.connected_components(G)))
    # Extract the endpoints (degree == 1)
    edges = list(map(lambda s: [v for v, d2 in G.degree(list(s)) if d2 == 1], subs))
    return edges


# Connect separate subgraphs into one big subgraph
# Try to add the shortest edges possible to do this
def ConnectSubgraphs(G, nnodes, dnodes):
    sub_graphs = list(nx.connected_components(G))
    all_coord = [dnodes[x] for x in list(G.nodes())]
    A = NodeMap(all_coord)

    # More efficient way?
    # shift up and left, connect if include other, repeat if not

    # Repeat until there's only onesubgraph
    while len(sub_graphs) > 1:
        # For each subgraph, connect it to the closest node in a different subgraph
        for i, sg in enumerate(sub_graphs):
            print(i)
            if i == len(sub_graphs) - 1:
                continue
            # Get all nodes connected to first node in the subgraph
            sg = nx.node_connected_component(G, list(sg)[0])
            sg_node_num = list(sg)
            sg_node_coord = [dnodes[x] for x in sg_node_num]

            # Grow the border of the subgraph and find new edges
            newedge = GrowBorder(A, sg_node_coord)
            # If we found a new edge, add it to the graph
            if newedge != None:
                G.add_edge(nnodes[newedge[0]], nnodes[newedge[1]], weight=newedge[2])

        sub_graphs = list(nx.connected_components(G))

    return G


# Add a parallel edge to any degree 1 node with an odd node as a neighbor
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
        if G.degree(n1) != 1:
            continue
        G.add_edge(n1, ne, weight=G.get_edge_data(n1, ne)[0]['weight'])
    return G


# Add a parallel edge to any odd node with an odd node as a neighbor
def EasyLinkOdd(G):
    nO = nOdeg(G)  # Odd deg nodes
    nOnei = list(map(lambda e: list(G.neighbors(e))[0], nO))  # Neighbors of odd degree nodes
    neideg = list(map(lambda e: G.degree(e) % 2 == 1, nOnei))  # Degree of each neighbor

    # Find where odd degree node has neighbor that is also odd
    filternO = [i for indx, i in enumerate(nO) if neideg[indx] is True]
    filterne = [i for indx, i in enumerate(nOnei) if neideg[indx] is True]

    # Match them up
    for nO, ne in zip(filternO, filterne):
        # If we matched one of these already and it's now even degree, skip
        if G.degree(nO) % 2 != 1:
            continue
        G.add_edge(nO, ne, weight=G.get_edge_data(nO, ne)[0]['weight'])

    return G


# Add a parallel edge from any remaining deg 1 node to closest node
# May make some new nodes odd
def LinkDegOne(G, nnodes, dnodes):
    nodes_one_degree = n1deg(G)
    one_coord = [dnodes[p] for p in nodes_one_degree]

    nodes_all_degree = list(G.nodes())
    all_coord = [dnodes[p] for p in nodes_all_degree]

    for i, p in enumerate(one_coord):
        # If no longer odd, skip
        if G.degree(nnodes[p]) != 1:
            continue

        # Get distances to all other nodes
        ld = distance_matrix([p], all_coord)
        ld[0, nnodes[p]] = 100000
        # Find closest and connect
        nei = ld.argmin()
        G.add_edge(nnodes[p], nnodes[all_coord[nei]], weight=ld[0, nei])

    return G


# Add a parallel edge from any remaining odd degree node to closest odd deg node
# May not make all odd points even
def LinkDegOdd(G, nnodes, dnodes):
    diagDistance = pythag(Ig.shape, (0, 0))
    distance_threshold = 0.02 * diagDistance
    nodes_odd_degree = nOdeg(G)
    odd_coord = [dnodes[p] for p in nodes_odd_degree]

    for i, p in enumerate(odd_coord):
        if G.degree(nnodes[p]) % 2 != 1:
            continue

        # Get distances to all other odd nodes
        ld = distance_matrix([p], odd_coord)
        ld[0, i] = 100000
        # Find closest
        nei = ld.argmin()
        # If shorter than some threshold, connect
        if ld[0, nei] < distance_threshold:
            G.add_edge(nnodes[p], nnodes[odd_coord[nei]], weight=ld[0, nei])

    return G


# Link remaining odd nodes by adding parallel paths between them
# Paths run through even nodes but add 2 edges to each, so still even
# Try to find minimum weight matching
# https://brooksandrew.github.io/simpleblog/articles/intro-to-graph-optimization-solving-cpp/
def PathFinder(G, dnodes):
    nO = nOdeg(G)
    # Build an empty distance matrix between all nodes (through the graph)
    A = np.zeros((len(list(G.nodes())), len(list(G.nodes()))))
    # Get path lengths from all odd nodes to all nodes
    # Populate a matrix
    print('Get all path lengths')
    for t in nO:
        # For each odd node, get distance through graph to all other nodes
        # Does it improve speed if only get distances to odd nodes?
        length = nx.single_source_dijkstra_path_length(G, t)
        lk = list(length.keys())
        lv = list(length.values())
        # Populate distance matrix
        A[t, lk] = lv

    # Only look at odd to odd paths
    np.fill_diagonal(A, 1000000)  # No self loops
    A2 = A[nO, :]
    A2 = A2[:, nO]

    # Get all combinations of two odd nodes
    odd_node_pairs = list(itertools.combinations(nO, 2))
    print('Num pairs', len(odd_node_pairs))
    # Construct a temp complete graph of odd nodes
    # edge weights = path lengths through original graph
    # (negative for max weight matching)
    OddComplete = nx.Graph()
    for op in odd_node_pairs:
        OddComplete.add_edge(op[0], op[1], weight=-A[op[0], op[1]])

    # This gets optimal matching, probably could do near-optimal matching
    print('Begin matching')
    odd_matching_dupes = nx.algorithms.max_weight_matching(OddComplete, True)
    print('End matching')
    # Pull out optimal matching pairs
    if type(odd_matching_dupes) is set:
        odd_matching = list(pd.unique([tuple(sorted([k, v])) for k, v in list(odd_matching_dupes)]))
    else:
        odd_matching = list(pd.unique([tuple(sorted([k, v])) for k, v in odd_matching_dupes.items()]))

    # For each match that is created, add path through original graph
    print('Add parallel paths')
    for a in odd_matching:
        # Get the full path between the two
        p = nx.dijkstra_path(G, a[0], a[1])
        # Add parallel edges from the path
        for i in range(0, len(p) - 1):
            w = pythag(dnodes[p[i]], dnodes[p[i+1]])
            G.add_edge(p[i], p[i+1], weight=w)

    return G


def PlotGraph(G, ndnodes):
    if not interactive:
        return
    nx.draw_networkx_edges(G, ndnodes, node_size=0.01, width=.2)
    nx.draw_networkx_nodes(G, ndnodes, nodelist=nOdeg(G), node_size=0.02)
    plt.axis('scaled')
    plt.show()


if __name__ == '__main__':
    main()
