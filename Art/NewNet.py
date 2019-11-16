import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import sys
import matplotlib.pyplot as plt
# from postman_problems.solver import cpp
# from postman_problems.stats import calculate_postman_solution_stats
import itertools
import atexit
from tempfile import TemporaryFile
import time
# import pandas as pd

print('Line done')
cities = []
cities = np.array(cities)

def ConnectSubgraphs(G):
    sub_graphs = nx.connected_component_subgraphs(G)
    # print(list(sub_graphs))
    # For each subgraph, connect it to the closest node in a different subgraph
    for i, sg in enumerate(sub_graphs):
        # Find closest node
        no = np.array(list(sg.nodes()))
        # print(len(no))
        mindist = 10000000
        minN = 0 # Node in this subgraph closest to any other node in a different graph
        # Don't look at nodes in same subgraph
        a,b = np.meshgrid(no, no)
        d[a,b] =  100000
        A = d[no,:]
        (z, nei) = np.unravel_index(A.argmin(), A.shape)
        minN = no[z]
        # for n in sg.nodes():
        #     # for n2 in sg.nodes():
        #     # d[n, no] = 100000
        #     themin = min(d[n, :])
        #     if themin < mindist:
        #         minN = n
        #         mindist = themin

        # minInd = mindists.index(min(mindists))
        # minN = no[minInd]
        ndists = d[minN, :]
        # nei = ndists.find(min(ndists))+1
        print(minN, nei)
        G.add_edge(minN, nei, weight=A.min())

    return G

d = distance_matrix(cities, cities)
# # # Don't look at self-edges
np.fill_diagonal(d, 100000)

np.save('d', d)

# d = np.load('d.npy')
print('2')
# Begin looking at neighbors 1 away (incl diag)
a1 = np.multiply(d, (d < 1.5))
print(len(a1 != 0))
# Mark that we've already added these edges
d = d + a1 * 100000
np.fill_diagonal(a1, 0)

# G = nx.from_numpy_matrix(a1)
# G = ConnectSubgraphs(G)
# nx.write_weighted_edgelist(G,"new.adjlist")
T = nx.MultiGraph()
Ge = nx.read_weighted_edgelist('new.adjlist', nodetype=int, create_using=T)

def CloseDegreeOneNodes(T):
    nodes_one_degree = [v for v, d2 in T.degree() if d2 == 1]
    dOne = d[nodes_one_degree, :]
    dOne = dOne[:, nodes_one_degree]
    closedOne = dOne < 10
    clipped = np.tril(closedOne)
    pairs = np.argwhere(clipped)
    for p in pairs:
        T.add_edge(nodes_one_degree[p[0]], nodes_one_degree[p[1]])
    return T

def FixDegreeOneNodes(T):
    nodes_one_degree = [v for v, d2 in T.degree() if d2 == 1]
    # Any degree-one node gets a duplicate edge
    for one in nodes_one_degree:
        bors = list(T.neighbors(one))
        T.add_edge(one, bors[0])

    return T

def FixEasyOddDegree(T):
    nodes_odd_degree = [v for v, d2 in T.degree() if d2 % 2 == 1]
    # Any odd-degree node with an odd-degree neighbor gets a duplicate edge
    for odd in nodes_odd_degree:
        bors = list(T.neighbors(odd))
        for b in bors:
            if b in nodes_odd_degree:
                nodes_odd_degree.remove(b)
                T.add_edge(odd, b)
                break

    return T

T = CloseDegreeOneNodes(T)

T = FixDegreeOneNodes(T)

T = FixEasyOddDegree(T)

print('3')

# # Pair up each odd degree node such that 
# # the total cost to connect each pair (through other edges) is minimized
# # Add paths btwn pairs, edges increase degree of intermediate nodes by 2

def CreateLMat(T):
    nodes_odd_degree = [v for v, d2 in T.degree() if d2 % 2 == 1]
    print(len(nodes_odd_degree))
    lMat = np.ones((len(nodes_odd_degree), len(T.nodes())))*100000
    i = 0
    for odd in nodes_odd_degree:
        length=nx.single_source_dijkstra_path_length(T,odd)
        v = length.values()
        k = length.keys()
        Z = [x for _,x in sorted(zip(k,v))]
        lMat[i, :] = Z
        i += 1
    lMat = lMat[:, nodes_odd_degree]
    np.fill_diagonal(lMat, 100000)
    return lMat

lMat = CreateLMat(T)
# np.save('lmat', lMat)
print('3A')


# lMat = np.load('lmat.npy')

def LinSum(T, lMat):
    nodes_odd_degree = [v for v, d2 in T.degree() if d2 % 2 == 1]
    row_ind, col_ind = linear_sum_assignment(lMat)

    added = []
    for j in range(0,len(nodes_odd_degree)):
        if j == col_ind[col_ind[j]]:
            if not (j in added or col_ind[j] in added):
                added.append(j)
                added.append(col_ind[j])
                path=nx.dijkstra_path(T,nodes_odd_degree[j], nodes_odd_degree[col_ind[j]])
                # print(path)
                T.add_path(path)
    return T

T = LinSum(T, lMat)

nodes_odd_degree = [v for v, d2 in T.degree() if d2 % 2 == 1]
print(len(nodes_odd_degree),sorted(nodes_odd_degree))
print('4')


lMat = CreateLMat(T)

def RANSAC(T):
    nodes_odd_degree = [v for v, d2 in T.degree() if d2 % 2 == 1]
    nOdd = len(nodes_odd_degree)

    # randomly split nodes_odd_degree into 2 halves
    # linsumassign row vs column, compare

    bestSum = 100000000
    bestL = []
    bestR = []
    for x in range(0, 300):
        oddNodes = np.random.choice(nOdd, nOdd, replace=False)
        L, R = oddNodes[:int(nOdd/2)], oddNodes[int(nOdd/2):]
        submat = lMat[L, :]
        submat = submat[:, R]

        r, c = linear_sum_assignment(submat)
        thisSum = submat[r, c].sum()
        if thisSum < bestSum:
            print(thisSum)
            bestSum = thisSum
            bestL = L
            bestR = R

    for j in range(0,len(L)):
        path=nx.dijkstra_path(T,nodes_odd_degree[bestL[j]], nodes_odd_degree[bestR[j]])
        T.add_path(path)
    return T

T = RANSAC(T)

print(len(list(T.edges())))
nodes_odd_degree = [v for v, d2 in T.degree() if d2 % 2 == 1]
stops = np.array([u for u, v in nx.eulerian_circuit(T)])
print(len(stops))

def RemoveStutter(stops):
    diff2 = stops[:-2] - stops[2:]
    adiff2 = np.array(diff2[:-1])
    bdiff2 = np.array(diff2[1:])
    rem = np.logical_not(np.logical_and(adiff2 == 0, bdiff2 == 0))
    rem = np.insert(rem, 0, [True, True, True], axis=0)
    stops = stops[rem]
    return stops
print(len(stops))
stops = RemoveStutter(stops)

def RemoveColinear(pa):

    spa = np.array(pa[1:, :])
    # spa.append([0, 0])
    spa = np.append(spa, [[0, 0]], axis=0)
    dpa = pa - spa
    sdpa = dpa[1:, :]
    # sdpa.append([0, 0])
    sdpa = np.append(sdpa, [[0, 0]], axis=0)
    ddpa = dpa - sdpa
    q = np.sum(ddpa, axis=1) != 0
    print(len(q), np.sum(q))
    return pa[q]

print(len(stops))
out = RemoveColinear(cities[stops])
print(len(out))
# print(cities[stops])
np.set_printoptions(threshold=sys.maxsize)
print(np.array2string(out, separator=','))