import networkx as nx
import modin.pandas as pd
import itertools

from artUtils import *

# Connect separate subgraphs into one big subgraph
# Try to add the shortest edges possible to do this
def ConnectSubgraphs(G, coord_to_ind, ind_to_coord):
  sub_graphs = list(nx.connected_components(G))
  all_coord = [ind_to_coord[x] for x in list(G.nodes())]
  A = NodeMap(all_coord)

  # More efficient way?
  # shift up and left, connect if include other, repeat if not

  def add_edge(sg):
    # Get all nodes connected to first node in the subgraph
    sg_node_num = list(sg)
    sg_node_coord = [ind_to_coord[x] for x in sg_node_num]
    # Grow the border of the subgraph and find new edges
    return GrowBorder(A, sg_node_coord)

  # Repeat until there's only onesubgraph
  while len(sub_graphs) > 1:
    print(len(sub_graphs))
    # For each subgraph, connect it to the closest node in a different subgraph
    df = pd.Series(sub_graphs)
    df2 = df.apply(lambda row : add_edge( row))
    for newedge in df2:
      if newedge != None:
        G.add_edge(coord_to_ind[newedge[0]], coord_to_ind[newedge[1]], weight=newedge[2])

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
def LinkDegOne(G, coord_to_ind, ind_to_coord):
  nodes_one_degree = n1deg(G)
  one_coord = [ind_to_coord[p] for p in nodes_one_degree]

  nodes_all_degree = list(G.nodes())
  all_coord = [ind_to_coord[p] for p in nodes_all_degree]

  def match_deg_one(p):
    dm = distance_matrix([ind_to_coord[p]], all_coord)
    dm[dm==0] = 100000
    nei = dm.argmin()
    return [p, coord_to_ind[all_coord[nei]], dm[0][nei]]

  df = pd.Series(nodes_one_degree)
  df2 = df.apply(lambda row : match_deg_one(row))

  for edge in df2:
    if edge == None:
      continue
    [n1, n2, w] = edge
    if G.degree(n1) == 1:
      G.add_edge(n1, n2, weight=w)

  return G


# Add a parallel edge from any remaining odd degree node to closest odd deg node
# May not make all odd points even
def LinkDegOdd(G, coord_to_ind, ind_to_coord):
  point_coords = ind_to_coord.values()
  xcoord = [p[0] for p in ind_to_coord.values()]
  ycoord = [p[1] for p in ind_to_coord.values()]

  distance_threshold = 0.01 * pythag((max(xcoord), max(ycoord)), (min(xcoord), min(ycoord)))
  nodes_odd_degree = nOdeg(G)
  odd_coord = [ind_to_coord[p] for p in nodes_odd_degree]

  def match_deg_odd(p):
    dm = distance_matrix([ind_to_coord[p]], odd_coord)
    dm[dm==0] = 100000
    nei = dm.argmin()
    edge_distance = dm[0][nei]
    if edge_distance > distance_threshold:
      return None
    return [p, coord_to_ind[odd_coord[nei]], dm[0][nei]]

  df = pd.Series(nodes_odd_degree)
  df2 = df.apply(lambda row : match_deg_odd(row))

  for edge in df2:
    if edge == None:
      continue
    [n1, n2, w] = edge
    if G.degree(n1) % 2 == 1:
      G.add_edge(n1, n2, weight=w)

  return G


# Link remaining odd nodes by adding parallel paths between them
# Paths run through even nodes but add 2 edges to each, so still even
# Try to find minimum weight matching
# https://brooksandrew.github.io/simpleblog/articles/intro-to-graph-optimization-solving-cpp/
def FinalizeGraph(G, ind_to_coord):
  nO = nOdeg(G)
  # Build an empty distance matrix between all nodes (through the graph)
  A = np.zeros((len(list(G.nodes())), len(list(G.nodes()))))
  # Get path lengths from all odd nodes to all nodes
  # Populate a matrix
  print('Get all path lengths')

  def calc_path_lengths(n):
    # For each odd node, get distance through graph to all other nodes
    # Does it improve speed if only get distances to odd nodes?
    length = nx.single_source_dijkstra_path_length(G, n)
    lk = list(length.keys())
    lv = list(length.values())
    return [n, lk, lv]

  df = pd.Series(nO)
  df2 = df.apply(lambda row : calc_path_lengths(row))
  for t in df2:
    # Populate distance matrix
    [n, lk, lv] = t
    A[n, lk] = lv

  # Only look at odd to odd paths
  np.fill_diagonal(A, 1000000)  # No self loops

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
      w = pythag(ind_to_coord[p[i]], ind_to_coord[p[i+1]])
      G.add_edge(p[i], p[i+1], weight=w)

  return G