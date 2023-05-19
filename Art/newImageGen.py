import sys
import os
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt

from modin.config import ProgressBar
ProgressBar.enable()

from artUtils import *
from imageParser import GenerateEdges
from graphMods import *

# If interactive, show GUI for parameter tuning
# If interactive == 'check' also show progress after each step
interactive = False
if len(sys.argv) > 2:
  interactive = False
print('Interactive', interactive)       # Edge detection parameters

# Pull the image
folder = '/Users/Ben/Desktop/Files/Etch/'
jpgname = sys.argv[1]if 1 < len(sys.argv) else 'Supercar2'
im_path = os.path.join(folder, jpgname+'.jpg')
Im = np.array(Image.open(im_path).convert('RGB'))
Im_gray = rgb2gray(Im)


def main():
  # Take the grayscale image and generate main set of edges
  alledges = GenerateEdges(Im_gray, interactive, im_path)

  # At this point we're ready to build the graph
  # and start connecting edges as necessary
  # Need an eulerian circuit
  # One component, no odd degree nodes

  print('Start graph work')
  G, coord_to_ind, ind_to_coord, neg_coord = genGraph(alledges)

  # Connect any separate subgraphs into one
  # Will add edges
  print('Connect Subgraphs')
  G = ConnectSubgraphs(G, coord_to_ind, ind_to_coord)
  print("Num components", len(list(nx.connected_components(G))))
  PlotGraph(G, neg_coord)

  # Add a parallel edge to any degree-1 node with an degree-odd node as a neighbor
  print('EasyLinkOne')
  G = EasyLinkOne(G)
  PlotGraph(G, neg_coord)

  # # Link remaining d1 nodes to a good neighbor
  # # This may create new edges
  print('LinkDegOne')
  G = LinkDegOne(G, coord_to_ind, ind_to_coord)
  PlotGraph(G, neg_coord)

  # Add a parallel edge to any odd node with an odd node as a neighbor
  print('EasyLinkOdd')
  G = EasyLinkOdd(G)
  PlotGraph(G, neg_coord)

  # Link odd nodes together but only if they are each other's best odd
  # This may create new edges
  print('LinkDegOdd')
  G = LinkDegOdd(G, coord_to_ind, ind_to_coord)
  counter = 1
  previous_nOdeg = nOdeg(G)
  this_nOdeg = None
  while len(nOdeg(G)) > 300 and previous_nOdeg != this_nOdeg:
    counter += 1
    previous_nOdeg = this_nOdeg
    print('LinkDegOdd '+str(counter))
    G = LinkDegOdd(G, coord_to_ind, ind_to_coord)
    this_nOdeg = len(nOdeg(G))
    print("Num Odd", len(nOdeg(G)))  # Number of odd degree nodes
    print("Num One", len(n1deg(G)))  # Number of degree one nodes

  PlotGraph(G, neg_coord)

  # Match remaining odd nodes with longer parallel paths
  print('FinalOdd')
  G = FinalizeGraph(G, ind_to_coord)

  # This outputs the waypoints of the path
  print('Begin Eulerian')
  stops = [list(ind_to_coord[u[0]]) for u in nx.eulerian_path(G)]
  stops = np.array(stops)

  print('Write to file')
  np.set_printoptions(threshold=sys.maxsize)
  stopstring = np.array2string(stops, separator=',')
  file1 = open('../rPi/public/art/'+jpgname+'.json', "w")
  file1.write(FormatFile(stopstring))
  file1.close()


# Do the graph theory work
def genGraph(alledges):
  G = nx.MultiGraph()

  # Add weights to edges based on pythag distance
  weighted_edges = list(map(lambda e: (e[0], e[1], pythag(e[0], e[1])), alledges))

  # Get all unique nodes in all edges
  allnodes = list(map(lambda e: [e[0], e[1]], alledges))
  allnodes = [item for sublist in allnodes for item in sublist]
  allnodes = list(set(allnodes))
  # Create dicts that can translate between node # and node coord
  coord_to_ind = {v: k for k, v in enumerate(allnodes)} # nnodes
  ind_to_coord = {k: v for k, v in enumerate(allnodes)} # dnodes
  neg_coord = {k: (v[0], -v[1]) for k, v in enumerate(allnodes)}

  # Add the edges with node #'s instead of coords (makes things easier)
  nweighted_edges = list(map(lambda e: 
    (coord_to_ind[e[0]], coord_to_ind[e[1]], e[2]), weighted_edges))
  G.add_weighted_edges_from(nweighted_edges)
  
  return G, coord_to_ind, ind_to_coord, neg_coord


def PlotGraph(G, neg_coord):
  print("Num Odd", len(nOdeg(G)))  # Number of odd degree nodes
  print("Num One", len(n1deg(G)))  # Number of degree one nodes

  if interactive != 'check':
    return
  nx.draw_networkx_edges(G, neg_coord, node_size=0.01, width=.2)
  nx.draw_networkx_nodes(G, neg_coord, nodelist=nOdeg(G), node_size=0.02)
  plt.axis('scaled')
  plt.show()


def FormatFile(stopstring):
  pretext = '{"name":"'+jpgname+'","pxSpeed":50,"pxPerRev":200,"points":'
  return pretext+stopstring+'}'


if __name__ == '__main__':
    main()
