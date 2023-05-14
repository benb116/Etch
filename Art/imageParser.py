from math import comb
from venv import create
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from canny3 import extractEdges
from artUtils import *


# Edge detection returns a binary image of edge points
# Link up adjacent points returned from edge detection
# And return them as graph edges
def ConnectCanny(cIm):
  G = nx.Graph()

  G = AddNeighborEdges(G, cIm, 2)
  G = AddNeighborEdges(G, cIm, 4)
  G = AddNeighborEdges(G, cIm, 1)
  G = AddNeighborEdges(G, cIm, 3)
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
  G = AddNeighborEdges(nx.Graph(), mask, (2+ori))  # Connect adjacent points

  # Each sub_graph is a long edge
  sub_graphs = (list(nx.connected_components(G)))
  # Extract the endpoints (degree == 1)
  edges = list(map(lambda s: [v for v, d2 in G.degree(list(s)) if d2 == 1], sub_graphs))
  return edges


# Run edge detection and turn results into endpoint edges
def getCannyEdges(im_path, parameters):
  print('Extract Canny edges')
  Im_canny = extractEdges(im_path, parameters[0], parameters[1])
  Im_canny[Im_canny > 0] = 1
  print('Link edge detection')
  canedges = ConnectCanny(Im_canny)
  return Im_canny, canedges


def generateBinHatch(bin_image, cutoff, orientation, spacing, offset):
  print(f'Generating hatch: {cutoff}, {orientation}, {spacing}, {offset}')
  full_hatch = buildHatch(bin_image.shape, orientation, spacing, offset)  # Build the full matrix hatch
  masked_hatch = (full_hatch & (bin_image <= cutoff)) # Find pixels that are in the hatch and cutoff
  hatched_edges = createDiagEdges(masked_hatch, orientation)
  return masked_hatch, hatched_edges


def combineImages(edge_image, hatched_images):
  edge_image = edge_image > 0
  for hatch_image in hatched_images:
    edge_image = edge_image | hatch_image
  return edge_image


def GenerateEdges(Ig, interactive, im_path):
  # Binning and hatching parameters
  cutoffs = np.array([10, 30, 50, 65, 255])  # Brightness cutoffs
  orientation = np.array([-1, 1, -1, 1, -1])  # Direction (not all the same way)
  spacing = np.array([2, 6, 10, 15, 20])      # Corresponding line densities
  offset = np.array([0, 0, 0, 0, 100000])     # Any offsets
  edge_parameters = [157, 85]                # Edge detection parameters

  Ig = despeck(Ig)
  Im_blur = blur(Ig)  # Smooth out any small specks
  
  # Get initial images and edge sets
  Im_canny, canedges = getCannyEdges(im_path, edge_parameters)  # Use Canny edge detection
  bin_image = bin(Im_blur, cutoffs)  # Bin all pixel values into one of N intervals
  hatch_images, hatch_edge_sets = [], []
  for i in range(len(cutoffs)):
    hatch_image, hatch_edge_set = generateBinHatch(bin_image, cutoffs[i], orientation[i], spacing[i], offset[i])
    hatch_images.append(hatch_image)
    hatch_edge_sets.append(hatch_edge_set)

  sumImage = combineImages(Im_canny, hatch_images)
  
  # Show a GUI that represents current collection of edges
  # And allows parameter tuning
  def SliderFigure():

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.45, bottom=0.35)
    ax.imshow(1-sumImage, cmap='gray', vmin=0, vmax=1)

    sax1 = fig.add_axes([0.15, 0.05, 0.3, 0.03])
    sax2 = fig.add_axes([0.15, 0.10, 0.3, 0.03])
    sax3 = fig.add_axes([0.15, 0.15, 0.3, 0.03])
    sax4 = fig.add_axes([0.15, 0.20, 0.3, 0.03])
    sax5 = fig.add_axes([0.15, 0.25, 0.3, 0.03])
    s1 = Slider(sax1, 'Int1', 0, 255, valfmt='%0.0f', valinit=cutoffs[0])
    s2 = Slider(sax2, 'Int2', 0, 255, valfmt='%0.0f', valinit=cutoffs[1])
    s3 = Slider(sax3, 'Int3', 0, 255, valfmt='%0.0f', valinit=cutoffs[2])
    s4 = Slider(sax4, 'Int4', 0, 255, valfmt='%0.0f', valinit=cutoffs[3])
    s5 = Slider(sax5, 'Int5', 0, 255, valfmt='%0.0f', valinit=cutoffs[4])
    s1.on_changed(lambda x: slid(1, x))
    s2.on_changed(lambda x: slid(2, x))
    s3.on_changed(lambda x: slid(3, x))
    s4.on_changed(lambda x: slid(4, x))
    s5.on_changed(lambda x: slid(5, x))
    tax1 = fig.add_axes([0.55, 0.05, 0.3, 0.03])
    tax2 = fig.add_axes([0.55, 0.10, 0.3, 0.03])
    tax3 = fig.add_axes([0.55, 0.15, 0.3, 0.03])
    tax4 = fig.add_axes([0.55, 0.20, 0.3, 0.03])
    tax5 = fig.add_axes([0.55, 0.25, 0.3, 0.03])
    t1 = Slider(tax1, 'Spa1', 1, 30, valfmt='%0.0f', valinit=spacing[0])
    t2 = Slider(tax2, 'Spa2', 1, 30, valfmt='%0.0f', valinit=spacing[1])
    t3 = Slider(tax3, 'Spa3', 1, 30, valfmt='%0.0f', valinit=spacing[2])
    t4 = Slider(tax4, 'Spa4', 1, 30, valfmt='%0.0f', valinit=spacing[3])
    t5 = Slider(tax5, 'Spa5', 1, 30, valfmt='%0.0f', valinit=spacing[4])
    t1.on_changed(lambda x: space(1, x))
    t2.on_changed(lambda x: space(2, x))
    t3.on_changed(lambda x: space(3, x))
    t4.on_changed(lambda x: space(4, x))
    t5.on_changed(lambda x: space(5, x))

    eax1 = fig.add_axes([0.15, 0.5, 0.2, 0.03])
    eax2 = fig.add_axes([0.15, 0.55, 0.2, 0.03])
    e1 = Slider(eax1, 'Thr1', 50, 300, valfmt='%0.0f', valinit=edge_parameters[0])
    e2 = Slider(eax2, 'Thr0', 50, 300, valfmt='%0.0f', valinit=edge_parameters[1])
    e1.on_changed(lambda x: thresh(1, x))
    e2.on_changed(lambda x: thresh(2, x))

    def slid(ind, val):
      ind = ind-1
      cutoffs[ind] = int(round(val))
      bin_image = bin(Im_blur, cutoffs)  # Bin all pixel values into one of N intervals
      hatch_images[ind], hatch_edge_sets[ind] = generateBinHatch(
        bin_image, cutoffs[ind], orientation[ind], spacing[ind], offset[ind])
      hatch_images[ind+1], hatch_edge_sets[ind+1] = generateBinHatch(
        bin_image, cutoffs[ind+1], orientation[ind+1], spacing[ind+1], offset[ind+1])
      sumImage = combineImages(Im_canny, hatch_images)

      ax.imshow(1-sumImage, cmap='gray', vmin=0, vmax=1)
      fig.canvas.draw()

    def space(ind, val):
      ind = ind-1
      spacing[ind] = val
      hatch_images[ind], hatch_edge_sets[ind] = generateBinHatch(
        bin_image, cutoffs[ind], orientation[ind], spacing[ind], offset[ind])
      sumImage = combineImages(Im_canny, hatch_images)

      ax.imshow(1-sumImage, cmap='gray', vmin=0, vmax=1)
      fig.canvas.draw()

    def thresh(ind, val):
      ind = ind-1
      edge_parameters[ind] = val
      Im_canny, canedges = getCannyEdges(im_path, edge_parameters)  # Use Canny edge detection
      sumImage = combineImages(Im_canny, hatch_images)

      ax.imshow(1-sumImage, cmap='gray', vmin=0, vmax=1)
      fig.canvas.draw()

    plt.show()
    plt.close()

  if interactive:
    SliderFigure()

  print('Final parameters')
  print(cutoffs, orientation, spacing, offset, edge_parameters)

  alledges =  canedges + [item for sublist in hatch_edge_sets for item in sublist]
  print(f'Num edges: {len(alledges)}')
  return alledges
