#! python3
import cv2
import os
import sys
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib
from tsp_solver.greedy import solve_tsp
from scipy.spatial import distance_matrix

# path_output_dir = './frames/'

# def video_to_frames(video):
#     # Extract frames from a video and save to directory as 'x.png' where x is the frame index
#     vidcap = cv2.VideoCapture(video)
#     # How many frames were extracted?
#     count = 0
#     # Loop and extract all frames
#     while vidcap.isOpened():
#         success, image = vidcap.read()
#         # If there's still frames to extract
#         if success:
#             cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
#             count += 1
#         else:
#             break
#     cv2.destroyAllWindows()
#     vidcap.release()
#     return count

# def findDerivatives(I_gray):
#   # TODO: your code here

#   dy = np.array([1,-1]).reshape(2,1)
#   dx = np.array([1,-1]).reshape(1,2)

#   # Convolve image with derivative of gaussian to get horz and vert gradient
#   Magx = signal.convolve2d(I_gray,dx,'same')
#   Magy = signal.convolve2d(I_gray,dy,'same')

#   # Gradient magnitude and orientation
#   Mag = np.sqrt(Magx*Magx + Magy*Magy)
#   Ori = np.arctan2(-Magx,Magy)

#   return Mag, Magx, Magy, Ori

# def nonMaxSup(Mag, Ori):
#   # TODO: your code here
#   # print('Start NMS')
  
#   # Set up interpolation of gradient magnitude
#   xmax, ymax = Mag.shape
#   x = np.arange(0, xmax)
#   y = np.arange(0, ymax)

#   interp_spline = RectBivariateSpline(x, y, Mag)

#   # Find each offset values to use for interpolation
#   # Use a unit length in the direction of the gradient at each point
#   Xoffset = np.cos(Ori)
#   Yoffset = np.sin(Ori)

#   Y, X = np.meshgrid(y,x)

#   PlusPointX = X+Xoffset
#   PlusPointY = Y+Yoffset
#   MinuPointX = X-Xoffset
#   MinuPointY = Y-Yoffset

#   # Interpolate to get magnitude at those values

#   PlusMag = interp_spline.ev(PlusPointX, PlusPointY)
#   MinuMag = interp_spline.ev(MinuPointX, MinuPointY)

#   # Compare pixel to two neighbors
#   GreatPlus = ((Mag-PlusMag) > 0)
#   GreatMinu = ((Mag-MinuMag) > 0)

#   # Only pixels that are greater than both neighbors (1x1)
#   M = np.multiply(GreatPlus, GreatMinu).astype(int)

#   return M

frame = cv2.imread('/Users/Ben/Desktop/im2.png', 0)
edges = cv2.Canny(frame,100,200)

plt.imshow(edges)
plt.show()

frame = (frame < 200) & (frame > 10 )
print(np.sum(frame))

# Set up interpolation of gradient magnitude
xmax, ymax = frame.shape
x = np.arange(0, xmax)
y = np.arange(0, ymax)
Y, X = np.meshgrid(y,x)
xp = Y[frame]
yp = X[frame];

cities = (np.transpose([xp, yp]))

d = distance_matrix(cities, cities)
# print(d)
np.set_printoptions(threshold=sys.maxsize)
# print(cities)
plt.imshow(frame)
plt.show()

path = solve_tsp(d)
# print(path)
# print(cities[path])
print(np.array2string(cities[path], separator=','))

# print(np.sum(frame))
# for x in range(1,3):
  # Mag, magX, magy, Ori = findDerivatives(frame)
  # frame = frame & (Mag < .5)
  # print(np.sum(frame))



# matplotlib.image.imsave("/Users/Ben/Desktop/im2.png", frame>0.5)

# # video_to_frames('./vid.m4v')
# frameCount = 490

# frameOld = cv2.imread(os.path.join(path_output_dir, str(0)+'.png'), 0)
# frameOld = (frameOld < 150)


# frameOld = (M & frameOld)

# # Set up interpolation of gradient magnitude
# xmax, ymax = frameOld.shape
# x = np.arange(0, xmax)
# y = np.arange(0, ymax)
# Yf, Xf = np.meshgrid(y,x)

# for f in range(frameCount-1):
#     print(f)
#     frameNew = cv2.imread(os.path.join(path_output_dir, str(f+1)+'.png'), 0)
#     frameNew = (frameNew < 150)
#     # Mag, magX, magy, Ori = findDerivatives(frameNew)
#     # M = nonMaxSup(Mag, Ori)
#     # frameNew = (M & frameNew)
#     diff = frameNew & ~frameOld
#     plt.imshow(diff, interpolation='nearest')
#     plt.show()
#     # print(np.sum(frameNew))
#     # print(np.transpose(np.vstack((Xf[diff], Yf[diff]))))
#     frameOld = frameNew
