import numpy as np
from scipy.ndimage import convolve
from scipy.spatial import distance_matrix
from math import sqrt
import sys
import math

left = []

done = []

start = [38,143]

# Greedy sequential norm-based
# done.append(start)
# while len(left) > 0:
#     # print(len(left))
#     last = done[-1]
#     dist = np.subtract(left, last)
#     norm = np.linalg.norm(dist, axis=1)
#     ind = np.argmin(norm)
#     done.append(left[ind])
#     left = np.delete(left, ind, 0)

# https://pythonprogramming.net/how-to-program-best-fit-line-slope-machine-learning-tutorial/
def best_fit_slope(xs,ys):
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)**2) - np.mean(xs**2)))
    if math.isnan(m):
        m = 1000
    return m

np.set_printoptions(threshold=sys.maxsize)

# Greedy sequential norm + direction
done.append(start)
for x in range(1,3):
    last = done[-1]
    dist = np.subtract(left, last)
    norm = np.linalg.norm(dist, axis=1)
    ind = np.argmin(norm)
    done.append(left[ind])
    left = np.delete(left, ind, 0)

# done = np.array(done)
while len(done) < len(left):
    # print(len(left))
    last = done[-1]
    dist = np.subtract(left, last)
    norm = np.linalg.norm(dist, axis=1)

    last3 = np.array(done[-3:])
    # print(last3)
    diff = np.subtract(last3[1], last)
    angle = np.arctan2(diff[1], diff[0])
    m = best_fit_slope(last3[:,0],last3[:,1])
    a2 = np.arctan(m) * np.sign(angle)

    # print(angle, m, a2)
    slopes = np.arctan2(dist[:,1], dist[:,0])
    slopediff = np.abs(np.subtract(slopes, a2))

    score = norm
    ind = np.argmin(score)
    # print(score[ind], slopediff[ind])
    # if score[ind] > 200:
        # break
    done.append(left[ind])
    left = np.delete(left, ind, 0)

print(np.array2string(np.array(done), separator=','))