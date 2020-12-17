import time
import threading
from math import sqrt

from . import pi_utils

pxPerRev = 80 # pixels per revolution (XY, not full vector) - REPLACED
speed = 40 # pix per sec - REPLACED
stepsPerRev = 200 # how many motor steps are in a revolution
onboard = False
if pi_utils.IsRPi():
    onboard = True
    from . import motors
    stepsPerRev = motors.stepsPerRev() # Stepper motor takes in n steps to turn a full 360 deg

# Returns norm of two distances
def pythag(a, b):
    return sqrt(a*a + b*b)

# Returns two threads that generate steps to follow a line between (x1, y1) and (x2, y2)
# Also returns the time it will take to move that distance (for use in delaying thread starts)
def linInterp(x1, y1, x2, y2):

    # Pixel distances
    d1 = abs(y2 - y1)
    d2 = abs(x2 - x1)

    te = pythag(d1, d2)/speed # Time elapsed

    # H and V motor rotations required
    r1 = d1 / pxPerRev
    r2 = d2 / pxPerRev

    # H and V steps required to rotate
    s1 = round(r1 * stepsPerRev)
    s2 = round(r2 * stepsPerRev)

    # Step frequencies
    f1 = s1 / te
    f2 = s2 / te

    # Which direction to rotate each motor (1 or -1)
    dir1 = -1 + 2 * ((y2 - y1) < 0)
    dir2 = 1 - 2 * ((x2 - x1) < 0)

    # print(te, d1, r1, s1, f1, dir1)
    # print(te, d2, r2, s2, f2, dir2)
    # print(s1, s2)
    # Return interval threads
    th1 = threading.Thread(target = motors.setDirAndFreq, args = (1, dir1, int(f1)))
    th2 = threading.Thread(target = motors.setDirAndFreq, args = (2, dir2, int(f2)))
    return te, th1, th2

# Begin the motor threads that were already created
def genF(a1, a2):
    def fn():
        # print('start')
        a1.start()
        a2.start()
        a1.join()
        a2.join()
    return fn

# Dynamimcally create and start motor threads for a set of points
# Wait until startT to begin stepping
# To reduce number of active threads at any one time, delays based on active count
def genThreads(pts, startT, pxSpeed, pxRev):
    global pxPerRev, speed
    pxPerRev = pxRev
    speed = pxSpeed
    pretime = startT; # Updated timestamp at which the next threads will start

    motors.motorsOn(True)
    for i in range(len(pts) - 1):
        a = pts[i][0]
        b = pts[i][1]
        c = pts[i+1][0]
        d = pts[i+1][1]
        te, th1, th2 = linInterp(a,b,c,d)

        # Wait until pretime to begin
        z = threading.Timer( pretime - time.time(), genF(th1, th2) )
        z.start()
        z.join()

        pretime += te # next pretime = old pretime + time elapsed of last threads
        # Delay to reduce the # of threads active at once
        sleeptime = max((te - 0.1 + threading.active_count()*.003), 0.001)
        time.sleep(sleeptime)

    motors.motorsOn(False)