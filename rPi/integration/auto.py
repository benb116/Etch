import time
import threading
from math import sqrt

from . import pi_utils

onboard = False
if pi_utils.IsRPi():
    onboard = True
    from . import motors

# motors.turnOn()

stepsPerRev = motors.stepsPerRev() # Stepper motor takes in n steps to turn a full 360 deg
pxPerRev = 80 # pixels per revolution (XY, not full vector) - REPLACED
speed = 40 # pix per sec - REPLACED

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

    t1 = 0 if s1 == 0 else te / s1
    t2 = 0 if s2 == 0 else te / s2

    # Which direction to rotate each motor (1 or -1)
    dir1 = -1 + 2 * ((y2 - y1) < 0)
    dir2 = 1 - 2 * ((x2 - x1) < 0)

    # print(d1, r1, s1, t1, dir1)
    # print(d2, r2, s2, t2, dir2)
    # print(s1, s2)
    # Return interval threads
    th1 = threading.Thread(target = initInterval, args = (t1*1000, s1, createStepFn(1, dir1), 1))
    th2 = threading.Thread(target = initInterval, args = (t2*1000, s2, createStepFn(2, dir2), 2))
    return te, th1, th2

# Create recursive timeouts for a motor to step every tms n times
# fn is the function that is called each time
def initInterval(tms, n, fn, intN):
    global next_call, timer_MS, nCount, fnlist
    
    try:
        next_call
    except NameError:
        next_call = [];
        timer_MS = [];
        nCount = [];
        fnlist = [];

    while len(next_call) < intN:
        next_call.append(0)
        timer_MS.append(0)
        nCount.append(0)
        fnlist.append(0)

    i = intN - 1
    next_call[i] = time.time()
    timer_MS[i] = tms
    nCount[i] = 0
    fnlist[i] = fn
    if n == 0:
        return

    def foo():
        global next_call, timer_MS, nCount, fnlist
        tf = fnlist[i]
        tf()
        
        nCount[i] += 1
        if nCount[i] >= n & n >= 0:
            return
        next_call[i] = next_call[i] + timer_MS[i]/float(1000)
        threading.Timer( next_call[i] - time.time(), foo ).start()
       
    foo()

# Returns a fn that makes a motor step in a direction
def createStepFn(mn, dir):
    def s():
        if onboard:
            motors.step(mn, dir)
    return s

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