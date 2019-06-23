import time
import threading
# import concurrent.futures
import datetime
from math import sqrt

# import motors

stepsPerRev = 200 # Stepper motor takes in n steps to turn a full 360 deg
pxPerRev = 40 # pixels per revolution (XY, not full vector)
speed = 10 # pix per sec

def r200(n):
    return round(n*200)/200

def pythag(a, b):
    return sqrt(a*a + b*b)

def linInterp(x1, y1, x2, y2):

    d1 = abs(x2 - x1)
    d2 = abs(y2 - y1)

    te = pythag(d1, d2)/speed
    # print('TE', te)
    r1 = d1 / pxPerRev
    r2 = d2 / pxPerRev

    s1 = round(r1 * stepsPerRev)
    s2 = round(r2 * stepsPerRev)

    t1 = 0 if s1 == 0 else te / s1
    t2 = 0 if s2 == 0 else te / s2

    dir1 = 1 - 2 * ((x2 - x1) < 0)
    dir2 = 1 - 2 * ((y2 - y1) < 0)

    # print(d1, r1, s1, t1, dir1)
    # print(d2, r2, s2, t2, dir2)

    t1 = threading.Thread(target = initInterval, args = (t1*1000, s1, genStep(1, dir1), 1))
    t2 = threading.Thread(target = initInterval, args = (t2*1000, s2, genStep(2, dir2), 2))
    return te, t1, t2
    
def initInterval(tms, n, fn, intN):
    global next_call, timer_MS, nCount, fnlist
    
    try:
        next_call
    except NameError:
        # clearTimerInfo()
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

def clearTimerInfo():
    next_call = [];
    timer_MS = [];
    nCount = [];
    fnlist = [];

def genStep(mn, dir):
    def s():
        # print(mn*dir)
        # motors.step(mn, dir)
        pass
    return s

def genF(a1, a2):
    def fn():
        print('start')
        a1.start()
        a2.start()
        a1.join()
        a2.join()
    return fn

def lineThread(pretime, th1, th2):
    def fn():
        return threading.Timer( pretime - time.time(), genF(th1, th2) ).start()
    return fn

def genThreads(pts, startT, pxSpeed, pxRev):
    global pxPerRev, speed
    pxPerRev = pxRev
    speed = pxSpeed
    pretime = startT;
    for i in range(len(pts)-1):
        a = pts[i][0]
        b = pts[i][1]
        c = pts[i+1][0]
        d = pts[i+1][1]
        te, th1, th2 = linInterp(a,b,c,d)

        threading.Timer( pretime - time.time(), genF(th1, th2) ).start()

        pretime += te
        # Delay to reduce the # of threads active at once
        sleeptime = max((te - 0.1 + threading.active_count()*.003), 0.001)
        time.sleep(sleeptime)