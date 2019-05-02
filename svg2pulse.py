# Image > SVG 
import time
# import RPi.GPIO as GPIO
import threading
import datetime

stepsPerRev = 200
pxPerRev = 50 # pixels per revolution
speed = 25 # pix per sec

def r200(n):
    return round(n*200)/200

def linInterp(x1, y1, x2, y2):

    d1 = r200(abs(x2 - x1))
    d2 = r200(abs(y2 - y1))

    te = max(d1, d2)/speed
    print('TE', te)
    r1 = r200(d1 / pxPerRev)
    r2 = r200(d2 / pxPerRev)

    s1 = round(r1 * stepsPerRev)
    s2 = round(r2 * stepsPerRev)
    t1 = te / s1
    t2 = te / s2

    dir1 = 1 - 2 * ((x2 - x1) < 0)
    dir2 = 1 - 2 * ((y2 - y1) < 0)

    print(d1, r1, s1, t1, dir1)
    print(d2, r2, s2, t2, dir2)

    t1 = threading.Thread(target = initInterval, args = (t1*1000, s1, genStep(1, dir1), 1))
    t2 = threading.Thread(target = initInterval, args = (t2*1000, s2, genStep(2, dir2), 2))
    # t1.start()
    # t2.start()
    # t1.join()
    # t2.join()
    return t1, t2

def genStep(mn, dir):
    def s():
        # step
        print(mn)
    return s

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

    def foo():
        global next_call, timer_MS, nCount, fnlist
        tf = fnlist[i]
        tf()
        
        nCount[i] += 1
        if nCount[i] >= n:
            return
        next_call[i] = next_call[i] + timer_MS[i]/float(1000)
        threading.Timer( next_call[i] - time.time(), foo ).start()
       
    foo()

a1, a2 = linInterp(1835.9187,671.7737,1830.856,676.93994)
a1.start()
a2.start()

def genF(x1, y1, x2, y2):
    return linInterp(x1, y1, x2, y2)

iTime =  time.time()
# for l in range(svglength):
#     x1, y1, x2, y2 = vals[l] # Get from SVG
#     run = genF(x1, y1, x2, y2)
#     startT = starts[l]
#     threading.Timer( (startT + iTime - time.time()), run ).start()
