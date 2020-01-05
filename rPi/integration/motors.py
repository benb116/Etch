import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)

ENABLED = 0

resolution = {'Full': (0, 0, 0),
              'Half': (1, 0, 0),
              '1/4': (0, 1, 0),
              '1/8': (1, 1, 0),
              '1/16': (1, 1, 1)}

MODE = resolution['1/8']

pSlp = 12
pRes = (14, 15, 18)

pStp1 = 21
pDir1 = 20
pStp2 = 26
pDir2 = 19

GPIO.setup(pSlp, GPIO.OUT)
GPIO.setup(pRes, GPIO.OUT)
GPIO.setup(pStp1, GPIO.OUT)
GPIO.setup(pDir1, GPIO.OUT)
GPIO.setup(pStp2, GPIO.OUT)
GPIO.setup(pDir2, GPIO.OUT)

GPIO.output(pSlp, 0)
GPIO.output(pRes, 0)
GPIO.output(pStp1, 0)
GPIO.output(pDir1, 0)
GPIO.output(pStp2, 0)
GPIO.output(pDir2, 0)

stepDelay = 0.001

def toggle(onoff):
    ENABLED = onoff
    GPIO.output(pSlp, ENABLED)

def turnOn():
    ENABLED = 1
    GPIO.output(pSlp, ENABLED)

def turnOff():
    ENABLED = 0
    GPIO.output(pSlp, ENABLED)

def setRes(res):
    MODE = resolution[res]
    GPIO.output(pRes, resolution[res])

def step(mn, mdir):
    if not ENABLED:
        toggle(1)

    msPin = pStp1 if mn == 1 else pStp2
    mdPin = pDir1 if mn == 1 else pDir2
    GPIO.output(mdPin, 1 if mdir == 1 else 0)
    GPIO.output(msPin, 1)
    time.sleep(stepdelay)
    GPIO.output(msPin, 0)
    time.sleep(stepdelay)

def stepsPerRev():
    exp = MODE[2]*4 + MODE[1]*2 + MODE[0]
    return 200 * 2**exp

toggle(ENABLED)
setRes('1/8')

# for x in range(1,1600):
#     step(1, 1)
#     time.sleep(0.005)

# good practise to cleanup GPIO at some point before exit
GPIO.cleanup()