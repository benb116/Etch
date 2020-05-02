import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)

resolution = {'Full': (0, 0, 0),
              'Half': (1, 0, 0),
              '1/4': (0, 1, 0),
              '1/8': (1, 1, 0),
              '1/16': (1, 1, 1)}

# MODE = resolution['1/16']

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

stepdelay = 0.0001

def turnOn():
    GPIO.output(pSlp, 0) # Active low

def turnOff():
    GPIO.output(pSlp, 1)

def setRes(res):
    MODE = resolution[res]
    print(resolution[res])
    GPIO.output(pRes, resolution[res])

def step(mn, mdir):
    print(mn, mdir)
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

# DISABLED = 0
setRes('1/4')
print('222ee')
# turnOff()
# time.sleep(2)
# turnOn()
# time.sleep(2)
# turnOff()
# time.sleep(2)
# turnOn()
# time.sleep(2)
# turnOff()
# time.sleep(2)
turnOn()
# DISABLED = 0
# print('333')
# for x in range(1,400):
#     print('st')
#     step(1, 0)
#     time.sleep(0.01)
# # for x in range(1,200):
#     # print('st')
#     step(0, 1)
#     # time.sleep(0.01)
# # time.sleep(2)

# turnOff()
# time.sleep(200)

# # turnOn()
# # good practise to cleanup GPIO at some point before exit
# GPIO.cleanup()