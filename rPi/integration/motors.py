import RPi.GPIO as GPIO
import pigpio
import time
GPIO.setmode(GPIO.BCM)
PIGPIO = pigpio.pi()


# Microstepping resolution mapping to select pins hi or lo
resolution = {'Full': (0, 0, 0),
              'Half': (1, 0, 0),
              '1/4': (0, 1, 0),
              '1/8': (1, 1, 0),
              '1/16': (1, 1, 1)}
MODE = resolution['1/16']

pEnab = 7 # Enable pin
pRes = (14, 15, 17) # Microstep resolution pins

pStp1 = 12 # Motor 1 step pin
pDir1 = 20 # Motor 1 direction pin
pStp2 = 13 # Motor 2 step pin
pDir2 = 6  # Motor 2 direction pin

GPIO.setup(pEnab, GPIO.OUT)
GPIO.setup(pRes, GPIO.OUT)
GPIO.setup(pStp1, GPIO.OUT)
GPIO.setup(pDir1, GPIO.OUT)
GPIO.setup(pStp2, GPIO.OUT)
GPIO.setup(pDir2, GPIO.OUT)

PIGPIO.set_mode(pStp1, pigpio.ALT5)
PIGPIO.set_mode(pStp2, pigpio.ALT5)

GPIO.output(pEnab, 0)
GPIO.output(pRes, 0)
GPIO.output(pStp1, 0)
GPIO.output(pDir1, 0)
GPIO.output(pStp2, 0)
GPIO.output(pDir2, 0)

stepdelay = 0.0002

# Turn the motors on or off using the enable pin
def motorsOn(on):
    GPIO.output(pEnab, 1-on) # Active low
    if not on:
        setDirAndFreq(0, 0, 0)
        setDirAndFreq(1, 0, 0)

# Set the microstepping resolution
def setRes(res):
    MODE = resolution[res]
    GPIO.output(pRes, resolution[res])

# Send a single step command to a motor in a direction
def step(mn, mdir):
    # print(mn, mdir)
    msPin = pStp1 if mn == 1 else pStp2
    mdPin = pDir1 if mn == 1 else pDir2
    GPIO.output(mdPin, 1 if mdir == 1 else 0)
    GPIO.output(msPin, 1)
    time.sleep(stepdelay)
    GPIO.output(msPin, 0)
    time.sleep(stepdelay)

# Set up PWM at 50% duty and a certain direction and frequency
def setDirAndFreq(mn, mdir, freq):
    msPin = pStp1 if mn == 1 else pStp2
    mdPin = pDir1 if mn == 1 else pDir2
    GPIO.output(mdPin, 1 if mdir == 1 else 0)
    if freq == 0:
        PIGPIO.write(msPin, 0)
        return
    PIGPIO.hardware_PWM(msPin, freq, 500000)

# Return the total number of steps in one revolution based on the current microstep resolution
def stepsPerRev():
    exp = MODE[2]*4 + MODE[1]*2 + MODE[0]
    if exp == 7:
        exp = 4
    return 200 * 2**exp

setRes('1/16')