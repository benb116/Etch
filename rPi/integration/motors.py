import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)

resolution = {'Full': (0, 0, 0),
              'Half': (1, 0, 0),
              '1/4': (0, 1, 0),
              '1/8': (1, 1, 0),
              '1/16': (1, 1, 1)}

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

GPIO.output(pEnab, 0)
GPIO.output(pRes, 0)
GPIO.output(pStp1, 0)
GPIO.output(pDir1, 0)
GPIO.output(pStp2, 0)
GPIO.output(pDir2, 0)

stepdelay = 0.0001

# Turn the motors on or off using the enable pin
def motorsOn(on):
    GPIO.output(pEnab, 1-on) # Active low

# Set the microstepping resolution
def setRes(res):
    MODE = resolution[res]
    print(resolution[res])
    GPIO.output(pRes, resolution[res])

# Send a step command to a motor in a direction
def step(mn, mdir):
    # print(mn, mdir)
    msPin = pStp1 if mn == 1 else pStp2
    mdPin = pDir1 if mn == 1 else pDir2
    GPIO.output(mdPin, 1 if mdir == 1 else 0)
    GPIO.output(msPin, 1)
    time.sleep(stepdelay)
    GPIO.output(msPin, 0)
    time.sleep(stepdelay)

def setDirAndFreq(mn, mdir, freq):
    msPin = pStp1 if mn == 1 else pStp2
    mdPin = pDir1 if mn == 1 else pDir2
    GPIO.output(mdPin, 1 if mdir == 1 else 0)
    pwm = GPIO.PWM(msPin, freq)
    pwm.start(50)

# Return the total number of steps in one revolution based on the microstep resolution
def stepsPerRev():
    exp = MODE[2]*4 + MODE[1]*2 + MODE[0]
    return 200 * 2**exp

MODE = resolution['Full']
# # DISABLED = 0
setRes('Full')
# motorsOn(False)

# try:
  # setDirAndFreq(1, 1, 1000)

# except Exception as e:
#   print("Ctl C pressed - ending program")

#   pwm.stop()                         # stop PWM
#   GPIO.cleanup()                     # resets GPIO ports used back to input mode
# print('222ee')
# # turnOff()
# # time.sleep(2)
# turnOn()
# # time.sleep(2)
# turnOff()
# # time.sleep(2)
# # turnOn()
# # time.sleep(2)
# # turnOff()
# # time.sleep(2)
# turnOn()
# DISABLED = 0
# print('333')
# for x in range(1,400):
#     print('st')
#     step(2, 0)
#     time.sleep(0.01)
# # for x in range(1,200):
#     # print('st')
#     # step(0, 1)
#     # time.sleep(0.01)
# # time.sleep(2)

# # turnOff()
# # time.sleep(200)

# # # turnOn()
# # # good practise to cleanup GPIO at some point before exit
# GPIO.cleanup()