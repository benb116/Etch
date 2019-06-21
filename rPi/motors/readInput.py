import time
import threading
import datetime

from timeThread import genThread

hpin = 4
vpin = 5

hval = 0
vval = 0

hold = 0
vold = 0

# import RPi.GPIO as GPIO           # import RPi.GPIO module  
# GPIO.setmode(GPIO.BCM)            # choose BCM or BOARD  
# GPIO.setup(hpin, GPIO.IN)  # set a port/pin as an input  
# GPIO.setup(hpin, GPIO.IN)


# i = GPIO.input(port_or_pin) 
def readPins():
    # hval = GPIO.input(hpin)
    # vval = GPIO.input(vpin)
    print(datetime.datetime.now())
    if (hval & !vval):
        


readms = 10
readThread = genInterval(readms, -1, readPins, 3)
readThread.start()