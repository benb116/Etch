# import RPi.GPIO as GPIO
from adafruit_motor import stepper
from adafruit_motorkit import MotorKit
import atexit

kit = MotorKit()

def turnOff():
    kit.stepper1.release()
    kit.stepper2.release()

atexit.register(turnOff)

def step(mn, dir):
    stepdir = stepper.FORWARD if dir else stepper.BACKWARD
    if mn == 1:
        kit.stepper1.onestep(direction=stepdir, style=stepper.SINGLE)
    else:
        kit.stepper2.onestep(direction=stepdir, style=stepper.SINGLE)
