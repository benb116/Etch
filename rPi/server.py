#! python3
from flask import Flask
from flask_socketio import SocketIO, emit
import time
import threading
from multiprocessing import Process
import json
import sys

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

import eventlet
eventlet.monkey_patch()

from integration.pi_utils import IsRPi

def readAngle():
    pass

def genThreads(a, b, c, d):
    pass

onboard = False
if IsRPi():
    onboard = True
    from RPi import GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    from integration.encoder import readAngle
    from integration.motors import motorsOn
    motorsOn(False)

from integration.auto import genThreads

app = Flask(__name__, static_folder='public')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SECRET_KEY'] = 'secret!benwashere'
socketio = SocketIO(app)

url = '/art/Test.json'
AUTO = False # Current mode

# Serve static files
@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/index2.html')
def root2():
    return app.send_static_file('index2.html')

@app.route('/socket.io.js')
def socketioFile():
    return app.send_static_file('socket.io.js')

# Socket stuff
@socketio.on('connect')
def on_connect():
    print('connected')

@socketio.on('AUTO')
def on_modeChange(a):
    AUTO = bool(a)
    print('mode changed to '+ ('auto' if AUTO else 'manual'))
    if AUTO:
        InitAuto()
    else:
        InitManual()


## AUTO MODE ##

# Send an art file
@app.route('/art/<path:path>')
def send_art(path):
    print('art/'+path)
    return app.send_static_file('art/'+path)

def SendArtLink(url):
    emit('link', url)

@socketio.on('clientArtReady')
def on_clientArtReady(url):
    # pull file that we sent
    with open('public/'+url) as json_file:  
        data = json.load(json_file)
        points = data['points']
        pxSpeed = data['pxSpeed']
        pxPerRev = data['pxPerRev']

    # Determine unix start time
    TS = time.time() + 0.5
    # Begin stepping at the start time
    threading.Thread(target=genThreads, args=(points, TS, pxSpeed, pxPerRev)).start()
    # Tell the client when the start time is
    emit('startTime', TS);


## MANUAL MODE ##
stepsPerRev = 200
bitsPerStep = 20

oldVal = [0, 0]

def InitManual():
    # if onboard:
        # motors.toggle(0)
    oldVal[0] = readAngle(0)
    oldVal[1] = readAngle(1)
    # print('InitManual')
    # Repeatedly check both for changes
    while ~AUTO:
        # print('check')
        eventlet.sleep(0.01)
        checkTick(0)
        checkTick(1)
        # print('checkEnd')

def checkTick(mn):
    o = oldVal[mn]
    # print(o)
    n = readAngle(mn)
    # print(n)
    diff = (n - o)
    # If the diff is > half a rotation, assume it was less than half the other way
    if abs(diff) > 4096/2:
        diff = diff + 4096 * (-1 + 2*(o > n))
    # If above some threshold for a tick
    if abs(diff) >= bitsPerStep:
        # print('tick', (mn, round(diff/bitsPerStep)))
        socketio.emit('tick', (mn, round(diff/bitsPerStep)))
    oldVal[mn] = n

eventlet.spawn(InitManual)

if __name__ == '__main__':
    print('begin')
    socketio.run(app)

    try:
        socketio.run(app)
    except:
        pass
    finally:
        motorsOn(False)
        sys.exit(0)