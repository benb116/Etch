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

def readAngle(m):
    pass

def genThreads(a, b, c, d):
    pass

def motorsOn(bool):
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
    if onboard:
        threading.Thread(target=genThreads, args=(points, TS, pxSpeed, pxPerRev)).start()
    # Tell the client when the start time is
    emit('startTime', TS);


## MANUAL MODE ##
oldVal = [0, 0]

def InitManual():
    oldVal[0] = readAngle(0)
    oldVal[1] = readAngle(1)
    if not onboard:
        return

    while ~AUTO:
        eventlet.sleep(0.01)
        checkTick(0)
        checkTick(1)

def checkTick(mn):
    o = oldVal[mn]
    n = readAngle(mn)
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