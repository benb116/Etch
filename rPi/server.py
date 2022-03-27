#! python3
from flask import Flask
from flask_socketio import SocketIO, emit
import time
import threading
from multiprocessing import Process
import json
import sys
import logging
import eventlet

from integration.pi_utils import IsRPi
from integration.auto import genThreads

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
eventlet.monkey_patch()


# Dummy functions that are overwritten if on rPi
def readAngle(m):
    pass


def genThreads(a, b, c, d):
    pass


def motorsOn(bool):
    pass


# Are we running on the rPi
onboard = False
# Want to be able to run the server off the Pi, but local doesn't have libraries
if IsRPi():
    onboard = True
    from RPi import GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    from integration.encoder import readAngle
    from integration.motors import motorsOn
    motorsOn(False)

app = Flask(__name__, static_folder='public')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SECRET_KEY'] = 'secret!benwashere'
socketio = SocketIO(app)


# Serve static files
@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/canvas.html')
def root2():
    return app.send_static_file('canvas.html')


@app.route('/socket.io.js')
def socketioFile():
    return app.send_static_file('socket.io.js')


# Socket stuff
@socketio.on('connect')
def on_connect():
    print('connected')


# AUTO MODE #
# Send an art file
@app.route('/art/<path:path>')
def send_art(path):
    print('art/'+path)
    return app.send_static_file('art/'+path)


# After the frontend has downloaded and precomputed the points, it emits a "clientArtReady" message
@socketio.on('clientArtReady')
def on_clientArtReady(url):
    # pull file that we sent
    with open('public/'+url) as json_file:
        data = json.load(json_file)
        points = data['points']
        pxSpeed = data['pxSpeed']
        pxPerRev = data['pxPerRev']

    # Determine unix start time
    # The front and backends will attempt to start the drawing at the same timestamp
    TS = time.time() + 0.5
    # Begin stepping at the start time
    if onboard:
        threading.Thread(target=genThreads, args=(points, TS, pxSpeed, pxPerRev)).start()
    # Tell the client when the start time is
    emit('startTime', TS)


# MANUAL MODE #
oldVal = [0, 0]


def InitManual():
    if not onboard:
        return
    # Read initial values from sensors
    oldVal[0] = readAngle(0)
    oldVal[1] = readAngle(1)

    while ~AUTO:
        eventlet.sleep(0.01)
        checkTick(0)
        checkTick(1)


# Check if a motor position has changed
def checkTick(mn):
    bitsPerStep = 10
    o = oldVal[mn]
    n = readAngle(mn)
    diff = (n - o)
    # If the diff is > half a rotation, assume it was less than half the other way
    if abs(diff) > 4096/2:
        diff = diff + 4096 * (-1 + 2*(o > n))
    # If above some threshold for a tick
    if abs(diff) >= bitsPerStep:
        # Emit a tick event to the frontend
        socketio.emit('tick', (mn, round(diff/bitsPerStep)))
        oldVal[mn] = n


eventlet.spawn(InitManual)

if __name__ == '__main__':
    print('begin')
    try:
        socketio.run(app)
    finally:
        # Turn off motors on exit
        motorsOn(False)
        sys.exit(0)
