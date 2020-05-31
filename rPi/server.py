#! python3
from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, emit
import time
import threading
import json

from integration.pi_utils import IsRPi

onboard = False
if IsRPi():
    onboard = True
    from RPi import GPIO
    GPIO.setmode(GPIO.BCM)
    from integration.encoder import readAngle

def readAngle():
    pass

def genThreads(a, b, c, d):
    pass

from integration.auto import genThreads

app = Flask(__name__, static_folder='public')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SECRET_KEY'] = 'secret!benwashere'
socketio = SocketIO(app)

url = '/art/Test.json'
AUTO = True # Current mode
isConnected = False

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/index2.html')
def root2():
    return app.send_static_file('index2.html')

@app.route('/index3.html')
def root3():
    return app.send_static_file('index3.html')

@app.route('/socket.io.js')
def socketioFile():
    return app.send_static_file('socket.io.js')

@socketio.on('connect')
def on_connect():
    print('connected')
    isConnected = True
    # SendArtLink(url)

@socketio.on('AUTO')
def on_modeChange(a):
    AUTO = bool(a)
    print('mode changed to '+ ('auto' if AUTO else 'manual'))
    if AUTO:
        InitAuto()
    else:
        InitManual()


## AUTO MODE ##

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
    print(len(points))
    TS = time.time() + 0.00025 * len(points) + 0.5

    print('Init')
    threading.Thread(target=genThreads, args=(points, TS, pxSpeed, pxPerRev)).start()
    emit('startTime', TS);


## MANUAL MODE ##
stepsPerRev = 200
bitsPerStep = 20

oldVal = [0, 0]

def InitManual():
    if onboard:
        motors.toggle(0)
    oldVal[0] = readAngle(0)
    oldVal[1] = readAngle(1)
    print('InitManual')
    while ~AUTO:
        # print('check')
        time.sleep(0.01)
        checkTick(0)
        checkTick(1)
        # print('checkEnd')

def checkTick(mn):
    o = oldVal[mn]
    # print(o)
    n = readAngle(mn)
    print(n)
    diff = (n - o)
    if abs(diff) > 4096/2:
        diff = diff + 4096 * (-1 + 2*(o > n))
    if abs(diff) >= bitsPerStep:
        # emit('tick', (mn, round(diff/bitsPerStep)))
        # print('tick', (mn, round(diff/bitsPerStep)))
        oldVal[mn] = n

if __name__ == '__main__':
    print('begin')
    socketio.run(app)
    # InitManual()
