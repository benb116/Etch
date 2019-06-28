#! python3
from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, emit
import time
import threading
import json
# from RPi import GPIO
# GPIO.setmode(GPIO.BCM)

from integration.auto import genThreads

app = Flask(__name__, static_folder='public')
app.config['SECRET_KEY'] = 'secret!benwashere'
socketio = SocketIO(app)

url = '/art/test.json'
AUTO = True # Current mode
isConnected = False

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/socket.io.js')
def socketioFile():
    return app.send_static_file('socket.io.js')

@socketio.on('connect')
def on_connect():
    print('connected')
    isConnected = True
    SendArtLink(url)

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
    print('art'+path)
    return app.send_static_file('art/'+path)

def SendArtLink(url):
    emit('link', url)

@socketio.on('clientArtReady')
def on_clientArtReady():
    # pull file that we sent
    with open('public/'+url) as json_file:  
        data = json.load(json_file)
        points = data['points']
        pxSpeed = data['pxSpeed']
        pxPerRev = data['pxPerRev']

    # Determine unix start time
    TS = time.time() + 0.500

    print('Init')
    threading.Thread(target=genThreads, args=(points, TS, pxSpeed, pxPerRev)).start()
    emit('startTime', TS);


## MANUAL MODE ##

# Input pins
ha = 17 # Horizontal motor A pin
hb = 18
va = 19
vb = 20

# TODO: Determine pull up or down
# GPIO.setup(ha, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(hb, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(va, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(vb, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# haLastState = GPIO.input(ha)
# vaLastState = GPIO.input(va)

def horCallback(channel):  
    # haState = GPIO.input(ha)
    if haState != haLastState:
        haLastState = haState
        # hbState = GPIO.input(hb)
        if hbState == haState:
            emit('tick', (1, 1)) 
        else:
            emit('tick', (1, -1)) 

def verCallback(channel):  
    # vaState = GPIO.input(va)
    if vaState != vaLastState:
        vaLastState = vaState
        # vbState = GPIO.input(vb)
        if vbState == vaState:
            emit('tick', (2, 1)) 
        else: 
            emit('tick', (2, -1))

def InitManual():
    # motors.turnOff()
    # GPIO.add_event_detect(ha, GPIO.FALLING  , callback=horCallback, bouncetime=300)
    # GPIO.add_event_detect(va, GPIO.FALLING  , callback=verCallback, bouncetime=300)
    pass


if __name__ == '__main__':
    print('begin')
    socketio.run(app)
