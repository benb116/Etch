from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, emit
import time
import threading
import json

from integration.auto import genThreads
app = Flask(__name__, static_folder='public')
app.config['SECRET_KEY'] = 'secret!benwashere'
socketio = SocketIO(app)

@app.route('/')
def root():
    print('ee')
    return app.send_static_file('index.html')

@app.route('/socket.io.js')
def socketioFile():
    return app.send_static_file('socket.io.js')

@app.route('/art/<path:path>')
def send_art(path):
    print('art'+path)
    return app.send_static_file('art/'+path)

@socketio.on('connect')
def on_connect():
    print('connected')
    SendArtLink(url)

url = '/art/1.json'

def SendArtLink(url):
    emit('link', url)
    # Init all linethreads in an array

@socketio.on('clientArtReady')
def on_clientArtReady():
    # Determine unix start time
    TS = time.time() + 0.500

    with open('public/'+url) as json_file:  
        data = json.load(json_file)
        points = data['points']

    threading.Thread(target=Init, args=(points, TS)).start()
    emit('startTime', TS);

def Init(pts, ts):
    print('Init')
    threadlist = genThreads(pts, ts)

points = []

# points = [[1, 1], [1, 10], [10, 10], [10, 1], [1, 1], [5, 5], [10, 10], [10, 1], [1, 1]]

# threading.Thread(target=Init, args=(points, time.time()+0.500)).start()

if __name__ == '__main__':
    socketio.run(app)
