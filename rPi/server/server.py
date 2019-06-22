from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!benwashere'
socketio = SocketIO(app)

@app.route('/')
def home():
    return "Hello, World!"  # return a string

@socketio.on('connect')
def on_connect():
    print('connected')

def SendArtLink(url):
    emit('link', url)
    # Init all linethreads in an array

@socketio.on('clientArtReady')
def on_clientArtReady():
    # Determine unix start time
    TS = time.time() + 0.500
    print(time.time())
    Init(PTS, TS)
    print(time.time())
    emit('startTime', TS);

if __name__ == '__main__':
    socketio.run(app)
