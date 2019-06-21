from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!benwashere'
socketio = SocketIO(app)

@app.route('/')
def home():
    return "Hello, World!"  # return a string

@socketio.on('message')
def handle_message(message):
    print(message)
    print('received message: ' + message)

@socketio.on('connect')
def test_connect():
    print('eee')
    emit('my response', {'data': 'Connected'})

if __name__ == '__main__':
    socketio.run(app)
