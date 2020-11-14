# Raspberry Pi Code

The RPi acts as the brain of the system, directing the motors, receiving user input, and orchestrating and displaying the artwork. It consists of a Python webserver integrated with motor control and I2C data firmware that communicates with a web-based frontend over websockets.

## Webserver

The Pi runs a Flask webserver with websocket capabilities. It has three main functions:

1. Communicate with the front webpage that displays the animated artwork. It does this via websocket communication and serving requests for data JSON files.
2. Send precisely timed step commands to the motors that are synchronized with the art animation (this occurs in AUTO mode).
3. Repeatedly read the position data from both sensors and calculate the corresponding pointer movement (this occurs in MANUAL mode).

## Frontend

The Pi is also connected to the TV and displays a webpage in Chromium. The webpage operates differently based on the mode:

1. In AUTO mode, the page communicates with the server using websockets and AJAX requests to send and receive art data. It then creates an SVG with the data and animates the path being drawn.
2. In MANUAL mode, the page receives websocket "ticks" that tell it to draw horizontally or vertically on a Canvas element.

## Art files

Art data is stored in JSON objects with the following properties:

* Name
* pxSpeed - How fast should the pointer move across the screen (pixels per second)
* pxPerRev - How many pixels to move the pointer for a full knob rotation
* points - an array of points to draw lines between sequentially (each point is [x,y])