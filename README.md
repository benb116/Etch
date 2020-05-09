# Etch

Etch is an auto-drawing digital Etch a Sketch built on top of a Raspberry Pi. Stepper motors mounted under a television turn 3D printed knobs to match the drawing motion on the screen. The knobs can also be turned by hand to manually draw lines on the display.

This repository contains the following:

* 3D models of the shell, knobs, and other support pieces
* Altium PCB design files and Gerber files for the custom circuit boards
* RPi files for the server and motor/sensor integration
* Python scripts for generating art files from images

### Hardware

Etch is built around a Raspberry Pi connected to a TV and two stepper motors. The Pi is connected to a custom breakout board that links with two other custom boards mounted to the back of the motors.

The motor boards serve two functions:

1. AUTO Mode - Take in step and direction signals from the breakout board and route them to the onboard stepper driver.
2. MANUAL Mode - Sense the orientation of a diametrically polarized magnet on the back of the motor axis. The motor position is then sent to the Pi to monitor the user's input.

The motors, shaft couplers, and knobs are mounted to two brackets that are adhered to the TV. Lasercut polystyrene sheets form the outer shell and are positioned using internal ribs.

### Software

The Pi runs a Python webserver that integrates with the motors and sensors. The server communicates with a webpage running in a browser that is displayed on the screen. A websocket connection passes information back and forth about art data, timing, and user input. In AUTO Mode, the server also runs through a script of artwork to display.

Art files contain the lists of points in order of travel. They also include information about drawing speed and knob rotation rates. These files can be generated with different methods:

* [ImageGen.py](https://github.com/benb116/Etch/blob/master/Art/ImageGen.py) - Take a full image and generate a path that traces contours and fills in darker regions.
* [PathFinder.py](https://github.com/benb116/Etch/blob/master/Art/pathfinder.py) - Identifies the path in a single line drawing
* DIY - You can use any method that outputs a series of line segments

## Further work
- [ ] Improve artwork generation algorithms
- [ ] Upgrade the custom PCBs
- [ ] Develop a show script
- [ ] Add user input to toggle between AUTO and MANUAL

## Acknowledgements

A big thanks to many people and organizations for their help and advice on this project.

* Everyone at [Bresslergroup](https://www.bresslergroup.com/), especially John, Jason, Steve, Nick, and Mike
* [Evil Mad Scientist](https://wiki.evilmadscientist.com/StippleGen)
* [Andrew Brooks](https://brooksandrew.github.io/simpleblog/articles/intro-to-graph-optimization-solving-cpp/)
* [Sunny Days](http://sunnybala.com/2018/09/10/python-etch-a-sketch.html)
* [Adrian Rosebrock](https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/)
* [Joe Freeman](https://web.archive.org/web/20140918131540/http://joefreeman.co.uk/blog/2009/09/lineographic-interpretations-of-images-with-an-etch-a-sketch/)
* [MIT](http://web.mit.edu/urban_or_book/www/book/chapter6/6.4.4.html)