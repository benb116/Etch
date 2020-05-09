# Electrical Hardware

Along with a Raspberry Pi, Etch utilized two unique custom PCBs to route signals and power to various components.

1. Motor Board - Mounted to the back of each stepper motor, this PCB has a stepper motor driver to control the knob and a magnetic encoder to read the knob's position.
2. Breakout board - Mounted in the center of the TV, this PCB routes signals from the Pi to the correct board. It also distributes motor power.

## Motor Board
![Image](https://github.com/benb116/Etch/raw/master/EE/MotorBoard/Motor%20Board%20Schematic.png)

The Motor Board is designed to fit on the back of a standard NEMA-17 stepper motor. It accepts power and a 10-pin signal cable from the Breakout Board. The signals are:

1. **-ENABLE** | *Active low* | Pull this pin high to disable the motor (used in MANUAL mode)
2. **MS1** | *Pulled low* | Microstepping control pin 1
3. **GND**
4. **MS2** | *Pulled low* | Microstepping control pin 2
5. **SDA** | *Pulled high* | Encoder I2C Data pin
6. **MS3** | *Pulled low* | Microstepping control pin 3
7. **SCL** | *Pulled high* | Encoder I2C Clock pin
8. **STEP** | *Pulled low* | Step pulse pin
9. **3v3**
10. **DIR** | *Pulled low* | Direction control pin

Along with support components and connectors, the PCB has two main parts:

1. Headers that match an A4988 stepper motor driver (or similar) on the top layer
2. An [AS5600 magnetic rotary position sensor](https://ams.com/as5600) on the bottom layer. The sensor is centered over the motor axle which has a [diametrically polarized magnet](https://www.kjmagnetics.com/magdir.asp) glued to it. As the motor turns, the magnetic field rotates and is read by the sensor.
	- Note: Data is read from the sensor over I2C. The sensor has a fixed I2C address, so a 2-channel i2c bus multiplexer is required (see Breakout Board)

## Breakout Board

The Breakout Board mainly routes signals and power between the Pi and the two Motor Boards. It accepts power from a DC jack and 12 signals from the Pi via jumper cable:

1. **3v3**
2. **-ENABLE** | *Active low* | Pull this pin high to disable the motors (used in MANUAL mode)
3. **SCL** | *Pulled high* | Multiplexer I2C Clock pin
4. **DIRB** | *Pulled low* | Direction control pin for motor B
5. **SDA** | *Pulled high* | Multiplexer I2C Data pin
6. **STEPB** | *Pulled low* | Step pulse pin for motor B
7. **MS3** | *Pulled low* | Microstepping control pin 3
8. **DIRA** | *Pulled low* | Direction control pin for motor A
9. **MS2** | *Pulled low* | Microstepping control pin 2
10. **STEPA** | *Pulled low* | Step pulse pin for motor A
11. **MS1** | *Pulled low* | Microstepping control pin 1
12. **GND**

Along with connectors, the Breakout Board has a [PCA9540 2-channel I2C multiplexer](https://www.nxp.com/docs/en/data-sheet/PCA9540B.pdf). Because the AS5600 has a fixed address, the Pi can use the mux to select which sensor to read before pulling the value. This is done by writing the desired channel to a mux register before reading the value.

## Connectors

The 10-pin connections between Breakout and Motor Boards are sent over [2-foot Samtec cables (FFSD-05-D-24.00-01-N-R)](https://www.samtec.com/products/ffsd-05-d-24.00-01-n-r) to [surface-mount headers (FTSH-105-01-F-DV-K)](https://www.samtec.com/products/ftsh-105-01-f-dv-k). These were chosen because they are compact, long enough, keyed to prevent miswiring, and because free samples are available from the manufacturer. Future versions of the boards will use through-hole connectors which are easier to solder. If there is interest in a general-use Motor Board, the connectors may be swapped for 0.1" pitch connectors to preserve compatibility with existing DIY electronics kits.