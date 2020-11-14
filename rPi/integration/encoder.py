import smbus
asAddress = 0x36

# Get I2C bus
bus = smbus.SMBus(1)

# left is 0 or 1 for different mux addresses
def readAngle(left):

    # Tell the mux which sensor to connect to
    bus.write_byte_data(0x70, 0, 0x00)
    bus.write_byte_data(0x70, 0, 0x04 + left)

    # Read data registers and calculate raw angle value
    byte1 = bus.read_byte_data(asAddress, 0x0E) << 8
    byte2 = bus.read_byte_data(asAddress, 0x0F)
    return byte1 + byte2
