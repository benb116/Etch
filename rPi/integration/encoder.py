import smbus
asAddress = 0x36

# Get I2C bus
bus = smbus.SMBus(3)

#Reading Data from i2c bus3 3

def readAngle(left):
    # Set select pin hi or lo

    byte1 = bus.read_byte_data(asAddress, 0x0E) << 8
    byte2 = bus.read_byte_data(asAddress, 0x0F)
    return byte1 + byte2
