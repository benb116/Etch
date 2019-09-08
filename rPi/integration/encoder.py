import smbus
asAddress = 0x36

# Get I2C bus
bus3 = smbus.SMBus(3)
bus4 = smbus.SMBus(4)

#Reading Data from i2c bus3 3

def readAngle(left):
    bus = (bus3 if left else bus4)
    byte1 = bus.read_byte_data(asAddress, 0x0E) << 8
    byte2 = bus.read_byte_data(asAddress, 0x0F)
    return byte1 + byte2
