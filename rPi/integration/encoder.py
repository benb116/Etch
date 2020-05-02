import smbus
asAddress = 0x36

# Get I2C bus
bus = smbus.SMBus(1)

#Reading Data from i2c bus3 3

def readAngle(left):
    # Set select pin hi or lo
    # print(0x04 + left)
    bus.write_byte_data(0x70, 0, 0x00)
    bus.write_byte_data(0x70, 0, 0x04 + left)

    byte1 = bus.read_byte_data(asAddress, 0x0E) << 8
    byte2 = bus.read_byte_data(asAddress, 0x0F)
    return byte1 + byte2
