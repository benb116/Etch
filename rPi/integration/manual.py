# https://gist.github.com/codelectron/ca794fe1fef6c71b08b27008e5501776#file-rotaryencoder_rpi_int-py

# from RPi import GPIO
# GPIO.setmode(GPIO.BCM)

# Input pins
ha = 17 # Horizontal motor A pin
hb = 18
va = 19
vb = 20

# TODO: Determine pull up or down
# GPIO.setup(ha, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(hb, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(va, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(vb, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# haLastState = GPIO.input(ha)
# vaLastState = GPIO.input(va)

def horCallback(channel):  
    try:
        # haState = GPIO.input(ha)
        if haState != haLastState:
            haLastState = haState
            # hbState = GPIO.input(hb)
            if hbState == haState:
                return (1, 1)
            else:
                return (1, -1)
        return (0, 0)
    finally:
        print "Ending"

def verCallback(channel):  
    try:
        # vaState = GPIO.input(va)
        if vaState != vaLastState:
            vaLastState = vaState
            # vbState = GPIO.input(vb)
            if vbState == vaState:
                return (2, 1)
            else:
                return (2, -1)
        return (0, 0)
    finally:
        print "Ending"

# TODO: Change to rising if necessary
# TODO: Need a debounce?
# GPIO.add_event_detect(ha, GPIO.FALLING  , callback=horCallback, bouncetime=300)  
# GPIO.add_event_detect(va, GPIO.FALLING  , callback=verCallback, bouncetime=300)  

# GPIO.cleanup()