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
    global haLastState
    try:
        # haState = GPIO.input(ha)
        if haState != haLastState:
            # hbState = GPIO.input(hb)
            if hbState != haState:
                Tick(1, 1)
            else:
                Tick(1, -1)
        haLastState = haState
    finally:
        print "Ending"



counter = 0
# clkLastState = GPIO.input(clk)
# TODO: Change to rising if necessary
# TODO: Need a debounce?
# GPIO.add_event_detect(clk, GPIO.FALLING  , callback=horCallback, bouncetime=300)  

# GPIO.cleanup()