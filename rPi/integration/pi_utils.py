def IsRPi():
    try:
        import RPi.GPIO as gpio
        test_environment = True
    except (ImportError, RuntimeError):
        test_environment = False

    return test_environment