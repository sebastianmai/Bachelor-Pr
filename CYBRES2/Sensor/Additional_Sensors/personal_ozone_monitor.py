import serial

class POM():
    def __init__(self):
        self.ser = serial.Serial(
            port='/dev/ttyACM0',
            baudrate=19200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1,
            xonxoff=False,
            rtscts=True,
            dsrdtr=True
        )
        
        self.ser.flushInput()
        self.ser.flushOutput() # Just in case...
        self.ser.close()
        self.ser.open()

        self.ser.write(b'l')

    def getData():
        pass

pom = POM()

