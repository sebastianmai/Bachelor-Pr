#!/usr/bin/env python3
import serial


class Cybres_MU:

    def __init__(self, port_name, baudrate=460800):
        self.ser = serial.Serial(
            port=port_name,
            baudrate=baudrate,
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

        self.start_char = "Z"
        

    def return_serial(self):
        return self.ser.read(1).decode('ascii')

    # Finds the start of the next data set
    def find_start(self):
        start_found = False
        while not start_found:
            char = self.return_serial()
            if char == 'A':
                start_found = True
        self.start_char = char
    
    # Returns the next complete data set
    def get_next(self):
        line = ""
        end_found = False
        if self.start_char != 'A':
            self.find_start()
        while not end_found:
            next_char = self.return_serial()
            if (next_char == 'Z'):
                end_found = True
            else:
                line += next_char
            self.start_char = next_char
        return line[:-1]

    def restart(self):
        # restart MU
        self.ser.write(b'sr*')


    def start_measurement(self):
        # start measurement
        self.ser.write(b'ms*')
        

    def stop_measurement(self):
        # stop measurement
        self.ser.write(b'mp*')

    
    def set_measurement_interval(self, interval):
        # set interval between measurements
        set_interval = 'mi{:05}*'.format(interval)
        self.ser.write(set_interval.encode())

