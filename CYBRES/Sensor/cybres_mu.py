#!/usr/bin/env python3
import serial
import time

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
        
    # Finds the start of the next data set
    def find_start(self):
        start_found = False
        while not start_found:
            char = self.ser.read(1).decode('ascii')
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
            next_char = self.ser.read(1).decode('ascii')
            if (next_char == 'Z'):
                end_found = True
            else:
                line += next_char
            self.start_char = next_char
        return line[:-1]

    def get_initial_status(self):
        self.ser.write(b',ss*')
        return self._get_response(sleep_time=0.5)

    def restart(self):
        self.ser.write(b',sr*')

    def start_measurement(self):
        self.ser.write(b',ms*')
        
    def stop_measurement(self):
        self.ser.write(b',mp*')

    def set_measurement_interval(self, interval):
        set_interval = ',mi{:05}*'.format(interval)
        self.ser.write(set_interval.encode())
        return self._get_response(sleep_time=0.5)

    def to_flash(self):
        self.ser.write(b'sf2*')

    def read_all_lines(self):
        self.ser.write(b'f1*') #f1, mr
        while True:
            line = self.get_next()
            print(line)

    def read_all(self):
        self.ser.write(b'f1*')
        counter = 0
        while True:
            char = self.return_serial()
            print(char, end='')
            if char == 'A':
                counter +=1
                print(f"-----------------{counter}---------------------------")
    
    def _get_response(self, sleep_time=0.1):
        time.sleep(sleep_time)
        response = ""
        while self.ser.in_waiting:
            response += self.ser.read(1).decode('ascii')
        return response


def main():
    
    mu = Cybres_MU('/dev/ttyACM0')
    mu.set_measurement_interval(1000)
    mu.to_flash()
    mu.start_measurement()
    time.sleep(180)
    print("Now reading")
    mu.read_all()


if __name__ == '__main__':
    main()