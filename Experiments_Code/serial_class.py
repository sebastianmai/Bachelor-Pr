import serial
from datetime import datetime
import csv
import os

class PhytNode_Serial:
    def __init__(self, port: str, device_no: str, baudrate=19200, parity=serial.PARITY_ODD, timeout=1)->None:
        """
        Initializes the serial connection and creates the directory for the data
        Args:
            port (str): port of the serial connection
            device_no (str): device number of the phytosensor
            baudrate (int): baudrate of the serial connection
            parity (serial.PARITY_ODD): parity of the serial connection
            timeout (int): timeout of the serial connection
        """
        self.ser = serial.Serial(port, baudrate, timeout=timeout, parity=parity)
        self.ser.flushInput()  # Clear any existing data in the input buffer
        self.wrong_data_counter = 0

        self.directory = '/home/pi/Measurements/'
        self.device_no = None

        while not self.read():
            self.read()
        self.file_prefix = "P" + str(self.device_no)  # reading the number based on the number sent in buffbuff[1]
        # self.file_prefix = "P" + str(device_no)
        self.specific_directory = self.directory + self.file_prefix

        if not os.path.exists(self.specific_directory):
            os.makedirs(self.specific_directory)

        self.header = ["timestamp", "differential_potential", "filtered"]  # headers for the csv file
        self.start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3]
        self.last_csv_time = datetime.now()
        self.file_path = os.path.join(self.specific_directory, f'{self.file_prefix}_{self.start_timestamp}.csv')
        print("started: ", self.start_timestamp, self.file_path)


    def int_from_bytes(self, xbytes: bytes) -> int:
        """
        Converts bytes to int
        Args:
            xbytes (bytes): bytes to convert
        """
        return int.from_bytes(xbytes, 'big')

    def read(self)-> int:
        """
        Reads the data from the serial connection
        Args:
        """
        data = self.ser.readline()
        if (len(data) == 0 or len(data) != 6 or data[0] != 80): # check if data is correct
            # print(self.file_prefix+ " " + datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], "not correct", data)
            self.wrong_data_counter += 1
            self.ser.flushInput()
            return None
        self.device_no = data[1]
        data = self.int_from_bytes(data[2:-1])

        return data

    def getVolt(self, data):
        Vref = 2.5
        Gain = 4
        databits = 8388608

        volt = data / databits
        volt1 = volt - 1
        volt2 = volt1 * Vref / Gain
        return volt2 * 1000

    def write2csv(self)->None:
        """
        Writes the data to the csv file
        Args:
        """

        data = self.read()
        print(data)
        if data is None:
            return
        current_time = datetime.now()
        if current_time.hour in {0, 12} and current_time.hour != self.last_csv_time.hour: # create new file every 12 hours
            self.last_csv_time = datetime.now()
            self.start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3]
            self.file_path = os.path.join(self.specific_directory, f'{self.file_prefix}_{self.start_timestamp}.csv')
            print("new file: ", self.start_timestamp)

        saved_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3]
        with open(self.file_path, "a") as f:
            writer = csv.writer(f, delimiter=",")

            if os.path.getsize(self.file_path) == 0:
                writer.writerow(self.header)

            writer.writerow([saved_time, data, 0])

    def write2csv_thread(self)->None:
        """
        Writes the data to the csv file
        Args:
        """
        while(True):
            data = self.read()
            #print(data, self.wrong_data_counter, self.file_prefix)
            if data is None:
                continue
            current_time = datetime.now()
            if current_time.hour in {0, 12} and current_time.hour != self.last_csv_time.hour: # create new file every 12 hours
                self.last_csv_time = datetime.now()
                self.start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3]
                self.file_path = os.path.join(self.specific_directory, f'{self.file_prefix}_{self.start_timestamp}.csv')
                print("new file: ", self.start_timestamp)

            saved_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3]
            with open(self.file_path, "a") as f:
                writer = csv.writer(f, delimiter=",")

                if os.path.getsize(self.file_path) == 0:
                    writer.writerow(self.header)

                writer.writerow([saved_time, self.getVolt(data), 0])

