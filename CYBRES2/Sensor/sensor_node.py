#!/usr/bin/env python3
import sys
import re
import time
import logging
import datetime
import numpy as np

from cybres_mu import Cybres_MU
# from Additional_Sensors.rgbtcs34725 import RGB_TCS34725
from zmq_publisher import ZMQ_Publisher

sys.path.append("..") # Adds higher directory to python modules path.
from Utilities.data2csv import data2csv

class Sensor_Node():

    def __init__(self, hostname, port, baudrate, meas_interval, address, file_path):
        self.mu = Cybres_MU(port, baudrate)
        self.pub = ZMQ_Publisher(address)
        self.hostname = hostname
        self.measurment_interval = meas_interval
        self.file_path = file_path
        self.csv_object = None
        self.msg_count = 0
        self.start_time = None
        self.mu_id = 0
        self.mu_mm = 0

        # Add the names of the additional data columns to the list
        # e.g. ['ozon-conc', 'intensity-red', 'intensity-blue']
        # self.rgb = RGB_TCS34725()
        self.additionalSensors = []#['light-red', 'light-green', 'light-blue', 'color-temperature', 'light-intensity']

    def start(self):
        """
        Start the measurements. Continue to publish over MQTT and store to csv.
        """

        # Measure at set interval.
        self.mu.set_measurement_interval(self.measurment_interval)
        self.mu.start_measurement()

        # Record the starting time and notify the user.
        self.start_time = datetime.datetime.now()
        logging.info("Measurement started at %s.", self.start_time.strftime("%d.%m.%Y. %H:%M:%S"))
        logging.info("Saving data to: %s", self.file_path)

        # Create the file for storing measurement data.
        file_name = f"{self.hostname}_{self.start_time.strftime('%Y_%m_%d-%H_%M_%S')}.csv"
        self.csv_object = data2csv(self.file_path, file_name, self.additionalSensors)
        last_time = datetime.datetime.now()

        while True:
            # Create a new csv file after the specified interval.
            current_time = datetime.datetime.now()
            if current_time.hour in {0, 12} and current_time.hour != last_time.hour:
                logging.info("Creating a new csv file.")
                self.csv_object.close_file()
                file_name = f"{self.hostname}_{current_time.strftime('%Y_%m_%d-%H_%M_%S')}.csv"
                self.csv_object = data2csv(self.file_path, file_name, self.additionalSensors)
                last_time = current_time

            # Get the next data set
            next_line = self.mu.get_next()
            header, payload = self.classify_message(next_line)

            # Check for invalid data
            if header != None:
                self.pub.publish(header, self.additionalSensors, payload)

            # Store the data to the csv file.
            if header[1] == 1:
                self.msg_count += 1
                e = self.csv_object.write2csv([self.hostname] + payload.tolist())
                if e is not None:
                    logging.error("Writing to csv file failed with error:\n%s\n\n\
                        Continuing because this is not a fatal error.", e)

            # Print out a status message roughly every 30 mins
            if self.msg_count % 180 == 0 and self.msg_count > 0:
                td = datetime.datetime.now() - self.start_time
                hms = (td.seconds // 3600, td.seconds // 60 % 60, td.seconds % 60)
                duration = f"{td.days} days, {hms[0] :02}:{hms[1] :02}:{hms[2] :02} [HH:MM:SS]"
                logging.info("I am measuring for %s and I collected %d datapoints.", duration, self.msg_count)


    def classify_message(self, mu_line):
        """
        Determines the message type.

        Args:
            mu_line (str): Complete MU data line

        Returns:
            A tuple containing a header and payload for the MQTT message.
        """
        counter = mu_line.count('#')
        if counter == 0:
            # Line is pure data message
            messagetype = 1
            transfromed_data = self.transform_data(mu_line)
            # ID and MM are manually added
            payload = np.append([self.mu_mm, self.mu_id], transfromed_data)

        elif counter == 2:
            # Line is data message/id/measurement mode
            # Every 100 measurements the MU sends also its own
            # ID and measurement mode
            messagetype = 2
            messages = mu_line.split('#')
            mu_id = int(messages[1].split(' ')[1])
            mu_mm = int(messages[2].split(' ')[1])
            # ID and mm get attached at the back of the data array
            payload = np.append([mu_mm, mu_id], self.transform_data(messages[0]))

        elif counter == 4:
            # Line is header
            messagetype = 0
            payload = mu_line
            # ID and MM are saved from the header
            lines = mu_line.split('\r\n')
            self.mu_id = int(lines[3].split()[1])
            self.mu_mm = int(lines[4].split()[1])
        else:
            logging.warning("Unknown data type: \n%s", mu_line)
            return None, None

        # Add data from additional external sensors
        if self.additionalSensors and messagetype in {1, 2}:
            # Call here a get_data() method, e.g.:
            # additionalValues = [getOzonValues(), getRGBValues()]
            # Important: len(self.sensors) == len(additionalValues), otherwise
            # it won't work
            additionalValues = []#self.rgb.getData()#[]
            payload = np.append(payload, additionalValues)


        header = (self.hostname, messagetype, bool(self.additionalSensors))
        return header, payload


    def transform_data(self, string_data):
        """
        Transform MU data from string to numpy array.

        Args:
            string_data (str): MU data in string format.

        Returns:
            A numpy array containing the MU data
        """
        split_data = string_data.split(' ')
        timestamp = [int(time.mktime(datetime.datetime.now().timetuple()))]
        measurements = [int(elem) for elem in split_data[1:]]
        return np.array(timestamp + measurements)


    def stop(self):
        """
        Stop the measurement and clean up.
        """
        logging.info("Measurement stopped at %s.", datetime.datetime.now().strftime("%d.%m.%Y. %H:%M:%S"))
        _ = self.mu.return_serial()
        self.mu.stop_measurement()
        if self.csv_object is not None:
            self.csv_object.close_file()


    def shutdown(self):
        """
        Perform final clean up on shutdown.
        """
        self.mu.restart()
        self.mu.ser.close()
        self.pub.socket.close()
        self.pub.context.term()
