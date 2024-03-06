#!/usr/bin/env python3
import sys, os
import re
import socket
import logging
import argparse
from sensor_node import Sensor_Node

sys.path.append("../../..") # Adds higher directory to python modules path.
from Utilities.log_formatter import ColoredFormatter, setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for the sensor node.")
    parser.add_argument('--port', action='store', default='/dev/ttyACM0',
        help="Port where the measurement unit is connected.")
    parser.add_argument('--baud', action='store', type=int, default=460800,
        help="Baudrate for communicating with the measurement unit.")
    parser.add_argument('--int', action='store', type=int, default=10000,
        help="Time interval between two measurements (in miliseconds).")
    parser.add_argument('--addr', action='store', default='localhost',
        help='Address of the MQTT subscriber. Can be IP, localhost, *.local, etc.')
    parser.add_argument('--dir', action='store', default='/home/' + os.getenv('USER') + '/measurements/' )
    args = parser.parse_args()
    
    if args.addr == 'localhost':
        hostname = args.port.split('/')[-1]
    else:
        hostname = socket.gethostname()

    setup_logger(hostname)
    logging.info('Starting sensor node.')

    csv_dir = args.dir
    if csv_dir[-1] != '/':
        csv_dir += '/'

    SN = Sensor_Node(hostname, args.port, args.baud, args.int, args.addr, csv_dir)
    while True:
        try:
            SN.check()
            SN.start()
        except KeyboardInterrupt:
            logging.info("Interrupted!")
            SN.stop()

            next_command = input('\nEnter a command:\n\tnew --> start new measurement\n\texit --> exit from the script\n> ')
            if next_command != 'new':
                if next_command != 'exit':
                    print("Unknow command. Exiting.")
                SN.shutdown()
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)                    
            
