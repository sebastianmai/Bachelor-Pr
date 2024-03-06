from serial_class import PhytNode_Serial
import threading
from pyHS100 import SmartPlug
from datetime import datetime, timedelta
import pandas as pd
import os
import time

def main():
    PN_1 = PhytNode_Serial("/dev/ttyACM1", 1)
    PN_2 = PhytNode_Serial("/dev/ttyACM2", 2)
    PN_controll = PhytNode_Serial("/dev/ttyACM3", 9)
    thread_PN_1 = threading.Thread(target=PN_1.write2csv_thread)
    thread_PN_2 = threading.Thread(target=PN_2.write2csv_thread)
    thread_P_controll = threading.Thread(target=PN_controll.write2csv_thread)

    thread_PN_1.start()
    thread_PN_2.start()
    thread_P_controll.start()

    #thread_PN_1.join()
    #thread_PN_2.join()
    #thread_P_controll.join()

if __name__ == '__main__':
    main()
