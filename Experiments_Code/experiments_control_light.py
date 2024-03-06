from serial_class import PhytNode_Serial
import threading
from pyHS100 import SmartPlug
from datetime import datetime, timedelta
import pandas as pd
import os
import time

def main():
    WP03 = "134.34.225.167"  # 70-4F-57-FF-AE-F5
    plug = WP03
    growLight = SmartPlug(plug)
    growLight.turn_off()

    wait_time = datetime.now()

    while (True):
        current_time = datetime.now()

        if wait_time + timedelta(minutes=5) < current_time <= wait_time + timedelta(minutes=15):
            growLight.turn_on()
        elif wait_time + timedelta(minutes=15) < current_time < (wait_time + timedelta(minutes=20)):
            growLight.turn_off()

        if current_time >= (wait_time + timedelta(minutes=20)):
            print("break")
            break

if __name__ == '__main__':
    main()
