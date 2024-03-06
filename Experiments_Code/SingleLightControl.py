from constants import *
from pyHS100 import Discover, SmartPlug
from datetime import datetime

WP00 = "134.34.225.132"  # D8-0D-17-5c-FC-93
# WP01 = "134.34.225.235" #D8-0D-17-5c-FC-7D
# WP02 = "134.34.225.133" #D8-0D-17-5c-FC-51
WP03 = "134.34.225.167"  # 70-4F-57-FF-AE-F5
# WP04 = "134.34.225.135" #3C-84-6A-50-E3-E3


plug = WP03
growLight = SmartPlug(plug)
growLight.turn_off()

on = 0
print("light control is running")

while (True):
    current_time = datetime.now()
    print(current_time.hour, current_time.minute)

    if 8 <= current_time.hour < 22:

        specific_times = [
            (9, 0),
            (11, 10),
            (13, 20),
            (15, 30),
            (17, 40),
            (19, 50)
        ]

        time_match = False

        for hour, minute in specific_times:
            if current_time.hour == hour and minute <= current_time.minute < minute + 10:
                print("turnedon")
                growLight.turn_on()
                on = 1
                time_match = True
                break

        if not time_match and on:
                print("turned off")
                growLight.turn_off()
                on = 0

    else:
        print("turned off")
        growLight.turn_off()
        on = 0
