from serial_class import PhytNode_Serial
import threading
from pyHS100 import SmartPlug
from datetime import datetime, timedelta
import pandas as pd
import os
import time

scaling_factors = {
    "temp-external": lambda t: t / 10000,
    "humidity-external": lambda t: t / 1,
    # actual function should be ((x * 3 / 4200000-0.1515) / (0.006707256-0.0000137376 * (temp-external / 10000.0)))
    "light-external": lambda t: (t / 799.4) - 0.75056,
    "differential_potential_CH1": lambda t: (t - 512000) / 1000,
    "differential_potential_CH2": lambda t: (t - 512000) / 1000,
    "transpiration": lambda t: t / 1000,
    "soil_moisture": lambda t: t,
    "soil_temperature": lambda t: t / 10
}

def main():
    WP03 = "134.34.225.167"  # 70-4F-57-FF-AE-F5
    plug = WP03
    growLight = SmartPlug(plug)
    growLight.turn_off()

    wait_time = datetime.now()
    start_temp = get_temp(0)

    while (True):
        current_time = datetime.now()

        if wait_time + timedelta(minutes=5) < current_time <= wait_time + timedelta(minutes=15):
            current_temp = get_temp(-1)
            print(current_temp, start_temp, start_temp + 5)

            if current_temp <= start_temp + 5:
                growLight.turn_on()
                time.sleep(1)
            elif current_temp > start_temp + 5:
                growLight.turn_off()
                time.sleep(1)
        elif wait_time + timedelta(minutes=15) < current_time <= (wait_time + timedelta(minutes=20)):
            growLight.turn_off()
        elif current_time > (wait_time + timedelta(minutes=20)):
            break


def get_temp(position):
    folder_path = '/home/pi/measurements/'
    files = os.listdir(folder_path)
    full_name = [os.path.join(folder_path, file) for file in files]
    sorted_files = sorted(full_name, key=os.path.getmtime, reverse=True)
    data_cybres = pd.read_csv(sorted_files[0])

    data_cybres["temp-external"] = scaling_factors["temp-external"](data_cybres["temp-external"])
    last_temp = data_cybres["temp-external"].iloc[position]

    return last_temp

if __name__ == '__main__':
    main()
