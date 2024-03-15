import asyncio

from serial_class import PhytNode_Serial
import threading
from kasa import SmartPlug
from datetime import datetime, timedelta


async def main():
    #WP03 = "134.34.225.167"  # 70-4F-57-FF-AE-F5
    WP00 = "134.34.225.132"  # D8-0D-17-5c-FC-93
    plug = WP00
    # plug = WP03
    growLight = SmartPlug(plug)
    await growLight.turn_off()

    wait_time = datetime.now()

    while (True):
        current_time = datetime.now()

        if wait_time + timedelta(minutes=5) < current_time <= wait_time + timedelta(minutes=15):
            await growLight.turn_on()
        elif wait_time + timedelta(minutes=15) < current_time < (wait_time + timedelta(minutes=20)):
            await growLight.turn_off()

        if current_time >= (wait_time + timedelta(minutes=20)):
            print("break")
            break

if __name__ == '__main__':
    asyncio.run(main())
