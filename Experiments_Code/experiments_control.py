from serial_class import PhytNode_Serial
import threading
from pyHS100 import SmartPlug
from datetime import datetime, timedelta


def main():
    PN_1 = PhytNode_Serial("/dev/ttyACM1", 1)
    PN_2 = PhytNode_Serial("/dev/ttyACM2", 2)
    #PN_3 = PhytNode_Serial("/dev/ttyACM2",3)
    #PN_controll = PhytNode_Serial("/dev/ttyACM3",9)
    thread_PN_1 = threading.Thread(target=PN_1.write2csv_thread, daemon=True)
    thread_PN_2 = threading.Thread(target=PN_2.write2csv_thread, daemon=True)
    #thread_PN_3 = threading.Thread(target=PN_3.write2csv_thread, daemon=True)
    #thread_P_controll = threading.Thread(target=PN_controll.write2csv_thread, daemon=True)

    thread_PN_1.start()
    thread_PN_2.start()
    # thread_PN_3.start()
    #thread_P_controll.start()
    # thread_PN_1.join()
    # thread_PN_2.join()
    # thread_PN_3.join()
    # thread_P_controll.join()

    WP03 = "134.34.225.167"  # 70-4F-57-FF-AE-F5
    plug = WP03
    growLight = SmartPlug(plug)
    growLight.turn_off()

    wait_time = datetime.now()

    while (True):
        # PN_1.write2csv()
        # PN_2.write2csv()
        # PN_3.write2csv()
        # PN_controll.write2csv()
        # print(f"\r Wrong data counter: PN_1: {PN_1.wrong_data_counter} Wrong data counter: PN_2: {PN_2.wrong_data_counter} "
        #      f"Wrong data counter: Control: {PN_controll.wrong_data_counter}", end="\r")

        current_time = datetime.now()
        #print(current_time, wait_time + timedelta(minutes=1), wait_time + timedelta(minutes=2), wait_time + timedelta(minutes=3), wait_time + timedelta(minutes=))


        if wait_time + timedelta(minutes=10) < current_time <= wait_time + timedelta(minutes=20):
            print("turnedon")
            growLight.turn_on()
        elif wait_time + timedelta(minutes=20) < current_time <= (wait_time + timedelta(minutes=30)):
            print("turnedoff")
            growLight.turn_off()
        elif current_time > (wait_time + timedelta(minutes=30)):
            break


if __name__ == '__main__':
    main()
    exit(0)
