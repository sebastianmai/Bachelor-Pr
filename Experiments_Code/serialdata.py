from serial_class import PhytNode_Serial
import threading


def main():
    """
    Creates the serial connection and writes the data to a csv file
    """
    PN3 = PhytNode_Serial("/dev/ttyACM0",3)
    PN1 = PhytNode_Serial("/dev/ttyACM1",1)
    PN_controll = PhytNode_Serial("/dev/ttyACM2",9)
    thread_P3 = threading.Thread(target=PN3.write2csv_thread)
    thread_P1 = threading.Thread(target=PN1.write2csv_thread)
    thread_P_controll = threading.Thread(target=PN_controll.write2csv_thread)

    thread_P3.start()
    thread_P1.start()
    thread_P_controll.start()
    #thread_P3.join()
    #thread_P1.join()
    #thread_P_controll.join()


    while(True):
        #PN3.write2csv()
        ##PN1.write2csv()
        #PN_controll.write2csv()
        print(f"\r Wrong data counter: PN3: {PN3.wrong_data_counter}, PN1: {PN1.wrong_data_counter}, PN_controll: {PN_controll.wrong_data_counter}", end="\r")






if __name__ == '__main__':
    main()
