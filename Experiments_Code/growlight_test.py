from kasa import SmartPlug
import time

def main():

    #WP03 = "134.34.225.167"  # 70-4F-57-FF-AE-F5nano
    WP00 = "134.34.225.132"  # D8-0D-17-5c-FC-93
    plug = WP00
    #plug = WP00
    growLight = SmartPlug(plug)

    try:


        # Toggle the grow light on and off continuously
        while True:
            growLight.turn_on()
            print('ON')
            print('')
            #growLight.turn_off()
            #print(growLight.state)

            #if growLight.state == 'ON':
                #print('On')
                #time.sleep(3)  # Wait for 0.5 seconds
            #if growLight.state == 'off':
                #print('Off')
                #time.sleep(3)  # Wait for 0.5 seconds # Wait for 0.5 seconds

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()