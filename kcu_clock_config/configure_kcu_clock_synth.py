import os
import sys
import serial
import subprocess 

def open_kcu_serial_port(): 

    try: 
        serial.__version__
    except: 
        print("You probably need to install pyserial (pip install pyserial)")
        sys.exit(1) 

    ports = subprocess.Popen(['bash','find_serial_ports.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode().split("\n")[:-1]
    print('\nAvaliable USB ports:')
    print(ports)

    print('\nWelcome to KCU105 clock configuration')
    print('\n')
    usb = input('Which USB port is the board connected to? Type a number: ')
    try: 
        portName = '/dev/ttyUSB'+ str(usb)
        ser = serial.Serial()
        ser.baudrate = 115200
        ser.port = portName
        ser.timeout = 1
        ser.open()
        return ser
    except:
        print('\nERROR: unavaliable USB Port')
        sys.exit(1)

def pprint(line): 
    for l in line: 
        print(l)


def configure_kcu_clocks():

    def dump(line, msg=""):
        print("Unexpected read, looking for \"" + str(msg) + "\" but got:")
        pprint(line)
        return_to_main()
        sys.exit(1)

    def check_and_write(msg, expect, action=None):
        line = ser.readlines()
        if expect not in line:
            dump(line, expect)
        else:
            print(msg)
            if action is not None:
                ser.write(action)

    def return_to_main():
        """get back to the main menu from a nested menu"""

        ser.write(b'\n')
        line = ser.readlines()

        while b'\r0. Return to Main Menu\r\n' in line or b'\r0. Return to Clock Menu\r\n' in line:
            print("Returning to main menu")
            ser.write(b'\n0\n')
            line = ser.readlines()

        ser.write(b'\n')

    ser = open_kcu_serial_port()
    
    print("Configuring KCU105 at: " + str(ser.name))

    if ser.is_open:
        check = input('\nAbout to setup KCU105 Si570, is this okay?  [y/n]: ')
    else:
        check=""

    if (check != 'y'):
        print('\nDid not set up KCU105 Si570\n')
        sys.exit(0)

    return_to_main()

    # transition from main menu to the clock menu
    check_and_write("Main Menu -> Clock Menu",
                    b'\r      - Main Menu -\r\n',
                    b'\n1\n')

    check_and_write("Clock Menu -> Si750 User Clock Frequency Menu (1)",
                    b'\r      - Clock Menu -\r\n',
                    b'\n1\n')

    check_and_write("Si750 User Clock Frequency Menu: Enter Clock Freq (320.64)",
                    b'Enter the Si570 frequency (10-810MHz):\r\n',
                    b'\n320.64\n')

    check_and_write("Clock Menu -> Save Clocks to EEPROMs (3)",
                    b'\r3. Save    KCU105 Clock Frequency  to  EEPROM\r\n',
                    b'\n3\n')

    check_and_write("Save Clock frequency to EEPROM (1)",
                    b'\r        - Save Menu -\r\n',
                    b'\n1\n')

    check_and_write("Return to Clock Menu (0)",
                    b'\r        - Save Menu -\r\n',
                    b'\n0\n')

    check_and_write("View KCU105 Saved Clocks (5)",
                    b'\r      - Clock Menu -\r\n',
                    b'\n5\n')

    check_and_write("Checking Clock frequency",
                    b'\rSi570  User Clock:  320.64000000 MHz\r\n',
                    b'\n')

    check_and_write("Setting up automatic restore (6)",
                    b'\r      - Clock Menu -\r\n',
                    b'\n6\n')

    check_and_write("Enabling automatic restore (2)",
                    b'\r      - Options Menu -\r\n',
                    b'\n2\n')

    return_to_main()

    print('End of Program\n')
    ser.close()


if __name__ == "__main__": 
    configure_kcu_clocks()

