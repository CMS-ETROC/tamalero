from tamalero.ReadoutBoard import ReadoutBoard
from tamalero.utils import header, make_version_header, get_kcu, check_repo_status

import time
import random
import sys

if __name__ == '__main__':
    import argparse
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--verbose', action='store_true', default=False, help="Verbose power up sequence")
    argParser.add_argument('--power_up', action='store_true', default=False, help="Do lpGBT power up init?")
    argParser.add_argument('--reconfigure', action='store_true', default=False, help="Configure the RB electronics: SCA and lpGBT?")
    argParser.add_argument('--adcs', action='store_true', default=False, help="Read ADCs?")
    argParser.add_argument('--i2c_temp', action='store_true', default=False, help="Do temp monitoring on I2C from lpGBT?")
    argParser.add_argument('--i2c_sca', action='store_true', default=False, help="I2C tests on SCA?")
    argParser.add_argument('--kcu', action='store', default="192.168.0.10", help="Specify the IP address for KCU")
    argParser.add_argument('--control_hub', action='store_true', default=False, help="Use control hub for communication?")
    argParser.add_argument('--host', action='store', default='localhost', help="Specify host for control hub")
    argParser.add_argument('--configuration', action='store', default='modulev0', choices=['default', 'emulator', 'modulev0'], help="Specify a configuration of the RB, e.g. emulator or modulev0")
    argParser.add_argument('--devel', action='store_true', default=False, help="Don't check repo status (not recommended)")
    argParser.add_argument('--connection_test', action='store_true', default=False, help="Check the PCB connections.")
    args = argParser.parse_args()

    verbose = args.verbose

    #-------------------------------------------------------------------------------
    # Try to Connect to the KCU105
    #-------------------------------------------------------------------------------
    print ("Using KCU at address: %s"%args.kcu)

    kcu = None
    rb_0 = None

    # write to the loopback node of the KCU105 to check ethernet communication
    kcu = get_kcu(args.kcu, control_hub=args.control_hub, host=args.host, verbose=args.verbose)
    if (kcu == 0):
        # if not basic connection was established the get_kcu function returns 0
        # this would cause the RB init to fail.
        sys.exit(1)

    rb_0 = ReadoutBoard(0, kcu=kcu, config=args.configuration)
    data = 0xabcd1234
    kcu.write_node("LOOPBACK.LOOPBACK", data)
    if (data != kcu.read_node("LOOPBACK.LOOPBACK")):
        print("No communications with KCU105... quitting")
        sys.exit(1)

    is_configured = rb_0.DAQ_LPGBT.is_configured()
    if not is_configured:
        print("RB is not configured, exiting.")
        exit(0)
    header(configured=is_configured)

    if not args.devel:
        check_repo_status(kcu_version=kcu.get_firmware_version(verbose=True))

    rb_0.VTRX.get_version()

    if not hasattr(rb_0, "TRIG_LPGBT"):
        rb_0.get_trigger()

    res = rb_0.DAQ_LPGBT.get_board_id()
    res['trigger'] = 'yes' if rb_0.trigger else 'no'

    if (verbose):
        make_version_header(res)

    if args.adcs:
        print("\n\nReading GBT-SCA ADC values:")
        rb_0.SCA.read_adcs(check=True)

        print("\n\nReading DAQ lpGBT ADC values:")
        rb_0.DAQ_LPGBT.read_adcs(check=True)

        # High level reading of temperatures
        temp = rb_0.read_temp(verbose=True)

    #-------------------------------------------------------------------------------
    # Read SCA
    #-------------------------------------------------------------------------------
    if args.i2c_sca:
        print("Writing and Reading I2C_ctrl register:")
        for n in range(10):
            wr = random.randint(0, 100)
            rb_0.SCA.I2C_write_ctrl(channel=3, data=wr)
            rd = rb_0.SCA.I2C_read_ctrl(channel=3)
            print("write: {} \t read: {}".format(wr, rd))

        print("Testing multi-byte read:")
        multi_out = rb_0.SCA.I2C_read_multi(channel=3, servant = 0x48, nbytes=2)
        print("servant: 0x48, channel: 3, nbytes: 2, output = {}".format(multi_out))

        print("Testing multi-byte write:")

        write_value = [0x2, 25, (27&240)]
        print("servant: 0x48, channel: 3, nbytes: 2, data:{}".format(write_value))
        rb_0.SCA.I2C_write_multi(write_value, channel=3, servant=0x48)
        read_value = rb_0.SCA.I2C_read_multi(channel=3, servant=0x48, nbytes = 2, reg=0x2)

        if read_value == write_value[1:]:
            print ("write/read successful!")
        print("read value = {}".format(rb_0.SCA.I2C_read_multi(channel=3, servant=0x48, nbytes = 2, reg=0x2)))

    #-------------------------------------------------------------------------------
    # Success LEDs
    #-------------------------------------------------------------------------------
    rb_0.DAQ_LPGBT.set_gpio("LED_1", 1) # Set LED1 after tamalero finishes succesfully
    rb_0.DAQ_LPGBT.set_gpio("LED_RHETT", 1) # Set LED1 after tamalero finishes succesfully
    if rb_0.DAQ_LPGBT.ver == 1 and args.connection_test:
        print("Toggling FEAST and 2.5V on/off for 1min")
        t_end = time.time() + 60
        while time.time() < t_end:
            rb_0.DAQ_LPGBT.set_gpio("LED_1", 1) # Let Rhett LED blink for 10s
            rb_0.DAQ_LPGBT.set_gpio("LED_RHETT", 1) # Let Rhett LED blink for 10s
            rb_0.SCA.set_gpio("mod_d00", 1)
            rb_0.SCA.set_gpio("mod_d01", 1)
            rb_0.SCA.set_gpio("mod_d08", 1)
            rb_0.SCA.set_gpio("mod_d09", 1)
            rb_0.SCA.set_gpio("mod_d16", 1)
            rb_0.SCA.set_gpio("mod_d17", 1)
            time.sleep(2.0)
            rb_0.DAQ_LPGBT.set_gpio("LED_1", 0)
            rb_0.DAQ_LPGBT.set_gpio("LED_RHETT", 0)
            rb_0.SCA.set_gpio("mod_d00", 0)
            rb_0.SCA.set_gpio("mod_d01", 0)
            rb_0.SCA.set_gpio("mod_d08", 0)
            rb_0.SCA.set_gpio("mod_d09", 0)
            rb_0.SCA.set_gpio("mod_d16", 0)
            rb_0.SCA.set_gpio("mod_d17", 0)
            time.sleep(2.0)
        rb_0.DAQ_LPGBT.set_gpio("LED_1", 1)
        rb_0.DAQ_LPGBT.set_gpio("LED_RHETT", 1)

        # disabling 2.5V
        rb_0.SCA.set_gpio("mod_d00", 1)
        rb_0.SCA.set_gpio("mod_d08", 1)
        rb_0.SCA.set_gpio("mod_d16", 1)

        # enabling FEAST
        rb_0.SCA.set_gpio("mod_d01", 1)
        rb_0.SCA.set_gpio("mod_d09", 1)
        rb_0.SCA.set_gpio("mod_d17", 1)
