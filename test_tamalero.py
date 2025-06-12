from tamalero.ReadoutBoard import ReadoutBoard
from tamalero.utils import header, make_version_header, get_kcu, check_repo_status
from tamalero.FIFO import FIFO
from tamalero.Module import Module

from test_ETROC import vth_scan, fromPixNum

import time
import random
import sys
import os
import json
import datetime
import numpy as np
from emoji import emojize
from flask import Flask, request

summary_tests = {}
def add_test_to_summary(test_name: str, module: int, etroc: int, test_results) -> dict:
    # Initialize summary_tests if necessary
    summary_tests.setdefault(module, {}).setdefault(etroc, {})
    # Add or update the test results
    summary_tests[module][etroc][test_name] = test_results

def create_app(rb, modules=[]):
    # FIXME this should live somewhere else in the future
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    @app.route('/summary_tests')
    def get_summary_tests():
        global summary_tests #YIKES but I think this is the only way haha
        posted_summary_tests = summary_tests
        summary_tests = {} #empty it after this is hit
        return posted_summary_tests

    @app.route('/rb_temp')
    def temperatures():
        temp = rb.read_temp()
        temp['time'] = datetime.datetime.now().isoformat()
        return temp

    @app.route('/rb_mux64/<ch>')
    def get_mux64_output(ch):
        volts = {}
        # mux64 should have 64 entries anyway but could be nice check configuration maybe? 
        for i in [ch]:
            volts[str(ch)] = rb.read_mux_test_board(int(ch))
        volts['time'] = datetime.datetime.now().isoformat()
        return volts

    @app.route('/module_links')
    def get_link_status():
        link_status = {}
        for i, m in enumerate(modules):
            any_connected = False
            etrocs = {}
            for j, etroc in enumerate(m.ETROCs):
                links = {}
                elinks = etroc.elinks
                status = etroc.get_elink_status()
                ilink = 0
                any_connected |= etroc.is_connected() > 0
                for lpgbt in elinks:
                    for k, link in enumerate(elinks[lpgbt]):
                        links[ilink] = {'lpGBT': lpgbt, 'elink': elinks[lpgbt][k], 'locked': status[lpgbt][k]}
                        ilink += 1
                etrocs[str(j)] = links

            link_status[i] = etrocs
            link_status[i]['connected'] = any_connected
        return link_status

    @app.route('/etroc_status')
    def get_etroc_status():
        etroc_status = {}
        for i, module in enumerate(modules):
            etroc_status[i] = {}
            for j, etroc in enumerate(module.ETROCs):  # 4 ETROCs expected per module
                if etroc.is_connected():
                    stat = etroc.pixel_sanity_check(return_matrix=True)
                    etroc_status[i][j] = stat.astype(int).tolist()
                    #add it to the summmary of tests dictionary
                    add_test_to_summary('etroc_status',i,j,etroc_status[i][j])
        return etroc_status

    @app.route('/threshold_scan', methods=['POST'])
    def run_threshold_scan():
        payload = json.loads(request.data)
        print(f"Threshold Scan on module {payload['module']}, etroc {payload['etroc']}")

        module = modules[int(payload['module'])]
        etroc = module.ETROCs[int(payload['etroc'])]
        fifo = FIFO(rb=rb)

        vth_scan_data = vth_scan(
            etroc,
            vth_min = 0,
            vth_max = 200,
            decimal = True,
            fifo = fifo,
            absolute = True,
        )

        vth_axis    = np.array(vth_scan_data[0])
        hit_rate    = np.array(vth_scan_data[1])
        N_pix       = len(hit_rate) # total # of pixels
        N_pix_w     = int(round(np.sqrt(N_pix))) # N_pix in NxN layout
        max_indices = np.argmax(hit_rate, axis=1)
        maximums    = vth_axis[max_indices]
        max_matrix  = np.empty([N_pix_w, N_pix_w])

        for pix in range(N_pix):
            r, c = fromPixNum(pix, N_pix_w)
            max_matrix[r][c] = maximums[pix]

        threshold_status  = {}
        threshold_status['vth_axis'] = vth_axis.tolist()
        threshold_status['hit_rate'] = hit_rate.tolist()
        threshold_status['max_matrix'] = max_matrix.tolist()
        return threshold_status
    return app

if __name__ == '__main__':
    import argparse
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--verbose', action='store_true', default=False, help="Verbose power up sequence")
    argParser.add_argument('--power_up', action='store_true', default=False, help="Do lpGBT power up init?")
    argParser.add_argument('--reconfigure', action='store_true', default=False, help="Configure the RB electronics: SCA and lpGBT?")
    argParser.add_argument('--adcs', action='store_true', default=False, help="Read ADCs?")
    argParser.add_argument('--i2c_temp', action='store_true', default=False, help="Do temp monitoring on I2C from lpGBT?")
    argParser.add_argument('--i2c_sca', action='store_true', default=False, help="I2C tests on SCA?")
    argParser.add_argument('--run_pattern_checker', action='store_true', default=False, help="Read pattern checker?")
    argParser.add_argument('--reset_pattern_checker', action='store', choices=[None, 'prbs', 'upcnt'], default=None, help="Reset pattern checker?")
    argParser.add_argument('--kcu', action='store', default="192.168.0.10", help="Specify the IP address for KCU")
    argParser.add_argument('--force_no_trigger', action='store_true', help="Never initialize the trigger lpGBT.")
    argParser.add_argument('--allow_bad_links', action='store_true', help="Select to allow bad link initialization.")
    argParser.add_argument('--read_fifo', action='store', default=-1, help='Read 3000 words from link N')
    argParser.add_argument('--alignment', action='store', nargs='?', default=False, const=True, help='Load/scan alignment? If load, pass in file path')
    argParser.add_argument('--etroc', action='store', default="ETROC2", help='Specify ETROC version.')
    argParser.add_argument('--eyescan', action='store_true', default=False, help="Run eyescan?")
    argParser.add_argument('--recal_lpgbt', action='store_true', default=False, help="Recalibrate ADC in LPGBT? (instead of using saved values)")
    argParser.add_argument('--host', action='store', default='localhost', help="Specify host for control hub")
    argParser.add_argument('--configuration', action='store', default='default', choices=['default', 'emulator', 'modulev0', 'modulev0b', 'multimodule', 'mux64', 'modulev1'], help="Specify a configuration of the RB, e.g. emulator or modulev0")
    argParser.add_argument('--devel', action='store_true', default=False, help="Don't check repo status (not recommended)")
    argParser.add_argument('--monitor', action='store_true', default=False, help="Start up montoring threads in the background")
    argParser.add_argument('--strict', action='store_true', default=False, help="Enforce strict limits on ADC reads for SCA and LPGBT")
    argParser.add_argument('--server', action='store_true', default=False, help="Start server")
    argParser.add_argument('--port', action='store', default=5000, type=int, help="Port to use for server")
    argParser.add_argument('--rb', action='store', default=0, type=int, help="Specify Readout Board")
    argParser.add_argument('--rbver', action='store', type=int, help="Specify Readout Board version")
    argParser.add_argument('--multi_board', action = 'store_true')
    argParser.add_argument('--power_board', action='store_true', help="Enable power board usage, and show status.")
    args = argParser.parse_args()

    verbose = args.verbose
    data_mode = args.etroc in ['ETROC1', 'ETROC2']

    #-------------------------------------------------------------------------------
    # Try to Connect to the KCU105
    #-------------------------------------------------------------------------------
    print ("Using KCU at address: %s"%args.kcu)

    kcu = None
    rb = None

    if args.multi_board:
        temp = get_kcu(args.kcu, control_hub=True, host=args.host, verbose=args.verbose)
        if temp == 0:
            sys.exit(1)
        for i in range(3):
            try:
                print(f'Checking ReadoutBoard {i}')
                rb = ReadoutBoard(i, trigger=(not args.force_no_trigger), kcu=temp, config=args.configuration)
                print(f'Checking data loopback')
                temp.write_node("LOOPBACK.LOOPBACK", data)
                if (data != temp.read_node("LOOPBACK.LOOPBACK")):
                    print(f"No communications with KCU105 for board {i}")
                else:
                    print(f'Successfully connected to ReadoutBoard {i}')
                    rb.get_trigger()
                    rb.read_temp()
            except:
                print(f'Connecting to ReadoutBoard {i} failed')
        temp.status()

    if args.alignment:
        if isinstance(args.alignment, str):
            print ("Loading uplink alignemnt from file:", args.alignment)
            from tamalero.utils import load_alignment_from_file
            alignment = load_alignment_from_file(args.alignment)
        else:
            alignment = None
    else:
        alignment = False

    # write to the loopback node of the KCU105 to check ethernet communication
    kcu = get_kcu(args.kcu, control_hub=True, host=args.host, verbose=args.verbose)
    if (kcu == 0):
        # if not basic connection was established the get_kcu function returns 0
        # this would cause the RB init to fail.
        sys.exit(1)

    # check that the KCU is actually connected
    data = 0xabcd1234
    kcu.write_node("LOOPBACK.LOOPBACK", data)
    if (data != kcu.read_node("LOOPBACK.LOOPBACK")):
        print("No communications with KCU105... quitting")
        sys.exit(1)

    print(f'Utilizing ReadoutBoard {args.rb}')
    rb = ReadoutBoard(
        args.rb,
        trigger=(not args.force_no_trigger),
        kcu=kcu,
        config=args.configuration,
        alignment=alignment,
        data_mode=data_mode,
        etroc=args.etroc,
        verbose=args.verbose,
        allow_bad_links = args.allow_bad_links,
        #ver=args.rbver,
    )
    
    # IDEA Loop over boards for configuration?
    print(kcu.readout_boards)

    #is_configured = rb.DAQ_LPGBT.is_configured()
    header(configured=rb.configured)

    if args.recal_lpgbt:
        rb.DAQ_LPGBT.calibrate_adc(recalibrate=True)
        if rb.trigger:
            rb.TRIG_LPGBT.calibrate_adc(recalibrate=True)

    if not args.devel:
        check_repo_status(kcu_version=kcu.get_firmware_version(verbose=True))

    #-------------------------------------------------------------------------------
    # Power up
    #-------------------------------------------------------------------------------
    if args.power_up:

        print("Power up init sequence for: DAQ")

        #rb.DAQ_LPGBT.power_up_init()

        rb.VTRX.get_version()
        if (verbose):
            print ("VTRX status at power up:")
            _ = rb.VTRX.status()
            print (rb.VTRX.ver)

        rb.get_trigger()

        if rb.trigger:
            #print("Configuring Trigger lpGBT")
            print (" > Power up init sequence for: Trigger")
            rb.TRIG_LPGBT.power_up_init()
            print("Done Configuring Trigger lpGBT")
    else:
        rb.VTRX.get_version()

    if not hasattr(rb, "TRIG_LPGBT"):
        rb.get_trigger()

    if args.power_up or args.reconfigure:
        print("Configuring readout board")
        rb.configure()

    res = rb.DAQ_LPGBT.get_board_id()
    res['trigger'] = 'yes' if rb.trigger else 'no'

    if (verbose):
        make_version_header(res)

    if args.power_up or args.reconfigure:
        # FIXME this is taken out because it sometimes sends the RB into the Nirvana.
        # Daniel will fix it when he has time.
        #rb.reset_problematic_links(
        #    max_retries=10,
        #    allow_bad_links=args.allow_bad_links)
        if verbose:
            rb.status()

    if (verbose):
        kcu.status()

    rb.VTRX.get_version()
    if (verbose):
        _ = rb.VTRX.status()

    if args.power_up or args.reconfigure:
        print("Link inversions")
        rb.DAQ_LPGBT.invert_links()
        if rb.trigger:
            rb.TRIG_LPGBT.invert_links()

    #-------------------------------------------------------------------------------
    # Module Status
    #-------------------------------------------------------------------------------
    rb.enable_etroc_readout()
    rb.enable_etroc_readout(slave=True)

    modules = []
    pb_channel = []
    if args.configuration == 'emulator' or args.configuration.count('modulev0'):
        print("Configuring ETROCs")
        for i in range(res['n_module']):
            modules.append(Module(rb, i+1, enable_power_board=args.power_board))
            if args.power_board:
                pb_channel.append(modules[-1].get_power_good())
        print()
        print("Querying module status")
        for m in modules:
            #m.configure()
            m.show_status()

        # Monitoring threads
        if args.monitor:
            from tamalero.Monitoring import Monitoring, module_mon
            #mon1 = module_mon(modules[0])
            monitoring_threads = []
            for i in range(res['n_module']):
                if modules[i].ETROCs[0].connected:
                    monitoring_threads.append(module_mon(modules[i]))

    if args.server:
        app = create_app(rb, modules=modules)
        app.run(port=args.port, threaded=False)

    #-------------------------------------------------------------------------------
    # Read ADCs
    #-------------------------------------------------------------------------------
    if args.adcs:
        if rb.ver < 3:
            print("\n\nReading GBT-SCA ADC values:")
            rb.SCA.read_adcs(check=True, strict_limits=args.strict)

        print("\n\nReading DAQ lpGBT ADC values:")
        rb.DAQ_LPGBT.read_adcs(check=True, strict_limits=args.strict)

        # High level reading of temperatures
        temp = rb.read_temp(verbose=True)

    #-------------------------------------------------------------------------------
    # I2C Test
    #-------------------------------------------------------------------------------
    if args.i2c_temp:

        for i in range(100):
            print ( rb.DAQ_LPGBT.read_temp_i2c() )
            time.sleep(1)

    #-------------------------------------------------------------------------------
    # Read SCA
    #-------------------------------------------------------------------------------
    if args.i2c_sca:

        print("Writing and Reading I2C_ctrl register:")
        for n in range(10):
            wr = random.randint(0, 100)
            rb.SCA.I2C_write_ctrl(channel=3, data=wr)
            rd = rb.SCA.I2C_read_ctrl(channel=3)
            print("write: {} \t read: {}".format(wr, rd))

        print("Testing multi-byte read:")
        multi_out = rb.SCA.I2C_read_multi(channel=3, servant = 0x48, nbytes=2)
        print("servant: 0x48, channel: 3, nbytes: 2, output = {}".format(multi_out))

        print("Testing multi-byte write:")

        write_value = [0x2, 25, (27&240)]
        print("servant: 0x48, channel: 3, nbytes: 2, data:{}".format(write_value))
        rb.SCA.I2C_write_multi(write_value, channel=3, servant=0x48)
        read_value = rb.SCA.I2C_read_multi(channel=3, servant=0x48, nbytes = 2, reg=0x2)

        if read_value == write_value[1:]:
            print ("write/read successful!")
        print("read value = {}".format(rb.SCA.I2C_read_multi(channel=3, servant=0x48, nbytes = 2, reg=0x2)))

    #-------------------------------------------------------------------------------
    # Pattern Checkers
    #-------------------------------------------------------------------------------
    if args.reset_pattern_checker:
        print ("\nResetting the pattern checker.")
        rb.DAQ_LPGBT.set_uplink_group_data_source("normal")
        rb.DAQ_LPGBT.set_downlink_data_src(args.reset_pattern_checker)
        time.sleep(0.1)
        rb.DAQ_LPGBT.reset_pattern_checkers()
        time.sleep(0.1)

    if args.run_pattern_checker:
        print ("\nReading the pattern checker counter. Waiting 1 sec.")
        time.sleep(1)
        rb.DAQ_LPGBT.read_pattern_checkers()

    #-------------------------------------------------------------------------------
    # Eyescan
    #-------------------------------------------------------------------------------
    if args.eyescan:
        rb.DAQ_LPGBT.eyescan()

    all_tests_passed = True  # FIXME this should be properly defined
    if all_tests_passed:
        rb.DAQ_LPGBT.set_configured()

    #-------------------------------------------------------------------------------
    # Success LEDs
    #-------------------------------------------------------------------------------
    if rb.ver > 1:
        rb.is_configured()
        t_end = time.time() + 10
        if args.power_up:
            from tamalero.Monitoring import Monitoring, blink_rhett
            print("RB configured successfully. Rhett is happy " + emojize(":dog_face:"))
            rb.disable_rhett()
            time.sleep(0.5)
            rb.enable_rhett()
            time.sleep(0.5)
            rb.disable_rhett()
            time.sleep(0.5)
            rb.enable_rhett()
            #b = blink_rhett(rb, iterations=3)
