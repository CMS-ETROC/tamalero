#!/usr/bin/env python3
#TODO: This file has a ton of commented out lines
# Do they need to be here? Have they been fixed?
# Should they be updated to find the right values?
'''
RB0 and RB1 connected to KCU 192.168.0.10 #TODO: Is this always true
on PSU 192.168.2.1 ch1 and ch2
'''
import time
import os
import glob
from emoji import emojize
import argparse

from yaml import load, CLoader as Loader, CDumper as Dumper
from tamalero.ReadoutBoard import ReadoutBoard
from tamalero.utils import get_kcu, load_yaml
from tamalero.FIFO import FIFO
from tamalero.DataFrame import DataFrame

'''
Configuration of the module telescope
'''

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--kcu', action='store', default='192.168.0.10', help="IP Address of KCU105 board")
    argParser.add_argument('--configuration', action='store', default='desy', help="Telescope configuration to be loaded")
    argParser.add_argument('--host', action='store', default='localhost', help="Hostname for control hub")
    argParser.add_argument('--enable_power_board', action='store_true', help="Enable Power Board (all modules). Jumpers must still be set as well.")
    argParser.add_argument('--test_config', action='store_true', help="Use a test configuration.")
    argParser.add_argument('--power_up', action='store_true', help="Turn power on (BU setup only)")
    argParser.add_argument('--power_down', action='store_true', help="Turn power off (BU setup only)")
    argParser.add_argument('--subset', action='store_true', help="Use subset of pixels for tests")
    argParser.add_argument('--reuse_thresholds', action='store_true', help="Reuse thresholds from last run")
    argParser.add_argument('--offset', action='store', default='auto', help="The offset from the baseline")
    argParser.add_argument('--delay', action='store', default=15, type=int, help="Set the L1A delay")
    argParser.add_argument('--power_mode', action='store', default='high', choices=['low', 'high'], help="ETROC power mode")
    argParser.add_argument('--dark_mode', action='store_true', help="Turn all LEDs off that can be turned off.")
    argParser.add_argument('--etroc_num_for_threshold', action='store', help="Used during sps october test beam to reuse thresholds for a module with one etroc")
    args = argParser.parse_args()

    config = load_yaml(f"configs/telescope_{args.configuration}.yaml")
    pm = args.power_mode
    shut_down = False
    connect_modules = True
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")

    all_out_dir = glob.glob(f'telescope_config_data/{args.configuration}*')
    all_out_dir.sort(reverse=True)
    if len(all_out_dir) > 0:
        latest_out_dir = all_out_dir[0]
    else:
        out_dir = f"telescope_config_data/{args.configuration}_{timestamp}"

    if args.reuse_thresholds:
        print(f"Using thresholds from {latest_out_dir}")
    else:
        out_dir = f"telescope_config_data/{args.configuration}_{timestamp}"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

    print(emojize(':atom_symbol:'), " Telescope code draft")
    print(f"Using time stamp: {timestamp}")

    PSUs = {}
    if args.power_up:
        from cocina.PowerSupply import PowerSupply
        print(emojize(':battery:'), " Power Supply")
        for layer in config:
            if "psu" in config[layer]:
                for psu_ip, psu_ch in config[layer]["psu"]:
                    if psu_ip not in PSUs:
                        PSUs[psu_ip] = PowerSupply(ip=psu_ip, name='PSU')
                    print(f"Powering up channel {psu_ch} of PSU at {psu_ip}")
                    #psu_tmp = PowerSupply(ip=psu_ip, name='PSU')
                    PSUs[psu_ip].power_up(psu_ch)
                    time.sleep(1)  # PSUs are sloooow
        #            del psu_tmp  # ???

        for ip in PSUs:
            PSUs[ip].monitor()
        #psu1 = PowerSupply(ip='192.168.2.1', name='PS1')
        #psu1.power_up('ch1')
        #psu1.power_up('ch2')
        #psu2 = PowerSupply(ip='192.168.2.2', name='PS2')
        #psu2.power_up('ch1')
        time.sleep(1)
    print(emojize(":check_mark_button:"), " Ready")

    print("Getting the KCU")
    kcu = get_kcu(args.kcu, control_hub=True, verbose=True)

    print("Configuring Readout Boards")
    rbs = {}
    for layer in config:
        print(layer)
        rbs[layer] = ReadoutBoard(rb=layer, trigger=True, kcu=kcu, config=config[layer]['type'], verbose=False)

    #print("Scanning Readout Boards")
    #irbs = []
    #for i in range(5):
    #    # we can at most connect 5 RBs
    #    try:
    #        rb_tmp = ReadoutBoard(rb=i, trigger=False, kcu=kcu, config=args.configuration, verbose=False)
    #        #rbs.append(rb_tmp)
    #        #rbs.append(ReadoutBoard(rb=i, trigger=True, kcu=kcu, config=args.configuration, verbose=False))
    #        print(f"Added RB #{i}")
    #        irbs.append(i)
    #    except:
    #        print(f"Could not add RB#{i}")
    #        #raise

    #print("Configuring Readout Boards")
    #rbs = {}
    #for i in irbs:
    #    rbs[i] = ReadoutBoard(rb=i, trigger=True, kcu=kcu, config=args.configuration, verbose=False)

    if connect_modules:
        print("Connecting Modules")
        for rb in rbs:
            moduleids = [x[0] if len(x)>0 else 0 for x in config[rb]['modules']]
            print(moduleids)
            rbs[rb].connect_modules(moduleids=moduleids)
        #    rb.rerun_bitslip()

            for mod in rbs[rb].modules:
                mod.show_status()

        print("Configuring ETROCs")
        for rb in rbs:
            for mod in rbs[rb].modules:
                if mod.connected:
                    if args.test_config:
                        for etroc in mod.ETROCs:
                            etroc.test_config(occupancy=10)
                    else:
                        if args.subset:
                            test_pixels = [
                                #(0,0),
                                (1,1),
                                (7,7),
                                (7,8),
                                (8,8),
                                (8,7),
                                (0,15),
                                (15,0),
                                (15,15),
                            ]
                        else:
                            test_pixels = False
                        if args.offset == 'auto':
                            offset = args.offset
                        else:
                            offset = int(args.offset)

                        for etroc in mod.ETROCs:
                            if args.reuse_thresholds:
                                
                                with open(f'{latest_out_dir}/thresholds_module_{etroc.module_id}_etroc_{etroc.chip_no}.yaml', 'r') as f:
                                    thresholds = load(f, Loader=Loader)
                                etroc.physics_config(offset=offset, L1Adelay=int(args.delay), subset=test_pixels, thresholds=thresholds, out_dir=latest_out_dir, powerMode=pm)
                            else:
                                etroc.physics_config(offset=offset, L1Adelay=int(args.delay), subset=test_pixels, out_dir=out_dir, powerMode=pm)
                    for etroc in mod.ETROCs:
                        etroc.reset()

        if args.dark_mode:
            for rb in rbs:
                rbs[rb].dark_mode()

        fifos = []
        for rb in rbs:
            fifos.append(FIFO(rbs[rb]))
        #fifo_0 = FIFO(rbs[0])
        #fifo_1 = FIFO(rbs[1])
        #fifo_2 = FIFO(rbs[2])
        df = DataFrame("ETROC2")

        fifos[0].send_l1a(1)
        for fifo in fifos:
            fifo.reset()
        #fifo_0.reset()
        #fifo_1.reset()
        #fifo_2.reset()

        #rbs[1].modules[0].ETROCs[0].wr_reg("readoutClockDelayGlobal", 1)
        #rbs[0].modules[0].ETROCs[0].wr_reg("readoutClockDelayGlobal", 31)

        for rb in rbs:
            for mod in rbs[rb].modules:
                for etroc in mod.ETROCs:
                    etroc.reset()
        #rbs[0].modules[0].ETROCs[0].reset()
        #rbs[1].modules[0].ETROCs[0].reset()
        #rbs[2].modules[0].ETROCs[0].reset()

        # doesn't matter which FIFO to choose, the L1A is universial
        print(emojize(':factory:'), " Producing some test data")
        fifos[0].send_l1a(10)

        for i, fifo in enumerate(fifos):
            print(emojize(':closed_mailbox_with_raised_flag:'), f" Data in FIFO {i}:")
            for x in fifos[i].pretty_read(df):
                #if x[0] == 'data': print ('!!!!!!!!!!', x)
                print(x)

        #print(emojize(':closed_mailbox_with_raised_flag:'), " Data in FIFO 1:")
        #for x in fifo_1.pretty_read(df):
        #    #if x[0] == 'data': print ('!!!!!!!!!!', x)
        #    print(x)

        #print(emojize(':closed_mailbox_with_raised_flag:'), " Data in FIFO 2:")
        #for x in fifo_2.pretty_read(df):
        #    #if x[0] == 'data': print ('!!!!!!!!!!', x)
        #    print(x)

        # This script was verified to work with noise at the BU test stands
        # and it actually sees noise on the wirebonded pixels, as expected

    ## code below is specific for module 40, please change to your convenience
    #thresholds = load_yaml('results/40/2024-03-12-02-19-28/thresholds.yaml')

    ## deactivate hot pixels in Module 40
    #rbs[0].modules[0].ETROCs[0].reset()  # soft reset
    #rbs[0].modules[0].ETROCs[0].disable_data_readout(row=2, col=10, broadcast=False)
    #rbs[0].modules[0].ETROCs[0].disable_data_readout(row=1, col=15, broadcast=False)
    #rbs[0].modules[0].ETROCs[0].disable_data_readout(row=7, col=15, broadcast=False)

    ##rbs[0].modules[0].ETROCs[0].wr_reg("Bypass_THCal", 1, broadcast=True)
    # for row in range(16):
    #    for col in range(16):
    #        rbs[0].modules[0].ETROCs[0].wr_reg('DAC', int(thresholds[row][col]), row=row, col=col)
    #pixels = [
    #    (10,6),
    #    (10,7),
    #    #(10,8),
    #    #(9,6),
    #    #(9,7),
    #    #(9,8),
    #]
    #rbs[0].modules[0].ETROCs[0].disable_data_readout(broadcast=True)
    #for row, col in pixels:
    #    rbs[0].modules[0].ETROCs[0].enable_data_readout(row=row, col=col, broadcast=False)
    #    rbs[0].modules[0].ETROCs[0].wr_reg('DAC', int(thresholds[row][col])+25, row=row, col=col)

    run_timing_scan = False
    #rbs[0].modules[0].ETROCs[0].disable_data_readout(broadcast=True)
    #rbs[0].modules[0].ETROCs[0].enable_data_readout(row=8, col=8, broadcast=False)
    #rbs[0].modules[0].ETROCs[0].wr_reg("DAC", 302, row=8, col=8, broadcast=False)
    if run_timing_scan:
        #rbs[0].modules[0].ETROCs[0].disable_data_readout(row=2, col=10, broadcast=False)
        #rbs[0].modules[0].ETROCs[0].disable_data_readout(row=1, col=15, broadcast=False)
        #rbs[0].modules[0].ETROCs[0].disable_data_readout(row=7, col=15, broadcast=False)
        rbs[0].enable_external_trigger()
        rbs[0].kcu.hw.dispatch()
        for i in range(40):
            rbs[0].modules[0].ETROCs[0].wr_reg("L1Adelay", i, broadcast=True)  # broadcast was missing before.
            rbs[0].reset_data_error_count()
            kcu.write_node(f"READOUT_BOARD_0.EVENT_CNT_RESET", 0x1)
            fifo_0.reset()
            print(kcu.read_node(f"READOUT_BOARD_0.EVENT_CNT").value())
            # while kcu.read_node(f"READOUT_BOARD_0.EVENT_CNT").value()<30:
            #     time.sleep(0.01)
            #time.sleep(10)
            #fifo_0.send_l1a(100)
            #time.sleep(0.1)
            #res = fifo_0.pretty_read(df)
            data_count = rbs[0].read_data_count(elink=0, slave=False)
            trigger_count = kcu.read_node(f"READOUT_BOARD_0.EVENT_CNT").value()
            print(i, data_count, trigger_count)
            #start_time = time.time()
            #while time.time() - start_time < 10:

    if args.power_down:
        psu1.power_down('ch1')
        psu1.power_down('ch2')
        #psu2.power_down('ch1')
