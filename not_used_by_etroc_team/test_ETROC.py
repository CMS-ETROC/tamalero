from tamalero.ETROC import ETROC
from tamalero.ETROC_Emulator import ETROC2_Emulator as software_ETROC2
from tamalero.DataFrame import DataFrame
from tamalero.utils import get_kcu
from tamalero.ReadoutBoard import ReadoutBoard
from tamalero.PixelMask import PixelMask
from tamalero.colors import red, green, yellow

import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
import sys
import json
import time
import datetime
from yaml import load, dump
import traceback
import pickle

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

# ====== HELPER FUNCTIONS ======

# run N L1A's and return packaged ETROC2 dataformat
def run(ETROC, N, fifo=None):
    # currently uses the software ETROC to produce fake data
    if ETROC.isfake:
        return ETROC2.run(N)
    else:
        fifo.send_l1a(N)
        return fifo.pretty_read(None, raw=True)

def toPixNum(row, col, w):
    return col*w+row

def fromPixNum(pix, w):
    row = pix%w
    col = int(np.floor(pix/w))
    return row, col


def sigmoid(k,x,x0):
    return 1/(1+np.exp(k*(x-x0)))

# take x,y values and perform fit to sigmoid function
# return steepness(k) and mean(x0)
def sigmoid_fit(x_axis, y_axis):
    res = curve_fit(
        #sigmoid,
        lambda x,a,b: 1/(1+np.exp(a*(x-b))),  # for whatever reason this fit only works with a lambda function?
        x_axis-x_axis[0],
        y_axis,
        maxfev=10000,
        bounds=([-np.inf,-np.inf],[0,np.inf])
    )
    return res[0][0], res[0][1]+x_axis[0]

# parse ETROC dataformat into 1D list of # of hits per pixel
def parse_data(data, N_pix):
    results = np.zeros(N_pix)
    pix_w = int(round(np.sqrt(N_pix)))

    for word in data:
        datatype, res = DF.read(word)
        if datatype == 'data':
            pix = toPixNum(res['row_id'], res['col_id'], pix_w)
            results[pix] += 1

    return results

def vth_scan(ETROC2, vth_min=693, vth_max=709, vth_step=1, decimal=False, fifo=None, absolute=False):
    # N_l1a    =  3200 # how many L1As to send
    N_l1a    =  3200 # how many L1As to send
    vth_min  =   vth_min # scan range
    vth_max  =   vth_max
    if not decimal:
        vth_step =   ETROC2.DAC_step # step size
    else:
        vth_step = vth_step
    N_steps  = int((vth_max-vth_min)/vth_step)+1 # number of steps
    N_pix    = 16*16 # total number of pixels

    vth_axis    = np.linspace(vth_min, vth_max, N_steps)
    run_results = np.empty([N_steps, N_pix])
    
    for vth in vth_axis:
        print(f"Working on threshold {vth=}")
        if decimal:
            ETROC2.wr_reg('DAC', int(vth), broadcast=True)
            #print("Acc value", ETROC2.get_ACC(row=0, col=0)) #doesn't work here
        else:
            ETROC2.set_Vth_mV(vth)
        i = int((vth-vth_min)/vth_step)
        run_results[i] = parse_data(run(ETROC2, N_l1a, fifo=fifo), N_pix)
        
    # transpose so each 1d list is for a pixel & normalize
    if absolute:
        run_results = run_results.transpose()
    else:
        run_results = run_results.transpose()/N_l1a
    return [vth_axis.tolist(), run_results.tolist()]

def vth_scan_internal(ETROC2, row=0, col=0, dac_min=0, dac_max=500, dac_step=1):
    N_steps  = int((dac_max-dac_min)/dac_step)+1 # number of steps
    dac_axis = np.linspace(dac_min, dac_max, N_steps)

    ETROC2.setup_accumulator(row, col)
    results = []
    for i in range(dac_min, dac_max+1, dac_step):
        results.append(
            etroc.check_accumulator(DAC=i, row=row, col=col)
        )
    run_results = np.array(results)
    return [dac_axis, run_results]

# initiate
ETROC2 = software_ETROC2()  # currently using Software ETROC2 (fake)
print("ETROC2 emulator instantiated, base configuration successful")
DF = DataFrame('ETROC2')

if __name__ == '__main__':
    # argsparser
    import argparse
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--test_readwrite', action='store_true', default=False, help="Test simple read/write functionality?")
    argParser.add_argument('--test_chip', action='store_true', default=False, help="Test simple read/write functionality for real chip?")
    argParser.add_argument('--config_chip', action='store_true', default=False, help="Configure chip?")
    argParser.add_argument('--configuration', action='store', default='modulev1', choices=['modulev0', 'modulev0b', 'multimodule', 'modulev1'], help="Board configuration to be loaded")
    argParser.add_argument('--vth', action='store_true', default=False, help="Parse Vth scan plots?")
    argParser.add_argument('--rerun', action='store_true', default=False, help="Rerun Vth scan and overwrite data?")
    argParser.add_argument('--fitplots', action='store_true', default=False, help="Create individual vth fit plots for all pixels?")
    argParser.add_argument('--kcu', action='store', default='192.168.0.10', help="IP Address of KCU105 board")
    argParser.add_argument('--module', action='store', default=0, choices=['1','2','3'], help="Module to test")
    argParser.add_argument('--etroc', action='store', default=0, choices=['0','1','2','3'], help="Module to test")
    argParser.add_argument('--rb', action='store', default=0, choices=['0','1','2','3'], help="Module to test")
    argParser.add_argument('--host', action='store', default='localhost', help="Hostname for control hub")
    argParser.add_argument('--partial', action='store_true', default=False, help="Only read data from corners and edges")
    argParser.add_argument('--qinj_scan', action='store_true', default=False, help="Run the phase scan for Qinj")
    argParser.add_argument('--qinj', action='store_true', default=False, help="Run some charge injection tests")
    argParser.add_argument('--qinj_vth_scan', action='store_true', default=False, help="Run some charge injection tests")
    argParser.add_argument('--charge', action='store', default=15, help="Charge to be injected")
    argParser.add_argument('--hard_reset', action='store_true', default=False, help="Hard reset of selected ETROC2 chip")
    argParser.add_argument('--skip_sanity_checks', action='store_true', default=False, help="Don't run sanity checks of ETROC2 chip")
    argParser.add_argument('--pixelscan', action='store', default='none', choices=['none', 'full', 'simple', 'internal'], help="Which threshold scan to run with ETROC2")
    argParser.add_argument('--mode', action='store', default='dual', choices=['dual', 'single'], help="Port mode for ETROC2")
    argParser.add_argument('--threshold', action='store', default='auto', help="Use thresholds from manual or automatic scan?")
    argParser.add_argument('--internal_data', action='store_true', help="Set up internal data generation")
    argParser.add_argument('--enable_power_board', action='store_true', help="Enable Power Board (all modules). Jumpers must still be set as well.")
    argParser.add_argument('--timing_scan', action='store_true', help="Perform L1Adelay timing scan")
    argParser.add_argument('--row', action='store', default=4,type=int, help="Pixel row to be tested")
    argParser.add_argument('--col', action='store', default=3,type=int, help="Pixel column to be tested")
    argParser.add_argument('--pixel_mask', action='store', default=None, help="Pixel mask to apply")
    argParser.add_argument('--moduleid', action='store', default=0, help="")
    argParser.add_argument('--charges', action = 'store', default = [15], nargs = '*', type = int, help = 'Charges to inject')
    argParser.add_argument('--vth_axis', action = 'store', default = [415, 820, 406], nargs = '*', type = int, help = 'Threshold DAC values to scan over')
    argParser.add_argument('--nl1a', action = 'store', type = int, default = 3200, help = 'Number of Qinj/L1A pulses to send')
    argParser.add_argument('--multi', action='store_true', help="Run multiple modules at once (for data taking only!)")
    args = argParser.parse_args()

    print(type(args.row))

    if args.test_chip or args.config_chip:
        assert int(args.moduleid)>0, "Module ID is not specified. This is a new feature, please run with --moduleid MODULEID, where MODULEID is the number on the module test board."
        MID = args.moduleid
    else:
        MID = "software"

    ietroc = int(args.etroc)

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")

    result_dir = f"results/{MID}/{timestamp}/"
    out_dir = f"outputs/{MID}/{timestamp}/"

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    print(f"Will test module with ID {MID}.")
    print(f"All results will be stored under timestamp {timestamp}")

    if args.test_readwrite:
        # ==============================
        # === Test simple read/write ===
        # ==============================
        print("<--- Test simple read/write --->")
        print("Testing read/write to addresses...")

        test_val = 0x2
        print(f"Broadcasting {test_val=} to CLSel in-pixel registers")
        ETROC2.wr_reg('CLSel', test_val, broadcast=True)
        assert ETROC2.rd_reg('CLSel', row=2, col=3) == test_val, "Did not read back the expected value"
        print("Test passed.\n")

        test_val = 2**8 + 2**5
        print(f"Broadcasting {test_val=} to DAC in-pixel registers")
        ETROC2.wr_reg('DAC', test_val, broadcast=True)
        assert ETROC2.rd_reg('DAC', row=5, col=4) == test_val, "Did not read back the expected value"
        print("Test passed.\n")

        test_val = 2**11 + 2**5
        print(f"Trying to broadcast too large value {test_val=} to DAC in-pixel registers")
        try:
            ETROC2.wr_reg('DAC', test_val, broadcast=True)
            raise NotImplementedError("Test failed.")
        except RuntimeError:
            print("Succesfully failed, as expected.")
            pass
        print("Test passed.\n")

        test_val = 700.25
        print(f"Trying to set the threshold to {test_val=}mV")
        ETROC2.set_Vth_mV(test_val)
        read_val = ETROC2.get_Vth_mV(row=4, col=5)
        if abs(read_val-test_val)>ETROC2.DAC_step:
            raise RuntimeError("Returned discriminator threshold is off.")
        else:
            print(f"Threshold is currently set to {read_val=} mV")
            print("Test passed.\n")

    elif args.multi:
        from tamalero.FIFO import FIFO
        from tamalero.DataFrame import DataFrame
        df = DataFrame()

        kcu = get_kcu(args.kcu, control_hub=True, host=args.host, verbose=False)
        if (kcu == 0):
            # if not basic connection was established the get_kcu function returns 0
            # this would cause the RB init to fail.
            sys.exit(1)

        rb_0 = ReadoutBoard(int(args.rb), kcu=kcu, config=args.configuration)
        data = 0xabcd1234
        kcu.write_node("LOOPBACK.LOOPBACK", data)
        if (data != kcu.read_node("LOOPBACK.LOOPBACK")):
            print("No communications with KCU105... quitting")
            sys.exit(1)

        fifo = FIFO(rb=rb_0)
        fifo.send_l1a(1)
        fifo.reset()

        from tamalero.Module import Module

        print("\n - Getting modules")
        modules = []
        connected_modules = []
        for i in [1,2,3]:
            # FIXME we might want to hard reset all ETROCs at this point?
            moduleid = int(args.moduleid) if i==int(args.module) else (i+200)  # NOTE: at some point we should also include the RB number, e.g. (rb << 4) | (module)
            m_tmp = Module(rb=rb_0, i=i, enable_power_board=args.enable_power_board, moduleid = moduleid)
            modules.append(m_tmp)
            if m_tmp.ETROCs[ietroc].is_connected():  # NOTE assume that module is connected if first ETROC is connected
                connected_modules.append(i)
                for e_tmp in m_tmp.ETROCs:
                    if args.hard_reset:
                        print(f"Running hard reset and default config on module {i}")
                        e_tmp.reset(hard=True)
                        e_tmp.default_config()
                        time.sleep(1.1)
                        #e_tmp.default_config()
                        #
                    print("Setting ETROCs into workMode 0")
                    e_tmp.wr_reg("workMode", 0, broadcast=True)

        time.sleep(0.1)
        rb_0.enable_etroc_readout()
        rb_0.enable_etroc_readout(slave=True)
        rb_0.reset_data_error_count()

        for module in modules:
            module.show_status()

        fifo.reset()
        print("\n - Testing fast command communication - Sending two L1As")
        fifo.send_l1a(2)
        for x in fifo.pretty_read(df):
            print(x)

    elif args.config_chip:
        start_time = time.time()
        kcu = get_kcu(args.kcu, control_hub=True, host=args.host, verbose=False)
        if (kcu == 0):
            # if not basic connection was established the get_kcu function returns 0
            # this would cause the RB init to fail.
            sys.exit(1)

        int_time = time.time()
        rb_0 = ReadoutBoard(int(args.rb), kcu=kcu, config=args.configuration)
        data = 0xabcd1234
        kcu.write_node("LOOPBACK.LOOPBACK", data)
        if (data != kcu.read_node("LOOPBACK.LOOPBACK")):
            print("No communications with KCU105... quitting")
            sys.exit(1)

        from tamalero.Module import Module

        int2_time = time.time()
        # FIXME the below code is still pretty stupid
        modules = []
        for i in [1,2,3]:
            m_tmp = Module(rb=rb_0, i=i)
            if m_tmp.ETROCs[ietroc].connected:  # NOTE assume that module is connected if first ETROC is connected
                modules.append(m_tmp)

        end_time = time.time()

        print("KCU init done in     {:.2f}s".format(int_time-start_time))
        print("RB init done in      {:.2f}s".format(int2_time-int_time))
        print("Module init done in  {:.2f}s".format(end_time-int2_time))  # default config is what's slow

    elif args.test_chip:
        kcu = get_kcu(args.kcu, control_hub=True, host=args.host, verbose=False)
        if (kcu == 0):
            # if not basic connection was established the get_kcu function returns 0
            # this would cause the RB init to fail.
            sys.exit(1)

        rb_0 = ReadoutBoard(int(args.rb), kcu=kcu, config=args.configuration)
        data = 0xabcd1234
        kcu.write_node("LOOPBACK.LOOPBACK", data)
        if (data != kcu.read_node("LOOPBACK.LOOPBACK")):
            print("No communications with KCU105... quitting")
            sys.exit(1)

        from tamalero.Module import Module

        # FIXME the below code is still pretty stupid
        modules = []
        connected_modules = []

        for i in [1,2,3]:
            # FIXME we might want to hard reset all ETROCs at this point?
            moduleid = int(args.moduleid) if i==int(args.module) else (i+200)
            m_tmp = Module(rb=rb_0, i=i, enable_power_board=args.enable_power_board, moduleid = moduleid)
            modules.append(m_tmp)
            if m_tmp.ETROCs[ietroc].is_connected():  # NOTE assume that module is connected if first ETROC is connected
                connected_modules.append(i)
                for e_tmp in m_tmp.ETROCs:
                    if args.hard_reset:
                        print(f"Running hard reset and default config on module {i}")
                        e_tmp.reset(hard=True)
                        e_tmp.default_config()
                        time.sleep(1.1)
                        #e_tmp.default_config()
                        #
                    print("Setting ETROCs into workMode 0")
                    e_tmp.wr_reg("workMode", 0, broadcast=True)

        #for i in [1,2,3]:
        #    m_tmp = Module(rb=rb_0, i=i, enable_power_board=args.enable_power_board)
        #    modules.append(m_tmp)
        #    if m_tmp.ETROCs[0].connected:  # NOTE assume that module is connected if first ETROC is connected
        #        connected_modules.append(i)

        print(f"Found {len(connected_modules)} connected modules")
        if int(args.module) > 0:
            module = int(args.module)
        else:
            module = connected_modules[0]

        print(f"Will proceed with testing Module {module}")
        print("Module status:")
        modules[module-1].show_status()

        etroc = modules[module-1].ETROCs[ietroc]
        if args.hard_reset:
            etroc.reset(hard=True)
            etroc.default_config()

        if args.mode == 'single':
            print(f"Setting the ETROC in single port mode ('right')")
            etroc.set_singlePort("right")
            etroc.set_mergeTriggerData("separate")
        elif args.mode == 'dual':
            print(f"Setting the ETROC in dual port mode ('both')")
            etroc.set_singlePort("both")
            etroc.set_mergeTriggerData("merge")

        #etroc = ETROC(rb=rb_0, i2c_adr=96, i2c_channel=1, elinks={0:[0,2]})

        # Load a pixel mask if specified
        masked_pixels = []
        if args.pixel_mask:
            mask = PixelMask.from_file(args.pixel_mask)
            masked_pixels = mask.get_masked_pixels()

            print("\n - Will apply the following pixel mask:")
            mask.show()

        if not args.skip_sanity_checks:

            print("\n - Checking peripheral configuration:")
            etroc.print_perif_conf()

            print("\n - Checking peripheral status:")
            etroc.print_perif_stat()

            print("\n - Running pixel sanity check:")
            res = etroc.pixel_sanity_check(verbose=False)
            if res:
                print("Passed!")
            else:
                print("Failed")

            print("\n - Running pixel random check:")
            res = etroc.pixel_random_check(verbose=False)
            if res:
                print("Passed!")
            else:
                print("Failed")

            print("\n - Checking configuration for pixel (4,5):")
            etroc.print_pixel_conf(row=4, col=5)

            print("\n - Checking status for pixel (4,5):")
            etroc.print_pixel_stat(row=4, col=5)

            ## pixel broadcast
            print("\n - Checking pixel broadcast.")
            etroc.wr_reg('workMode', 0, broadcast=True)
            tmp = etroc.rd_reg('workMode', row=10, col=10)
            etroc.wr_reg('workMode', 1, broadcast=True)
            test0 = True
            for row in range(16):
                for col in range(16):
                    test0 &= (etroc.rd_reg('workMode', row=row, col=col) == 1)
            tmp2 = etroc.rd_reg('workMode', row=10, col=10)
            tmp3 = etroc.rd_reg('workMode', row=3, col=12)
            etroc.wr_reg('workMode', 0, broadcast=True)
            tmp4 = etroc.rd_reg('workMode', row=10, col=10)
            test1 = (tmp != tmp2)
            test2 = (tmp2 == tmp3)
            test3 = (tmp == tmp4)
            if test0 and test1 and test2 and test3:
                print("Passed!")
            else:
                print(f"Failed: {test0=}, {test1=}, {test2=}, {test3=}")
        else:
            # still run sanity check but quietly
            # this is needed so that we can deactivate problematic / hot pixels
            res = etroc.pixel_sanity_check(verbose=False)

        print("\n - Deactivating problematic / hot pixels")
        etroc.deactivate_hot_pixels(pixels=masked_pixels)

        # NOTE below is WIP code for tests of the actual data readout
        from tamalero.FIFO import FIFO
        from tamalero.DataFrame import DataFrame
        df = DataFrame()
        # NOTE this is for single port tests right now, where we only get elink 2
        fifo = FIFO(rb=rb_0)
        #fifo.select_elink(0)
        #fifo.ready()

        print("\n - Checking elinks")

        #print("Disabling readout for all elinks but the ETROC under test")
        ## FIXME this is still problematic...
        #rb_0.disable_etroc_readout(all=True)
        #rb_0.reset_data_error_count()
        ##rb_0.enable_etroc_readout()
        #for lpgbt in etroc.elinks:
        #    if lpgbt == 0:
        #        slave = False
        #    else:
        #        slave = True
        #    for link in etroc.elinks[lpgbt]:
        #        print(f"Enabling elink {link}, slave is {slave}")
        #        rb_0.enable_etroc_readout(link, slave=slave)
        #        #time.sleep(0.5)
        #        #rb_0.reset_data_error_count()
        #        #fifo.select_elink(link, slave)
        #        #fifo.ready()
        #        rb_0.rerun_bitslip()
        #        time.sleep(1.5)
        #        rb_0.reset_data_error_count()
        #        stat = rb_0.get_link_status(link, slave=slave, verbose=False)
        #        if stat:
        #            rb_0.get_link_status(link, slave=slave)
        #        start_time = time.time()
        #        while not stat:
        #            #rb_0.disable_etroc_readout(link, slave=slave)
        #            rb_0.enable_etroc_readout(link, slave=slave)
        #            #time.sleep(0.5)
        #            #time.sleep(0.1)
        #            #rb_0.reset_data_error_count()
        #            #fifo.select_elink(link, slave)
        #            #fifo.ready()
        #            rb_0.rerun_bitslip()
        #            time.sleep(1.5)
        #            rb_0.reset_data_error_count()
        #            stat = rb_0.get_link_status(link, slave=slave, verbose=False
        #                                        )
        #            if stat:
        #                rb_0.get_link_status(link, slave=slave)
        #                break
        #            if time.time() - start_time > 2:
        #                print('Link not good, but continuing')
        #                rb_0.get_link_status(link, slave=slave)
        #                break

        ## Bloc below to be retired
        ## Keeping it for one more iteration
        #locked = kcu.read_node(f"READOUT_BOARD_0.ETROC_LOCKED").value()
        #if (locked & 0b101) == 5:
        #    print(green('Both elinks (0 and 2) are locked.'))
        #elif (locked & 1) == 1:
        #    print(yellow('Only elink 0 is locked.'))
        #elif (locked & 4) == 4:
        #    print(yellow('Only elink 2 is locked.'))
        #else:
        #    print(red('No elink is locked.'))

        # The block below is a bit unclear
        # The first FIFO read (fifo.pretty_read(df)) will always fail,
        # but with a reset afterwards it should get in a stable state
        start_time = time.time()
        while True:
            try:
                fifo.reset()
                fifo.send_l1a(10)
                _ = fifo.pretty_read(df)
                fifo.reset()
                break
            except:
                print("Initial (re)set of FIFO.")
                if time.time() - start_time > 1:
                    print("FIFO state is unexpected.")
                    raise
        etroc.reset()

        print("\n - Getting internal test data")

        etroc.wr_reg("DAC", 0, broadcast=True)  # make sure that we're not additionally picking up any noise
        etroc.wr_reg("selfTestOccupancy", 2, broadcast=True)
        if not args.partial:
            etroc.wr_reg("workMode", 0x1, broadcast=True)
        else:
            etroc.wr_reg("selfTestOccupancy", 70, broadcast=True)
            etroc.wr_reg("workMode", 0x0, broadcast=True)
            ## center pixels
            #etroc.wr_reg("workMode", 0x1, row=15, col=7)
            etroc.wr_reg("workMode", 0x1, row=7, col=7)
            etroc.wr_reg("workMode", 0x1, row=7, col=8)
            etroc.wr_reg("workMode", 0x1, row=8, col=7)
            etroc.wr_reg("workMode", 0x1, row=8, col=8)
            # corner pixels
            etroc.wr_reg("workMode", 0x1, row=0, col=0)
            etroc.wr_reg("workMode", 0x1, row=15, col=15)
            etroc.wr_reg("workMode", 0x1, row=0, col=15)
            etroc.wr_reg("workMode", 0x1, row=15, col=0)
            # edge pixels
            etroc.wr_reg("workMode", 0x1, row=7, col=0)
            etroc.wr_reg("workMode", 0x1, row=8, col=0)
            etroc.wr_reg("workMode", 0x1, row=0, col=7)
            etroc.wr_reg("workMode", 0x1, row=0, col=8)
            etroc.wr_reg("workMode", 0x1, row=7, col=15)
            etroc.wr_reg("workMode", 0x1, row=8, col=15)
            etroc.wr_reg("workMode", 0x1, row=15, col=7)
            etroc.wr_reg("workMode", 0x1, row=15, col=8)

        etroc.deactivate_hot_pixels(pixels=masked_pixels)
        etroc.wr_reg("onChipL1AConf", 0x2)  # NOTE: internal L1A is around 1MHz, so we're only turning this on for the shortest amount of time.
        etroc.wr_reg("onChipL1AConf", 0x0)
        test_data = []
        while fifo.get_occupancy() > 0:
            test_data += fifo.pretty_read(df)

        import hist
        import matplotlib.pyplot as plt
        import mplhep as hep
        plt.style.use(hep.style.CMS)

        hits_total = np.zeros((16,16))
        row_axis = hist.axis.Regular(16, -0.5, 15.5, name="row", label="row")
        col_axis = hist.axis.Regular(16, -0.5, 15.5, name="col", label="col")
        hit_matrix = hist.Hist(col_axis,row_axis)
        n_events_total = 0
        n_events_hit   = 0
        n_events_err   = 0
        for d in test_data:
            if d[0] == 'trailer':
                n_events_total += 1
                if d[1]['hits'] > 0:
                    n_events_hit += 1
            if d[0] == 'data':
                hit_matrix.fill(row=d[1]['row_id'], col=d[1]['col_id'])
                hits_total[d[1]['row_id']][d[1]['col_id']] += 1

                try:
                    if d[1]['row_id'] != d[1]['row_id2']:
                        print("Unpacking error in row ID")
                        n_events_err += 1
                except KeyError:
                    print("Unpacking error in row ID")
                    n_events_err += 1

                try:
                    if d[1]['col_id'] != d[1]['col_id2']:
                        print("Unpacking error in col ID")
                        n_events_err += 1
                except KeyError:
                    print("Unpacking error in col ID")
                    n_events_err += 1

                try:
                    if d[1]['test_pattern'] != 0xaa:
                        print(f"Unpacking error in test pattern, expected 0xAA but got {d[1]['test_pattern']=}")
                        n_events_err += 1
                except KeyError:
                    print(f"Unpacking error in test pattern: didn't find a test pattern!")
                    n_events_err += 1

        print(f"Got number of total events {n_events_total=}")
        print(f"Events with at least one hit {n_events_hit=}")
        print(f"Events with some error in data unpacking {n_events_err=}")

        fig, ax = plt.subplots(1,1,figsize=(7,7))
        hit_matrix.plot2d(
            ax=ax,
        )
        ax.set_ylabel(r'$Row$')
        ax.set_xlabel(r'$Column$')
        hep.cms.label(
                "ETL Preliminary",
                data=True,
                lumi='0',
                com=0,
                loc=0,
                ax=ax,
                fontsize=15,
            )
        name = 'hit_matrix_internal_test_pattern'
        fig.savefig(os.path.join(result_dir, f"{name}.pdf"))
        fig.savefig(os.path.join(result_dir, f"{name}.png"))

        fifo.reset()
        print("\n - Testing fast command communication - Sending two L1As")
        fifo.send_l1a(2)
        for x in fifo.pretty_read(df):
            print(x)

        #etroc.QInj_unset(broadcast=True)
        fifo.reset()
        if not args.partial:
            print("Will use workMode 1 to get some occupancy (no noise or charge injection)")
            etroc.wr_reg("workMode", 0x1, broadcast=True)  # this overwrites disDataReadout!
            etroc.deactivate_hot_pixels(pixels=masked_pixels)
            #etroc.deactivate_hot_pixels(pixels=[(i,1) for i in range(16)])

            for j in range(1):
                # One go is enough, but can run through this loop many times if there's any issue
                # with FIFO / data readout at any point
                print(j)
                ### Another occupancy map
                i = 0
                occupancy = 0
                print("\n - Will send L1As until FIFO is full.")

                #etroc.QInj_set(30, 0, row=3, col=3, broadcast=False)
                start_time = time.time()
                with tqdm(total=65536, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
                    while not fifo.is_full():
                        fifo.send_l1a()
                        #fifo.send_QInj(delay=j)
                        #fifo.send_QInj()
                        i +=1
                        if i%100 == 0:
                            tmp = fifo.get_occupancy()
                            pbar.update(tmp-occupancy)
                            occupancy = tmp
                        #if time.time()-start_time>5:
                        #    print("Time out")
                        #    break

                test_data = []
                while fifo.get_occupancy() > 0:
                    test_data += fifo.pretty_read(df)

                hits_total = np.zeros((16,16))
                hit_matrix = hist.Hist(col_axis,row_axis)
                n_events_total = 0
                n_events_hit   = 0
                for d in test_data:
                    if d[0] == 'trailer':
                        n_events_total += 1
                        if d[1]['hits'] > 0:
                            n_events_hit += 1
                    if d[0] == 'data':
                        hit_matrix.fill(row=d[1]['row_id'], col=d[1]['col_id'])
                        hits_total[d[1]['row_id']][d[1]['col_id']] += 1
                        # NOTE could do some CRC check.

                print(f"Got number of total events {n_events_total=}")
                print(f"Events with at least one hit {n_events_hit=}")

                fig, ax = plt.subplots(1,1,figsize=(7,7))
                hit_matrix.plot2d(
                    ax=ax,
                )
                ax.set_ylabel(r'$Row$')
                ax.set_xlabel(r'$Column$')
                hep.cms.label(
                        "ETL Preliminary",
                        data=True,
                        lumi='0',
                        com=0,
                        loc=0,
                        ax=ax,
                        fontsize=15,
                    )
                name = 'hit_matrix_external_L1A'
                fig.savefig(os.path.join(result_dir, "{}.pdf".format(name)))
                fig.savefig(os.path.join(result_dir, "{}.png".format(name)))


                print("\nOccupancy vs column:")
                hit_matrix[{"row":sum}].show(columns=100)
                print("\nOccupancy vs row:")
                hit_matrix[{"col":sum}].show(columns=100)

        # Set the chip back into well-defined workMode 0
        etroc.wr_reg("workMode", 0x0, broadcast=True)

        if args.threshold == 'auto':
            # using broadcast.
            # NOTE: not working in ETROC2, so this part is currently being skipped
            if False:
                print ("Using auto-threshold calibration with broadcast")
                baseline, noise_width = etroc.auto_threshold_scan(broadcast=True)

                fig, ax = plt.subplots(1,1,figsize=(7,7))
                cax = ax.matshow(baseline)
                ax.set_ylabel(r'$Row$')
                ax.set_xlabel(r'$Column$')
                fig.colorbar(cax,ax=ax)
                name = 'baseline_auto_broadcast'
                fig.savefig(os.path.join(plot_dir, "{}.pdf".format(name)))
                fig.savefig(os.path.join(plot_dir, "{}.png".format(name)))

                fig, ax = plt.subplots(1,1,figsize=(7,7))
                cax = ax.matshow(noise_width)
                ax.set_ylabel(r'$Row$')
                ax.set_xlabel(r'$Column$')
                fig.colorbar(cax,ax=ax)
                name = 'noisewidth_auto_broadcast'
                fig.savefig(os.path.join(plot_dir, "{}.pdf".format(name)))
                fig.savefig(os.path.join(plot_dir, "{}.png".format(name)))

            # not using broadcast
            print ("\n - Using auto-threshold calibration for individual pixels")
            print ("Info: if progress is slow, probably most pixel threshold calibrations time out because of high noise levels.")
            baseline = np.empty([16, 16])
            noise_width = np.empty([16, 16])
            #pixel = 0
            with tqdm(total=256, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
                for pixel in range(256):
                    row = pixel & 0xF
                    col = (pixel & 0xF0) >> 4
                    #print(pixel, row, col)
                    baseline[row][col], noise_width[row][col] = etroc.auto_threshold_scan(row=row, col=col, broadcast=False)
                    #print(pixel)
                    pbar.update()
            #pbar.close()
                    #pixel += 1
            #for i, j in pixels: in range(16):
            #    for j in range(16):

            print ("Done with threshold scan")
            fig, ax = plt.subplots(1,1,figsize=(7,7))
            cax = ax.matshow(baseline)
            ax.set_ylabel(r'$Row$')
            ax.set_xlabel(r'$Column$')
            fig.colorbar(cax,ax=ax)
            name = 'baseline_auto_individual'
            fig.savefig(os.path.join(result_dir, "{}.pdf".format(name)))
            fig.savefig(os.path.join(result_dir, "{}.png".format(name)))

            fig, ax = plt.subplots(1,1,figsize=(7,7))
            cax = ax.matshow(noise_width)
            ax.set_ylabel(r'$Row$')
            ax.set_xlabel(r'$Column$')
            fig.colorbar(cax,ax=ax)
            name = 'noisewidth_auto_individual'
            fig.savefig(os.path.join(result_dir, "{}.pdf".format(name)))
            fig.savefig(os.path.join(result_dir, "{}.png".format(name)))

            with open(f'{result_dir}/thresholds.yaml', 'w') as f:
                dump((baseline+noise_width).tolist(), f)
            
            with open(f'{result_dir}/baseline.yaml', 'w') as f:
                dump((baseline).tolist(), f)

            with open(f'{result_dir}/noise_width.yaml', 'w') as f:
                dump((noise_width).tolist(), f)

        elif args.threshold == "manual":
            # Prescanning a random pixel to get an idea of the threshold
            row = int(args.row)
            col = int(args.col)

            elink, slave = etroc.get_elink_for_pixel(row, col)

            rb_0.reset_data_error_count()
            print("\n - Running simple threshold scan on single pixel")
            print(f"Found this pixel on elink {elink}, lpGBT is servant: {slave}")
            vth     = []
            count   = []
            #etroc.reset(hard=True)
            #etroc.default_config()

            rb_0.get_link_status(elink, slave=slave)

            print("Coarse scan to find the peak location")
            first_val = 1023
            for i in range(0, 1000, 3):
                etroc.wr_reg("DAC", i, row=row, col=col)
                fifo.send_l1a(2000)
                vth.append(i)
                data_cnt = rb_0.read_data_count(elink, slave=slave)
                count.append(data_cnt)
                if data_cnt > 0:
                    print(i, data_cnt)
                    first_val = i
                rb_0.reset_data_error_count()
                if i > (first_val + 10):
                    # break the loop early because this scan is sloooow
                    print("I've seen enough, breaking coarse scan.")
                    break

            vth_a = np.array(vth)
            count_a = np.array(count)
            vth_max = vth_a[np.argmax(count_a)]
            print(f"Found maximum count at DAC setting vth_max={vth_max}")

            ### threshold scan draft
            dac_min = max(0, vth_max - 75)  # don't run into negatives!
            dac_max = vth_max + 75
            vth_scan_data = vth_scan(
                etroc,
                vth_min = dac_min,
                vth_max = dac_max,
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
            noise_matrix  = np.empty([N_pix_w, N_pix_w])
            threshold_matrix = np.empty([N_pix_w, N_pix_w])
            
            rawout = {vth_axis[i]:hit_rate.T[i].tolist() for i in range(len(vth_axis))}
            with open(out_dir + '/manual_thresh_scan_data.json', 'w') as f:
                json.dump(rawout, f)

            # for pix in range(N_pix):
            #     plt.plot(vth_axis[hit_rate[pix] > 0], hit_rate[pix][hit_rate[pix] > 0])
            #     plt.savefig(f"./hit_rate/hit_rate_{pix}.png")
            #     plt.close()

            for pix in range(N_pix):
                r, c = fromPixNum(pix, N_pix_w)
                max_matrix[r][c] = maximums[pix]
                noise_matrix[r][c] = np.size(np.nonzero(hit_rate[pix]))
                max_value = vth_axis[hit_rate[pix]==max(hit_rate[pix])]
                if isinstance(max_value, np.ndarray):
                    max_value = max_value[-1]
                zero_dac_values = vth_axis[((vth_axis>(max_value)) & (hit_rate[pix]==0))]
                if len(zero_dac_values)>0:
                    threshold_matrix[r][c] = zero_dac_values[0] + 2
                else:
                    threshold_matrix[r][c] = dac_max + 2

            # 2D histogram of the mean
            # this is based on the code for automatic sigmoid fits
            # for software emulator data below
            fig, ax = plt.subplots(2,1, figsize=(15,15))
            ax[0].set_title("Peak values of threshold scan")
            ax[1].set_title("Noise width of threshold scan")
            cax1 = ax[0].matshow(max_matrix)
            cax2 = ax[1].matshow(noise_matrix)
            fig.colorbar(cax1,ax=ax[0])
            fig.colorbar(cax2,ax=ax[1])

            ax[0].set_xticks(np.arange(N_pix_w))
            ax[0].set_yticks(np.arange(N_pix_w))

            ax[1].set_xticks(np.arange(N_pix_w))
            ax[1].set_yticks(np.arange(N_pix_w))

            for i in range(N_pix_w):
                for j in range(N_pix_w):
                    text = ax[0].text(j, i, int(max_matrix[i,j]),
                            ha="center", va="center", color="w", fontsize="xx-small")

                    text1 = ax[1].text(j, i, int(noise_matrix[i,j]),
                            ha="center", va="center", color="w", fontsize="xx-small")

            fig.savefig(f'{result_dir}/peak_and_noiseWidth_thresholds.png')
            plt.show()

            plt.close(fig)
            del fig, ax

            fig, ax = plt.subplots()
            plt.title("Thresholds from manual scan")
            cax = ax.matshow(threshold_matrix)

            fig.colorbar(cax)
            ax.set_xticks(np.arange(N_pix_w))
            ax.set_yticks(np.arange(N_pix_w))

            for i in range(N_pix_w):
                for j in range(N_pix_w):
                    text = ax.text(j, i, int(threshold_matrix[i,j]),
                            ha="center", va="center", color="w", fontsize="xx-small")

            ax.set_xlabel("Column")
            ax.set_ylabel("Row")

            fig.savefig(f'{result_dir}/thresholds.png')
            plt.show()

            plt.close(fig)
            del fig, ax

            with open(f'{result_dir}/thresholds.yaml', 'w') as f:
                dump(threshold_matrix.tolist(), f)

            with open(f'{result_dir}/baseline.yaml', 'w') as f:
                dump(max_matrix.tolist(), f)
            
            with open(f'{result_dir}/noise_width.yaml', 'w') as f:
                dump(noise_matrix.tolist(), f)

        else:
            print(f"Trying to load tresholds from the following file: {args.threshold}")
            with open(args.threshold, 'r') as f:
                threshold_matrix = load(f)

            for row in range(16):
                for col in range(16):
                    etroc.wr_reg('DAC', int(threshold_matrix[row][col]), row=row, col=col)

        if args.pixelscan == 'simple':
            row = 4
            col = 3

            elink, slave = etroc.get_elink_for_pixel(row, col)

            rb_0.reset_data_error_count()
            print("\n - Running simple threshold scan on single pixel")
            vth     = []
            count   = []
            etroc.reset(hard=True)
            etroc.default_config()
            print("Coarse scan to find the peak location")
            for i in range(0, 1000, 5):
                # this could use a tqdm
                etroc.wr_reg("DAC", i, row=row, col=col)
                fifo.send_l1a(2000)
                vth.append(i)
                count.append(rb_0.read_data_count(elink, slave=slave))
                print(i, rb_0.read_data_count(elink, slave=slave))
                rb_0.reset_data_error_count()

            vth_a = np.array(vth)
            count_a = np.array(count)
            vth_max = vth_a[np.argmax(count_a)]
            print(f"Found maximum count at DAC setting vth_max={vth_max}")

            vth     = []
            count   = []
            print("Fine scanning around this DAC value now")
            for i in range(vth_max-15, vth_max+15):
                #etroc.wr_reg("DAC", i, row=3, col=4)
                etroc.wr_reg("DAC", i, row=row, col=col)
                fifo.send_l1a(5000)
                vth.append(i)
                count.append(rb_0.read_data_count(elink, slave=slave))
                print(i, rb_0.read_data_count(elink, slave=slave))
                rb_0.reset_data_error_count()

            print(vth)
            print(count)

        elif args.pixelscan =="internal":
            row = int(args.row)
            col = int(args.col)

            elink, slave = etroc.get_elink_for_pixel(row, col)

            # turn off data readout for all pixels
            etroc.wr_reg("disDataReadout", 1, broadcast=True)
            etroc.wr_reg("disDataReadout", 0, row=row, col=col, broadcast=False)

            print(f"\n - Running internal threshold scan for pixel {row}, {col}")
            print(f"Found this pixel on elink {elink}, lpGBT is servant: {slave}")

            dac, res = vth_scan_internal(etroc, row=row, col=col, dac_min=0, dac_max=1000)
            slope = dac[((res>0) & (res<max(res)))]
            slope_vals = res[((res>0) & (res<max(res)))]
            ten_percent_occupancy = dac[(res/max(res)<0.1)][0]
            mid_slope = int(slope[int(len(slope)/2)])
            threshold = dac[res==0][2]  # take a DAC value a bit above the threshold
            print("Found the following results:")
            print(f"Threshold [DAC]: {threshold}")
            print(f"Mid-slope [DAC]: {mid_slope}")

            res_normalized = res / max(res)

            fig, ax = plt.subplots()
            plt.title(f"S-curve for pixel ({row},{col})")
            ax.plot(dac, res_normalized, '.-', color='blue', label='internal (acc)')

            # compare with external scan
            rb_0.reset_data_error_count()
            print("\n - Running external threshold scan on single pixel")
            vth     = []
            count   = []
            print("Fine scanning around the mid-slope DAC value now")
            rb_0.get_link_status(elink, slave=slave)
            for i in range(mid_slope-20, mid_slope+35):
                #etroc.wr_reg("DAC", i, row=3, col=4)
                etroc.wr_reg("DAC", i, row=row, col=col)
                fifo.send_l1a(5000)
                vth.append(i)
                c = rb_0.read_data_count(elink, slave=slave)
                print(i,c)
                count.append(c)
                rb_0.reset_data_error_count()

            count = np.array(count)
            count_normalized = count / max(count)

            ax.plot(vth, count_normalized, '.-', color='red', label='external')

            ax.set_xlim(mid_slope-20, mid_slope+20)
            ax.set_xlabel("DAC")
            ax.set_ylabel("normalized count")
            plt.legend()

            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fig.savefig(f'{result_dir}/scan_internal_row_{row}_col_{col}_{now}.png')

        if args.internal_data:
            # this still gives type == 0 data (with TOA, TOT, CAL)
            # but those should be random (?) values
            print("Setting ETROC in internal test data mode")
            etroc.wr_reg("workMode", 0x1, broadcast=True)  # this was missing
            etroc.wr_reg("selfTestOccupancy", 2, broadcast=True)

        if args.qinj_vth_scan:
            fifo.reset()
            delay = 10
            i = int(args.row)
            j = int(args.col)
            L1Adelay = 501
            
            if len(args.vth_axis) == 3:
                vth_axis = np.linspace(*args.vth_axis)
            elif len(args.vth_axis) < 3:
                vth_axis = range(*args.vth_axis)
            else:
                vth_axis    = np.linspace(415, 820, 406)
            #charges = [1,5,10,15,20,25,30,32]
            #charges = [5,10,15,20,25,30,32]
            charges = args.charges#[4,6,8,12]
            results =[[] for i in range(0,len(charges))]
            TOA = [[] for i in range(0,len(charges))]
            TOT =  [[] for i in range(0,len(charges))]
            CAL = [[] for i in range(0,len(charges))]
            k=0
            for q in charges:
                print(f"\n - Will send L1a/QInj pulse with delay of {delay} cycles and charge of {q} fC")
                print(f"\n - to pixel at Row {i}, Col {j}.")
                for vth in vth_axis:
                    worked = False
                    qinjatt = 0
                    while(not worked) and (qinjatt < 10):
                        try:
                            etroc.QInj_set(q, delay, L1Adelay, row=i, col=j, broadcast = False) #set reg on ETROC
                            etroc.wr_reg('DAC', int(vth), row=i, col=j, broadcast=False) #set vth on ETROC
                            fifo.send_QInj(count=int(args.nl1a), delay=504) #send Qinj pulses with L1Adelay
                            result = fifo.pretty_read(df)
                            worked = True
                        except KeyboardInterrupt:
                            print(f'Interupted by keyboard. Terminating run')
                            raise
                        except:
                            print(f'Attempt Number {qinjatt} failed. Trying again.')
                            print(traceback.format_exc())
                        qinjatt += 1
                    hits=0
                    toa=[]
                    tot=[]
                    cal=[]
                    for word in result:
                        if(word[0] == 'data'):
                          toa.append(int(word[1]['toa']))
                          tot.append(int(word[1]['tot']))
                          cal.append(int(word[1]['cal']))
                          
                        if(word[0] == 'trailer'):
                            hits+=word[1]['hits']
                    results[k].append(int(hits))
                    TOT[k].append(tot)
                    TOA[k].append(toa)
                    CAL[k].append(cal)
                k+=1

                scan_df = {'vth': list(vth_axis),
                           'hits': list(results[k-1]),
                           'toa' : TOA[k-1],
                           'tot' : TOT[k-1],
                           'cal' : CAL[k-1]}
                #with open(f"{out_dir}/Qinj_scan_L1A_504_{q}.pkl", 'w') as f:
                #    pickle.dump(scan_df, f)
                with open(f"{out_dir}/Qinj_scan_L1A_504_{q}.yaml", 'w') as f:
                    dump(scan_df, f)
            fig, ax = plt.subplots()

            plt.title("S curve for Qinj")
            plt.xlabel("Vth")
            plt.ylabel("hit rate")
            for i in range(0,len(charges)):
                plt.plot(vth_axis, results[i], '.-')
            
            #plt.xlim(410,820)
            plt.grid(True)
            plt.legend(loc='best')
            
            fig.savefig(f'{result_dir}/Scurve_Qinj.png')
            plt.close(fig)
            del fig, ax

        if args.qinj:
            etroc.reset()
            etroc.wr_reg("disDataReadout", 1, broadcast=True)

            N_pix   = 16*16 # total # of pixels
            N_pix_w = 16 # N_pix in NxN layout
            pixels = [
                (0,0),
                (15,15),
                (15,0),
                (0,15),
                (6,6),
                (6,9),
                (9,9),
                (9,6),
                (3,3),
                (3,12),
                (12,3),
                (12,12),
                (7,0),
                (8,0),
                (7,15),
                (8,15),
                (0,7),
                (0,8),
                (15,7),
                (15,8),
                (7,11),
                (3,15),
                (9,14),
            ]

            charge = int(args.charge) - 1
            for row, col in pixels:
                etroc.wr_reg("disDataReadout", 0, row=row, col=col, broadcast=False)
                etroc.wr_reg("QSel", charge, row=row, col=col)
                etroc.wr_reg("QInjEn", 1, row=row, col=col)

            fifo.reset()
            fifo_depth = 8192*2  # number of words the FIFO can hold
            events_max = int(fifo_depth/(len(pixels)+4))  # calculate the maximum number if l1as that the FIFO can handle
            # without getting drained in between
            print(f"Sending {events_max} L1As and charge injection pulses")
            fifo.send_QInj(events_max, delay=etroc.QINJ_delay)
            data = fifo.pretty_read(df)

            event_counter = 0
            hit_counter = 0
            hit_matrix = np.zeros([N_pix_w, N_pix_w])
            for x in data:
                if x[0] == 'header':
                    event_counter += 1
                if x[0] == 'data':
                    hit_counter += 1
                    hit_matrix[x[1]['row_id']][x[1]['col_id']] += 1

            print(f"Found {event_counter} headers")
            print(f"... with {hit_counter} hits total")

            fig, ax = plt.subplots()
            plt.title("Hits from charge injection")
            cax = ax.matshow(hit_matrix)

            fig.colorbar(cax)
            ax.set_xticks(np.arange(N_pix_w))
            ax.set_yticks(np.arange(N_pix_w))

            for i in range(N_pix_w):
                for j in range(N_pix_w):
                    text = ax.text(j, i, int(hit_matrix[i,j]),
                            ha="center", va="center", color="w", fontsize="xx-small")

            fig.savefig(f'{result_dir}/hit_matrix_qinj.png')
            plt.show()

            plt.close(fig)
            del fig, ax

            import struct
            fifo.reset()
            with open(f"{out_dir}/output_qinj_{args.charge}fC.dat", mode="wb") as f:
                for i in range(50):
                    fifo.send_QInj(1000, delay=etroc.QINJ_delay)
                    data = fifo.read(dispatch=True)
                    f.write(struct.pack('<{}I'.format(len(data)), *data))

        if args.qinj_scan:
            print("\n - Running scan for charge injection test now.")

            # counter based charge injection
            #q = 30
            #delay = 10
            row = 1
            col = 1

            # select the correct elink for the counter
            elink, slave = etroc.get_elink_for_pixel(row, col)

            # reset the counter
            rb_0.reset_data_error_count()

            # turn off data readout for all pixels
            etroc.wr_reg("disDataReadout", 1, broadcast=True)
            etroc.wr_reg("disDataReadout", 0, row=row, col=col, broadcast=False)

            # test the settings and get the proper threshold for the pixel,
            # using internal accumulator
            # NOTE: add a nice threshold histogram here?
            print("Running internal threshold scan for pixel under test")
            dac, res = vth_scan_internal(etroc, row=row, col=col, dac_min=0, dac_max=750)
            slope = dac[((res>0) & (res<max(res)))]
            slope_vals = res[((res>0) & (res<max(res)))]
            ten_percent_occupancy = dac[(res/max(res)<0.1)][0]
            mid_slope = int(slope[int(len(slope)/2)])
            threshold = dac[res==0][2]  # take a DAC value a bit above the threshold

            # check that everything actually works
            rb_0.reset_data_error_count()
            etroc.wr_reg("DAC", mid_slope, row=row, col=col)
            fifo.send_l1a(5000)
            hits = rb_0.read_data_count(elink, slave=slave)
            print(f"Found {hits} hits when sitting mid-slope and sending 5000 L1As")
            rb_0.reset_data_error_count()

            # set threshold to threshold
            print(f"Using found threshold at {dac[res==0][0]}, using value {dac[res==0][2]} for DAC.")
            etroc.wr_reg("DAC", int(threshold), row=row, col=col)

            # enable Q injection like the ETROC people
            etroc.wr_reg("QSel", 0x0e, row=row, col=col)
            etroc.wr_reg("QInjEn", 1, row=row, col=col)

            delays = []
            counts = []
            for i in range(500, 510, 1):
                rb_0.reset_data_error_count()
                fifo.send_QInj(5000, delay=i)
                hits = rb_0.read_data_count(elink, slave=slave)
                print(i, hits)
                delays.append(i)
                counts.append(hits)

            delays = np.array(delays)
            counts = np.array(counts)

            delay = delays[counts>1000][0]  # this should catch the rising edge

            fifo.reset()
            fifo.send_QInj(5000, delay=delay)
            data = fifo.pretty_read(df)

            cal_axis = hist.axis.Regular(2**10, 0, 2**10, name="cal", label="Cal code")
            toa_axis = hist.axis.Regular(1000, 0, 25, name="toa", label="TOA")
            toa_hist = hist.Hist(cal_axis)
            tot_hist = hist.Hist(cal_axis)
            cal_hist = hist.Hist(cal_axis)
            toa_code = []
            tot_code = []
            cal_code = []
            toa = []
            tot = []
            cal = []
            for x in data:
                if x[0] == 'data':
                    bin = 3.125/x[1]['cal']
                    toa_code.append(x[1]['toa'])
                    tot_code.append(x[1]['tot'])
                    cal_code.append(x[1]['cal'])
                    toa.append(12.5-bin * x[1]['toa'])
                    tot.append(x[1]['tot']*2 - np.floor(x[1]['tot']/32)*bin)

            print("\n - TOA_code:")
            toa_hist.fill(cal=toa_code)
            toa_min = np.mean(toa_code) - 10
            toa_max = np.mean(toa_code) + 10
            toa_hist[complex(0,toa_min):complex(0,toa_max):1j].show(columns=100)

            print("\n - TOT:")
            tot_hist.fill(cal=tot_code)
            tot_min = np.mean(tot_code) - 10
            tot_max = np.mean(tot_code) + 10
            tot_hist[complex(0,tot_min):complex(0,tot_max):1j].show(columns=100)

            print("\n - Cal Code:")
            cal_hist.fill(cal=cal_code)
            cal_min = np.mean(cal_code) - 20
            cal_max = np.mean(cal_code) + 20
            cal_hist[complex(0,cal_min):complex(0,cal_max):1j].show(columns=100)

        elif args.timing_scan:
            import pickle
            print("Running timing scan")
            rb_0.enable_external_trigger()
            rb_0.external_trigger_status()
            etroc.wr_reg("disDataReadout", 0, row=args.col, col=args.row, broadcast=False)
            etroc.wr_reg("DAC", 108, row=args.row, col=args.col)  # FIXME this should not be hard coded.
            isSlave = args.col<8 
            print( f"The pixel chosen is in the slave lpGBT? {isSlave}")
            #data = []
            results = []
            fifo.reset()
            rb_0.reset_data_error_count()
            #data_count = 0
            for j in range(0, 512):  # max delay is 511 bunch crossings
            #for j in range(230, 245):  # max delay is 511 bunch crossings
                data_count = 0
                trigger_count = 0
                data = []
                etroc.wr_reg("L1Adelay", j, broadcast=True)  # broadcast was missing before.
                for i in range(100):
                    #etroc.wr_reg("L1Adelay", 0x01f5)
                    if fifo.is_full():
                        print("Fifo is full!")
                        fifo.reset()
                    if rb_0.read_data_count(0, slave=isSlave):
                        print("There was a hit (or noise)")
                        data += fifo.pretty_read(df)
                        trigger_count += rb_0.read_packet_count(0, slave=isSlave)
                        data_count += rb_0.read_data_count(0, slave=isSlave)
                        rb_0.reset_data_error_count()
                        #data_count += rb_0.read_data_count(0, slave=isSlave)
                        #fifo.reset()
                results.append((j, data_count, trigger_count))
                print(j,data_count,trigger_count)

            now = time.time()
            with open(f"timing_scan_{now}.pkl", "wb") as f:
                pickle.dump(results, f)

            rb_0.disable_external_trigger()

    elif args.vth:
        # ==============================
        # ======= Test Vth scan ========
        # ==============================
        print("<--- Testing Vth scan --->")

        # run only if no saved data or we want to rerun
        if (not os.path.isfile("results/vth_scan.json")) or args.rerun:

            # scan
            print("No data. Run new vth scan...")
            result_data = vth_scan(ETROC2)

            #save
            if not os.path.isdir('results'):
                os.makedirs('results')

            with open("results/vth_scan.json", "w") as outfile:
                json.dump(result_data, outfile)
                print("Data saved to results/vth_scan.json\n")


        # read and parse vth scan data
        with open('results/vth_scan.json', 'r') as openfile:
            vth_scan_data = json.load(openfile)

        vth_axis = np.array(vth_scan_data[0])
        hit_rate = np.array(vth_scan_data[1])

        vth_min = vth_axis[0]  # vth scan range
        vth_max = vth_axis[-1]
        N_pix   = len(hit_rate) # total # of pixels
        N_pix_w = int(round(np.sqrt(N_pix))) # N_pix in NxN layout

        # ======= PERFORM FITS =======
        # fit to sigmoid and save to NxN layout
        slopes = np.empty([N_pix_w, N_pix_w])
        means  = np.empty([N_pix_w, N_pix_w])
        widths = np.empty([N_pix_w, N_pix_w])

        for pix in range(N_pix):
            fitresults = sigmoid_fit(vth_axis, hit_rate[pix])
            r, c = fromPixNum(pix, N_pix_w)
            slopes[r][c] = fitresults[0]
            means[r][c]  = fitresults[1]
            widths[r][c] = 4/fitresults[0]

        # print out results nicely
        for r in range(N_pix_w):
            for c in range(N_pix_w):
                pix = toPixNum(r, c, N_pix_w)
                print("{:8s}".format("#"+str(pix)), end='')
            print("")
            for c in range(N_pix_w):
                print("%4.2f"%means[r][c], end='  ')
            print("")
            for c in range(N_pix_w):
                print("+-%2.2f"%widths[r][c], end='  ')
            print("\n")

        # ======= PLOT RESULTS =======
        # fit results per pixel & save
        if args.fitplots:
            print('Creating plots and saving in ./results/...')
            print('This may take a while.')
            for expix in range(256):
                exr   = expix%N_pix_w
                exc   = int(np.floor(expix/N_pix_w))

                fig, ax = plt.subplots()

                plt.title("S curve fit example (pixel #%d)"%expix)
                plt.xlabel("Vth")
                plt.ylabel("hit rate")

                plt.plot(vth_axis, hit_rate[expix], '.-')
                fit_func = sigmoid(slopes[exr][exc], vth_axis, means[exr][exc])
                plt.plot(vth_axis, fit_func)
                plt.axvline(x=means[exr][exc], color='r', linestyle='--')
                plt.axvspan(means[exr][exc]-widths[exr][exc], means[exr][exc]+widths[exr][exc],
                            color='r', alpha=0.1)

                plt.xlim(vth_min, vth_max)
                plt.grid(True)
                plt.legend(["data","fit","baseline"])

                fig.savefig(f'results/pixel_{expix}.png')
                plt.close(fig)
                del fig, ax

        # 2D histogram of the mean
        fig, ax = plt.subplots()
        plt.title("Mean values of baseline voltage")
        cax = ax.matshow(means)

        fig.colorbar(cax)
        ax.set_xticks(np.arange(N_pix_w))
        ax.set_yticks(np.arange(N_pix_w))

        for i in range(N_pix_w):
            for j in range(N_pix_w):
                text = ax.text(j, i, "%.2f\n+/-%.2f"%(means[i,j],widths[i,j]),
                        ha="center", va="center", color="w", fontsize="xx-small")

        fig.savefig(f'results/sigmoid_mean_2D.png')
        plt.show()

        plt.close(fig)
        del fig, ax

        # 2D histogram of the width
        fig, ax = plt.subplots()
        plt.title("Width of the sigmoid")
        cax = ax.matshow(
            widths,
            cmap='RdYlGn_r',
            vmin=0, vmax=5,
        )

        fig.colorbar(cax)
        ax.set_xticks(np.arange(N_pix_w))
        ax.set_yticks(np.arange(N_pix_w))

        #cax.set_zlim(0, 10)

        fig.savefig(f'results/sigmoid_width_2D.png')
        plt.show()

    else:
        thresholds = [706-x*ETROC2.DAC_step for x in range(10)]
        print("Sending 10 L1As and reading back data, for the following thresholds:")
        print(thresholds)
        for th in thresholds:
            ETROC2.set_Vth_mV(th)  # anything between 196 and 203 should give reasonable numbers of hits
            print(f'Threshold at {ETROC2.get_Vth_mV(row=4, col=5)}mV')
            data = ETROC2.runL1A()  # this will spit out data for a single event, with an occupancy corresponding to the previously set threshold
            unpacked = [DF.read(d) for d in data]
            for d in data:
                print(DF.read(d))

        if unpacked[-1][1]['hits'] == len(unpacked)-2:
            print("Very simple check passed.")
            sys.exit(0)
        else:
            print("Data looks inconsistent.")
            sys.exit(1)
