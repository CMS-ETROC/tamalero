#!/usr/bin/env python3
#TODO: Check which commented code is still needed
import time
import argparse

from tamalero.ReadoutBoard import ReadoutBoard
from tamalero.utils import get_kcu
from tamalero.FIFO import FIFO
from tamalero.DataFrame import DataFrame
from tamalero.Module import Module
from tamalero.PixelMask import PixelMask
import yaml
import numpy as np

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--kcu', action='store', default='192.168.0.10', help="IP Address of KCU105 board")
    argParser.add_argument('--configuration', action='store', default='modulev0b', choices=['modulev0', 'modulev0b', 'modulev1'], help="Board configuration to be loaded")
    argParser.add_argument('--rbs', action='store', default='0', help="Which RBs to read from")
    argParser.add_argument('--modules', action='store', default='1', help="Modules to read from")
    argParser.add_argument('--etrocs', action='store', default='0', help="ETROCs to read from")
    argParser.add_argument('--change_timing', action='store', type=int, default=-1, help="ETROCs to read from")
    argParser.add_argument('--mask', action='store', default=None, help="Pixel mask to apply to a single ETROC")
    argParser.add_argument('--run', action='store', default=None, help="Run number for book keeping")
    argParser.add_argument('--temperature', action='store_true', help="Read temperature from ETROC")
    argParser.add_argument('--time', action='store_true', help="Show time stamp")
    argParser.add_argument('--dark_mode', action='store_true', help="Turn off all LEDs on the RB")
    argParser.add_argument('--light_mode', action='store_true', help="Turn on all LEDs on the RB")
    argParser.add_argument('--bitslip', action='store_true', help="Rerun the bitslip")
    argParser.add_argument('--reset_fifo', action='store_true', help="Rerun the bitslip")
    argParser.add_argument('--verbose', action='store_true', help="Rerun the bitslip")
    args = argParser.parse_args()

    rbs = [int(m) for m in args.rbs.split(',')]
    modules = [int(m) for m in args.modules.split(',')]
    etrocs = [int(m) for m in args.etrocs.split(',')]
    temps = []

    # this is the slowest part
    start_time = time.time()
    kcu = get_kcu(args.kcu, control_hub=True, verbose=False, quiet=True)
    kcu_time = time.time()

    for irb in rbs:
        rb = ReadoutBoard(rb=irb, trigger=True, kcu=kcu, config=args.configuration, verbose=False, poke=True)
        if args.temperature:
            for i in modules:
                module = Module(rb, i=i, poke=True)
                for j in etrocs:
                    etroc = module.ETROCs[j]
                    if etroc.is_connected():
                        try:
                            temp = etroc.check_temp()
                        except:
                            temp = -1
                        temps.append({'rb': irb, 'mod': etroc.module_id, 'etroc': etroc.chip_no, 'temp': temp})

        if args.dark_mode:
            rb.dark_mode()
        if args.light_mode:
            rb.light_mode()
        if args.bitslip:
            rb.rerun_bitslip()

    s=time.gmtime(time.time())
    current=time.strftime("%Y-%m-%d %H:%M:%S", s)
    ret_string = ''
    if args.time:
        ret_string += f'{current} '
    if args.run != None:
        ret_string += f'{args.run} '
    if args.temperature:
        ret_string += str(temps)
        print(ret_string)

    if args.mask:
        rb = ReadoutBoard(rb=rbs[0], trigger=True, kcu=kcu, config=args.configuration, verbose=False, poke=True)
        module = Module(rb, i=1, poke=True)
        etroc = module.ETROCs[2]

        with open(args.mask, 'r') as f:
            test = yaml.load(f)
        #mask = PixelMask(ar=np.array(test)==0)
        mask = PixelMask(ar=np.array(test)<8)
        mask.show()
        masked_pixels = mask.get_masked_pixels()
        etroc.deactivate_hot_pixels(pixels=masked_pixels)
        for row, col in masked_pixels:
            etroc.wr_reg("DAC", 1023, row=row, col=col, broadcast=False)
        #etroc.deactivate_hot_pixels([(7,5),(8,5),(10,6),(7,15),(14,8)])

    if args.change_timing>0:
        timing = int(args.change_timing)
        rb = ReadoutBoard(rb=0, trigger=True, kcu=kcu, config=args.configuration, verbose=False, poke=True)
        module = Module(rb, i=1, poke=True)
        etroc = module.ETROCs[2]
        delay_now = etroc.rd_reg("L1Adelay")
        print(f"Changing L1Adelay from {delay_now} to {timing}.")
        etroc.disable_data_readout(row=4, col=14, broadcast=False)
        etroc.wr_reg("L1Adelay", timing, broadcast=True)

    if args.reset_fifo:
        fifo = FIFO(rb)
        df = DataFrame("ETROC2")
        fifo.reset()
        fifo.send_l1a(2)
        fifo.reset()

    final_time = time.time()

    if args.verbose:
        print("Total time:", final_time-start_time)
        print("- KCU:", kcu_time-start_time)

    #print(etroc.rd_reg('DAC', row=10, col=10))
    #etroc.run_threshold_scan(use=False)
    #print(etroc.rd_reg('DAC', row=10, col=10))
    #etroc.run_threshold_scan(use=True)
    #print(etroc.rd_reg('DAC', row=10, col=10))
