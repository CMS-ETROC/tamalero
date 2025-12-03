#!/usr/bin/env python3
import time
from emoji import emojize
import argparse

from tamalero.ReadoutBoard import ReadoutBoard
from tamalero.utils import get_kcu

from cocina.PowerSupply import PowerSupply

from daq import stream_daq, stream_daq_multi

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--kcu', action='store', default='192.168.0.10', help="KCU address")
    argParser.add_argument('--l1a_rate', action='store', default=0, type=int, help="L1A rate in Hz")
    argParser.add_argument('--ext_l1a', action='store_true', help="Enable external trigger input")
    argParser.add_argument('--run_physics', action='store_true', help="Run on physics data / noise")
    argParser.add_argument('--run_time', action='store', default=10, type=int, help="Time in [s] to take data")
    argParser.add_argument('--run', action='store', default=1, type=int, help="Run number")
    args = argParser.parse_args()

    power_down = False
    kcu = args.kcu
    run_physics = args.run_physics
    run_time = args.run_time
    run = args.run
    l1a_rate = args.l1a_rate

    print("Getting the KCU")
    print(emojize(':atom_symbol:'), " Throughput test code draft")
    print(emojize(':battery:'), " Power Supply")
    psu1 = PowerSupply(ip='192.168.2.1', name='PS1')
    psu1.power_up('ch1')
    psu1.power_up('ch2')
    time.sleep(1)
    print(emojize(":check_mark_button:"), " Ready")

    kcu = get_kcu(kcu, control_hub=True, verbose=True)
    rb_0 = ReadoutBoard(0, kcu=kcu, config='modulev0b', verbose=False)
    rb_0.connect_modules()
    rb_0.modules[0].show_status()

    rb_1 = ReadoutBoard(1, kcu=kcu, config='modulev0b', verbose=False)
    rb_1.connect_modules()
    rb_1.modules[0].show_status()

    if run_physics:
        rb_0.modules[0].ETROCs[0].physics_config(offset='auto')  # works well for module 12
        rb_1.modules[0].ETROCs[0].physics_config(offset=4)  # offset of 3 gets us in trouble for module 25
    else:
        # NOTE double check if it's an occupancy issue for 50k events?
        rb_0.modules[0].ETROCs[0].test_config(occupancy=30)
        rb_1.modules[0].ETROCs[0].test_config(occupancy=30)

    rb_0.rerun_bitslip()
    rb_1.rerun_bitslip()

    rb_0.modules[0].show_status()
    rb_1.modules[0].show_status()

    stream_0 = stream_daq_multi(
        stream_daq,
        {'kcu':kcu, 'rb':0, 'l1a_rate':l1a_rate, 'run_time':run_time, 'run':run, 'ext_l1a':True},
    )

    stream_1 = stream_daq_multi(
        stream_daq,
        {'kcu':kcu, 'rb':1, 'l1a_rate':l1a_rate, 'run_time':run_time+0.2, 'run':run, 'ext_l1a':True},
    )

    print("Taking data")
    while stream_0._running or stream_1._running:
        time.sleep(1)
    print("Done with all streams")

    if power_down:
        psu1.power_down('ch1')
        psu1.power_down('ch2')
