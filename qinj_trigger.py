#!/usr/bin/env python3
import argparse
import time
from tamalero.utils import get_kcu

def run_qinj(kcu, rb=0, max_pulse_time=4, cycle_time=60, max_pulses=10000):
    start_time = time.time()
    qinj_counter = 0

    now = time.time()
    while now - start_time < max_pulse_time and qinj_counter < max_pulses:
        kcu.write_node("READOUT_BOARD_%s.L1A_QINJ_PULSE"%rb, 0x01)
        qinj_counter += 1
        now = time.time()

    print(f"Sent {qinj_counter} Qinj pulses in {now-start_time} seconds")
    print(f"Sleeping for {cycle_time - now + start_time} seconds now")
    time.sleep(cycle_time - (now - start_time))

def run_trigger(kcu, rb=0, max_pulse_time=4, cycle_time=60, max_pulses=10000):
    start_time = time.time()
    qinj_counter = 0
    now = time.time()
    while now - start_time < max_pulse_time and qinj_counter < max_pulses:
        kcu.write_node("READOUT_BOARD_%s.L1A_PULSE"%rb, 0x01)
        qinj_counter += 1
        now = time.time()
    print(f"Sent {qinj_counter} L1A in {now-start_time} seconds")
    print(f"Sleeping for {now - start_time} seconds now")
    time.sleep(cycle_time - (now - start_time))

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--kcu', action='store', default='192.168.0.10', help="KCU address")
    argParser.add_argument('--rb', action='store', default=0, type=int, help="RB number (default 0)")
    argParser.add_argument('--cycles', action='store', default=2, type=int, help="Number of cycles")
    argParser.add_argument('--cycle_time', action='store', default=60, type=int, help="Cycle time in [s]")
    argParser.add_argument('--pulses', action='store', default=10000, type=int, help="Number of pulses")
    argParser.add_argument('--pulse_time', action='store', default=4, type=int, help="Maximum pulse time in [s]")
    argParser.add_argument('--trigger', action='store_true', help="Run pure L1A mode")
    args = argParser.parse_args()

    rb = int(args.rb)
    kcu = get_kcu(args.kcu)
    kcu.write_node("READOUT_BOARD_%s.L1A_INJ_DLY"%rb, 504)

    for i in range(args.cycles):
        run_qinj(kcu, rb=rb, cycle_time=args.cycle_time, max_pulses=args.pulses, max_pulse_time=args.pulse_time)
