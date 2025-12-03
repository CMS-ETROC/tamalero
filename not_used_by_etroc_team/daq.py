#!/usr/bin/env python3
import struct
import os
import uhal
import argparse
import time
from tamalero.utils import get_kcu
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

here = os.path.dirname(os.path.abspath(__file__))

class MultiThread:
    def __init__(self, fun, args):
        self._running = True
        self.fun = fun
        self.args = args

    def terminate(self):
        self._running = False

    def run(self, sleep=60):
        while self._running and True:
            self.fun()
            time.sleep(sleep)

    def run_limited(self, iterations=1):
        for it in range(iterations):
            self.fun(**self.args)
        self._running = False

def stream_daq_multi(fun, args):
    from threading import Thread
    mon = MultiThread(fun, args)
    t = Thread(target = mon.run_limited, args=(1,))
    t.start()
    return mon

def get_kcu_flag(lock=os.path.expandvars('../ScopeHandler/Lecroy/Acquisition/running_acquitision.txt')):
    # NOTE where to put the locks?
    with open(lock) as f:
        res = f.read()
    return res.rstrip()
    #return open(f"/home/daq/ETROC2_Test_Stand/ScopeHandler/Lecroy/Acquisition/running_acquitision.txt").read()

def write_run_done(log=os.path.expandvars(here+'/daq_log.txt'), run=0):
    with open(log, 'a') as f:
        f.write(f'{run}\n')
    return run

def get_occupancy(hw, rb):
    try:
        occupancy = hw.getNode(f"READOUT_BOARD_{rb}.RX_FIFO_OCCUPANCY").read()
        hw.dispatch()
        occ = occupancy.value()
    except uhal._core.exception:
        print("uhal UPDP error when trying to get occupancy. Returning 0.")
        occ = 0
    return occ * 4  # not sure where the factor of 4 comes from, but it's needed

def stream_daq(kcu=None, rb=0, l1a_rate=0, run_time=10, n_events=1000, superblock=100, block=128, run=1, ext_l1a=False, lock=None, verbose=False):
    uhal.disableLogging()
    hw = kcu.hw
    rate_setting = l1a_rate / 25E-9 / (0xffffffff) * 10000

    print(f"Start data taking with rb {rb}")
    # reset fifo
    hw.getClient().write(hw.getNode(f"READOUT_BOARD_{rb}.FIFO_RESET").getAddress(), 0x1)
    hw.dispatch()
    kcu.write_node("SYSTEM.L1A_PULSE", 2)
    time.sleep(0.05)
    hw.getClient().write(hw.getNode(f"READOUT_BOARD_{rb}.FIFO_RESET").getAddress(), 0x1)
    hw.dispatch()
    kcu.write_node(f"READOUT_BOARD_{rb}.EVENT_CNT_RESET", 0x1)

    # set l1a rate
    hw.getNode("SYSTEM.L1A_RATE").write(int(rate_setting))
    hw.dispatch()

    time.sleep(0.05)

    if ext_l1a:
        # enable external trigger
        hw.getNode("SYSTEM.EN_EXT_TRIGGER").write(0x1)
        hw.dispatch()

    log = {}
    start = time.time()
    log['start_time'] = start
    len_data = 0
    data = []
    occupancy = 0
    f_out = f"ETROC_output/output_run_{run}_rb{rb}.dat"
    log_out = f"ETROC_output/log_run_{run}_rb{rb}.yaml"
    #f_out = f"output/output_rb_{rb}_run_{run}_time_{start}.dat"  # USED TO BE THIS, keeping for reference and debugging
    occupancy_block = []
    reads = []
    with open(f_out, mode="wb") as f:
        if lock is not None:
            # External lock file based DAQ
            iteration = 0
            Running = get_kcu_flag(lock=lock)
            while (Running.lower() == "false" or Running.lower() == "stop"):
                if iteration == 0:
                    print("Waiting for the start command.")
                Running = get_kcu_flag(lock=lock)
                iteration += 1

            print("Start data taking")
            Running = get_kcu_flag(lock=lock)
            while (Running.lower() != "false" and Running.lower() != "stop"):
                Running = get_kcu_flag(lock=lock)
                num_blocks_to_read = 0
                occupancy = get_occupancy(hw, rb)
                num_blocks_to_read = occupancy // block
                occupancy_block.append(num_blocks_to_read)

                # read the blocks
                if (num_blocks_to_read)>0:
                    try:
                        for x in range(num_blocks_to_read):
                            reads += [hw.getNode(f"DAQ_RB{rb}").readBlock(block)]
                            hw.dispatch()  # NOTE this is necessary
                    except uhal._core.exception:
                        print("uhal UDP error in reading FIFO")

        else:
            while (start + run_time > time.time()):
                # Time based DAQ
                # We read the data into memory (small)
                # and write to disk once we're done
                num_blocks_to_read = 0
                occupancy = get_occupancy(hw, rb)
                num_blocks_to_read = occupancy // block
                occupancy_block.append(num_blocks_to_read)

                # read the blocks
                if (num_blocks_to_read)>0:
                    #if num_blocks_to_read>1 and verbose:
                    #    print(occupancy, num_blocks_to_read)
                    try:
                        for x in range(num_blocks_to_read):
                            reads += [hw.getNode(f"DAQ_RB{rb}").readBlock(block)]
                            hw.dispatch()  # NOTE this is necessary
                    except uhal._core.exception:
                        print("uhal UDP error in reading FIFO")

        print("Resetting L1A rate back to 0")
        hw.getNode("SYSTEM.L1A_RATE").write(0)
        hw.dispatch()
        print(f"Done with data taking with rb {rb}")
        
        # Read data that might still be in the FIFO
        occupancy = get_occupancy(hw, rb)
        print(f"Occupancy before last read: {occupancy}")
        num_blocks_to_read = occupancy // block
        remainder = occupancy % block
        if (num_blocks_to_read)>0:
            for x in range(num_blocks_to_read):
                reads += [hw.getNode(f"DAQ_RB{rb}").readBlock(block)]
                hw.dispatch()
            reads += [hw.getNode(f"DAQ_RB{rb}").readBlock(remainder)]
        else:
            reads += [hw.getNode(f"DAQ_RB{rb}").readBlock(occupancy)]

        hw.dispatch()
        for i, read in enumerate(reads):
            data+= read.value()

        len_data += len(data)
        # The two prints below can be useful to check if we miss the first or last event
        # when we convert in the following steps
        #print(data[:10])
        #print(data[-10:])

        # The FIFO should be empty by now, but we do check another time
        occupancy = get_occupancy(hw, rb)
        while occupancy>0:
            print("Found stuff in FIFO. This should not have happened!")
            num_blocks_to_read = occupancy // block
            last_block = occupancy % block
            if (num_blocks_to_read or last_block):
                reads = []
                for x in range(num_blocks_to_read):
                    reads += [hw.getNode(f"DAQ_RB{rb}").readBlock(block)]
                    hw.dispatch()
                reads += [hw.getNode(f"DAQ_RB{rb}").readBlock(last_block)]
                hw.dispatch()
                for read in reads:
                    data += read.value()
            occupancy = get_occupancy(hw, rb)

        len_data += len(data)

        # Get some stats
        timediff = time.time() - start
        speed = 32*len_data  / timediff / 1E6
        occupancy = hw.getNode(f"READOUT_BOARD_{rb}.RX_FIFO_OCCUPANCY").read()
        lost = hw.getNode(f"READOUT_BOARD_{rb}.RX_FIFO_LOST_WORD_CNT").read()
        rate = hw.getNode(f"READOUT_BOARD_{rb}.PACKET_RX_RATE").read()
        l1a_rate_cnt = hw.getNode("SYSTEM.L1A_RATE_CNT").read()
        hw.dispatch()

        nevents = kcu.read_node(f"READOUT_BOARD_{rb}.EVENT_CNT").value()

        l1a_rate = l1a_rate_cnt.value()/1000.0
        occ = occupancy.value()
        lost_events = lost.value()
        rate_log = rate.value()
        print("L1A rate = %f kHz" % (l1a_rate))
        print("Occupancy = %d words" % occ)
        print("Number of events = %d"%nevents)
        print("Lost events = %d events" % lost_events)
        print("Packet rate = %d Hz" % rate_log)
        print("Speed = %f Mbps" % speed)

        # Actually write to disk
        f.write(struct.pack('<{}I'.format(len(data)), *data))

    hw.getClient().write(hw.getNode(f"READOUT_BOARD_{rb}.FIFO_RESET").getAddress(), 0x1)
    hw.dispatch()

    if ext_l1a:
        # disable external trigger again
        hw.getNode("SYSTEM.EN_EXT_TRIGGER").write(0x0)
        hw.dispatch()

    log['l1a_rate'] = l1a_rate
    log['occupancy'] = occ
    log['nevents'] = nevents
    log['lost_events'] = lost_events
    log['rate'] = rate_log
    log['speed'] = speed
    log['stop_time'] = time.time()

    with open(log_out, 'w') as f:
        dump(log, f)

    print(f"Data stored in {f_out}\n")
    write_run_done(run=run)

    return f_out

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--kcu', action='store', default='192.168.0.10', help="KCU address")
    argParser.add_argument('--rb', action='store', default=0, help="RB numbers (default 0)")
    argParser.add_argument('--l1a_rate', action='store', default=0, type=int, help="L1A rate in Hz")
    argParser.add_argument('--ext_l1a', action='store_true', help="Enable external trigger input")
    argParser.add_argument('--run_time', action='store', default=10, type=int, help="Time in [s] to take data")
    argParser.add_argument('--n_events', action='store', default=1000, type=int, help="N events")
    argParser.add_argument('--lock', action='store', default=None, help="Lock file for the scope acquisition status (relative or absolute path)")
    argParser.add_argument('--run', action='store', default=1, type=int, help="Run number")
    args = argParser.parse_args()

    start_time = time.time()

    kcu = get_kcu(args.kcu)

    rbs = [int(x) for x in args.rb.split(',')]

    # scanning the RBs can cause problems with the trigger link, not exactly sure why.
    # therefore, it's safer to just give a list of RBs that are connected
    #rb = int(args.rb)
    #rbs = []
    #for i in range(5):
    #    # we can at most connect 5 RBs
    #    try:
    #        rbs.append(ReadoutBoard(i, kcu=kcu, config='modulev0b', verbose=False))
    #        print(f"Added RB #{i}")
    #    except:
    #        pass

    #if len(rbs) == 0:
    #    print (f"No RB connect to {args.kcu}. Exiting.")
    #    exit()

    #print(f"Resetting global event counter of RB #{rb}")
    #kcu.write_node(f"READOUT_BOARD_{rb}.EVENT_CNT_RESET", 0x1)

    print(f"Preparing DAQ streams.\n ...")

    streams = []
    for rb in rbs:
        streams.append(
            stream_daq_multi(
                stream_daq,
                {
                    'kcu':kcu,
                    'rb':rb,
                    'l1a_rate':args.l1a_rate,
                    'run_time':args.run_time,
                    'run':args.run,
                    'ext_l1a':args.ext_l1a,
                    'lock': args.lock,
                    'verbose': True,
                },
            )
        )

    init_time = time.time()
    print(f"Init of ETROC DAQ took {round(init_time-start_time, 1)}s")
    print("Taking data now")
    while any([stream._running for stream in streams]):
       # stream_0._running or stream_1._running:
        time.sleep(1)
    print("Done with all streams")

    print(f"Run {args.run} has ended.")
    ## NOTE this would be the place to also dump the ETROC configs
