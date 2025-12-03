#!/usr/bin/env python3
import argparse
import sys
from tamalero.Module import Module
from tamalero.ReadoutBoard import ReadoutBoard
from tamalero.utils import get_kcu
from daq import stream_daq

#IPB_PATH = "ipbusudp-2.0://192.168.0.10:50001?max_payload_size=1500"
# IPB_PATH = "ipbusudp-2.0://192.168.0.10:50001"
# ADR_TABLE = "./address_table/generic/etl_test_fw.xml"

def set_ETROC_occupancy(etroc, occ):
    assert occ < 128, "Occupancy value out of accepted range [0-127] (7bits)"
    assert type(occ)==int, "Occupancy value type not int"
    for i in range (0,16): 
        for j in range (0,16): 
            etroc.set_selftest_occupancy(occ=occ,row=i,col=j) 
    print ( "ETROC occupancy set to {}%".format(round(occ/1.28,2)) )

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--kcu', action='store', default='192.168.0.10', help="KCU address")
    argParser.add_argument('--rb', action='store', default=0, type=int, help="RB number (default 0)")
    argParser.add_argument('--moduleid', action='store', default=0,required=True, type=int, help="ID of the module under test")
    argParser.add_argument('--run_time', action='store', default=10, type=int, help="Time in [s] to take data")
    argParser.add_argument('--run', action='store', default=1, type=int, help="Run number")
    args = argParser.parse_args()
    
    occupancies=[1,5,10,40,70]
    rates=[100,500,1000,2000,5000]#,50000]
    
    rbID = int(args.rb)
    kcu = get_kcu(args.kcu)
    rb = ReadoutBoard(rbID,kcu=kcu, config="modulev0b")
    module = Module(rb=rb, i=1, enable_power_board=False, moduleid = args.moduleid)
    if module.ETROCs[0].is_connected():
        etroc = module.ETROCs[0]
    else: 
        print ("ETROC not connected")
        sys.exit(1)
    
    print("\nStarting the throughput test")
    for occ in occupancies:
        set_ETROC_occupancy(etroc, occ)
        for rate in rates:
            run="occ_{}_rate_{}".format(round(occ/128*100,1),rate)
            print ("\nTesting taking data at {} Hz".format(rate))
            kcu.write_node(f"READOUT_BOARD_{rbID}.EVENT_CNT_RESET", 0x1)
            f_out= None
            while f_out is None:
                try:
                    f_out = stream_daq(kcu, l1a_rate=rate, run_time=args.run_time, run=run)
                except:
                    print("Probably UHAL hang-up, trying again...")
                    pass

            print(f"Run {run} has ended.")
            print(f"Stored data in file: {f_out}")
            nevents = kcu.read_node(f"READOUT_BOARD_{rbID}.EVENT_CNT").value()
            print(f"Recorded {nevents=}")
