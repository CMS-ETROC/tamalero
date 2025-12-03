#!/usr/bin/env python3

import time
import os
import copy
import glob
from emoji import emojize

from yaml import load
from yaml import CLoader as Loader, CDumper as Dumper
from tamalero.ReadoutBoard import ReadoutBoard
from tamalero.utils import get_kcu, load_yaml
from tamalero.FIFO import FIFO
from tamalero.DataFrame import DataFrame
from tamalero.Module import Module

if __name__ == '__main__':

    kcu = get_kcu('192.168.0.10', control_hub=True, verbose=True)
    rb = ReadoutBoard(rb=0, trigger=True, kcu=kcu, config='modulev1', verbose=False)

    rb.enable_external_trigger()

    module = Module(rb, i=1, poke=True)
    etroc = module.ETROCs[2]

    
    for i in range(4,30):
        for offset in [-2, -3, -5]:
            first = True
            start_time = time.time()
            print(f"Probing delay {i} with offset {offset}")
            
            #etroc.wr_reg("L1Adelay", i, broadcast=True)
            thresholds = load_yaml("/home/etl/Test_Stand/ETL_TestingDAQ/Test_Beam_Data/DebugTB/thresholds/noise_width_module_110_etroc_2.yaml")
            etroc.physics_config(
                offset = offset, 
                L1Adelay = i, 
                thresholds = thresholds, 
                out_dir = "/home/etl/Test_Stand/ETL_TestingDAQ/Test_Beam_Data/DebugTB/thresholds", 
                powerMode = "high")

            rb.reset_data_error_count()
            kcu.write_node(f"READOUT_BOARD_0.EVENT_CNT_RESET", 0x1)
            while kcu.read_node(f"READOUT_BOARD_0.EVENT_CNT").value()<5000:
                time.sleep(0.01)
                if kcu.read_node(f"READOUT_BOARD_0.EVENT_CNT").value()>1000 and first:
                    s=time.gmtime(time.time())
                    current=time.strftime("%Y-%m-%d %H:%M:%S", s)
                    print(f"{current} - Received 1000 L1As, still waiting...")
                    first=False
                if time.time() - start_time > 70:
                    print("Have been in the same setting for 1mins, breaking")
                    break


            data_count = rb.read_data_count(elink=0, slave=False)
            trigger_count = kcu.read_node(f"READOUT_BOARD_0.EVENT_CNT").value()
            if data_count > 0:
                print(f"!!! Found data for delay {i} !!!")
            print(i, data_count, trigger_count)
