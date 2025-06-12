#!/usr/bin/env python3

from tamalero.ETROC import ETROC
from tamalero.ETROC_Emulator import ETROC2_Emulator as software_ETROC2
from tamalero.DataFrame import DataFrame
from tamalero.utils import get_kcu
from tamalero.ReadoutBoard import ReadoutBoard
from tamalero.colors import red, green, yellow

import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import ROOT
import os
import sys
import json
import time
import pdb
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

DF = DataFrame('ETROC2')

if __name__ == '__main__':
    # argsparser
    import argparse
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--kcu', action='store', default='192.168.0.10', help="IP Address of KCU105 board")
    argParser.add_argument('--configuration', action='store', default='modulev0b', choices=['modulev0', 'modulev0b'], help="Board configuration to be loaded")
    argParser.add_argument('--module', action='store', default=0, choices=['1','2','3'], help="Module to test")
    argParser.add_argument('--host', action='store', default='localhost', help="Hostname for control hub")
    argParser.add_argument('--hard_reset', action='store_true', default=False, help="Hard reset of selected ETROC2 chip")
    argParser.add_argument('--scan', action='store', default=['full'], choices=['none', 'full', 'simple', 'internal'], help="Which threshold scan to run with ETROC2")
    argParser.add_argument('--mode', action='store', default=['dual'], choices=['dual', 'single'], help="Port mode for ETROC2")
    argParser.add_argument('--timing_scan', action='store_true', help="Set up internal data generation")
    argParser.add_argument('--enable_power_board', action='store_true', help="Enable Power Board (all modules). Jumpers must still be set as well.")
    argParser.add_argument('--row', action='store', default=4, help="Pixel row to be tested")
    argParser.add_argument('--col', action='store', default=3, help="Pixel column to be tested")
    argParser.add_argument('--offset', action='store', default=0, help="The offset from the baseline")
    args = argParser.parse_args()

    kcu = get_kcu(args.kcu, control_hub=True, host=args.host, verbose=False)
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

    from tamalero.Module import Module

    # FIXME the below code is still pretty stupid
    modules = []
    connected_modules = []
    for i in [1,2,3]:
        m_tmp = Module(rb=rb_0, i=i, enable_power_board=args.enable_power_board)
        modules.append(m_tmp)
        if m_tmp.ETROCs[0].connected:  # NOTE assume that module is connected if first ETROC is connected
            connected_modules.append(i)

    print(f"Found {len(connected_modules)} connected modules")
    if int(args.module) > 0:
        module = int(args.module)
    else:
        module = connected_modules[0]

    print(f"Will proceed with testing Module {module}")
    print("Module status:")
    modules[module-1].show_status()

    etroc = modules[module-1].ETROCs[0]
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

    # NOTE below is WIP code for tests of the actual data readout
    from tamalero.FIFO import FIFO
    from tamalero.DataFrame import DataFrame
    df = DataFrame()
    # NOTE this is for single port tests right now, where we only get elink 2
    fifo = FIFO(rb=rb_0)
    fifo.select_elink(0)
    fifo.ready()

    print("\n - Checking elinks")

    print("Disabling readout for all elinks but the ETROC under test")
    rb_0.disable_etroc_readout(all=True)
    rb_0.reset_data_error_count()
    #rb_0.enable_etroc_readout()
    for lpgbt in etroc.elinks:
        if lpgbt == 0:
            slave = False
        else:
            slave = True
        for link in etroc.elinks[lpgbt]:
            rb_0.enable_etroc_readout(link, slave=slave)
            #time.sleep(0.1)
            #rb_0.reset_data_error_count()
            rb_0.rerun_bitslip()
            time.sleep(1.5)
            rb_0.reset_data_error_count()
            stat = rb_0.get_link_status(link, slave=slave, verbose=False)
            if stat:
                rb_0.get_link_status(link, slave=slave)
            start_time = time.time()
            while not stat:
                #rb_0.disable_etroc_readout(link, slave=slave)
                rb_0.enable_etroc_readout(link, slave=slave)
                #time.sleep(0.1)
                #rb_0.reset_data_error_count()
                rb_0.rerun_bitslip()
                time.sleep(1.5)
                rb_0.reset_data_error_count()
                stat = rb_0.get_link_status(link, slave=slave, verbose=False)
                if stat:
                    rb_0.get_link_status(link, slave=slave)
                    break
                if time.time() - start_time > 2:
                    print('Link not good, but continuing')
                    rb_0.get_link_status(link, slave=slave)
                    break
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

    # with open('results/thresholds.yaml', 'r') as f:
    #     threshold_matrix = load(f, Loader)

    # with open('results/no_temperature_control/N13/185V/noise_scan/thresholds.yaml', 'r') as f:
    #     threshold_matrix = load(f, Loader)
    with open('results/noise_scan/thresholds.yaml', 'r') as f:
        threshold_matrix = load(f, Loader)

    rb_0.enable_external_trigger()
    offset_from_baseline = int(args.offset)
    etroc.wr_reg("disDataReadout", 1, broadcast=True)
    w = 1000
    scan_canvas = ROOT.TCanvas("Autocalibration", "Autocalibration")
    scan_canvas.SetCanvasSize(w, 2*w)
    baseline    = ROOT.TH2I("Autocalibration baseline", "Autocalibration baseline", 16, 0, 16, 16, -16, 0)
    noise_width = ROOT.TH2I("Noise width", "Noise width", 16, 0, 16, 16, -16, 0)
    ROOT.gStyle.SetOptStat(0)
    scan_canvas.Divide(1, 2)
    for i in range(16):
        for j in range(16):
            etroc.wr_reg("disDataReadout", 0, row=i, col=j, broadcast=False)
            # current_offset = etroc.get_THoffset(row = i, col = j)
            threshold = etroc.auto_threshold_scan(row=i, col=j, offset="auto")
            print(i+1, j+1, threshold)
            baseline.SetBinContent(j+1, i+1, threshold[0])
            noise_width.SetBinContent(j+1, i+1, threshold[1])
            etroc.wr_reg("disDataReadout", 1, row=i, col=j, broadcast=False)
    noise_width.GetZaxis().SetRangeUser(-2, 20)
    scan_canvas.cd(1)
    baseline.Draw("colz TEXT")
    scan_canvas.cd(2)
    noise_width.Draw("colz TEXT")
    scan_canvas.SaveAs("Autocalibration.png")
    L1Adelay = 14  # NOTE this is what we've found for the laser setup at FNAL.
    # We previously found a value of 17. Needs to be checked why it is different now.
    if args.timing_scan:
        import pickle
        print("Running timing scan")
        #data = []
        results = []
        fifo.reset()
        rb_0.reset_data_error_count()
        #data_count = 0
        #for j in range(0, 512):  # max delay is 511 bunch crossings
        for j in range(0, 40):  # max delay is 511 bunch crossings
            data_count = 0
            trigger_count = 0
            data = []
            etroc.wr_reg("L1Adelay", j, broadcast=True)  # broadcast was missing before.
            for i in range(1000):
                #etroc.wr_reg("L1Adelay", 0x01f5)
                if fifo.is_full():
                    print("Fifo is full!")
                    fifo.reset()
                if rb_0.read_data_count(0, slave=True) or rb_0.read_packet_count(0, slave=True):
                    #print("There was a hit (or noise)")
                    data += fifo.pretty_read(df)
                    trigger_count += rb_0.read_packet_count(0, slave=True)
                    data_count += rb_0.read_data_count(0, slave=True)
                    rb_0.reset_data_error_count()
                    #data_count += rb_0.read_data_count(0, slave=True)
                    #fifo.reset()
            
            results.append((j, data_count, trigger_count))
            print(j,data_count,trigger_count)
            if data_count>0:
                L1Adelay = j
                print(f"Found L1Adelay of {L1Adelay}")

    etroc.wr_reg("L1Adelay", L1Adelay, broadcast=True)
    rb_0.disable_external_trigger()
