#!/usr/bin/env python3
import json
import ROOT as rt
from array import array
import os
import re
import time

def setVector(v_, l_):
    v_.clear()
    for i in l_:
        v_.push_back(i)

def dump_to_root(output, input_file):
    # Create an empty root file so that the merger step is always happy and does not get stuck
    filename = os.path.basename(input_file)
    name, ext = os.path.splitext(filename)
    if ext != '.json':
        raise ValueError("Inputted file needs to be json from data dumper")

    f = rt.TFile(output, "RECREATE")
    tree = rt.TTree("pulse", "pulse")
    print(output)

    if os.path.isfile(input_file):
        with open(input_file) as f_in:
            print("Now reading from {}".format(input_file))
            jsonString = json.load(f_in)
            jsonData = json.loads(jsonString)

            event_       = array('I',[0])
            l1counter_   = array('I',[0])
            row_         = rt.std.vector[int]()
            col_         = rt.std.vector[int]()
            tot_code_    = rt.std.vector[int]()
            toa_code_    = rt.std.vector[int]()
            cal_code_    = rt.std.vector[int]()
            elink_       = rt.std.vector[int]()
            #raw_         = rt.std.vector[rt.std.string]()
            #crc_         = rt.std.vector[int]()
            chipid_      = rt.std.vector[int]()
            #bcid_        = rt.std.vector[int]()
            bcid_        = array("I",[0]) # rt.std.vector[int]()
            #counter_a_   = rt.std.vector[int]()
            nhits_       = rt.std.vector[int]()
            nhits_trail_ = rt.std.vector[int]()

            tree.Branch("event",       event_, "event/I")
            tree.Branch("l1counter",   l1counter_, "l1counter/I")
            tree.Branch("row",         row_)
            tree.Branch("col",         col_)
            tree.Branch("tot_code",    tot_code_)
            tree.Branch("toa_code",    toa_code_)
            tree.Branch("cal_code",    cal_code_)
            tree.Branch("elink",       elink_)
            #tree.Branch("raw",         raw_)
            #tree.Branch("crc",         crc_)
            tree.Branch("chipid",      chipid_)
            tree.Branch("bcid",        bcid_, "bcid/I")
            #tree.Branch("counter_a",   counter_a_)
            # tree.Branch("nhits",       nhits_, "nhits/I")
            tree.Branch("nhits",       nhits_)
            tree.Branch("nhits_trail", nhits_trail_)

            for i, event in enumerate(jsonData):
                # print(event["bcid"])
                event_[0] =             event["event"]
                l1counter_[0] =         event["l1counter"]
                setVector(row_,         event["row"])
                setVector(col_,         event["col"])
                setVector(tot_code_,    event["tot_code"])
                setVector(toa_code_,    event["toa_code"])
                setVector(cal_code_,    event["cal_code"])
                setVector(elink_,       event["elink"])
                # setVector(raw_,         event["raw"])
                #setVector(crc_,         event["crc"])
                setVector(chipid_,      event["chipid"])
                # print(event["bcid"])
                bcid_[0] =              int(event["bcid"][0])
                # setVector(bcid_,        event["bcid"])
                #setVector(counter_a_,   event["counter_a"])
                setVector(nhits_,           event["nhits"])
                # setVector(nhits_trail_, event["nhits_trail"])

                tree.Fill()
    
        print(f"Found {i+1} events")
        f.WriteObject(tree, "pulse")
        print(f"Output written to {output} ...")
    else:
        print("-----File does not exist-----")

def get_run_number(path: str) -> int:
    pattern = r'output_run_(\d+)_rb0\.json'
    match = re.search(pattern, path)
    if match:
        return int(match.group(1))
    return -99999
    
if __name__ == '__main__':

    dump_to_root(
        "/home/etl/Test_Stand/ETL_TestingDAQ/unit_test/asserted_output/run_6000/output_run_6000_rb0.root", 
        "/home/etl/Test_Stand/ETL_TestingDAQ/unit_test/asserted_output/run_6000/output_run_6000_rb0.json"
    )

    # RUN_START = 10610
    # output_etroc_root = lambda run: f"/home/etl/Test_Stand/ETROC2_Test_Stand/ScopeHandler/ScopeData/ETROCData/output_run_{run}_rb0.root"
    # etroc_json_input_directory = "/home/etl/Test_Stand/module_test_sw/ETROC_output/"
    # force = False
    # run_minimum = RUN_START
    # while True:
    #     # first get all the files in the dirctory matching this
    #     print(f"Looking for any new etroc json files up to this run: {run_minimum}")
    #     etroc_json_files = sorted(
    #         [file for file in os.listdir(etroc_json_input_directory) if get_run_number(file) >= run_minimum], 
    #         key =get_run_number
    #     )

    #     if not etroc_json_files:
    #         time.sleep(3)
    #         continue

    #     print(f"Root dumping these files: {etroc_json_files}")
    #     for etroc_file in etroc_json_files:
    #         run_number = get_run_number(etroc_file)
    #         if os.path.exists(output_etroc_root(run_number)) and not force:
    #             print(f"ROOT DUMPED FILE ALREADY EXISTS ({run_number=}), passing this file!")
    #             continue
    #         print("-------------------------")
    #         print(f"DUMPING RUN: {run_number}")
    #         dump_to_root(
    #             output_etroc_root(run_number), 
    #             os.path.join(etroc_json_input_directory, etroc_file)
    #         )
    #         print("-------------------------")
    #         print("")

    #     run_minimum = get_run_number(etroc_json_files[-1])+1
    #     print(f"Moving run minimum to run_minimum: {run_minimum}")
        
            

