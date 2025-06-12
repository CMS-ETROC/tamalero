#!/usr/bin/env python3
import subprocess
import time
from pathlib import Path
from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import shutil

if __name__ == '__main__':
    which_root = shutil.which("root")
    if which_root == "/cvmfs/sft.cern.ch/lcg/releases/LCG_103/ROOT/6.28.00/x86_64-ubuntu2004-gcc9-opt/bin/root":
        print("Found expected ROOT version")
    else:
        print("This script relies on pyROOT, but found unexpected root version.")
        print(" > In case of issues, run:")
        print(" > source /cvmfs/sft.cern.ch/lcg/releases/LCG_103/ROOT/6.28.00/x86_64-ubuntu2004-gcc9-opt/ROOT-env.sh")

    from root_dumper import dump_to_root
    from data_dumper import data_dumper

    skip_stageout = True
    td02_dir = '/home/daq/ETROC_output/'
    data_dir = './ETROC_output/'

    while True:
        with open('daq_log.txt', 'r') as f:
            available_runs = f.readlines()
            available_runs = [int(x) for x in available_runs]

        Path('process_log.txt').touch()
        with open('process_log.txt', 'r') as f:
            processed_runs = f.readlines()
            processed_runs = [int(x) for x in processed_runs]

        for run in available_runs:
            if run not in processed_runs:

                print(f"\n\n >>> Starting to process run {run} <<<")
                print(f" > Trying to load DAQ log")
                try:
                    with open(data_dir + f"/log_run_{run}_rb0.yaml", "r") as f:
                        log = load(f, Loader=Loader)
                    print(f" > Nevents according to log: {log['nevents']}")
                    if log['lost_events'] > 0:
                        print(" > Lost events detected, will not process the data.")
                        with open('process_log.txt', 'a') as f:
                            f.write(f'{run}\n')
                        continue
                    if log['nevents'] < 1:
                        print(" > Empty run detected, can't process any data.")
                        with open('process_log.txt', 'a') as f:
                            f.write(f'{run}\n')
                        continue
                except FileNotFoundError:
                    print("Couldn't find log")
                print(f" > Converting binary to json")
                n_events, events = data_dumper(
                    f"{data_dir}/output_run_{run}_rb0.dat",
                    skip_trigger_check=True,
                )
                continue_processing = n_events==log['nevents']
                #print("Number of L1A and events in agreement?", continue_processing)

                if continue_processing:
                    #subprocess.call(f"python3 data_dumper.py --input {run} --rbs 0 --skip_trigger_check", shell=True)
                    outfile = f'ETROC_merged_run_{run}.root'
                    print(f" > Converting json to root")
                    dump_to_root(f'{data_dir}/{outfile}', f'{data_dir}/output_run_{run}_rb0.json')
                else:
                    print(" ! Data and number of L1As not in agreement, did not further process!")
                if not skip_stageout:
                    print(f" > Stage out root file.")
                    subprocess.call(f"scp {data_dir}/{outfile} daq@timingdaq02.dhcp.fnal.gov:{td02_dir}/{outfile}", shell=True)
                    #print(f" > Backup of raw data to EOS")
                with open('process_log.txt', 'a') as f:
                    f.write(f'{run}\n')
                print(f" > Done.")
        print("Done with all runs, sleeping for 1min")
        time.sleep(60)
