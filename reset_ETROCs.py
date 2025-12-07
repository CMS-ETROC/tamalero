import argparse
import time
from datetime import datetime, timedelta, timezone

# Tamalero imports
from tamalero.FIFO import FIFO
from tamalero.colors import green, yellow, red

# Local Module imports
from settings import DAQConfig
from hardware_init import ETROCSystem
from calibration import CalibrationManager
from data_handler import DataWriter
from daq_utils import TerminalHandler

def main():
    # 0. Parse Arguments
    parser = argparse.ArgumentParser(description='Reset lpGBT Frame Aligner')
    args = parser.parse_args()

    # 1. Setup Configuration
    # We initialize default config, then override with command line args
    config = DAQConfig()

    # 2. Initialize Hardware
    system = ETROCSystem(config)
    system.connect()

    #print(system.rb.DAQ_LPGBT)

    piodir = system.rb.DAQ_LPGBT.rd_reg('LPGBT.RWF.PIO.PIODIRL')
    piodir = piodir | 0x1111
    system.rb.DAQ_LPGBT.wr_reg('LPGBT.RWF.PIO.PIODIRL', piodir)
    #print(system.rb.DAQ_LPGBT.rd_reg('LPGBT.RWF.PIO.PIODIRL'))

    #print(system.rb.DAQ_LPGBT.rd_reg('LPGBT.RWF.PIO.PIOPULLENABLEL'))
    #print(system.rb.DAQ_LPGBT.rd_reg('LPGBT.RWF.PIO.PIODRIVESTRENGTHL'))

    state = system.rb.DAQ_LPGBT.rd_reg('LPGBT.RWF.PIO.PIOOUTL')

    state = state | 0x1111
    system.rb.DAQ_LPGBT.wr_reg('LPGBT.RWF.PIO.PIOOUTL', state)

    time.sleep(1)

    state = state & 0x11110000
    system.rb.DAQ_LPGBT.wr_reg('LPGBT.RWF.PIO.PIOOUTL', state)

    time.sleep(3)

    state = state | 0x1111
    system.rb.DAQ_LPGBT.wr_reg('LPGBT.RWF.PIO.PIOOUTL', state)

if __name__ == "__main__":
    main()
