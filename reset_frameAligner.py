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

    #print(system.rb.DAQ_LPGBT.rd_reg('LPGBT.RW.RESET.RSTFRAMEALIGNER'))

    system.rb.DAQ_LPGBT.wr_reg('LPGBT.RW.RESET.RSTFRAMEALIGNER', 0)

    time.sleep(1)

    system.rb.DAQ_LPGBT.wr_reg('LPGBT.RW.RESET.RSTFRAMEALIGNER', 1)

    time.sleep(1)

    system.rb.DAQ_LPGBT.wr_reg('LPGBT.RW.RESET.RSTFRAMEALIGNER', 0)

if __name__ == "__main__":
    main()
