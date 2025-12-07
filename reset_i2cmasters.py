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
    parser = argparse.ArgumentParser(description='Reset I2C Masters')
    parser.add_argument('-a', '--all', action='store_true', help='Reset all lpGBT I2C masters')
    parser.add_argument('-0', '--mzero', action='store_true', help='Reset lpGBT I2C master 0')
    parser.add_argument('-1', '--mone', action='store_true', help='Reset lpGBT I2C master 1')
    parser.add_argument('-2', '--mtwo', action='store_true', help='Reset lpGBT I2C master 2')
    args = parser.parse_args()

    # 1. Setup Configuration
    # We initialize default config, then override with command line args
    config = DAQConfig()

    # 2. Initialize Hardware
    system = ETROCSystem(config)
    system.connect()

    # print(system.rb.DAQ_LPGBT)

    # print(system.rb.DAQ_LPGBT.rd_reg('LPGBT.RW.RESET.RSTI2CM0'))

    # Need to pulse 0 -> 1 - > 0 to trigger reset
    if args.all or args.mzero:
        system.rb.DAQ_LPGBT.wr_reg('LPGBT.RW.RESET.RSTI2CM0', 0)
    if args.all or args.mone:
        system.rb.DAQ_LPGBT.wr_reg('LPGBT.RW.RESET.RSTI2CM1', 0)
    if args.all or args.mtwo:
        system.rb.DAQ_LPGBT.wr_reg('LPGBT.RW.RESET.RSTI2CM2', 0)

    time.sleep(1)

    if args.all or args.mzero:
        system.rb.DAQ_LPGBT.wr_reg('LPGBT.RW.RESET.RSTI2CM0', 0b100)
    if args.all or args.mone:
        system.rb.DAQ_LPGBT.wr_reg('LPGBT.RW.RESET.RSTI2CM1', 0b010)
    if args.all or args.mtwo:
        system.rb.DAQ_LPGBT.wr_reg('LPGBT.RW.RESET.RSTI2CM2', 0b001)

    time.sleep(1)

    if args.all or args.mzero:
        system.rb.DAQ_LPGBT.wr_reg('LPGBT.RW.RESET.RSTI2CM0', 0)
    if args.all or args.mone:
        system.rb.DAQ_LPGBT.wr_reg('LPGBT.RW.RESET.RSTI2CM1', 0)
    if args.all or args.mtwo:
        system.rb.DAQ_LPGBT.wr_reg('LPGBT.RW.RESET.RSTI2CM2', 0)

if __name__ == "__main__":
    main()
