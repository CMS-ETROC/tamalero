#!/usr/bin/env python3
from tamalero.FIFO import FIFO
from tamalero.ETROC import ETROC
from tamalero.LPGBT import LPGBT
from tamalero.utils import get_kcu
from tamalero.DataFrame import DataFrame
from tamalero.colors import green, red, yellow
from tamalero.ReadoutBoard import ReadoutBoard
import os
import sys
import tty
import time
import select
import pickle
import termios
import numpy as np
from tqdm import tqdm
from random import randint
from datetime import datetime, timezone

KCU_IP = "192.168.0.10" ## If your KCU ip is diff, modify it.

READOUTBOARD_ID = 0
READOUTBOARD_CONFIG = 'default'

ETROC_I2C_ADDRESSES = [0x60, 0x61, 0x62, 0x63]
ETROC_I2C_CHANNEL = 1
ETROC_ELINKS_MAP = {0: [0, 4, 8, 12]}

# Test parameters
TH_OFFSET = 50              # Threshold offset above baseline
TRIGGER_ENABLE_MASK = 0x2
TRIGGER_DATA_SIZE = 1
TRIGGER_DELAY_SEL = 472

CHARGE_FC = 30 
QINJ_COUNT = 1000

PIXEL_ROW = 2
PIXEL_COL = 2
NUM_ETROC = len(ETROC_I2C_ADDRESSES)

stop_acquisition = False
cosmic_data = []
hit_counter = 0

def setup_terminal():
    """Setup terminal for non-blocking input"""
    try:
        # Save old terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        tty.cbreak(sys.stdin.fileno())
        return old_settings
    except:
        return None

def restore_terminal(old_settings):
    """Restore terminal settings"""
    try:
        if old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    except:
        pass

def check_for_quit():
    """Check if user pressed 'q' to quit"""
    global stop_acquisition
    try:
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            char = sys.stdin.read(1)
            if char.lower() == 'q':
                print(yellow("\nUser pressed 'q', stopping cosmic ray detection..."))
                stop_acquisition = True
                return True
    except:
        pass
    return False

# def save_to_pickle():
#     data_to_save = {
#     'baseline': baseline_storage,
#     'NW' : nw_storage
#     }

#     with open('etroc_NW_scan.pkl', 'wb') as f:
#         pickle.dump(data_to_save)

#     print('Data saved')


def main():
    global stop_acquisition, cosmic_data, hit_counter
    
    print('ETROC COSMIC RAY TEST - HARDWARE INITIALIZATION')
    
    # ======================================================================================
    # 1. INITIALIZE KCU
    # ======================================================================================

    kcu = get_kcu(
        KCU_IP,
        control_hub=True,
        host='localhost',
        verbose=False
    )
    print(green("Successfully connected to KCU."))

    kcu.status() # Prints LpGBT link statuses from KCU 
    # fw_ver = kcu.get_firmware_version() #
    # kcu.check_clock_frequencies() # Verifies KCU clock stability

    # Perform a simple loopback register test to confirm communication
    loopback_val = 0xABCD1234
    kcu.write_node("LOOPBACK.LOOPBACK", loopback_val) #
    read_val = kcu.read_node("LOOPBACK.LOOPBACK").value()

    if read_val == loopback_val:
        print(green(f"KCU Loopback test PASSED: Wrote 0x{loopback_val:X}, Read 0x{read_val:X}"))
    else:
        print(red(f"KCU Loopback test FAILED: Wrote 0x{loopback_val:X}, Read 0x{read_val:X}"))

    # ======================================================================================
    # 2. INITIALIZE READOUT BOARD
    # ======================================================================================
    rb = ReadoutBoard(
        rb=READOUTBOARD_ID,
        kcu=kcu,
        config=READOUTBOARD_CONFIG, 
        trigger=False,     
        verbose=False
    )
    print(green(f"Readout Board version detected: {rb.ver}"))

    # ======================================================================================
    # 3. INITIALIZE FOUR ETROC
    # ======================================================================================

    print("\n3. Initializing ETROC chips...")
    etroc_chips = []
    chip_names = []

    for i, addr in enumerate(ETROC_I2C_ADDRESSES):
        chip_name = f"Chip{i+1}"
        chip_names.append(chip_name)
        
        print(f"\nInitializing {chip_name} (I2C: 0x{addr:02X})...")
        
        try:
            etroc = ETROC(
                rb,
                master='lpgbt',
                i2c_adr=addr,
                i2c_channel=ETROC_I2C_CHANNEL,
                elinks=ETROC_ELINKS_MAP,
                strict=False,
                verbose=False
            )
            etroc_chips.append(etroc)
            
            # Verify communication
            if etroc.is_connected():
                # Check key registers
                scrambler_status = etroc.rd_reg("disScrambler")
                controller_state = etroc.rd_reg("controllerState")
                pll_unlock_count = etroc.rd_reg("pllUnlockCount")
                
                print(green(f"✓ {chip_name} connected successfully"))
                print(f"  Controller state: {controller_state} (should be 11)")
                print(f"  PLL unlock count: {pll_unlock_count}")
                
                if scrambler_status == 1:
                    print(green("  ✓ Register communication verified"))
                else:
                    print(red("  ✗ Register communication issue"))
            else:
                print(red(f"✗ {chip_name} not responding"))
                
        except Exception as e:
            print(red(f"✗ Failed to initialize {chip_name}: {e}"))
            etroc_chips.append(None)

    print(green("\n✓ Hardware initialization completed successfully!"))
    print(f"Initialized {len([c for c in etroc_chips if c is not None])} ETROC chips")
    print("\nETROC to E-link Mapping:")
    elink_list = ETROC_ELINKS_MAP[0]  # [0, 4, 8, 12]
    for i, (addr, elink) in enumerate(zip(ETROC_I2C_ADDRESSES, elink_list)):
        chip_name = f"Chip{i+1}"
        if i < len(etroc_chips) and etroc_chips[i] is not None:
            status = "✓ Connected"
        else:
            status = "✗ Failed"
        print(f"  {chip_name} (I2C: 0x{addr:02X}) <-> E-link {elink} - {status}")

    
    # ======================================================================================
    # 4. COSMIC RAY DETECTION SETUP
    # ======================================================================================
    
    print("\nETROC COSMIC RAY TEST - CONTINUOUS DETECTION")
    
    print(f"2. Calibrating {PIXEL_ROW * PIXEL_COL} pixel baselines...")
    
    # Store baseline for all 256 pixels (16x16 array)
    baseline_storage = {}
    etroc_configs = []
    failed_pixels = {}
    nw_storage ={}

    def save_to_pickle():
        data_to_save = {
        'baseline': baseline_storage,
        'NW' : nw_storage
        }

        with open('etroc_NW_scan.pkl', 'wb') as f:
            pickle.dump(data_to_save,f)

        print('Data saved')

    print("\n3. Generating test pixel configuration...")
    all_pixels_per_chip = []
    for _ in range(NUM_ETROC):
        chip_pixels = []
        for row in range(PIXEL_ROW):
            for col in range(PIXEL_COL):
                # if len(chip_pixels) < 255:
                chip_pixels.append((row, col))
        all_pixels_per_chip.append(chip_pixels)

    for i, (etroc, chip_name) in enumerate(zip(etroc_chips, chip_names)):
        if etroc is not None and i < len(all_pixels_per_chip):
            etroc_configs.append((etroc, chip_name, all_pixels_per_chip[i]))

    print("Test pixel assignments:")
    for etroc, chip_name, pixels in etroc_configs:
        print(f"  {chip_name}: {len(pixels)} pixels")

    for etroc, chip_name, test_pixels in etroc_configs:
        baseline_storage[chip_name] = {}
        nw_storage[chip_name] = {}
        failed_pixels[chip_name] = []
        print(f"\nScanning {chip_name}...")
        
        for pixel_row, pixel_col in tqdm(test_pixels, desc = f"{chip_name} pixels"):
            # print(f"  Calibrating pixel ({pixel_row}, {pixel_col})...")
            try:
                baseline, nw = etroc.auto_threshold_scan(
                    row=pixel_row,
                    col=pixel_col,
                    broadcast=False,
                    offset='auto',
                    use=False,
                    verbose=True
                )
            
                baseline_storage[chip_name][(pixel_row, pixel_col)] = baseline
                nw_storage[chip_name][(pixel_row, pixel_col)] = nw
                # print(f"Baseline for {pixel_row}-{pixel_col} : {baseline_storage[chip_name][(pixel_row, pixel_col)]}")
                # print(f"NW for {pixel_row}-{pixel_col} : {nw_storage[chip_name][(pixel_row, pixel_col)]}")
            except Exception as e:
                print(red(f"  Pixel ({pixel_row},{pixel_col}): SCAN FAILED - {e}"))
                failed_pixels[chip_name].append((pixel_row, pixel_col))
        save_to_pickle()   
    if failed_pixels[chip_name]:
        print(red(f"  Found {len(failed_pixels[chip_name])} pixels with scan failures during sampling"))

    def load_pickle():
        with open('etroc_NW_scan.pkl','rb') as f:
            data = pickle.load(f)
        return data['baseline'], data['NW']
    
if __name__ == "__main__":
    main()