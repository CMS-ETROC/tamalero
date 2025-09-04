from tamalero.ETROC import ETROC
from tamalero.utils import get_kcu
from tamalero.colors import green, red, yellow
from tamalero.ReadoutBoard import ReadoutBoard
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timezone

### Custom function
from etroc_utils import convert_dict_to_pandas, save_baselines

KCU_IP = "192.168.0.10" 

READOUTBOARD_ID = 0
READOUTBOARD_CONFIG = 'default'

ETROC_I2C_ADDRESSES = [0x60, 0x61, 0x62, 0x63]
ETROC_NAMES = ['ET2p02_PT_NH39_CE', 'ET2p02_PT_NH42_CE', 'ET2p02_PT_NH41_CE', 'ET2p02_PT_NH36_CE']

ETROC_I2C_CHANNEL = 1
ETROC_ELINKS_MAP = {0: [0, 4, 8, 12]}

NUM_ETROC = len(ETROC_I2C_ADDRESSES)


### Variables for plot
path_to_figure = '/home/daq/KCU105_NEW/ETROC-figures'
path_to_hist = '/home/daq/KCU105_NEW/ETROC-History'
custom_note = ''

# ======================================================================================
# HARDWARE INITIALIZATION FUNCTIONS
# ======================================================================================

def initialize_kcu():
    """Initialize KCU connection"""
    print('ETROC COSMIC RUN TEST - HARDWARE INITIALIZATION')
    
    kcu = get_kcu(
        KCU_IP,
        control_hub=True,
        host='localhost',
        verbose=False
    )
    print(green("Successfully connected to KCU."))

    kcu.status()

    # Perform a simple loopback register test to confirm communication
    loopback_val = 0xABCD1234
    kcu.write_node("LOOPBACK.LOOPBACK", loopback_val) #
    read_val = kcu.read_node("LOOPBACK.LOOPBACK").value()

    if read_val == loopback_val:
        print(green(f"KCU Loopback test PASSED: Wrote 0x{loopback_val:X}, Read 0x{read_val:X}"))
    else:
        print(red(f"KCU Loopback test FAILED: Wrote 0x{loopback_val:X}, Read 0x{read_val:X}"))
    
    return kcu

def initialize_readout_board(kcu):
    """Initialize readout board"""
    rb = ReadoutBoard(
        rb=READOUTBOARD_ID,
        kcu=kcu,
        config=READOUTBOARD_CONFIG, 
        trigger=False,     
        verbose=False
    )
    print(green(f"Readout Board version detected: {rb.ver}"))

    #print("\nReading DAQ LpGBT ADCs:")
    #try:
    #    rb.DAQ_LPGBT.read_adcs(check=True, strict_limits=True) #
    #except ValueError as e:
    #    print(f"LpGBT ADC check issue: {e}")

    return rb

def initialize_etroc_chips(rb):
    """Initialize all ETROC chips"""
    print("\n3. Initializing ETROC chips...")
    
    etroc_chips = []

    for i, addr in enumerate(ETROC_I2C_ADDRESSES):
        
        chip_name = ETROC_NAMES[i]
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
                    print(green("  Register communication verified"))
                else:
                    print(red("   Register communication issue"))
            else:
                print(red(f"✗ {chip_name} not responding"))
                
        except Exception as e:
            print(red(f"Failed to initialize {chip_name}: {e}"))
            etroc_chips.append(None)

    print(green("\n Hardware initialization completed successfully!"))
    print(f"Initialized {len([c for c in etroc_chips if c is not None])} ETROC chips")
    print("\nETROC to E-link Mapping:")
    
    elink_list = ETROC_ELINKS_MAP[0]  # [0, 4, 8, 12]
    
    for i, (addr, elink) in enumerate(zip(ETROC_I2C_ADDRESSES, elink_list)):
        chip_name = f"Chip{i+1}"
        if i < len(etroc_chips) and etroc_chips[i] is not None:
            status = "Connected"
        else:
            status = "Failed"
        print(f"  {ETROC_NAMES[i]} (I2C: 0x{addr:02X}) <-> E-link {elink} - {status}")
    
    return etroc_chips

# ======================================================================================
# CALIBRATION AND CONFIGURATION FUNCTIONS
# ======================================================================================

def calibrate_baselines(etroc_chips):
    """Calibrate baseline for all pixels"""
    print(f"\n2. Calibrating 256 pixel baselines...")
    
    baseline_storage = {}
    etroc_configs = []
    failed_pixels = {}
    
    all_pixels = [(row, col) for row in range(16) for col in range(16)]

    for i, etroc, in enumerate(etroc_chips):

        chip_name = ETROC_NAMES[i]
        baseline_storage[chip_name] = {
            'row': [], 'col': [], 'baseline': [],
            'noise_width': [], 'timestamp': []
        }
        
        for pixel_row, pixel_col in tqdm(all_pixels, desc = f"{chip_name} pixels"):

            baseline, noise_width = 0, 0  # Default values for failure case
            
            try:
                baseline, noise_width = etroc.auto_threshold_scan(
                    row=pixel_row,
                    col=pixel_col,
                    broadcast=False,
                    offset='auto',
                    use=False,
                    verbose=True
                )
            
                time.sleep(0.03)
            
            except Exception as e:
                print(red(f"  Pixel ({pixel_row},{pixel_col}): SCAN FAILED - {e}"))

            # 4. Append data for EVERY pixel (values are None on failure)
            baseline_storage[chip_name]['row'].append(pixel_row)
            baseline_storage[chip_name]['col'].append(pixel_col)
            baseline_storage[chip_name]['baseline'].append(baseline)
            baseline_storage[chip_name]['noise_width'].append(noise_width)
            baseline_storage[chip_name]['timestamp'].append(datetime.now().isoformat(sep=' '))

    for key, val in baseline_storage.items():
        bl_nw_df = convert_dict_to_pandas(val, key)
        tmp_timestamp = datetime.now().isoformat(sep=' ', timespec='seconds')
        my_note = tmp_timestamp + ' ' + custom_note
        save_baselines(bl_nw_df, key,
                       hist_dir=path_to_hist,
                       fig_dir=path_to_figure,
                       save_notes=my_note)    

# ======================================================================================
# MAIN FUNCTION
# ======================================================================================

def main(args = None):
    
    # Hardware initialization
    kcu = initialize_kcu()
    rb = initialize_readout_board(kcu)
    etroc_chips = initialize_etroc_chips(rb)

    for etroc in etroc_chips:
        etroc.set_power_mode(mode='high', row=0, col=0, broadcast=True)
    
    # Setup and calibration
    print("\nRun Auto Calibration")
    calibrate_baselines(etroc_chips)
    
if __name__ == "__main__":
    """
    main function
    Args:
        max_running_time: max running time (minutes)
                        None means no limit(press 'q' to stop)
    """
    import argparse

    parser = argparse.ArgumentParser(
            prog='PlaceHolder',
            description='Run Cable Eliminator DAQ!',
    )

    # parser.add_argument(
    #     '-o',
    #     '--outdir',
    #     metavar = 'NAME',
    #     type = str,
    #     help = 'output directory name',
    #     required = True,
    #     dest = 'outdir',
    # )

    args = parser.parse_args()

    main(args)
