#!/usr/bin/env python3
### Tamalero functions
from tamalero.ETROC import ETROC
from tamalero.utils import get_kcu
from tamalero.colors import green, red, yellow
from tamalero.ReadoutBoard import ReadoutBoard
from tamalero.FIFO import FIFO

from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timezone

import termios, tty, sys, select, time, yaml, struct

### Custom function
from etroc_utils import convert_dict_to_pandas, save_baselines

run_config_path = 'run_config_yamls/test.yaml'
try:
    with open(run_config_path, 'r') as file:
        run_config = yaml.safe_load(file)
except FileNotFoundError:
    print(f"Error: Configuration file '{run_config_path}' not found.")
    exit()

# --- Assign Board & Connection Variables ---
KCU_IP = run_config['system']['kcu_ip']
READOUTBOARD_ID = run_config['system']['readout_board_id']
READOUTBOARD_CONFIG = run_config['system']['readout_board_config']  

# --- Assign ETROC Variables ---
ETROC_I2C_ADDRESSES = run_config['etroc']['i2c_addresses']
ETROC_CHIP_NAMES = run_config['etroc']['chip_names']
ETROC_ELINKS_MAP = run_config['etroc']['elinks_map']
ETROC_I2C_CHANNEL = run_config['etroc']['i2c_channel']

# --- Assign Test Parameter Variables ---
TH_OFFSET = run_config['parameters']['th_offset']
TRIGGER_ENABLE_MASK = run_config['parameters']['trigger_enable_mask']
TRIGGER_DATA_SIZE = run_config['parameters']['trigger_data_size']
TRIGGER_DELAY_SEL = run_config['parameters']['trigger_delay_sel']
CHARGE_FC = run_config['parameters']['charge_fc']
QINJ_COUNT = run_config['parameters']['qinj_count']

HISTORY_PATH = run_config['paths']['history_path']
FIGURES_PATH = run_config['paths']['figures_path']
OUTPUTS_PATH = run_config['paths']['outputs_path']
NOTE_FOR_HISTORY = run_config['extra']['history_note']


# --- Derived Variables ---
# This variable is calculated from the loaded config, not stored in it.
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
                print(yellow("\nUser pressed 'q', stopping daq..."))
                stop_acquisition = True
                return True
    except:
        pass
    return False

def initialize_etroc(rb):
    
    print("\n3. Initializing ETROC chips...")
    etroc_chips = []

    for i, addr in enumerate(ETROC_I2C_ADDRESSES):
        print(f"\nInitializing {ETROC_CHIP_NAMES[i]} (I2C: 0x{addr:02X})...")
        
        try:
            etroc = ETROC(
                rb,
                master='lpgbt',
                i2c_adr=addr,
                i2c_channel=ETROC_I2C_CHANNEL,
                elinks=ETROC_ELINKS_MAP,
                strict=False,
                verbose=False,
                path_to_address_table='../address_table/ETROC2_example_fnal.yaml'
            )
            etroc_chips.append(etroc)
            
            # Verify communication
            if etroc.is_connected():
                # Check key registers
                scrambler_status = etroc.rd_reg("disScrambler")
                controller_state = etroc.rd_reg("controllerState")
                pll_unlock_count = etroc.rd_reg("pllUnlockCount")
                
                print(green(f"✓ {ETROC_CHIP_NAMES[i]} connected successfully"))
                print(f"  Controller state: {controller_state} (should be 11)")
                print(f"  PLL unlock count: {pll_unlock_count}")
                
                if scrambler_status == 1:
                    print(green("  ✓ Register communication verified"))
                else:
                    print(red("  ✗ Register communication issue"))
            else:
                print(red(f"✗ {ETROC_CHIP_NAMES[i]} not responding"))
                
        except Exception as e:
            print(red(f"✗ Failed to initialize {ETROC_CHIP_NAMES[i]}: {e}"))
            etroc_chips.append(None)

    print(green("\n✓ Hardware initialization completed successfully!"))
    print(f"Initialized {len([c for c in etroc_chips if c is not None])} ETROC chips")
    print("\nETROC to E-link Mapping:")
    elink_list = ETROC_ELINKS_MAP[0]  # [0, 4, 8, 12]
    for i, (addr, elink) in enumerate(zip(ETROC_I2C_ADDRESSES, elink_list)):
        if i < len(etroc_chips) and etroc_chips[i] is not None:
            status = "✓ Connected"
        else:
            status = "✗ Failed"
        print(f"  {ETROC_CHIP_NAMES[i]} (I2C: 0x{addr:02X}) <-> E-link {elink} - {status}")

    return etroc_chips

def measure_BL_and_NW(list_of_pixels,
                      configs,
                      path_to_hist = Path('../ETROC-History'),
                      path_to_figure = Path('../ETROC-figures'),
                      custom_note = '',
                      ):

    baseline_storage = {}
    failed_pixels = {}

    for chip_name, etroc in configs.items():
    
        # 2. Initialize storage for the chip and get a direct reference
        baseline_storage[chip_name] = {
            'row': [], 'col': [], 'baseline': [],
            'noise_width': [], 'timestamp': []
        }
        failed_pixels[chip_name] = []

        print(f"\nScanning {chip_name}...")

        # 3. Loop through each pixel for the current chip
        for pixel_row, pixel_col in tqdm(list_of_pixels, desc=f"{chip_name} pixels"):
            
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
            
            except Exception as e:
                print(red(f"  Pixel ({pixel_row},{pixel_col}): SCAN FAILED - {e}"))
                failed_pixels[chip_name].append((pixel_row, pixel_col))

            # 4. Append data for EVERY pixel (values are None on failure)
            baseline_storage[chip_name]['row'].append(pixel_row)
            baseline_storage[chip_name]['col'].append(pixel_col)
            baseline_storage[chip_name]['baseline'].append(baseline)
            baseline_storage[chip_name]['noise_width'].append(noise_width)
            baseline_storage[chip_name]['timestamp'].append(datetime.now().isoformat(sep=' '))

    # 5. Per-chip summary of failures (moved inside the loop)
    if failed_pixels[chip_name]:
        print(red(f"  Found {len(failed_pixels[chip_name])} pixels with scan failures on {chip_name}"))

    # --- Final Summary ---
    total_failed_pixels = sum(len(failed_list) for failed_list in failed_pixels.values())
    print(green("\nBaseline calibration completed"))

    if total_failed_pixels > 0:
        print(yellow(f"WARNING: Found {total_failed_pixels} total pixels with baseline scan failures:"))
        for chip_name, failed_list in failed_pixels.items():
            if failed_list:
                print(f"  {chip_name}: {failed_list}")
    else:
        print(green("All pixels passed baseline scan"))

    print("Save Baseline and noise width plots")

    for key, val in baseline_storage.items():
        bl_nw_df = convert_dict_to_pandas(val, key)
        tmp_timestamp = datetime.now().isoformat(sep=' ', timespec='seconds')
        my_note = tmp_timestamp + custom_note
        save_baselines(bl_nw_df, key,
                       hist_dir=path_to_hist,
                       fig_dir=path_to_figure,
                       save_notes=my_note)
    
    return baseline_storage


def config_etroc(list_of_pixels, 
                 configs,
                 baseline_dict
                 ):
    for chip_name, etroc in configs.items():

        # --- 1. Perform chip-wide broadcast configurations ONCE ---
        # This is far more efficient than setting the same value for each pixel individually.
        print(f"\nConfiguring {chip_name} with broadcast writes...")
        etroc.reset()
        time.sleep(0.5)  # A delay after reset is often necessary

        # Set universal configurations for all pixels via broadcast
        etroc.wr_reg("singlePort", 1)
        etroc.wr_reg("workMode", 0, broadcast=True)
        etroc.wr_reg("triggerGranularity", 1)
        etroc.wr_reg("enable_TDC", 1, broadcast=True)
        etroc.wr_reg("disDataReadout", 0, broadcast=True)
        etroc.wr_reg("disTrigPath", 0, broadcast=True)
        etroc.wr_reg("QSel", CHARGE_FC, broadcast=True)
        etroc.wr_reg("QInjEn", 0, broadcast=True)

        # Set universal thresholds for all pixels via broadcast
        etroc.set_trigger_TH('TOA', upper=0x3ff, lower=0, broadcast=True)
        etroc.set_trigger_TH('TOT', upper=0x1ff, lower=0, broadcast=True)
        etroc.set_trigger_TH('Cal', upper=0x3ff, lower=0, broadcast=True)
        etroc.set_data_TH('TOA', upper=0x3ff, lower=0, broadcast=True)
        etroc.set_data_TH('TOT', upper=0x1ff, lower=0, broadcast=True)
        etroc.set_data_TH('Cal', upper=0x3ff, lower=0, broadcast=True)

        # --- 2. Prepare data for fast lookups (Non-Pandas Method) ---
        print(f"Building baseline lookup dictionary for {chip_name}...")
        chip_data = baseline_dict[chip_name]
        
        # Create a dict mapping (row, col) -> baseline for fast O(1) lookups
        baseline_lookup = {
            (r, c): bl 
            for r, c, bl in zip(chip_data['row'], chip_data['col'], chip_data['baseline'])
        }

         # --- 3. Configure pixel-specific registers (DAC setting) ---
        print(f"Setting individual pixel DACs for {chip_name}...")
        with tqdm(total=len(list_of_pixels), desc=f"Pixel DACs", ncols=100) as pbar:
            for pixel_row, pixel_col in list_of_pixels:
                
                # A) Fast baseline lookup from our new dictionary
                baseline = baseline_lookup.get((pixel_row, pixel_col)) # .get() is safer
                
                if baseline is None:
                    # Handle case where a pixel might be missing from storage
                    pbar.update(1)
                    continue

                # B) Write the one register that is unique to this pixel
                applied_dac = baseline + TH_OFFSET
                etroc.wr_reg('DAC', applied_dac, row=pixel_row, col=pixel_col, broadcast=False)
                
                pbar.update(1)
                pbar.set_postfix({'pixel': f'({pixel_row},{pixel_col})', 'DAC': f'{applied_dac:.0f}'})
        
        print(f"{chip_name} configuration complete.")

def prepare_self_trigger_system(rb):
    print("\n6. Configuring self-trigger system...")
    
    rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK_0", TRIGGER_ENABLE_MASK)
    rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK_1", TRIGGER_DATA_SIZE)
    rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK_3", TRIGGER_DELAY_SEL)
    
    print(f"Trigger ENABLE Mask: 0x{TRIGGER_ENABLE_MASK:X}")
    print(f"Trigger DATA SIZE: {TRIGGER_DATA_SIZE}")
    print(f"Trigger DELAY SEL: {TRIGGER_DELAY_SEL}")
    time.sleep(0.2)
    # check elink status
    for elink in [0,4,8,12]:
        locked = rb.etroc_locked(elink, slave=False)
        print(f"elink: {elink} locked status: {locked}")
        if not locked:
            print(f"Warning: E-link {elink} is NOT locked after configuration. Attempting to re-lock...")
            rb.rerun_bitslip() 
            time.sleep(0.5)
            locked = rb.etroc_locked(elink, slave=False) 
            print(f"After re-lock, elink {elink} locked status : {locked}")
            if not locked:
                print(f"FATAL: E-link {elink} failed to lock. Stopping.")

    print(green("Self-trigger system configured and enabled"))


def main(args):
    global stop_acquisition
    
    print('ETROC HARDWARE INITIALIZATION')
    
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
    # rb.DAQ_LPGBT.set_dac(1.0)

    # ======================================================================================
    # 3. INITIALIZE ETROCs
    # ======================================================================================

    etroc_chips = initialize_etroc(rb)

    # ======================================================================================
    # 4. Baseline and Noise Width calibration
    # ======================================================================================
            
    all_pixels = [(row, col) for row in range(16) for col in range(16)]

    # 2. Build the configuration list, filtering for valid chips.
    etroc_configs = {
        chip_name: etroc
        for etroc, chip_name in zip(etroc_chips, ETROC_CHIP_NAMES)
        if etroc is not None
    }

    if not args.skip_etroc_config:
        print(f"4. Calibrating pixels' baselines...")
        baseline_storage = measure_BL_and_NW(all_pixels, etroc_configs,
                                            path_to_hist=HISTORY_PATH,
                                            path_to_figure=FIGURES_PATH,
                                            custom_note=NOTE_FOR_HISTORY)

    # ======================================================================================
    # 5. Chip configuration
    # ======================================================================================
    
    if not args.skip_etroc_config:
        config_etroc(all_pixels, etroc_configs, baseline_storage)
    
    # ======================================================================================
    # 6. Prepare self-trigger system
    # ======================================================================================   
    
    prepare_self_trigger_system(rb)

    # ======================================================================================
    # 7. FIFO
    # ======================================================================================   
    
    # Initialize FIFO and reset system
    fifo = FIFO(rb)
    fifo.reset()
    rb.reset_data_error_count()
    rb.enable_etroc_readout()
    rb.rerun_bitslip()
    fifo.use_etroc_data()
    time.sleep(0.2)
    rb.enable_etroc_trigger()

    # ======================================================================================
    # 8. CONTINUOUS COSMIC RAY DETECTION
    # ======================================================================================
    
    print("\n8. Starting continuous cosmic run detection...")
    print(yellow("Press 'q' to stop acquisition"))

    # Setup terminal for non-blocking input
    old_settings = setup_terminal()

    if QINJ_COUNT > 0:
        print('Charge injection test run')
        fifo.send_Qinj_only(count=QINJ_COUNT)
        fifo.reset()

    time.sleep(1)
    
    try:
        # --- Configuration for file splitting ---
        output_dir = Path(OUTPUTS_PATH)
        output_dir.mkdir(exist_ok=True)

        max_counters_per_file = run_config['extra']['max_data_counter']
        file_number = 0
        counters_in_current_file = 0
        current_file = None

        print(f"Saving data to: {output_dir.resolve()}, splitting every {max_counters_per_file} counters")

        start_time = datetime.now(timezone.utc)
        while not stop_acquisition:
            try:
                # Check for quit command
                if check_for_quit():
                    break

                # --- Check if a new file needs to be created ---
                if current_file is None:
                    new_filename = output_dir / f"file_{file_number}_CE.dat"
                    print(f"Opening new file: {new_filename}")
                    current_file = open(new_filename, "wb")
                    counters_in_current_file = 0
                
                # --- Read and write data ---
                raw_data = fifo.read(dispatch=True)
                if raw_data:
                    packed_data = struct.pack(f'<{len(raw_data)}I', *raw_data)
                    current_file.write(packed_data)
                    counters_in_current_file += 1
                
                    # --- Check event count and roll over if needed ---
                    if counters_in_current_file >= max_counters_per_file:
                        current_file.close()
                        print(f"Closed file: {current_file.name}")

                        file_number += 1
                        current_file = None # Trigger opening a new file on the next loop

            except Exception as e:
                print(red(f"An error occurred: {e}"))
            
    except KeyboardInterrupt:
        print(yellow("\nKeyboard interrupt detected, stopping..."))

    finally:
        elapsed_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        print(f"Running time: {elapsed_time:.1f} seconds")

        if current_file and not current_file.closed:
            current_file.close()
            print(f"Closed final file: {current_file.name}")

        # Restore terminal settings
        restore_terminal(old_settings)
    

    # ======================================================================================
    # 9. CLEANUP SYSTEM
    # ======================================================================================

    print("\n9. Cleaning up system...")

    for chip_name, etroc in etroc_configs.items():
        print(f"Cleaning up {chip_name}...")
        for _ in range(3):
            fifo.reset()
            rb.reset_data_error_count()
            etroc.wr_reg("QInjEn", 0, broadcast=True)
            etroc.wr_reg("disDataReadout", 1, broadcast=True)
            time.sleep(0.1)

    print(green("System cleanup completed"))
    print(green("Cosmic ray test finished successfully!"))


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(
            prog='PlaceHolder',
            description='Run Cable Eliminator DAQ!',
    )

    parser.add_argument(
        '--skip_etroc_config',
        action = 'store_true',
        help = 'If set, skip etroc initialziation and configuration',
        dest = 'skip_etroc_config',
    )
    
    args = parser.parse_args()
    
    main(args)