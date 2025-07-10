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
TRIGGER_ENABLE_MASK = 0xF
TRIGGER_DATA_SIZE = 1
TRIGGER_DELAY_SEL = 472

CHARGE_FC = 30 
QINJ_COUNT = 1

PIXEL_ROW = 16
PIXEL_COL = 16
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
    
    print("\n3. Generating test pixel configuration...")
    all_pixels_per_chip = []
    for _ in range(NUM_ETROC):
        chip_pixels = []
        for row in range(PIXEL_ROW):
            for col in range(PIXEL_COL):
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
        failed_pixels[chip_name] = []
        print(f"\nScanning {chip_name}...")
        
        for pixel_row, pixel_col in tqdm(test_pixels, desc = f"{chip_name} pixels"):
            # print(f"  Calibrating pixel ({pixel_row}, {pixel_col})...")
            try:
                baseline, _ = etroc.auto_threshold_scan(
                    row=pixel_row,
                    col=pixel_col,
                    broadcast=False,
                    offset='auto',
                    use=False,
                    verbose=True
                )
            
                baseline_storage[chip_name][(pixel_row, pixel_col)] = baseline
            except Exception as e:
                print(red(f"  Pixel ({pixel_row},{pixel_col}): SCAN FAILED - {e}"))
                failed_pixels[chip_name].append((pixel_row, pixel_col))

    if failed_pixels[chip_name]:
        print(red(f"  Found {len(failed_pixels[chip_name])} pixels with scan failures during sampling"))
    
    # Print summary of baseline calibration
    total_failed_pixels = sum(len(failed_list) for failed_list in failed_pixels.values())
    print(green("Baseline calibration completed"))

    if total_failed_pixels > 0:
        print(yellow(f"WARNING: Found {total_failed_pixels} pixels with baseline scan failures:"))
        for chip_name, failed_list in failed_pixels.items():
            if failed_list:
                print(f"  {chip_name}: {failed_list}")
    else:
        print(green("All pixels passed baseline scan"))

    time.sleep(2)
    print(green("Baseline calibration completed"))
    
    print(f"\n4. Configuring all {PIXEL_ROW * PIXEL_COL} pixels for cosmic run detection...")
    
    # Reset and configure all chips
    for etroc, chip_name, all_pixels in etroc_configs:
        print(f"Configuring {chip_name}-{PIXEL_ROW * PIXEL_COL} pixels pixels)...")
        etroc.reset()
        time.sleep(1)
        etroc.wr_reg("singlePort", 1)
        # Disable all pixels initially
        etroc.wr_reg("disDataReadout", 1, broadcast=True)
        etroc.wr_reg("QInjEn", 0, broadcast=True) 
        etroc.wr_reg("enable_TDC", 0, broadcast=True)
        etroc.wr_reg("disTrigPath", 1, broadcast=True)
        etroc.wr_reg("workMode", 0, broadcast=True)
        etroc.wr_reg('triggerGranularity', 1)
        # etroc.wr_reg("enable_TDC", 1, broadcast=True)
        # etroc.wr_reg("disDataReadout", 0, broadcast=True)
        # etroc.wr_reg("disTrigPath", 0, broadcast=True)
        
        # Configure all pixels for cosmic ray detection
        with tqdm(total=len(all_pixels), desc=f"{chip_name} pixels", ncols=100) as pbar:
            for pixel_row, pixel_col in all_pixels:
                # Set DAC threshold (baseline + Offset)
                etroc.wr_reg("workMode", 0, row=pixel_row, col=pixel_col, broadcast=False)
                etroc.wr_reg("enable_TDC", 1, row=pixel_row, col=pixel_col, broadcast=False)
                etroc.wr_reg("disDataReadout", 0, row=pixel_row, col=pixel_col, broadcast=False)
                etroc.wr_reg("disTrigPath", 0, row=pixel_row, col=pixel_col, broadcast=False)
                time.sleep(1)
                baseline = baseline_storage[chip_name][(pixel_row, pixel_col)]
                applied_dac = baseline + TH_OFFSET
                etroc.wr_reg('DAC', applied_dac, row=pixel_row, col=pixel_col, broadcast=False)
                etroc.wr_reg("QSel", CHARGE_FC, row=pixel_row, col=pixel_col, broadcast=False)
                etroc.wr_reg("QInjEn", 1, row=pixel_row, col=pixel_col, broadcast=False)
                time.sleep(0.1)
                pbar.update(1)
                pbar.set_postfix({
                    'pixel': f'({pixel_row},{pixel_col})', 
                    'DAC': f'{applied_dac:.0f}'
                })
                
                # Small delay to prevent communication issues
                if (pixel_row * 16 + pixel_col) % 32 == 0:  # Every 32 pixels
                    time.sleep(0.01)

            # NO charge injection for cosmic run
            # etroc.wr_reg("QInjEn", 0, broadcast=True)
            # etroc.wr_reg("QSel", 29, broadcast=True)
            # etroc.wr_reg("QInjEn", 1, broadcast=True)
        # print(f"  ✓ Configured all 256 pixels with DAC threshold = baseline + {TH_OFFSET}")
    
    print(green(f"All {PIXEL_ROW * PIXEL_COL} pixel configuration completed"))
    
    print("\n6. Configuring self-trigger system...")
    
    rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK_0", TRIGGER_ENABLE_MASK)
    rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK_1", TRIGGER_DATA_SIZE)
    rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK_3", TRIGGER_DELAY_SEL)
    
    print(f"Trigger ENABLE Mask: 0x{TRIGGER_ENABLE_MASK:X}")
    print(f"Trigger DATA SIZE: {TRIGGER_DATA_SIZE}")
    print(f"Trigger DELAY SEL: {TRIGGER_DELAY_SEL}")
    time.sleep(1)
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

    # Initialize FIFO and reset system
    df = DataFrame()
    fifo = FIFO(rb)
    fifo.reset()
    rb.reset_data_error_count()
    rb.enable_etroc_readout()
    rb.rerun_bitslip()
    fifo.use_etroc_data()
    time.sleep(1)
    rb.enable_etroc_trigger()
    time.sleep(1)
    
    print(green("Self-trigger system configured and enabled"))
    
    # ======================================================================================
    # 5. CONTINUOUS COSMIC RAY DETECTION
    # ======================================================================================
    
    print("\n7. Starting continuous cosmic run detection...")
    print(yellow("Press 'q' to stop acquisition"))
    
    # Setup terminal for non-blocking input
    old_settings = setup_terminal()
    fifo.send_Qinj_only(count=QINJ_COUNT)
    time.sleep(1)
    try:
        # Continuous data acquisition loop
        start_time = datetime.now(timezone.utc)
        last_report_time = time.time()
        report_interval = 60  # Report every 60 seconds
        
        while not stop_acquisition:
            try:
                # Check for quit command
                if check_for_quit():
                    break
                
                # Read data from FIFO
                data = fifo.pretty_read(df)
                time.sleep(1)
                if len(data) > 0:
                    cosmic_data.extend(data)
                    
                    # Count hits
                    for event in data:
                        if event and len(event) >= 2 and event[0] == 'data':
                            hit_counter += 1
                            hit_data = event[1]
                            
                            # Print hit information for monitoring
                            row = hit_data.get('row_id', 'N/A')
                            col = hit_data.get('col_id', 'N/A')
                            toa = hit_data.get('toa', 'N/A')
                            tot = hit_data.get('tot', 'N/A')
                            elink = hit_data.get('elink', 'N/A')
                            
                            # print(green(f"Cosmic hit #{hit_counter}: Pixel({row},{col}) "
                            #           f"elink={elink} ToA={toa} ToT={tot}"))
                
                # Periodic status report
                current_time = time.time()
                if current_time - last_report_time >= report_interval:
                    elapsed_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                    rate = hit_counter / elapsed_time if elapsed_time > 0 else 0
                    print(f"\n--- Status Report ---")
                    print(f"Running time: {elapsed_time:.1f} seconds")
                    print(f"Total cosmic hits: {hit_counter}")
                    print(f"Hit rate: {rate:.3f} hits/second")
                    print(f"Total events: {len(cosmic_data)}")
                    print("Press 'q' to stop\n")
                    last_report_time = current_time
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                print(red(f"Data acquisition error: {e}"))
                time.sleep(1)
                continue
    
    except KeyboardInterrupt:
        print(yellow("\nKeyboard interrupt detected, stopping..."))
    
    finally:
        # Restore terminal settings
        restore_terminal(old_settings)
    
    # ======================================================================================
    # 6. DATA ANALYSIS AND SAVE
    # ======================================================================================
    
    # Final data analysis and save
    end_time = datetime.now(timezone.utc)
    total_time = (end_time - start_time).total_seconds()
    
    print(f"\n8. Cosmic ray detection completed!")
    print(f"Total running time: {total_time:.1f} seconds")
    print(f"Total cosmic hits detected: {hit_counter}")
    print(f"Average hit rate: {hit_counter/total_time:.3f} hits/second")
    
    print("\n9. Analyzing cosmic ray data...")
    
    # Analyze the collected data
    header_count = hit_count = filler_count = trailer_count = 0
    pixel_hits = {}
    elink_hits = {}
    
    for event in cosmic_data:
        if event is None or len(event) < 2:
            continue
        
        data_type, event_data = event[0], event[1]
        
        if data_type == 'header':
            header_count += 1
        elif data_type == 'filler':
            filler_count += 1
        elif data_type == 'trailer':
            trailer_count += 1
        elif data_type == 'data':
            hit_count += 1
            
            # Extract hit information
            row = event_data.get('row_id', 'N/A')
            col = event_data.get('col_id', 'N/A')
            elink = event_data.get('elink', 'N/A')
            
            # Count hits per pixel
            pixel_key = f"({row},{col})"
            pixel_hits[pixel_key] = pixel_hits.get(pixel_key, 0) + 1
            
            # Count hits per elink
            elink_hits[elink] = elink_hits.get(elink, 0) + 1
    
    print(f"\nCosmic Run Analysis Summary:")
    print(f"  Total events: {len(cosmic_data)}")
    print(f"  Headers: {header_count}")
    print(f"  Cosmic hits: {hit_count}")
    print(f"  Trailers: {trailer_count}")
    print(f"  Fillers: {filler_count}")
    
    print(f"\nHits by E-link:")
    for elink in sorted(elink_hits.keys()):
        print(f"  Elink {elink}: {elink_hits[elink]} hits")
    
    print(f"\nTop 10 pixels:")
    sorted_pixels = sorted(pixel_hits.items(), key=lambda x: x[1], reverse=True)
    for i, (pixel, count) in enumerate(sorted_pixels[:10]):
        print(f"  {i+1}. Pixel {pixel}: {count} hits")
    
    print("\n10. Saving cosmic run results...")
    
    # Prepare results for saving
    cosmic_results = {
        "test_parameters": {
            "start_time_utc": start_time.isoformat(),
            "end_time_utc": end_time.isoformat(),
            "total_time_seconds": total_time,
            "num_ETROCs": len(etroc_configs),
            "total_pixels": sum(len(pixels) for _, _, pixels in etroc_configs),
            "threshold_offset": TH_OFFSET
        },
        "statistics": {
            "total_events": len(cosmic_data),
            "cosmic_hits": hit_count,
            "hit_rate_per_second": hit_count / total_time if total_time > 0 else 0,
            "pixel_hits": pixel_hits,
            "elink_hits": elink_hits
        },
        "parsed_hits": [event[1] for event in cosmic_data if event[0] == 'data'],
        "raw_events": cosmic_data
    }
    
    output_dir = "Cosmic_Output"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"cosmic_run_{len(etroc_configs)}_chips_{timestamp}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(cosmic_results, f)
        print(green(f"Cosmic ray results saved to {filepath}"))
    except Exception as e:
        print(red(f"Failed to save results: {e}"))
    
    # ======================================================================================
    # 7. CLEANUP SYSTEM
    # ======================================================================================
    
    print("\n11. Cleaning up system...")
    
    for etroc, chip_name, _ in etroc_configs:
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
    main()