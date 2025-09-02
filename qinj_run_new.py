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
from datetime import datetime, timezone,timedelta


KCU_IP = "192.168.0.10" 

READOUTBOARD_ID = 0
READOUTBOARD_CONFIG = 'default'

ETROC_I2C_ADDRESSES = [0x60]
ETROC_I2C_CHANNEL = 1
ETROC_ELINKS_MAP = {0: [0, 4, 8, 12]}

# Test parameters
TH_OFFSET = 20              # Threshold offset above baseline
TRIGGER_ENABLE_MASK = 0x1
TRIGGER_DATA_SIZE = 1
TRIGGER_DELAY_SEL = 469

CHARGE_FC = 30 
QINJ_COUNT = 100
CHUNK_SIZE = 500000     # number of events for each saved file
running_time = 2       # minutes, None means no limit

PIXEL_ROW = 1
PIXEL_COL = 2
NUM_ETROC = len(ETROC_I2C_ADDRESSES)

stop_acquisition = False
hit_counter = 0

# ======================================================================================
# CHUNKED DATA SAVER CLASS
# ======================================================================================

class ChunkedDataSaver:
    def __init__(self, base_dir="Cosmic_Data_Chunks", chunk_size=50000):

        self.base_dir = base_dir
        self.chunk_size = chunk_size
        self.current_chunk = []
        self.chunk_number = 0
        self.total_events = 0
        
        # 创建输出目录
        self.session_dir = os.path.join(base_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        print(f"Data will be saved into: {self.session_dir}")
        print(f"Every chunk size: {chunk_size} events")
    
    def add_events(self, events):
        if not events:
            return
            
        self.current_chunk.extend(events)
        self.total_events += len(events)
        
        if len(self.current_chunk) >= self.chunk_size:
            self._save_current_chunk()
    
    def _save_current_chunk(self):
        if not self.current_chunk:
            return
            
        filename = f"chunk_{self.chunk_number:04d}.pkl"
        filepath = os.path.join(self.session_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.current_chunk, f)
            
            print(f"Already saved chunks {self.chunk_number}: {len(self.current_chunk)} events -> {filename}")
            
            # empty current chunk and prepare for the next
            self.current_chunk = []
            self.chunk_number += 1
            
        except Exception as e:
            print(f" Saving chunk {self.chunk_number} failed: {e}")
    
    def finalize(self):
        # save the last chunk if exist
        if self.current_chunk:
            self._save_current_chunk()
        
        metadata = {
            "total_events": self.total_events,
            "total_chunks": self.chunk_number,
            "chunk_size": self.chunk_size,
            "session_dir": self.session_dir,
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_file = os.path.join(self.session_dir, "metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\nData dumping completed:")
        print(f"- Total events: {self.total_events:,}")
        print(f"- Total chunks: {self.chunk_number}")
        print(f"- Directory: {self.session_dir}")

# ======================================================================================
# TERMINAL CONTROL FUNCTIONS
# ======================================================================================

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
    """Check if pressed 'q' to quit"""
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
    return rb

def initialize_etroc_chips(rb):
    """Initialize all ETROC chips"""
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
        print(f"  {chip_name} (I2C: 0x{addr:02X}) <-> E-link {elink} - {status}")
    
    return etroc_chips, chip_names

# ======================================================================================
# CALIBRATION AND CONFIGURATION FUNCTIONS
# ======================================================================================

def calibrate_baselines(etroc_chips, chip_names):
    """Calibrate baseline for all pixels"""
    print(f"\n2. Calibrating {PIXEL_ROW * PIXEL_COL} pixel baselines...")
    
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

    time.sleep(1)
    print(green("Baseline calibration completed"))
    
    return etroc_configs, baseline_storage

def configure_etroc_for_cosmic(etroc_configs, baseline_storage):
    """Configure all ETROC chips for cosmic run detection"""
    print(f"\n4. Configuring all {PIXEL_ROW * PIXEL_COL} pixels for cosmic run detection...")
    
    # Reset and configure all chips
    for _, (etroc, chip_name, all_pixels) in enumerate(etroc_configs):
        print(f"Configuring {chip_name}-{PIXEL_ROW * PIXEL_COL} pixels pixels)...")
        etroc.reset()
        time.sleep(0.1)
        etroc.wr_reg("singlePort", 1)
        # Disable all pixels initially
        etroc.wr_reg("disDataReadout", 1, broadcast=True)
        etroc.wr_reg("QInjEn", 0, broadcast=True) 
        etroc.wr_reg("enable_TDC", 0, broadcast=True)
        etroc.wr_reg("disTrigPath", 1, broadcast=True)
        etroc.wr_reg("workMode", 0, broadcast=True)
        etroc.wr_reg('triggerGranularity', 1)
        time.sleep(0.1)
        
        # Configure all pixels for cosmic ray detection
        with tqdm(total=len(all_pixels), desc=f"{chip_name} pixels", ncols=100) as pbar:
            for pixel_row, pixel_col in all_pixels:
                # Set DAC threshold (baseline + Offset)
                etroc.wr_reg("workMode", 0, row=pixel_row, col=pixel_col, broadcast=False)
                etroc.wr_reg("enable_TDC", 1, row=pixel_row, col=pixel_col, broadcast=False)
                etroc.wr_reg("disDataReadout", 0, row=pixel_row, col=pixel_col, broadcast=False)
                etroc.wr_reg("disTrigPath", 0, row=pixel_row, col=pixel_col, broadcast=False)
                time.sleep(0.1)
                baseline = baseline_storage[chip_name][(pixel_row, pixel_col)]
                applied_dac = baseline + TH_OFFSET
                etroc.wr_reg('DAC', applied_dac, row=pixel_row, col=pixel_col, broadcast=False)
                etroc.wr_reg("QSel", CHARGE_FC - 1, row=pixel_row, col=pixel_col, broadcast=False)
                etroc.wr_reg("QInjEn", 1, row=pixel_row, col=pixel_col, broadcast=False)
                etroc.set_trigger_TH('TOA', upper=0x3ff, lower=0, row=pixel_row, col=pixel_col, broadcast=False)
                etroc.set_trigger_TH('TOT', upper=0x1ff, lower=0, row=pixel_row, col=pixel_col, broadcast=False)
                etroc.set_trigger_TH('Cal', upper=0x3ff, lower=0, row=pixel_row, col=pixel_col, broadcast=False)
                etroc.set_data_TH('TOA', upper=0x3ff, lower=0 ,row=pixel_row, col=pixel_col, broadcast=False)
                etroc.set_data_TH('TOT', upper=0x1ff, lower=0 ,row=pixel_row, col=pixel_col, broadcast=False)
                etroc.set_data_TH('Cal', upper=0x3ff, lower=0 ,row=pixel_row, col=pixel_col, broadcast=False)
                pbar.update(1)
                pbar.set_postfix({
                    'pixel': f'({pixel_row},{pixel_col})', 
                    'DAC': f'{applied_dac:.0f}'
                })
                
                # Small delay to prevent communication issues
                if (pixel_row * 16 + pixel_col) % 32 == 0:  # Every 32 pixels
                    time.sleep(0.01)
    
    print(green(f"All {PIXEL_ROW * PIXEL_COL} pixel configuration completed"))

def configure_trigger_system(rb):
    """Configure self-trigger system"""
    print("\n6. Configuring self-trigger system...")
    
    rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK_0", TRIGGER_ENABLE_MASK)
    rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK_1", TRIGGER_DATA_SIZE)
    rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK_3", TRIGGER_DELAY_SEL)
    
    print(f"Trigger ENABLE Mask: 0x{TRIGGER_ENABLE_MASK:X}")
    print(f"Trigger DATA SIZE: {TRIGGER_DATA_SIZE}")
    print(f"Trigger DELAY SEL: {TRIGGER_DELAY_SEL}")
    time.sleep(0.1)

    # check elink status
    for elink in [0,4,8,12]:
        locked = rb.etroc_locked(elink, slave=False)
        print(f"elink: {elink} locked status: {locked}")
        if not locked:
            print(yellow(f"Warning: E-link {elink} is NOT locked after configuration. Attempting to re-lock..."))
            rb.rerun_bitslip() 
            time.sleep(0.5)
            locked = rb.etroc_locked(elink, slave=False) 
            print(f"After re-lock, elink {elink} locked status : {locked}")
            if not locked:
                print(red(f"FATAL: E-link {elink} failed to lock. Stopping."))

    print(green("Self-trigger system configured and enabled"))

# ======================================================================================
# DATA ACQUISITION FUNCTION
# ======================================================================================

def run_cosmic_detection(rb, max_running_time):
    """Run continuous cosmic ray detection with chunked data saving"""
    global stop_acquisition, hit_counter
    
    print("\n7. Starting continuous cosmic run detection...")

    if max_running_time:
        print(yellow(f"Maximum running time being set: {max_running_time} minutes"))
        print(yellow(f"Press 'q' to stop acquisition or wait {max_running_time} minutes to auto stop"))
    else:
        print(yellow("Press 'q' to stop acquisition"))
                
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
    
    # Initialize chunked data saver
    chunk_saver = ChunkedDataSaver(chunk_size=CHUNK_SIZE)
    
    # Setup terminal for non-blocking input
    old_settings = setup_terminal()
    fifo.send_Qinj_only(count=QINJ_COUNT)
    time.sleep(1)
    
    try:
        # Continuous data acquisition loop
        start_time = datetime.now(timezone.utc)
        last_report_time = time.time()
        last_save_time = time.time()
        report_interval = 10  # seconds
        save_interval = 300   # save data every 300 s
        trigger_cnt = 0

        end_time = None
        if max_running_time:
            end_time = start_time + timedelta(minutes=max_running_time)
            print(f"Start time: {start_time.strftime('%H:%M:%S')}")
            print(f"Estimate ending time: {end_time.strftime('%H:%M:%S')}")

        while not stop_acquisition:
            try:
                # Check for quit command
                if check_for_quit():
                    break
                    
                if max_running_time:
                    current_time = datetime.now(timezone.utc)
                    if current_time >= end_time:
                        elapsed_time = (current_time - start_time).total_seconds() / 60
                        print(yellow(f"Reached time setting ({elapsed_time:.1f} minutes), auto stopped"))
                        break
                
                #fifo.send_Qinj_only(count=QINJ_COUNT)
                data = fifo.pretty_read(df)
                
                if len(data) > 0:
                    chunk_saver.add_events(data)
                    
                    # Count hits
                    for event in data:
                        if event and len(event) >= 2:
                            if event[0] == 'header':
                                trigger_cnt += 1
                            elif event[0] == 'data':
                                hit_counter += 1
                
                current_time = time.time()
                
                # Periodic status report
                if current_time - last_report_time >= report_interval:
                    elapsed_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                    elapsed_minutes = elapsed_time / 60

                    print(f"\n--- Status Report ---")
                    print(f"Running time: {elapsed_time:.1f} seconds")
                    print(f"Total cosmic hits: {hit_counter}")
                    print(f"Trigger count: {trigger_cnt}")
                    print(f"current saved events: {chunk_saver.total_events:,}")
                    print(f"Current chunk size: {len(chunk_saver.current_chunk):,}")

                    if max_running_time:
                        remaining_time = (max_running_time * 60) - elapsed_time
                        if remaining_time > 60:
                            remaining_min = int(remaining_time // 60)
                            remaining_sec = int(remaining_time % 60)
                            print(yellow(f"Press 'q' to stop acquisition earlier or wait {remaining_min} minutes and {remaining_sec} secs to auto stop"))
                        elif remaining_time > 0:
                            print(yellow(f"remaining time: {int(remaining_time)} sec"))
                    else:
                        print("Press 'q' to stop\n")
                    last_report_time = current_time

                if current_time - last_save_time >= save_interval:
                    if chunk_saver.current_chunk: 
                        chunk_saver._save_current_chunk()
                    last_save_time = current_time
                
                time.sleep(0.05)  # Small delay
                
            except Exception as e:
                print(red(f"Data acquisition error: {e}"))
                time.sleep(1)
                continue
        
    except KeyboardInterrupt:
        print(yellow("\nKeyboard interrupt detected, stopping..."))
    
    finally:
        # Restore terminal settings
        restore_terminal(old_settings)
        
        chunk_saver.finalize()

    end_time = datetime.now(timezone.utc)
    total_time = (end_time - start_time).total_seconds()
    
    print(f"\n8. Cosmic ray detection completed!")
    print(f"Total running time: {total_time:.1f} seconds")
    print(f"Total cosmic hits detected: {hit_counter}")

def cleanup_system(etroc_configs, rb):
    """Cleanup system"""
    print("\n11. Cleaning up system...")
    
    fifo = FIFO(rb)
    
    for etroc, chip_name, _ in etroc_configs:
        print(f"Cleaning up {chip_name}...")
        for _ in range(2):  
            fifo.reset()
            rb.reset_data_error_count()
            etroc.wr_reg("QInjEn", 0, broadcast=True)
            etroc.wr_reg("disDataReadout", 1, broadcast=True)
            time.sleep(0.1)
    
    print(green("System cleanup completed"))
    print(green("Cosmic ray test finished successfully!"))

# ======================================================================================
# MAIN FUNCTION
# ======================================================================================

def main(max_running_time = None):
    global stop_acquisition, hit_counter
    
    # Hardware initialization
    kcu = initialize_kcu()
    rb = initialize_readout_board(kcu)
    etroc_chips, chip_names = initialize_etroc_chips(rb)
    
    # Setup and calibration
    print("\nETROC COSMIC RAY TEST - CONTINUOUS DETECTION")
    etroc_configs, baseline_storage = calibrate_baselines(etroc_chips, chip_names)
    configure_etroc_for_cosmic(etroc_configs, baseline_storage)
    configure_trigger_system(rb)
    
    # Data acquisition
    run_cosmic_detection(rb,max_running_time)
    
    # Cleanup
    cleanup_system(etroc_configs, rb)

if __name__ == "__main__":
    """
    main function
    Args:
        max_running_time: max running time (minutes)
                        None means no limit(press 'q' to stop)
    """
    main(running_time)
