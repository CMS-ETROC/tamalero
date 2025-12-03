from tamalero.FIFO import FIFO
from tamalero.ETROC import ETROC
from tamalero.LPGBT import LPGBT
from tamalero.utils import get_kcu
from tamalero.DataFrame import DataFrame
from tamalero.colors import green, red, yellow
from tamalero.ReadoutBoard import ReadoutBoard
from tamalero.KCU import KCU
import os
import sys
import tty
import time
import select
import pickle
import termios
import struct
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from random import randint
from datetime import datetime, timezone,timedelta

### Custom function
from etroc_utils import convert_dict_to_pandas, save_baselines

KCU_IP = "192.168.0.10"

READOUTBOARD_ID = 0
READOUTBOARD_CONFIG = 'default'

ETROC_I2C_ADDRESSES = [0x60, 0x61, 0x62, 0x63]
ETROC_NAMES = ['ET2p02_PT_IH2', 'ET2p02_PT_IH3', 'ET2p02_PT_IH5', 'ET2p02_PT_IH24']
ETROC_I2C_CHANNEL = 1
ETROC_ELINKS_MAP = {0: [0, 4, 8, 12]}

# Test parameters
TH_OFFSET = 10              # 20 DAC Threshold offset above baseline
TH_OFFSETS = {
    'ET2p02_PT_IH2': 20,
    'ET2p02_PT_IH3': 20,
    'ET2p02_PT_IH5': 20,
    'ET2p02_PT_IH24': 20,
}

TRIGGER_ENABLE_MASK = 0x1 # 0001, 0x1 (0x63, 0x62, 0x61, 0x60) - trigger now the last plane NH39
TRIGGER_DATA_SIZE = 1
TRIGGER_DELAY_SEL = 470

CHARGE_FC = 5
QINJ_COUNT = 0
CHUNK_SIZE = 500     # number of events for each saved file
MAX_FILE_SIZE_BYTES = 120*1024*1024 

PIXEL_ROW = 16
PIXEL_COL = 16
NUM_ETROC = len(ETROC_I2C_ADDRESSES)

### Variables for plot
path_to_figure = '/home/daq/ETROC2_KCU105/ETROC-figures'
path_to_hist = '/home/daq/ETROC2_KCU105/ETROC-History'

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
    ipb_path = f"chtcp-2.0://localhost:10203?target={KCU_IP}:50001"
    generic_xml_path = os.path.expandvars("$TAMALERO_BASE/address_table/generic/etl_test_fw.xml")

    kcu = KCU(
        name="kcu",
        ipb_path=ipb_path,
        adr_table=generic_xml_path
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
        print(f"  {chip_name} (I2C: 0x{addr:02X}) <-> E-link {elink} - {status}")

    return etroc_chips

# ======================================================================================
# CALIBRATION AND CONFIGURATION FUNCTIONS
# ======================================================================================

def calibrate_baselines(etroc_chips, chip_names, custom_note):
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
        baseline_storage[chip_name] = {
            'row': [], 'col': [], 'baseline': [],
            'noise_width': [], 'timestamp': []
        }
        failed_pixels[chip_name] = []
        print(f"\nScanning {chip_name}...")

        for pixel_row, pixel_col in tqdm(test_pixels, desc = f"{chip_name} pixels"):
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

                baseline_storage[chip_name][(pixel_row, pixel_col)] = baseline
            except Exception as e:
                print(red(f"  Pixel ({pixel_row},{pixel_col}): SCAN FAILED - {e}"))
                # failed_pixels[chip_name].append((pixel_row, pixel_col))

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

        etroc.set_trigger_TH('TOA', upper=0x3ff, lower=0, row=0, col=0, broadcast=True)
        etroc.set_trigger_TH('TOT', upper=0x1ff, lower=0, row=0, col=0, broadcast=True)
        etroc.set_trigger_TH('Cal', upper=0x3ff, lower=0, row=0, col=0, broadcast=True)
        etroc.set_data_TH('TOA', upper=0x3ff, lower=0 ,row=0, col=0, broadcast=True)
        etroc.set_data_TH('TOT', upper=0x1ff, lower=0 ,row=0, col=0, broadcast=True)
        etroc.set_data_TH('Cal', upper=0x3ff, lower=0 ,row=0, col=0, broadcast=True)

        chip_data = baseline_storage[chip_name]

        # Create a dict mapping (row, col) -> baseline for fast O(1) lookups
        baseline_lookup = {
            (r, c): bl
            for r, c, bl in zip(chip_data['row'], chip_data['col'], chip_data['baseline'])
        }

        # Configure all pixels for cosmic ray detection
        with tqdm(total=len(all_pixels), desc=f"{chip_name} pixels", ncols=100) as pbar:
            for pixel_row, pixel_col in all_pixels:
                # Set DAC threshold (baseline + Offset)
                etroc.wr_reg("enable_TDC", 1, row=pixel_row, col=pixel_col, broadcast=False)
                etroc.wr_reg("disDataReadout", 0, row=pixel_row, col=pixel_col, broadcast=False)
                etroc.wr_reg("disTrigPath", 0, row=pixel_row, col=pixel_col, broadcast=False)
                baseline = baseline_lookup.get((pixel_row, pixel_col))
                applied_dac = baseline + TH_OFFSETS[chip_name]
                etroc.wr_reg('DAC', applied_dac, row=pixel_row, col=pixel_col, broadcast=False)
                # etroc.wr_reg("QSel", CHARGE_FC - 1, row=pixel_row, col=pixel_col, broadcast=False)
                # etroc.set_trigger_TH('TOA', upper=0x3ff, lower=0, row=pixel_row, col=pixel_col, broadcast=False)
                # etroc.set_trigger_TH('TOT', upper=0x1ff, lower=0, row=pixel_row, col=pixel_col, broadcast=False)
                # etroc.set_trigger_TH('Cal', upper=0x3ff, lower=0, row=pixel_row, col=pixel_col, broadcast=False)
                # etroc.set_data_TH('TOA', upper=0x3ff, lower=0 ,row=pixel_row, col=pixel_col, broadcast=False)
                # etroc.set_data_TH('TOT', upper=0x1ff, lower=0 ,row=pixel_row, col=pixel_col, broadcast=False)
                # etroc.set_data_TH('Cal', upper=0x3ff, lower=0 ,row=pixel_row, col=pixel_col, broadcast=False)
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

    rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK", TRIGGER_ENABLE_MASK)
    rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_DATA_SIZE", TRIGGER_DATA_SIZE)
    rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_DLY_SEL", TRIGGER_DELAY_SEL)

    print(f"Trigger ENABLE Mask: 0x{TRIGGER_ENABLE_MASK:X}")
    print(f"Trigger DATA SIZE: {TRIGGER_DATA_SIZE}")
    print(f"Trigger DELAY SEL: {TRIGGER_DELAY_SEL}")
    time.sleep(0.1)

    max_retries = 10
    retry_delay_sec = 0.5
    all_links_locked = True
    # check elink status
    for elink in [0,4,8,12]:
        locked = rb.etroc_locked(elink, slave=False)
        print(f"elink: {elink} locked status: {locked}")

        retries = 0
        while not locked and retries < max_retries:
            retries += 1
            print(yellow(f"  Warning: E-link {elink} NOT locked. Retrying... (Attempt {retries}/{max_retries})"))

            rb.rerun_bitslip()
            time.sleep(retry_delay_sec)

            locked = rb.etroc_locked(elink, slave=False)
            print(f"  E-link {elink} status after retry: {locked}")

        if not locked:
            print(red(f"FATAL: E-link {elink} failed to lock after {max_retries} attempts. Stopping."))
            all_links_locked = False

    if all_links_locked:
        print(green("Success: All E-links are locked."))
        print(green("Self-trigger system configured and enabled"))
    else:
        print(red("Script terminating due to E-link lock failure."))
        sys.exit(1)

# ======================================================================================
# DATA ACQUISITION FUNCTION
# ======================================================================================

def run_cosmic_detection(rb, args):
    """Run continuous cosmic ray detection with chunked data saving"""
    global stop_acquisition, hit_counter

    print("\n7. Starting continuous cosmic run detection...")
    max_running_time = args.max_run_time
    if max_running_time > 1440:
        max_running_time = 1440
        print('DAQ maximum time is set to over 24 hrs. Current DAQ may not stable after 24 hrs.')
        print('So, reducing maximum running time to 24 hrs.')

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

    # Setup terminal for non-blocking input
    old_settings = setup_terminal()
    time.sleep(1)

    try:
        # Continuous data acquisition loop
        start_time = datetime.now(timezone.utc)
        end_time = None
        
        if max_running_time:
            end_time = start_time + timedelta(minutes=max_running_time)
            print(f"Start time: {start_time.strftime('%H:%M:%S')}")
            print(f"Estimate ending time: {end_time.strftime('%H:%M:%S')}")

        output_dir = Path(args.outdir)
        output_dir.mkdir(exist_ok=True, parents=True)
        file_number = 0
        counters_in_current_file = 0
        current_file = None

        start_time = datetime.now(timezone.utc)
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

                # --- Check if a new file needs to be created ---
                if current_file is None:
                    new_filename = output_dir / f"file_{file_number}_CE.dat"
                    print(f"Opening new file: {new_filename}")
                    current_file = open(new_filename, "wb")
                    counters_in_current_file = 0

                # --- Read and write data ---
                raw_data = fifo.read(dispatch=True)

                time.sleep(0.1) ## slow down daq speed to avoid "uhal UDP error in FIFO.get_occupancy, trying again" error

                if raw_data:
                    packed_data = struct.pack(f'<{len(raw_data)}I', *raw_data)
                    current_file.write(packed_data)
                    counters_in_current_file += 1
                    
                    # Get current file size +after+ writing
                    current_size_bytes = current_file.tell()
                    # Check if either limit is reached
                    split_by_count = (counters_in_current_file >= CHUNK_SIZE)
                    split_by_size = (current_size_bytes >= MAX_FILE_SIZE_BYTES)

                    if split_by_count or split_by_size:
                        # --- Optional: Log the reason for splitting ---
                        if split_by_count:
                            print(f"Reached event count limit ({counters_in_current_file} events).")
                        else:
                            size_mb = current_size_bytes / (1024 * 1024)
                            print(f"Reached file size limit ({size_mb:.2f} MB).")
                        # --- End optional logging ---

                        current_file.close()
                        print(f"Closed file: {current_file.name}")

                        file_number += 1
                        current_file = None # Trigger opening a new file on the next loop

                    time.sleep(0.1) ## slow down daq speed to avoid "uhal UDP error in FIFO.get_occupancy, trying again" error
        
            except Exception as e:
                print(red(f"Data acquisition error: {e}"))
                time.sleep(1)
                continue

    except KeyboardInterrupt:
        print(yellow("\nKeyboard interrupt detected, stopping..."))

    finally:
        elapsed_time = (datetime.now(timezone.utc) - start_time)
        print(f"Running time: {str(elapsed_time).split('.')[0]}")

        if current_file and not current_file.closed:
            current_file.close()
            print(f"Closed final file: {current_file.name}")
        # Restore terminal settings
        restore_terminal(old_settings)

        # chunk_saver.finalize()

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

## --------------------------------------
def read_BLNW_history_chip_measurements(sqlite_file: Path, chip_name: str):
    with sqlite3.connect(sqlite_file) as sqlite3_connection:
        timestamp_df = pd.read_sql_query(f"SELECT timestamp, save_notes FROM baselines WHERE chip_name='{chip_name}' AND ROW=0 AND COL=0", sqlite3_connection)
        max_timestamp_df = pd.read_sql_query(f"SELECT timestamp FROM baselines WHERE chip_name='{chip_name}' AND ROW=15 AND COL=15", sqlite3_connection)
        timestamp_df['min_timestamp'] = pd.to_datetime(timestamp_df['timestamp'])
        timestamp_df['max_timestamp'] = pd.to_datetime(max_timestamp_df['timestamp'])

        return timestamp_df.drop('timestamp', axis=1).copy()

## --------------------------------------
def read_BLNW_history_chip_measurement_df(sqlite_file: Path, chip_name: str, min_timestamp, max_timestamp):
    with sqlite3.connect(sqlite_file) as sqlite3_connection:
        data_df = pd.read_sql_query(f"SELECT * FROM baselines WHERE chip_name='{chip_name}'", sqlite3_connection)
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        data_df = data_df.loc[data_df.timestamp >= min_timestamp]
        data_df = data_df.loc[data_df.timestamp <= max_timestamp]

        return data_df.copy()

def main(args = None):
    global stop_acquisition, hit_counter

    # Hardware initialization
    kcu = initialize_kcu()
    rb = initialize_readout_board(kcu)
    etroc_chips = initialize_etroc_chips(rb)

    return

    for etroc in etroc_chips:
        etroc.set_power_mode(mode='high', row=0, col=0, broadcast=True)

    # Setup and calibration
    print("\nETROC COSMIC RAY TEST - CONTINUOUS DETECTION")

    if not args.skip_baseline:
        etroc_configs, baseline_storage = calibrate_baselines(etroc_chips, ETROC_NAMES, args.note)
    else:
        
        ### Build etroc_configs
        etroc_configs = []
        all_pixels_per_chip = []
        for _ in range(NUM_ETROC):
            chip_pixels = []
            for row in range(PIXEL_ROW):
                for col in range(PIXEL_COL):
                    chip_pixels.append((row, col))
            all_pixels_per_chip.append(chip_pixels)

        for i, (etroc, chip_name) in enumerate(zip(etroc_chips, ETROC_NAMES)):
            if etroc is not None and i < len(all_pixels_per_chip):
                etroc_configs.append((etroc, chip_name, all_pixels_per_chip[i]))

        ### Build baseline_storage
        sqlite_file = Path(path_to_hist) / 'BaselineHistory.sqlite'
        baseline_storage = {}
        for unique_name in ETROC_NAMES:
            timestamp_df = read_BLNW_history_chip_measurements(sqlite_file, unique_name)
            timestamp_idx = len(timestamp_df) - 1

            min_timestamp = timestamp_df.min_timestamp[timestamp_idx]
            max_timestamp = timestamp_df.max_timestamp[timestamp_idx]

            data_df = read_BLNW_history_chip_measurement_df(sqlite_file, unique_name, min_timestamp, max_timestamp)

            baseline_storage[unique_name] = {
                'row': data_df.row.to_list(),
                'col': data_df.col.to_list(),
                'baseline': data_df.baseline.to_list(),
            }
        
    configure_etroc_for_cosmic(etroc_configs, baseline_storage)
    configure_trigger_system(rb)

    # Data acquisition
    run_cosmic_detection(rb, args)

    # Cleanup
    cleanup_system(etroc_configs, rb)

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

    parser.add_argument(
        '-o',
        '--outdir',
        metavar = 'NAME',
        type = str,
        help = 'output directory name',
        required = True,
        dest = 'outdir',
    )

    parser.add_argument(
        '--note',
        metavar = 'NAME',
        type = str,
        help = 'note for BL and NW history',
        default = '',
        dest = 'note',
    )

    parser.add_argument(
        '--max_run_time',
        metavar = 'NUM',
        type = int,
        help = 'Maximum running time of DAQ in minutes, default is 8 hours',
        default = 480,
        dest = 'max_run_time',
    )

    parser.add_argument(
        '--skip_baseline',
        action='store_true',
        help='Call latest baseline values from the baseline history file.',
        dest='skip_baseline'
    )

    args = parser.parse_args()

    main(args)
