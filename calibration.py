import time, sys
import sqlite3
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm
from tamalero.colors import green, red, yellow

import traceback

# If you still want to use the original utils for saving, keep this import.
# Otherwise, we can write a native saver. For now, I'll wrap the logic you had.
from etroc_utils import convert_dict_to_pandas, save_baselines

class CalibrationManager:
    def __init__(self, system, config):
        """
        Args:
            system: The ETROCSystem instance (from hardware_init.py)
            config: The DAQConfig instance (from settings.py)
        """
        self.sys = system
        self.cfg = config
        self.db_path = Path(self.cfg.path_to_hist) / 'BaselineHistory.sqlite'
        self.fig_path = Path(self.cfg.path_to_figure)

    def run_calibration(self, note="", charge_injection_mode=False):
        """
        Runs the threshold scan on all connected chips.
        If an I2C failure occurs, the script halts.
        """
        baseline_storage = {}

        # 1. Determine Pixel List
        if charge_injection_mode:
            print(yellow(f"[Calibration] Charge Injection Mode: Scanning only {self.cfg.test_pixels}..."))
            pixels_to_scan = self.cfg.test_pixels
        else:
            print(f"[Calibration] Scanning {self.cfg.pixel_row * self.cfg.pixel_col} pixels per chip...")
            pixels_to_scan = [
                (row, col)
                for row in range(self.cfg.pixel_row)
                for col in range(self.cfg.pixel_col)
            ]

        # 2. Scan Loop
        for i, etroc in enumerate(self.sys.etroc_chips):
            chip_name = self.cfg.etroc_names[i]

            if etroc is None:
                print(yellow(f"Skipping {chip_name} (Not connected)"))
                continue

            print(f"Scanning {chip_name}...")
            chip_data = {
                'row': [], 'col': [], 'baseline': [],
                'noise_width': [], 'timestamp': []
            }

            try:
                for row, col in tqdm(pixels_to_scan, desc=f"{chip_name}", leave=False):
                    baseline, noise_width = etroc.auto_threshold_scan(row=row, col=col)

                    chip_data['row'].append(row)
                    chip_data['col'].append(col)
                    chip_data['baseline'].append(baseline)
                    chip_data['noise_width'].append(noise_width)
                    chip_data['pixel_timestamp_utc'].append(datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=' ', timespec='milliseconds'))

                    # Small sleep to prevent bus congestion
                    time.sleep(0.01)

                print(green(f"   [Calibration] Scan completed for {chip_name}"))

            except Exception as e:
                print(red(f"\n[Calibration] FATAL: I2C fault during calibration on {chip_name}: {e}"))
                print(red("Halting DAQ preparation."))
                sys.exit(1)

            # 3. Save Data
            if not charge_injection_mode:
                self._save_to_history(chip_name, chip_data, note)
            else:
                print(yellow(f"   [Note] Charge Injection: Skipping DB save for {chip_name}"))

            # Store in memory for immediate use
            baseline_storage[chip_name] = self._format_data_for_lookup(chip_data)

        print(green("[Calibration] Scan phase completed."))
        return baseline_storage

    def disable_trigger(self):
        print(f"\n[Configuration] Disabling trigger path for all ETROC...")
        for i, etroc in enumerate(self.sys.etroc_chips):
            if etroc is None: continue

            etroc.wr_reg("disTrigPath", 1, broadcast=True)

    def disable_data(self):
        print(f"\n[Configuration] Disabling data readout for all ETROC...")
        for i, etroc in enumerate(self.sys.etroc_chips):
            if etroc is None: continue

            etroc.wr_reg("disDataReadout", 1, broadcast=True)

    def disable_tdc(self):
        print(f"\n[Configuration] Disabling TDC for all ETROC...")
        for i, etroc in enumerate(self.sys.etroc_chips):
            if etroc is None: continue

            etroc.wr_reg("enable_TDC", 0, broadcast=True)

    def enable_trigger(self):
        print(f"\n[Configuration] Enabling trigger path for all ETROC...")
        for i, etroc in enumerate(self.sys.etroc_chips):
            if etroc is None: continue

            etroc.wr_reg("disTrigPath", 0, broadcast=True)

    def enable_trigger_boardIdxs(self, board_idxs):
        for i, etroc in enumerate(self.sys.etroc_chips):
            if i not in board_idxs:
                continue
            etroc.wr_reg("disTrigPath", 0, broadcast = True)

    def enable_trigger_boardIdxs_row_col(self, board_idxs, enable_rows = None, enable_cols = None):
        for i, etroc in enumerate(self.sys.etroc_chips):
            if i not in board_idxs:
                continue
            for row in range(16):
                if enable_rows is None or row in enable_rows:
                    for col in range(16):
                        if enable_cols is None or col in enable_cols:
                            etroc.wr_reg("disTrigPath", 0, row=row, col=col, broadcast = False)

    def set_trigger_window(self, board_idxs = None, reg="TOA", min=0, max=0x3ff):
        if reg == "TOT" and max > 0x1ff:
            max = 0x1ff

        for i, etroc in enumerate(self.sys.etroc_chips):
            if board_idxs is not None and i not in board_idxs:
                continue

            etroc.set_trigger_TH(reg, max, min, 15, 15, broadcast=True)

    def set_data_window(self, board_idxs = None, reg="TOA", min=0, max=0x3ff):
        if reg == "TOT" and max > 0x1ff:
            max = 0x1ff

        for i, etroc in enumerate(self.sys.etroc_chips):
            if board_idxs is not None and i not in board_idxs:
                continue

            etroc.set_data_TH(reg, max, min, 15, 15, broadcast=True)

    def enable_data(self):
        print(f"\n[Configuration] Enabling data readout for all ETROC...")
        for i, etroc in enumerate(self.sys.etroc_chips):
            if etroc is None: continue

            etroc.wr_reg("disDataReadout", 0, broadcast=True)

    def enable_tdc(self):
        print(f"\n[Configuration] Enabling TDC for all ETROC...")
        for i, etroc in enumerate(self.sys.etroc_chips):
            if etroc is None: continue

            etroc.wr_reg("enable_TDC", 1, broadcast=True)

    def standard_enable(self):
        print(f"\n[Configuration] Enabling trigger path, data readout and TDC for all ETROC...")
        for i, etroc in enumerate(self.sys.etroc_chips):
            if etroc is None: continue

            etroc.wr_reg("enable_TDC", 1, broadcast=True)
            etroc.wr_reg("disDataReadout", 0, broadcast=True)
            etroc.wr_reg("disTrigPath", 0, broadcast=True)

    def apply_configuration(self, baseline_storage, charge_injection_mode=False):
        """
        Writes the thresholds (Baseline + Offset) to the chips.
        If an I2C failure occurs, the script halts.
        """
        print(f"\n[Configuration] Configuring pixels for DAQ run...")

        for i, etroc in enumerate(self.sys.etroc_chips):
            chip_name = self.cfg.etroc_names[i]
            if etroc is None: continue

            # Determine which pixels to configure
            if charge_injection_mode:
                pixels_to_config = self.cfg.test_pixels
            else:
                pixels_to_config = [
                    (r, c)
                    for r in range(self.cfg.pixel_row)
                    for c in range(self.cfg.pixel_col)
                ]

            # Apply Pixel Specifics
            lookup = baseline_storage.get(chip_name, {})
            offset = self.cfg.th_offsets.get(chip_name)

            try:
                ## --- Part A: Global/Broadcast Settings ---
                print(f"   Configuring {chip_name} with broadcast...")
                etroc.reset()
                time.sleep(0.1)
                etroc.wr_reg("singlePort", 1)
                etroc.wr_reg("disDataReadout", 1, broadcast=True)
                etroc.wr_reg("QInjEn", 0, broadcast=True)
                etroc.wr_reg("enable_TDC", 0, broadcast=True)
                etroc.wr_reg("disTrigPath", 1, broadcast=True)
                etroc.wr_reg("workMode", 0, broadcast=True)
                etroc.wr_reg("L1Adelay", self.cfg.l1a_delays.get(chip_name), broadcast=True)
                etroc.wr_reg('triggerGranularity', 1)

                ## Global Thresholds (Safe defaults)
                for reg in ['TOA', 'TOT', 'Cal']:
                    max_val = 0x1ff if reg == "TOT" else 0x3ff
                    etroc.set_trigger_TH(reg, max_val, 0, 0, 0, broadcast=True)
                    etroc.set_data_TH(reg, max_val, 0, 0, 0, broadcast=True)

                ## --- Part B: Pixel Specific Settings ---
                print(f"   Configuring {chip_name} ({len(pixels_to_config)} pixels) and (Offset={offset})...")
                count = 0

                for row, col in pixels_to_config:
                    # Enable Pixel
                    etroc.wr_reg("enable_TDC", 1, row=row, col=col, broadcast=False)
                    etroc.wr_reg("disDataReadout", 0, row=row, col=col, broadcast=False)
                    etroc.wr_reg("disTrigPath", 0, row=row, col=col, broadcast=False)

                    # Calculate DAC
                    baseline = lookup.get((row, col), 0)
                    if baseline == 0:
                        applied_dac = 1020
                    else:
                        applied_dac = min(int(baseline + offset), 1023)

                    if charge_injection_mode:
                        print(f"   Applied DAC: {applied_dac}")
                    etroc.wr_reg('DAC', applied_dac, row=row, col=col, broadcast=False)

                    # --- Charge Injection Specific Pixel Settings ---
                    if charge_injection_mode:
                        etroc.wr_reg("QSel", self.cfg.charge_fc, row=row, col=col, broadcast=False)
                        etroc.wr_reg("QInjEn", 1, row=row, col=col, broadcast=False)

                    count += 1
                    if count % 32 == 0: time.sleep(0.01)

                print(green(f"   [Configuration] {chip_name} configured successfully."))

            except Exception as e:
                print(red(f"\n[Configuration] FATAL: I2C fault during chip configuration for {chip_name}: {e}"))
                print(red("Halting DAQ preparation."))
                sys.exit(1)

        print(green("[Configuration] All pixels configured."))

    # --- Internal Helpers ---

    def _format_data_for_lookup(self, data_dict):
        """Converts parallel lists into a (row, col) -> baseline dict."""
        return {
            (r, c): b
            for r, c, b in zip(data_dict['row'], data_dict['col'], data_dict['baseline'])
        }

    def load_from_history(self):
        """Loads the latest baseline values from SQLite."""
        print("\n[Calibration] Loading historical baselines from database...")
        baseline_storage = {}
        hv_values = {}

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")

        for chip_name in self.cfg.etroc_names:
            try:
                df, min_timestamp, max_timestamp = self._fetch_latest_run_df(chip_name)

                # Convert DataFrame to lookup dict: data['row'], data['baseline']
                data_dict = {
                    'row': df.row.tolist(),
                    'col': df.col.tolist(),
                    'baseline': df.baseline.tolist()
                }
                baseline_storage[chip_name] = self._format_data_for_lookup(data_dict)
                print(green(f"   Loaded {len(df)} pixels for {chip_name} between {min_timestamp}, {max_timestamp}"))
                hv_values[chip_name] = df.loc[df['chip_name'] == chip_name, 'hv'].iloc[0]

            except Exception as e:
                print(red(f"   Failed to load history for {chip_name}: {e}"))
                baseline_storage[chip_name] = {} # Empty dict on failure

        return baseline_storage

    def _save_to_history(self, chip_name, data, note):
        """Wrapper for the existing logic to save data."""
        # Using the existing logic you had, leveraging etroc_utils
        try:
            current_hv = self.cfg.hvs.get(chip_name, 0.0)
            df = convert_dict_to_pandas(data, chip_name, current_hv)
            save_baselines(df, chip_name,
                           hist_dir=self.cfg.path_to_hist,
                           fig_dir=self.cfg.path_to_figure,
                           save_notes=note)
        except Exception as e:
            error_details = traceback.format_exc()
            print(red(f"Error saving history for {chip_name}: {e} ({error_details})"))

    def _fetch_latest_run_df(self, chip_name):
        with sqlite3.connect(self.db_path) as conn:
            # 1. Get the LATEST timestamp for the end of the run (15, 15)
            # We use ORDER BY and LIMIT 1 to get just the single latest value immediately.
            q_max = """
                SELECT pixel_timestamp_utc
                FROM baselines
                WHERE chip_name = ? AND ROW = 15 AND COL = 15
                ORDER BY pixel_timestamp_utc DESC
                LIMIT 1
            """
            end_time_df = pd.read_sql_query(q_max, conn, params=(chip_name,))

            if end_time_df.empty:
                raise ValueError("No history found (End of run missing)")

            max_ts_str = end_time_df.iloc[0]['pixel_timestamp_utc']

            # 2. Get the LATEST timestamp for the start of the run (0, 0)
            # We look for the latest (0,0) that happened BEFORE or AT the max_ts
            q_min = """
                SELECT pixel_timestamp_utc
                FROM baselines
                WHERE chip_name = ? AND ROW = 0 AND COL = 0 AND pixel_timestamp_utc <= ?
                ORDER BY pixel_timestamp_utc DESC
                LIMIT 1
            """
            start_time_df = pd.read_sql_query(q_min, conn, params=(chip_name, max_ts_str))

            if start_time_df.empty:
                raise ValueError("No history found (Start of run missing)")

            min_ts_str = start_time_df.iloc[0]['pixel_timestamp_utc']

            # 3. Fetch ONLY the data within that window
            # We bind the specific start and end times to the query.
            q_data = """
                SELECT * FROM baselines
                WHERE chip_name = ?
                AND pixel_timestamp_utc BETWEEN ? AND ?
            """

            bl_data_df = pd.read_sql_query(
                q_data,
                conn,
                params=(chip_name, min_ts_str, max_ts_str)
            )

            # Convert to datetime only for the small result set
            bl_data_df['pixel_timestamp_utc'] = pd.to_datetime(bl_data_df['pixel_timestamp_utc'])

            return bl_data_df, min_ts_str, max_ts_str