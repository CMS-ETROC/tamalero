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
from data_handler import DataWriter, generate_run_dir
from daq_utils import TerminalHandler

def run_daq_loop(system, config, charge_injection_mode=False):
    """
    Main DAQ Execution Loop.
    Handles FIFO reading, file writing via DataWriter, and time/keyboard limits.
    """
    mode_str = "Charge Injection" if charge_injection_mode else "Cosmic/Beam"
    print(f"\n7. Starting continuous {mode_str} run detection...")

    # 1. Setup FIFO and Readout Board
    fifo = FIFO(system.rb)
    fifo.reset()
    system.rb.reset_data_error_count()
    system.rb.enable_etroc_readout()
    system.rb.rerun_bitslip()
    fifo.use_etroc_data()
    system.rb.enable_etroc_trigger()

    # 2. Timing Logic
    start_time = datetime.now(timezone.utc)
    max_minutes = min(config.max_run_time, 1440)  # Cap at 24 hrs
    end_time = start_time + timedelta(minutes=max_minutes)

    print(f"   Start: {start_time.strftime('%H:%M:%S')}")
    print(f"   End:   {end_time.strftime('%H:%M:%S')} (Max {max_minutes} mins)")
    print(yellow("   Press 'q' to stop acquisition"))

    # 3. Setup Terminal (Non-blocking input)
    terminal = TerminalHandler()
    terminal.start_non_blocking()

    # 4. The Data Loop
    try:
        # 'with' block automatically handles file opening/closing/chunking
        with DataWriter(config) as writer:

            while True:
                # A. Check Limits
                if datetime.now(timezone.utc) >= end_time:
                    print(yellow("   Time limit reached."))
                    break

                if terminal.check_for_q():
                    break

                # B. Read Hardware
                try:
                    if charge_injection_mode:
                        fifo.send_Qinj_only(count=config.qinj_count)
                    raw_data = fifo.read(dispatch=True)
                    time.sleep(0.05)  # Small delay to prevent UDP spam
                except Exception as e:
                    print(red(f"FIFO Error: {e}"))
                    time.sleep(1)
                    continue

                # C. Write Data
                if raw_data:
                    writer.write(raw_data)

    except KeyboardInterrupt:
        print(yellow("\n   Keyboard Interrupt (Ctrl+C)"))

    finally:
        # 5. Cleanup
        system.rb.disable_etroc_trigger()
        system.rb.disable_etroc_readout()
        terminal.restore()
        elapsed = datetime.now(timezone.utc) - start_time
        print(green(f"\nRun Complete."))
        print(f"Duration: {str(elapsed).split('.')[0]}")

def main():
    # 0. Parse Arguments
    parser = argparse.ArgumentParser(description='Run Cable Eliminator DAQ')
    parser.add_argument('-o', '--rootdir', type=str, required=True, dest='rootdir', help='Root directory where the Run_XX folder will be created')
    parser.add_argument('--note', type=str, default='', help='Note for baseline history')
    parser.add_argument('--max_run_time', type=int, default=480, help='Max run time in mins')
    parser.add_argument('--skip_baseline', action='store_true', help='Use latest history')
    parser.add_argument('--charge_injection', action='store_true', help='Run in Charge Injection Mode')
    args = parser.parse_args()

    # 1. Setup Configuration
    # We initialize default config, then override with command line args
    config = DAQConfig()
    config.max_run_time = args.max_run_time

    # Define run_type
    run_type = "QInj" if args.charge_injection else "beam"

    config.outdir = generate_run_dir(
        root_path=args.rootdir,
        run_type=run_type,
        note=args.note
    )

    # 2. Initialize Hardware
    system = ETROCSystem(config)
    system.connect()

    # 3. Calibration / Configuration
    cal_mgr = CalibrationManager(system, config)

    if args.skip_baseline:
        # Load most recent baselines from SQLite DB
        baselines = cal_mgr.load_from_history()
    else:
        # Run new hardware scan
        baselines = cal_mgr.run_calibration(note=args.note, charge_injection_mode=args.charge_injection)

    # Apply thresholds (Configuring pixels)
    cal_mgr.apply_configuration(baselines, charge_injection_mode=args.charge_injection)

    # 4. Final Hardware Trigger Setup
    # (Must be done after chip configuration)
    system.configure_trigger()

    # 5. Run DAQ Loop
    run_daq_loop(system, config, args.charge_injection)

    # 6. Final Cleanup
    cal_mgr.disable_trigger()
    cal_mgr.disable_data()
    cal_mgr.disable_tdc()
    print(green("\nRun finished."))

if __name__ == "__main__":
    main()
