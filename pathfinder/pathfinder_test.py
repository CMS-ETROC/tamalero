#!/usr/bin/env python3
"""CLI example derived from pathfinder_test.ipynb, sections 0-12.
Hardware sequences are retained; WS sections are not included.
Default run is an offline preview. Hardware execution has not been validated.
Place this file in the repository's pathfinder directory.
"""
import argparse
import json
from pathlib import Path


def selection(value):
    try:
        obj = json.loads(value)
        if not isinstance(obj, dict) or not obj:
            raise ValueError("selection must be a nonempty JSON object")
        return {int(key): val for key, val in obj.items()}
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mapping", default="pathfinder_mapping.yaml",
                        help="Absolute path or filename relative to repo/pathfinder")
    parser.add_argument("--stage", choices=["preview", "motherboard", "blnw", "qinj"], default="preview",
                        help="Run all prerequisite sections through this stage")
    parser.add_argument("--selection", type=selection, default={1: "all", 3: [1]},
                        help='JSON, e.g. {"1":"all","3":[1]}')
    parser.add_argument("--trigger-selection", type=selection, default=None,
                        help="Defaults to the readout selection")
    parser.add_argument("--trigger-logic", choices=["or", "and"], default="or")
    parser.add_argument("--ip", default="192.168.0.10")
    parser.add_argument("--rb-id", type=int, default=0)
    parser.add_argument("--board-config", default="default")
    parser.add_argument("--monitor-seconds", type=int, default=10)
    parser.add_argument("--pixels", choices=["sample", "all"], default="sample")
    parser.add_argument("--no-save", action="store_true", help="Do not save BL/NW pickle results")
    parser.add_argument("--no-plots", action="store_true", help="Skip BL/NW map and histogram display")
    parser.add_argument("--trigger-delay", type=int, default=466)
    parser.add_argument("--charge-fc", type=int, default=15)
    parser.add_argument("--qinj-count", type=int, default=1)
    parser.add_argument("--interactive", action="store_true",
                        help="Open repeatable diagnostics after a hardware stage")
    return parser



CURRENT_STEP = "Startup"


def announce(title, purpose, expected):
    global CURRENT_STEP
    CURRENT_STEP = title
    print("\n" + "=" * 72, flush=True)
    print(f"[STEP] {title}", flush=True)
    print(f"Action: {purpose}", flush=True)
    print(f"Check:  {expected}", flush=True)
    print("=" * 72, flush=True)


def show_run_plan(args):
    print("\nPATHFINDER TEST - RUN PLAN", flush=True)
    print(f"Requested stage: {args.stage}")
    print(f"Mapping: {args.mapping}")
    print(f"Readout selection: {args.selection}")
    print(f"Trigger selection: {args.trigger_selection if args.trigger_selection is not None else args.selection}")
    print(f"Trigger combination: {args.trigger_logic.upper()}")
    print(f"KCU: {args.ip}; readout board: {args.rb_id}")
    print(f"BL/NW pixels: {args.pixels}; save BL/NW: {not args.no_save}; show BL/NW plots: {not args.no_plots}")
    if args.stage == "preview":
        print("Offline preview only. No hardware initialization will run.")
    elif args.stage == "motherboard":
        print("Hardware configuration will run. Stop point: before ETROC initialization.")
    else:
        print("Hardware configuration will run. Selected ETROC daughter boards must be connected.")
    if args.stage == "qinj":
        print(f"Includes a fresh BL/NW scan, trigger setup and QInj: {args.charge_fc} fC, count={args.qinj_count}.")
    print("Test stages run automatically; an interactive menu follows a hardware stage when --interactive is set.")
    print("On failure, the script stops. Partial hardware configuration may remain.", flush=True)



def interactive_checks(kcu, rb, *, pathfinder_profile=False):
    """Repeat diagnostics without reconstructing KCU or ReadoutBoard."""
    from pathfinder_control import (
        read_optical_link_status, monitor_optical_links,
        read_clock_configuration, reset_optical_fec_counters,
        initialize_pathfinder_optical_links,
    )

    actions = {
        "1": ("KCU status", "READ ONLY", "Read KCU status, clocks and available E-link lock flags."),
        "2": ("Uplink READY/FEC", "READ ONLY", "Read current counters without clearing them."),
        "3": ("Monitor uplinks for 10 seconds", "READ ONLY", "Sample READY/FEC once per second; counters are not reset."),
        "4": ("Clock register readback", "READ ONLY", "Read ECLK0-27 frequency and drive strength; this does not measure waveforms."),
        "5": ("Reset FEC counters", "WRITES HARDWARE", "Clear historical FEC counts on both uplinks."),
        "6": ("Configure and verify optical links", "WRITES HARDWARE", "Apply the tested Pathfinder lpGBT/VTRx+ profile, clear FEC counts and verify for 10 seconds."),
    }
    while True:
        print("\n" + "=" * 72)
        print("Interactive diagnostics - reuse the current hardware connection")
        for key, (title, kind, _) in actions.items():
            suffix = " (unavailable for this mapping)" if key == "6" and not pathfinder_profile else ""
            print(f"{key}. {title} [{kind}]{suffix}")
        print("0. Exit (does not power down hardware)")
        try:
            choice = input("Select an action: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT] Leaving diagnostics; hardware is not restored or powered down.")
            return
        if choice == "0":
            print("[EXIT] Leaving diagnostics; hardware is not powered down.")
            return
        if choice not in actions:
            print("[WARNING] Enter a number from 0 to 6.")
            continue
        if choice == "6" and not pathfinder_profile:
            print("[SKIPPED] This configuration profile is restricted to pathfinder_mapping.yaml.")
            continue
        title, kind, explanation = actions[choice]
        print(f"\n[START] {title} [{kind}]")
        print(f"[INFO] {explanation}")
        try:
            if choice == "1":
                kcu.status()
            elif choice == "2":
                read_optical_link_status(rb)
            elif choice == "3":
                monitor_optical_links(rb, seconds=10)
                print("[CHECK] A saturated counter at 65535 cannot demonstrate error-free operation.")
            elif choice == "4":
                results = read_clock_configuration(rb)
                errors = sum(item["error"] is not None for rows in results.values() for item in rows)
                print(f"[CHECK] Clock register read errors: {errors}")
            elif choice == "5":
                reset_optical_fec_counters(rb)
            elif choice == "6":
                initialize_pathfinder_optical_links(rb, monitor_seconds=10)
                print("[PASS] Optical configuration and verification completed.")
            if choice != "6":
                print("[DONE] Operation completed. Review the readings above; this is not an automatic pass verdict.")
        except KeyboardInterrupt:
            print("\n[STOPPED] Action interrupted. Partial configuration may remain; no rollback performed.")
        except Exception as exc:
            import traceback
            print(f"[FAILED] {title}: {type(exc).__name__}: {exc}")
            print("[INFO] No automatic rollback performed. Review before retrying.")
            traceback.print_exc()
        print("[MENU] Returning to action selection.")


def main(args):
    show_run_plan(args)
    if args.monitor_seconds < 1 or args.qinj_count < 1:
        raise ValueError("monitor-seconds and qinj-count must be positive")

    # Notebook code cell 2
    announce('0. Repository setup', 'Set repository paths and locate drivers.', 'Repository and driver paths must match this test installation.')
    import os
    import sys
    from pathlib import Path

    # Replace this with the repository root on the computer running this kernel.
    # The root contains configs/, address_table/, tamalero/, and pathfinder/.
    TAMALERO_ROOT = args.repo_root.expanduser().resolve()

    if not (TAMALERO_ROOT / "tamalero" / "ETROC.py").is_file():
        raise FileNotFoundError(f"Incorrect repository root: {TAMALERO_ROOT}")

    if not (TAMALERO_ROOT / "pathfinder" / "pathfinder_control.py").is_file():
        raise FileNotFoundError(f"Missing pathfinder_control.py in {TAMALERO_ROOT / 'pathfinder'}")

    # Restart the kernel before using this cell to switch driver repositories.
    sys.path.insert(0, str(TAMALERO_ROOT))
    sys.path.insert(0, str(TAMALERO_ROOT / "pathfinder"))
    os.environ["TAMALERO_BASE"] = str(TAMALERO_ROOT)
    # Legacy drivers resolve configs/ relative to the repository root.
    os.chdir(TAMALERO_ROOT)

    print("Python:", sys.executable)
    print("Repository:", TAMALERO_ROOT)
    print("Working directory:", Path.cwd())

    # Notebook code cell 4
    announce('1. Test parameters', 'Load control functions and apply command-line selections.', 'Review mapping, chip selection and connection parameters.')
    from html import escape
    from IPython.display import HTML, display
    from tqdm.auto import tqdm


    def show_status(message, level="info"):
        print(f"[{level.upper()}] {message}")


    from pathlib import Path
    import os
    import inspect

    from pathfinder_control import (
        read_clock_configuration,
        load_mapping, expand_selection, build_elinks, build_readout_masks,
        initialize_etrocs, InitializationError,
        check_connections, check_links, configure_readout_links, build_trigger_mask, initialize_pathfinder_optical_links,
    )

    MAPPING_PATH = Path(args.mapping) if Path(args.mapping).is_absolute() else TAMALERO_ROOT / "pathfinder" / args.mapping
    ENABLE_SELECTION = args.selection
    # ENABLE_SELECTION = {m: "all" for m in range(1, 8)}
    # For Eliminator, replace both settings above with:
    # MAPPING_PATH = TAMALERO_ROOT / "pathfinder" / "eliminator_mapping.yaml"
    # ENABLE_SELECTION = {1: [1], 3: [1], 4: [1]}
    # ENABLE_SELECTION = {m: "all" for m in range(1, 5)}

    KCU_IP = args.ip
    READOUTBOARD_ID = args.rb_id
    READOUTBOARD_CONFIG = args.board_config
    ELINK_WIDTH = 0x3  # Retained from the existing test notebook.
    TAMALERO_BASE = os.environ.get("TAMALERO_BASE")

    basic_checks_passed = False

    # Notebook code cell 6
    announce('2. Wiring preview', 'Resolve selected ETROCs using the mapping YAML.', 'I2C routes and readout masks should match the connected hardware.')
    mapping = load_mapping(MAPPING_PATH)
    ids = expand_selection(mapping, ENABLE_SELECTION)
    USE_TRIG = "trig" in mapping["lpgbts"]
    basic_checks_passed = False

    print(f"Selected {len(ids)} ETROCs: {ids}")
    for identity in ids:
        cfg = mapping["etrocs"][identity]
        bus = cfg["i2c"]
        print(f"{identity}: I2C={bus['lpgbt']}/M{bus['channel']}/"
              f"0x{bus['address']:02X}, elinks={build_elinks(mapping, identity)}")

    preview_masks = build_readout_masks(mapping, ids)
    for bank, mask in preview_masks.items():
        print(f"Bank {bank} disable mask: 0x{mask:07X}")

    # Notebook code cell 8
    announce('2.1. Trigger preview', 'Calculate the trigger mask without writing hardware.', 'Trigger sources must be included in the readout selection.')
    # Select trigger sources independently from readout selection.
    TRIGGER_SELECTION = args.trigger_selection if args.trigger_selection is not None else args.selection
    TRIGGER_COMBINATION_LOGIC = 1 if args.trigger_logic == "and" else 0

    trigger_ids = expand_selection(mapping, TRIGGER_SELECTION)

    if not set(trigger_ids).issubset(set(ids)):
        raise ValueError("Trigger sources must be included in ENABLE_SELECTION")

    preview_trigger_mask = build_trigger_mask(mapping, trigger_ids)

    print("Trigger ETROCs:", trigger_ids)
    print(f"Trigger mask: 0x{preview_trigger_mask:07X}")

    if args.stage == "preview":
        if args.interactive:
            print("[INFO] Interactive hardware diagnostics are skipped in offline preview mode.")
        print('Offline preview complete. Next: --stage motherboard to configure and check the motherboard.', flush=True)
        return

    # Notebook code cell 10
    announce('3. KCU connection', 'Connect to KCU and perform the register loopback test.', 'Loopback must pass. Initial optical status is not the final post-configuration result.')
    from tamalero.KCU import KCU
    from tamalero.ReadoutBoard import ReadoutBoard
    from tamalero.ETROC import ETROC

    if "i2c_master" not in inspect.signature(ETROC.__init__).parameters:
        raise RuntimeError("The imported ETROC driver does not support i2c_master")
    if not TAMALERO_BASE:
        raise RuntimeError("Set TAMALERO_BASE to the existing Tamalero project root")

    xml_path = Path(TAMALERO_BASE) / "address_table" / "generic" / "etl_test_fw.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)

    print(f"ETROC driver: {inspect.getfile(ETROC)}")
    print(f"Address table: {xml_path}")
    kcu = KCU(
        name="kcu",
        ipb_path=f"chtcp-2.0://localhost:10203?target={KCU_IP}:50001",
        adr_table=str(xml_path),
    )
    kcu.status()
    loopback_value = 0xABCD1234
    kcu.write_node("LOOPBACK.LOOPBACK", loopback_value)
    received = kcu.read_node("LOOPBACK.LOOPBACK").value()
    if received != loopback_value:
        raise RuntimeError(f"Loopback failed: wrote 0x{loopback_value:X}, read 0x{received:X}")
    show_status("KCU loopback passed", "success")

    # Notebook code cell 12
    announce('4. Motherboard initialization', 'Create rb, configure declared lpGBTs, mask data links and configure TRIG clocks.', 'Both declared lpGBT objects must be available; ELINK_WIDTH must read back correctly.')
    USE_TRIG = "trig" in mapping["lpgbts"]
    if USE_TRIG:
        upstream = mapping["lpgbts"]["trig"]["control"]
        if (upstream["via"], upstream["channel"], upstream["address"]) != ("daq", 0, 0x72):
            raise RuntimeError("The YAML TRIG route differs from the current DAQ M0 / 0x72 driver route")

    basic_checks_passed = False
    chips = {}
    initialized_ids = ()
    etroc_register_checks_done = False
    etroc_power_checks_done = False
    blnw_scan_passed = False
    rb = ReadoutBoard(
        rb=READOUTBOARD_ID, kcu=kcu, config=READOUTBOARD_CONFIG,
        trigger=USE_TRIG, verbose=True,
    )
    configure_readout_links(rb, mapping, [])
    if USE_TRIG and (not rb.trigger or getattr(rb, "TRIG_LPGBT", None) is None):
        raise RuntimeError("TRIG lpGBT was not initialized; readout links remain masked")

    # Retain the TRIG clock configuration verified during hardware testing.
    if USE_TRIG:
        rb.TRIG_LPGBT.configure_clocks(0x0fffffff)
        show_status("TRIG ECLK0-27 configured; verify readback below", "info")

    width_node = f"READOUT_BOARD_{rb.rb}.ELINK_WIDTH"
    rb.kcu.write_node(width_node, ELINK_WIDTH)
    width = rb.kcu.read_node(width_node).value()
    if width != ELINK_WIDTH:
        raise RuntimeError(f"ELINK_WIDTH mismatch: expected {ELINK_WIDTH}, received {width}")
    print(f"Initialized controllers: {list(mapping['lpgbts'])}; declared banks are masked")

    # Notebook code cell 14
    announce('4.1. Initial link diagnostics', 'Read existing link status and reset FEC counters after the existing DAQ check.', 'This is before optical configuration; use section 4.3 for final dual-link verification.')
    print("Checking DAQ/TRIG LpGBT base configuration:")
    daq_link_good = rb.DAQ_LPGBT.link_status()
    show_status(f"DAQ LpGBT Link Good: {daq_link_good}", "success" if daq_link_good else "error")
    if USE_TRIG:
        trig_link_good = rb.TRIG_LPGBT.link_status()
        show_status(f"TRIG LpGBT Link Good: {trig_link_good}", "success" if trig_link_good else "error")
    fec_errors_before_reset = rb.get_FEC_error_count()
    # print(f"FEC Errors: DAQ LpGBT = {fec_errors_before_reset.get('DAQ', 'N/A')}")

    if not daq_link_good:
        basic_checks_passed = False
        raise RuntimeError("DAQ lpGBT link is not good; stop before ETROC initialization")

    # Preserve the observed counters above before starting a new counting interval.
    # rb.reset_FEC_error_count()

    # Notebook code cell 16
    announce('4.2. Clock register readback', 'Read ECLK frequency and drive registers.', 'Register readback does not measure a physical clock waveform.')
    clock_readback = read_clock_configuration(rb)

    clock_read_errors = sum(
        result["error"] is not None
        for results in clock_readback.values()
        for result in results
    )
    show_status(f"Clock register read errors: {clock_read_errors}", "success" if clock_read_errors == 0 else "error")

    # Notebook code cell 18
    announce('4.3. Optical-link configuration', 'Apply the Pathfinder transmitter profile when selected, then verify readback and monitor uplinks.', 'For Pathfinder, both READY flags must stay 1 and FEC counters stay 0 at the sampled times.')
    optical_checks_passed = False
    optical_init_result = None

    if Path(MAPPING_PATH).name == "pathfinder_mapping.yaml":
        optical_init_result = initialize_pathfinder_optical_links(
            rb,
            monitor_seconds=args.monitor_seconds,
        )
        optical_checks_passed = True
    else:
        show_status(
            "Skipping Pathfinder optical initialization for this mapping",
            "info",
        )

    # ADC monitoring uses the selected YAML, not the driver's legacy default map.
    announce('4.4. lpGBT ADC monitoring',
             'Load mapped ADC channels, calibrate and read DAQ/TRIG monitoring values.',
             'Limits and conversion factors are provisional; missing or unpowered daughter cards may show ERR.')
    if "adc_monitoring" not in mapping:
        show_status("Skipping ADC monitoring: selected YAML has no adc_monitoring section", "info")
    else:
        from pathfinder_control import configure_adc_mapping
        adc_groups = configure_adc_mapping(rb, mapping)
        for adc_bank, adc_entries in adc_groups.items():
            if not adc_entries:
                continue
            adc_lpgbt = rb.DAQ_LPGBT if adc_bank == "daq" else rb.TRIG_LPGBT
            print(f"\n{adc_bank.upper()} ADC monitoring ({len(adc_entries)} entries):")
            adc_lpgbt.calibrate_adc()
            adc_lpgbt.read_adcs(check=True, strict_limits=False)
        show_status("ADC reads completed. Review the table; completion does not mean all values passed.", "info")

    if args.stage == "motherboard":
        print('Motherboard stage complete. With no daughter boards, stop here. Review whether the Pathfinder optical profile ran or was skipped.', flush=True)
        if args.interactive:
            interactive_checks(kcu, rb, pathfinder_profile=Path(MAPPING_PATH).name == "pathfinder_mapping.yaml")
        return

    # Notebook code cell 20
    announce('5. ETROC initialization', 'Create the selected ETROC objects.', 'Selected chips must be connected; motherboard-only runs stop before this step.')
    basic_checks_passed = False
    chips = {}
    initialized_ids = ()
    etroc_register_checks_done = False
    etroc_power_checks_done = False
    blnw_scan_passed = False
    configure_readout_links(rb, mapping, [])
    try:
        chips = initialize_etrocs(rb, mapping, ids, strict=False, verbose=False)
    except InitializationError as exc:
        chips = exc.initialized_chips
        print(f"Failed ETROC: {exc.failed_id}; completed: {list(chips)}")
        raise
    initialized_ids = tuple(ids)
    show_status(f"Initialized {len(chips)} ETROCs: {list(chips)}", "success")

    # Notebook code cell 22
    announce('5.1. ETROC communication', 'Check communication with the initialized chips.', 'Inspect the per-chip connection results.')
    basic_checks_passed = False
    if tuple(ids) != initialized_ids or list(chips) != ids:
        raise RuntimeError("Selection changed or initialization is incomplete; rerun initialization")
    connection_results = check_connections(chips)
    for identity, result in connection_results.items():
        show_status(f"{identity}: connected={result['connected']}, error={result['error']}", "success" if result["connected"] else "error")
    if not all(result["connected"] for result in connection_results.values()):
        raise RuntimeError("One or more ETROCs failed the I2C check")

    # Notebook code cell 24
    announce('5.2. ETROC register checks', 'Run the notebook register configuration and checks.', 'Any reported register failure must be investigated.')
    basic_checks_passed = False
    etroc_register_checks_done = False
    etroc_power_checks_done = False
    blnw_scan_passed = False
    if not ids or tuple(ids) != initialized_ids or list(chips) != ids:
        raise RuntimeError("Selection is empty or initialization is incomplete; rerun initialization")
    configure_readout_links(rb, mapping, [])
    etroc_register_results = {}
    for identity, etroc in chips.items():
        values = {}
        etroc_register_results[identity] = values
        try:
            if not etroc.is_connected():
                raise RuntimeError("ETROC is not responding")
            etroc.wr_reg("PS_CapRst", 1)
            etroc.wr_reg("PS_CapRst", 0)
            for register in ("disScrambler", "controllerState", "pllUnlockCount", "L1Adelay"):
                values[register] = etroc.rd_reg(register)
            print(f"{identity}: {values}")
            if values["disScrambler"] != 1:
                show_status(f"{identity}: disScrambler differs from the legacy expected value 1", "warning")
            if values["controllerState"] != 11:
                show_status(f"{identity}: controllerState differs from the legacy expected value 11", "warning")
        except Exception as exc:
            values["error"] = str(exc)
            raise RuntimeError(f"ETROC register configuration failed for {identity}; readout remains masked") from exc
    etroc_register_checks_done = True

    # Notebook code cell 26
    announce('5.3. ETROC power configuration', 'Apply and check the existing ETROC power settings.', 'Power-mode checks must pass before scanning.')
    basic_checks_passed = False
    etroc_power_checks_done = False
    blnw_scan_passed = False
    if tuple(ids) != initialized_ids or list(chips) != ids:
        raise RuntimeError("Selection changed or initialization is incomplete")
    if not etroc_register_checks_done:
        raise RuntimeError("Complete the ETROC register configuration cell first")
    configure_readout_links(rb, mapping, [])
    etroc_power_results = {}
    for identity, etroc in chips.items():
        try:
            etroc.set_power_mode(mode="high", row=0, col=0, broadcast=True)
        except Exception as exc:
            raise RuntimeError(f"High power configuration failed for {identity}; readout remains masked") from exc

    for identity, etroc in chips.items():
        try:
            mode = etroc.get_power_mode(row=8, col=8)
            etroc_power_results[identity] = mode
            print(f"{identity}: power mode at pixel (8, 8) = {mode}")
            if mode != "high":
                raise RuntimeError(f"Expected high power mode, received {mode!r}")
        except Exception as exc:
            raise RuntimeError(f"Power mode verification failed for {identity}; readout remains masked") from exc
    etroc_power_checks_done = True
    show_status("High power mode configured; sampled pixel readbacks passed", "success")

    # Notebook code cell 28
    announce('6. BL/NW scan settings', 'Apply pixel coverage and BL/NW save options.', 'sample scans 8 pixels per chip; all scans 256 pixels per chip.')
    # Choose "sample" for 8 pixels or "all" for the full 16 x 16 matrix.
    BLNW_SCAN_MODE = args.pixels
    BLNW_SAVE_RESULTS = not args.no_save
    BLNW_OUTPUT_DIR = TAMALERO_ROOT / "pathfinder" / "results" / "blnw" 

    # Notebook code cell 30
    announce('6.1. BL/NW scan', 'Scan selected pixels using the original notebook sequence.', 'Follow the progress bar and failure summary; saved results use BLNW_OUTPUT_DIR.')
    import pickle
    import io
    from contextlib import redirect_stdout
    from datetime import datetime

    def build_blnw_pixels(mode):
        if mode == "sample":
            return [(row, col) for row in (0, 15) for col in (0, 7, 8, 15)]
        if mode == "all":
            return [(row, col) for row in range(16) for col in range(16)]
        raise ValueError("BLNW_SCAN_MODE must be 'sample' or 'all'")


    basic_checks_passed = False
    blnw_scan_passed = False
    scan_pixels = build_blnw_pixels(BLNW_SCAN_MODE)
    if not ids or tuple(ids) != initialized_ids or list(chips) != ids:
        raise RuntimeError("Selection is empty or initialization is incomplete")
    if not etroc_register_checks_done or not etroc_power_checks_done:
        raise RuntimeError("Complete ETROC register and power configuration first")
    configure_readout_links(rb, mapping, [])
    blnw_save_enabled = BLNW_SAVE_RESULTS
    blnw_output_path = None
    if blnw_save_enabled:
        BLNW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        blnw_output_path = BLNW_OUTPUT_DIR / (
            "blnw_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".pkl"
        )
    baseline_dict = {identity: {} for identity in ids}
    nw_dict = {identity: {} for identity in ids}
    failed_pixels = {identity: [] for identity in ids}
    blnw_results = {
        "baseline": baseline_dict,
        "NW": nw_dict,
        "chip_name": list(ids),
        "failed_pixels": failed_pixels,
        "records": [],
        "scan_mode": BLNW_SCAN_MODE,
        "pixels": scan_pixels,
        "mapping_path": str(MAPPING_PATH),
        "started_at": datetime.now().isoformat(),
        "completed": False,
    }


    def save_blnw_results():
        if not blnw_save_enabled:
            return
        temporary_path = blnw_output_path.with_suffix(".tmp")
        with temporary_path.open("wb") as handle:
            pickle.dump(blnw_results, handle)
        temporary_path.replace(blnw_output_path)


    save_blnw_results()
    print(f"Scanning {len(scan_pixels)} pixels per ETROC on {len(chips)} ETROCs")
    if blnw_save_enabled:
        print(f"Results: {blnw_output_path.resolve()}")
    else:
        print("Local saving is disabled; results remain in blnw_results")

    scan_failure_count = 0
    try:
        for identity, etroc in chips.items():
            progress = tqdm(total=len(scan_pixels), desc=f"BL/NW scan: {identity}",
                            unit="pixel", leave=True)
            try:
                for row, col in scan_pixels:
                    record = {"chip": identity, "row": row, "col": col,
                              "baseline": None, "noise_width": None, "error": None}
                    driver_output = io.StringIO()
                    progress.set_postfix(row=row, col=col,
                                         failed=len(failed_pixels[identity]), refresh=False)
                    try:
                        # Retain unsolicited driver output in the result, not in the cell output.
                        with redirect_stdout(driver_output):
                            baseline, noise_width = etroc.auto_threshold_scan(
                                row=row, col=col, broadcast=False,
                                offset=1023, use=False, verbose=False,
                            )
                        if baseline is None or noise_width is None:
                            raise RuntimeError("Scan returned an incomplete result")
                        record.update(baseline=baseline, noise_width=noise_width)
                    except Exception as exc:
                        record["error"] = str(exc)
                        failed_pixels[identity].append((row, col))
                        scan_failure_count += 1
                    record["driver_output"] = driver_output.getvalue()
                    record["timestamp"] = datetime.now().isoformat()
                    baseline_dict[identity][(row, col)] = record["baseline"]
                    nw_dict[identity][(row, col)] = record["noise_width"]
                    blnw_results["records"].append(record)
                    save_blnw_results()
                    progress.set_postfix(failed=len(failed_pixels[identity]), refresh=False)
                    progress.update(1)
            finally:
                progress.close()
        blnw_results["completed"] = True
    finally:
        blnw_results["last_saved_at"] = datetime.now().isoformat()
        save_blnw_results()

    failure_count = sum(len(pixels) for pixels in failed_pixels.values())
    show_status(f"BL/NW scan finished: {len(blnw_results['records'])} pixels, "
                f"{failure_count} failed", "error" if failure_count else "success")
    for identity in ids:
        count = len(failed_pixels[identity])
        show_status(f"{identity}: {len(scan_pixels) - count}/{len(scan_pixels)} pixels passed",
                    "warning" if count else "success")
    if failure_count:
        raise RuntimeError(f"{failure_count} pixels failed; inspect failed_pixels and blnw_results; readout remains masked")
    blnw_scan_passed = True

    # Notebook code cell 32
    announce('6.2. BL/NW figures', 'Display maps and applicable histograms unless --no-plots is set.', 'A plot window may block execution; close it to continue.')
    if not args.no_plots:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, FormatStrFormatter
        from datetime import datetime


        def format_scan_time(results):
            """Use the saved scan start time; never substitute the plotting time."""
            timestamp = results.get("started_at")
            if not timestamp:
                return "Scan time unavailable"
            try:
                return "Scan: " + datetime.fromisoformat(str(timestamp)).isoformat(sep=" ", timespec="seconds")
            except (TypeError, ValueError):
                return "Scan: " + str(timestamp)


        def build_blnw_matrix(values):
            matrix = np.full((16, 16), np.nan)
            for (row, col), value in values.items():
                if value is not None:
                # if value is not None and value != 0:
                    matrix[row, col] = float(value)
            return matrix


        def plot_blnw_maps(results):
            figures = []
            for identity in results["chip_name"]:
                fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
                for ax, key, title in zip(axes, ("baseline", "NW"), ("Baseline", "Noise width")):
                    matrix = build_blnw_matrix(results[key][identity])
                    cmap = plt.get_cmap("viridis").copy()
                    cmap.set_bad("white")
                    view = ax.imshow(np.ma.masked_invalid(matrix), origin="lower",
                                     cmap=cmap, interpolation="nearest")
                    # ax.set_title(f"{identity} - {title}")
                    ax.text(0.01, 1.00, "CMS", transform=ax.transAxes,
                            fontsize=13, fontweight="bold", va="bottom")
                    ax.text(0.11, 1.00, "ETL ETROC", transform=ax.transAxes,
                            fontsize=10, style="italic", va="bottom")
                    ax.text(1.0, 1.025, f"{identity}-{title} 2D", transform=ax.transAxes,
                            fontsize=9, ha="right", va="bottom")
                    ax.text(1.0, 1.00, format_scan_time(results), transform=ax.transAxes,
                            fontsize=9, ha="right", va="bottom")
                    ax.set_xlabel("Column")
                    ax.set_ylabel("Row")
                    ax.set_xticks(range(16))
                    ax.set_yticks(range(16))
                    ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
                    ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
                    ax.grid(which="minor", color="lightgray", linewidth=0.4)
                    ax.tick_params(which="minor", bottom=False, left=False)
                    for row, col in np.argwhere(np.isfinite(matrix)):
                        value = matrix[row, col]
                        rgba = view.cmap(view.norm(value))
                        brightness = sum(weight * component for weight, component in
                                         zip((0.299, 0.587, 0.114), rgba[:3]))
                        label = f"{value:.0f}" if key == "NW" else f"{value:g}"
                        ax.text(col, row, label, ha="center", va="center",
                                fontsize=6, color="black" if brightness > 0.5 else "white")
                    if np.isfinite(matrix).any():
                        colorbar = fig.colorbar(view, ax=ax, label="DAC code")
                        if key == "NW":
                            colorbar.locator = MaxNLocator(integer=True)
                            colorbar.formatter = FormatStrFormatter("%.0f")
                            colorbar.update_ticks()
                    else:
                        ax.set_title(f"{identity} - {title} (no valid results)")
                # fig.suptitle("Unscanned or failed pixels are blank")
                figures.append(fig)
                plt.show()
            return figures


        def plot_blnw_histograms(results):
            figures = []
            if results.get("scan_mode") != "all":
                return figures
            for identity in results["chip_name"]:
                fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
                for ax, key, title in zip(axes, ("baseline", "NW"), ("Baseline", "Noise width")):
                    matrix = build_blnw_matrix(results[key][identity])
                    values = matrix[np.isfinite(matrix) & (matrix!=0)]
                    # ax.set_title(f"{identity} - {title} (valid pixels: {values.size}/256)")
                    ax.text(0.01, 1.00, "CMS", transform=ax.transAxes,
                            fontsize=13, fontweight="bold", va="bottom")
                    ax.text(0.11, 1.00, "ETL ETROC", transform=ax.transAxes,
                            fontsize=10, style="italic", va="bottom")
                    ax.text(1.0, 1.025, f"{identity}-{title} Histo", transform=ax.transAxes,
                            fontsize=9, ha="right", va="bottom")
                    ax.text(1.0, 1.00, format_scan_time(results), transform=ax.transAxes,
                            fontsize=9, ha="right", va="bottom")
                    ax.set_xlabel(f"{title} (DAC code)")
                    ax.set_ylabel("Pixel count")
                    if key == "baseline":
                        bins = np.linspace(0, 1023, 65)
                        ax.set_xlim(0, 1023)
                        ax.set_xticks([0, 128, 256, 384, 512, 640, 768, 896, 1023])
                    else:
                        bins = np.arange(-0.5, 18, 1)
                        ax.set_xlim(0, 17)
                        ax.set_xticks(range(18))
                    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                    if values.size:
                        # ax.hist(values, bins=bins, edgecolor="black", alpha=0.8)
                        ax.hist(values, bins=bins, histtype="step", color="#7ec8e3", linewidth=1.3)
                        ax.text(0.97, 0.95,
                                # f"Mean: {values.mean():.2f}\nStd: {values.std():.2f}",
                                # transform=ax.transAxes, ha="right", va="top")
                                f"Mean: {values.mean():.2f}, Std: {values.std():.2f}",
                                transform=ax.transAxes, ha="right", va="top")
                    else:
                        ax.text(0.5, 0.5, "No valid results", transform=ax.transAxes,
                                ha="center", va="center")
                    # ax.grid(axis="y", alpha=0.25)
                figures.append(fig)
                plt.show()
            return figures


        # This cell uses in-memory results and does not save image files.
        blnw_figures = plot_blnw_maps(blnw_results)

        blnw_histogram_figures = plot_blnw_histograms(blnw_results)

    if args.stage == "blnw":
        print('BL/NW stage complete. Review scan failures and figures. Use --stage qinj for a new run including initialization and a fresh scan.', flush=True)
        if args.interactive:
            interactive_checks(kcu, rb, pathfinder_profile=Path(MAPPING_PATH).name == "pathfinder_mapping.yaml")
        return

    # Notebook code cell 34
    announce('7. Enable selected data links', 'Enable mapped data links and check ETROC link lock.', 'This checks ETROC E-links, separately from the optical uplinks.')
    basic_checks_passed = False
    if tuple(ids) != initialized_ids or list(chips) != ids:
        raise RuntimeError("Selection changed or initialization is incomplete")
    if not etroc_register_checks_done or not etroc_power_checks_done:
        raise RuntimeError("Complete ETROC register and power configuration before enabling links")
    if not blnw_scan_passed:
        raise RuntimeError("Complete the BL/NW scan successfully before enabling links")
    try:
        connection_results = check_connections(chips)
        if not all(result["connected"] for result in connection_results.values()):
            raise RuntimeError(f"I2C checks failed: {connection_results}")
        applied_masks = configure_readout_links(rb, mapping, ids)
        rb.rerun_bitslip()
        link_results = check_links(rb, mapping, ids)
        for identity, result in link_results.items():
            for link in result["links"]:
                print(f"{identity}: {link['lpgbt']} E-link {link['elink']}, "
                      f"locked={link['locked']}, error={link['error']}")
        if not all(result["all_locked"] for result in link_results.values()):
            raise RuntimeError("One or more selected data links are not locked")
    except Exception:
        try:
            configure_readout_links(rb, mapping, [])
        except Exception as cleanup_error:
            print(f"Failed to mask readout links: {cleanup_error}")
        raise

    basic_checks_passed = True
    show_status("Selected ETROCs respond and selected data links are locked", "success")
    print("Only selected readout links are enabled; no acquisition has been started")

    # Notebook code cell 36
    announce('8. Pixel thresholds', 'Apply baseline-derived DAC thresholds and activate scanned pixels.', 'Threshold configuration must complete before acquisition.')
    import time

    THRESHOLD_OFFSET = 50   # DAC code; threshold = baseline + this offset
    L1A_DELAY_DEFAULT = 0x01f5   # = 501 decimal; matches the L1Adelay readback in section 5.2

    # Full-range TOA/TOT/Cal trigger- and data-path windows: accept every hit
    # instead of filtering by pulse shape. Narrow these later if you actually
    # want TOA/TOT/Cal-based selection.
    TOA_RANGE = (0, 0x3ff)
    TOT_RANGE = (0, 0x1ff)
    CAL_RANGE = (0, 0x3ff)

    if not basic_checks_passed:
        raise RuntimeError("Complete section 7 (link enable) successfully before configuring thresholds")
    if not blnw_scan_passed:
        raise RuntimeError("Complete the BL/NW scan before configuring thresholds")


    def configure_thresholds(chips, blnw_results, offset=THRESHOLD_OFFSET,
                              l1a_delay=L1A_DELAY_DEFAULT, reset_first=True,
                              toa_range=TOA_RANGE, tot_range=TOT_RANGE, cal_range=CAL_RANGE):
        """Apply DAC threshold (baseline + offset), activate the pixels that
        were BL/NW-scanned in section 6.1, and open their TOA/TOT/Cal trigger-
        and data-path windows. Reuses baseline values already in memory; does
        not reload from disk. A pixel with baseline=None (failed scan) is
        skipped, not assigned a threshold.

        reset_first=True calls etroc.reset() per chip, which internally also
        calls rb.rerun_bitslip() (see tamalero/ETROC.py). Re-run the link-lock
        check in section 7 afterward if you need to confirm links are still
        locked.
        """
        scan_pixels = blnw_results["pixels"]
        threshold_results = {}
        skipped_pixels = {}

        for identity, etroc in chips.items():
            if reset_first:
                etroc.reset()
                time.sleep(0.1)
                etroc.wr_reg("singlePort", 1)
                etroc.wr_reg("disDataReadout", 1, broadcast=True)
                etroc.wr_reg("QInjEn", 0, broadcast=True)
                etroc.wr_reg("enable_TDC", 0, broadcast=True)
                etroc.wr_reg("disTrigPath", 1, broadcast=True)
                etroc.wr_reg("workMode", 0, broadcast=True)

            applied = {}
            skipped = []
            for row, col in scan_pixels:
                baseline = blnw_results["baseline"][identity][(row, col)]
                if baseline is None:
                    skipped.append((row, col))
                    continue
                dac = baseline + offset

                etroc.wr_reg("workMode", 0, row=row, col=col, broadcast=False)
                etroc.wr_reg("enable_TDC", 1, row=row, col=col, broadcast=False)
                etroc.wr_reg("disDataReadout", 0, row=row, col=col, broadcast=False)
                etroc.wr_reg("disTrigPath", 0, row=row, col=col, broadcast=False)
                etroc.wr_reg("L1Adelay", l1a_delay, row=row, col=col, broadcast=False)
                etroc.wr_reg("DAC", dac, row=row, col=col, broadcast=False)

                etroc.set_trigger_TH("TOA", lower=toa_range[0], upper=toa_range[1], row=row, col=col, broadcast=False)
                etroc.set_trigger_TH("TOT", lower=tot_range[0], upper=tot_range[1], row=row, col=col, broadcast=False)
                etroc.set_trigger_TH("Cal", lower=cal_range[0], upper=cal_range[1], row=row, col=col, broadcast=False)
                etroc.set_data_TH("TOA", lower=toa_range[0], upper=toa_range[1], row=row, col=col, broadcast=False)
                etroc.set_data_TH("TOT", lower=tot_range[0], upper=tot_range[1], row=row, col=col, broadcast=False)
                etroc.set_data_TH("Cal", lower=cal_range[0], upper=cal_range[1], row=row, col=col, broadcast=False)

                applied[(row, col)] = {"baseline": baseline, "dac": dac}

            threshold_results[identity] = applied
            skipped_pixels[identity] = skipped
            level = "warning" if skipped else "success"
            show_status(f"{identity}: threshold applied to {len(applied)} pixels "
                        f"(offset={offset}), {len(skipped)} skipped (no baseline)", level)

        return threshold_results, skipped_pixels


    threshold_results, threshold_skipped_pixels = configure_thresholds(chips, blnw_results)

    # Notebook code cell 38
    announce('9. Self-trigger configuration', 'Apply the selected trigger mask, combination logic and delay.', 'Inspect mask, lock and FC diagnostic output.')
    import time

    if not basic_checks_passed:
        raise RuntimeError("Complete section 7 (link enable) successfully before configuring self-trigger")

    TRIGGER_DELAY_DEFAULT = args.trigger_delay


    def configure_self_trigger(rb, mapping, ids, chips, trigger_delay=TRIGGER_DELAY_DEFAULT,
                                trigger_data_size=0x1, reset_pulses=True,
                                trigger_board_ids=None, trigger_combination_logic=0):
        """Readout-board-level self-trigger enable.

        trigger_board_ids: which board(s) generate the self-trigger, used to
        compute TRIG_ENABLE_MASK. Independent from `ids` (which boards are read
        out) - a board can be read out without being a trigger source. Defaults
        to only the first selected board.
        """

        if trigger_combination_logic not in (0, 1):
            raise ValueError("Trigger combination must be 0 (OR) or 1 (AND)")

        if trigger_board_ids is None:
            trigger_board_ids = [ids[0]]

        if reset_pulses:
            rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.LINK_RESET_PULSE", 0x1)
            rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.ECR_PULSE", 0x1)
            rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.BC0_PULSE", 0x1)

        trigger_enable_mask = build_trigger_mask(mapping, trigger_board_ids)

        rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK", trigger_enable_mask)
        rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_COMBINATION_LOGIC", trigger_combination_logic)
        rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_DATA_SIZE", trigger_data_size)
        rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.TRIG_DLY_SEL", trigger_delay)

        show_status(f"Trigger board(s): {trigger_board_ids}", "info")
        show_status(f"TRIG_ENABLE_MASK: 0x{trigger_enable_mask:X}", "info")
        show_status(f"TRIG_DATA_SIZE: {trigger_data_size}", "info")
        show_status(f"TRIG_DLY_SEL: {trigger_delay}", "info")
        time.sleep(0.1)

        for identity in ids:
            for bank, elinks in build_elinks(mapping, identity).items():
                for elink in elinks:
                    locked = rb.etroc_locked(elink, slave=(bank == 1))
                    print(f"{identity} elink {elink} (bank {bank}): locked={locked}")
                    if not locked:
                        show_status(f"{identity} elink {elink} not locked, re-running bitslip", "warning")
                        rb.rerun_bitslip()
                        time.sleep(0.5)
                        locked = rb.etroc_locked(elink, slave=(bank == 1))
                        print(f"{identity} elink {elink}: locked={locked} (after re-lock)")
                        if not locked:
                            show_status(f"{identity} elink {elink} FAILED to lock", "error")

        show_status("Self-trigger configured", "success")
        for identity, etroc in chips.items():
            fc_status = etroc.FC_status()
            invalid_count = etroc.get_invalidFCCount()
            print(f"{identity}: FC status={fc_status}, invalid FC count={invalid_count}")


    configure_self_trigger(rb, mapping, ids, chips, trigger_delay=TRIGGER_DELAY_DEFAULT, trigger_board_ids=trigger_ids, trigger_combination_logic=TRIGGER_COMBINATION_LOGIC,)

    # Notebook code cell 40
    announce('10. Charge injection', 'Run the original charge-injection and readout sequence.', 'Inspect returned data and errors; completion alone is not a physics-data validation.')
    import time
    from datetime import datetime, timezone
    from tamalero.FIFO import FIFO
    from tamalero.DataFrame import DataFrame   # adjust the import path if DataFrame lives elsewhere in this driver

    if not basic_checks_passed:
        raise RuntimeError("Complete section 7 (link enable) successfully before running charge injection")

    QINJ_COUNT_DEFAULT = args.qinj_count
    EMPTY_SLOT_BCID = 500   # must be < 3563
    CHARGE_FC_DEFAULT = args.charge_fc


    def enable_qinj_pixels(chips, blnw_results, charge_fc=CHARGE_FC_DEFAULT):
        """Enable charge injection on the pixels scanned in section 6.1.

        QSel selects the injected charge amplitude (QSel = charge_fc - 1, per
        the reference confChip()). QInjEn=1 turns injection on for that pixel;
        it defaults to 0 (off). Only pixels with a valid baseline (see
        configure_thresholds()) are enabled.

        QInjEn stays set on the chip until cleared - call disable_qinj_pixels()
        once you are done with charge injection on this selection, so these
        pixels don't keep responding to injection pulses during a later
        self-trigger/cosmic run.
        """
        scan_pixels = blnw_results["pixels"]
        qsel = charge_fc - 1
        enabled = {}
        for identity, etroc in chips.items():
            count = 0
            for row, col in scan_pixels:
                baseline = blnw_results["baseline"][identity][(row, col)]
                if baseline is None:
                    continue
                etroc.wr_reg("QSel", qsel, row=row, col=col, broadcast=False)
                etroc.wr_reg("QInjEn", 1, row=row, col=col, broadcast=False)
                count += 1
            enabled[identity] = count
            show_status(f"{identity}: QInj enabled on {count} pixels (QSel={qsel}, charge={charge_fc}fC)", "info")
        return enabled


    def disable_qinj_pixels(chips, blnw_results):
        """Turn QInjEn back off on the pixels scanned in section 6.1."""
        scan_pixels = blnw_results["pixels"]
        for identity, etroc in chips.items():
            for row, col in scan_pixels:
                etroc.wr_reg("QInjEn", 0, row=row, col=col, broadcast=False)
        show_status("QInj disabled on scanned pixels", "info")


    qinj_enabled_pixels = enable_qinj_pixels(chips, blnw_results, charge_fc=CHARGE_FC_DEFAULT)


    def run_charge_injection_test(chips, qinj_count=QINJ_COUNT_DEFAULT,
                                   empty_slot_bcid=EMPTY_SLOT_BCID,
                                   report_interval=10, max_iterations=1):
        """Self-triggered charge-injection (QInj) readout test.

        Assumes configure_thresholds() and configure_self_trigger() have already
        run (DAC/L1Adelay applied, TRIG_ENABLE_MASK/TRIG_DLY_SEL set, links
        locked). This function only sets emptySlotBCID, drives the FIFO and
        injection pulse, and reads back events.
        """
        df = DataFrame()
        fifo = FIFO(rb)
        fifo.reset()
        rb.reset_data_error_count()
        rb.enable_etroc_readout()
        rb.rerun_bitslip()
        fifo.use_etroc_data()
        time.sleep(0.1)

        rb.enable_etroc_trigger()
        time.sleep(0.1)

        for identity, etroc in chips.items():
            etroc.wr_reg("emptySlotBCID", empty_slot_bcid)

        trigger_cnt = 0
        hit_counter = 0
        trailer_cnt = 0
        ea_clean = 0
        ea_issue = 0

        start_time = datetime.now(timezone.utc)
        last_report_time = time.time()

        fifo.send_Qinj_only(count=qinj_count)  # per pixel

        rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.EVENT_CNT_RESET", 0x1)
        time.sleep(1)
        event_cnt = rb.kcu.read_node(f"READOUT_BOARD_{rb.rb}.EVENT_CNT").value()
        show_status(f"Event count in 1 second: {event_cnt}", "info")
        time.sleep(0.01)

        # Expected elinks computed from the mapping instead of hardcoded, so this
        # works regardless of which/how many boards are selected.
        expected_elinks = set()
        for identity in chips:
            for bank, elinks in build_elinks(mapping, identity).items():
                expected_elinks.update(elinks)

        for iteration in range(max_iterations):
            try:
                print(f"Occupancy: {fifo.get_occupancy()}")
                data = fifo.pretty_read(df)
                time.sleep(0.1)  # slow down DAQ to avoid uhal UDP errors

                if len(data) > 0:
                    flashing_headers = {}
                    for event in data:
                        if event and len(event) >= 2:
                            if event[0] == "header":
                                trigger_cnt += 1
                                if event[1]["bcid"] == "1":
                                    flashing_headers[event[1]["elink"]] = event
                            elif event[0] == "data":
                                print(event)
                                if event[1]["ea"] == 0:
                                    ea_clean += 1
                                else:
                                    show_status("ea is not 0", "error")
                                    ea_issue += 1
                                hit_counter += 1
                            elif event[0] == "trailer":
                                trailer_cnt += 1

                    if expected_elinks.issubset(flashing_headers.keys()):
                        show_status(f"Flashing bit detected on all {len(expected_elinks)} expected elinks", "success")
                        for elink in sorted(flashing_headers.keys()):
                            print(f"  elink {elink}: {flashing_headers[elink]}")

                time.sleep(0.1)
                current_time = time.time()
                if current_time - last_report_time >= report_interval:
                    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                    print(f"\n--- Status Report ---")
                    print(f"Running time: {elapsed:.1f} s")
                    print(f"Total hits: {hit_counter}")
                    print(f"Trigger count: {trigger_cnt}")
                    print(f"Trailer count: {trailer_cnt}")
                    last_report_time = current_time
                time.sleep(0.05)

            except Exception as exc:
                show_status(f"Data acquisition error: {exc}", "error")
                time.sleep(1)
                continue

        current_mask = rb.kcu.read_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK").value()
        show_status(f"Readback TRIG_ENABLE_MASK: 0x{current_mask:X}", "info")
        show_status(f"Total hits: {hit_counter}, trigger count: {trigger_cnt}, trailer count: {trailer_cnt}", "info")
        if hit_counter:
            show_status(f"ea clean: {ea_clean}, ea issue: {ea_issue}",
                        "success" if ea_issue == 0 else "warning")

        return {
            "trigger_cnt": trigger_cnt,
            "hit_counter": hit_counter,
            "trailer_cnt": trailer_cnt,
            "ea_clean": ea_clean,
            "ea_issue": ea_issue,
        }


    qinj_results = run_charge_injection_test(chips, qinj_count=QINJ_COUNT_DEFAULT)

    # Uncomment once you're done running charge injection on this selection,
    # so these pixels stop responding to injection pulses in later runs:
    # disable_qinj_pixels(chips, blnw_results)

    # Notebook code cell 42
    announce('11. End of test', 'Mask the declared readout banks using the notebook cleanup step.', 'This does not power down chips or disable every external trigger source.')
    configure_readout_links(rb, mapping, [])
    print(f"Basic checks passed: {basic_checks_passed}")
    show_status("All declared readout banks are masked", "info")

    print("QInj workflow finished; declared data links are masked. Review readout results above.", flush=True)
    if args.interactive:
        interactive_checks(kcu, rb, pathfinder_profile=Path(MAPPING_PATH).name == "pathfinder_mapping.yaml")


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print(f"\n[STOPPED] Interrupted during: {CURRENT_STEP}", flush=True)
        print("Hardware is not automatically restored. Inspect its state before restarting.", flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[FAILED] Step: {CURRENT_STEP}", flush=True)
        print(f"Reason: {type(exc).__name__}: {exc}", flush=True)
        print("No later stage will run. Keep the traceback below for diagnosis.", flush=True)
        raise

