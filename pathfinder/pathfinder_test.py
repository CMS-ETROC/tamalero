#!/usr/bin/env python3
"""CLI example derived from pathfinder_test.ipynb, sections 0-12.
Notebook-aligned staged diagnostics. WS sections are not included.
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
    parser.add_argument("--stage", choices=["preview", "motherboard", "links", "phase-scan", "blnw", "qinj", "fc-scan"], default="preview",
                        help="Run all prerequisite sections through this stage")
    parser.add_argument("--selection", type=selection, default={0: "all"},
                        help='JSON, e.g. {"1":"all","3":[1]}')
    parser.add_argument("--trigger-selection", type=selection, default=None,
                        help="Defaults to the readout selection")
    parser.add_argument("--trigger-logic", choices=["or", "and"], default="or")
    parser.add_argument("--ip", default="192.168.0.10")
    parser.add_argument("--rb-id", type=int, default=0)
    parser.add_argument("--board-config", default="default")
    parser.add_argument("--monitor-seconds", type=int, default=3)
    parser.add_argument("--blnw", "--pixels", dest="pixels", choices=["sample", "all"], default="sample", help="BL/NW coverage: sample=4 pixels/chip; all=256 pixels/chip (default: sample)")
    parser.add_argument("--no-save", action="store_true", help="Do not save BL/NW pickle results")
    parser.add_argument("--blnw-load", metavar="PKL|latest", default=None, help="Skip the BL/NW scan and load saved results (a .pkl path, or latest = newest valid file in the results folder)")
    plot_options = parser.add_mutually_exclusive_group()
    plot_options.add_argument("--plot", "--plots", dest="no_plots", action="store_false", help="Include BL/NW plotting in the workflow")
    plot_options.add_argument("--no-plots", "--no-plot", dest="no_plots", action="store_true", help="Skip workflow plots (default); interactive menu 22 remains available")
    parser.set_defaults(no_plots=True)
    parser.add_argument("--trigger-delay", type=int, default=469)
    parser.add_argument("--qinj", "--charge-fc", dest="charge_fc", type=int, default=15, help="Injected charge in fC (1..32; default: 15)")
    parser.add_argument("--qinj-cnt", "--qinj-count", dest="qinj_count", type=int, default=1, help="Injection pulse count (default: 1)")
    parser.add_argument("--offset", type=int, default=50, help="Threshold DAC offset above measured baseline (default: 50)")
    parser.add_argument("--interactive", action="store_true",
                        help="Open repeatable diagnostics after a hardware stage")
    parser.add_argument("--rate", type=int, choices=[320, 640], default=640)
    parser.add_argument("--trigger-bits", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--l1a-delay", type=int, default=501, help="ETROC pixel L1Adelay; distinct from FPGA trigger delay and FC input delay bits")
    parser.add_argument("--scan-selection", type=selection, help="Subset of initialized D/H to phase-scan; defaults to --selection")
    parser.add_argument("--phase-settings", type=json.loads, default={}, help='Explicit fixed phases, e.g. {"D3H1":4}; applied after initialization')
    parser.add_argument("--fc-target", help="One initialized ETROC to scan FC input delay bits, e.g. D3H1")
    parser.add_argument("--scan-output", type=Path, default=Path("diagnostic_results"))
    parser.add_argument("--print-records", action="store_true", help="Print full header/data/trailer records grouped by E-link")
    parser.add_argument("--fifo-reads", type=int, default=3, help="Read passes after one QInj burst; does not send extra injections")
    parser.add_argument("--step", action="store_true", help="Pause for debugging after each completed hardware step; retry failed steps in the same session")
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
    print(f"Rate: {args.rate} Mbps; merged trigger bits: {args.trigger_bits}; pixel L1Adelay: {args.l1a_delay}")
    print("Test stages run automatically; an interactive menu follows a hardware stage when --interactive is set.")
    print("On failure, the script stops. Partial hardware configuration may remain.", flush=True)



def interactive_checks(kcu, rb, *, pathfinder_profile=False, mapping=None, chips=None, ids=None, qinj_context=None, output_dir="diagnostic_results", recovery=False, failed=False, blnw_results=None): 
    """Repeat diagnostics without reconstructing KCU or ReadoutBoard."""
    from pathfinder_control import (
        read_optical_link_status, monitor_optical_links,
        read_clock_configuration, reset_optical_fec_counters,
        initialize_pathfinder_links,
    )

    actions = {
        "1": ("KCU status", "READ ONLY", "Read KCU status, clocks and available E-link lock flags."),
        "2": ("Uplink READY/FEC", "READ ONLY", "Read current counters without clearing them."),
        "3": ("Monitor uplinks for 10 seconds", "READ ONLY", "Sample READY/FEC once per second; counters are not reset."),
        "4": ("Clock register readback", "READ ONLY", "Read ECLK0-27 frequency and drive strength; this does not measure waveforms."),
        "5": ("Reset FEC counters", "WRITE HARDWARE", "Clear historical FEC counts on both uplinks."),
        "6": ("Configure and verify optical links", "WRITE HARDWARE", "Apply the tested Pathfinder lpGBT/VTRx+ profile, clear FEC counts and verify for 10 seconds."),
    }
    if mapping is not None and chips:
        actions["7"] = ("Compare ETROC links", "READ ONLY", "Compare initialized ETROCs side by side; no resets or configuration writes.")
        actions["8"] = ("Scan phase select", "WRITE HARDWARE", "Stop acquisition first. Scan selected D/H; restore phases, group modes and readout masks. Shared bitslip may realign other channels.")
        actions["9"] = ("Apply phase settings", "WRITE HARDWARE", "Apply explicit phases to selected D/H and retain fixed group mode; restore readout masks afterwards.")
        actions["10"] = ("Scan trigger delay", "WRITE HARDWARE", "Requires prepared QInj. Scan FPGA delay with fixed pixel L1Adelay; restore FPGA delay and readout masks.")
        actions["11"] = ("Set trigger delay", "WRITE HARDWARE", "Set and verify FPGA TRIG_DLY_SEL; pixel L1Adelay is unchanged.")
        actions["12"] = ("Charge injection acquisition", "WRITE HARDWARE", "Reuse prepared thresholds/self-trigger. Set charge and pulse count, acquire and save per-link records.")
        actions["13"] = ("Scan FC input delay", "WRITE HARDWARE", "Scan 00/01/10/11 on one initialized ETROC using QInj; restore original FC bits even on failure.")
    if mapping is not None and rb is not None:
        actions["21"] = ("Read ADC monitoring", "READ ONLY", "Configure mapped ADC channels, calibrate and read values. Missing daughter boards may show out-of-range readings.")
        actions["15"] = ("Retry direct I2C reads", "READ ONLY", "Read mapped ETROC register 0 repeatedly, including chips not initialized yet. No reset/default configuration.")
        actions["17"] = ("Read receiver masks", "READ ONLY", "Read DAQ/TRIG disable masks without changing them.")
    if chips:
        actions["14"] = ("Set FC input delay", "WRITE HARDWARE", "Set and verify fcClkDelayEn/fcDataDelayEn on explicit targets; keep the requested settings.")
        actions["16"] = ("Rerun bitslip and check links", "WRITE HARDWARE", "Rerun shared bitslip; report selected link status. Does not change masks.")
        actions["18"] = ("Check link rates", "READ ONLY", "Compare configured FPGA/lpGBT/ETROC rates.")
        actions["19"] = ("Enable selected readout links", "WRITE HARDWARE", "Apply masks for initialized selected ETROCs; other receivers remain disabled.")
        actions["20"] = ("Reset one global readout", "WRITE HARDWARE", "Stop triggers first. Pulse ETROC global-readout reset and read controllerState; does not hard reset or force a state value.")
    if blnw_results is not None:
        actions["22"] = ("Plot BL/NW results", "READ ONLY", "Plot current in-memory maps; full scans also show histograms. Close plot windows to return.")
    if kcu is None:
        actions.pop("1", None)
    if rb is None:
        for key in ("2", "3", "4", "5", "6"):
            actions.pop(key, None)
    if qinj_context is None:
        for key in ("10", "12", "13"):
            actions.pop(key, None)
    while True:
        print("\n" + "=" * 72)
        print("Interactive diagnostics - reuse the current hardware connection")
        for key, (title, kind, _) in sorted(actions.items(), key=lambda item: int(item[0])):
            suffix = " (unavailable for this mapping)" if key == "6" and not pathfinder_profile else ""
            print(f"{key}. {title} [{kind}]{suffix}")
        if recovery:
            print("Enter / c: Continue (unavailable until this step passes)" if failed else "Enter / c: Continue to the next step")
            print("R: Rerun the current step")
        print("0. Exit (does not power down hardware)")
        try:
            choice = input("Select an action: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT] Leaving diagnostics; hardware is not restored or powered down.")
            return
        if recovery and choice.lower() == "r":
            return "retry"
        if recovery and choice.lower() in ("", "c"):
            if failed:
                print("Step failed. Debug, then press R to retry; continuing is unavailable.")
                continue
            return "continue"
        if choice == "0":
            print("[EXIT] Leaving diagnostics; hardware is not powered down.")
            return
        if choice not in actions:
            print("[WARNING] Select one of the displayed action numbers.")
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
                initialize_pathfinder_links(rb, monitor_seconds=10)
                print("[PASS] Optical configuration and verification completed.")
            elif choice == "7":
                from pathfinder_control import compare_etroc_links
                available = [i for i in (ids if ids is not None else chips) if i in chips]
                print("Initialized ETROCs:", ", ".join(available))
                raw = input("ETROC IDs separated by commas (Enter = all): ").strip()
                targets = list(dict.fromkeys(i.strip() for i in raw.split(",") if i.strip())) if raw else available
                if not targets or any(i not in available for i in targets):
                    print("[SKIPPED] Select only initialized IDs listed above.")
                    continue
                differences = input("Only differences? [y/N]: ").strip().lower() in ("y", "yes")
                compare_etroc_links(rb, mapping, chips, targets, only_differences=differences)
            elif choice == "8":
                from pathfinder_control import expand_selection, scan_etroc_phases
                selected = selection(input('Scan selection JSON, e.g. {"3":[1]}: ').strip())
                targets = expand_selection(mapping, selected)
                if any(i not in chips for i in targets):
                    raise ValueError("All scan targets must already be initialized")
                with temporary_readout_selection(rb, mapping, list(chips)):
                    saved = set_fixed_phase_groups(rb, mapping, targets)
                    try:
                        scan_etroc_phases(rb, mapping, chips, selected, output_dir=output_dir)
                    finally:
                        restore_phase_groups(saved)
            elif choice == "9":
                from pathfinder_control import apply_etroc_phases
                settings = json.loads(input('Phase settings JSON, e.g. {"D3H1":4,"D2H3":5}: ').strip())
                if not isinstance(settings, dict) or not settings or any(
                    i not in chips or type(p) is not int or not 0 <= p <= 14 for i, p in settings.items()
                ):
                    raise ValueError("Use initialized IDs with integer phases 0..14")
                with temporary_readout_selection(rb, mapping, list(chips)):
                    saved = set_fixed_phase_groups(rb, mapping, list(settings))
                    try:
                        result = apply_etroc_phases(rb, mapping, chips, settings)
                        print(result)
                    except BaseException:
                        restore_phase_groups(saved)
                        raise
                print("Applied phases and fixed group modes retained; inspect lock results above.")
            elif choice == "10":
                if qinj_context is None:
                    print("[SKIPPED] Run --stage qinj --interactive first: thresholds, injection and self-trigger must be prepared.")
                    continue
                from pathfinder_control import scan_trigger_delay
                start = int(input("Start delay [440]: ").strip() or "440")
                stop = int(input("Stop delay, exclusive [500]: ").strip() or "500")
                latency = int(input("Fixed pixel L1Adelay [501]: ").strip() or "501")
                count = int(input("QInj pulses per delay [1]: ").strip() or "1")
                if not 0 <= start < stop <= 512 or not 0 <= latency <= 511 or count < 1:
                    raise ValueError("Invalid delay range, L1Adelay or pulse count")
                with temporary_readout_selection(rb, mapping, list(chips)):
                    report = scan_trigger_delay(rb, chips, mapping, scan_pixels=qinj_context["pixels"],
                                                delays=range(start, stop), l1a_delay=latency, qinj_count=count)
                save_diagnostic(output_dir, "interactive_trigger_delay", report)
                for delay, item in report["results"].items():
                    print(f"delay={delay}: {item['per_elink']}")
                print("Pixel L1Adelay remains at the requested value; FPGA trigger delay restored. No best delay applied.")
            elif choice == "11":
                value = int(input("FPGA trigger delay (0..511): ").strip())
                if not 0 <= value <= 511:
                    raise ValueError("Trigger delay must be in 0..511")
                node = f"READOUT_BOARD_{rb.rb}.TRIG_DLY_SEL"
                old = kcu.read_node(node)
                old = int(old.value() if hasattr(old, "value") else old)
                try:
                    kcu.write_node(node, value)
                    actual = kcu.read_node(node)
                    actual = int(actual.value() if hasattr(actual, "value") else actual)
                    if actual != value:
                        raise RuntimeError(f"Trigger delay readback mismatch: {actual}")
                except BaseException:
                    kcu.write_node(node, old)
                    raise
                print(f"TRIG_DLY_SEL: {old} -> {actual}; L1Adelay unchanged")
            elif choice in ("12", "13"):
                if qinj_context is None or "acquire" not in qinj_context:
                    print("[SKIPPED] Use --stage qinj --step --interactive to prepare thresholds, self-trigger and QInj first.")
                    continue
                target = None
                if choice == "13":
                    print("Initialized ETROCs:", ", ".join(chips))
                    target = input("FC scan target, e.g. D5H1: ").strip()
                    if target not in chips:
                        raise ValueError("FC target must already be initialized")
                charge = qinj_context["charge_fc"]        # set with --qinj on the command line
                count = qinj_context["qinj_count"]       # set with --qinj-cnt on the command line
                print(f"QInj charge={charge} fC, pulses={count} (from the command line)")
                reads = int(input("FIFO read passes [3]: ").strip() or "3")
                records = input("Print complete records by E-link? [y/N]: ").strip().lower() in ("y", "yes")
                if not 1 <= charge <= 32 or count < 1 or reads < 1:
                    raise ValueError("Charge must be 1..32 fC; pulse count and read passes must be positive")
                def acquire():
                    return qinj_context["acquire"](charge, count, reads, records)
                with temporary_readout_selection(rb, mapping, list(chips)):
                    if choice == "12":
                        report = acquire()
                        save_diagnostic(output_dir, "interactive_qinj", report)
                    else:
                        run_fc_scan(chips, target, acquire, output_dir)
                qinj_context["charge_fc"] = charge
            elif choice == "22":
                show_blnw_plots(blnw_results)
            elif choice == "21":
                if "adc_monitoring" not in mapping:
                    print("No adc_monitoring section in this mapping.")
                else:
                    from pathfinder_control import configure_adc_mapping
                    for bank, entries in configure_adc_mapping(rb, mapping).items():
                        if not entries:
                            continue
                        lp = rb.DAQ_LPGBT if bank == "daq" else rb.TRIG_LPGBT
                        print(f"{bank.upper()} ADC monitoring:")
                        lp.calibrate_adc()
                        lp.read_adcs(check=True, strict_limits=False)
                    print("ADC read complete; review individual values and limits.")
            elif choice == "14":
                names = [i.strip().strip("\"'") for i in input('ETROC IDs, e.g. "D0H0", "D0H1": ').split(",") if i.strip().strip("\"'")]
                pair = input("FC clk/data: 00, 01 or 10: ").strip()
                if not names or any(i not in chips for i in names) or pair not in ("00", "01", "10"):
                    raise ValueError("Use initialized IDs and one of 00/01/10")
                set_fc_settings(chips, names, tuple(map(int, pair)))
            elif choice == "15":
                from pathfinder_control import expand_selection
                selected = selection(input('I2C selection JSON, e.g. {"5":"all"}: ').strip())
                targets = expand_selection(mapping, selected)
                repeats = int(input("Read repetitions [3]: ").strip() or "3")
                if not 1 <= repeats <= 100:
                    raise ValueError("Repetitions must be 1..100")
                probe_i2c(rb, mapping, targets, repeats)
            elif choice == "16":
                from pathfinder_control import check_links
                rb.rerun_bitslip()
                print(check_links(rb, mapping, list(chips)))
            elif choice == "17":
                for bank, suffix in (("daq", ""), ("trig", "_SLAVE")):
                    if bank in mapping["lpgbts"]:
                        node = f"READOUT_BOARD_{rb.rb}.ETROC_DISABLE{suffix}"
                        print(node, rb.kcu.read_node(node))
            elif choice == "18":
                from pathfinder_control import check_link_rate
                check_link_rate(rb, chips, mapping=mapping)
            elif choice == "19":
                from pathfinder_control import configure_readout_links
                print(configure_readout_links(rb, mapping, list(chips)))
            elif choice == "20":
                import time
                identity = input("ETROC ID to reset: ").strip()
                if identity not in chips:
                    raise ValueError("ETROC must be initialized")
                chip = chips[identity]
                try:
                    chip.wr_reg("asyResetGlobalReadout", 0)
                    time.sleep(0.1)
                finally:
                    chip.wr_reg("asyResetGlobalReadout", 1)
                time.sleep(0.2)
                print(identity, "controllerState:", chip.rd_reg("controllerState"))
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


from contextlib import contextmanager


@contextmanager
def temporary_readout_selection(rb, mapping, targets):
    """Enable initialized targets for diagnostics, restoring exact original masks."""
    from pathfinder_control import configure_readout_links
    nodes = [f"READOUT_BOARD_{rb.rb}.ETROC_DISABLE"]
    if "trig" in mapping["lpgbts"]:
        nodes.append(f"READOUT_BOARD_{rb.rb}.ETROC_DISABLE_SLAVE")
    def integer(v):
        return int(v.value() if hasattr(v, "value") else v)
    saved = {node: integer(rb.kcu.read_node(node)) for node in nodes}
    try:
        configure_readout_links(rb, mapping, targets)
        rb.rerun_bitslip()
        yield
    finally:
        errors = []
        for node, value in saved.items():
            try:
                rb.kcu.write_node(node, value)
                if integer(rb.kcu.read_node(node)) != value:
                    raise RuntimeError("mask restoration readback mismatch")
            except Exception as exc:
                errors.append(f"{node}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))


def restore_phase_groups(saved):
    errors = []
    for lp, node, value in saved:
        try:
            lp.wr_reg(node, value)
            if int(lp.rd_reg(node)) != value:
                raise RuntimeError("TRACKMODE restore mismatch")
        except Exception as exc:
            errors.append(f"{node}: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))


def set_fixed_phase_groups(rb, mapping, ids):
    from pathfinder_control import build_elinks
    targets = {}
    for identity in ids:
        for bank, links in build_elinks(mapping, identity).items():
            for elink in links:
                targets[bank, elink // 4] = getattr(rb, "DAQ_LPGBT" if bank == 0 else "TRIG_LPGBT")
    saved = [(lp, f"LPGBT.RWF.EPORTRX.EPRX{group}TRACKMODE",
              int(lp.rd_reg(f"LPGBT.RWF.EPORTRX.EPRX{group}TRACKMODE")))
             for (bank, group), lp in targets.items()]
    try:
        for lp, node, old in saved:
            print(f"{node}: {old} -> 0 (shared by four EPRX inputs)")
            lp.wr_reg(node, 0)
            if int(lp.rd_reg(node)) != 0:
                raise RuntimeError("TRACKMODE fixed-mode readback mismatch")
    except BaseException:
        restore_phase_groups(saved)
        raise
    return saved


def save_diagnostic(folder, name, report):
    from datetime import datetime, timezone
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ".json")
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Saved: {path.resolve()}")
    return path


def run_fc_scan(chips, identity, acquire, output_dir):
    """Scan FC clk/data delay bits 00/01/10/11. Restore exact original bits.

    Does not change pixel L1Adelay or FPGA trigger delay.
    """
    chip = chips[identity]
    regs = ("fcClkDelayEn", "fcDataDelayEn")
    def integer(v):
        return int(v.value() if hasattr(v, "value") else v)
    original = {reg: integer(chip.rd_reg(reg)) for reg in regs}
    report = {"identity": identity, "original": original, "results": {}, "status": "running"}
    try:
        for clk, data in ((0, 0), (0, 1), (1, 0), (1, 1)):
            for reg, value in zip(regs, (clk, data)):
                chip.wr_reg(reg, value)
                if integer(chip.rd_reg(reg)) != value:
                    raise RuntimeError(f"{identity}: {reg} readback mismatch")
            result = acquire()
            report["results"][f"clk={clk},data={data}"] = result
            print(f"{identity} clk={clk} data={data}: {result['per_elink']}")
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        errors = []
        for reg, value in original.items():
            try:
                chip.wr_reg(reg, value)
                if integer(chip.rd_reg(reg)) != value:
                    raise RuntimeError("restore readback mismatch")
            except Exception as exc:
                errors.append(f"{reg}: {exc}")
        report["restore_errors"] = errors
        save_diagnostic(output_dir, "fc_delay", report)
        if errors:
            raise RuntimeError("FC settings restoration failed: " + "; ".join(errors))
    return report


def probe_i2c(rb, mapping, targets, repeats):
    import time
    for attempt in range(repeats):
        for identity in targets:
            bus = mapping["etrocs"][identity]["i2c"]
            try:
                lp = getattr(rb, "DAQ_LPGBT" if bus["lpgbt"] == "daq" else "TRIG_LPGBT")
                value = lp.I2C_read(reg=0, master=bus["channel"], slave_addr=bus["address"], nbytes=1, adr_nbytes=2)
                print(f"{attempt+1}/{repeats} {identity} {bus}: READ OK {value!r}")
            except Exception as exc:
                print(f"{attempt+1}/{repeats} {identity} {bus}: {type(exc).__name__}: {exc}")
        time.sleep(0.2)


def set_fc_settings(chips, names, pair):
    regs = ("fcClkDelayEn", "fcDataDelayEn")
    def integer(v):
        return int(v.value() if hasattr(v, "value") else v)
    saved = {(i,r): integer(chips[i].rd_reg(r)) for i in names for r in regs}
    try:
        for identity in names:
            for reg, value in zip(regs, pair):
                chips[identity].wr_reg(reg, value)
                if integer(chips[identity].rd_reg(reg)) != value:
                    raise RuntimeError(f"{identity}: {reg} mismatch")
            print(identity, dict(zip(regs,pair)))
    except BaseException:
        errors = []
        for (i,r), v in saved.items():
            try:
                chips[i].wr_reg(r,v)
                if integer(chips[i].rd_reg(r)) != v:
                    raise RuntimeError("restore mismatch")
            except Exception as exc:
                errors.append(f"{i}/{r}: {exc}")
        if errors:
            print("FC restore errors:", errors)
        raise


def workflow_menu(state, args, failed=False):
    # A snapshot of references to live objects; no reconstruction of KCU/chips.
    context = None
    if "interactive_acquire" in state:
        context = {"pixels": state["blnw_results"]["pixels"], "charge_fc": args.charge_fc, "qinj_count": args.qinj_count,
                   "acquire": state["interactive_acquire"]}
    print("Use R to retry the failed step; continuing requires a successful check." if failed else "Step complete. Debug now or continue.")
    return interactive_checks(state.get("kcu"), state.get("rb"),
                              pathfinder_profile=Path(args.mapping).name == "pathfinder_mapping.yaml",
                              mapping=state.get("mapping"), chips=state.get("chips"), ids=state.get("ids"),
                              qinj_context=context, output_dir=args.scan_output, recovery=True, failed=failed, blnw_results=state.get("blnw_results"))


def show_blnw_plots(blnw_results):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    from datetime import datetime
    def format_scan_time(results):
        """Use the saved scan start time; never substitute the plotting time."""
        timestamp = results.get('started_at')
        if not timestamp:
            return 'Scan time unavailable'
        try:
            return 'Scan: ' + datetime.fromisoformat(str(timestamp)).isoformat(sep=' ', timespec='seconds')
        except (TypeError, ValueError):
            return 'Scan: ' + str(timestamp)
    def build_blnw_matrix(values):
        matrix = np.full((16, 16), np.nan)
        for (row, col), value in values.items():
            if value is not None:
                matrix[row, col] = float(value)
        return matrix
    def plot_blnw_maps(results):
        figures = []
        for identity in results['chip_name']:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
            for ax, key, title in zip(axes, ('baseline', 'NW'), ('Baseline', 'Noise width')):
                matrix = build_blnw_matrix(results[key][identity])
                cmap = plt.get_cmap('viridis').copy()
                cmap.set_bad('white')
                view = ax.imshow(np.ma.masked_invalid(matrix), origin='lower', cmap=cmap, interpolation='nearest')
                ax.text(0.01, 1.0, 'CMS', transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom')
                ax.text(0.11, 1.0, 'ETL ETROC', transform=ax.transAxes, fontsize=10, style='italic', va='bottom')
                ax.text(1.0, 1.025, f'{identity}-{title} 2D', transform=ax.transAxes, fontsize=9, ha='right', va='bottom')
                ax.text(1.0, 1.0, format_scan_time(results), transform=ax.transAxes, fontsize=9, ha='right', va='bottom')
                ax.set_xlabel('Column')
                ax.set_ylabel('Row')
                ax.set_xticks(range(16))
                ax.set_yticks(range(16))
                ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
                ax.grid(which='minor', color='lightgray', linewidth=0.4)
                ax.tick_params(which='minor', bottom=False, left=False)
                for row, col in np.argwhere(np.isfinite(matrix)):
                    value = matrix[row, col]
                    rgba = view.cmap(view.norm(value))
                    brightness = sum((weight * component for weight, component in zip((0.299, 0.587, 0.114), rgba[:3])))
                    label = f'{value:.0f}' if key == 'NW' else f'{value:g}'
                    ax.text(col, row, label, ha='center', va='center', fontsize=6, color='black' if brightness > 0.5 else 'white')
                if np.isfinite(matrix).any():
                    colorbar = fig.colorbar(view, ax=ax, label='DAC code')
                    if key == 'NW':
                        colorbar.locator = MaxNLocator(integer=True)
                        colorbar.formatter = FormatStrFormatter('%.0f')
                        colorbar.update_ticks()
                else:
                    ax.set_title(f'{identity} - {title} (no valid results)')
            figures.append(fig)
            plt.show()
        return figures
    def plot_blnw_histograms(results):
        figures = []
        if results.get('scan_mode') != 'all':
            return figures
        for identity in results['chip_name']:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
            for ax, key, title in zip(axes, ('baseline', 'NW'), ('Baseline', 'Noise width')):
                matrix = build_blnw_matrix(results[key][identity])
                values = matrix[np.isfinite(matrix) & (matrix != 0)]
                ax.text(0.01, 1.0, 'CMS', transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom')
                ax.text(0.11, 1.0, 'ETL ETROC', transform=ax.transAxes, fontsize=10, style='italic', va='bottom')
                ax.text(1.0, 1.025, f'{identity}-{title} Histo', transform=ax.transAxes, fontsize=9, ha='right', va='bottom')
                ax.text(1.0, 1.0, format_scan_time(results), transform=ax.transAxes, fontsize=9, ha='right', va='bottom')
                ax.set_xlabel(f'{title} (DAC code)')
                ax.set_ylabel('Pixel count')
                if key == 'baseline':
                    bins = np.linspace(0, 1023, 65)
                    ax.set_xlim(0, 1023)
                    ax.set_xticks([0, 128, 256, 384, 512, 640, 768, 896, 1023])
                else:
                    bins = np.arange(-0.5, 18, 1)
                    ax.set_xlim(0, 17)
                    ax.set_xticks(range(18))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                if values.size:
                    ax.hist(values, bins=bins, histtype='step', color='#7ec8e3', linewidth=1.3)
                    ax.text(0.97, 0.95, f'Mean: {values.mean():.2f}, Std: {values.std():.2f}', transform=ax.transAxes, ha='right', va='top')
                else:
                    ax.text(0.5, 0.5, 'No valid results', transform=ax.transAxes, ha='center', va='center')
            figures.append(fig)
            plt.show()
        return figures
    blnw_figures = plot_blnw_maps(blnw_results)
    blnw_histogram_figures = plot_blnw_histograms(blnw_results)

def main(args):
    show_run_plan(args)
    if args.step:
        args.interactive = True
    if args.monitor_seconds < 1 or args.qinj_count < 1 or args.fifo_reads < 1:
        raise ValueError("monitor-seconds and qinj-count must be positive")

    if not 0 <= args.trigger_delay <= 511 or not 0 <= args.l1a_delay <= 511:
        raise ValueError("trigger-delay and l1a-delay must be in 0..511")
    if not 1 <= args.charge_fc <= 32:
        raise ValueError("charge-fc must be in 1..32")
    if args.stage == "fc-scan" and not args.fc_target:
        raise ValueError("fc-scan requires --fc-target, e.g. D3H1")

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
        check_connections, check_links, configure_readout_links, build_trigger_mask, initialize_pathfinder_links,
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
    ELINK_WIDTH = {320: 2, 640: 3}[args.rate]
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

    if args.fc_target and args.fc_target not in ids:
        raise ValueError("fc-target must be in --selection")
    if args.scan_selection and not set(expand_selection(mapping, args.scan_selection)).issubset(ids):
        raise ValueError("scan-selection must be a subset of --selection")
    if not isinstance(args.phase_settings, dict) or any(
        i not in ids or type(p) is not int or not 0 <= p <= 14 for i, p in args.phase_settings.items()
    ):
        raise ValueError("phase-settings must map selected IDs to integer phases 0..14")
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
    while True:
        try:
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
        except Exception as exc:
            if not args.interactive:
                raise
            print(f"[FAILED] {CURRENT_STEP}: {type(exc).__name__}: {exc}")
            if workflow_menu(locals(), args, failed=True) == "retry":
                continue
            return
        if args.step:
            action = workflow_menu(locals(), args)
            if action == "retry":
                continue
            if action != "continue":
                return
        break
    while True:
        try:
            announce('4. Motherboard initialization', 'Create rb, configure declared lpGBTs, configure TRIG clocks and enable the selected data links.', 'Both declared lpGBT objects must be available; ELINK_WIDTH must read back correctly.')
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
            if USE_TRIG and (not rb.trigger or getattr(rb, "TRIG_LPGBT", None) is None):
                raise RuntimeError("TRIG lpGBT was not initialized")

            # Retain the TRIG clock configuration verified during hardware testing.
            if USE_TRIG:
                rb.TRIG_LPGBT.configure_clocks(0x0fffffff)
                show_status("TRIG ECLK0-27 configured; verify readback below", "info")

            width_node = f"READOUT_BOARD_{rb.rb}.ELINK_WIDTH"
            rb.kcu.write_node(width_node, ELINK_WIDTH)
            width = rb.kcu.read_node(width_node).value()
            if width != ELINK_WIDTH:
                raise RuntimeError(f"ELINK_WIDTH mismatch: expected {ELINK_WIDTH}, received {width}")
            # Enable the selected data links here; every other link stays masked.
            applied_masks = configure_readout_links(rb, mapping, ids)
            print(f"Initialized controllers: {list(mapping['lpgbts'])}; selected data links enabled: {ids}")

            # Notebook code cell 14
        except Exception as exc:
            if not args.interactive:
                raise
            print(f"[FAILED] {CURRENT_STEP}: {type(exc).__name__}: {exc}")
            if workflow_menu(locals(), args, failed=True) == "retry":
                continue
            return
        if args.step:
            action = workflow_menu(locals(), args)
            if action == "retry":
                continue
            if action != "continue":
                return
        break
    while True:
        try:
            announce('4.1. Optical-link configuration and checks', 'Configure optical links, verify readback and monitor uplinks, then check lpGBT status.', 'For Pathfinder, both READY flags must stay 1 and FEC counters stay 0 at the sampled times.')
            optical_checks_passed = False
            optical_init_result = None

            if Path(MAPPING_PATH).name == "pathfinder_mapping.yaml":
                optical_init_result = initialize_pathfinder_links(
                    rb,
                    monitor_seconds=args.monitor_seconds,
                )
                optical_checks_passed = True
            else:
                show_status(
                    "Skipping Pathfinder optical initialization for this mapping",
                    "info",
                )

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


        except Exception as exc:
            if not args.interactive:
                raise
            print(f"[FAILED] {CURRENT_STEP}: {type(exc).__name__}: {exc}")
            if workflow_menu(locals(), args, failed=True) == "retry":
                continue
            return
        if args.step:
            action = workflow_menu(locals(), args)
            if action == "retry":
                continue
            if action != "continue":
                return
        break
    if args.stage == 'motherboard':
        print('Motherboard stage complete. With no daughter boards, stop here. Review whether the Pathfinder optical profile ran or was skipped.', flush=True)
        if args.interactive:
            interactive_checks(kcu, rb, pathfinder_profile=Path(MAPPING_PATH).name == 'pathfinder_mapping.yaml', mapping=mapping, chips=chips, ids=ids, blnw_results=locals().get("blnw_results"))
        return
    while True:
        try:
            announce('5. ETROC initialization', 'Create the selected ETROC objects.', 'Selected chips must be connected; motherboard-only runs stop before this step.')
            basic_checks_passed = False
            chips = dict(chips)
            initialized_ids = ()
            etroc_register_checks_done = False
            etroc_power_checks_done = False
            blnw_scan_passed = False
            existing = dict(chips)
            routes = {(mapping["etrocs"][i]["i2c"]["lpgbt"], mapping["etrocs"][i]["i2c"]["channel"],
                       mapping["etrocs"][i]["i2c"]["address"]): chip for i, chip in existing.items()}
            def reuse_etroc(*pos, **kw):
                bank = "daq" if kw.get("i2c_master") is rb.DAQ_LPGBT else "trig"
                key = (bank, kw.get("i2c_channel"), kw.get("i2c_adr"))
                return routes[key] if key in routes else ETROC(*pos, **kw)
            try:
                chips = initialize_etrocs(rb, mapping, ids, etroc_factory=reuse_etroc, strict=False, verbose=False)
            except InitializationError as exc:
                chips.update(exc.initialized_chips)
                print(f"Failed ETROC: {exc.failed_id}; completed: {list(chips)}")
                raise
            initialized_ids = tuple(ids)
            show_status(f"Initialized {len(chips)} ETROCs: {list(chips)}", "success")

            # Notebook code cell 22
        except Exception as exc:
            if not args.interactive:
                raise
            print(f"[FAILED] {CURRENT_STEP}: {type(exc).__name__}: {exc}")
            if workflow_menu(locals(), args, failed=True) == "retry":
                continue
            return
        if args.step:
            action = workflow_menu(locals(), args)
            if action == "retry":
                continue
            if action != "continue":
                return
        break
    while True:
        try:
            announce('5.1. ETROC communication and register checks', 'Check I2C communication with the initialized chips, then read the key registers.', 'A failed I2C check names the ETROC and the error; register readings are printed per chip.')
            basic_checks_passed = False
            etroc_register_checks_done = False
            etroc_power_checks_done = False
            blnw_scan_passed = False
            if not ids or tuple(ids) != initialized_ids or list(chips) != ids:
                raise RuntimeError("Selection is empty or initialization is incomplete; rerun initialization")
            connection_results = check_connections(chips)
            for identity, result in connection_results.items():
                show_status(f"{identity}: connected={result['connected']}, error={result['error']}", "success" if result["connected"] else "error")
            failed_i2c = {identity: result["error"] for identity, result in connection_results.items() if not result["connected"]}
            if failed_i2c:
                raise RuntimeError("I2C check failed for " + "; ".join(f"{identity}: {error}" for identity, error in failed_i2c.items()))

            etroc_register_results = {}
            for identity, etroc in chips.items():
                values = {}
                etroc_register_results[identity] = values
                try:
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
                    raise RuntimeError(f"ETROC register check failed for {identity}: {exc}") from exc
            etroc_register_checks_done = True

            # Notebook code cells 24 and 26
        except Exception as exc:
            if not args.interactive:
                raise
            print(f"[FAILED] {CURRENT_STEP}: {type(exc).__name__}: {exc}")
            if workflow_menu(locals(), args, failed=True) == "retry":
                continue
            return
        if args.step:
            action = workflow_menu(locals(), args)
            if action == "retry":
                continue
            if action != "continue":
                return
        break
    while True:
        try:
            announce('5.2. Power mode configuration', 'Apply and check the existing ETROC power settings.', 'Power-mode checks must pass before scanning.')
            basic_checks_passed = False
            etroc_power_checks_done = False
            blnw_scan_passed = False
            if tuple(ids) != initialized_ids or list(chips) != ids:
                raise RuntimeError("Selection changed or initialization is incomplete")
            if not etroc_register_checks_done:
                raise RuntimeError("Complete the ETROC register configuration cell first")
            etroc_power_results = {}
            for identity, etroc in chips.items():
                try:
                    etroc.set_power_mode(mode="high", row=0, col=0, broadcast=True)
                except Exception as exc:
                    raise RuntimeError(f"High power configuration failed for {identity}") from exc

            for identity, etroc in chips.items():
                try:
                    mode = etroc.get_power_mode(row=8, col=8)
                    etroc_power_results[identity] = mode
                    print(f"{identity}: power mode at pixel (8, 8) = {mode}")
                    if mode != "high":
                        raise RuntimeError(f"Expected high power mode, received {mode!r}")
                except Exception as exc:
                    raise RuntimeError(f"Power mode verification failed for {identity}") from exc
            etroc_power_checks_done = True
            show_status("High power mode configured; sampled pixel readbacks passed", "success")

            import time
        except Exception as exc:
            if not args.interactive:
                raise
            print(f"[FAILED] {CURRENT_STEP}: {type(exc).__name__}: {exc}")
            if workflow_menu(locals(), args, failed=True) == "retry":
                continue
            return
        if args.step:
            action = workflow_menu(locals(), args)
            if action == "retry":
                continue
            if action != "continue":
                return
        break
    while True:
        try:
            announce("5.3. Rate and merge configuration", "Synchronize rate and trigger width.", "Subsequent link retries preserve these settings.")
            # Configure merged data and keep ETROC/FPGA trigger widths consistent.
            TRIGGER_BITS = args.trigger_bits  # Current test: 0, 1, or 2 bits

            def as_int(value):
                return int(value.value() if hasattr(value, "value") else value)

            if TRIGGER_BITS not in (0, 1, 2):
                raise ValueError("This test supports TRIGGER_BITS = 0, 1, or 2")

            if not ids or any(identity not in chips for identity in ids):
                raise RuntimeError("Initialize all selected ETROCs first")

            for identity in ids:
                chip = chips[identity]
                chip.set_mergeTriggerData("merge")
                chip.set_triggerGranularity(TRIGGER_BITS)

                merge = as_int(chip.rd_reg("mergeTriggerData"))
                bits = as_int(chip.get_triggerGranularity())
                if merge != 1 or bits != TRIGGER_BITS:
                    raise RuntimeError(
                        f"{identity}: merge={merge}, trigger_bits={bits}"
                    )
                print(f"{identity}: merge={merge}, trigger_bits={bits}")

            node = f"READOUT_BOARD_{rb.rb}.TRIG_DATA_SIZE"
            rb.kcu.write_node(node, TRIGGER_BITS)
            actual = as_int(rb.kcu.read_node(node))
            if actual != TRIGGER_BITS:
                raise RuntimeError(f"FPGA TRIG_DATA_SIZE readback={actual}")

            print(f"FPGA: TRIG_DATA_SIZE={actual}")
            from pathfinder_control import set_link_rate, apply_etroc_phases
            set_link_rate(rb, chips, mbps=args.rate)
            configure_readout_links(rb, mapping, ids)
            if args.phase_settings:
                set_fixed_phase_groups(rb, mapping, list(args.phase_settings))
                apply_etroc_phases(rb, mapping, chips, args.phase_settings)
            rb.rerun_bitslip()
            time.sleep(1.0)
        except Exception as exc:
            if not args.interactive:
                raise
            print(f"[FAILED] {CURRENT_STEP}: {type(exc).__name__}: {exc}")
            if workflow_menu(locals(), args, failed=True) == "retry":
                continue
            return
        if args.step:
            action = workflow_menu(locals(), args)
            if action == "retry":
                continue
            if action != "continue":
                return
        break
    while True:
        try:
            announce("6. E-link lock check", "Check selected E-link locks before pixel calibration.", "On failure use compare, masks, phase scan/apply, then retry this check.")
            if args.stage != "phase-scan":
                configure_readout_links(rb, mapping, ids)
                rb.rerun_bitslip()
                current_links = check_links(rb, mapping, ids)
                for identity, result in current_links.items():
                    for link in result["links"]:
                        print(f"{identity}: {link['lpgbt'].upper()} E-link {link['elink']}, "
                              f"locked={link['locked']}, error={link['error']}")
                locked_count = sum(bool(link["locked"]) for result in current_links.values() for link in result["links"])
                link_count = sum(len(result["links"]) for result in current_links.values())
                print(f"E-link lock: {locked_count}/{link_count}")
                if not all(item["all_locked"] for item in current_links.values()):
                    raise RuntimeError("E-link check failed before calibration")
                basic_checks_passed = True
            if args.stage in ("links", "phase-scan"):
                announce("6.1. Link diagnostics", "Read settings, masks, controllerState and lock status.",
                         "Check optical link, rate/merge, ENABLE/INVERT/TERM and masks before scanning phase.")
                if args.stage == "phase-scan":
                    from pathfinder_control import scan_etroc_phases
                    scan_ids = expand_selection(mapping, args.scan_selection or args.selection)
                    saved_groups = set_fixed_phase_groups(rb, mapping, scan_ids)
                    try:
                        scan_etroc_phases(rb, mapping, chips, args.scan_selection or args.selection,
                                         output_dir=args.scan_output)
                    finally:
                        restore_phase_groups(saved_groups)
                    print("Scan restored original phases. Select a candidate explicitly with --phase-settings.")
                print("Diagnostic stage ended; selected receivers remain enabled for inspection.")
                if args.interactive:
                    interactive_checks(kcu, rb, pathfinder_profile=Path(MAPPING_PATH).name == "pathfinder_mapping.yaml",
                                       mapping=mapping, chips=chips, ids=ids, blnw_results=locals().get("blnw_results"))
                return


            # Notebook code cell 28
        except Exception as exc:
            if not args.interactive:
                raise
            print(f"[FAILED] {CURRENT_STEP}: {type(exc).__name__}: {exc}")
            if workflow_menu(locals(), args, failed=True) == "retry":
                continue
            return
        if args.step:
            action = workflow_menu(locals(), args)
            if action == "retry":
                continue
            if action != "continue":
                return
        break
    while True:
        try:
            announce('7. BL/NW scan settings', 'Apply pixel coverage and BL/NW save options.', 'sample scans 4 pixels per chip; all scans 256 pixels per chip.')
            # Choose "sample" for 4 pixels or "all" for the full 16 x 16 matrix.
            # Change it on the command line with --blnw sample|all (default: sample).
            # The scanned pixels are also the ones given thresholds and QInj later.
            BLNW_SCAN_MODE = args.pixels
            BLNW_SAVE_RESULTS = not args.no_save
            BLNW_OUTPUT_DIR = TAMALERO_ROOT / "pathfinder" / "results" / "blnw" 

            def show_blnw_mode():
                print(f"Current BL/NW mode: {BLNW_SCAN_MODE}")
                if args.blnw_load:
                    print(f"Load saved results: {args.blnw_load} (scan skipped; saved mode must cover --blnw {BLNW_SCAN_MODE})")
                else:
                    print(f"Save results: {BLNW_SAVE_RESULTS}; output folder: {BLNW_OUTPUT_DIR}")

            show_blnw_mode()

            # Notebook code cell 30
        except Exception as exc:
            if not args.interactive:
                raise
            print(f"[FAILED] {CURRENT_STEP}: {type(exc).__name__}: {exc}")
            if workflow_menu(locals(), args, failed=True) == "retry":
                continue
            return
        if args.step:
            action = workflow_menu(locals(), args)
            if action == "retry":
                continue
            if action != "continue":
                return
        break
    while True:
        try:
            announce('7.1. BL/NW scan', 'Scan selected pixels using the original notebook sequence.', 'Follow the progress bar and failure summary; saved results use BLNW_OUTPUT_DIR.')
            import pickle
            import io
            from contextlib import redirect_stdout
            from datetime import datetime

            def build_blnw_pixels(mode):
                if mode == "sample":
                    return [(row, col) for row in (0, 15) for col in (0, 15)]
                if mode == "all":
                    return [(row, col) for row in range(16) for col in range(16)]
                raise ValueError("BLNW_SCAN_MODE must be 'sample' or 'all'")


            blnw_scan_passed = False
            scan_pixels = build_blnw_pixels(BLNW_SCAN_MODE)
            if not ids or tuple(ids) != initialized_ids or list(chips) != ids:
                raise RuntimeError("Selection is empty or initialization is incomplete")
            if not etroc_register_checks_done or not etroc_power_checks_done:
                raise RuntimeError("Complete ETROC register and power configuration first")
            if args.blnw_load:
                def blnw_load_error(data):
                    if not isinstance(data, dict) or "baseline" not in data or "pixels" not in data:
                        return "not a BL/NW result file"
                    if not data.get("completed"):
                        return "scan was not completed"
                    if any(p not in data["pixels"] for p in scan_pixels):
                        return f"saved mode '{data.get('scan_mode')}' does not cover requested '{BLNW_SCAN_MODE}'"
                    missing = [i for i in ids if i not in data.get("chip_name", [])]
                    if missing:
                        return f"no data for {missing}"
                    for i in ids:
                        if any((p not in data["baseline"].get(i, {})) for p in scan_pixels):
                            return f"{i}: baseline missing for some pixels"
                    return None

                if args.blnw_load == "latest":
                    candidates = sorted(BLNW_OUTPUT_DIR.glob("blnw_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
                else:
                    candidates = [Path(args.blnw_load)]
                blnw_results = None
                for candidate in candidates:
                    try:
                        with open(candidate, "rb") as handle:
                            data = pickle.load(handle)
                        error = blnw_load_error(data)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    if error is None:
                        blnw_results, loaded_path = data, candidate
                        break
                    print(f"Skip {candidate.name}: {error}")
                if blnw_results is None:
                    raise RuntimeError(f"No usable saved BL/NW result for {list(ids)} ({args.blnw_load})")
                # A larger saved scan (all) can serve a smaller request (sample); use only the requested pixels.
                blnw_results["pixels"] = list(scan_pixels)
                blnw_results["scan_mode"] = BLNW_SCAN_MODE
                failure_count = sum(1 for i in ids for p in scan_pixels if blnw_results["baseline"][i].get(p) is None)
                show_status(f"Loaded BL/NW results: {loaded_path.resolve()}", "success")
                show_status(f"mode={BLNW_SCAN_MODE}, {len(scan_pixels)} pixels/chip, "
                            f"{failure_count} failed (skipped at thresholds)", "warning" if failure_count else "success")
                blnw_scan_passed = True
            else:
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
                    raise RuntimeError(f"{failure_count} pixels failed; inspect failed_pixels and blnw_results")
                blnw_scan_passed = True

            # Notebook code cell 32
        except Exception as exc:
            if not args.interactive:
                raise
            print(f"[FAILED] {CURRENT_STEP}: {type(exc).__name__}: {exc}")
            if workflow_menu(locals(), args, failed=True) == "retry":
                continue
            return
        if args.step:
            action = workflow_menu(locals(), args)
            if action == "retry":
                continue
            if action != "continue":
                return
        break
    if not args.no_plots:
        while True:
            try:
                announce('7.2. BL/NW figures', 'Display maps and applicable histograms unless --no-plots is set.', 'A plot window may block execution; close it to continue.')
                if not args.no_plots:
                    show_blnw_plots(blnw_results)
            except Exception as exc:
                if not args.interactive:
                    raise
                print(f'[FAILED] {CURRENT_STEP}: {type(exc).__name__}: {exc}')
                if workflow_menu(locals(), args, failed=True) == 'retry':
                    continue
                return
            if args.step:
                action = workflow_menu(locals(), args)
                if action == 'retry':
                    continue
                if action != 'continue':
                    return
            break
    if args.stage == 'blnw':
        print('BL/NW stage complete. Review scan failures and figures. Use --stage qinj for a new run including initialization and a fresh scan.', flush=True)
        if args.interactive:
            interactive_checks(kcu, rb, pathfinder_profile=Path(MAPPING_PATH).name == 'pathfinder_mapping.yaml', mapping=mapping, chips=chips, ids=ids, blnw_results=locals().get("blnw_results"))
        return
    while True:
        try:
            announce('8. Pixel thresholds', 'Apply baseline-derived DAC thresholds and activate scanned pixels.', 'Threshold configuration must complete before acquisition.')
            import time

            THRESHOLD_OFFSET = args.offset   # DAC code; threshold = baseline + this offset
            L1A_DELAY_DEFAULT = args.l1a_delay

            # Full-range TOA/TOT/Cal trigger- and data-path windows: accept every hit
            # instead of filtering by pulse shape. Narrow these later if you actually
            # want TOA/TOT/Cal-based selection.
            TOA_RANGE = (0, 0x3ff)
            TOT_RANGE = (0, 0x1ff)
            CAL_RANGE = (0, 0x3ff)

            if not basic_checks_passed:
                raise RuntimeError("Complete the E-link lock check (6) successfully before configuring thresholds")
            if not blnw_scan_passed:
                raise RuntimeError("Complete the BL/NW scan before configuring thresholds")


            def configure_thresholds(chips, blnw_results, offset=THRESHOLD_OFFSET,
                                      l1a_delay=L1A_DELAY_DEFAULT, reset_first=False,
                                      toa_range=TOA_RANGE, tot_range=TOT_RANGE, cal_range=CAL_RANGE):
                """Apply DAC threshold (baseline + offset), activate the pixels that
                were BL/NW-scanned in section 7.1, and open their TOA/TOT/Cal trigger-
                and data-path windows. Reuses baseline values already in memory; does
                not reload from disk. A pixel with baseline=None (failed scan) is
                skipped, not assigned a threshold.

                reset_first=True calls etroc.reset() per chip, which internally also
                calls rb.rerun_bitslip() (see tamalero/ETROC.py). Re-run the link-lock
                check in section 6 afterward if you need to confirm links are still
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
        except Exception as exc:
            if not args.interactive:
                raise
            print(f"[FAILED] {CURRENT_STEP}: {type(exc).__name__}: {exc}")
            if workflow_menu(locals(), args, failed=True) == "retry":
                continue
            return
        if args.step:
            action = workflow_menu(locals(), args)
            if action == "retry":
                continue
            if action != "continue":
                return
        break
    while True:
        try:
            announce('9. Self-trigger configuration', 'Apply the selected trigger mask, combination logic and delay.', 'Inspect mask, lock and FC diagnostic output.')
            import time

            if not basic_checks_passed:
                raise RuntimeError("Complete the E-link lock check (6) successfully before configuring self-trigger")

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
                                    raise RuntimeError(f"{identity} elink {elink} FAILED to lock after self-trigger setup")

                show_status("Self-trigger configured", "success")
                # FC status: read invalidFCCount twice; unchanged means no new invalid
                # fast commands. A constant nonzero count is not an error.
                for identity, etroc in chips.items():
                    first = etroc.get_invalidFCCount()
                    time.sleep(0.5)
                    second = etroc.get_invalidFCCount()
                    fc_status = second == first
                    clk_delay = etroc.rd_reg("fcClkDelayEn")
                    data_delay = etroc.rd_reg("fcDataDelayEn")
                    print(f"{identity}: FC status={fc_status}, FC delay: clk={clk_delay}, data={data_delay}"
                          + ("" if fc_status else f" (invalid FC count {first} -> {second})"))


            configure_self_trigger(rb, mapping, ids, chips, trigger_delay=TRIGGER_DELAY_DEFAULT, trigger_board_ids=trigger_ids, trigger_combination_logic=TRIGGER_COMBINATION_LOGIC, trigger_data_size=args.trigger_bits)

        except Exception as exc:
            if not args.interactive:
                raise
            print(f"[FAILED] {CURRENT_STEP}: {type(exc).__name__}: {exc}")
            if workflow_menu(locals(), args, failed=True) == "retry":
                continue
            return
        if args.step:
            action = workflow_menu(locals(), args)
            if action == "retry":
                continue
            if action != "continue":
                return
        break
    while True:
        try:
            announce('10. QInj preparation', 'Enable injection on calibrated pixels.', 'No automatic trigger/FC delay tuning is performed.')
            import time
            from datetime import datetime, timezone
            from tamalero.FIFO import FIFO
            from tamalero.DataFrame import DataFrame   # adjust the import path if DataFrame lives elsewhere in this driver

            if not basic_checks_passed:
                raise RuntimeError("Complete the E-link lock check (6) successfully before running charge injection")

            QINJ_COUNT_DEFAULT = args.qinj_count
            EMPTY_SLOT_BCID = 500   # must be < 3563
            CHARGE_FC_DEFAULT = args.charge_fc   # placeholder - confirm the injected charge (fC) used on this setup


            def enable_qinj_pixels(chips, blnw_results, charge_fc=CHARGE_FC_DEFAULT):
                """Enable charge injection on the pixels scanned in section 7.1.

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
                    show_status(f"{identity}: QInj enabled on {count} pixels (offset={THRESHOLD_OFFSET}, charge={charge_fc}fC)", "info")
                return enabled


            def disable_qinj_pixels(chips, blnw_results):
                """Turn QInjEn back off on the pixels scanned in section 7.1."""
                scan_pixels = blnw_results["pixels"]
                for identity, etroc in chips.items():
                    for row, col in scan_pixels:
                        etroc.wr_reg("QInjEn", 0, row=row, col=col, broadcast=False)
                show_status("QInj disabled on scanned pixels", "info")


            qinj_enabled_pixels = enable_qinj_pixels(chips, blnw_results, charge_fc=CHARGE_FC_DEFAULT)


            def run_charge_injection_test(chips, qinj_count=QINJ_COUNT_DEFAULT,
                                           empty_slot_bcid=EMPTY_SLOT_BCID,
                                           report_interval=10, max_iterations=1,
                                           show_per_elink=True, print_records=False):
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
                configure_readout_links(rb, mapping, list(chips))
                rb.rerun_bitslip()
                if not all(item["all_locked"] for item in check_links(rb, mapping, list(chips)).values()):
                    raise RuntimeError("Link lock lost during QInj preparation; no injection sent")
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

                # Per-E-link bookkeeping (added; does not change the totals above).
                # The E-link number in an event word is 28 * bank + lpGBT elink
                # (DAQ lpGBT 0..27, TRIG lpGBT 28..55), e.g. TRIG E4 -> 32, E10 -> 38.
                elink_names = {}
                for identity in chips:
                    for bank, elinks in build_elinks(mapping, identity).items():
                        for e in elinks:
                            elink_names[28 * bank + e] = identity
                per_elink = {n: {"header": 0, "data": 0, "trailer": 0} for n in elink_names}
                records = {n: [] for n in elink_names}
                unexpected_elinks = {}

                start_time = datetime.now(timezone.utc)
                last_report_time = time.time()

                rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.EVENT_CNT_RESET", 0x1)
                fifo.send_Qinj_only(count=qinj_count)  # one burst, shared by enabled pixels
                time.sleep(1)
                event_cnt = rb.kcu.read_node(f"READOUT_BOARD_{rb.rb}.EVENT_CNT").value()
                show_status(f"Event count in 1 second: {event_cnt}", "info")
                time.sleep(0.01)

                # Expected elinks computed from the mapping instead of hardcoded, so this
                # works regardless of which/how many boards are selected.
                expected_elinks = set()
                for identity in chips:
                    for bank, elinks in build_elinks(mapping, identity).items():
                        expected_elinks.update(28 * bank + e for e in elinks)

                for iteration in range(max_iterations):
                    try:
                        print(f"Occupancy: {fifo.get_occupancy()}")
                        data = fifo.pretty_read(df)
                        time.sleep(0.1)  # slow down DAQ to avoid uhal UDP errors

                        if len(data) > 0:
                            flashing_headers = {}
                            for event in data:
                                if (event and len(event) >= 2 and event[0] in ("header", "data", "trailer")
                                        and isinstance(event[1], dict) and "elink" in event[1]):
                                    n = event[1]["elink"]
                                    if n not in elink_names:
                                        # Not an E-link of the selected ETROCs: do not count or print it here.
                                        unexpected_elinks[n] = unexpected_elinks.get(n, 0) + 1
                                        continue
                                    per_elink[n][event[0]] += 1
                                    records[n].append(event)
                                if event and len(event) >= 2:
                                    if event[0] == "header":
                                        if print_records:
                                            pass  # Complete records are printed by E-link below.
                                        trigger_cnt += 1
                                        if int(event[1]["bcid"]) == 1:
                                            flashing_headers[event[1]["elink"]] = event
                                    elif event[0] == "data":
                                        if print_records:
                                            pass  # Complete records are printed by E-link below.
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
                        raise  # UDP/read errors invalidate this acquisition; do not report a partial success.

                current_mask = rb.kcu.read_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK").value()
                show_status(f"Readback TRIG_ENABLE_MASK: 0x{current_mask:X}", "info")
                show_status(f"Total hits: {hit_counter}, trigger count: {trigger_cnt}, trailer count: {trailer_cnt}", "info")
                if hit_counter:
                    show_status(f"ea clean: {ea_clean}, ea issue: {ea_issue}",
                                "success" if ea_issue == 0 else "warning")

                if show_per_elink:
                    print("\nPer E-link (header / data / trailer):")
                    print(f"  {'E-link':>6}  {'ETROC':<8} {'header':>7} {'data':>7} {'trailer':>8}")
                    for n in sorted(per_elink):
                        c = per_elink[n]
                        print(f"  {n:>6}  {elink_names.get(n, '?'):<8} {c['header']:>7} {c['data']:>7} {c['trailer']:>8}")
                    silent = [n for n in elink_names if not any(per_elink[n].values())]
                    if silent:
                        show_status("No packets at all from E-link(s): "
                                    + ", ".join(f"{n} ({elink_names[n]})" for n in sorted(silent)), "warning")
                if unexpected_elinks:
                    show_status("Ignored packets from E-links outside the selection: "
                                + ", ".join(f"{n} (x{c})" for n, c in sorted(unexpected_elinks.items())), "warning")
                if print_records:
                    for n in sorted(records):
                        print(f"\n===== E-link {n} ({elink_names.get(n, '?')}): {len(records[n])} records =====")
                        if not records[n]:
                            print("No records")
                        for item in records[n]:
                            print(item)

                return {
                    "trigger_cnt": trigger_cnt,
                    "hit_counter": hit_counter,
                    "trailer_cnt": trailer_cnt,
                    "ea_clean": ea_clean,
                    "ea_issue": ea_issue,
                    "per_elink": per_elink,
                    "records": records,
                    "elink_names": elink_names,
                }


            def interactive_acquire(charge, count, reads, records):
                current = check_links(rb, mapping, ids)
                if not all(result["all_locked"] for result in current.values()):
                    raise RuntimeError("Selected links are not locked; use compare/phase diagnostics before QInj")
                enable_qinj_pixels(chips, blnw_results, charge_fc=charge)
                result = run_charge_injection_test(chips, qinj_count=count,
                                                   max_iterations=reads, print_records=records)
                result["settings"] = {"charge_fc": charge, "qinj_count": count, "fifo_reads": reads}
                return result

        except Exception as exc:
            if not args.interactive:
                raise
            print(f"[FAILED] {CURRENT_STEP}: {type(exc).__name__}: {exc}")
            if workflow_menu(locals(), args, failed=True) == "retry":
                continue
            return
        if args.step:
            action = workflow_menu(locals(), args)
            if action == "retry":
                continue
            if action != "continue":
                return
        break
    while True:
        try:
            announce("11. Data Acquisition", "Use current prepared settings.", "Errors return to debugging; retry resends an injection burst and resets FIFO.")
            if args.stage == "fc-scan":
                announce("11.1. FC input delay scan", "Scan all four clk/data delay-bit combinations: 00, 01, 10, 11.",
                         "Original FC bits are restored, even on acquisition failure; other chips remain unchanged.")
                run_fc_scan(chips, args.fc_target,
                            lambda: run_charge_injection_test(chips, qinj_count=QINJ_COUNT_DEFAULT,
                                                              max_iterations=args.fifo_reads, print_records=args.print_records),
                            args.scan_output)
            else:
                qinj_results = run_charge_injection_test(chips, qinj_count=QINJ_COUNT_DEFAULT,
                                                        max_iterations=args.fifo_reads, print_records=args.print_records)
                save_diagnostic(args.scan_output, "qinj", qinj_results)

            # Uncomment once you're done running charge injection on this selection,
            # so these pixels stop responding to injection pulses in later runs:
            # disable_qinj_pixels(chips, blnw_results)

            # Notebook code cell 42
        except Exception as exc:
            if not args.interactive:
                raise
            print(f"[FAILED] {CURRENT_STEP}: {type(exc).__name__}: {exc}")
            if workflow_menu(locals(), args, failed=True) == "retry":
                continue
            return
        if args.step:
            action = workflow_menu(locals(), args)
            if action == "retry":
                continue
            if action != "continue":
                return
        break
    while True:
        try:
            announce('12. End of test', 'Mask the declared readout banks using the notebook cleanup step.', 'This does not power down chips or disable every external trigger source.')
            configure_readout_links(rb, mapping, [])
            print(f"Basic checks passed: {basic_checks_passed}")
            show_status("All declared readout banks are masked", "info")

            print("QInj workflow finished; declared data links are masked. Review readout results above.", flush=True)
            if args.interactive:
                interactive_checks(kcu, rb, pathfinder_profile=Path(MAPPING_PATH).name == "pathfinder_mapping.yaml",
                                       mapping=mapping, chips=chips, ids=ids,
                                       qinj_context={"pixels": blnw_results["pixels"], "charge_fc": args.charge_fc, "qinj_count": args.qinj_count,
                                                     "acquire": interactive_acquire}, output_dir=args.scan_output, blnw_results=locals().get("blnw_results"))
        except Exception as exc:
            if not args.interactive:
                raise
            print(f"[FAILED] {CURRENT_STEP}: {type(exc).__name__}: {exc}")
            if workflow_menu(locals(), args, failed=True) == "retry":
                continue
            return
        if args.step:
            action = workflow_menu(locals(), args)
            if action == "retry":
                continue
            if action != "continue":
                return
        break


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

