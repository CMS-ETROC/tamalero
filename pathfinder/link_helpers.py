"""
link_helpers.py -- two small helpers for the Pathfinder notebook.

Why: after any change that affects the ETROC -> lpGBT -> FPGA elink data
(rate change, ETROC reset / PLL / serRate / singlePort writes, power cycle of
a daughter card) the FPGA receiver's bit alignment is lost and elinks show
locked=0, filler=0, err=65535 until rb.rerun_bitslip() is called again.

    from link_helpers import set_link_rate, ensure_locked

    # change the rate in ONE place (FPGA + lpGBT + ETROC) and re-align:
    set_link_rate(rb, chips, 320)          # or 640
    # verify; re-aligns and retries if something is not locked:
    ensure_locked(rb, mapping, ids)

All calls used here exist in the tamalero files you have:
  rb.kcu.write_node, lpgbt.wr_reg, etroc.set_dataRate(port, rate),
  rb.rerun_bitslip, pathfinder_control.check_links.
"""

import time

# FPGA READOUT_BOARD_n.ELINK_WIDTH encoding (see ReadoutBoard.set_elink_width)
_FPGA_WIDTH = {320: 2, 640: 3, 1280: 4}
# lpGBT EPRXnDATARATE encoding
_EPRX_RATE = {320: 1, 640: 2, 1280: 3}


def set_link_rate(rb, chips, mbps=320, include_trig=False, settle=0.2):
    """Set FPGA, lpGBT and ETROC uplink rate together, then re-run bitslip.

    chips: dict of initialized ETROC objects (their serRateLeft/Right are set).
    include_trig: also write the TRIG lpGBT EPRX rates (default: DAQ only,
        same as ReadoutBoard.set_elink_width).
    """
    if mbps not in _FPGA_WIDTH:
        raise ValueError(f"mbps must be one of {sorted(_FPGA_WIDTH)}")

    rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.ELINK_WIDTH", _FPGA_WIDTH[mbps])

    lpgbts = [rb.DAQ_LPGBT]
    if include_trig and getattr(rb, "TRIG_LPGBT", None) is not None:
        lpgbts.append(rb.TRIG_LPGBT)
    for lp in lpgbts:
        for group in range(7):
            lp.wr_reg(f"LPGBT.RWF.EPORTRX.EPRX{group}DATARATE", _EPRX_RATE[mbps])

    for etroc in chips.values():
        for port in ("Left", "Right"):
            etroc.set_dataRate(port, mbps)

    time.sleep(settle)
    rb.rerun_bitslip()
    print(f"[set_link_rate] FPGA/lpGBT/ETROC set to {mbps} Mbps, bitslip re-run")


def ensure_locked(rb, mapping, ids, retries=3, wait=0.5, verbose=True):
    """check_links(); if any selected elink is unlocked, re-run bitslip and
    check again, up to `retries` times. Returns the last check_links() result."""
    from pathfinder_control import check_links

    result = check_links(rb, mapping, ids)
    for attempt in range(retries):
        if all(r["all_locked"] for r in result.values()):
            break
        if verbose:
            bad = [(i, l["elink"]) for i, r in result.items()
                   for l in r["links"] if not l["locked"]]
            print(f"[ensure_locked] not locked: {bad}; re-running bitslip "
                  f"({attempt + 1}/{retries})")
        rb.rerun_bitslip()
        time.sleep(wait)
        result = check_links(rb, mapping, ids)
    if verbose:
        ok = all(r["all_locked"] for r in result.values())
        print(f"[ensure_locked] {'all selected elinks locked' if ok else 'STILL NOT LOCKED'}")
    return result


def probe_bitslip(rb, elink=14, slave=False, times=(0.0, 0.05, 0.2, 0.5, 1.0)):
    """Find out WHY an unlocked elink locks. Run it while the elink is NOT
    locked. It prints lock status at several delays after (a) a bare
    rerun_bitslip() and, if still unlocked, (b) after reset_data_error_count().

    Reading: locked only at the later delays in (a)  -> just timing, wait
    longer after rerun_bitslip(); locked only in (b) -> counter reset matters;
    random between runs -> bitslip is not deterministic, retry it.
    """
    def sample(tag):
        t0 = time.time()
        out = []
        for t in times:
            dt = t - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
            out.append((t, bool(rb.etroc_locked(elink, slave=slave))))
        print(f"  {tag:<28}", "  ".join(f"{t:>4}s:{'LOCK' if l else '----'}" for t, l in out))
        return out[-1][1]

    print(f"[probe_bitslip] elink {elink} ({'TRIG' if slave else 'DAQ'})")
    print(f"  locked before anything       : {bool(rb.etroc_locked(elink, slave=slave))}")
    rb.rerun_bitslip()
    if sample("after bare rerun_bitslip"):
        return "locked after bare rerun_bitslip (timing/OK)"
    rb.reset_data_error_count()
    if sample("after reset_data_error_count"):
        return "locked only after reset_data_error_count"
    return "still not locked after both"


def find_lock_step(rb, elink=14, slave=False, wait=1.0):
    """From an UNLOCKED state, try actions from least to most invasive and
    report the first one after which the elink is locked. Stops at the first
    success, so run it again (after it unlocks) to get a second data point.
    """
    def locked():
        return bool(rb.etroc_locked(elink, slave=slave))

    steps = [
        ("1. do nothing, just wait", lambda: None),
        ("2. read counters only (filler/error/packet/data)",
         lambda: [f(elink, slave=slave) for f in (rb.read_filler_rate, rb.read_error_count,
                                                   rb.read_packet_count, rb.read_data_count)]),
        ("3. reset_data_error_count()", rb.reset_data_error_count),
        ("4. rerun_bitslip()", rb.rerun_bitslip),
        ("5. rerun_bitslip() + reset_data_error_count()",
         lambda: (rb.rerun_bitslip(), rb.reset_data_error_count())),
    ]
    print(f"[find_lock_step] elink {elink} ({'TRIG' if slave else 'DAQ'}); "
          f"locked at start: {locked()}")
    if locked():
        print("  already locked -- run this while the elink is NOT locked")
        return None
    for name, action in steps:
        action()
        time.sleep(wait)
        ok = locked()
        print(f"  {name:<52} -> {'LOCKED' if ok else 'not locked'}")
        if ok:
            return name
    print("  never locked with any of these steps")
    return None
