"""
check_elink_lock.py
===================
Diagnose why one elink (e.g. D5H1 -> elink 14 on the DAQ/MASTER lpGBT) shows
ETROC_LOCKED=0, filler_rate=0, error_cnt=65535 even though I2C (bl/nw scan)
works and the lpGBT uplink is READY.

Run from INSIDE your notebook kernel (it needs the live `rb` object):

    import importlib, check_elink_lock
    importlib.reload(check_elink_lock)
    check_elink_lock.check_elink(rb, elink=14)                 # DAQ side
    check_elink_lock.check_elink(rb, elink=14, slave=True)     # TRIG side
    check_elink_lock.check_elink(rb, elink=14, etroc=<your ETROC object>)

Only API calls that exist in the ReadoutBoard.py you shared are used
(etroc_locked, read_filler_rate, read_error_count, read_packet_count,
read_data_count, reset_data_error_count, rerun_bitslip, DAQ_LPGBT.rd_reg,
kcu.read_node). Nothing is guessed about the ETROC class: pass your ETROC
object and it just lists candidate method names for the ETROC-side check.

By default the script is read-only except for step 5, which pulses the
auto-bitslip (harmless; pass try_bitslip=False to skip it).
"""

import time

# FPGA ctrl.elink_width encoding (set_elink_width writes widths[w]+1):
#   2 -> 320 Mbps (all elinks), 3 -> 640 Mbps (even), 4 -> 1280 Mbps (every 4th)
FPGA_WIDTH_TO_MBPS = {2: 320, 3: 640, 4: 1280}
FPGA_WIDTH_STEP = {2: 1, 3: 2, 4: 4}      # elink % step must be 0 to be enabled
# lpGBT EPRXnDATARATE encoding: 1 -> 320, 2 -> 640, 3 -> 1280 (= fpga_width - 1)

EXPECTED_FILLER_RATE = 16_500_000          # same threshold as get_link_status()


def _node(rb, name):
    return rb.kcu.read_node(f"READOUT_BOARD_{rb.rb}.{name}").value()


def _lpgbt(rb, slave):
    return rb.TRIG_LPGBT if slave else rb.DAQ_LPGBT


def _rd(lpgbt, reg):
    for m in ("rd_reg", "read_reg"):
        f = getattr(lpgbt, m, None)
        if f is None:
            continue
        try:
            return f(reg)
        except Exception as e:
            return f"<err {e}>"
    return "<no rd_reg method>"


def _counters(rb, elink, slave):
    return dict(
        locked=rb.etroc_locked(elink, slave=slave),
        filler=rb.read_filler_rate(elink, slave=slave),
        err=rb.read_error_count(elink, slave=slave),
        pkt=rb.read_packet_count(elink, slave=slave),
        data=rb.read_data_count(elink, slave=slave),
    )


def _show(tag, c):
    print(f"  {tag:<22} locked={c['locked']!s:<5} filler={c['filler']:<10} "
          f"err={c['err']:<6} pkt={c['pkt']:<8} data={c['data']}")


def check_elink(rb, elink=14, slave=False, etroc=None, try_bitslip=True):
    sfx = "_SLAVE" if slave else ""
    side = "TRIG(slave)" if slave else "DAQ(master)"
    print(f"===== elink {elink} on {side} lpGBT =====\n")

    # ------------------------------------------------------------------
    print("--- 1. FPGA receiver enable / lock masks ---")
    dis = _node(rb, "ETROC_DISABLE" + sfx)
    lck = _node(rb, "ETROC_LOCKED" + sfx)
    print(f"  ETROC_DISABLE{sfx} = {dis:#09x}   bit{elink} = {(dis >> elink) & 1}"
          f"   ({'DISABLED -> call rb.enable_etroc_readout(only=%d%s)' % (elink, ', slave=True' if slave else '') if (dis >> elink) & 1 else 'enabled'})")
    print(f"  ETROC_LOCKED{sfx}  = {lck:#09x}   bit{elink} = {(lck >> elink) & 1}")
    enabled = [i for i in range(28) if not (dis >> i) & 1]
    print(f"  enabled elinks: {enabled}")

    # ------------------------------------------------------------------
    print("\n--- 2. elink width: FPGA vs lpGBT ---")
    fw = _node(rb, "ELINK_WIDTH")
    mbps = FPGA_WIDTH_TO_MBPS.get(fw)
    print(f"  FPGA ELINK_WIDTH = {fw}  -> {mbps} Mbps" if mbps
          else f"  FPGA ELINK_WIDTH = {fw}  (unexpected value, expect 2/3/4)")
    if fw in FPGA_WIDTH_STEP and elink % FPGA_WIDTH_STEP[fw] != 0:
        print(f"  !! At {mbps} Mbps the FPGA only enables elinks with "
              f"elink % {FPGA_WIDTH_STEP[fw]} == 0; elink {elink} is DISABLED by rate.")

    lp = _lpgbt(rb, slave)
    group = elink // 4
    print(f"  lpGBT EPRX group for elink {elink}: group {group} (elinks {group*4}-{group*4+3})")
    drs = {}
    for g in range(7):
        drs[g] = _rd(lp, f"LPGBT.RWF.EPORTRX.EPRX{g}DATARATE")
    print("  lpGBT EPRXnDATARATE (1=320, 2=640, 3=1280): " +
          ", ".join(f"g{g}={v}" for g, v in drs.items()))
    exp = fw - 1
    got = drs.get(group)
    if isinstance(got, int):
        ok = (got == exp)
        print(f"  expected group {group} datarate = {exp}, read {got} -> "
              f"{'MATCH' if ok else '!! MISMATCH between FPGA and lpGBT'}")
    # EPRX enable / tracking / per-channel settings, exactly as written by
    # LPGBT.configure_eprx() (expected: ENABLE=1, TRACKMODE=2, PHASESELECT=5,
    # INVERT=0, ACBIAS=1, TERM=1, EQ1=0). configure_eprx() only runs inside
    # power_up_init(), i.e. only when power_up_done() is False and only for the
    # DAQ lpGBT -- so check that it actually took effect.
    print("  expected after configure_eprx(): ENABLE=1 TRACKMODE=2 PHASESELECT=5 "
          "INVERT=0 ACBIAS=1 TERM=1 EQ1=0")
    for fld in (f"EPRX{group}TRACKMODE",
                f"EPRX{group}0ENABLE", f"EPRX{group}1ENABLE",
                f"EPRX{group}2ENABLE", f"EPRX{group}3ENABLE"):
        print(f"  LPGBT.RWF.EPORTRX.{fld} = {_rd(lp, 'LPGBT.RWF.EPORTRX.' + fld)}")
    for fld in ("PHASESELECT", "INVERT", "ACBIAS", "TERM", "EQ1"):
        reg = f"LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL.EPRX{elink}{fld}"
        print(f"  {reg} = {_rd(lp, reg)}")
    print(f"  LPGBT.RWF.EPORTRX.EPRXLOCKTHRESHOLD = "
          f"{_rd(lp, 'LPGBT.RWF.EPORTRX.EPRXLOCKTHRESHOLD')}  (expected 5)")

    # ------------------------------------------------------------------
    print("\n--- 3. FPGA bitslip / current counters ---")
    try:
        print(f"  BITSLIP_AUTO_EN = {_node(rb, 'BITSLIP_AUTO_EN')}")
    except Exception as e:
        print(f"  BITSLIP_AUTO_EN unreadable ({e})")
    c0 = _counters(rb, elink, slave)
    _show("now", c0)

    # ------------------------------------------------------------------
    print("\n--- 4. counters after reset + 1 s (is the error count still running?) ---")
    rb.reset_data_error_count()
    time.sleep(1.0)
    c1 = _counters(rb, elink, slave)
    _show("after reset+1s", c1)
    if c1["err"] >= 65535:
        print("  -> error counter saturates within 1 s: receiver sees a stream of "
              "invalid words (wrong width/rate/phase, or ETROC not sending filler).")
    elif c1["err"] == 0 and c1["filler"] == 0:
        print("  -> no errors AND no filler: line looks silent/constant; ETROC likely "
              "not transmitting, or elink not routed to the receiver.")

    # ------------------------------------------------------------------
    if try_bitslip:
        print("\n--- 5. re-run auto bitslip ---")
        rb.rerun_bitslip()
        rb.reset_data_error_count()
        time.sleep(1.0)
        c2 = _counters(rb, elink, slave)
        _show("after bitslip", c2)
        if c2["locked"] and c2["filler"] > EXPECTED_FILLER_RATE and c2["err"] < 1:
            print("  -> LOCKED after bitslip: it was an alignment problem.")

    # ------------------------------------------------------------------
    print("\n--- 6. ETROC2 side (needs your ETROC object) ---")
    if etroc is None:
        print("  no etroc passed. Re-run with etroc=chips['D5H1'] to read the "
              "ETROC registers that default_config() sets.")
    else:
        # register names taken from ETROC.default_config() in your tamalero
        # (expected there: singlePort=1 'right', mergeTriggerData=1,
        #  disScrambler=1, serRateLeft=1, serRateRight=1 (=640 Mbps),
        #  PLL_ENABLEPLL=1, controllerState=11, pllUnlockCount small/stable)
        for reg in ("singlePort", "serRateLeft", "serRateRight", "mergeTriggerData",
                    "disScrambler", "PLL_ENABLEPLL", "controllerState",
                    "pllUnlockCount", "disDataReadout"):
            try:
                print(f"  {reg:<18} = {etroc.rd_reg(reg)}")
            except Exception as e:
                print(f"  {reg:<18} <err {e}>")
        try:
            print("  get_dataRate('Right') =", etroc.get_dataRate('Right'),
                  "| get_singlePort() =", etroc.get_singlePort())
        except Exception as e:
            print("  get_dataRate/get_singlePort <err", e, ">")
        try:
            print("  FC_status() =", etroc.FC_status())
            print("  get_invalidFCCount() =", etroc.get_invalidFCCount())
        except Exception as e:
            print("  FC_status/get_invalidFCCount <err", e, ">")
    print("  consistent 640 Mbps chain should read: FPGA ELINK_WIDTH=3, lpGBT "
          "EPRXnDATARATE=2, ETROC serRateRight=1 + singlePort=1.")

    print("\nDone. Paste this whole output back.")
    return c0, c1


if __name__ == "__main__":
    print(__doc__)
