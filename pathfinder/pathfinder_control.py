"""Pathfinder mapping and initialization utilities.

Importing this module and running its CLI do not access hardware.
Calling initialize_etrocs without a test factory imports the ETROC driver
and may configure hardware.

Requires PyYAML: python -m pip install PyYAML
Run an offline mapping check:
    python pathfinder_control.py pathfinder_mapping.yaml
"""

import argparse
import re
from pathlib import Path

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently overwriting values."""


def _construct_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ValueError("YAML mapping keys must be hashable") from exc
        if duplicate:
            raise ValueError(
                f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _require_dict(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer from {low} to {high}")


def edin_to_elink(pin):
    """Convert EDINgc to the confirmed software index: 4 * group + channel."""
    match = re.fullmatch(r"EDIN([0-6])([0-3])", pin) if isinstance(pin, str) else None
    if match is None:
        raise ValueError(f"Invalid data input pin: {pin!r}")
    group, channel = map(int, match.groups())
    return group * 4 + channel


def build_elinks(mapping, etroc_id):
    """Return one ETROC's link mapping, retaining both DAQ and TRIG banks."""
    if etroc_id not in mapping["etrocs"]:
        raise ValueError(f"Unknown ETROC ID: {etroc_id}")
    result = {0: [], 1: []}
    for link in mapping["etrocs"][etroc_id]["data"]:
        bank = mapping["lpgbts"][link["lpgbt"]]["elink_bank"]
        result[bank].append(edin_to_elink(link["pin"]))
    return result


def build_trigger_mask(mapping, ids):
    """Build a 28-bit trigger mask from selected ETROC data links."""
    if isinstance(ids, str):
        raise ValueError("ids must be a list of ETROC IDs")

    trigger_mask = 0

    for identity in ids:
        for ilpgbt, elinks in build_elinks(mapping, identity).items():
            if ilpgbt not in (0, 1):
                raise ValueError("ilpgbt must be 0 (DAQ) or 1 (TRIG)")

            for elink in elinks:
                if elink not in range(0, 28, 2):
                    raise ValueError(
                        f"{identity}: trigger E-link must be even "
                        f"and within 0..26, got {elink}"
                    )

                trigger_bit = ilpgbt * 14 + elink // 2
                trigger_mask |= 1 << trigger_bit

    return trigger_mask


def validate_mapping(mapping):
    """Validate wiring entries without reading or changing any hardware.

    A subset of the 28 positions is allowed. Selection of missing positions
    fails explicitly. Clock fields are validated as metadata only.
    """
    _require_dict(mapping, "Root")
    if type(mapping.get("schema_version")) is not int or mapping["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    controllers = _require_dict(mapping.get("lpgbts"), "lpgbts")
    if "daq" not in controllers or set(controllers) - {"daq", "trig"}:
        raise ValueError("lpgbts must declare daq and optionally trig only")
    for name, hardware_id, bank in (("daq", 1, 0), ("trig", 2, 1)):
        if name not in controllers:
            continue
        cfg = _require_dict(controllers.get(name), f"lpgbts.{name}")
        for field, expected in (("hardware_id", hardware_id), ("elink_bank", bank)):
            if type(cfg.get(field)) is not int or cfg[field] != expected:
                raise ValueError(f"lpgbts.{name}.{field} must be {expected}")

    upstream = None
    if "trig" in controllers:
        upstream = _require_dict(controllers["trig"].get("control"), "trig.control")
        if upstream.get("via") != "daq":
            raise ValueError("trig.control.via must be 'daq'")
        _integer(upstream.get("channel"), 0, 2, "trig.control.channel")
        _integer(upstream.get("address"), 0, 127, "trig.control.address")

    etrocs = _require_dict(mapping.get("etrocs"), "etrocs")
    if not etrocs:
        raise ValueError("etrocs must not be empty")
    positions = {}
    buses = {}
    if upstream is not None:
        buses[("daq", upstream["channel"], upstream["address"])] = "TRIG control"
    links = {}
    for identity, cfg in etrocs.items():
        _require_dict(cfg, str(identity))
        daughter, hybrid = cfg.get("daughter"), cfg.get("hybrid")
        _integer(daughter, 1, 7, f"{identity}.daughter")
        _integer(hybrid, 1, 4, f"{identity}.hybrid")
        if identity != f"D{daughter}H{hybrid}":
            raise ValueError(f"ID {identity!r} does not match its Daughter/Hybrid position")
        position = (daughter, hybrid)
        if position in positions:
            raise ValueError(f"Duplicate physical position: {position}")
        positions[position] = identity
        i2c = _require_dict(cfg.get("i2c"), f"{identity}.i2c")
        if i2c.get("lpgbt") not in controllers:
            raise ValueError(f"{identity}: invalid I2C controller")
        _integer(i2c.get("channel"), 0, 2, f"{identity}.i2c.channel")
        _integer(i2c.get("address"), 0, 127, f"{identity}.i2c.address")
        route = (i2c["lpgbt"], i2c["channel"], i2c["address"])
        if route in buses:
            raise ValueError(f"I2C conflict between {identity} and {buses[route]}: {route}")
        buses[route] = identity
        data = cfg.get("data")
        if not isinstance(data, list) or not data:
            raise ValueError(f"{identity}.data must be a non-empty list")
        for link in data:
            _require_dict(link, f"{identity}.data entry")
            if link.get("lpgbt") not in controllers:
                raise ValueError(f"{identity}: invalid data controller")
            route = (link["lpgbt"], edin_to_elink(link.get("pin")))
            if route in links:
                raise ValueError(f"Data link conflict between {identity} and {links[route]}: {route}")
            links[route] = identity
        if "clock" in cfg:
            clock = _require_dict(cfg["clock"], f"{identity}.clock")
            if clock.get("lpgbt") not in controllers:
                raise ValueError(f"{identity}: invalid clock controller")
            pin = clock.get("pin")
            if not isinstance(pin, str) or not re.fullmatch(r"ECLK(?:[0-9]|[12][0-9])", pin):
                raise ValueError(f"{identity}: invalid clock output pin")
    return mapping


def load_mapping(path):
    """Load YAML, reject duplicate keys, and validate wiring constraints."""
    with Path(path).open(encoding="utf-8") as stream:
        mapping = yaml.load(stream, Loader=UniqueKeyLoader)
    return validate_mapping(mapping)


def expand_selection(mapping, selection):
    """Expand {1: 'all', 7: [1]} into IDs in the supplied selection order.

    'all' selects the Hybrids actually declared for that module, sorted by
    Hybrid number. This does not enable or disable hardware links.
    """
    _require_dict(selection, "selection")
    result = []
    for module, hybrids in selection.items():
        _integer(module, 1, 7, "Module ID")
        if isinstance(hybrids, str) and hybrids == "all":
            hybrids = sorted(cfg["hybrid"] for cfg in mapping["etrocs"].values()
                             if cfg["daughter"] == module)
        elif not isinstance(hybrids, (list, tuple)):
            raise ValueError(f"Module {module}: use 'all' or a Hybrid ID list")
        if not hybrids:
            raise ValueError(f"Module {module}: Hybrid selection is empty")
        for hybrid in hybrids:
            _integer(hybrid, 1, 4, f"Module {module} Hybrid ID")
            identity = f"D{module}H{hybrid}"
            if identity not in mapping["etrocs"]:
                raise ValueError(f"ETROC {identity} is missing from the mapping")
            if identity in result:
                raise ValueError(f"Duplicate ETROC selection: {identity}")
            result.append(identity)
    if not result:
        raise ValueError("No ETROCs selected")
    return result


class InitializationError(RuntimeError):
    """Expose completed devices after a partial initialization failure.

    The failing device may also have been partially configured. No rollback
    or automatic reset is performed.
    """

    def __init__(self, message, failed_id, initialized_chips):
        super().__init__(message)
        self.failed_id = failed_id
        self.initialized_chips = dict(initialized_chips)


def initialize_etrocs(rb, mapping, ids, *, etroc_factory=None,
                      strict=False, verbose=False):
    """Initialize selected ETROCs and return objects indexed by position ID.

    Preflight the complete selection before constructing any ETROC.
    Construction may write registers and perform driver-defined resets.
    Stop on the first failure and preserve completed objects in the exception.
    Supply etroc_factory to test routing without hardware.
    """
    validate_mapping(mapping)
    if isinstance(ids, (str, bytes)):
        raise ValueError("ids must be a sequence of ETROC IDs, not a string")
    ids = list(ids)
    if not ids:
        raise ValueError("No ETROCs selected")

    seen = set()
    controllers = {}
    for identity in ids:
        if not isinstance(identity, str):
            raise ValueError("ETROC IDs must be strings")
        if identity in seen:
            raise ValueError(f"Duplicate ETROC selection: {identity}")
        if identity not in mapping["etrocs"]:
            raise ValueError(f"Unknown ETROC ID: {identity}")
        seen.add(identity)
        name = mapping["etrocs"][identity]["i2c"]["lpgbt"]
        attribute = {"daq": "DAQ_LPGBT", "trig": "TRIG_LPGBT"}[name]
        controller = getattr(rb, attribute, None)
        if controller is None:
            raise RuntimeError(f"{identity}: controller {attribute} is not available")
        if name == "trig" and not getattr(rb, "trigger", False):
            raise RuntimeError(f"{identity}: TRIG lpGBT is not enabled")
        controllers[name] = controller

    if etroc_factory is None:
        from tamalero.ETROC import ETROC
        etroc_factory = ETROC

    chips = {}
    for identity in ids:
        i2c = mapping["etrocs"][identity]["i2c"]
        try:
            chip = etroc_factory(
                rb,
                master="lpgbt",
                i2c_master=controllers[i2c["lpgbt"]],
                i2c_channel=i2c["channel"],
                i2c_adr=i2c["address"],
                elinks=build_elinks(mapping, identity),
                strict=strict,
                verbose=verbose,
            )
            if not chip.is_connected():
                raise RuntimeError("ETROC did not respond")
        except Exception as exc:
            completed = ", ".join(chips) or "none"
            raise InitializationError(
                f"Failed to initialize {identity} via {i2c['lpgbt']}/"
                f"M{i2c['channel']}/0x{i2c['address']:02X}. "
                f"Previously initialized ETROCs: {completed}",
                identity, chips,
            ) from exc
        chips[identity] = chip
    return chips


def check_connections(chips):
    """Check I2C connectivity per ETROC without configuring or resetting it.

    Return {id: {connected: bool, error: str | None}}. A failed device does
    not prevent checking the remaining devices. Connectivity follows the
    supplied driver's is_connected() semantics; it does not verify identity.
    """
    _require_dict(chips, "chips")
    results = {}
    for identity, chip in chips.items():
        try:
            if chip is None:
                raise ValueError("ETROC object is missing")
            connected = bool(chip.is_connected())
            results[identity] = {
                "connected": connected,
                "error": None if connected else "ETROC did not respond",
            }
        except Exception as exc:
            results[identity] = {
                "connected": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return results


def check_links(rb, mapping, ids):
    """Read lock status for selected ETROCs without enabling or realigning links.

    Return {id: {all_locked: bool, links: list}}. Each link record includes
    lpgbt, bank, elink, locked and error. Read failures are recorded per link
    and do not prevent checking the remaining links. False lock status alone
    is not a read exception, so its error field is None.
    """
    validate_mapping(mapping)
    if isinstance(ids, (str, bytes)):
        raise ValueError("ids must be a sequence of ETROC IDs, not a string")
    ids = list(ids)
    if not ids:
        raise ValueError("No ETROCs selected")
    seen = set()
    # Reject invalid selections before any status reads.
    for identity in ids:
        if not isinstance(identity, str) or identity not in mapping["etrocs"]:
            raise ValueError(f"Unknown ETROC ID: {identity!r}")
        if identity in seen:
            raise ValueError(f"Duplicate ETROC selection: {identity}")
        seen.add(identity)

    results = {}
    for identity in ids:
        records = []
        for bank, elinks in build_elinks(mapping, identity).items():
            for elink in elinks:
                record = {
                    "lpgbt": "daq" if bank == 0 else "trig",
                    "bank": bank,
                    "elink": elink,
                    "locked": False,
                    "error": None,
                }
                try:
                    record["locked"] = bool(rb.etroc_locked(elink, slave=(bank == 1)))
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                records.append(record)
        results[identity] = {
            "all_locked": bool(records) and all(r["locked"] for r in records),
            "links": records,
        }
    return results


def build_readout_masks(mapping, ids):
    """Build 28-bit disable masks for declared lpGBT banks only.

    A set bit disables a link. An empty ID list disables all declared banks.
    DAQ-only maps return {0: mask}, without an entry for absent TRIG hardware.
    This function does not access hardware.
    """
    validate_mapping(mapping)
    if isinstance(ids, (str, bytes)):
        raise ValueError("ids must be a sequence of ETROC IDs, not a string")
    masks = {cfg["elink_bank"]: (1 << 28) - 1
             for cfg in mapping["lpgbts"].values()}
    seen = set()
    for identity in ids:
        if not isinstance(identity, str) or identity not in mapping["etrocs"]:
            raise ValueError(f"Unknown ETROC ID: {identity!r}")
        if identity in seen:
            raise ValueError(f"Duplicate ETROC selection: {identity}")
        seen.add(identity)
        for bank, links in build_elinks(mapping, identity).items():
            for elink in links:
                masks[bank] &= ~(1 << elink)
    return masks


def configure_readout_links(rb, mapping, ids):
    """Replace declared FPGA readout bank masks and verify their readback.

    Call with acquisition stopped. Writes are sequential, not atomic.
    Unselected links, including unmapped links, are disabled. Existing FIFO
    contents are not cleared. No power, I2C, injection, reset or bitslip
    settings are changed. A failure raises without automatic rollback;
    either bank may already have changed and must be checked before resuming.
    Undeclared banks are not accessed. Returns after all readbacks match.
    """
    masks = build_readout_masks(mapping, ids)
    nodes = {
        0: f"READOUT_BOARD_{rb.rb}.ETROC_DISABLE",
        1: f"READOUT_BOARD_{rb.rb}.ETROC_DISABLE_SLAVE",
    }
    for bank, node in nodes.items():
        if bank not in masks:
            continue
        try:
            rb.kcu.write_node(node, masks[bank])
            actual = int(rb.kcu.read_node(node).value())
            if actual != masks[bank]:
                raise RuntimeError(
                    f"Readback mismatch: expected 0x{masks[bank]:07X}, "
                    f"received 0x{actual:07X}"
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to configure {node}: {exc}. "
                "Readout masks may be partially updated; no rollback was performed."
            ) from exc
    return masks


import time
from html import escape
from IPython.display import HTML, display


def read_optical_link_status(rb, *, verbose=True):
    """Read initialized uplinks without changing configuration or counters."""
    result = {}
    for ilpgbt, name in [(0, "DAQ"), (1, "TRIG")]:
        if ilpgbt == 1 and not getattr(rb, "trigger", False):
            continue
        prefix = f"READOUT_BOARD_{rb.rb}.LPGBT.UPLINK_{ilpgbt}"
        result[name] = {
            "ready": rb.kcu.read_node(f"{prefix}.READY").value(),
            "fec": rb.kcu.read_node(f"{prefix}.FEC_ERR_CNT").value(),
        }
    if verbose:
        print(format_optical_link_status(result))
    return result


def format_optical_link_status(result):
    """Format a previously read sample without accessing hardware."""
    return " | ".join(
        f"{name}: READY={link['ready']}, FEC_ERR_CNT={link['fec']}"
        for name, link in result.items()
    )


def monitor_optical_links(rb, seconds=10, *, verbose=True):
    """Sample uplinks immediately and once per second; do not reset counters."""
    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 1:
        raise ValueError("seconds must be a positive integer")
    samples = [read_optical_link_status(rb, verbose=False)]
    for second in range(1, seconds + 1):
        time.sleep(1)
        sample = read_optical_link_status(rb, verbose=False)
        samples.append(sample)
        if verbose:
            print(f"{second:2d}s | {format_optical_link_status(sample)}")
    return samples


def reset_optical_fec_counters(rb):
    """Print before/after values and clear both uplink FEC counters."""
    print("Before reset:")
    before = read_optical_link_status(rb)
    rb.reset_FEC_error_count(quiet=True)
    print("After reset:")
    after = read_optical_link_status(rb)
    return {"before": before, "after": after}


def read_clock_configuration(rb, *, verbose=True):
    """Read ECLK0-27 on initialized lpGBTs; retain per-channel errors."""
    chips = {"daq": rb.DAQ_LPGBT}
    if getattr(rb, "TRIG_LPGBT", None) is not None:
        chips["trig"] = rb.TRIG_LPGBT
    results = {}
    for name, chip in chips.items():
        results[name] = []
        if verbose:
            print(f"\n{name.upper()} clock configuration")
        for channel in range(28):
            prefix = f"LPGBT.RWF.EPORTCLK.EPCLK{channel}"
            entry = {"channel": channel, "frequency": None,
                     "drive_strength": None, "error": None}
            try:
                entry["frequency"] = chip.rd_reg(prefix + "FREQ")
                entry["drive_strength"] = chip.rd_reg(prefix + "DRIVESTRENGTH")
                if verbose:
                    print(f"ECLK{channel:02d}: FREQ={entry['frequency']}, DRIVESTRENGTH={entry['drive_strength']}")
            except Exception as exc:
                entry["error"] = str(exc)
                if verbose:
                    print(f"ECLK{channel:02d}: read failed: {exc}")
            results[name].append(entry)
    return results


def initialize_pathfinder_optical_links(rb, monitor_seconds=10):
    """Configure and verify both Pathfinder optical uplinks.

    Run after the existing motherboard and ECLK/E-link initialization.
    Current-board settings are based on the successfully tested setup.
    """
    def status(message, ok=True):
        color = "#16803c" if ok else "#c62828"
        display(HTML(
            f'<div style="color:{color};font-weight:600">'
            f'{escape(message)}</div>'
        ))

    if monitor_seconds < 1:
        raise ValueError("monitor_seconds must be at least 1")

    if not getattr(rb, "trigger", False) or not hasattr(rb, "TRIG_LPGBT"):
        raise RuntimeError("TRIG lpGBT was not detected during initialization")

    chips = {
        "DAQ": rb.DAQ_LPGBT,
        "TRIG": rb.TRIG_LPGBT,
    }

    # Apply these settings on every initialization, even when PUSM is ready.
    lpgbt_settings = {
        "LPGBT.RWF.LINE_DRIVER.LDMODULATIONCURRENT": 0x7F,
        "LPGBT.RWF.LINE_DRIVER.LDEMPHASISENABLE": 0,
        "LPGBT.RWF.LINE_DRIVER.LDEMPHASISAMP": 0,
        "LPGBT.RWF.LINE_DRIVER.LDEMPHASISSHORT": 0,
        "LPGBT.RWF.CHIPCONFIG.HIGHSPEEDDATAOUTINVERT": 1,
        "LPGBT.RW.DEBUG.ULDPBYPASSINTERLEAVER": 0,
        "LPGBT.RW.DEBUG.ULDPBYPASSSCRAMBLER": 0,
        "LPGBT.RW.DEBUG.ULDPBYPASSFECCODER": 0,
    }

    vtrx_settings = {
        "CHxBIAS": 0x30,
        "CHxMOD": 0x20,
        "CHxMODEN": 1,
        "CHxEN": 1,
    }

    # Validate field availability before making changes.
    for chip in chips.values():
        for field in lpgbt_settings:
            chip.rd_reg(field)

    for field in vtrx_settings:
        if field not in rb.VTRX.regs:
            raise RuntimeError(
                f"VTRx+ register table does not support {field}"
            )

    # Prototype-only enables: configure only if supported.
    for field in ("CHxLAEN", "CHxBEN"):
        if field in rb.VTRX.regs:
            vtrx_settings[field] = 1

    def snapshot():
        return {
            "lpGBT": {
                name: {
                    field: chip.rd_reg(field)
                    for field in lpgbt_settings
                }
                for name, chip in chips.items()
            },
            "VTRx+": {
                name: {
                    field: rb.VTRX.rd_reg_ch(field, ch)
                    for field in vtrx_settings
                }
                for ch, name in [(0, "TX1"), (1, "TX2")]
            },
        }

    print("1. Checking lpGBT modes")
    expected_modes = {"DAQ": 0xB, "TRIG": 0x9}

    for name, chip in chips.items():
        mode = chip.rd_reg("LPGBT.RO.LPGBTSETTINGS.LPGBTMODE")
        state = chip.rd_reg("LPGBT.RO.PUSM.PUSMSTATE")
        print(f"{name}: MODE=0x{mode:X}, PUSMSTATE={state}")

        if mode != expected_modes[name]:
            raise RuntimeError(
                f"{name}: expected MODE=0x{expected_modes[name]:X}, "
                f"read 0x{mode:X}; check MODE configuration"
            )

    before = snapshot()

    print("\n2. Configuring both lpGBT transmitters")
    for name, chip in chips.items():
        for field, value in lpgbt_settings.items():
            chip.wr_reg(field, value)
            actual = chip.rd_reg(field)
            if actual != value:
                raise RuntimeError(
                    f"{name}: {field} write/read mismatch: "
                    f"expected {value}, read {actual}"
                )

    print("\n3. Configuring VTRx+ TX1 and TX2")
    print(f"VTRx+ version: {getattr(rb.VTRX, 'ver', 'unknown')}")

    for ch in (0, 1):
        for field, value in vtrx_settings.items():
            rb.VTRX.wr_reg_ch(field, ch, value)
            actual = rb.VTRX.rd_reg_ch(field, ch)
            if actual != value:
                raise RuntimeError(
                    f"TX{ch + 1}: {field} write/read mismatch: "
                    f"expected {value}, read {actual}"
                )

    after = snapshot()

    print("\n4. Configuration comparison: before -> after")
    for section, devices in after.items():
        print(f"\n[{section}]")
        for field in next(iter(devices.values())):
            values = []
            for name, registers in devices.items():
                old = before[section][name][field]
                new = registers[field]
                values.append(f"{name}: 0x{old:02X} -> 0x{new:02X}")
            print(f"{field.split('.')[-1]:30s} {' | '.join(values)}")

    # Allow link acquisition before clearing historical errors.
    print("\n5. Waiting for both uplinks")
    deadline = time.monotonic() + 10
    while True:
        links = read_optical_link_status(rb, verbose=False)
        if all(link["ready"] == 1 for link in links.values()):
            break
        if time.monotonic() >= deadline:
            status(f"Uplink acquisition failed: {links}", ok=False)
            raise RuntimeError("Both uplinks did not become READY")
        time.sleep(0.2)

    # Shared reset clears both uplink FEC counters.
    rb.reset_FEC_error_count(quiet=True)

    print("\n6. Monitoring READY and FEC counters")
    samples = monitor_optical_links(rb, seconds=monitor_seconds)

    passed = all(
        link["ready"] == 1 and link["fec"] == 0
        for sample in samples
        for link in sample.values()
    )

    if not passed:
        status("Optical-link verification failed", ok=False)
        raise RuntimeError("READY dropped or FEC errors were observed")

    status(
        f"Both uplinks passed: READY=1 and FEC=0 "
        f"at all samples during the {monitor_seconds}-second check"
    )

    return {"before": before, "after": after, "samples": samples}


def main():
    """Print the wiring map without accessing hardware."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapping", type=Path)
    args = parser.parse_args()
    mapping = load_mapping(args.mapping)
    print(f"Validated {len(mapping['etrocs'])} ETROC entries")
    for identity, cfg in mapping["etrocs"].items():
        bus = cfg["i2c"]
        print(f"{identity}: I2C={bus['lpgbt']}/M{bus['channel']}/"
              f"0x{bus['address']:02X}, elinks={build_elinks(mapping, identity)}")


class TamaleroWaveformSampler:
    def __init__(self, rb, lpgbt, ws_addr, i2c_master):
        self.rb = rb
        self.lpgbt = lpgbt
        self.kcu = rb.kcu
        self.ws_addr = ws_addr
        self.i2c_master = i2c_master
        self.adr_nbytes = 1

    def i2c_write(self, reg, val):
        self.lpgbt.I2C_write(reg=reg, val=val, master=self.i2c_master,
                             slave_addr=self.ws_addr, adr_nbytes=self.adr_nbytes)

    def i2c_read(self, reg, nbytes=1):
        return self.lpgbt.I2C_read(reg=reg, master=self.i2c_master,
                                   slave_addr=self.ws_addr, nbytes=nbytes, adr_nbytes=self.adr_nbytes)

    def modify_reg_bit(self, reg, bit_mask, val_bool):
        current = self.i2c_read(reg)
        if isinstance(current, list):
            current = current[0]
        if val_bool:
            new_val = current | bit_mask
        else:
            new_val = current & (~bit_mask)
        self.i2c_write(reg, new_val)

    def init_ws(self):
        print(f"  [{hex(self.ws_addr)}] Initializing WS Global Config...")
        self.i2c_write(0x1F, 0x2B)
        self.i2c_write(0x0F, 0x00)
        self.i2c_write(0x0E, 0x00)
        self.i2c_write(0x0D, 0x10)

    def arm_ws(self):
        print(f"  [{hex(self.ws_addr)}] Configing WS Memory Config...")
        self.i2c_write(0x1F, 0x22)
        self.i2c_write(0x1F, 0x0B)

    def trigger_ws_start(self):
        print(f"  [{hex(self.ws_addr)}] Sending WS_START_PULSE via KCU...")
        self.kcu.write_node(f"READOUT_BOARD_{self.rb.rb}.WS_START_PULSE", 0x1)

    def trigger_ws_stop(self):
        print(f"  [{hex(self.ws_addr)}] Sending WS_STOP_PULSE via KCU...")
        self.kcu.write_node(f"READOUT_BOARD_{self.rb.rb}.WS_STOP_PULSE", 0x1)

    def stop_and_close_ws(self):
        self.i2c_write(0x1F, 0x09)

    def read_data(self):
        import pandas as pd
        import numpy as np
        from tqdm import tqdm
        print(f"  [{hex(self.ws_addr)}] Reading 1024 words from WS RAM...")
        self.modify_reg_bit(0x1F, 0b00000100, True)

        max_steps = 1024
        base_data = []
        coeff = 0.085
        time_coeff = 0.390625

        for address in tqdm(range(max_steps), desc="  Reading WS", ncols=80):
            rd_add_MSB = address // 4
            rd_add_LSB = (address % 4) * 64

            self.i2c_write(0x1D, rd_add_MSB)
            self.i2c_write(0x1C, rd_add_LSB)

            tmp_data = None
            for _ in range(3):
                tmp_data = self.i2c_read(0x20, nbytes=2)
                if tmp_data is not None and len(tmp_data) >= 2:
                    break
                time.sleep(0.005)

            if tmp_data is None or len(tmp_data) < 2:
                raise RuntimeError(f"Incomplete WS RAM read at address {address}")

            data_int = (tmp_data[0] >> 2) | (tmp_data[1] << 6)
            pointer = (data_int >> 13) & 1
            Dout_S1 = (data_int >> 7) & 0x3F

            Dout_S2 = (((data_int >> 6) & 1) * 24 +
                       ((data_int >> 5) & 1) * 16 +
                       ((data_int >> 4) & 1) * 10 +
                       ((data_int >> 3) & 1) * 6 +
                       ((data_int >> 2) & 1) * 4 +
                       ((data_int >> 1) & 1) * 2 +
                        (data_int & 1))

            base_data.append({
                "Address": address,
                "Data": data_int,
                "Raw": f"{data_int:014b}",
                "pointer": pointer,
                "Dout_S1": Dout_S1,
                "Dout_S2": Dout_S2,
                "Dout": Dout_S1 - (coeff * Dout_S2),
            })

        df = pd.DataFrame(base_data)
        if len(df) == 0:
            self.modify_reg_bit(0x1F, 0b00000100, False)
            return df

        channels = 8
        steps_per_ch = len(df) // channels

        df['Channel'] = np.repeat(np.arange(1, channels + 1), steps_per_ch)
        df['Step'] = np.tile(np.arange(steps_per_ch), channels)

        pointer_mask = (df['Channel'] == channels) & (df['pointer'] != 0)
        if pointer_mask.any():
            pointer_step = df.loc[pointer_mask, 'Step'].iloc[0]
            df['Step'] = (df['Step'] - (pointer_step + 1)) % steps_per_ch

        df['Time_Index'] = df['Step'] * channels + (channels - df['Channel'])
        df['Time_ns'] = df['Time_Index'] * time_coeff
        df = df.sort_values('Time_Index').reset_index(drop=True)

        self.modify_reg_bit(0x1F, 0b00000100, False)
        return df

def capture_ws_all(samplers, rb, hold_seconds):
    """Arm all, request shared FC START/QInj/STOP; host timing is not deterministic."""
    try:
        for identity, ws in samplers.items():
            print(f"Arming {identity}")
            ws.arm_ws()
        time.sleep(0.1)
        print("Sending shared WS_START")
        rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.WS_START_PULSE", 1)
        print("Sending shared QInj FC")
        rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.QINJ_PULSE", 1)
        time.sleep(hold_seconds)
    finally:
        # Also stop on interrupted/failed arm or START; never read before STOP.
        print("Sending shared WS_STOP")
        rb.kcu.write_node(f"READOUT_BOARD_{rb.rb}.WS_STOP_PULSE", 1)
    time.sleep(0.1)


def run_ws_test(rb, mapping, chips, target_ids, *, samples=1, hold_seconds=0.1,
                output_dir="waveform_data", plots=True):
    import math
    from datetime import datetime

    if not target_ids or len(set(target_ids)) != len(target_ids):
        raise ValueError("WS targets must be nonempty and unique")
    if samples < 1 or not math.isfinite(hold_seconds) or hold_seconds <= 0:
        raise ValueError("WS samples and hold time must be positive")
    samplers = {}
    routes = set()
    # Validate all routes before writing any WS registers.
    for identity in target_ids:
        if identity not in chips:
            raise ValueError(f"{identity}: ETROC must be selected and initialized")
        bus = mapping["etrocs"][identity]["i2c"]
        addr, channel, bank = bus["ws_address"], bus["channel"], bus["lpgbt"]
        if type(addr) is not int or not 0 <= addr <= 127:
            raise ValueError(f"{identity}: invalid WS address")
        if type(channel) is not int or channel not in (0, 1, 2):
            raise ValueError(f"{identity}: invalid I2C master channel")
        if bank not in ("daq", "trig"):
            raise ValueError(f"{identity}: invalid lpGBT")
        if bank == "trig" and not getattr(rb, "trigger", False):
            raise RuntimeError("TRIG lpGBT is not initialized")
        route = (bank, channel, addr)
        if route in routes:
            raise ValueError(f"Duplicate WS route: {route}")
        routes.add(route)
        lpgbt = rb.DAQ_LPGBT if bank == "daq" else rb.TRIG_LPGBT
        samplers[identity] = TamaleroWaveformSampler(rb, lpgbt, addr, channel)
        print(f"WS {identity}: {bank.upper()} / M{channel} / 0x{addr:02X}")
    output = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    print("WS 1. Check I2C and initialize all samplers")
    for identity, ws in samplers.items():
        val = ws.i2c_read(0)
        if isinstance(val, list):
            val = val[0] if val else None
        if val is None:
            raise RuntimeError(f"{identity}: WS probe failed")
        print(f"{identity}: register 0 = 0x{val:02X}")
        ws.init_ws()
    print("WS 2. Configure original notebook test pixel (0, 14)")
    WS_TEST_PIXELS, WS_OFFSET = [(0, 14)], 10
    for chip_name in target_ids:
        etroc = chips[chip_name]
        for row, col in WS_TEST_PIXELS:
            baseline, nw = etroc.auto_threshold_scan(row=row, col=col, broadcast=False, offset=1023, use=False, verbose=False)
            print(f"  {chip_name} ({row},{col}): baseline={baseline}, NW={nw}")

            etroc.wr_reg("workMode", 0, row=row, col=col, broadcast=False)
            etroc.wr_reg("enable_TDC", 1, row=row, col=col, broadcast=False)
            etroc.wr_reg("disDataReadout", 0, row=row, col=col, broadcast=False)
            etroc.wr_reg("disTrigPath", 0, row=row, col=col, broadcast=False)
            etroc.wr_reg("DAC", baseline + WS_OFFSET, row=row, col=col, broadcast=False)
            etroc.wr_reg("QSel", 0x1E, row=row, col=col, broadcast=False)
            etroc.wr_reg("QInjEn", 1, row=row, col=col, broadcast=False)
            etroc.wr_reg("RfSel", 0x00, row=row, col=col, broadcast=False)


    print("WS 3. Shared FC START/QInj/STOP; sequential readout. Host timing is not precise.")
    try:
        for sample_idx in range(samples):
            capture_ws_all(samplers, rb, hold_seconds)
            captured_at = datetime.now().isoformat()
            for identity, ws in samplers.items():
                print(f"Reading {identity}, capture {sample_idx + 1}/{samples}")
                df = ws.read_data()
                if len(df) != 1024:
                    raise RuntimeError(f"{identity}: incomplete capture ({len(df)} words)")
                df["sample_idx"] = sample_idx
                df["timestamp"] = captured_at
                df["etroc_id"] = identity
                bus = mapping["etrocs"][identity]["i2c"]
                for field in ("lpgbt", "channel", "ws_address"):
                    df[field] = bus[field]
                # Filename uses validated selection index; ID is recorded in the CSV.
                stem = output / f"ws_{target_ids.index(identity):02d}_sample_{sample_idx:04d}"
                df.to_csv(stem.with_suffix(".csv"), index=False)
                print(f"Saved {identity}: {stem.with_suffix('.csv')}")
                if plots:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    ax.plot(df["Time_ns"], df["Dout"])
                    ax.set(xlabel="Time [ns]", ylabel="Dout",
                           title=f"{identity}: raw WS capture {sample_idx + 1}")
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(stem.with_suffix(".png"), dpi=150)
                    plt.close(fig)
    finally:
        for identity, ws in samplers.items():
            try:
                ws.stop_and_close_ws()
            except Exception as exc:
                print(f"WARNING: WS cleanup failed for {identity}: {exc}")
    print(f"WS complete. Results: {output.resolve()}")
    return output



def configure_adc_mapping(rb, mapping):
    """Assign legacy-format ADC entries to each lpGBT; no hardware writes.

    Call after ReadoutBoard initialization. read_adcs() still performs ADC
    conversions using the existing driver. Missing limits display N/A.
    This does not change ADC current-source settings or calibration.
    """
    from copy import deepcopy
    import math

    entries = mapping["adc_monitoring"]["adc"]
    grouped = {"daq": {}, "trigger": {}}
    for name, source in entries.items():
        entry = deepcopy(source)
        bank = entry["lpgbt"]
        if bank == "trig":
            bank = "trigger"
        if bank not in grouped:
            raise ValueError(f"{name}: unknown lpGBT {bank}")
        pins = [entry["pin_pos"], entry["pin_neg"]] if entry.get("differential", False) else [entry["pin"]]
        if any(type(pin) is not int or not 0 <= pin < 8 for pin in pins):
            raise ValueError(f"{name}: external ADC pins must be integers 0..7")
        if len(pins) == 2 and pins[0] == pins[1]:
            raise ValueError(f"{name}: H and L must differ")
        if entry.get("current", 0) != 0:
            raise ValueError(f"{name}: this passive monitoring profile requires current=0")
        if not math.isfinite(entry["conv"]) or entry["conv"] <= 0:
            raise ValueError(f"{name}: invalid conversion factor")
        grouped[bank][name] = entry
    if grouped["trigger"] and (not getattr(rb, "trigger", False) or not hasattr(rb, "TRIG_LPGBT")):
        raise RuntimeError("ADC mapping requires an initialized TRIG lpGBT")
    rb.DAQ_LPGBT.adc_mapping = grouped["daq"]
    if hasattr(rb, "TRIG_LPGBT"):
        rb.TRIG_LPGBT.adc_mapping = grouped["trigger"]
    provisional = [name for name, entry in entries.items() if entry.get("conversion_verified") is False]
    if provisional:
        print("WARNING: legacy conversion factors need Pathfinder verification: " + ", ".join(provisional))
    print(f"ADC mapping loaded: DAQ={len(grouped['daq'])}, TRIG={len(grouped['trigger'])}")
    return grouped


if __name__ == "__main__":
    main()
