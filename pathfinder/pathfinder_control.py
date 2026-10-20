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


def _index_base(mapping):
    base = mapping.get("index_base", 1)
    _integer(base, 0, 1, "index_base")
    return base


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
        _integer(daughter, _index_base(mapping), _index_base(mapping) + 6, f"{identity}.daughter")
        _integer(hybrid, _index_base(mapping), _index_base(mapping) + 3, f"{identity}.hybrid")
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
    """Expand selections using the mapping's index_base (default 1) into IDs in the supplied selection order.

    'all' selects the Hybrids actually declared for that module, sorted by
    Hybrid number. This does not enable or disable hardware links.
    """
    _require_dict(selection, "selection")
    result = []
    for module, hybrids in selection.items():
        _integer(module, _index_base(mapping), _index_base(mapping) + 6, "Module ID")
        if isinstance(hybrids, str) and hybrids == "all":
            hybrids = sorted(cfg["hybrid"] for cfg in mapping["etrocs"].values()
                             if cfg["daughter"] == module)
        elif not isinstance(hybrids, (list, tuple)):
            raise ValueError(f"Module {module}: use 'all' or a Hybrid ID list")
        if not hybrids:
            raise ValueError(f"Module {module}: Hybrid selection is empty")
        for hybrid in hybrids:
            _integer(hybrid, _index_base(mapping), _index_base(mapping) + 3, f"Module {module} Hybrid ID")
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


def configure_fc_termination(rb, mapping, chips, ids, *, etroc_factory=None):
    """Apply explicit mapping FC end-termination policy to affected daughters.

    Requires stopped acquisition and all four physical ETROCs accessible.
    Unselected devices use no_init=True; no reset/default configuration is run.
    Only FC_EnTer is written. Returned chips/readout selection is not expanded.
    Reapply after any subsequent reset that restores termination defaults.
    """
    policy = mapping.get("fc_termination")
    if policy is None:
        return {}
    base = _index_base(mapping)
    endpoints = policy.get("terminated_hybrids")
    if not isinstance(endpoints, list) or any(
        type(h) is not int or h not in range(base, base + 4) for h in endpoints
    ) or len(set(endpoints)) != len(endpoints):
        raise ValueError("fc_termination.terminated_hybrids must be a list of unique valid hybrid indices")
    daughters = sorted({mapping["etrocs"][i]["daughter"] for i in ids})
    targets = []
    for daughter in daughters:
        for hybrid in range(base, base + 4):
            identity = f"D{daughter}H{hybrid}"
            if identity not in mapping["etrocs"]:
                raise ValueError(f"FC termination requires mapping for {identity}")
            targets.append(identity)
    if etroc_factory is None:
        from tamalero.ETROC import ETROC
        etroc_factory = ETROC
    devices, report = {}, {}
    # Complete device-access preflight before changing any termination.
    for identity in targets:
        bus = mapping["etrocs"][identity]["i2c"]
        if identity in chips:
            chip = chips[identity]
        else:
            lp = getattr(rb, "DAQ_LPGBT" if bus["lpgbt"] == "daq" else "TRIG_LPGBT", None)
            if lp is None:
                raise RuntimeError(f"{identity}: required lpGBT unavailable")
            chip = etroc_factory(rb, master="lpgbt", i2c_master=lp,
                                 i2c_channel=bus["channel"], i2c_adr=bus["address"],
                                 elinks=build_elinks(mapping, identity), no_init=True,
                                 no_hard_reset_on_init=True, strict=False, verbose=False)
        if not chip.is_connected():
            raise RuntimeError(f"{identity}: FC endpoint configuration requires all four chips accessible")
        before = _link_int(chip.get_Ter("FC"))
        devices[identity] = chip
        report[identity] = {"before": before}
    # Enable each physical endpoint before removing intermediate terminations.
    for daughter in daughters:
        order = endpoints + [h for h in range(base, base + 4) if h not in endpoints]
        for hybrid in order:
            identity = f"D{daughter}H{hybrid}"
            expected = int(hybrid in endpoints)
            devices[identity].set_Ter("FC", bool(expected))
            actual = _link_int(devices[identity].get_Ter("FC"))
            if actual != expected:
                raise RuntimeError(f"{identity}: FC_EnTer expected={expected}, readback={actual}; partial configuration")
            report[identity]["after"] = actual
            print(f"{identity}: FC_EnTer={actual}")
    return report


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
    if mapping.get("fc_termination") is not None:
        try:
            configure_fc_termination(rb, mapping, chips, ids, etroc_factory=etroc_factory)
        except Exception as exc:
            raise InitializationError(
                "ETROC objects were created, but FC termination configuration failed; do not start acquisition",
                "FC termination", chips,
            ) from exc
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


def initialize_pathfinder_eprx(rb):
    """Enable and invert the 14 wired even inputs on each Pathfinder lpGBT.

    Run with acquisition stopped, after base lpGBT initialization. Does not
    change rates, phase, termination, odd inputs, or FPGA readout masks.
    Every write is verified; failure can leave a partially applied setup.
    """
    controllers = [("DAQ", getattr(rb, "DAQ_LPGBT", None), "uplink"),
                   ("TRIG", getattr(rb, "TRIG_LPGBT", None), "trigger")]
    for name, lp, key in controllers:
        if lp is None:
            raise RuntimeError(f"{name} lpGBT is required for Pathfinder EPRX initialization")
    def number(value):
        return int(value.value() if hasattr(value, "value") else value)
    used_elinks = list(range(0, 28, 2))
    report = {}
    for name, lp, key in controllers:
        lp.link_inversions[key] = used_elinks.copy()
        report[name] = {}
        for elink in used_elinks:
            group, channel = divmod(elink, 4)
            fields = {
                "INVERT": f"LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL.EPRX{elink}INVERT",
                "ENABLE": f"LPGBT.RWF.EPORTRX.EPRX{group}{channel}ENABLE",
                "TERM":   f"LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL.EPRX{elink}TERM",
            }
            report[name][elink] = {}
            for field, node in fields.items():
                lp.wr_reg(node, 1)
                actual = number(lp.rd_reg(node))
                if actual != 1:
                    raise RuntimeError(f"{name} E-link {elink}: {field} expected=1, readback={actual}")
                report[name][elink][field] = actual
        print(f"{name}: {len(used_elinks)} wired E-links verified (ENABLE=1, INVERT=1)")
    return report


def initialize_pathfinder_links(rb, monitor_seconds=10):
    """Configure wired EPRX inputs and verify both Pathfinder optical uplinks.

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

    print("\n1.1. Enabling and inverting wired EPRX inputs")
    eprx = initialize_pathfinder_eprx(rb)

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

    return {"before": before, "after": after, "samples": samples, "eprx": eprx}


# Compatibility for notebooks using the previous entry-point name.
initialize_pathfinder_optical_links = initialize_pathfinder_links


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


# E-link configuration and diagnostics. Acquisition must be stopped for writes.
_LINK_RATES = {320: (2, 1), 640: (3, 2)}


def _link_int(value):
    return int(value.value() if hasattr(value, "value") else value)


def set_link_rate(rb, chips, mbps=320, include_trig=None, settle=0.2):
    """Set and verify FPGA width, all receiver groups, and all supplied ETROCs.

    Supports 320/640 Mbps on the validated 10.24 Gbps lpGBT modes 9 and 11.
    include_trig=None automatically includes an existing TRIG object.
    FPGA width is shared: pass every ETROC that will remain enabled.
    Does not alter masks, inversion, merge, trigger width, or run bitslip.
    On failure writes may be partial; readbacks are required before acquisition.
    """
    import time
    if isinstance(mbps, bool) or mbps not in _LINK_RATES:
        raise ValueError("Supported rates are 320 and 640 Mbps")
    if not chips:
        raise ValueError("Pass the initialized ETROCs that will be enabled")
    if include_trig is not None and type(include_trig) is not bool:
        raise ValueError("include_trig must be None, True, or False")
    if not isinstance(settle, (int, float)) or not 0 <= settle <= 60:
        raise ValueError("settle must be between 0 and 60 seconds")
    trig = getattr(rb, "TRIG_LPGBT", None)
    use_trig = trig is not None if include_trig is None else include_trig
    if use_trig and trig is None:
        raise RuntimeError("TRIG lpGBT requested but not initialized")
    controllers = [("DAQ", rb.DAQ_LPGBT)]
    if use_trig:
        controllers.append(("TRIG", trig))
    # Complete preflight reads before the first write.
    for name, lp in controllers:
        mode = _link_int(lp.rd_reg("LPGBT.RO.LPGBTSETTINGS.LPGBTMODE"))
        if mode not in (9, 11):
            raise ValueError(f"{name}: unvalidated optical MODE={mode}; no rate writes made")
    for identity, chip in chips.items():
        for port in ("Left", "Right"):
            _link_int(chip.rd_reg("serRate" + port))
    width, rate = _LINK_RATES[mbps]
    report = {}
    def write_checked(key, value, writer, reader):
        writer(value)
        actual = _link_int(reader())
        report[key] = actual
        if actual != value:
            raise RuntimeError(f"{key}: expected {value}, read {actual}; configuration may be partial")
    node = f"READOUT_BOARD_{rb.rb}.ELINK_WIDTH"
    write_checked("ELINK_WIDTH", width, lambda v: rb.kcu.write_node(node, v),
                  lambda: rb.kcu.read_node(node))
    for name, lp in controllers:
        for group in range(7):
            node = f"LPGBT.RWF.EPORTRX.EPRX{group}DATARATE"
            write_checked(f"{name}.group{group}", rate,
                          lambda v: lp.wr_reg(node, v), lambda: lp.rd_reg(node))
    for identity, chip in chips.items():
        for port in ("Left", "Right"):
            node = "serRate" + port
            write_checked(f"{identity}.{node}", {320: 0, 640: 1}[mbps],
                          lambda v: chip.wr_reg(node, v), lambda: chip.rd_reg(node))
    time.sleep(settle)
    print(f"Rate verified: {mbps} Mbps; receivers={[n for n, _ in controllers]}, ETROCs={list(chips)}")
    return report


_FPGA_RATE = {2: 320, 3: 640, 4: 1280}
_LPGBT_RATE = {1: 320, 2: 640, 3: 1280}
_ETROC_RATE = {0: 320, 1: 640, 2: 1280}
_FPGA_STEP = {320: 1, 640: 2, 1280: 4}      # only elinks with elink % step == 0 are enabled


def check_link_rate(rb, chips, mbps=None, *, mapping=None, include_trig=None, verbose=True):
    """Read-only check that FPGA, lpGBT receivers and ETROCs share one rate.

    mbps=None expects whatever the FPGA ELINK_WIDTH says; otherwise every
    layer must equal mbps (320, 640 or 1280). Checks FPGA ELINK_WIDTH, the 7
    EPRXnDATARATE groups of the DAQ (and an existing TRIG) lpGBT, and
    serRateLeft/Right of each ETROC; with singlePort=1 only Right is used,
    so only Right must match, and a Left difference is reported as a note.
    With mapping, selected ETROC elinks that the FPGA rate disables
    (elink % step != 0) are reported. Writes nothing.
    Returns {"rate", "consistent", "mismatches", "notes", "layers"}.
    """
    def val(x):
        return int(x.value() if hasattr(x, "value") else x)

    layers, mismatches, notes = {}, [], []

    def read(key, reader, table):
        try:
            code = val(reader())
        except Exception as exc:
            layers[key] = {"code": None, "mbps": None, "error": str(exc)}
            mismatches.append(f"{key}: READ FAILED ({exc})")
            return None
        layers[key] = {"code": code, "mbps": table.get(code)}
        return table.get(code)

    fpga = read("FPGA.ELINK_WIDTH",
                lambda: rb.kcu.read_node(f"READOUT_BOARD_{rb.rb}.ELINK_WIDTH"), _FPGA_RATE)
    expected = mbps if mbps is not None else fpga
    if mbps is not None and mbps not in (320, 640, 1280):
        raise ValueError("mbps must be None, 320, 640 or 1280")
    if expected is None:
        mismatches.append("expected rate unknown (FPGA ELINK_WIDTH unreadable or unexpected)")

    lps = [("DAQ", rb.DAQ_LPGBT)]
    use_trig = (getattr(rb, "TRIG_LPGBT", None) is not None) if include_trig is None else include_trig
    if use_trig:
        lps.append(("TRIG", rb.TRIG_LPGBT))
    for name, lp in lps:
        try:
            mode = val(lp.rd_reg("LPGBT.RO.LPGBTSETTINGS.LPGBTMODE"))
        except Exception:
            mode = None
        if mode not in (9, 11):
            notes.append(f"{name}: LPGBTMODE={mode}; rate codes are only validated for modes 9/11")
        for g in range(7):
            read(f"{name}.EPRX{g}DATARATE",
                 lambda lp=lp, g=g: lp.rd_reg(f"LPGBT.RWF.EPORTRX.EPRX{g}DATARATE"), _LPGBT_RATE)

    for ident, chip in chips.items():
        port = None
        try:
            port = val(chip.rd_reg("singlePort"))
        except Exception as exc:
            mismatches.append(f"{ident}.singlePort: READ FAILED ({exc})")
        layers[f"{ident}.singlePort"] = {"code": port}
        for side in ("Left", "Right"):
            r = read(f"{ident}.serRate{side}",
                     lambda chip=chip, side=side: chip.rd_reg("serRate" + side), _ETROC_RATE)
            if side == "Left" and port == 1:
                if r is not None and expected is not None and r != expected:
                    notes.append(f"{ident}.serRateLeft={r} Mbps differs, but singlePort=1 uses Right only")
                layers[f"{ident}.serRateLeft"]["unused"] = True

    for key, info in layers.items():
        if "mbps" not in info or info.get("unused") or info.get("error"):
            continue
        if expected is not None and info["mbps"] != expected:
            mismatches.append(f"{key}: {info['mbps']} Mbps (code {info['code']}), expected {expected}")

    if mapping is not None and expected in _FPGA_STEP:
        step = _FPGA_STEP[expected]
        for ident in chips:
            for bank, links in build_elinks(mapping, ident).items():
                for elink in links:
                    if elink % step:
                        mismatches.append(f"{ident} elink {elink} ({'DAQ' if bank == 0 else 'TRIG'}) "
                                          f"is disabled by the FPGA at {expected} Mbps")

    result = {"rate": expected, "consistent": not mismatches,
              "mismatches": mismatches, "notes": notes, "layers": layers}
    if verbose:
        print(f"Expected rate: {expected} Mbps" + ("" if mbps is not None else " (taken from FPGA)"))
        for key, info in layers.items():
            if "mbps" in info:
                tag = "" if info.get("unused") or info.get("error") or info["mbps"] == expected else "   <-- MISMATCH"
                print(f"  {key:<24} code={info['code']}  {info['mbps']} Mbps{tag}"
                      + ("  (unused: singlePort=1)" if info.get("unused") else ""))
            else:
                print(f"  {key:<24} {info['code']}")
        for n in notes:
            print("  note:", n)
        print("ALL LAYERS AGREE: " + str(expected) + " Mbps" if result["consistent"]
              else "RATE CHECK FAILED:\n  " + "\n  ".join(mismatches))
    return result


def configure_receivers(rb, mapping, ids, *, phase=5, invert=None, verify=True):
    """Enable the lpGBT elink receivers (EPRX) used by the selected ETROCs.

    DAQ receivers are set up by LPGBT.configure_eprx() during power-up, but
    the TRIG lpGBT never runs it (it only loads configs/config_slave_v1.txt),
    so TRIG elinks can be disabled. For every data link of every selected
    ETROC, on the matching lpGBT, this writes: group TRACKMODE=2, the
    channel ENABLE=1, PHASESELECT=phase, ACBIAS=1, TERM=1, and the
    global lock/relock thresholds=5. Equalization is left unchanged. Data rate and polarity are NOT touched
    unless invert={(bank, elink): 0 or 1} is given (bank 0 = DAQ, 1 = TRIG),
    e.g. invert={(0, 14): 1}. Rewriting the same values on DAQ is harmless.
    Writes are verified by readback; a mismatch raises. Run before
    ensure_locked(), with acquisition stopped.
    Returns {(bank, elink): {"identity", "applied": {...}}}.
    """
    validate_mapping(mapping)
    if isinstance(ids, (str, bytes)):
        raise ValueError("ids must be a sequence of ETROC IDs, not a string")
    invert = dict(invert or {})
    for key, value in invert.items():
        if (not isinstance(key, tuple) or len(key) != 2 or key[0] not in (0, 1)
                or value not in (0, 1) or isinstance(value, bool)):
            raise ValueError("invert must look like {(bank, elink): 0 or 1}")
    targets = {}
    for identity in ids:
        if identity not in mapping["etrocs"]:
            raise ValueError(f"Unknown ETROC ID: {identity!r}")
        for bank, elinks in build_elinks(mapping, identity).items():
            for elink in elinks:
                targets[(bank, elink)] = identity
    unknown = set(invert) - set(targets)
    if unknown:
        raise ValueError(f"invert refers to links not used by the selected ETROCs: {sorted(unknown)}")
    lpgbts = {0: rb.DAQ_LPGBT, 1: getattr(rb, "TRIG_LPGBT", None)}
    for bank, _ in targets:
        if lpgbts[bank] is None:
            raise RuntimeError("A selected ETROC uses the TRIG lpGBT, which is not initialized")

    def put(lp, reg, value, label):
        lp.wr_reg(reg, value)
        if verify:
            actual = _link_int(lp.rd_reg(reg))
            if actual != value:
                raise RuntimeError(f"{label}: {reg} wrote {value}, read {actual}")

    report = {}
    prefix = "LPGBT.RWF.EPORTRX."
    for (bank, elink), identity in sorted(targets.items()):
        lp, name = lpgbts[bank], ("DAQ" if bank == 0 else "TRIG")
        group, channel = divmod(elink, 4)
        label = f"{name} elink {elink} ({identity})"
        applied = {}
        put(lp, f"{prefix}EPRX{group}TRACKMODE", 2, label)
        put(lp, f"{prefix}EPRX{group}{channel}ENABLE", 1, label)
        for field, value in (("PHASESELECT", phase), ("ACBIAS", 1), ("TERM", 1)):
            put(lp, f"{prefix}EPRX_CHN_CONTROL.EPRX{elink}{field}", value, label)
        if (bank, elink) in invert:
            put(lp, f"{prefix}EPRX_CHN_CONTROL.EPRX{elink}INVERT", invert[(bank, elink)], label)
            applied["INVERT"] = invert[(bank, elink)]
        report[(bank, elink)] = {"identity": identity, "applied": applied}
    for bank in sorted({b for b, _ in targets}):
        for reg in ("EPRXLOCKTHRESHOLD", "EPRXRELOCKTHRESHOLD"):
            put(lpgbts[bank], prefix + reg, 5, "DAQ" if bank == 0 else "TRIG")
    print("Receivers enabled: " + ", ".join(
        f"{'DAQ' if b == 0 else 'TRIG'}:{e}({i})" for (b, e), i in sorted(targets.items())))
    return report


def ensure_locked(rb, mapping, ids, retries=3, wait=0.5, verbose=True):
    """Check selected links; retry automatic alignment without changing masks.

    Raises before alignment if a selected link is disabled. Each retry enables
    automatic alignment for `wait` seconds, then restores its previous setting.
    Returns check_links() results, even if some channels remain unlocked.
    """
    import time
    if isinstance(ids, (str, bytes)):
        raise ValueError("ids must be a sequence, not a string")
    ids = list(ids)
    if not ids:
        raise ValueError("Select at least one ETROC")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a non-negative integer")
    if not isinstance(wait, (int, float)) or not 0 < wait <= 60:
        raise ValueError("wait must be > 0 and <= 60 seconds")
    build_readout_masks(mapping, ids)  # Validate the selection without writes.
    prefix = f"READOUT_BOARD_{rb.rb}"
    for identity in ids:
        for bank, links in build_elinks(mapping, identity).items():
            if not links:
                continue
            node = "ETROC_DISABLE" if bank == 0 else "ETROC_DISABLE_SLAVE"
            mask = _link_int(rb.kcu.read_node(f"{prefix}.{node}"))
            for elink in links:
                if mask & (1 << elink):
                    raise RuntimeError(f"{identity} bank {bank} E-link {elink} DISABLED; "
                                       "call configure_readout_links(rb, mapping, ids) first")
    result = check_links(rb, mapping, ids)
    def all_locked():
        return bool(result) and all(r["all_locked"] for r in result.values())
    for attempt in range(retries):
        if all_locked():
            break
        if verbose:
            print(f"Alignment attempt {attempt + 1}/{retries}")
        node = f"{prefix}.BITSLIP_AUTO_EN"
        previous = _link_int(rb.kcu.read_node(node))
        try:
            rb.kcu.write_node(node, 1)
            time.sleep(wait)
            result = check_links(rb, mapping, ids)
        finally:
            rb.kcu.write_node(node, previous)
    if verbose:
        print("All selected links locked" if all_locked() else "Some selected links remain unlocked")
    return result


def _phase_target(rb, mapping, chips, identity):
    """Resolve one initialized ETROC's single data link without writes."""
    if identity not in chips:
        raise ValueError(f"{identity}: ETROC has not been initialized")
    links = [(b, e) for b, es in build_elinks(mapping, identity).items() for e in es]
    if len(links) != 1:
        raise ValueError(f"{identity}: phase operations require exactly one mapped data link")
    bank, elink = links[0]
    lp = getattr(rb, "DAQ_LPGBT" if bank == 0 else "TRIG_LPGBT", None)
    if lp is None:
        raise ValueError(f"{identity}: required lpGBT is unavailable")
    return bank, elink, lp


def scan_etroc_phases(rb, mapping, chips, selection, *,
                      phases=None, output_dir="phase_scan_results", html=False):
    """Sequential fixed-phase scans selected by a D/H selection dictionary.

    Requires stopped acquisition. Each scan restores its phase and shared
    auto-bitslip setting. Non-fixed-mode/disabled links are recorded as skipped;
    missing ETROC objects are rejected before writes. Hardware errors stop the
    batch and save completed results. No best phase is automatically applied.
    Five lock samples per tap are a screening test, not a BER measurement.
    """
    import csv
    import json
    from datetime import datetime, timezone
    from uuid import uuid4

    validate_mapping(mapping)
    ids = expand_selection(mapping, selection)
    phases = list(range(15)) if phases is None else list(phases)
    if not phases or any(type(p) is not int or not 0 <= p <= 14 for p in phases):
        raise ValueError("phases must contain integers from 0 to 14")
    if len(set(phases)) != len(phases):
        raise ValueError("Duplicate phases are not allowed")
    targets = {i: _phase_target(rb, mapping, chips, i) for i in ids}
    folder = Path(output_dir).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8]
    json_path, csv_path = folder / f"phase_scan_{stamp}.json", folder / f"phase_scan_{stamp}.csv"
    report = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
              "selection": selection, "identities": ids, "phases": phases,
              "sample_count": 5, "alignment_seconds": 0.5, "sample_interval_seconds": 0.1,
              "status": "running", "results": {},
              "files": {"json": str(json_path), "csv": str(csv_path)}}
    def save():
        # Separate files per run; atomic replacement avoids a half-written JSON.
        tmp = json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(json_path)
        tmp = csv_path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["identity", "bank", "elink", "status", "original_phase",
                             "fpga_width", "eprx_rate", "phase", "locked_samples", "total_samples", "error"])
            for identity, item in report["results"].items():
                for row in item.get("rows") or [{}]:
                    cfg = item.get("configuration", {})
                    writer.writerow([identity, item["bank"], item["elink"], item["status"],
                                     item.get("original_phase", ""), cfg.get("ELINK_WIDTH", ""),
                                     cfg.get("EPRX_DATARATE", ""), row.get("phase", ""),
                                     row.get("locked_samples", ""), len(row.get("samples", [])),
                                     item.get("error", "")])
        tmp.replace(csv_path)
    save()  # Check output is writable before making hardware changes.
    print("Phase scans require stopped acquisition; shared auto-bitslip can realign other channels.")
    print(f"Results: {json_path}")
    try:
        for identity in ids:
            bank, elink, lp = targets[identity]
            item = {"bank": bank, "elink": elink, "status": "reading", "rows": [],
                    "i2c": dict(mapping["etrocs"][identity]["i2c"])}
            report["results"][identity] = item
            prefix = f"READOUT_BOARD_{rb.rb}"
            item["original_phase"] = _link_int(lp.rd_reg(
                f"LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL.EPRX{elink}PHASESELECT"))
            cfg = {}
            item["configuration"] = cfg
            for name in ("ELINK_WIDTH", "TRIG_DATA_SIZE", "BITSLIP_AUTO_EN"):
                cfg[name] = _link_int(rb.kcu.read_node(f"{prefix}.{name}"))
            for suffix in ("DATARATE", "TRACKMODE"):
                cfg["EPRX_" + suffix] = _link_int(lp.rd_reg(f"LPGBT.RWF.EPORTRX.EPRX{elink // 4}{suffix}"))
            for suffix in ("INVERT", "TERM", "ACBIAS", "EQ1"):
                cfg["EPRX_" + suffix] = _link_int(lp.rd_reg(
                    f"LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL.EPRX{elink}{suffix}"))
            for name in ("serRateLeft", "serRateRight", "singlePort", "mergeTriggerData"):
                cfg[name] = _link_int(chips[identity].rd_reg(name))
            cfg["trigger_bits"] = _link_int(chips[identity].get_triggerGranularity())
            mask_name = "ETROC_DISABLE" if bank == 0 else "ETROC_DISABLE_SLAVE"
            disabled = (_link_int(rb.kcu.read_node(f"{prefix}.{mask_name}")) >> elink) & 1
            if cfg["EPRX_TRACKMODE"] != 0 or disabled:
                item.update(status="skipped", error="Requires TRACKMODE=0 and an enabled FPGA receiver")
                print(f"{identity}: SKIPPED ({item['error']})")
            else:
                item["status"] = "scanning"
                save()
                item.update(_scan_etroc_phase(rb, mapping, identity, phases=phases))
                item["status"] = "complete"
            save()
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        if "identity" in locals():
            report["results"][identity].update(status=report["status"], error=f"{type(exc).__name__}: {exc}")
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print(f"Saved JSON: {json_path}\nSaved CSV: {csv_path}")

    table = []
    for phase in phases:
        cells = []
        for identity in ids:
            item = report["results"][identity]
            row = next((r for r in item["rows"] if r["phase"] == phase), None)
            cells.append(f"{row['locked_samples']}/{len(row['samples'])}" if row else item["status"])
        table.append(("Phase scan (lock samples, not BER)", f"phase {phase}", cells, len(set(cells)) > 1))
    if html:
        _cmp_show_html(ids, table, {i: {} for i in ids})
    else:
        _cmp_show_text(ids, table, {i: {} for i in ids})
    return report


def apply_etroc_phases(rb, mapping, chips, phase_settings):
    """Apply explicit {ETROC_ID: phase} choices, verify and align once.

    Requires stopped acquisition and fixed-phase mode. Uses phases 0..14.
    Settings stay applied on success, even when a link fails to lock. On an
    error, attempts to restore changed phases. No mapping file is modified.
    """
    import time
    if not isinstance(phase_settings, dict) or not phase_settings:
        raise ValueError("phase_settings must be a nonempty {ETROC_ID: phase} mapping")
    validate_mapping(mapping)
    targets = {}
    for identity, phase in phase_settings.items():
        if type(phase) is not int or not 0 <= phase <= 14:
            raise ValueError(f"{identity}: phase must be an integer from 0 to 14")
        bank, elink, lp = _phase_target(rb, mapping, chips, identity)
        if _link_int(lp.rd_reg(f"LPGBT.RWF.EPORTRX.EPRX{elink // 4}TRACKMODE")) != 0:
            raise RuntimeError(f"{identity}: requires TRACKMODE=0")
        mask = "ETROC_DISABLE" if bank == 0 else "ETROC_DISABLE_SLAVE"
        if _link_int(rb.kcu.read_node(f"READOUT_BOARD_{rb.rb}.{mask}")) & (1 << elink):
            raise RuntimeError(f"{identity}: FPGA receiver is disabled")
        node = f"LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL.EPRX{elink}PHASESELECT"
        targets[identity] = (bank, elink, lp, node, _link_int(lp.rd_reg(node)))
    auto = f"READOUT_BOARD_{rb.rb}.BITSLIP_AUTO_EN"
    old_auto = _link_int(rb.kcu.read_node(auto))
    changed, report = [], {}
    try:
        for identity, phase in phase_settings.items():
            bank, elink, lp, node, original = targets[identity]
            changed.append(identity)
            lp.wr_reg(node, phase)
            if _link_int(lp.rd_reg(node)) != phase:
                raise RuntimeError(f"{identity}: phase readback mismatch")
            report[identity] = {"bank": bank, "elink": elink, "before": original, "phase": phase}
        rb.kcu.write_node(auto, 1)
        time.sleep(0.5)
        rb.kcu.write_node(auto, 0)
        for identity in phase_settings:
            bank, elink, _, _, _ = targets[identity]
            samples = []
            for _ in range(5):
                time.sleep(0.1)
                samples.append(bool(rb.etroc_locked(elink, slave=(bank == 1))))
            report[identity]["lock_samples"] = samples
            print(f"{identity}: phase={phase_settings[identity]}, locked {sum(samples)}/5")
    except BaseException:
        for identity in reversed(changed):
            _, _, lp, node, original = targets[identity]
            try:
                lp.wr_reg(node, original)
                if _link_int(lp.rd_reg(node)) != original:
                    raise RuntimeError("readback mismatch")
            except Exception as exc:
                print(f"RESTORE FAILED for {identity}: {exc}")
        raise
    finally:
        rb.kcu.write_node(auto, old_auto)
        if _link_int(rb.kcu.read_node(auto)) != old_auto:
            raise RuntimeError("Failed to restore auto-bitslip setting")
    return report


def _scan_etroc_phase(rb, mapping, identity, *, phases=None):
    """Explicit hardware diagnostic: scan one mapped link, then restore settings.

    Requires fixed-phase mode and stopped acquisition. Shared FPGA bitslip
    may realign other channels. Lock samples are not a bit-error-rate test.
    """
    import time
    phases = list(range(15)) if phases is None else list(phases)
    if not phases or any(type(p) is not int or not 0 <= p <= 14 for p in phases):
        raise ValueError("phases must contain integers from 0 to 14")
    if len(set(phases)) != len(phases):
        raise ValueError("Duplicate phases are not allowed")
    links = [(b, e) for b, es in build_elinks(mapping, identity).items() for e in es]
    if len(links) != 1:
        raise ValueError("Phase scan requires exactly one mapped E-link per ETROC")
    bank, elink = links[0]
    lp = getattr(rb, "DAQ_LPGBT" if bank == 0 else "TRIG_LPGBT", None)
    if lp is None:
        raise RuntimeError("Required lpGBT is not initialized")
    prefix = f"READOUT_BOARD_{rb.rb}"
    mask_node = prefix + (".ETROC_DISABLE" if bank == 0 else ".ETROC_DISABLE_SLAVE")
    if _link_int(rb.kcu.read_node(mask_node)) & (1 << elink):
        raise RuntimeError("Selected FPGA E-link is disabled; enable it before scanning")
    mode_node = f"LPGBT.RWF.EPORTRX.EPRX{elink // 4}TRACKMODE"
    if _link_int(lp.rd_reg(mode_node)) != 0:
        raise RuntimeError("Phase scan requires TRACKMODE=0; group mode was not changed")
    phase_node = f"LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL.EPRX{elink}PHASESELECT"
    auto_node = prefix + ".BITSLIP_AUTO_EN"
    old_phase = _link_int(lp.rd_reg(phase_node))
    old_auto = _link_int(rb.kcu.read_node(auto_node))
    rows = []
    print(f"Phase scan: {identity}, {'DAQ' if bank == 0 else 'TRIG'} E{elink}; shared bitslip will run")
    try:
        for phase in phases:
            lp.wr_reg(phase_node, phase)
            if _link_int(lp.rd_reg(phase_node)) != phase:
                raise RuntimeError(f"Phase {phase}: readback mismatch")
            rb.kcu.write_node(auto_node, 1)
            time.sleep(0.5)
            rb.kcu.write_node(auto_node, 0)
            samples = []
            for _ in range(5):
                time.sleep(0.1)
                samples.append(bool(rb.etroc_locked(elink, slave=(bank == 1))))
            rows.append({"phase": phase, "locked_samples": sum(samples), "samples": samples})
            print(f"E{elink} phase={phase:2d}: locked {sum(samples)}/5")
    finally:
        try:
            lp.wr_reg(phase_node, old_phase)
            if _link_int(lp.rd_reg(phase_node)) != old_phase:
                raise RuntimeError("Original phase restoration failed")
        finally:
            rb.kcu.write_node(auto_node, old_auto)
            if _link_int(rb.kcu.read_node(auto_node)) != old_auto:
                raise RuntimeError("Original auto-bitslip restoration failed")
    candidates = [r["phase"] for r in rows if r["locked_samples"] == 5]
    print("Original phase and auto-bitslip setting restored.")
    print("Candidate phases:", candidates)
    return {"identity": identity, "bank": bank, "elink": elink,
            "original_phase": old_phase, "original_auto": old_auto,
            "rows": rows, "candidates": candidates}


def debug_etroc_link(rb, mapping, chips, identity, pll_samples=3, *, interval=1.0, verbose=True, scan_phase=False, phases=None):
    """Read ETROC, lpGBT receiver, and FPGA link status without writes.

    verbose=False reads everything and returns the results dict without
    printing (used by compare_etroc_links).
    """
    import time
    if type(scan_phase) is not bool:
        raise ValueError("scan_phase must be a bool")
    if phases is not None and not scan_phase:
        raise ValueError("phases requires scan_phase=True")
    _p = print if verbose else (lambda *args, **kwargs: None)

    def as_int(value):
        return int(value.value() if hasattr(value, "value") else value)

    if isinstance(pll_samples, bool) or not isinstance(pll_samples, int) or pll_samples < 1:
        raise ValueError("pll_samples must be a positive integer")
    if not isinstance(interval, (int, float)) or not 0 <= interval <= 60:
        raise ValueError("interval must be between 0 and 60 seconds")
    results = {}

    def read(label, reader):
        try:
            value = as_int(reader())
            results[label] = value
            _p(f"  {label}: {value}")
            return value
        except Exception as exc:
            results[label] = {"error": str(exc)}
            _p(f"  {label}: READ FAILED — {exc}")
            return None

    if identity not in mapping["etrocs"]:
        raise ValueError(f"Unknown ETROC: {identity}")
    if identity not in chips:
        raise ValueError(f"{identity} has not been initialized")

    chip = chips[identity]
    routes = build_elinks(mapping, identity)
    prefix = f"READOUT_BOARD_{rb.rb}"

    _p(f"\n=== {identity}: mapping ===")
    _p("  I2C:", mapping["etrocs"][identity]["i2c"])
    _p("  Data links:", routes)

    _p("\n=== ETROC output ===")
    for name in (
        "singlePort",
        "serRateLeft",
        "serRateRight",
        "mergeTriggerData",
        "disScrambler",
        "controllerState", "fcAlignFinalState", "fcAlignStatus",
        "asyLinkReset", "TDCStrobeTest",
    ):
        read(name, lambda name=name: chip.rd_reg(name))

    read("trigger_bits", lambda: chip.get_triggerGranularity())
    _p("\n=== FPGA configuration ===")
    for name in ("ELINK_WIDTH", "BITSLIP_AUTO_EN", "TRIG_DATA_SIZE"):
        read(name, lambda name=name:
             rb.kcu.read_node(f"{prefix}.{name}"))

    for ilpgbt, elinks in routes.items():
        if not elinks:
            continue
        if ilpgbt not in (0, 1):
            raise ValueError(f"Unsupported ilpgbt: {ilpgbt}")

        label = "DAQ" if ilpgbt == 0 else "TRIG"
        lpgbt = getattr(
            rb, "DAQ_LPGBT" if ilpgbt == 0 else "TRIG_LPGBT", None
        )
        if lpgbt is None:
            _p(f"\n{label}: lpGBT object unavailable")
            continue

        _p(f"\n=== {label}: optical uplink ===")
        for name in ("READY", "FEC_ERR_CNT"):
            node = f"{prefix}.LPGBT.UPLINK_{ilpgbt}.{name}"
            read(f"{label}.{name}",
                 lambda node=node: rb.kcu.read_node(node))

        read(f"{label}.LPGBTMODE",
             lambda: lpgbt.rd_reg("LPGBT.RO.LPGBTSETTINGS.LPGBTMODE"))

        # Software initialization marker, not hardware READY.
        read(f"{label}.power_up_done",
             lambda: lpgbt.power_up_done())

        disable_name = (
            "ETROC_DISABLE" if ilpgbt == 0 else "ETROC_DISABLE_SLAVE"
        )
        disable_mask = read(
            f"{label}.{disable_name}",
            lambda: rb.kcu.read_node(f"{prefix}.{disable_name}")
        )

        for elink in elinks:
            group, channel = divmod(elink, 4)
            tag = f"{label}.E{elink}"

            _p(
                f"\n=== {identity}: {label} E-link {elink}, "
                f"Group {group}, Channel {channel} ==="
            )

            nodes = [
                f"LPGBT.RWF.EPORTRX.EPRX{group}{channel}ENABLE",
                f"LPGBT.RWF.EPORTRX.EPRX{group}DATARATE",
                f"LPGBT.RWF.EPORTRX.EPRX{group}TRACKMODE",
            ]

            for suffix in ("PHASESELECT", "INVERT", "ACBIAS", "TERM", "EQ1"):
                nodes.append(
                    "LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL."
                    f"EPRX{elink}{suffix}"
                )

            for node in nodes:
                read(f"{tag}.{node.split('.')[-1]}",
                     lambda node=node: lpgbt.rd_reg(node))

            # Raw status addresses verified for driver ver=1.
            if lpgbt.ver == 1:
                rx = read(
                    f"{tag}.RX_STATUS_RAW",
                    lambda: lpgbt.rd_adr(0x152 + 3 * group)
                )
                dll = read(
                    f"{tag}.DLL_STATUS_RAW",
                    lambda: lpgbt.rd_adr(0x168 + group)
                )
                phase = read(
                    f"{tag}.PHASE_RAW",
                    lambda: lpgbt.rd_adr(
                        0x153 + 3 * group + channel // 2
                    )
                )

                if dll is not None:
                    read(f"{tag}.DLL_LOCKED",
                         lambda: (dll >> 7) & 1)
                if rx is not None:
                    read(f"{tag}.PHASE_TRACKING_LOCKED",
                         lambda: (rx >> (4 + channel)) & 1)
                if phase is not None:
                    read(f"{tag}.CURRENT_PHASE",
                         lambda: (phase >> (4 * (channel % 2))) & 0xF)
            else:
                _p("  DLL/phase status skipped: unsupported register version")

            if disable_mask is not None:
                read(f"{tag}.FPGA_DISABLED",
                     lambda: (disable_mask >> elink) & 1)

            read(
                f"{tag}.FPGA_FRAME_LOCKED",
                lambda: rb.etroc_locked(elink, slave=(ilpgbt == 1))
            )

    _p("\n=== Stability samples (read-only; no counter resets) ===")
    samples = []
    for i in range(pll_samples):
        sample = {}
        for name in ("controllerState", "pllUnlockCount", "invalidFCCount"):
            sample[name] = read(f"sample[{i}].{name}",
                                lambda name=name: chip.rd_reg(name))
        for bank, links in routes.items():
            if not links:
                continue
            label = "DAQ" if bank == 0 else "TRIG"
            for name in ("READY", "FEC_ERR_CNT"):
                key = f"{label}.{name}"
                sample[key] = read(f"sample[{i}].{key}", lambda bank=bank, name=name:
                    rb.kcu.read_node(f"{prefix}.LPGBT.UPLINK_{bank}.{name}"))
            for elink in links:
                key = f"{label}.E{elink}.LOCKED"
                sample[key] = read(f"sample[{i}].{key}", lambda bank=bank, elink=elink:
                                   rb.etroc_locked(elink, slave=(bank == 1)))
        samples.append(sample)
        if i + 1 < pll_samples:
            time.sleep(interval)

    notes = []
    def note(message):
        notes.append(message)
        _p("  " + message)

    _p("\n=== Interpretation ===")
    def decode(table, value):
        return table.get(value) if isinstance(value, int) else None
    fpga_rate = decode({2: 320, 3: 640, 4: 1280}, results.get("ELINK_WIDTH"))
    left = decode({0: 320, 1: 640, 2: 1280}, results.get("serRateLeft"))
    right = decode({0: 320, 1: 640, 2: 1280}, results.get("serRateRight"))
    note(f"Rate readbacks: FPGA={fpga_rate}, ETROC Left={left}, Right={right} Mbps")
    for bank, links in routes.items():
        label = "DAQ" if bank == 0 else "TRIG"
        for elink in links:
            tag = f"{label}.E{elink}"
            code = results.get(f"{tag}.EPRX{elink // 4}DATARATE")
            # Restrict interpretation to the two modes validated on this setup.
            mode = results.get(f"{label}.LPGBTMODE")
            rate = decode({1: 320, 2: 640, 3: 1280}, code) if mode in (9, 11) else None
            note(f"{tag}: lpGBT rate={rate} Mbps (mode={mode}, rate code={code})")
            if results.get("singlePort") == 1 and all(v is not None for v in (fpga_rate, right, rate)):
                note(f"{tag}: right-port rate " + ("MATCH" if fpga_rate == right == rate else "MISMATCH"))
            else:
                note(f"{tag}: rate match UNKNOWN; confirm optical mode and output-port routing")
            disabled = results.get(f"{tag}.FPGA_DISABLED")
            if disabled == 1:
                note(f"{tag}: DISABLED; an unlocked receiver does not diagnose the incoming signal")
            elif disabled == 0:
                note(f"{tag}: enabled; FPGA frame lock={results.get(f'{tag}.FPGA_FRAME_LOCKED')}")
            else:
                note(f"{tag}: enable status UNKNOWN")
            size = results.get("TRIG_DATA_SIZE")
            bits = results.get("trigger_bits")
            if isinstance(size, int) and isinstance(bits, int) and results.get("mergeTriggerData") == 1:
                effective = 0 if (size & 0x20 and elink % 4 == 0) else size & 0x1f
                note(f"{tag}: merge trigger width ETROC={bits}, FPGA={effective}: "
                     + ("MATCH" if bits == effective else "MISMATCH"))
            else:
                note(f"{tag}: merge trigger-width comparison UNKNOWN / not applicable")
    if results.get("BITSLIP_AUTO_EN") == 0:
        note("Automatic alignment is currently off; rerun_bitslip() normally turns it off after its pulse.")
    if results.get("controllerState") != 11:
        note("controllerState differs from driver expectation 11; no state-name interpretation is assumed.")
    if len(samples) > 1:
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            if any(v is None for v in values):
                note(f"{key}: incomplete reads; stability UNKNOWN")
            elif "Count" in key or key.endswith("FEC_ERR_CNT"):
                if any(b < a for a, b in zip(values, values[1:])):
                    note(f"{key}: decreased; possible reset/wrap, delta not inferred ({values})")
                elif values[-1] != values[0]:
                    note(f"{key}: increased by {values[-1] - values[0]} ({values})")
                else:
                    note(f"{key}: unchanged at {values[0]}; saturation or historical errors are not excluded")
            else:
                note(f"{key}: {'stable' if len(set(values)) == 1 else 'CHANGING'} {values}")
    results["samples"] = samples
    results["notes"] = notes
    if scan_phase:
        results["phase_scan"] = _scan_etroc_phase(rb, mapping, identity, phases=phases)
    return results


# ---------------------------------------------------------------------------
# Side-by-side view of debug_etroc_link for several ETROCs
# ---------------------------------------------------------------------------

_CMP_ERPX_NAME = re.compile(r"^EPRX\d+")
_CMP_LINK_KEY = re.compile(r"^(?:DAQ|TRIG)\.E\d+\.(?P<name>.+)$")
_CMP_BANK_KEY = re.compile(r"^(?:DAQ|TRIG)\.(?P<name>READY|FEC_ERR_CNT|LPGBTMODE|power_up_done|"
                           r"ETROC_DISABLE(?:_SLAVE)?)$")
_CMP_HEX_ROWS = {"ETROC_DISABLE", "TRIG_DATA_SIZE", "RX_STATUS_RAW", "DLL_STATUS_RAW", "PHASE_RAW"}
_CMP_SECTIONS = [
    ("ETROC", ["singlePort", "serRateLeft", "serRateRight", "mergeTriggerData", "disScrambler",
               "controllerState", "fcAlignFinalState", "fcAlignStatus", "asyLinkReset",
               "TDCStrobeTest", "trigger_bits"]),
    ("FPGA", ["ELINK_WIDTH", "BITSLIP_AUTO_EN", "TRIG_DATA_SIZE"]),
    ("lpGBT optical uplink", ["lpGBT.READY", "lpGBT.FEC_ERR_CNT", "lpGBT.LPGBTMODE",
                              "lpGBT.power_up_done", "lpGBT.ETROC_DISABLE"]),
    ("E-link receiver (lpGBT EPRX + FPGA)", [
        "link.EPRX_ENABLE", "link.EPRX_DATARATE", "link.EPRX_TRACKMODE", "link.EPRX_PHASESELECT",
        "link.EPRX_INVERT", "link.EPRX_ACBIAS", "link.EPRX_TERM", "link.EPRX_EQ1",
        "link.RX_STATUS_RAW", "link.DLL_STATUS_RAW", "link.PHASE_RAW", "link.DLL_LOCKED",
        "link.PHASE_TRACKING_LOCKED", "link.CURRENT_PHASE", "link.FPGA_DISABLED",
        "link.FPGA_FRAME_LOCKED"]),
]
_CMP_STABILITY = ["controllerState", "pllUnlockCount", "invalidFCCount",
                  "lpGBT.READY", "lpGBT.FEC_ERR_CNT", "link.LOCKED"]


def _cmp_normalize(results, routes):
    """Map one ETROC's debug results onto labels shared by all ETROCs.

    DAQ.E14.EPRX3DATARATE -> link.EPRX_DATARATE; TRIG.READY -> lpGBT.READY.
    The bank and the elink number are shown in the mapping rows instead, so
    an ETROC on the DAQ lpGBT lines up with one on the TRIG lpGBT. A second
    link or bank of the same ETROC gets link2./lpGBT#2. labels.
    """
    banks = [bank for bank, links in sorted(routes.items()) if links]
    links = [(bank, e) for bank, es in sorted(routes.items()) for e in es]
    bank_rank = {("DAQ", "TRIG")[b]: i for i, b in enumerate(banks)}
    link_rank = {f"{('DAQ', 'TRIG')[b]}.E{e}": i for i, (b, e) in enumerate(links)}

    def label(prefix, rank):
        return prefix if rank == 0 else f"{prefix}#{rank + 1}" if prefix == "lpGBT" else f"link{rank + 1}"

    out = {}
    for key, value in results.items():
        if key in ("samples", "notes", "error", "phase_scan") or key.startswith("sample["):
            continue                      # per-sample readings are summarised separately
        m = _CMP_LINK_KEY.match(key)
        if m:
            tag = key[:key.index(".", key.index(".") + 1)]
            name = _CMP_ERPX_NAME.sub("EPRX_", m.group("name"))
            out[f"{label('link', link_rank.get(tag, 0))}.{name}"] = value
            continue
        m = _CMP_BANK_KEY.match(key)
        if m:
            name = m.group("name").replace("ETROC_DISABLE_SLAVE", "ETROC_DISABLE")
            out[f"{label('lpGBT', bank_rank.get(key.split('.')[0], 0))}.{name}"] = value
            continue
        out[key] = value
    return out


def _cmp_cell(label, value):
    if isinstance(value, dict):
        return "ERR"
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return f"0x{value:X}" if label.split(".")[-1] in _CMP_HEX_ROWS else str(value)
    return str(value)


def _cmp_stability(samples, routes, name):
    """first -> last over the samples for one stability row (None if absent)."""
    tags = {"lpGBT.READY": "READY", "lpGBT.FEC_ERR_CNT": "FEC_ERR_CNT", "link.LOCKED": "LOCKED"}
    values = []
    for sample in samples or []:
        if name in tags:
            hits = [v for k, v in sample.items() if k.endswith("." + tags[name])]
            values.append(hits[0] if len(hits) == 1 else (None if not hits else tuple(hits)))
        else:
            values.append(sample.get(name))
    if not values:
        return None
    if any(v is None for v in values):
        return "?"
    if "Count" in name or name.endswith("ERR_CNT"):
        if all(type(v) is int for v in values):
            if len(values) < 2:
                return f"{values[0]} (one sample)"
            if any(b < a for a, b in zip(values, values[1:])):
                return f"{values[0]}->{values[-1]} (reset/wrap?)"
            delta = values[-1] - values[0]
            return f"{values[0]}->{values[-1]} (delta={'+' if delta else ''}{delta})"
    if len(set(values)) == 1:
        return f"{values[0]} (const)"
    return "changing " + ",".join(map(str, values))



def compare_etroc_links(rb, mapping, chips, identities=None, pll_samples=3, *,
                        interval=1.0, html=False, only_differences=False, overview=False, verbose=True):
    """Run debug_etroc_link for several ETROCs and show them side by side.

    One column per ETROC, one row per setting, so equal settings line up and
    a row that differs between ETROCs is marked with '*' (highlighted with
    html=True). Reads only; nothing is written. ETROCs are read one after
    another, each for about (pll_samples - 1) * interval seconds.

    identities: IDs to compare (default: every initialized ETROC, in mapping
    order). only_differences=True hides identical detail rows, but retains key checks
    and flagged read failures/statuses. ! marks a status needing attention;
    ? marks unknown results. * only means values differ, not a fault.
    overview=True adds an Overview section at the top with one verdict per
    ETROC (locked, mask, uplink, rate chain, FC, PLL) and a "problems" line;
    "none" means none of the checks it knows about fired. Off by default: the
    table alone already holds every compared status. A failed ETROC shows ERR
    in its column and the others are still shown.
    Returns a dict {"identities", "rows", "results"} (rows are
    (section, label, [cell per ETROC], differs)) whose repr is a single line,
    so ending a notebook cell with the call does not print the data again.
    """
    if identities is None:
        identities = [i for i in mapping["etrocs"] if i in chips]
    identities = list(identities)
    if not identities:
        raise ValueError("No initialized ETROCs to compare")
    for identity in identities:
        if identity not in mapping["etrocs"]:
            raise ValueError(f"Unknown ETROC: {identity}")

    raw, norm, routes = {}, {}, {}
    for identity in identities:
        routes[identity] = build_elinks(mapping, identity)
        try:
            raw[identity] = debug_etroc_link(rb, mapping, chips, identity, pll_samples,
                                             interval=interval, verbose=False)
            norm[identity] = _cmp_normalize(raw[identity], routes[identity])
        except Exception as exc:
            raw[identity] = {"error": f"{type(exc).__name__}: {exc}"}
            norm[identity] = {}

    def values(label):
        return [norm[i].get(label) if raw[i].get("error") is None else {"error": raw[i]["error"]}
                for i in identities]

    rows = []

    def add(section, label, cells, check_diff=True):
        differs = check_diff and len(set(cells)) > 1
        rows.append((section, label, cells, differs))

    # mapping (always different by nature, never flagged)
    def link_list(identity):
        return [(("DAQ", "TRIG")[b], e) for b, es in sorted(routes[identity].items()) for e in es]
    add("Mapping", "lpGBT", ["+".join(sorted({b for b, _ in link_list(i)})) or "-" for i in identities], False)
    add("Mapping", "I2C", [f"{mapping['etrocs'][i]['i2c']['lpgbt']}/M{mapping['etrocs'][i]['i2c']['channel']}"
                           f"/0x{mapping['etrocs'][i]['i2c']['address']:02X}" for i in identities], False)
    add("Mapping", "E-link", [",".join(str(e) for _, e in link_list(i)) or "-" for i in identities], False)

    shown = set()
    for section, labels in _CMP_SECTIONS:
        for label in labels:
            shown.add(label)
            add(section, label, [_cmp_cell(label, v) for v in values(label)])

    # derived rate / trigger-width verdicts (same rules as the notes in debug_etroc_link)
    dec = {"fpga": {2: 320, 3: 640, 4: 1280}, "etroc": {0: 320, 1: 640, 2: 1280},
           "lpgbt": {1: 320, 2: 640, 3: 1280}}
    def rate(kind, v):
        return dec[kind].get(v) if isinstance(v, int) and not isinstance(v, bool) else None
    fpga_r, etr_r, lp_r, chain, width = [], [], [], [], []
    for identity in identities:
        n = norm[identity]
        f, l, mode = rate("fpga", n.get("ELINK_WIDTH")), n.get("lpGBT.LPGBTMODE"), None
        port = n.get("singlePort")
        e = rate("etroc", n.get("serRateRight"))
        g = rate("lpgbt", n.get("link.EPRX_DATARATE")) if n.get("lpGBT.LPGBTMODE") in (9, 11) else None
        fpga_r.append(str(f) if f else "?"); etr_r.append(str(e) if e else "?"); lp_r.append(str(g) if g else "?")
        if port == 1 and all(v is not None for v in (f, e, g)):
            chain.append("MATCH" if f == e == g else "MISMATCH")
        else:
            chain.append("UNKNOWN")
        size, bits = n.get("TRIG_DATA_SIZE"), n.get("trigger_bits")
        first = link_list(identity)[0][1] if link_list(identity) else None
        if isinstance(size, int) and isinstance(bits, int) and n.get("mergeTriggerData") == 1 and first is not None:
            eff = 0 if (size & 0x20 and first % 4 == 0) else size & 0x1f
            width.append(f"{'MATCH' if bits == eff else 'MISMATCH'} (ETROC {bits}, FPGA {eff})")
        else:
            width.append("n/a")
    for label, cells in (("FPGA", fpga_r), ("ETROC right port", etr_r), ("lpGBT EPRX", lp_r)):
        add("Rate (Mbps)", label, cells)
    add("Rate (Mbps)", "FPGA = ETROC = lpGBT", chain)
    add("Rate (Mbps)", "merged trigger width", width)

    for label in _CMP_STABILITY:
        cells = []
        for identity in identities:
            cell = None if raw[identity].get("error") else _cmp_stability(
                raw[identity].get("samples"), routes[identity], label)
            cells.append("ERR" if raw[identity].get("error") else (cell if cell is not None else "-"))
        add("Stability (first -> last)", label, cells)

    # anything the single-ETROC view read that is not listed above is kept, not dropped
    extra = sorted({k for n in norm.values() for k in n} - shown)
    for label in extra:
        add("Other", label, [_cmp_cell(label, v) for v in values(label)])

    # Summary rows reuse existing reads; no extra hardware access.
    critical = [
        ("Key checks", "rate match", chain, len(set(chain)) > 1),
        ("Key checks", "trigger width match", width, len(set(width)) > 1),
    ]
    for label in ("lpGBT.READY", "lpGBT.FEC_ERR_CNT", "link.EPRX_ENABLE",
                  "link.EPRX_INVERT", "link.EPRX_TERM", "link.FPGA_DISABLED",
                  "link.FPGA_FRAME_LOCKED"):
        cells = [_cmp_cell(label, v) for v in values(label)]
        critical.append(("Key checks", label, cells, len(set(cells)) > 1))
    rows[3:3] = critical

    # optional overview: one verdict per ETROC, shown above the detail rows
    ov = {k: [] for k in ("locked", "readout mask", "lpGBT uplink", "rate chain",
                          "FC invalid count", "PLL", "problems")}
    for k, identity in enumerate(identities):
        n, err = norm[identity], raw[identity].get("error")
        if err:
            for key in ov:
                ov[key].append("ERR")
            ov["problems"][-1] = "READ FAILED"
            continue
        st = {label: _cmp_stability(raw[identity].get("samples"), routes[identity], label)
              for label in _CMP_STABILITY}
        rising = lambda text: isinstance(text, str) and "delta=+" in text
        locked, disabled = n.get("link.FPGA_FRAME_LOCKED"), n.get("link.FPGA_DISABLED")
        ready, fec = n.get("lpGBT.READY"), n.get("lpGBT.FEC_ERR_CNT")
        problems = []
        if disabled == 1:
            problems.append("readout masked")
        if n.get("link.EPRX_ENABLE") == 0:
            problems.append("EPRX disabled")
        if ready is not None and ready != 1:
            problems.append("uplink not ready")
        if rising(st["lpGBT.FEC_ERR_CNT"]):
            problems.append("FEC errors rising")
        if disabled != 1 and locked is not None and not locked:
            problems.append("not locked")
        if chain[k] == "MISMATCH":
            problems.append("rate mismatch")
        if rising(st["invalidFCCount"]):
            problems.append("FC invalid count rising")
        if rising(st["pllUnlockCount"]):
            problems.append("PLL unlocking")
        if n.get("controllerState") not in (None, 11):
            problems.append("controllerState != 11")
        ov["locked"].append("?" if locked is None else ("yes" if locked else "NO"))
        ov["readout mask"].append("?" if disabled is None else ("MASKED" if disabled else "open"))
        ov["lpGBT uplink"].append(f"READY={'?' if ready is None else ready}, FEC={'?' if fec is None else fec}")
        ov["rate chain"].append(f"{chain[k]} ({fpga_r[k]}/{etr_r[k]}/{lp_r[k]})")
        ov["FC invalid count"].append(st["invalidFCCount"] or "-")
        ov["PLL"].append(f"state {n.get('controllerState', '?')}, unlock {st['pllUnlockCount'] or '-'}")
        ov["problems"].append(", ".join(problems) if problems else "none")
    if overview:
        rows[3:3] = [("Overview", label, cells, len(set(cells)) > 1) for label, cells in ov.items()]

    if only_differences:
        rows = [r for r in rows if r[3] or r[0] in ("Mapping", "Overview", "Key checks") or any(_cmp_severity(r[1], c) for c in r[2])]

    if verbose:
        if html:
            _cmp_show_html(identities, rows, raw)
        else:
            _cmp_show_text(identities, rows, raw)
    return _CompareResult(identities=identities, rows=rows, results=raw)


class _CompareResult(dict):
    """Return value of compare_etroc_links: a normal dict with a one-line repr.

    The table is printed by the function itself; without this, a notebook cell
    ending in the call would echo the whole dict a second time.
    """

    def __repr__(self):
        differing = sum(1 for row in self["rows"] if row[3])
        return (f"<compare_etroc_links: {len(self['identities'])} ETROCs, "
                f"{len(self['rows'])} rows, {differing} differ; "
                f"keys: identities, rows, results>")

    __str__ = __repr__


def _cmp_severity(label, cell):
    """Classify explicit status evidence, not arbitrary configuration differences."""
    if cell in ("ERR", "?", "-", "UNKNOWN", "n/a"):
        return "unknown"
    if cell.startswith("MISMATCH"):
        return "attention"
    name = label.split(".")[-1]
    if name in ("READY", "EPRX_ENABLE", "DLL_LOCKED", "FPGA_FRAME_LOCKED", "LOCKED") and cell == "0":
        return "attention"
    if name == "FPGA_DISABLED" and cell == "1":
        return "attention"
    if name == "FEC_ERR_CNT" and cell.isdigit() and int(cell) > 0:
        return "attention"
    if ("Count" in name or name.endswith("ERR_CNT")) and "delta=+" in cell:
        return "attention"
    if name in ("READY", "LOCKED") and (cell.startswith("0 (const)") or cell.startswith("changing")):
        return "attention"
    return ""


def _cmp_show_text(identities, rows, raw):
    rows = [(sec, label, [("! " if _cmp_severity(label, c) == "attention" else "? " if _cmp_severity(label, c) == "unknown" else "") + c for c in cells], diff)
            for sec, label, cells, diff in rows]
    name_w = max(len(r[1]) for r in rows) + 2
    col_w = [max(len(i), *(len(r[2][k]) for r in rows)) + 2 for k, i in enumerate(identities)]
    print("* = values differ (not necessarily a fault); ! = attention; ? = unavailable/unknown.")
    section = None
    for sec, label, cells, differs in rows:
        if sec != section:
            section = sec
            print(f"\n== {sec} ==")
            print(" " * 2 + "setting".ljust(name_w) + "".join(i.ljust(w) for i, w in zip(identities, col_w)))
        print(("* " if differs else "  ") + label.split(".", 1)[-1].ljust(name_w)
              + "".join(c.ljust(w) for c, w in zip(cells, col_w)))
    for identity in identities:
        if raw[identity].get("error"):
            print(f"\n{identity}: READ FAILED: {raw[identity]['error']}")


def _cmp_show_html(identities, rows, raw):
    parts = ['<p>* / amber = differing values; red ! = attention; ? = unknown. Differences alone are not faults.</p>', '<table style="border-collapse:collapse;font-family:monospace;font-size:12px">']
    section = None
    for sec, label, cells, differs in rows:
        if sec != section:
            section = sec
            parts.append(f'<tr><th colspan="{len(identities) + 1}" style="text-align:left;'
                         f'padding-top:10px">{escape(sec)}</th></tr>'
                         '<tr><td></td>' + "".join(f'<td style="font-weight:bold;padding:1px 10px">'
                                                    f'{escape(i)}</td>' for i in identities) + '</tr>')
        bg = ' style="background:rgba(255,193,7,0.28)"' if differs else ""
        parts.append(f"<tr{bg}><td style=\"padding:1px 10px\">{escape(label.split('.', 1)[-1])}</td>"
                     + "".join(
                         '<td style="padding:1px 10px;' +
                         ('color:#c62828;font-weight:bold' if _cmp_severity(label, c) == 'attention' else '') +
                         '">' + escape(('! ' if _cmp_severity(label, c) == 'attention' else '? ' if _cmp_severity(label, c) == 'unknown' else '') + c) + '</td>'
                         for c in cells) + "</tr>")
    parts.append("</table>")
    display(HTML("".join(parts)))
    for identity in identities:
        if raw[identity].get("error"):
            display(HTML(f'<div style="color:#c62828">{escape(identity)}: READ FAILED: '
                         f'{escape(raw[identity]["error"])}</div>'))


# ---------------------------------------------------------------------------
# QInj delay scans (ETROC L1Adelay, FPGA TRIG_DLY_SEL)
# Needs: links locked, thresholds applied, TRIG_ENABLE_MASK != 0, QInj enabled
# on scan_pixels. Hardware is touched only when a scan function is called.
# ---------------------------------------------------------------------------

DEFAULT_L1A_DELAY = 0x01F5


def _qscan_expected_elinks(mapping, chips):
    out = []
    for identity in chips:
        for bank, elinks in build_elinks(mapping, identity).items():
            for e in elinks:
                out.append((bank, e, identity))
    return out


def _qscan_set_delay(chips, scan_pixels, delay, verify=False):
    for identity, etroc in chips.items():
        for row, col in scan_pixels:
            etroc.wr_reg("L1Adelay", delay, row=row, col=col, broadcast=False)
        if verify:
            row, col = scan_pixels[0]
            got = etroc.rd_reg("L1Adelay", row=row, col=col)
            if got != delay:
                raise RuntimeError(
                    f"{identity}: L1Adelay readback {got} != written {delay} "
                    f"(value out of the register range?)")


def _qscan_read_counts(fifo, df, expected):
    """Read everything currently in the FIFO; count events per FPGA E-link
    number (28 * bank + lpGBT elink)."""
    # Event words carry the FPGA E-link number: DAQ lpGBT 0..27, TRIG lpGBT
    # 28..55 (TRIG E4 -> 32, E6 -> 34, E8 -> 36, E10 -> 38). per_elink is keyed
    # by that number.
    per = {28 * bank + e: {"header": 0, "data": 0, "trailer": 0} for bank, e, _ in expected}
    tot = {"header": 0, "data": 0, "trailer": 0}
    events = fifo.pretty_read(df)
    for ev in events or []:
        if not ev or len(ev) < 2 or ev[0] not in tot:
            continue
        kind = ev[0]
        tot[kind] += 1
        try:
            elink = ev[1]["elink"]
        except Exception:
            continue
        if elink in per:
            per[elink][kind] += 1
    return tot, per


def _qscan_prepare(rb, chips, mapping, fifo, df, empty_slot_bcid):
    """Common checks + setup for both scans. Returns (fifo, df, expected)."""
    expected = _qscan_expected_elinks(mapping, chips)

    # Refuse to scan on links that are not locked (that cost us hours before).
    bad = [(i, e) for b, e, i in expected if not rb.etroc_locked(e, slave=(b == 1))]
    if bad:
        raise RuntimeError(f"not locked: {bad}. Open the mask with "
                           "configure_readout_links(rb, mapping, ids) and run "
                           "ensure_locked() first.")

    mask = rb.kcu.read_node(f"READOUT_BOARD_{rb.rb}.TRIG_ENABLE_MASK").value()
    if mask == 0:
        raise RuntimeError("TRIG_ENABLE_MASK is 0: run configure_self_trigger() "
                           "(section 9) first, otherwise no L1A is ever generated.")

    if fifo is None:
        from tamalero.FIFO import FIFO
        fifo = FIFO(rb)
    if df is None:
        from tamalero.DataFrame import DataFrame
        df = DataFrame()

    # Same preparation as run_charge_injection_test(), minus the call that
    # un-masks ALL 28 links (rb.enable_etroc_readout()): the mask you set in
    # section 7 is kept.
    fifo.reset()
    rb.reset_data_error_count()
    rb.rerun_bitslip()
    time.sleep(0.5)
    fifo.use_etroc_data()
    rb.enable_etroc_trigger()
    time.sleep(0.1)
    for etroc in chips.values():
        etroc.wr_reg("emptySlotBCID", empty_slot_bcid)
    return fifo, df, expected


def scan_l1a_delay(rb, chips, mapping, scan_pixels, delays=range(480, 512),
                   qinj_count=20, settle=0.05, empty_slot_bcid=500,
                   restore=DEFAULT_L1A_DELAY, fifo=None, df=None, verbose=True):
    if not chips:
        raise ValueError("chips is empty")
    scan_pixels = list(scan_pixels)
    if not scan_pixels:
        raise ValueError("scan_pixels is empty")
    delays = list(delays)

    fifo, df, expected = _qscan_prepare(rb, chips, mapping, fifo, df, empty_slot_bcid)

    results = {}
    try:
        _qscan_set_delay(chips, scan_pixels, delays[0], verify=True)   # range sanity check
        for n, d in enumerate(delays):
            _qscan_set_delay(chips, scan_pixels, d)
            fifo.reset()
            time.sleep(0.01)
            fifo.send_Qinj_only(count=qinj_count)
            time.sleep(settle)
            tot, per = _qscan_read_counts(fifo, df, expected)
            results[d] = {"total": tot, "per_elink": per}
            if verbose:
                bar = "#" * min(tot["data"], 60)
                print(f"L1Adelay {d:>4} (0x{d:03x}): headers={tot['header']:<4} "
                      f"data={tot['data']:<4} trailers={tot['trailer']:<4} {bar}")
    finally:
        # always put the pixels back, even if the scan is interrupted
        try:
            _qscan_set_delay(chips, scan_pixels, restore)
        except Exception as exc:                       # pragma: no cover
            print(f"WARNING: could not restore L1Adelay={restore}: {exc}")

    hits = {d: r["total"]["data"] for d, r in results.items()}
    best_hits = max(hits.values()) if hits else 0
    window = [d for d, h in hits.items() if best_hits and h >= 0.5 * best_hits]
    summary = {
        "results": results,
        "best_hits": best_hits,
        "window_half_max": (min(window), max(window)) if window else None,
        "centre": (min(window) + max(window)) / 2 if window else None,
        "restored_to": restore,
    }
    if verbose:
        if not window:
            print("\nNo data words at any delay. Check: QInjEn/QSel on the "
                  "pixels, thresholds, TRIG_ENABLE_MASK, FC_status(), or widen "
                  "the delay range.")
        else:
            lo, hi = summary["window_half_max"]
            print(f"\nPeak: {best_hits} data words; half-max window "
                  f"L1Adelay {lo}..{hi}, centre ~{summary['centre']:.1f}. "
                  f"Pixels restored to {restore}.")
    return summary


def scan_trigger_delay(rb, chips, mapping, scan_pixels, delays=range(440, 500),
                       l1a_delay=DEFAULT_L1A_DELAY, qinj_count=20, settle=0.05,
                       empty_slot_bcid=500, restore=None, fifo=None, df=None,
                       verbose=True):
    """Scan the FPGA self-trigger delay (READOUT_BOARD_n.TRIG_DLY_SEL) with
    the ETROC L1Adelay held fixed (default 501).

    TRIG_DLY_SEL selects a tap of a 512-stage, 40 MHz delay line (valid
    0..511, 25 ns per step) between the self-trigger and the L1A sent to the
    ETROC. The ETROC answers only if that L1A arrives L1Adelay bunch crossings
    after the hit, so data words appear in a narrow window of TRIG_DLY_SEL.
    Your current default is 466; the window should be close to
    (L1Adelay - fixed round-trip latency), i.e. near 466 if the latency is 35.

    TRIG_DLY_SEL is restored afterwards (to its value before the scan unless
    `restore` is given). L1Adelay stays at `l1a_delay` on the scanned pixels.
    """
    scan_pixels = list(scan_pixels)
    if not chips or not scan_pixels:
        raise ValueError("chips and scan_pixels must not be empty")
    delays = list(delays)
    if not delays or min(delays) < 0 or max(delays) > 511:
        raise ValueError("TRIG_DLY_SEL values must be within 0..511")

    fifo, df, expected = _qscan_prepare(rb, chips, mapping, fifo, df, empty_slot_bcid)

    node = f"READOUT_BOARD_{rb.rb}.TRIG_DLY_SEL"
    original = rb.kcu.read_node(node).value()
    if restore is None:
        restore = original
    _qscan_set_delay(chips, scan_pixels, l1a_delay, verify=True)
    if verbose:
        print(f"L1Adelay fixed at {l1a_delay}; TRIG_DLY_SEL was {original}; "
              f"scanning {delays[0]}..{delays[-1]} ({len(delays)} points)")

    results = {}
    try:
        for d in delays:
            rb.kcu.write_node(node, d)
            fifo.reset()
            time.sleep(0.01)
            fifo.send_Qinj_only(count=qinj_count)
            time.sleep(settle)
            tot, per = _qscan_read_counts(fifo, df, expected)
            results[d] = {"total": tot, "per_elink": per}
            if verbose:
                bar = "#" * min(tot["data"], 60)
                print(f"TRIG_DLY_SEL {d:>4}: headers={tot['header']:<4} "
                      f"data={tot['data']:<4} trailers={tot['trailer']:<4} {bar}")
    finally:
        try:
            rb.kcu.write_node(node, restore)
        except Exception as exc:                       # pragma: no cover
            print(f"WARNING: could not restore TRIG_DLY_SEL={restore}: {exc}")

    hits = {d: r["total"]["data"] for d, r in results.items()}
    best_hits = max(hits.values()) if hits else 0
    window = [d for d, h in hits.items() if best_hits and h >= 0.5 * best_hits]
    summary = {
        "results": results,
        "best_hits": best_hits,
        "window_half_max": (min(window), max(window)) if window else None,
        "centre": (min(window) + max(window)) / 2 if window else None,
        "l1a_delay": l1a_delay,
        "restored_to": restore,
    }
    if verbose:
        if not window:
            print("\nNo data words at any TRIG_DLY_SEL in this range. Widen the "
                  "range (0..511 is valid), and check QInjEn/QSel, thresholds, "
                  "TRIG_ENABLE_MASK and FC_status().")
        else:
            lo, hi = summary["window_half_max"]
            print(f"\nPeak: {best_hits} data words; half-max window "
                  f"TRIG_DLY_SEL {lo}..{hi}, centre ~{summary['centre']:.1f}. "
                  f"TRIG_DLY_SEL restored to {restore}.")
    return summary


if __name__ == "__main__":
    main()
