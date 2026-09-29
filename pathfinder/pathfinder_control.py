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


if __name__ == "__main__":
    main()
