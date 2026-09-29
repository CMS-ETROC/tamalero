"""Offline tests. Run: python -m unittest -v test_pathfinder_control.py"""

import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from pathfinder_control import (
    build_elinks, edin_to_elink, expand_selection, load_mapping, validate_mapping,
    initialize_etrocs, InitializationError,
    check_connections, check_links,
    build_readout_masks, configure_readout_links,
)


class MappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = load_mapping(Path(__file__).with_name("pathfinder_mapping.yaml"))

    def test_all_modules(self):
        ids = expand_selection(self.mapping, {m: "all" for m in range(1, 8)})
        self.assertEqual(len(ids), 28)
        self.assertEqual(len(set(ids)), 28)

    def test_partial_selection(self):
        self.assertEqual(expand_selection(self.mapping, {1: "all", 7: [1]}),
                         ["D1H1", "D1H2", "D1H3", "D1H4", "D7H1"])

    def test_known_links(self):
        self.assertEqual(build_elinks(self.mapping, "D1H1"), {0: [22], 1: []})
        self.assertEqual(build_elinks(self.mapping, "D7H1"), {0: [], 1: [2]})
        self.assertEqual(edin_to_elink("EDIN62"), 26)

    def test_bank_coverage(self):
        banks = {0: [], 1: []}
        for identity in self.mapping["etrocs"]:
            for bank, links in build_elinks(self.mapping, identity).items():
                banks[bank].extend(links)
        for links in banks.values():
            self.assertEqual(sorted(links), list(range(0, 28, 2)))

    def test_invalid_selections(self):
        for selection in ({}, {8: "all"}, {1: [5]}, {1: [1, 1]},
                          {1: "ALL"}, {1: []}, {True: "all"}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                expand_selection(self.mapping, selection)

    def test_i2c_conflict(self):
        mapping = copy.deepcopy(self.mapping)
        mapping["etrocs"]["D1H2"]["i2c"] = mapping["etrocs"]["D1H1"]["i2c"].copy()
        with self.assertRaisesRegex(ValueError, "I2C conflict"):
            validate_mapping(mapping)

    def test_upstream_conflict(self):
        mapping = copy.deepcopy(self.mapping)
        route = mapping["lpgbts"]["trig"]["control"]
        mapping["etrocs"]["D1H1"]["i2c"] = {
            "lpgbt": "daq", "channel": route["channel"], "address": route["address"]}
        with self.assertRaisesRegex(ValueError, "I2C conflict"):
            validate_mapping(mapping)

    def test_link_conflict(self):
        mapping = copy.deepcopy(self.mapping)
        mapping["etrocs"]["D1H2"]["data"] = copy.deepcopy(mapping["etrocs"]["D1H1"]["data"])
        with self.assertRaisesRegex(ValueError, "Data link conflict"):
            validate_mapping(mapping)

    def test_duplicate_yaml_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.yaml"
            path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate YAML key"):
                load_mapping(path)


class InitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = load_mapping(Path(__file__).with_name("pathfinder_mapping.yaml"))

    def setUp(self):
        self.rb = SimpleNamespace(DAQ_LPGBT=object(), TRIG_LPGBT=object(), trigger=True)
        self.factory = Mock(side_effect=lambda *a, **kw: SimpleNamespace(
            is_connected=lambda: True, settings=kw))

    def test_independent_control_and_data_routes(self):
        chips = initialize_etrocs(self.rb, self.mapping, ["D7H1", "D2H1"],
                                  etroc_factory=self.factory, strict=True)
        self.assertEqual(list(chips), ["D7H1", "D2H1"])
        first, second = self.factory.call_args_list
        self.assertIs(first.args[0], self.rb)
        self.assertIs(first.kwargs["i2c_master"], self.rb.DAQ_LPGBT)
        self.assertEqual(first.kwargs["i2c_channel"], 2)
        self.assertEqual(first.kwargs["i2c_adr"], 0x64)
        self.assertEqual(first.kwargs["elinks"], {0: [], 1: [2]})
        self.assertTrue(first.kwargs["strict"])
        self.assertIs(second.kwargs["i2c_master"], self.rb.TRIG_LPGBT)
        self.assertEqual(second.kwargs["i2c_channel"], 2)
        self.assertEqual(second.kwargs["i2c_adr"], 0x60)
        self.assertEqual(second.kwargs["elinks"], {0: [], 1: [18]})

    def test_invalid_ids_before_construction(self):
        for ids in ([], "D1H1", ["D1H1", "D8H1"], ["D1H1", "D1H1"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                initialize_etrocs(self.rb, self.mapping, ids, etroc_factory=self.factory)
        self.factory.assert_not_called()

    def test_missing_trig_before_construction(self):
        del self.rb.TRIG_LPGBT
        with self.assertRaisesRegex(RuntimeError, "TRIG_LPGBT"):
            initialize_etrocs(self.rb, self.mapping, ["D1H1", "D2H1"],
                              etroc_factory=self.factory)
        self.factory.assert_not_called()

    def test_disabled_trig_before_construction(self):
        self.rb.trigger = False
        with self.assertRaisesRegex(RuntimeError, "not enabled"):
            initialize_etrocs(self.rb, self.mapping, ["D2H1"], etroc_factory=self.factory)
        self.factory.assert_not_called()

    def test_daq_only_does_not_require_trig(self):
        del self.rb.TRIG_LPGBT
        self.rb.trigger = False
        chips = initialize_etrocs(self.rb, self.mapping, ["D1H1"],
                                  etroc_factory=self.factory)
        self.assertEqual(list(chips), ["D1H1"])

    def test_connection_failure_preserves_completed_objects(self):
        good = SimpleNamespace(is_connected=lambda: True)
        bad = SimpleNamespace(is_connected=lambda: False)
        self.factory.side_effect = [good, bad]
        with self.assertRaises(InitializationError) as context:
            initialize_etrocs(self.rb, self.mapping, ["D1H1", "D1H2", "D1H3"],
                              etroc_factory=self.factory)
        self.assertEqual(context.exception.failed_id, "D1H2")
        self.assertIs(context.exception.initialized_chips["D1H1"], good)
        self.assertEqual(self.factory.call_count, 2)

    def test_constructor_failure_preserves_cause(self):
        cause = TimeoutError("Simulated timeout")
        self.factory.side_effect = cause
        with self.assertRaises(InitializationError) as context:
            initialize_etrocs(self.rb, self.mapping, ["D1H1"], etroc_factory=self.factory)
        self.assertIs(context.exception.__cause__, cause)
        self.assertEqual(context.exception.initialized_chips, {})


class StatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = load_mapping(Path(__file__).with_name("pathfinder_mapping.yaml"))

    def test_connections_report_each_device(self):
        good = SimpleNamespace(is_connected=Mock(return_value=True))
        silent = SimpleNamespace(is_connected=Mock(return_value=False))
        broken = SimpleNamespace(is_connected=Mock(side_effect=TimeoutError("Read timed out")))
        results = check_connections({"D1H1": good, "D1H2": silent,
                                     "D1H3": broken, "D1H4": None})
        self.assertEqual(results["D1H1"], {"connected": True, "error": None})
        self.assertFalse(results["D1H2"]["connected"])
        self.assertIn("did not respond", results["D1H2"]["error"])
        self.assertIn("TimeoutError", results["D1H3"]["error"])
        self.assertIn("missing", results["D1H4"]["error"])
        for chip in (good, silent, broken):
            chip.is_connected.assert_called_once_with()

    def test_links_use_data_bank_not_i2c_controller(self):
        rb = SimpleNamespace(etroc_locked=Mock(side_effect=[True, False]))
        results = check_links(rb, self.mapping, ["D1H1", "D7H1"])
        self.assertEqual(rb.etroc_locked.call_args_list[0].args, (22,))
        self.assertEqual(rb.etroc_locked.call_args_list[0].kwargs, {"slave": False})
        self.assertEqual(rb.etroc_locked.call_args_list[1].args, (2,))
        self.assertEqual(rb.etroc_locked.call_args_list[1].kwargs, {"slave": True})
        self.assertTrue(results["D1H1"]["all_locked"])
        self.assertFalse(results["D7H1"]["all_locked"])
        self.assertEqual(results["D7H1"]["links"][0]["lpgbt"], "trig")
        self.assertIsNone(results["D7H1"]["links"][0]["error"])

    def test_link_read_failure_does_not_stop_remaining_checks(self):
        rb = SimpleNamespace(etroc_locked=Mock(side_effect=[TimeoutError("Read failed"), True]))
        results = check_links(rb, self.mapping, ["D1H1", "D7H1"])
        self.assertFalse(results["D1H1"]["all_locked"])
        self.assertIn("TimeoutError", results["D1H1"]["links"][0]["error"])
        self.assertTrue(results["D7H1"]["all_locked"])
        self.assertEqual(rb.etroc_locked.call_count, 2)

    def test_multiple_links_require_all_locked(self):
        mapping = copy.deepcopy(self.mapping)
        mapping["etrocs"]["D1H1"]["data"].append({"lpgbt": "trig", "pin": "EDIN01"})
        rb = SimpleNamespace(etroc_locked=Mock(side_effect=[True, False]))
        result = check_links(rb, mapping, ["D1H1"])["D1H1"]
        self.assertEqual(len(result["links"]), 2)
        self.assertFalse(result["all_locked"])

    def test_invalid_link_selection_before_reads(self):
        rb = SimpleNamespace(etroc_locked=Mock())
        for ids in ([], "D1H1", ["D1H1", "D8H1"], ["D1H1", "D1H1"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                check_links(rb, self.mapping, ids)
        rb.etroc_locked.assert_not_called()


class ReadoutMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = load_mapping(Path(__file__).with_name("pathfinder_mapping.yaml"))

    def make_board(self):
        registers = {}
        kcu = SimpleNamespace(
            write_node=Mock(side_effect=lambda node, value: registers.__setitem__(node, value)),
            read_node=Mock(side_effect=lambda node: SimpleNamespace(value=lambda: registers[node])),
        )
        return SimpleNamespace(rb=0, kcu=kcu), registers

    def test_partial_selection_uses_data_routes(self):
        masks = build_readout_masks(self.mapping, ["D1H1", "D7H1"])
        self.assertEqual(masks, {0: 0x0FFFFFFF & ~(1 << 22),
                                 1: 0x0FFFFFFF & ~(1 << 2)})

    def test_all_selection_keeps_unmapped_links_disabled(self):
        ids = expand_selection(self.mapping, {m: "all" for m in range(1, 8)})
        self.assertEqual(build_readout_masks(self.mapping, ids),
                         {0: 0x0AAAAAAA, 1: 0x0AAAAAAA})

    def test_empty_selection_disables_both_banks(self):
        rb, registers = self.make_board()
        configure_readout_links(rb, self.mapping, [])
        self.assertEqual(registers, {"READOUT_BOARD_0.ETROC_DISABLE": 0x0FFFFFFF,
                                    "READOUT_BOARD_0.ETROC_DISABLE_SLAVE": 0x0FFFFFFF})

    def test_repeated_configuration_is_idempotent(self):
        rb, registers = self.make_board()
        first = configure_readout_links(rb, self.mapping, ["D1H1", "D7H1"])
        snapshot = dict(registers)
        second = configure_readout_links(rb, self.mapping, ["D1H1", "D7H1"])
        self.assertEqual(first, second)
        self.assertEqual(registers, snapshot)
        self.assertEqual(rb.kcu.write_node.call_count, 4)

    def test_invalid_selection_does_not_write(self):
        rb, _ = self.make_board()
        for ids in (["D1H1", "D8H1"], ["D1H1", "D1H1"], "D1H1"):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                configure_readout_links(rb, self.mapping, ids)
        rb.kcu.write_node.assert_not_called()

    def test_readback_mismatch_reports_partial_update(self):
        rb, _ = self.make_board()
        rb.kcu.read_node.side_effect = lambda node: SimpleNamespace(value=lambda: 0)
        with self.assertRaisesRegex(RuntimeError, "Readback mismatch.*partially updated"):
            configure_readout_links(rb, self.mapping, ["D1H1"])
        self.assertEqual(rb.kcu.write_node.call_count, 1)

    def test_second_bank_write_failure_is_reported(self):
        rb, registers = self.make_board()
        def write(node, value):
            if node.endswith("_SLAVE"):
                raise TimeoutError("Simulated write timeout")
            registers[node] = value
        rb.kcu.write_node.side_effect = write
        with self.assertRaisesRegex(RuntimeError, "ETROC_DISABLE_SLAVE.*partially updated"):
            configure_readout_links(rb, self.mapping, ["D1H1"])
        self.assertEqual(len(registers), 1)


class EliminatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = load_mapping(Path(__file__).with_name("eliminator_mapping.yaml"))

    def test_all_uses_actual_hybrids(self):
        self.assertEqual(expand_selection(self.mapping, {2: "all"}), ["D2H1"])
        self.assertEqual(expand_selection(self.mapping, {m: "all" for m in range(1, 5)}),
                         ["D1H1", "D2H1", "D3H1", "D4H1"])

    def test_missing_hybrid_and_module_rejected(self):
        for selection in ({1: [2]}, {5: "all"}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                expand_selection(self.mapping, selection)

    def test_wiring_matches_legacy_system(self):
        for d, connector, addr, elink in ((1, "J401", 0x60, 0), (2, "J404", 0x61, 4),
                                         (3, "J402", 0x62, 8), (4, "J403", 0x63, 12)):
            cfg = self.mapping["etrocs"][f"D{d}H1"]
            self.assertEqual(cfg["connector"], connector)
            self.assertEqual(cfg["i2c"], {"lpgbt": "daq", "channel": 1, "address": addr})
            self.assertEqual(build_elinks(self.mapping, f"D{d}H1"), {0: [elink], 1: []})

    def test_absent_trig_references_rejected(self):
        for field in ("i2c", "data", "clock"):
            mapping = copy.deepcopy(self.mapping)
            entry = mapping["etrocs"]["D1H1"][field]
            if field == "data":
                entry = entry[0]
            entry["lpgbt"] = "trig"
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_mapping(mapping)

    def test_no_placeholder_address_reserved(self):
        mapping = copy.deepcopy(self.mapping)
        mapping["etrocs"]["D1H1"]["i2c"].update(channel=0, address=0x72)
        validate_mapping(mapping)
        self.assertNotIn("trig", mapping["lpgbts"])

    def test_daq_only_mask_never_accesses_slave(self):
        registers = {}
        def write(node, value):
            self.assertNotIn("SLAVE", node)
            registers[node] = value
        def read(node):
            self.assertNotIn("SLAVE", node)
            return SimpleNamespace(value=lambda: registers[node])
        rb = SimpleNamespace(rb=0, kcu=SimpleNamespace(write_node=write, read_node=read))
        ids = list(self.mapping["etrocs"])
        expected = 0x0FFFFFFF & ~sum(1 << e for e in (0, 4, 8, 12))
        self.assertEqual(configure_readout_links(rb, self.mapping, ids), {0: expected})
        self.assertEqual(configure_readout_links(rb, self.mapping, []), {0: 0x0FFFFFFF})

    def test_initialize_all_without_trig(self):
        rb = SimpleNamespace(DAQ_LPGBT=object(), trigger=False)
        factory = Mock(side_effect=lambda *a, **kw: SimpleNamespace(is_connected=lambda: True))
        chips = initialize_etrocs(rb, self.mapping, list(self.mapping["etrocs"]), etroc_factory=factory)
        self.assertEqual(len(chips), 4)
        for call in factory.call_args_list:
            self.assertIs(call.kwargs["i2c_master"], rb.DAQ_LPGBT)
            self.assertEqual(call.kwargs["i2c_channel"], 1)


if __name__ == "__main__":
    unittest.main()
