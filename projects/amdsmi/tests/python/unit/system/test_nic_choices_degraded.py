#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""One NIC whose summary getter fails must not take down parser construction.

``AMDSMIParser`` builds the NIC choices for every subcommand, so a library error from a single
NIC used to break ``amd-smi version`` and GPU-only commands as well.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

from common.common import cli_search_order, fake_module, find_cli_dir, load_cli_module, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(str(Path(__file__).parent)))
_CLI_FILES = {
    name: str(Path(_CLI_DIR) / f"{name}.py") if _CLI_DIR else None
    for name in ("amdsmi_helpers", "amdsmi_parser")
}


_GOOD_BDF = "0000:23:00.0"
_GOOD_UUID = "uuid-1"


class TestNicChoicesDegraded(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from amdsmi import amdsmi_exception, amdsmi_interface

        stub_modules(
            cls,
            {
                "_version": fake_module("_version", __version__="0.0.0+test"),
                "amdsmi_init": fake_module(
                    "amdsmi_init",
                    AMDSMI_INIT_FLAG=0,
                    AMDSMI_INITIALIZED=True,
                    amdsmi_interface=amdsmi_interface,
                    amdsmi_exception=amdsmi_exception,
                ),
                "amdsmi_helpers": None,
            },
        )
        cls.interface = amdsmi_interface
        # The parser imports ``amdsmi_helpers`` by bare name, so register the loaded module.
        cls.helpers_module = load_cli_module(
            "amdsmi_helpers", _CLI_FILES["amdsmi_helpers"], _CLI_DIR
        )
        sys.modules["amdsmi_helpers"] = cls.helpers_module
        cls.parser_module = load_cli_module("amdsmi_parser", _CLI_FILES["amdsmi_parser"], _CLI_DIR)

    @classmethod
    def _ainic_info(cls, handle):
        if handle == "bad-handle":
            raise cls.interface.AmdSmiLibraryException(
                cls.interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
            )
        return {"bdf": _GOOD_BDF, "UUID": _GOOD_UUID}

    def setUp(self):
        self.helpers = self.helpers_module.AMDSMIHelpers()
        for target, value in (
            ("get_ainic_handles", ["bad-handle", "good-handle"]),
            ("is_ainic_initialized", True),
        ):
            patcher = mock.patch.object(
                self.helpers_module.AMDSMIHelpers, target, autospec=True, return_value=value
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(
            self.interface, "amdsmi_get_ainic_info", autospec=True, side_effect=self._ainic_info
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_failing_nic_is_listed_with_unavailable_identity(self):
        with self.assertLogs(level="DEBUG"):
            choices, choices_str = self.helpers.get_nic_choices()

        self.assertEqual(set(choices), {"0", "1", "all"})
        self.assertEqual(choices["0"]["UUID"], "N/A")
        self.assertEqual(choices["0"]["bdf"], "N/A")
        self.assertEqual(choices["0"]["Device Handle"], "bad-handle")
        self.assertEqual(choices["1"]["UUID"], _GOOD_UUID)
        self.assertEqual(choices["1"]["bdf"], _GOOD_BDF)
        self.assertIn("ID: 0", choices_str)

    def test_parser_construction_survives_failing_nic(self):
        argv = ["amd-smi", "version"]
        with mock.patch.object(sys, "argv", argv):
            parser = self.parser_module.AMDSMIParser(
                *([mock.Mock()] * 20), sys_argv=argv, helpers=self.helpers
            )

        self.assertEqual(set(parser.subparsers.choices), {"version"})
        self.assertIn("1", parser.nic_choices)

    def test_bdf_selection_skips_the_failing_nic(self):
        choices, _ = self.helpers.get_nic_choices()
        del choices["all"]

        selected = self.helpers.get_device_handles_from_nic_selections([_GOOD_BDF], choices)

        self.assertEqual(selected, (True, ["good-handle"]))
