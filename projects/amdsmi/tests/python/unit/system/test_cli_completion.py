#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import unittest
from unittest import mock

from common.common import cli_search_order, fake_module, find_cli_dir, load_cli_module, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(str(Path(__file__).parent)))
_PARSER_PATH = str(Path(_CLI_DIR) / "amdsmi_parser.py") if _CLI_DIR else None


class TestCliCompletion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stub_modules(
            cls,
            {
                "_version": fake_module("_version", __version__="0.0.0+test"),
                "amdsmi_helpers": fake_module("amdsmi_helpers", AMDSMIHelpers=mock.Mock),
                "completion_parser_under_test": None,
            },
        )
        cls.module = load_cli_module("completion_parser_under_test", _PARSER_PATH, _CLI_DIR)

    def _parser(self, argv, completing=False):
        helpers = mock.Mock()
        for name in (
            "is_amdgpu_initialized",
            "is_brcm_switch_initialized",
            "is_ainic_initialized",
            "is_brcm_nic_initialized",
            "is_amd_hsmp_initialized",
            "is_hypervisor",
            "is_baremetal",
            "is_windows",
            "is_virtual_os",
        ):
            getattr(helpers, name).return_value = False
        helpers.is_linux.return_value = True
        helpers.os_info.return_value = "Linux Baremetal"
        helpers.get_rocm_version.return_value = "unknown"
        helpers.get_clock_types.side_effect = lambda: (["SCLK", "MCLK", "PCIE"], "")
        with mock.patch.dict(os.environ):
            os.environ.pop("_ARGCOMPLETE", None)
            if completing:
                os.environ["_ARGCOMPLETE"] = "1"
            return self.module.AMDSMIParser(*([mock.Mock()] * 20), sys_argv=argv, helpers=helpers)

    def test_completion_registers_subcommands_without_argv(self):
        expected = set(self._parser(["amd-smi", "--help"]).subparsers.choices)
        actual = set(self._parser(["amd-smi"], completing=True).subparsers.choices)
        self.assertIn("version", expected)
        self.assertEqual(actual, expected)

    def test_completion_registers_subcommands_with_partial_argv(self):
        expected = set(self._parser(["amd-smi", "--help"]).subparsers.choices)
        actual = set(self._parser(["amd-smi", "version"], completing=True).subparsers.choices)
        self.assertEqual(actual, expected)

    def test_normal_invocation_keeps_lazy_subcommand_registration(self):
        self.assertEqual(set(self._parser(["amd-smi"]).subparsers.choices), {"default"})
        self.assertEqual(set(self._parser(["amd-smi", "version"]).subparsers.choices), {"version"})

    def test_cli_script_has_global_completion_marker(self):
        head = (Path(_CLI_DIR) / "amdsmi_cli.py").read_bytes()[:1024]
        self.assertIn(b"PYTHON_ARGCOMPLETE_OK", head)
