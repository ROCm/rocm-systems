#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""``--nic`` is registered only on the subcommands that read ``args.nic``.

Elsewhere it parsed silently and the handler fell back to every GPU, so ``reset --nic 0 -G`` reset
all of them.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from common.common import cli_search_order, fake_module, find_cli_dir, load_cli_module, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(str(Path(__file__).parent)))
_PARSER_PATH = str(Path(_CLI_DIR) / "amdsmi_parser.py") if _CLI_DIR else None

_NIC_CONSUMERS = ("list", "static", "metric", "firmware")
_LINUX_BAREMETAL = {"is_amdgpu_initialized", "is_ainic_initialized", "is_linux", "is_baremetal"}
# ``profile`` registers only on a Windows hypervisor, where the full parser does not build.
_WINDOWS_HYPERVISOR = {
    "is_amdgpu_initialized",
    "is_ainic_initialized",
    "is_windows",
    "is_hypervisor",
}
_PLATFORMS = ((_LINUX_BAREMETAL, "--help"), (_WINDOWS_HYPERVISOR, "profile"))

_NIC_HANDLES = ["nic-0", "nic-1"]


def _select_nics(nic_selections, nic_choices=None):
    """Bare ``--nic`` arrives as ["all"]; an ID selects that NIC's handle."""
    if "all" in nic_selections:
        return True, list(_NIC_HANDLES)
    return True, [_NIC_HANDLES[int(selection)] for selection in nic_selections]


class TestNicArgScope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stub_modules(
            cls,
            {
                "_version": fake_module("_version", __version__="0.0.0+test"),
                "amdsmi_helpers": fake_module("amdsmi_helpers", AMDSMIHelpers=mock.Mock),
                "nic_scope_parser_under_test": None,
            },
        )
        cls.module = load_cli_module("nic_scope_parser_under_test", _PARSER_PATH, _CLI_DIR)
        cls.exceptions = sys.modules["amdsmi_cli_exceptions"]

    def _parser(self, argv, initialized=_LINUX_BAREMETAL):
        helpers = mock.MagicMock()
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
            "is_linux",
        ):
            getattr(helpers, name).return_value = name in initialized
        helpers.os_info.return_value = "Linux Baremetal"
        helpers.get_rocm_version.return_value = "unknown"
        helpers.get_clock_types.side_effect = lambda: (["SCLK", "MCLK", "PCIE"], "")
        helpers.get_gpu_choices.return_value = ({}, "")
        helpers.get_power_caps.return_value = (0, 0, 0, 0)
        helpers.get_accelerator_choices_types_indices.return_value = ({}, {})
        helpers.get_nic_choices.return_value = ({"0": {}}, "ID: 0\n")
        helpers.get_device_handles_from_nic_selections.side_effect = _select_nics
        with mock.patch.dict(os.environ):
            os.environ.pop("_ARGCOMPLETE", None)
            return self.module.AMDSMIParser(*([mock.Mock()] * 20), sys_argv=argv, helpers=helpers)

    def _parse(self, argv, initialized=_LINUX_BAREMETAL):
        argv_full = ["amd-smi"] + argv
        with mock.patch.object(sys, "argv", argv_full):
            return self._parser(argv_full, initialized).parse_args(argv)

    def _device_subcommands(self):
        """Primary name -> (subparser, platform) for every device-taking subcommand."""
        found = {}
        for initialized, registering_arg in _PLATFORMS:
            parser = self._parser(["amd-smi", registering_arg], initialized)
            for name, sub in parser.subparsers.choices.items():
                is_alias = any(sub is known for known, _ in found.values())
                if "--gpu" in sub._option_string_actions and not is_alias:
                    found.setdefault(name, (sub, initialized))
        return found

    @staticmethod
    def _required_flags(sub):
        """One flag from each required exclusive group, so a missing one cannot mask --nic."""
        return [
            group._group_actions[0].option_strings[0]
            for group in sub._mutually_exclusive_groups
            if group.required
        ]

    def test_every_device_subcommand_is_classified(self):
        # Each _add_device_arguments call site is one subcommand; the four consumers pass nic=True.
        source = Path(_PARSER_PATH).read_text()
        call_sites = source.count("self._add_device_arguments(")
        consumer_sites = source.count("required=False, nic=True)")
        self.assertEqual(
            len(self._device_subcommands()) - len(_NIC_CONSUMERS), call_sites - consumer_sites
        )

    def test_other_subcommands_reject_nic(self):
        subcommands = self._device_subcommands()
        others = sorted(set(subcommands) - set(_NIC_CONSUMERS))
        self.assertTrue(others)
        for command in others:
            for flag in ("--nic", "-N"):
                with self.subTest(command=command, flag=flag):
                    sub, initialized = subcommands[command]
                    argv = [command, *self._required_flags(sub), flag, "0"]
                    with self.assertRaises(
                        self.exceptions.AmdSmiInvalidParameterException
                    ) as caught:
                        self._parse(argv, initialized)
                    self.assertIn(flag, str(caught.exception.json_message))

    def test_consuming_subcommands_keep_bare_nic_distinct_from_an_id(self):
        for command in _NIC_CONSUMERS:
            for flag in ("--nic", "-N"):
                with self.subTest(command=command, flag=flag, selector="bare"):
                    self.assertEqual(self._parse([command, flag]).nic, _NIC_HANDLES)
                with self.subTest(command=command, flag=flag, selector="0"):
                    self.assertEqual(self._parse([command, flag, "0"]).nic, ["nic-0"])
