#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``amd-smi static --json`` with NICs selected by ``-N`` prints their JSON.

Selecting NICs returns from static() before the combined JSON printed at its
end, so that early return has to print it too: once, with every selected NIC,
and only for JSON.
"""

import argparse
import os
import unittest

from common.common import (
    amdsmi_path,
    cli_search_order,
    fake_module,
    find_cli_dir,
    load_cli_module,
    stub_modules,
)

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
STATIC_PATH = os.path.join(_CLI_DIR, "subcommands", "static.py") if _CLI_DIR else None


class _FakeLogger:
    def __init__(self, json_format):
        self.json_format = json_format
        self.output = {}
        self.store_nic_json_output = []
        self.combined_prints = 0

    def is_json_format(self):
        return self.json_format

    def clear_multiple_devices_output(self):
        pass

    def store_ainic_output(self, *_args):
        pass

    def store_multiple_device_output(self):
        pass

    def print_output(self, **_kwargs):
        pass

    def combine_arrays_to_json(self):
        self.combined_prints += 1


class _FakeHelpers:
    def __init__(self, nics):
        self.nics = nics

    def is_ainic_initialized(self):
        return True

    def is_brcm_nic_initialized(self):
        return False

    def os_info(self):
        return "mock-os"

    def get_ainic_id_from_device_handle(self, handle):
        return self.nics.index(handle)

    def handle_ainics(self, args, logger, subcommand):
        # As AMDSMIHelpers.handle_ainics: more than one NIC runs the subcommand per NIC.
        if isinstance(args.nic, list) and len(args.nic) > 1:
            for handle in args.nic:
                subcommand(args, multiple_devices=True, nic=handle)
            logger.print_output(multiple_device_enabled=True)
            return True, args.nic
        if isinstance(args.nic, list):
            args.nic = args.nic[0]
        return False, args.nic


class TestCliStaticNicJson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not STATIC_PATH or not os.path.isfile(STATIC_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI static.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        interface = fake_module(
            "amdsmi.amdsmi_interface", amdsmi_get_ainic_info=lambda _handle, _detail: {}
        )
        exception = fake_module("amdsmi.amdsmi_exception", AmdSmiLibraryException=Exception)
        stub_modules(
            cls,
            {
                "amdsmi": fake_module(
                    "amdsmi", amdsmi_interface=interface, amdsmi_exception=exception
                ),
                "amdsmi.amdsmi_interface": interface,
                "amdsmi.amdsmi_exception": exception,
                "amdsmi_helpers": fake_module("amdsmi_helpers", AMDSMIHelpers=object),
                "amdsmi_cli_exceptions": fake_module(
                    "amdsmi_cli_exceptions", AmdSmiInvalidParameterException=Exception
                ),
            },
        )
        cls.static_module = load_cli_module("static_nic_json_under_test", STATIC_PATH)

    def _static_for_selected_nics(self, count, json_format=True):
        nics = [object() for _ in range(count)]
        commands = object.__new__(self.static_module.StaticCommands)
        commands.logger = _FakeLogger(json_format)
        commands.helpers = _FakeHelpers(nics)
        # Supplied by ListDevicesCommands in AMDSMICommands; every NIC here is an AI NIC.
        commands._get_nics_from_args = lambda args: ([], list(args.nic))
        commands.static(argparse.Namespace(nic=list(nics), cpu=None, gpu=None))
        return commands.logger

    def test_json_prints_every_selected_nic_once(self):
        for count in (1, 2):
            with self.subTest(nics=count):
                logger = self._static_for_selected_nics(count)
                self.assertEqual(
                    [record["ai_nic"] for record in logger.store_nic_json_output],
                    list(range(count)),
                )
                self.assertEqual(logger.combined_prints, 1)

    def test_other_formats_do_not_print_json(self):
        logger = self._static_for_selected_nics(1, json_format=False)
        self.assertEqual(logger.combined_prints, 0)
