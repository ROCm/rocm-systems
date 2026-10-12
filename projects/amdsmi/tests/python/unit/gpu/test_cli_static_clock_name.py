#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``amd-smi static --clock`` with a clock name it does not know.

``--clock`` accepts free-form names, so an unknown one reaches the static
command's own check. That check must raise the CLI's invalid-parameter error
naming the bad clock, not a NameError traceback.
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


class _InvalidParameter(Exception):
    def __init__(self, command, arg, output_format):
        super().__init__(arg)
        self.command = command
        self.arg = arg
        self.output_format = output_format


class _FakeLogger:
    def is_json_format(self):
        return False

    def is_csv_format(self):
        return False

    def is_human_readable_format(self):
        return True


class _FakeHelpers:
    convert_clock_type = {"sys": 0, "mem": 1}

    def handle_gpus(self, args, _logger, _func):
        return False, args.gpu

    def get_gpu_id_from_device_handle(self, _handle):
        return 0

    def os_info(self):
        return "mock-os"

    def is_linux(self):
        return True

    def is_baremetal(self):
        return False

    def is_virtual_os(self):
        return True

    def is_hypervisor(self):
        return False

    def get_output_format(self):
        return "human"


class TestCliStaticClockName(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not STATIC_PATH or not os.path.isfile(STATIC_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI static.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        interface = fake_module("amdsmi.amdsmi_interface")
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
                    "amdsmi_cli_exceptions", AmdSmiInvalidParameterException=_InvalidParameter
                ),
            },
        )
        cls.static_module = load_cli_module("static_clock_name_under_test", STATIC_PATH)

    def test_unknown_clock_name_raises_invalid_parameter(self):
        commands = object.__new__(self.static_module.StaticCommands)
        commands.logger = _FakeLogger()
        commands.helpers = _FakeHelpers()
        commands.group_check_printed = True
        args = argparse.Namespace(
            gpu=object(),
            asic=False,
            bus=False,
            vbios=False,
            driver=False,
            ras=False,
            vram=False,
            cache=False,
            board=False,
            process_isolation=False,
            clock=["notaclock"],
            mem_carveout=False,
            partition=False,
        )

        with self.assertRaises(_InvalidParameter) as raised:
            commands.static_gpu(args)

        self.assertEqual(raised.exception.command, "static")
        self.assertEqual(raised.exception.arg, "notaclock")
