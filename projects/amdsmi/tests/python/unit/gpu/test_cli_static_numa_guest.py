#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``amd-smi static --numa`` on a Linux virtual OS (not passthrough).

The parser must register ``-u``/``--numa`` and ``static_gpu`` must report the
NUMA section on a virtual OS, not only on baremetal.
"""

import argparse
import os
import types
import unittest
from unittest import mock

from common.common import (
    amdsmi_path,
    cli_search_order,
    fake_module,
    find_cli_dir,
    load_cli_module,
    stub_modules,
)

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
PARSER_PATH = os.path.join(_CLI_DIR, "amdsmi_parser.py") if _CLI_DIR else None
STATIC_PATH = os.path.join(_CLI_DIR, "subcommands", "static.py") if _CLI_DIR else None

_STATUS_NOT_SUPPORTED = 2
_NUMA_SCOPE = 0
_SOCKET_SCOPE = 1


class _FakeLibraryException(Exception):
    def get_error_info(self):
        return str(self)

    def get_error_code(self):
        return _STATUS_NOT_SUPPORTED


class _FakeInterface(types.ModuleType):
    """``amdsmi_interface`` stub whose unmocked ``amdsmi_*`` calls raise the library error."""

    def __init__(self, **attrs):
        super().__init__("amdsmi.amdsmi_interface")
        for key, value in attrs.items():
            setattr(self, key, value)

    # A class-level hook, unlike a module __getattr__, also works on Python 3.6.
    def __getattr__(self, name):
        if not name.startswith("amdsmi_"):
            raise AttributeError(name)

        def _raise(*_args):
            raise _FakeLibraryException(f"mock: {name} not supported")

        return _raise


class _GuestHelpers:
    """Linux virtual OS guest, neither baremetal nor passthrough."""

    convert_clock_type = dict.fromkeys(
        ["sys", "mem", "df", "soc", "dcef", "vclk0", "vclk1", "dclk0", "dclk1"], 0
    )

    def is_linux(self):
        return True

    def is_virtual_os(self):
        return True

    def is_baremetal(self):
        return False

    def is_hypervisor(self):
        return False

    def is_amdgpu_initialized(self):
        return True

    def is_amd_hsmp_initialized(self):
        return False

    def is_ainic_initialized(self):
        return False

    def is_brcm_nic_initialized(self):
        return False

    def is_brcm_switch_initialized(self):
        return False

    def get_clock_types(self):
        return ["SYS", "MEM", "PCIE"], [0, 1, 2]

    def get_output_format(self):
        return "human"

    def handle_gpus(self, args, _logger, _func):
        return False, args.gpu

    def get_gpu_id_from_device_handle(self, _handle):
        return 0

    def get_bitmask_ranges(self, bitmask_dict):
        return dict.fromkeys(bitmask_dict, "0-7")

    def os_info(self):
        return "Linux Guest"


class _FakeLogger:
    def __init__(self, fmt="human"):
        self._fmt = fmt
        self.captured_values = None
        self.store_gpu_json_output = []

    def is_json_format(self):
        return self._fmt == "json"

    def is_csv_format(self):
        return self._fmt == "csv"

    def is_human_readable_format(self):
        return self._fmt == "human"

    def store_output(self, _gpu, key, value):
        if key == "values":
            self.captured_values = value

    def store_multiple_device_output(self):
        pass

    def print_output(self, *args, **kwargs):
        pass


class TestStaticNumaGuestParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PARSER_PATH or not os.path.isfile(PARSER_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI amdsmi_parser.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        interface = fake_module("amdsmi.amdsmi_interface")
        stub_modules(
            cls,
            {
                "amdsmi": fake_module("amdsmi", amdsmi_interface=interface),
                "amdsmi.amdsmi_interface": interface,
                "amdsmi_helpers": fake_module("amdsmi_helpers", AMDSMIHelpers=object),
                "_version": fake_module("_version", __version__="0.0.0-test"),
                # Drop any stale stub so the parser imports the real module.
                "amdsmi_cli_exceptions": None,
            },
        )
        cls.parser_mod = load_cli_module(
            "amdsmi_parser_static_numa_guest", PARSER_PATH, sys_path_dir=_CLI_DIR
        )

    def _build_static_parser(self):
        fake_self = object.__new__(self.parser_mod.AMDSMIParser)
        fake_self.helpers = _GuestHelpers()
        fake_self.description = "test"
        fake_self.gpu_choices = {}
        fake_self.gpu_choices_str = ""
        fake_self.cpu_choices_str = ""
        fake_self.nic_choices_str = ""
        fake_self.core_choices_str = ""
        fake_self.switch_choices_str = ""
        fake_self.vf_choices = []

        subparsers = argparse.ArgumentParser().add_subparsers()
        fake_self._add_static_parser(subparsers, func=lambda args: None)
        return subparsers.choices["static"]

    def test_numa_registered_on_guest(self):
        args = self._build_static_parser().parse_args(["--numa"])

        self.assertTrue(args.numa)


class TestStaticNumaGuestOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not STATIC_PATH or not os.path.isfile(STATIC_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI static.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        exception = fake_module(
            "amdsmi.amdsmi_exception", AmdSmiLibraryException=_FakeLibraryException
        )

        def _guest_cpu_affinity(_handle, scope):
            # No NUMA node in the guest, but local_cpulist still gives the socket CPUs.
            if scope == _NUMA_SCOPE:
                raise _FakeLibraryException("mock: no NUMA node in guest")
            return [0xFF]

        # Every other default static section degrades to N/A instead of needing a stub.
        interface = _FakeInterface(
            amdsmi_wrapper=types.SimpleNamespace(AMDSMI_STATUS_NOT_SUPPORTED=_STATUS_NOT_SUPPORTED),
            AmdSmiAffinityScope=types.SimpleNamespace(
                NUMA_SCOPE=_NUMA_SCOPE, SOCKET_SCOPE=_SOCKET_SCOPE
            ),
            # No host NUMA topology: PCI numa_node is -1 and KFD falls back to
            # CPU node 0, as on baremetal without firmware proximity info.
            amdsmi_topo_get_numa_node_number=lambda _handle: 0,
            amdsmi_get_gpu_topo_numa_affinity=lambda _handle: -1,
            amdsmi_get_cpu_affinity_with_scope=_guest_cpu_affinity,
        )
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
                    "amdsmi_cli_exceptions",
                    AmdSmiInvalidParameterException=type(
                        "AmdSmiInvalidParameterException", (Exception,), {}
                    ),
                ),
            },
        )
        cls.static_module = load_cli_module("static_numa_guest_under_test", STATIC_PATH)
        cls.interface = interface

    def _run_static(self, numa, fmt="human"):
        commands = object.__new__(self.static_module.StaticCommands)
        commands.logger = _FakeLogger(fmt)
        commands.helpers = _GuestHelpers()
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
            clock=False,
            mem_carveout=False,
            partition=False,
            numa=numa,
        )

        commands.static_gpu(args)

        static_dict = commands.logger.captured_values
        self.assertIsNotNone(static_dict, "static_gpu stored no values payload")
        return static_dict

    def test_numa_reported_on_guest(self):
        static_dict = self._run_static(numa=True)

        self.assertIn("numa", static_dict)
        self.assertEqual(static_dict["numa"]["node"], 0)
        self.assertEqual(static_dict["numa"]["affinity"], "NONE")

    def test_numa_in_default_output_on_guest(self):
        static_dict = self._run_static(numa=False)

        self.assertIn("numa", static_dict)

    def test_numa_csv_row_on_guest(self):
        # --csv keeps -1 rather than NONE and must not drop socket affinity without a NUMA node.
        row = self._run_static(numa=True, fmt="csv")

        self.assertEqual(row["node"], 0)
        self.assertEqual(row["affinity"], -1)
        self.assertEqual(row["cpu_list"], "cpu_list_0")
        self.assertEqual(row["bitmask"], "N/A")
        self.assertEqual(row["socket_bitmask"], "00000000000000FF")
        self.assertEqual(row["socket_cpu_cores_affinity"], "0-7")

    def test_numa_csv_row_without_socket_affinity(self):
        def _numa_scope_only(_handle, scope):
            if scope == _SOCKET_SCOPE:
                raise _FakeLibraryException("mock: no local_cpulist")
            return [0xFF]

        with mock.patch.object(
            self.interface, "amdsmi_get_cpu_affinity_with_scope", _numa_scope_only
        ):
            row = self._run_static(numa=True, fmt="csv")

        self.assertEqual(row["bitmask"], "00000000000000FF")
        self.assertEqual(row["cpu_cores_affinity"], "0-7")
        self.assertEqual(row["socket_bitmask"], "N/A")

    def test_numa_csv_row_without_any_cpu_affinity(self):
        def _no_cpu_affinity(_handle, _scope):
            raise _FakeLibraryException("mock: no CPU affinity")

        with mock.patch.object(
            self.interface, "amdsmi_get_cpu_affinity_with_scope", _no_cpu_affinity
        ):
            row = self._run_static(numa=True, fmt="csv")

        self.assertEqual(row["cpu_list"], "N/A")
        self.assertEqual(row["bitmask"], "N/A")
        self.assertEqual(row["socket_bitmask"], "N/A")
