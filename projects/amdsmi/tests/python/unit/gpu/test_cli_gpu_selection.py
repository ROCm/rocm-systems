#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for repeated device selections and repeated ``set`` values.

Drives the real parser and ``set`` handler with library initialization stubbed
and every device call mocked, so it runs without GPUs.
"""

import inspect
import os
import sys
import unittest
from unittest import mock

from common.common import (
    cli_search_order,
    fake_module,
    find_cli_dir,
    generated_version_stub,
    load_cli_module,
    stub_modules,
)

from amdsmi import amdsmi_exception, amdsmi_interface

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
PARSER_PATH = os.path.join(_CLI_DIR, "amdsmi_parser.py") if _CLI_DIR else None
SET_VALUE_PATH = os.path.join(_CLI_DIR, "subcommands", "set_value.py") if _CLI_DIR else None

_HANDLE = amdsmi_interface.amdsmi_wrapper.amdsmi_processor_handle


def _device_choices(first_value, count, domain):
    return {
        str(index): {
            "Device Handle": _HANDLE(first_value + index),
            "bdf": f"{domain:04x}:0{index + 1}:00.0",
            "UUID": f"{domain:08x}-0000-0000-0000-00000000000{index}",
            "CUID": f"cuid-{domain}-{index}",
        }
        for index in range(count)
    }


_GPUS = _device_choices(0x100, 3, 0)
_CPUS = _device_choices(0x200, 2, 1)
_CORES = _device_choices(0x300, 2, 2)
_NICS = _device_choices(0x400, 2, 3)
_SWITCHES = _device_choices(0x500, 2, 4)
_GPU_VALUES = [0x100, 0x101, 0x102]


def _values(handles):
    return [handle.value for handle in handles]


def _fresh_gpu_handles():
    # The library returns new handle objects on every query, so they never compare equal.
    return [_HANDLE(value) for value in _GPU_VALUES]


def _fake_modules():
    # amdsmi_helpers star-imports amdsmi_init, whose import initializes the library.
    return {
        "amdsmi_init": fake_module(
            "amdsmi_init",
            AMDSMI_INIT_FLAG=0,
            AMDSMI_INITIALIZED=True,
            amdsmi_interface=amdsmi_interface,
            amdsmi_exception=amdsmi_exception,
        ),
        "amdsmi_helpers": None,
        "amdsmi_cli_exceptions": None,
        **generated_version_stub(),
    }


def _baremetal_helpers(helpers_type, errors):
    class _Helpers(helpers_type):
        """Bare-metal Linux host with three GPUs and two of each other device type."""

        def __init__(self):
            # The base constructor probes the platform.
            self._count_of_sets_called = 0
            self.error_collector = errors.AmdSmiErrorCollector()

        def is_linux(self):
            return True

        def is_windows(self):
            return False

        def is_baremetal(self):
            return True

        def is_hypervisor(self):
            return False

        def is_amdgpu_initialized(self):
            return True

        def is_amd_hsmp_initialized(self):
            return True

        def is_ainic_initialized(self):
            return True

        def is_brcm_nic_initialized(self):
            return False

        def is_brcm_switch_initialized(self):
            return True

        def get_output_format(self):
            return "human_readable"

        def os_info(self, string_format=True):
            return "Linux Baremetal"

        def get_rocm_version(self):
            return "test"

        def get_gpu_choices(self):
            return dict(_GPUS), "0, 1, 2"

        def get_cpu_choices(self):
            return dict(_CPUS), "0, 1"

        def get_core_choices(self):
            return dict(_CORES), "0, 1"

        def get_nic_choices(self):
            return dict(_NICS), "0, 1"

        def get_switch_choices(self):
            return dict(_SWITCHES), "0, 1"

        def get_fan_support(self):
            return "0-255"

        def get_power_caps(self):
            return "0 W", "550 W", "0 W", "550 W"

        def get_max_node_power_limit(self):
            return "N/A"

        def get_soc_pstates(self):
            return []

        def get_xgmi_plpd_policies(self):
            return []

        def get_accelerator_choices_types_indices(self):
            return ["SPX", "DPX", "QPX", "CPX", "0", "1", "2", "3"], {}

        def confirm_changing_memory_partition_gpu_reload_warning(self):
            pass

        def get_gpu_id_from_device_handle(self, input_device_handle):
            return _GPU_VALUES.index(input_device_handle.value)

    return _Helpers()


class _CliFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PARSER_PATH or not os.path.isfile(PARSER_PATH):
            raise unittest.SkipTest(f"amd-smi CLI amdsmi_parser.py not found in {_CLI_DIR}")
        stub_modules(cls, _fake_modules())
        cls.parser_mod = load_cli_module(
            "amdsmi_parser_gpu_selection_test", PARSER_PATH, sys_path_dir=_CLI_DIR
        )
        cls.set_value_mod = load_cli_module(
            "set_value_gpu_selection_test", SET_VALUE_PATH, sys_path_dir=_CLI_DIR
        )
        cls.errors = cls.parser_mod.amdsmi_cli_exceptions

    def setUp(self):
        self.helpers = _baremetal_helpers(self.parser_mod.AMDSMIHelpers, self.errors)
        # AMDSMISubParser.error() builds its own helpers to read the output format.
        patcher = mock.patch.object(self.parser_mod, "AMDSMIHelpers", return_value=self.helpers)
        patcher.start()
        self.addCleanup(patcher.stop)

    def parse(self, command, *argv):
        names = inspect.signature(self.parser_mod.AMDSMIParser).parameters
        callbacks = {name: mock.Mock() for name in names if name not in ("sys_argv", "helpers")}
        full_argv = ["amd-smi", command, *argv]
        parser = self.parser_mod.AMDSMIParser(**callbacks, sys_argv=full_argv, helpers=self.helpers)
        with mock.patch.object(sys, "argv", full_argv):
            return parser.parse_args(full_argv[1:])


class TestRepeatedDeviceSelection(_CliFixture):
    def gpus(self, *argv, command="metric"):
        with mock.patch.object(
            amdsmi_interface, "amdsmi_get_processor_handles", side_effect=_fresh_gpu_handles
        ):
            return _values(self.parse(command, *argv).gpu)

    def test_repeated_gpu_options_combine(self):
        for argv in (
            ["-g", "0", "-g", "1"],
            ["--gpu", "0", "--gpu", "1"],
            ["-g=0", "-g=1"],
            ["--gpu=0", "--gpu=1"],
            ["-g", "0", "--gpu=1"],
        ):
            with self.subTest(argv=argv):
                self.assertEqual(self.gpus(*argv), _GPU_VALUES[:2])

    def test_repeated_gpu_options_match_one_list(self):
        for repeated, listed in (
            (["-g", "0", "1", "-g", "2"], ["-g", "0", "1", "2"]),
            (["-g", "2", "-g", "0"], ["-g", "2", "0"]),
            (["-g", "0", "-g", "all"], ["-g", "0", "all"]),
        ):
            with self.subTest(argv=repeated):
                self.assertEqual(
                    self.gpus(*repeated, command="list"), self.gpus(*listed, command="list")
                )

    def test_duplicate_gpus_are_selected_once(self):
        for argv, expected in (
            (["-g", "1", "-g", "1"], [0x101]),
            (["-g", "1", "1"], [0x101]),
            (["-g", "0", "-g", "all"], _GPU_VALUES),
            (["-g", "all", "-g", "0"], _GPU_VALUES),
            (["-g", "2", "-g", "0", "2", "1"], [0x102, 0x100, 0x101]),
        ):
            with self.subTest(argv=argv):
                self.assertEqual(self.gpus(*argv), expected)

    def test_repeated_gpu_options_accept_bdf_and_uuid(self):
        bdf = _GPUS["0"]["bdf"]
        uuid = _GPUS["2"]["UUID"]
        self.assertEqual(self.gpus("--gpu", bdf, "--gpu", uuid.upper()), [0x100, 0x102])

    def test_single_gpu_selections_unchanged(self):
        self.assertEqual(self.gpus("-g", "1"), [0x101])
        self.assertEqual(self.gpus("-g", "0", "2"), [0x100, 0x102])
        self.assertEqual(self.gpus("-g", "all"), _GPU_VALUES)
        self.assertIsNone(self.parse("metric").gpu)

    def test_other_selectors_combine_repeated_options(self):
        for argv, dest, expected in (
            (["-U", "0", "-U", "1"], "cpu", [0x200, 0x201]),
            (["--cpu", "1", "--cpu=1"], "cpu", [0x201]),
            (["-O", "0", "-O", "1"], "core", [0x300, 0x301]),
            (["-N", _NICS["0"]["bdf"], "--nic", _NICS["1"]["bdf"]], "nic", [0x400, 0x401]),
            (["--switch", "0", "--switch", "1", "--switch", "0"], "switch", [0x500, 0x501]),
        ):
            with self.subTest(argv=argv):
                self.assertEqual(_values(getattr(self.parse("list", *argv), dest)), expected)

    def test_other_selectors_select_all_once(self):
        cpus = {"processor_handles": [_HANDLE(0x200), _HANDLE(0x201)]}
        with mock.patch.object(amdsmi_interface, "amdsmi_get_cpu_handles", return_value=cpus):
            args = self.parse("list", "-U", "1", "-U", "all")
        self.assertEqual(_values(args.cpu), [0x201, 0x200])

    def test_invalid_repeated_gpu_is_rejected(self):
        for argv, error in (
            (["-g", "0", "-g", "9"], self.errors.AmdSmiDeviceNotFoundException),
            (["-g", "0", "-g", "bad"], self.errors.AmdSmiInvalidParameterValueException),
            (["-g", "0", "-g"], self.errors.AmdSmiMissingParameterValueException),
            (["-g", "0", "-U", "0"], self.errors.AmdSmiInvalidParameterException),
        ):
            with self.subTest(argv=argv):
                with self.assertRaises(error):
                    self.parse("metric", *argv)

    def test_comma_separated_gpu_values_stay_unsupported(self):
        with self.assertRaises(self.errors.AmdSmiInvalidParameterValueException):
            self.parse("metric", "-g=0,1")
        with self.assertRaises(self.errors.AmdSmiInvalidParameterException):
            self.parse("metric", "-g=0", "1")


class TestRepeatedSetValues(_CliFixture):
    """Single-value set options keep the last value; only different setters conflict."""

    def test_repeated_partition_setter_keeps_last_value(self):
        for argv, dest, expected in (
            (["-M", "NPS1", "-M", "NPS2"], "memory_partition", "NPS2"),
            (["-M", "NPS2", "--memory-partition=NPS1"], "memory_partition", "NPS1"),
            (["-C", "CPX", "-C", "DPX"], "compute_partition", "DPX"),
            (
                ["--compute-partition", "SPX", "--accelerator-partition", "CPX"],
                "compute_partition",
                "CPX",
            ),
            (["-a", "ALL", "-a", "CAPPING"], "compute_partition_mem_alloc_mode", "CAPPING"),
        ):
            with self.subTest(argv=argv):
                self.assertEqual(getattr(self.parse("set", *argv), dest), expected)

    def test_different_partition_setters_are_rejected(self):
        for argv in (
            ["-M", "NPS1", "-C", "SPX"],
            ["-C", "CPX", "-a", "ALL"],
            ["-M", "NPS1", "-a", "ALL"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaises(self.errors.AmdSmiInvalidParameterException):
                    self.parse("set", *argv)

    def set_memory_partition(self, *argv):
        args = self.parse("set", *argv)
        command = self.set_value_mod.SetValueCommands()
        command.helpers = self.helpers
        command.logger = mock.Mock(format="human_readable")
        command.group_check_printed = True
        with mock.patch.object(
            amdsmi_interface, "amdsmi_get_gpu_device_bdf", return_value="0000:01:00.0"
        ), mock.patch.object(
            amdsmi_interface,
            "amdsmi_get_gpu_memory_partition_config",
            return_value={"partition_caps": ["NPS1", "NPS2"], "mp_mode": "NPS1"},
        ), mock.patch.object(amdsmi_interface, "amdsmi_set_gpu_memory_partition_mode") as setter:
            command.set_gpu(args)
        return [(handle.value, mode) for (handle, mode), _ in setter.call_args_list]

    def test_repeated_gpus_receive_last_partition_value(self):
        nps2 = amdsmi_interface.AmdSmiMemoryPartitionType.NPS2
        self.assertEqual(
            self.set_memory_partition("-g", "0", "-M", "NPS1", "-g", "1", "-M", "NPS2"),
            [(0x100, nps2), (0x101, nps2)],
        )

    def test_duplicate_gpus_are_set_once(self):
        nps2 = amdsmi_interface.AmdSmiMemoryPartitionType.NPS2
        for argv in (["-g", "0", "-g", "0"], ["-g", "0", "0"]):
            with self.subTest(argv=argv):
                self.assertEqual(self.set_memory_partition(*argv, "-M", "NPS2"), [(0x100, nps2)])
