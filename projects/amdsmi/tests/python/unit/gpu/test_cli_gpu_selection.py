#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for repeated ``--gpu`` selections and repeated ``set`` values.

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

_HANDLES = ["gpu-handle-0", "gpu-handle-1", "gpu-handle-2"]
_GPU_CHOICES = {
    str(index): {
        "Device Handle": handle,
        "bdf": f"0000:0{index + 1}:00.0",
        "UUID": f"00000000-0000-0000-0000-00000000000{index}",
        "CUID": f"cuid-{index}",
    }
    for index, handle in enumerate(_HANDLES)
}


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
        # Another suite may have stubbed _version during discovery; find_spec rejects that.
        **({} if "_version" in sys.modules else generated_version_stub()),
    }


def _baremetal_helpers(helpers_type, errors):
    class _Helpers(helpers_type):
        """Bare-metal Linux host with three GPUs and no other device types."""

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
            return False

        def is_ainic_initialized(self):
            return False

        def is_brcm_nic_initialized(self):
            return False

        def is_brcm_switch_initialized(self):
            return False

        def get_output_format(self):
            return "human_readable"

        def os_info(self, string_format=True):
            return "Linux Baremetal"

        def get_rocm_version(self):
            return "test"

        def get_gpu_choices(self):
            return dict(_GPU_CHOICES), "0, 1, 2"

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
            return _HANDLES.index(input_device_handle)

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


class TestRepeatedGpuSelection(_CliFixture):
    def test_repeated_gpu_options_combine(self):
        for argv in (
            ["-g", "0", "-g", "1"],
            ["--gpu", "0", "--gpu", "1"],
            ["-g=0", "-g=1"],
            ["--gpu=0", "--gpu=1"],
            ["-g", "0", "--gpu=1"],
        ):
            with self.subTest(argv=argv):
                self.assertEqual(self.parse("metric", *argv).gpu, _HANDLES[:2])

    def test_repeated_gpu_options_match_one_list(self):
        for repeated, listed in (
            (["-g", "0", "1", "-g", "2"], ["-g", "0", "1", "2"]),
            (["-g", "2", "-g", "0"], ["-g", "2", "0"]),
            (["-g", "1", "-g", "1"], ["-g", "1", "1"]),
        ):
            with self.subTest(argv=repeated):
                self.assertEqual(self.parse("list", *repeated).gpu, self.parse("list", *listed).gpu)

    def test_repeated_gpu_options_accept_bdf_and_uuid(self):
        bdf = _GPU_CHOICES["0"]["bdf"]
        uuid = _GPU_CHOICES["2"]["UUID"]
        args = self.parse("metric", "--gpu", bdf, "--gpu", uuid.upper())
        self.assertEqual(args.gpu, [_HANDLES[0], _HANDLES[2]])

    def test_single_gpu_selections_unchanged(self):
        self.assertEqual(self.parse("metric", "-g", "1").gpu, [_HANDLES[1]])
        self.assertEqual(self.parse("metric", "-g", "0", "2").gpu, [_HANDLES[0], _HANDLES[2]])
        self.assertIsNone(self.parse("metric").gpu)
        with mock.patch.object(
            amdsmi_interface, "amdsmi_get_processor_handles", return_value=list(_HANDLES)
        ):
            self.assertEqual(self.parse("metric", "-g", "all").gpu, _HANDLES)

    def test_invalid_repeated_gpu_is_rejected(self):
        for argv, error in (
            (["-g", "0", "-g", "9"], self.errors.AmdSmiDeviceNotFoundException),
            (["-g", "0", "-g", "bad"], self.errors.AmdSmiInvalidParameterValueException),
            (["-g", "0", "-g"], self.errors.AmdSmiMissingParameterValueException),
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

    def test_repeated_gpus_receive_last_partition_value(self):
        args = self.parse("set", "-g", "0", "-M", "NPS1", "-g", "1", "-M", "NPS2")
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
        nps2 = amdsmi_interface.AmdSmiMemoryPartitionType.NPS2
        self.assertEqual(
            setter.call_args_list, [mock.call(_HANDLES[0], nps2), mock.call(_HANDLES[1], nps2)]
        )
