#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""``amd-smi set --compute-partition-mem-alloc-mode`` reports under the same
key as ``amd-smi static --partition`` (hardware-free)."""

import os
import types
import unittest

try:
    from common.common import (
        cli_search_order,
        fake_module,
        find_cli_dir,
        load_cli_module,
        stub_modules,
    )
except (ImportError, FileNotFoundError):  # pragma: no cover - harness/install unavailable
    cli_search_order = None
    find_cli_dir = None

_CLI_DIR = (
    find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
    if find_cli_dir and cli_search_order
    else None
)
SET_VALUE_PATH = os.path.join(_CLI_DIR, "subcommands", "set_value.py") if _CLI_DIR else ""
STATIC_PATH = os.path.join(_CLI_DIR, "subcommands", "static.py") if _CLI_DIR else ""

_KEY = "compute_partition_mem_alloc_mode"
_STATUS_INVAL = 1
_STATUS_NO_PERM = 10


class _FakeLibraryException(Exception):
    def __init__(self, err_code):
        super().__init__("mock error")
        self._err_code = err_code

    def get_error_code(self):
        return self._err_code

    def get_error_info(self, detailed=True):
        return "mock error"


class _RecordingLogger:
    def __init__(self):
        self.format = "human"
        self.keys = []

    def store_output(self, device, key, value):
        self.keys.append(key)

    def print_output(self, *args, **kwargs):
        pass

    def clear_multiple_devices_output(self):
        pass


class _StubErrorCollector:
    def record_library_error(self, error_code):
        pass

    def record(self, code):
        pass


class _StubHelpers:
    error_collector = _StubErrorCollector()

    def check_required_groups(self):
        pass

    def handle_gpus(self, args, logger, func):
        return (False, args.gpu)

    def is_baremetal(self):
        return True

    def get_gpu_id_from_device_handle(self, handle):
        return 0

    def store_device_error(self, logger, device, key, message, exception=None, code=None):
        logger.store_output(device, key, message)


class TestSetMemAllocModeKey(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(SET_VALUE_PATH):
            raise unittest.SkipTest(f"amd-smi CLI set_value.py not found at {SET_VALUE_PATH}")
        wrapper = fake_module("amdsmi.amdsmi_wrapper", AMDSMI_STATUS_NO_PERM=_STATUS_NO_PERM)
        interface = fake_module(
            "amdsmi.amdsmi_interface",
            amdsmi_wrapper=wrapper,
            AMDSMI_MAX_PPT_LIMIT=0,
            AMDSMI_MAX_UTIL=100,
            AMDSMI_MAX_STRING_LENGTH=256,
            AmdSmiAcceleratorPartitionMemAllocModeType={"ALL": 1, "CAPPING": 0},
            amdsmi_get_gpu_device_bdf=lambda _handle: "0000:00:00.0",
        )
        exception = fake_module(
            "amdsmi.amdsmi_exception", AmdSmiLibraryException=_FakeLibraryException
        )
        modules = {
            "amdsmi": fake_module("amdsmi", amdsmi_interface=interface, amdsmi_exception=exception),
            "amdsmi.amdsmi_interface": interface,
            "amdsmi.amdsmi_exception": exception,
            "amdsmi.amdsmi_wrapper": wrapper,
            "amdsmi_helpers": fake_module("amdsmi_helpers", AMDSMIHelpers=object),
            "_version": fake_module("_version", __version__="0.0.0-test"),
        }
        stub_modules(cls, modules)
        cls.interface = interface
        cls.module = load_cli_module("set_value_under_test_mem_alloc", SET_VALUE_PATH, _CLI_DIR)

    def _keys_for(self, mode, set_fn):
        self.interface.amdsmi_set_gpu_accelerator_partition_mem_alloc_mode = set_fn
        cmd = self.module.SetValueCommands()
        cmd.logger = _RecordingLogger()
        cmd.helpers = _StubHelpers()
        cmd.group_check_printed = True
        cmd.device_handles = ["gpu0"]
        args = types.SimpleNamespace(
            gpu="gpu0",
            fan=None,
            perf_level=None,
            profile=None,
            perf_determinism=None,
            compute_partition=None,
            memory_partition=None,
            power_cap=None,
            soc_pstate=None,
            xgmi_plpd=None,
            process_isolation=None,
            clk_limit=None,
            clk_level=None,
            ptl_status=None,
            ptl_format=None,
            mem_carveout=None,
            compute_partition_mem_alloc_mode=mode,
            ampp_activate=None,
            ampp_configure=None,
        )
        try:
            cmd.set_gpu(args)
        except PermissionError:
            pass
        return cmd.logger.keys

    def _raise(self, code):
        def _set(*_args):
            raise _FakeLibraryException(code)

        return _set

    def test_static_reports_the_same_key(self):
        with open(STATIC_PATH, encoding="utf-8") as f:
            self.assertIn(f'"{_KEY}"', f.read())

    def test_success(self):
        self.assertEqual(self._keys_for("ALL", lambda *_args: None), [_KEY])

    def test_invalid_mode(self):
        self.assertEqual(self._keys_for("BOGUS", lambda *_args: None), [_KEY])

    def test_no_permission(self):
        self.assertEqual(self._keys_for("ALL", self._raise(_STATUS_NO_PERM)), [_KEY])

    def test_library_error(self):
        self.assertEqual(self._keys_for("ALL", self._raise(_STATUS_INVAL)), [_KEY])
