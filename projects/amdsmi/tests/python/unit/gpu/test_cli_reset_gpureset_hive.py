#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for which GPUs ``amd-smi reset --gpureset`` resets.

A mode-1 reset of one GPU in an XGMI hive resets the whole hive, so the CLI
resets each XGMI hive once. UALink GPUs also report a shared hive ID but must
each be reset. These tests stub the C library so they run without GPU hardware.
"""

import os
import types
import unittest

# ``common.common`` bootstraps the real amdsmi package at import time, which fails
# on a stale or mismatched install. This suite fully stubs ``amdsmi``, so degrade
# gracefully: without the harness there is no resolver and the suite skips.
try:
    from common.common import (
        amdsmi_path,
        cli_search_order,
        find_cli_dir,
        load_cli_module,
        stub_modules,
    )
except (ImportError, FileNotFoundError):  # pragma: no cover - harness/install unavailable
    amdsmi_path = None
    cli_search_order = None
    find_cli_dir = None

_CLI_DIR = (
    find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
    if find_cli_dir and cli_search_order
    else None
)
RESET_PATH = os.path.join(_CLI_DIR, "subcommands", "reset.py") if _CLI_DIR else ""


class _FakeLibraryException(Exception):
    """Stand-in for ``amdsmi_exception.AmdSmiLibraryException``."""

    def get_error_code(self):
        return -1

    def get_error_info(self, detailed=True):
        return "mock error"


def _build_fake_amdsmi():
    """Build a stub ``amdsmi`` package so ``reset.py`` imports cleanly."""
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")
    wrapper = types.ModuleType("amdsmi.amdsmi_wrapper")

    wrapper.AMDSMI_STATUS_NO_PERM = 4
    interface.amdsmi_wrapper = wrapper
    exception.AmdSmiLibraryException = _FakeLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception

    return {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
        "amdsmi.amdsmi_wrapper": wrapper,
    }


class _Args(types.SimpleNamespace):
    """``reset`` arguments: any option not given here reads as None (not passed)."""

    def __getattr__(self, name):
        return None


class _NullLogger:
    format = "human"

    def store_output(self, device, key, value):
        pass

    def print_output(self, *args, **kwargs):
        pass

    def clear_multiple_devices_output(self):
        pass


class _StubHelpers:
    """Minimal ``self.helpers`` stub for the bare-metal ``reset --gpureset`` path."""

    def check_required_groups(self):
        pass

    def handle_gpus(self, args, logger, func):
        # Mirrors AMDSMIHelpers.handle_gpus: recurse once per handle in a list.
        if isinstance(args.gpu, list):
            for handle in args.gpu:
                func(args, multiple_devices=True, gpu=handle)
            return (True, args.gpu)
        return (False, args.gpu)

    def is_baremetal(self):
        return True

    def is_amd_device(self, handle):
        return True

    def get_gpu_id_from_device_handle(self, handle):
        return 0


class TestResetGpuresetHiveGrouping(unittest.TestCase):
    """Drives the real ``reset --gpureset`` path with the library stubbed."""

    GPUS = ["gpu0", "gpu1", "gpu2", "gpu3"]

    @classmethod
    def setUpClass(cls):
        if not RESET_PATH or not os.path.isfile(RESET_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI reset.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        modules = _build_fake_amdsmi()
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]
        cls.module = load_cli_module(
            "reset_gpureset_hive_under_test", RESET_PATH, sys_path_dir=_CLI_DIR
        )

    def _reset_all(self, hive_ids, fabric_types):
        # fabric_types[gpu] is None when the fabric query is unsupported.
        def _fabric(handle):
            if fabric_types.get(handle) is None:
                raise _FakeLibraryException()
            return {"fabric_type": fabric_types[handle]}

        reset_calls = []
        self.interface.amdsmi_get_xgmi_info = lambda h: {"xgmi_hive_id": hive_ids[h]}
        self.interface.amdsmi_get_gpu_fabric_info = _fabric
        self.interface.amdsmi_reset_gpu = reset_calls.append

        cmd = self.module.ResetCommands()
        cmd.logger = _NullLogger()
        cmd.helpers = _StubHelpers()
        cmd.group_check_printed = True
        cmd.device_handles = list(self.GPUS)
        cmd.reset(_Args(gpureset=True))
        return reset_calls

    def test_ualink_gpus_sharing_hive_id_are_each_reset(self):
        hive_ids = dict.fromkeys(self.GPUS, 16)
        resets = self._reset_all(hive_ids, dict.fromkeys(self.GPUS, "UALINK"))
        self.assertEqual(resets, self.GPUS)

    def test_ualoe_gpus_sharing_hive_id_are_each_reset(self):
        hive_ids = dict.fromkeys(self.GPUS, 16)
        resets = self._reset_all(hive_ids, dict.fromkeys(self.GPUS, "UALOE"))
        self.assertEqual(resets, self.GPUS)

    def test_xgmi_hive_is_reset_once(self):
        hive_ids = dict.fromkeys(self.GPUS, 7)
        self.assertEqual(self._reset_all(hive_ids, {}), ["gpu0"])

    def test_unknown_fabric_type_keeps_hive_grouping(self):
        hive_ids = dict.fromkeys(self.GPUS, 7)
        resets = self._reset_all(hive_ids, dict.fromkeys(self.GPUS, "UNKNOWN"))
        self.assertEqual(resets, ["gpu0"])

    def test_gpus_without_hive_are_each_reset(self):
        hive_ids = {"gpu0": 7, "gpu1": 7, "gpu2": 0, "gpu3": 0}
        self.assertEqual(self._reset_all(hive_ids, {}), ["gpu0", "gpu2", "gpu3"])


if __name__ == "__main__":
    unittest.main()
