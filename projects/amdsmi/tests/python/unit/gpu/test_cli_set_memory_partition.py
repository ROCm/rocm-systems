#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for the ``amd-smi set --memory-partition`` result message.

A memory partition change takes effect on the next driver reload, and asking for
the mode that is already current does not cancel a different mode set earlier.
These tests stub the C library so they run without GPU hardware, and check that
the CLI tells the user to reload only when the requested mode differs from the
current one.
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
SET_VALUE_PATH = os.path.join(_CLI_DIR, "subcommands", "set_value.py") if _CLI_DIR else ""

# Stand-ins for AmdSmiMemoryPartitionType values; set_value.py looks modes up by name.
_MODES = {"NPS1": 1, "NPS2": 2, "NPS4": 4, "NPS8": 8}


class _FakeLibraryException(Exception):
    """Stand-in for ``amdsmi_exception.AmdSmiLibraryException``."""

    def get_error_code(self):
        return -1

    def get_error_info(self, detailed=True):
        return "mock error"


def _build_fake_amdsmi():
    """Build a stub ``amdsmi`` package so ``set_value.py`` imports cleanly."""
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")
    wrapper = types.ModuleType("amdsmi.amdsmi_wrapper")

    # Constants set_value.py binds at import time; the values are irrelevant here.
    interface.AMDSMI_MAX_PPT_LIMIT = 0
    interface.AMDSMI_MAX_UTIL = 100
    wrapper.AMDSMI_STATUS_NO_PERM = 4
    wrapper.AMDSMI_STATUS_INVAL = 1
    interface.amdsmi_wrapper = wrapper
    interface.AmdSmiMemoryPartitionType = _MODES
    interface.amdsmi_get_gpu_device_bdf = lambda _handle: "0000:00:00.0"

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
    """``set`` arguments: any option not given here reads as None (not passed)."""

    def __getattr__(self, name):
        return None


class _RecordingLogger:
    """Minimal ``self.logger`` stub capturing ``store_output`` payloads."""

    def __init__(self):
        self.format = "human"
        self.outputs = []

    def store_output(self, device, key, value):
        self.outputs.append((device, key, value))

    def print_output(self, *args, **kwargs):
        pass

    def clear_multiple_devices_output(self):
        pass

    def last_output(self, key):
        for _device, stored_key, value in reversed(self.outputs):
            if stored_key == key:
                return value
        return None


class _StubHelpers:
    """Minimal ``self.helpers`` stub for the bare-metal GPU ``set`` path."""

    def check_required_groups(self):
        pass

    def handle_gpus(self, args, logger, func):
        # Single device: never recurse, echo the resolved handle back.
        return (False, args.gpu)

    def is_baremetal(self):
        return True

    def get_gpu_id_from_device_handle(self, handle):
        return 0

    def increment_set_count(self):
        pass

    def get_set_count(self):
        # Not the first set, so the interactive reload confirmation is skipped.
        return 2

    def confirm_changing_memory_partition_gpu_reload_warning(self):
        pass

    def store_device_error(self, logger, device, key, message, exception=None, code=None):
        logger.store_output(device, key, message)


class TestSetMemoryPartitionMessage(unittest.TestCase):
    """Drives the real ``set_gpu`` memory-partition branch with the library stubbed."""

    @classmethod
    def setUpClass(cls):
        if not SET_VALUE_PATH or not os.path.isfile(SET_VALUE_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI set_value.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        modules = _build_fake_amdsmi()
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]
        cls.module = load_cli_module(
            "set_value_memory_partition_under_test", SET_VALUE_PATH, sys_path_dir=_CLI_DIR
        )

    def _set(self, requested, current_mode=None):
        # current_mode=None makes the current-mode query fail, as on a GPU that
        # cannot report it.
        def _config(_handle):
            if current_mode is None:
                raise _FakeLibraryException()
            return {"partition_caps": ["NPS1", "NPS4"], "mp_mode": current_mode}

        set_calls = []
        self.interface.amdsmi_get_gpu_memory_partition_config = _config
        self.interface.amdsmi_set_gpu_memory_partition_mode = lambda _h, mode: set_calls.append(
            mode
        )

        logger = _RecordingLogger()
        cmd = self.module.SetValueCommands()
        cmd.logger = logger
        cmd.helpers = _StubHelpers()
        cmd.group_check_printed = True
        cmd.device_handles = ["gpu0"]
        cmd.set_gpu(_Args(gpu="gpu0", memory_partition=requested))
        return set_calls, logger.last_output("memory_partition")

    def test_current_mode_reports_no_change_without_reload_advice(self):
        # Reloading here would apply any mode set earlier, not the requested one.
        set_calls, message = self._set("NPS1", "NPS1")
        self.assertEqual(set_calls, [_MODES["NPS1"]])
        self.assertIn("already NPS1", message)
        self.assertIn("next driver reload still applies it", message)
        self.assertNotIn("modprobe", message)

    def test_different_mode_reports_reload_advice(self):
        set_calls, message = self._set("NPS4", "NPS1")
        self.assertEqual(set_calls, [_MODES["NPS4"]])
        self.assertIn("Successfully set memory partition to NPS4", message)
        self.assertIn("modprobe -r amdgpu", message)

    def test_unreadable_current_mode_keeps_reload_advice(self):
        set_calls, message = self._set("NPS1")
        self.assertEqual(set_calls, [_MODES["NPS1"]])
        self.assertIn("Successfully set memory partition to NPS1", message)
