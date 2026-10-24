#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for the FCLK step of ``amd-smi reset --clocks``.

The stubbed library lets these run without GPU hardware or the compiled
``amdsmi`` package.
"""

import os
import types
import unittest

try:
    from common.common import (
        amdsmi_path,
        cli_search_order,
        fake_module,
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

_STATUS_NOT_SUPPORTED = 2
_DF = "DF"

# MI300X FCLK DPM table, in Hz.
_FCLK_HZ = [1300_000_000, 1550_000_000, 1800_000_000]


class _FakeLibraryException(Exception):
    def __init__(self, err_code=_STATUS_NOT_SUPPORTED):
        super().__init__("mock error")
        self._err_code = err_code

    def get_error_code(self):
        return self._err_code


def _build_fake_amdsmi():
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")
    wrapper = types.ModuleType("amdsmi.amdsmi_wrapper")

    wrapper.AMDSMI_STATUS_NOT_SUPPORTED = _STATUS_NOT_SUPPORTED
    interface.amdsmi_wrapper = wrapper
    interface.AmdSmiClkType = types.SimpleNamespace(DF=_DF)
    exception.AmdSmiLibraryException = _FakeLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception
    return {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
        "amdsmi.amdsmi_wrapper": wrapper,
    }


class TestRestoreFclkCap(unittest.TestCase):
    """Unit tests for ``ResetCommands._restore_fclk_cap``."""

    @classmethod
    def setUpClass(cls):
        if not RESET_PATH or not os.path.isfile(RESET_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI reset.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        modules = _build_fake_amdsmi()
        modules["amdsmi_helpers"] = fake_module(
            "amdsmi_helpers", AMDSMIHelpers=type("AMDSMIHelpers", (), {})
        )
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]
        cls.module = load_cli_module("reset_under_test", RESET_PATH, sys_path_dir=_CLI_DIR)

    def _restore(self, levels_hz, max_clk):
        self.writes = []
        self.interface.amdsmi_get_clk_freq = lambda _gpu, clk: {"frequency": list(levels_hz)}
        self.interface.amdsmi_get_clock_info = lambda _gpu, clk: {"max_clk": max_clk}
        self.interface.amdsmi_set_gpu_clk_limit = lambda _gpu, clk, lim, val: self.writes.append(
            (clk, lim, val)
        )
        return self.module.ResetCommands._restore_fclk_cap(None)

    def test_lowered_cap_raised_to_top_level(self):
        self.assertTrue(self._restore(_FCLK_HZ, 1550))
        self.assertEqual(self.writes, [("fclk", "max", 1800)])

    def test_cap_at_top_level_not_written(self):
        self.assertFalse(self._restore(_FCLK_HZ, 1800))
        self.assertEqual(self.writes, [])

    def test_unreadable_cap_not_written(self):
        self.assertFalse(self._restore(_FCLK_HZ, "N/A"))
        self.assertEqual(self.writes, [])

    def test_empty_table_not_written(self):
        self.assertFalse(self._restore([], 1550))
        self.assertEqual(self.writes, [])

    def test_unsupported_clock_raises(self):
        # The caller treats NOT_SUPPORTED as "no FCLK table" and moves on.
        def _raise(_gpu, _clk):
            raise _FakeLibraryException(_STATUS_NOT_SUPPORTED)

        self.interface.amdsmi_get_clk_freq = _raise
        with self.assertRaises(_FakeLibraryException):
            self.module.ResetCommands._restore_fclk_cap(None)
