#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for the new CPU sections in ``amd-smi metric``.

These tests drive ``MetricCommands.metric_cpu`` with the C library, logger and
helpers stubbed, so they run without EPYC hardware or the compiled ``amdsmi``
package. They pin the ``metric.py`` display logic for the three flags added for
ROCm 10.2.0:

* ``--cpu-apb-status``      -> ``amdsmi_get_cpu_apb_status``
* ``--cpu-xgmi-link-width`` -> ``amdsmi_get_cpu_xgmi_width``
* ``--cpu-df-pstate-range`` -> ``amdsmi_get_cpu_df_pstate_range``

The behavior locked in here is the mapping from each getter's return dict to the
rendered section, including the APB ``apb_disabled`` -> ``ENABLED``/``DISABLED``
translation and the ``N/A`` fallback taken when a getter raises.
"""

import argparse
import os
import types
import unittest

from common.common import amdsmi_path, cli_search_order, find_cli_dir, load_cli_module, stub_modules

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CLI_DIR = find_cli_dir(*cli_search_order(_THIS_DIR))
METRIC_PATH = os.path.join(_CLI_DIR, "subcommands", "metric.py") if _CLI_DIR else None

# Every cpu_* arg metric_cpu reads when assembling curr_platform_cpu_values.
_CPU_ARGS = (
    "cpu_power_metrics",
    "cpu_prochot",
    "cpu_xgmi_link_width",
    "cpu_apb_status",
    "cpu_freq_metrics",
    "cpu_c0_res",
    "cpu_lclk_dpm_level",
    "cpu_pwr_svi_telemetry_rails",
    "cpu_io_bandwidth",
    "cpu_xgmi_bandwidth",
    "cpu_pwr_eff_mode",
    "cpu_df_pstate_range",
    "cpu_metrics_ver",
    "cpu_metrics_table",
    "cpu_socket_energy",
    "cpu_ddr_bandwidth",
    "cpu_temp",
    "cpu_dimm_temp_range_rate",
    "cpu_dimm_pow_consumption",
    "cpu_dimm_thermal_sensor",
    "cpu_xgmi_pstate_range",
    "cpu_railisofreq_policy",
    "cpu_dfcstate_ctrl",
    "cpu_pc6_enable",
    "cpu_cc6_enable",
    "cpu_dimm_sb_reg",
    "cpu_tdelta",
    "cpu_svi3_vr_controller_temp",
    "cpu_enabled_commands",
    "cpu_sdps_limit",
)


class _FakeAmdSmiException(Exception):
    """Stands in for ``AmdSmiException``."""


class _FakeLibraryException(_FakeAmdSmiException):
    def __init__(self, message="AMDSMI_STATUS_NOT_SUPPORTED"):
        super().__init__(message)
        self._message = message

    def get_error_info(self):
        return self._message


def _build_fake_amdsmi():
    """Build a stub ``amdsmi`` package so ``metric.py`` imports cleanly.

    Any ``amdsmi_*`` entry point a test does not stub explicitly raises
    ``_FakeLibraryException``, standing in for a sensor the host does not expose.
    """
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")

    interface.AMDSMI_MAX_RAIL_INDEX = 7
    interface.AmdSmiLibraryException = _FakeLibraryException

    def _unsupported(name):
        if name.startswith("amdsmi_"):

            def _raise(*_args, **_kwargs):
                raise _FakeLibraryException(f"{name}: AMDSMI_STATUS_NOT_SUPPORTED")

            return _raise
        raise AttributeError(name)

    interface.__getattr__ = _unsupported

    exception.AmdSmiException = _FakeAmdSmiException
    exception.AmdSmiLibraryException = _FakeLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception

    return {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
    }


def _load_metric_module():
    return load_cli_module("metric_cpu_under_test", METRIC_PATH, sys_path_dir=_CLI_DIR)


class _FakeLogger:
    """Captures the ``values`` payload that ``metric_cpu`` stores per CPU."""

    def __init__(self):
        self.captured_values = None
        self.store_cpu_json_output = []

    def is_json_format(self):
        return False

    def is_csv_format(self):
        return False

    def is_human_readable_format(self):
        return True

    def store_cpu_output(self, _cpu, key, value):
        if key == "values":
            self.captured_values = value

    def print_output(self, *args, **kwargs):
        pass

    def store_multiple_device_output(self, *args, **kwargs):
        pass


class _FakeHelpers:
    def handle_cpus(self, args, _logger, _func):
        # No multiple-device recursion; reuse the handle already on args.
        return False, args.cpu

    def get_cpu_id_from_device_handle(self, _handle):
        return 0

    def os_info(self):
        return "mock-os"


def _build_args(**overrides):
    """Namespace with every cpu_* arg off and a placeholder CPU handle."""
    defaults = {arg: None for arg in _CPU_ARGS}
    defaults["cpu"] = object()  # non-None so metric_cpu keeps the handle as-is
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCliMetricCpuFlags(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not METRIC_PATH or not os.path.isfile(METRIC_PATH):
            raise unittest.SkipTest(
                f"amd-smi CLI metric.py not found (looked in {_CLI_DIR or amdsmi_path})"
            )
        modules = _build_fake_amdsmi()
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]
        cls.metric_module = _load_metric_module()

    def _run_metric_cpu(self, **arg_overrides):
        commands = object.__new__(self.metric_module.MetricCommands)
        commands.logger = _FakeLogger()
        commands.helpers = _FakeHelpers()
        commands.cpu_handles = []

        commands.metric_cpu(_build_args(**arg_overrides))

        captured = commands.logger.captured_values
        self.assertIsNotNone(captured, "metric_cpu did not store a values payload")
        return captured

    def test_apb_status_disabled_reports_pstate(self):
        self.interface.amdsmi_get_cpu_apb_status = lambda _cpu: {"apb_disabled": 1, "pstate": 3}
        captured = self._run_metric_cpu(cpu_apb_status=True)
        self.assertEqual(captured["apb_status"], {"status": "DISABLED", "pstate": 3})

    def test_apb_status_enabled_hides_pstate(self):
        # apb_disabled == 0 means APB is enabled; the fixed P-state is then N/A.
        self.interface.amdsmi_get_cpu_apb_status = lambda _cpu: {"apb_disabled": 0, "pstate": 5}
        captured = self._run_metric_cpu(cpu_apb_status=True)
        self.assertEqual(captured["apb_status"], {"status": "ENABLED", "pstate": "N/A"})

    def test_apb_status_failure_reports_na(self):
        def _raise(_cpu):
            raise _FakeLibraryException()

        self.interface.amdsmi_get_cpu_apb_status = _raise
        captured = self._run_metric_cpu(cpu_apb_status=True)
        self.assertEqual(captured["apb_status"], {"status": "N/A", "pstate": "N/A"})

    def test_xgmi_link_width_reports_min_max(self):
        self.interface.amdsmi_get_cpu_xgmi_width = lambda _cpu: {"min_width": 2, "max_width": 16}
        captured = self._run_metric_cpu(cpu_xgmi_link_width=True)
        self.assertEqual(captured["xgmi_link_width"], {"min_width": 2, "max_width": 16})

    def test_xgmi_link_width_failure_reports_na(self):
        def _raise(_cpu):
            raise _FakeLibraryException()

        self.interface.amdsmi_get_cpu_xgmi_width = _raise
        captured = self._run_metric_cpu(cpu_xgmi_link_width=True)
        self.assertEqual(captured["xgmi_link_width"], {"min_width": "N/A", "max_width": "N/A"})

    def test_df_pstate_range_reports_min_max(self):
        self.interface.amdsmi_get_cpu_df_pstate_range = lambda _cpu: {
            "min_pstate": 0,
            "max_pstate": 2,
        }
        captured = self._run_metric_cpu(cpu_df_pstate_range=True)
        self.assertEqual(captured["df_pstate_range"], {"min_pstate": 0, "max_pstate": 2})

    def test_df_pstate_range_failure_reports_na(self):
        def _raise(_cpu):
            raise _FakeLibraryException()

        self.interface.amdsmi_get_cpu_df_pstate_range = _raise
        captured = self._run_metric_cpu(cpu_df_pstate_range=True)
        self.assertEqual(captured["df_pstate_range"], {"min_pstate": "N/A", "max_pstate": "N/A"})


if __name__ == "__main__":
    unittest.main()
