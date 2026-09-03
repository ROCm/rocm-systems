#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for APU section handling in ``amd-smi metric``.

These tests drive ``MetricCommands.metric_gpu`` with the C library, logger and
helpers stubbed, so they run without GPU hardware or the compiled ``amdsmi``
package. Every sensor entry point raises ``NOT_SUPPORTED`` and the metrics table
reports ``is_apu``, reproducing a Strix-class part.

The discrete-GPU sections are dropped from the default dump to keep it readable.
The behavior locked in here is that dropping them applies to the *default dump
only*: a section the user names explicitly reports ``N/A`` instead of printing
nothing, so ``--energy`` never exits 0 with an empty payload.
"""

import argparse
import importlib.util
import os
import sys
import types
import unittest

from common.common import amdsmi_path

# The amd-smi CLI ships alongside the amdsmi package: ``common`` resolves
# ``amdsmi_path`` to ``<rocm>/share/amd_smi`` and the CLI installs to the sibling
# ``<rocm>/libexec/amdsmi_cli``. ``setUpClass`` skips the suite if it is absent.
_ROCM_ROOT = os.path.dirname(os.path.dirname(amdsmi_path))
METRIC_PATH = os.path.join(_ROCM_ROOT, "libexec", "amdsmi_cli", "subcommands", "metric.py")

# Sections gated on APU parts, and the value each reports when named explicitly.
_SCALAR_NA_SECTIONS = ("ecc_blocks", "overdrive", "xgmi_err", "energy")
_DICT_NA_SECTIONS = ("pcie", "voltage_curve", "voltage")


class _FakeLibraryException(Exception):
    def __init__(self, message="AMDSMI_STATUS_NOT_SUPPORTED"):
        super().__init__(message)
        self._message = message

    def get_error_info(self):
        return self._message


class _ApuMetrics(dict):
    """gpu_metrics payload for an APU: ``is_apu`` set, every other field N/A."""

    def __missing__(self, _key):
        return "N/A"


class _EnumMeta(type):
    """Resolves any member access to the member name, mimicking an enum."""

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return f"{cls.__name__}.{name}"

    def __getitem__(cls, name):
        return f"{cls.__name__}.{name}"


class _FakeEnum(metaclass=_EnumMeta):
    pass


def _install_fake_amdsmi():
    """Register a stub ``amdsmi`` package so ``metric.py`` imports cleanly.

    Any ``amdsmi_*`` entry point that a test does not stub explicitly raises
    ``_FakeLibraryException``, standing in for a sensor the APU does not expose.
    """
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")

    interface.AMDSMI_MAX_NUM_GFX_CLKS = 8
    interface.AMDSMI_MAX_NUM_CLKS = 4
    interface.AMDSMI_MAX_RAIL_INDEX = 7
    interface.AMDSMI_NUM_VOLTAGE_CURVE_POINTS = 3
    interface.amdsmi_get_gpu_metrics_info = lambda _handle: _ApuMetrics(is_apu=True)
    interface._NA_amdsmi_get_gpu_metrics_info = lambda: _ApuMetrics(is_apu=True)
    interface.amdsmi_get_gpu_partition_metrics_info = lambda _handle: None
    # Activity succeeds on a real APU, so the usage section is a dict the
    # APU-specific fields can be merged into.
    interface.amdsmi_get_gpu_activity = lambda _handle: {
        "gfx_activity": "N/A",
        "umc_activity": "N/A",
        "mm_activity": "N/A",
    }
    interface.AmdSmiLibraryException = _FakeLibraryException

    def _unsupported(name):
        """Resolve any interface attribute the stub does not define explicitly.

        ``amdsmi_*`` entry points raise NOT_SUPPORTED so every sensor looks
        absent; ``AmdSmi*`` enum types resolve members to their own name.
        """
        if name.startswith("AmdSmi"):
            return _FakeEnum
        if name.startswith("amdsmi_"):

            def _raise(*_args, **_kwargs):
                raise _FakeLibraryException(f"{name}: AMDSMI_STATUS_NOT_SUPPORTED")

            return _raise
        raise AttributeError(name)

    interface.__getattr__ = _unsupported

    exception.AmdSmiLibraryException = _FakeLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception

    sys.modules["amdsmi"] = amdsmi_pkg
    sys.modules["amdsmi.amdsmi_interface"] = interface
    sys.modules["amdsmi.amdsmi_exception"] = exception
    return interface


def _load_metric_module():
    spec = importlib.util.spec_from_file_location("metric_apu_under_test", METRIC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLogger:
    """Captures the ``values`` payload that ``metric_gpu`` stores per GPU."""

    def __init__(self):
        self.captured_values = None
        self.store_gpu_json_output = []

    def is_json_format(self):
        return False

    def is_csv_format(self):
        return False

    def is_human_readable_format(self):
        return True

    def store_output(self, _gpu, key, value):
        if key == "values":
            self.captured_values = value

    def print_output(self, *args, **kwargs):
        pass

    def store_multiple_device_output(self):
        pass

    def store_watch_output(self, *args, **kwargs):
        pass


class _FakeHelpers:
    def is_hypervisor(self):
        return False

    def is_windows(self):
        return False

    def is_baremetal(self):
        return True

    def is_linux(self):
        return True

    def check_required_groups(self):
        pass

    def get_gpu_id_from_device_handle(self, _handle):
        return 0

    def os_info(self):
        return "mock-os"

    def _get_metric_version_and_partition_info(self, *args, **kwargs):
        return {"num_partition": 1}

    def get_gpu_board_temperatures(self, *args, **kwargs):
        return {}

    def get_base_board_temperatures(self, *args, **kwargs):
        return {}

    def build_xcp_dict(self, *args, **kwargs):
        return {}

    def unit_format(self, logger, value, unit):
        if isinstance(value, list):
            return [self.unit_format(logger, v, unit) for v in value]
        if value == "N/A":
            return "N/A"
        if unit:
            return f"{value} {unit}".rstrip()
        return f"{value}".rstrip()


def _build_args(**overrides):
    """Namespace with every attribute ``metric_gpu`` touches, all sections off.

    Leaving every section ``False`` reproduces a bare ``amd-smi metric``: the
    command then turns each platform-applicable arg on itself. Passing one
    section ``True`` reproduces an explicitly named section.
    """
    defaults = dict(
        gpu=object(),  # non-None, non-list placeholder device handle
        watch=False,
        watch_time=None,
        iterations=None,
        loglevel="INFO",
        partition=False,
        clock=False,
        usage=False,
        power=False,
        temperature=False,
        voltage=False,
        pcie=False,
        ecc=False,
        ecc_blocks=False,
        base_board=False,
        gpu_board=False,
        mem_usage=False,
        fan=False,
        voltage_curve=False,
        overdrive=False,
        perf_level=False,
        xgmi_err=False,
        energy=False,
        throttle=False,
        violation=False,
        schedule=False,
        guard=False,
        guest_data=False,
        fb_usage=False,
        xgmi=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCliMetricApuSections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(METRIC_PATH):
            raise unittest.SkipTest(f"amd-smi CLI metric.py not found at {METRIC_PATH}")
        cls.interface = _install_fake_amdsmi()
        cls.metric_module = _load_metric_module()

    def _run_metric(self, metrics=None, **arg_overrides):
        """Drive ``metric_gpu`` and return the stored ``values`` payload.

        ``metrics`` seeds extra ``gpu_metrics`` fields for the run; anything not
        named there resolves to N/A.
        """
        payload = _ApuMetrics(is_apu=True)
        if metrics:
            payload.update(metrics)
        self.interface.amdsmi_get_gpu_metrics_info = lambda _handle: payload

        commands = object.__new__(self.metric_module.MetricCommands)
        commands.logger = _FakeLogger()
        commands.helpers = _FakeHelpers()
        commands.group_check_printed = True
        commands.device_handles = []

        commands.metric_gpu(_build_args(**arg_overrides))

        captured = commands.logger.captured_values
        self.assertIsNotNone(captured, "metric_gpu did not store a values payload")
        return captured

    def test_explicit_section_reports_na_on_apu(self):
        # An explicitly named section must report N/A; emitting nothing leaves
        # `amd-smi metric --energy` printing only the GPU header and exiting 0.
        for section in _SCALAR_NA_SECTIONS:
            with self.subTest(section=section):
                captured = self._run_metric(**{section: True})
                self.assertIn(section, captured)
                self.assertEqual(captured[section], "N/A")

    def test_explicit_overdrive_reports_both_levels(self):
        captured = self._run_metric(overdrive=True)
        self.assertEqual(captured["overdrive"], "N/A")
        self.assertEqual(captured["mem_overdrive"], "N/A")

    def test_explicit_dict_sections_report_na_on_apu(self):
        # pcie, voltage_curve, and voltage carry per-field dicts rather than a scalar.
        pcie = self._run_metric(pcie=True)["pcie"]
        self.assertEqual(pcie["width"], "N/A")
        self.assertEqual(pcie["speed"], "N/A")

        curve = self._run_metric(voltage_curve=True)["voltage_curve"]
        self.assertEqual(curve["point_0_frequency"], "N/A")
        self.assertEqual(curve["point_0_voltage"], "N/A")

        voltage = self._run_metric(voltage=True)["voltage"]
        self.assertEqual(voltage["vddboard"], "N/A")

    def test_explicit_fan_reports_na_when_apu_pwm_absent(self):
        # On an APU the fan section is served by apu_metrics.fan_pwm; when that
        # field is absent the section still has to report N/A.
        captured = self._run_metric(fan=True)
        self.assertEqual(captured["fan"], {"apu_fan_pwm": "N/A"})

    def test_default_dump_omits_unsupported_sections_on_apu(self):
        # Bare `amd-smi metric` keeps the unsupported sections out so the dump
        # is not padded with N/A blocks.
        captured = self._run_metric()
        for section in _SCALAR_NA_SECTIONS + _DICT_NA_SECTIONS + ("fan",):
            with self.subTest(section=section):
                self.assertNotIn(section, captured)

    def test_apu_bandwidth_counters_report_mb_per_second(self):
        # DRAM and IPU read/write counters are both bandwidth in MB/s; the IPU
        # pair previously printed bare numbers while DRAM carried the unit.
        usage = self._run_metric(
            metrics={
                "apu_metrics.average_dram_reads": 428,
                "apu_metrics.average_dram_writes": 57,
                "apu_metrics.average_ipu_reads": 12,
                "apu_metrics.average_ipu_writes": 4,
            },
            usage=True,
        )["usage"]

        self.assertEqual(usage["apu_average_dram_reads"], "428 MB/s")
        self.assertEqual(usage["apu_average_dram_writes"], "57 MB/s")
        self.assertEqual(usage["apu_average_ipu_reads"], "12 MB/s")
        self.assertEqual(usage["apu_average_ipu_writes"], "4 MB/s")

    def test_zero_bandwidth_still_carries_unit(self):
        # 0 is a real reading, not an absent one, so it keeps its unit.
        usage = self._run_metric(metrics={"apu_metrics.average_ipu_reads": 0}, usage=True)["usage"]

        self.assertEqual(usage["apu_average_ipu_reads"], "0 MB/s")


if __name__ == "__main__":
    unittest.main()
