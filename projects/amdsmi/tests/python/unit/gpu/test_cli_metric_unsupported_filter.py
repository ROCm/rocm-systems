#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end output-shape tests for ``amd-smi metric`` unsupported-field filtering.

Drives ``MetricCommands.metric_gpu`` with the C library, logger and helpers
stubbed, so the whole path runs without GPU hardware: the header read, the
suppression set, the all-N/A gate that decides what reaches the logger, and the
rule that a section named on the command line is never emptied. The helper-level
tests in ``test_cli_metrics_field_support`` exercise those pieces in isolation
and cannot see a regression in how they are wired together, so every assertion
here compares default output against ``--show-unsupported`` output.

Filtering applies to human-readable output only. ``--json`` and ``--csv`` are
the machine-readable contract, so every class below pairs its human-format
assertions with the claim that the two machine formats are byte-for-byte what
``--show-unsupported`` produces, which is also what the feature replaced.

The stubbing pattern is the one in ``test_cli_metric_partition`` and
``test_vcn_busy_navi``. The one deliberate difference is that ``metric.py`` is
taken from the source checkout when there is one: the behavior itself is under
test, and an installed CLI predating it would make every assertion here vacuous.
"""

import argparse
import contextlib
import copy
import csv
import importlib.util
import io
import logging
import os
import sys
import types
import unittest

try:
    from common.common import amdsmi_path
except (ImportError, FileNotFoundError):  # pragma: no cover - harness/install unavailable
    amdsmi_path = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_CLI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "amdsmi_cli"))
_INSTALLED_CLI_DIR = (
    os.path.join(os.path.dirname(os.path.dirname(amdsmi_path)), "libexec", "amdsmi_cli")
    if amdsmi_path
    else ""
)


def _resolve_cli_dir():
    for cli_dir in (_SOURCE_CLI_DIR, _INSTALLED_CLI_DIR):
        if cli_dir and os.path.isfile(os.path.join(cli_dir, "subcommands", "metric.py")):
            return cli_dir
    return ""


CLI_DIR = _resolve_cli_dir()
METRIC_PATH = os.path.join(CLI_DIR, "subcommands", "metric.py") if CLI_DIR else ""

# Inside a source checkout metric.py must exist, so a miss is a failure rather
# than a skip. Only an installed CLI that predates the feature is skipped.
_RUNNABLE = bool(METRIC_PATH) or os.path.isdir(_SOURCE_CLI_DIR)
_SKIP_REASON = "amd-smi CLI metric.py not found"

NA = "N/A"


class _FakeClkType:
    GFX = "GFX"
    MEM = "MEM"
    VCLK0 = "VCLK0"
    DCLK0 = "DCLK0"
    SOC = "SOC"
    DF = "DF"


class _FakeTemperatureType:
    EDGE = "EDGE"
    HOTSPOT = "HOTSPOT"
    VRAM = "VRAM"


class _FakeTemperatureMetric:
    CURRENT = "CURRENT"
    CRITICAL = "CRITICAL"


class _FakeException(Exception):
    """Stands in for ``AmdSmiException``, the base of the two below."""


class _FakeLibraryException(_FakeException):
    def __init__(self, message="mock error"):
        super().__init__(message)
        self._message = message

    def get_error_info(self):
        return self._message


class _FakeParameterException(_FakeException):
    """Sibling of the library exception, and like the real one has no error info."""


def _raise_lib_exc(*_args, **_kwargs):
    raise _FakeLibraryException("mock error")


def _raise_param_exc(*_args, **_kwargs):
    raise _FakeParameterException("mock bad parameter")


class _FakeNamedValues:
    """Stands in for an interface enum: any member resolves to its own name."""

    def __getattr__(self, name):
        return types.SimpleNamespace(name=name, value=name)

    def __getitem__(self, name):
        return getattr(self, name)


class _FakeInterfaceModule(types.ModuleType):
    """Any ``amdsmi_*`` call not stubbed below behaves as an unsupported call.

    Plain ``amd-smi metric`` turns on every section, most of which these tests do
    not care about. Failing their reads is both the shortest stub and a faithful
    one: the CLI already prints "N/A" for a library call it cannot make.
    """

    def __getattr__(self, name):
        if name.startswith("amdsmi_"):
            return _raise_lib_exc
        raise AttributeError(name)


# Throttle rows whose residency accumulators report real numbers, and rows that
# report nothing. The second group is what a metrics version can make impossible.
_LIVE_THROTTLE_ROWS = ("prochot_thrm", "ppt_pwr", "socket_thrm", "vr_thrm", "hbm_thrm")
_EMPTY_THROTTLE_ROWS = (
    "gfx_clk_below_host_limit",
    "gfx_clk_below_host_limit_pwr",
    "gfx_clk_below_host_limit_thm",
    "gfx_clk_below_host_limit_total",
    "low_utilization",
)


def _violation_status(*_args, **_kwargs):
    # build_xcp_dict() mutates what it is handed, so hand it a fresh dict.
    status = {"acc_counter": 1000}
    for row in _LIVE_THROTTLE_ROWS:
        status[f"acc_{row}"] = 5
        status[f"active_{row}"] = False
        status[f"per_{row}"] = 1
    for row in _EMPTY_THROTTLE_ROWS:
        status[f"acc_{row}"] = NA
        status[f"active_{row}"] = NA
        status[f"per_{row}"] = NA
    return status


def _gpu_metrics(*_args, **_kwargs):
    return {
        "vcn_activity": NA,
        "jpeg_activity": NA,
        "temperature_hbm_stacks": NA,
        "temperature_mid": NA,
        "temperature_aid": NA,
        "xcp_stats.temperature_xcd": NA,
    }


def _pcie_info(*_args, **_kwargs):
    return {
        "pcie_metric": {
            "pcie_width": 16,
            "pcie_speed": 32000,
            "pcie_bandwidth": NA,
            "pcie_replay_count": 3,
            "pcie_l0_to_recovery_count": NA,
            "pcie_replay_roll_over_count": NA,
            "pcie_nak_received_count": NA,
            "pcie_nak_sent_count": NA,
            "pcie_lc_perf_other_end_recovery_count": NA,
        }
    }


def _install_fake_amdsmi():
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = _FakeInterfaceModule("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")

    interface.AMDSMI_MAX_NUM_GFX_CLKS = 8
    interface.AMDSMI_MAX_NUM_CLKS = 4
    interface.AMDSMI_MAX_RAIL_INDEX = 7
    interface.AMDSMI_NUM_VOLTAGE_CURVE_POINTS = 3
    interface.AmdSmiClkType = _FakeClkType
    interface.AmdSmiTemperatureType = _FakeTemperatureType
    interface.AmdSmiTemperatureMetric = _FakeTemperatureMetric
    interface.AmdSmiVoltageType = _FakeNamedValues()
    interface.AmdSmiVoltageMetric = _FakeNamedValues()
    interface.AmdSmiMemoryType = _FakeNamedValues()
    interface.AmdSmiRasErrState = _FakeNamedValues()
    interface.AmdSmiGpuBlock = _FakeNamedValues()
    interface.AmdSmiLibraryException = _FakeLibraryException

    interface.amdsmi_get_gpu_metrics_info = _gpu_metrics
    interface._NA_amdsmi_get_gpu_metrics_info = lambda: {}
    interface.amdsmi_get_gpu_partition_metrics_info = lambda _h: None
    interface.amdsmi_get_gpu_activity = lambda _h: {
        "gfx_activity": 30,
        "umc_activity": 12,
        "mm_activity": NA,
    }
    # Navi-style sysfs fallback is unavailable, so usage.vcn_busy reads "N/A".
    # It is unmapped, which makes it the control for "all-N/A but never dropped".
    interface.amdsmi_get_vcn_busy_percent = _raise_lib_exc
    interface.amdsmi_get_pcie_info = _pcie_info
    interface.amdsmi_get_gpu_pci_throughput = _raise_lib_exc
    interface.amdsmi_get_temp_metric = lambda _h, _type, _metric: 55
    interface.amdsmi_get_violation_status = _violation_status
    # Replaced per test with the version under test.
    interface.amdsmi_get_gpu_metrics_header_info = lambda _h: {}

    exception.AmdSmiException = _FakeException
    exception.AmdSmiLibraryException = _FakeLibraryException
    exception.AmdSmiParameterException = _FakeParameterException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception

    sys.modules["amdsmi"] = amdsmi_pkg
    sys.modules["amdsmi.amdsmi_interface"] = interface
    sys.modules["amdsmi.amdsmi_exception"] = exception
    return interface


def _load_metric_module():
    if not METRIC_PATH:
        raise AssertionError(f"metric.py not found in {_SOURCE_CLI_DIR!r} or an installed CLI")
    # metric.py imports its CLI-level siblings by bare name.
    if CLI_DIR not in sys.path:
        sys.path.insert(0, CLI_DIR)
    spec = importlib.util.spec_from_file_location("metric_under_test_unsupported", METRIC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_logger_module():
    """The real ``AMDSMILogger``, with only its helpers import replaced.

    The CSV column behavior asserted below belongs to the logger, so a
    hand-rolled stand-in would prove nothing. ``amdsmi_helpers`` pulls in
    ``amdsmi_init``, which talks to the driver, so that one import is stubbed.
    """
    if "amdsmi_helpers" not in sys.modules:
        helpers_module = types.ModuleType("amdsmi_helpers")
        helpers_module.AMDSMIHelpers = type("AMDSMIHelpers", (), {})
        sys.modules["amdsmi_helpers"] = helpers_module
    spec = importlib.util.spec_from_file_location(
        "amdsmi_logger_under_test_unsupported", os.path.join(CLI_DIR, "amdsmi_logger.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLogger:
    """Captures the ``values`` payload per device handle, in one output format."""

    FORMATS = ("human", "json", "csv")
    MACHINE_FORMATS = ("json", "csv")

    def __init__(self, output_format="human"):
        self.output_format = output_format
        self.captured = {}
        self.store_gpu_json_output = []

    def is_json_format(self):
        return self.output_format == "json"

    def is_csv_format(self):
        return self.output_format == "csv"

    def is_human_readable_format(self):
        return self.output_format == "human"

    def store_output(self, gpu, key, value):
        if key == "values":
            self.captured[id(gpu)] = value

    def print_output(self, *args, **kwargs):
        pass

    def store_multiple_device_output(self):
        pass

    def store_watch_output(self, *args, **kwargs):
        pass


class _FakeHelpers:
    def __init__(self, gpu_ids=None):
        self._gpu_ids = gpu_ids or {}

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

    def get_gpu_id_from_device_handle(self, handle):
        return self._gpu_ids.get(id(handle), 0)

    def os_info(self):
        return "mock-os"

    def _get_metric_version_and_partition_info(self, *args, **kwargs):
        return {"num_partition": NA}

    # Board temperatures come from a helper rather than the metrics blobs, and
    # an empty dict is how metric.py already spells "nothing to show".
    def get_gpu_board_temperatures(self, *_args, **_kwargs):
        return {}

    def get_base_board_temperatures(self, *_args, **_kwargs):
        return {}

    def build_xcp_dict(self, key, violation_status, num_partition):
        value = violation_status[key]
        if isinstance(value, list):
            return {f"xcp_{index}": value[index] for index in range(num_partition)}
        if "active_" in key and value != NA:
            return "ACTIVE" if value is True else "NOT ACTIVE"
        return value

    def unit_format(self, logger, value, unit):
        if isinstance(value, list):
            return [self.unit_format(logger, item, unit) for item in value]
        if value == NA:
            return NA
        if logger.is_json_format():
            return {"value": value, "unit": unit} if unit else value
        if unit:
            return f"{value} {unit}".rstrip()
        return f"{value}".rstrip()


# Every section flag `metric` registers on baremetal Linux, in the order
# metric_gpu builds them.
_SECTION_ARGS = (
    "mem_usage",
    "usage",
    "power",
    "clock",
    "temperature",
    "voltage",
    "pcie",
    "ecc",
    "ecc_blocks",
    "base_board",
    "gpu_board",
    "fan",
    "voltage_curve",
    "overdrive",
    "perf_level",
    "xgmi_err",
    "energy",
    "throttle",
)

# The metrics-backed sections, named explicitly, which is the common case here.
_DEFAULT_SECTIONS = ("usage", "pcie", "temperature", "throttle")


def _build_args(sections=_DEFAULT_SECTIONS, **overrides):
    """Namespace mirroring argparse on baremetal Linux.

    ``sections`` is what the user named; an empty tuple is plain ``amd-smi
    metric``, which metric_gpu then expands to every section.
    """
    defaults = dict(
        gpu=None,
        watch=False,
        watch_time=None,
        iterations=None,
        loglevel="INFO",
        show_unsupported=False,
        partition=False,
    )
    defaults.update({arg: arg in sections for arg in _SECTION_ARGS})
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _header(version):
    return {"structure_size": 512, "format_revision": version[0], "content_revision": version[1]}


def _key_paths(payload, prefix=""):
    """Every dotted key path in a values payload, for key-set comparisons."""
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        yield path
        if isinstance(value, dict):
            yield from _key_paths(value, path)


@unittest.skipUnless(_RUNNABLE, _SKIP_REASON)
class _MetricGpuHarness(unittest.TestCase):
    """Runs ``metric_gpu`` against the stubs and hands back the stored payload."""

    @classmethod
    def setUpClass(cls):
        cls.interface = _install_fake_amdsmi()
        cls.metric_module = _load_metric_module()

    def setUp(self):
        self.interface.amdsmi_get_violation_status = _violation_status

    def run_metric(
        self,
        *,
        show_unsupported=False,
        version=(1, 6),
        output="human",
        reader=None,
        sections=_DEFAULT_SECTIONS,
        loglevel="INFO",
        partition=False,
    ):
        """Drive one GPU and return the ``values`` payload the logger received."""
        self.header_calls = []

        def default_reader(handle):
            self.header_calls.append(handle)
            return _header(version)

        self.interface.amdsmi_get_gpu_metrics_header_info = reader or default_reader

        handle = object()
        commands = object.__new__(self.metric_module.MetricCommands)
        commands.logger = _FakeLogger(output)
        commands.helpers = _FakeHelpers()
        commands.group_check_printed = True
        commands.device_handles = []
        commands.metric_gpu(
            _build_args(
                sections=sections,
                gpu=handle,
                show_unsupported=show_unsupported,
                loglevel=loglevel,
                partition=partition,
            )
        )

        captured = commands.logger.captured.get(id(handle))
        self.assertIsNotNone(captured, "metric_gpu did not store a values payload")
        return captured

    def assertKeyPresent(self, values, section, key):
        self.assertIn(section, values, f"section {section} missing")
        self.assertIn(key, values[section], f"{section}.{key} was dropped")

    def assertKeyAbsent(self, values, section, key):
        self.assertNotIn(key, values.get(section, {}), f"{section}.{key} was kept")

    def assertMachineFormatsUnfiltered(self, *, version, sections=_DEFAULT_SECTIONS):
        """``--json`` and ``--csv`` must equal what ``--show-unsupported`` produces.

        Compared as whole payloads rather than key sets, so a machine format
        cannot start filtering by blanking a value instead of dropping a key.
        """
        for output in _FakeLogger.MACHINE_FORMATS:
            with self.subTest(output=output):
                unfiltered = self.run_metric(
                    show_unsupported=True, version=version, output=output, sections=sections
                )
                default = self.run_metric(version=version, output=output, sections=sections)
                self.assertEqual(default, unfiltered, f"--{output} was filtered")


class TestMachineFormatsAreNeverFiltered(_MetricGpuHarness):
    """``--json`` and ``--csv`` are a machine-readable contract and never filter.

    Scripts index fixed keys and columns, so the key set has to stay complete
    whatever the gpu_metrics version reports. These fail if the format gate in
    ``metric_gpu`` is ever dropped.
    """

    VERSIONS = ((1, 3), (1, 6), (1, 7), (1, 9), (9, 9))

    def test_machine_output_equals_show_unsupported_on_every_version(self):
        for version in self.VERSIONS:
            with self.subTest(version=version):
                self.assertMachineFormatsUnfiltered(version=version)

    def test_machine_output_does_not_depend_on_the_version(self):
        for output in _FakeLogger.MACHINE_FORMATS:
            with self.subTest(output=output):
                on_v13 = self.run_metric(version=(1, 3), output=output)
                on_v16 = self.run_metric(version=(1, 6), output=output)
                self.assertEqual(on_v13, on_v16)

    def test_machine_output_never_reads_the_header(self):
        # Nothing can consume it there, so metric_gpu must not pay for the read.
        for output in _FakeLogger.MACHINE_FORMATS:
            with self.subTest(output=output):
                self.run_metric(version=(1, 3), output=output)
                self.assertEqual(self.header_calls, [])

    def test_machine_output_still_reads_the_header_under_debug(self):
        for output in _FakeLogger.FORMATS:
            with self.subTest(output=output):
                with self.assertLogs(level=logging.DEBUG):
                    self.run_metric(
                        version=(1, 3), output=output, show_unsupported=True, loglevel="DEBUG"
                    )
                self.assertNotEqual(self.header_calls, [], "DEBUG lost the version log")

    def test_the_flag_is_a_no_op_in_machine_formats(self):
        for output in _FakeLogger.MACHINE_FORMATS:
            with self.subTest(output=output):
                values = self.run_metric(version=(1, 3), output=output, sections=())
                with_flag = self.run_metric(
                    version=(1, 3), output=output, sections=(), show_unsupported=True
                )
                self.assertEqual(values, with_flag)

    def test_a_section_the_version_empties_survives_in_machine_formats(self):
        # v1.3 can populate no throttle row, so human output drops the section.
        for output in _FakeLogger.MACHINE_FORMATS:
            with self.subTest(output=output):
                values = self.run_metric(version=(1, 3), output=output, sections=())
                self.assertIn("throttle", values)


class TestShowUnsupportedRestoresTheOldOutput(_MetricGpuHarness):
    def test_show_unsupported_never_reads_the_header(self):
        # The opt-out is total: with the flag, and outside DEBUG logging,
        # metric_gpu does not even query the version.
        for output in _FakeLogger.FORMATS:
            with self.subTest(output=output):
                self.run_metric(show_unsupported=True, version=(1, 6), output=output)
                self.assertEqual(self.header_calls, [])

    def test_show_unsupported_output_does_not_depend_on_the_version(self):
        for output in _FakeLogger.FORMATS:
            with self.subTest(output=output):
                on_v13 = self.run_metric(show_unsupported=True, version=(1, 3), output=output)
                on_v16 = self.run_metric(show_unsupported=True, version=(1, 6), output=output)
                self.assertEqual(on_v13, on_v16)

    def test_show_unsupported_keeps_every_key_the_default_drops(self):
        kept = self.run_metric(show_unsupported=True, version=(1, 3), output="human")
        dropped = self.run_metric(show_unsupported=False, version=(1, 3), output="human")
        self.assertNotEqual(kept, dropped, "the default suppressed nothing at all")
        for section, section_value in dropped.items():
            if isinstance(section_value, dict):
                self.assertLessEqual(set(section_value), set(kept[section]))
        self.assertLessEqual(set(dropped), set(kept))


class TestDefaultFilteringOnKnownVersion(_MetricGpuHarness):
    def test_v13_drops_only_the_impossible_all_na_keys(self):
        values = self.run_metric(version=(1, 3), output="human")

        # v1.3 has no hbm_stacks/mid/aid array and no XCD sensor.
        for key in ("hbm_stacks", "mid", "aid", "xcd"):
            self.assertKeyAbsent(values, "temperature", key)
        # The hwmon-backed sensors carry readings and are never mapped.
        for key in ("edge", "hotspot", "mem"):
            self.assertKeyPresent(values, "temperature", key)

        # v1.3 predates the instantaneous PCIe counters.
        for key in (
            "bandwidth",
            "l0_to_recovery_count",
            "replay_roll_over_count",
            "nak_sent_count",
            "nak_received_count",
            "lc_perf_other_end_recovery_count",
        ):
            self.assertKeyAbsent(values, "pcie", key)
        for key in ("width", "speed", "replay_count"):
            self.assertKeyPresent(values, "pcie", key)

        # v1.3 carries mm_activity and vcn_activity but not jpeg_activity.
        self.assertKeyAbsent(values, "usage", "jpeg_activity")
        for key in ("mm_activity", "vcn_activity"):
            self.assertKeyPresent(values, "usage", key)

        self.assertMachineFormatsUnfiltered(version=(1, 3))

    def test_v16_drops_the_legacy_keys_and_keeps_the_new_ones(self):
        values = self.run_metric(version=(1, 6), output="human")

        # v1.6 dropped these three from the table outright.
        for key in ("mm_activity", "vcn_activity", "jpeg_activity"):
            self.assertKeyAbsent(values, "usage", key)
        # ...and added the XCP busy arrays, so those stay even at "N/A".
        for key in ("gfx_busy_inst", "jpeg_busy"):
            self.assertKeyPresent(values, "usage", key)

        self.assertMachineFormatsUnfiltered(version=(1, 6))

    def test_a_suppressed_path_holding_a_reading_is_still_printed(self):
        # The all-N/A gate, end to end: v1.3 has none of the residency
        # accumulators, yet these rows report numbers and must survive.
        for output in _FakeLogger.FORMATS:
            with self.subTest(output=output):
                values = self.run_metric(version=(1, 3), output=output)
                self.assertKeyPresent(values, "throttle", "accumulation_counter")
                for row in ("prochot", "ppt", "socket_thermal", "vr_thermal", "hbm_thermal"):
                    for form in ("accumulated", "violation_status", "violation_activity"):
                        self.assertKeyPresent(values, "throttle", f"{row}_{form}")

    def test_an_unmapped_all_na_key_is_never_dropped(self):
        for output in _FakeLogger.FORMATS:
            with self.subTest(output=output):
                for version in ((1, 3), (1, 6), (1, 9)):
                    values = self.run_metric(version=version, output=output)
                    self.assertEqual(values["usage"]["vcn_busy"], NA)


class TestDerivedThrottleActivityRows(_MetricGpuHarness):
    """Derived throttle-activity rows: a percentage needs both its accumulator and the counter."""

    _GFX_LIMIT_ROWS = (
        "gfx_clk_below_host_limit",
        "gfx_clk_below_host_limit_power",
        "gfx_clk_below_host_limit_thermal",
        "total_gfx_clk_below_host_limit",
        "low_utilization",
    )
    # v1.6 carries accumulation_counter but none of the gfx-limit accumulators.
    # v1.7 adds the deprecated bare one back, so only that row survives.
    SUPPRESSED_ROWS = {
        (1, 6): frozenset(_GFX_LIMIT_ROWS),
        (1, 7): frozenset(_GFX_LIMIT_ROWS) - {"gfx_clk_below_host_limit"},
    }

    def test_every_form_of_an_impossible_row_is_dropped(self):
        for version, rows in sorted(self.SUPPRESSED_ROWS.items()):
            with self.subTest(version=version):
                values = self.run_metric(version=version, output="human")
                for row in self._GFX_LIMIT_ROWS:
                    for form in ("accumulated", "violation_status", "violation_activity"):
                        key = f"{row}_{form}"
                        if row in rows:
                            self.assertKeyAbsent(values, "throttle", key)
                        else:
                            self.assertKeyPresent(values, "throttle", key)

    def test_machine_formats_keep_every_form_of_an_impossible_row(self):
        for version in sorted(self.SUPPRESSED_ROWS):
            with self.subTest(version=version):
                self.assertMachineFormatsUnfiltered(version=version)
                for output in _FakeLogger.MACHINE_FORMATS:
                    values = self.run_metric(version=version, output=output)
                    for row in self._GFX_LIMIT_ROWS:
                        for form in ("accumulated", "violation_status", "violation_activity"):
                            self.assertKeyPresent(values, "throttle", f"{row}_{form}")


class TestSuppressionDegradesSafely(_MetricGpuHarness):
    def test_unknown_version_suppresses_nothing(self):
        for output in _FakeLogger.FORMATS:
            with self.subTest(output=output):
                baseline = self.run_metric(show_unsupported=True, output=output)
                values = self.run_metric(version=(9, 9), output=output)
                self.assertEqual(values, baseline)

    def test_unreadable_header_suppresses_nothing(self):
        # AmdSmiParameterException is a sibling of AmdSmiLibraryException, so
        # catching only the latter lets a bad handle escape to the user.
        for raiser in (_raise_lib_exc, _raise_param_exc):
            for output in _FakeLogger.FORMATS:
                with self.subTest(raiser=raiser.__name__, output=output):
                    baseline = self.run_metric(show_unsupported=True, output=output)
                    with self.assertLogs(level=logging.DEBUG) as captured_logs:
                        values = self.run_metric(output=output, reader=raiser)
                    self.assertEqual(values, baseline)
                    self.assertEqual(
                        [record for record in captured_logs.records if record.exc_info],
                        [],
                        "the failed header read produced a traceback",
                    )


class TestExplicitSectionIsNeverEmptied(_MetricGpuHarness):
    """A section the user named by flag must never come back empty.

    On v1.3 every throttle row is version-absent, and with the violation API
    unavailable every one of them reads "N/A", so filtering would leave nothing
    at all. That is the wall-of-N/A case the feature exists for when the user
    asked for everything, and an unacceptable answer when they asked for this.
    """

    def setUp(self):
        super().setUp()
        self.interface.amdsmi_get_violation_status = _raise_lib_exc

    def test_plain_metric_drops_a_section_filtering_empties(self):
        values = self.run_metric(version=(1, 3), output="human", sections=())
        self.assertNotIn("throttle", values)
        self.assertMachineFormatsUnfiltered(version=(1, 3), sections=())

    def test_a_named_section_survives_whole(self):
        for output in _FakeLogger.FORMATS:
            with self.subTest(output=output):
                asked = self.run_metric(version=(1, 3), output=output, sections=("throttle",))
                unfiltered = self.run_metric(
                    version=(1, 3), output=output, sections=("throttle",), show_unsupported=True
                )
                self.assertTrue(asked.get("throttle"), "--throttle answered with nothing")
                self.assertEqual(asked["throttle"], unfiltered["throttle"])

    def test_a_named_section_that_is_only_partly_suppressed_is_still_filtered(self):
        values = self.run_metric(version=(1, 6), output="human", sections=("usage",))
        for key in ("mm_activity", "vcn_activity", "jpeg_activity"):
            self.assertKeyAbsent(values, "usage", key)
        self.assertKeyPresent(values, "usage", "gfx_busy_inst")
        self.assertMachineFormatsUnfiltered(version=(1, 6), sections=("usage",))

    def test_partition_is_not_a_section_selector(self):
        """``--partition`` scopes the data, it does not name a section.

        It contributes nothing to the protected set, so the invocation filters
        exactly like plain ``amd-smi metric`` and drops the emptied section
        rather than protecting it the way a real section flag would.
        """
        plain = self.run_metric(version=(1, 3), output="human", sections=())
        with_partition = self.run_metric(
            version=(1, 3), output="human", sections=(), partition=True
        )
        self.assertEqual(with_partition, plain)
        self.assertNotIn("throttle", with_partition)
        # Contrast: a real section flag protects the section it names.
        named = self.run_metric(version=(1, 3), output="human", sections=("throttle",))
        self.assertTrue(named.get("throttle"))


@unittest.skipUnless(_RUNNABLE, _SKIP_REASON)
class TestPerGpuSuppression(unittest.TestCase):
    """Two GPUs on different metrics versions each get their own suppression set."""

    @classmethod
    def setUpClass(cls):
        cls.interface = _install_fake_amdsmi()
        cls.metric_module = _load_metric_module()
        cls.logger_module = _load_logger_module()

    def _drive_two_gpus(self, versions, make_logger, show_unsupported=False):
        handles = [object(), object()]
        by_handle = {id(handle): version for handle, version in zip(handles, versions)}
        self.interface.amdsmi_get_gpu_metrics_header_info = lambda handle: _header(
            by_handle[id(handle)]
        )
        helpers = _FakeHelpers({id(handle): index for index, handle in enumerate(handles)})

        commands = object.__new__(self.metric_module.MetricCommands)
        commands.logger = make_logger(helpers)
        commands.helpers = helpers
        commands.group_check_printed = True
        commands.device_handles = []
        commands.metric_gpu(_build_args(gpu=copy.copy(handles), show_unsupported=show_unsupported))
        return handles, commands.logger

    def _run_two_gpus(self, output, versions, show_unsupported=False):
        handles, logger = self._drive_two_gpus(
            versions, lambda _helpers: _FakeLogger(output), show_unsupported
        )
        self.assertEqual(len(logger.captured), 2, "expected one payload per GPU")
        return [logger.captured[id(handle)] for handle in handles]

    def _csv_rows(self, versions, show_unsupported=False):
        """One CSV row per GPU, off the real logger rather than a stand-in."""
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self._drive_two_gpus(
                versions,
                lambda helpers: self.logger_module.AMDSMILogger(format="csv", helpers=helpers),
                show_unsupported,
            )
        return list(csv.DictReader(io.StringIO(stdout.getvalue())))

    def test_each_gpu_is_filtered_against_its_own_version(self):
        v13, v16 = self._run_two_gpus("human", [(1, 3), (1, 6)])

        # jpeg_activity is impossible on both; mm/vcn_activity only on v1.6.
        self.assertNotIn("jpeg_activity", v13["usage"])
        self.assertNotIn("jpeg_activity", v16["usage"])
        for key in ("mm_activity", "vcn_activity"):
            self.assertIn(key, v13["usage"])
            self.assertNotIn(key, v16["usage"])

        # v1.6 gained the PCIe counters that v1.3 cannot report.
        self.assertNotIn("bandwidth", v13["pcie"])
        self.assertIn("bandwidth", v16["pcie"])

        self.assertNotEqual(set(v13["usage"]), set(v16["usage"]))

    def test_machine_formats_give_both_gpus_the_same_key_set(self):
        # Mixed versions are what would make a filtered machine format go ragged.
        for output in _FakeLogger.MACHINE_FORMATS:
            with self.subTest(output=output):
                v13, v16 = self._run_two_gpus(output, [(1, 3), (1, 6)])
                self.assertEqual(set(_key_paths(v13)), set(_key_paths(v16)))
                for key in ("mm_activity", "vcn_activity", "jpeg_activity"):
                    self.assertIn(key, v13["usage"])
                    self.assertIn(key, v16["usage"])
                self.assertIn("bandwidth", v13["pcie"])
                self.assertIn("bandwidth", v16["pcie"])

    # Version-absent on the v1.6 GPU only, on the v1.3 GPU only, and on both.
    _VERSION_ABSENT_COLUMNS = ("mm_activity", "vcn_activity", "bandwidth", "jpeg_activity")

    def test_csv_is_not_filtered_when_the_gpus_share_a_version(self):
        rows = self._csv_rows([(1, 3), (1, 3)])
        self.assertEqual(len(rows), 2, "expected one CSV row per GPU")
        for row in rows:
            for column in self._VERSION_ABSENT_COLUMNS:
                self.assertIn(column, row, f"{column} was filtered out of CSV")
                self.assertEqual(row[column], NA)

    def test_csv_is_not_filtered_when_the_gpus_are_on_different_versions(self):
        rows = self._csv_rows([(1, 3), (1, 6)])
        self.assertEqual(len(rows), 2, "expected one CSV row per GPU")
        self.assertEqual([row["gpu"] for row in rows], ["0", "1"])
        self.assertEqual(list(rows[0]), list(rows[1]), "the two GPUs produced ragged columns")
        for row in rows:
            for column in self._VERSION_ABSENT_COLUMNS:
                self.assertIn(column, row, f"{column} was filtered out of CSV")

    def test_csv_header_does_not_depend_on_the_flag(self):
        for versions in ([(1, 3), (1, 3)], [(1, 3), (1, 6)]):
            with self.subTest(versions=versions):
                self.assertEqual(
                    list(self._csv_rows(versions)[0]),
                    list(self._csv_rows(versions, show_unsupported=True)[0]),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
