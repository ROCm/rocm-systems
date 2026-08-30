#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for the ``amd-smi set --clk-limit`` DPM snap helper.

These stub the C library so they run without GPU hardware or the compiled
``amdsmi`` package, locking in the snap-down behavior (a ``max`` cap is reduced
to the largest reachable DPM level, never exceeding the request) and the safe
``None`` fallback on empty, missing, or library-error frequency lists.
"""

import collections
import importlib.util
import os
import sys
import types
import unittest

# ``common.common`` bootstraps the real amdsmi package at import time (sys.path
# insert + ``import amdsmi`` + module-level ``build_type_lists()``), which fails
# on a stale or mismatched install. This suite fully stubs ``amdsmi`` itself and
# only needs ``amdsmi_path`` to locate the *installed* CLI fallback, so degrade
# gracefully: if the shared harness cannot load, drop to ``None`` and rely on
# the in-tree source checkout (resolved first below). Keeps this file runnable
# from a plain checkout even when no matching amdsmi is installed.
try:
    from common.common import amdsmi_path
except (ImportError, FileNotFoundError):  # pragma: no cover - harness/install unavailable
    amdsmi_path = None

# set_value.py lives in the amd-smi CLI, which exists in two layouts:
#   * source checkout: <repo>/projects/amdsmi/amdsmi_cli (sibling of tests/)
#   * installed:       <rocm>/libexec/amdsmi_cli (amdsmi_path is the sibling
#                      <rocm>/share/amd_smi)
# Prefer the in-tree source when running from a checkout so the test exercises
# the code under review; fall back to the installed CLI otherwise.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_CLI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "amdsmi_cli"))
_INSTALLED_CLI_DIR = (
    os.path.join(os.path.dirname(os.path.dirname(amdsmi_path)), "libexec", "amdsmi_cli")
    if amdsmi_path
    else ""
)


def _resolve_cli_dir():
    for cli_dir in (_SOURCE_CLI_DIR, _INSTALLED_CLI_DIR):
        if cli_dir and os.path.isfile(os.path.join(cli_dir, "subcommands", "set_value.py")):
            return cli_dir
    return None


_CLI_DIR = _resolve_cli_dir()
SET_VALUE_PATH = os.path.join(_CLI_DIR, "subcommands", "set_value.py") if _CLI_DIR else ""

# AMDSMI_STATUS_NOT_SUPPORTED sentinel used by the stubbed library-error path.
_STATUS_NOT_SUPPORTED = 8


class _FakeLibraryException(Exception):
    """Stand-in for ``amdsmi_exception.AmdSmiLibraryException``.

    Mirrors the two accessors the helper and its callers use, ``get_error_code``
    and ``get_error_info``.
    """

    def __init__(self, err_code=_STATUS_NOT_SUPPORTED, message="mock error"):
        super().__init__(message)
        self._err_code = err_code
        self._message = message

    def get_error_code(self):
        return self._err_code

    def get_error_info(self):
        return self._message


def _install_fake_amdsmi():
    """Register a stub ``amdsmi`` package so ``set_value.py`` imports cleanly.

    Returns the fake ``amdsmi_interface`` module so individual tests can swap in
    per-case ``amdsmi_get_clk_freq`` return values or side effects.
    """
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")
    wrapper = types.ModuleType("amdsmi.amdsmi_wrapper")

    # Constants set_value.py binds at import time; the values are irrelevant here.
    interface.AMDSMI_MAX_PPT_LIMIT = 0
    interface.AMDSMI_MAX_UTIL = 100
    wrapper.AMDSMI_STATUS_NOT_SUPPORTED = _STATUS_NOT_SUPPORTED
    interface.amdsmi_wrapper = wrapper
    # Overwritten per-test; the empty default keeps the snap path inert.
    interface.amdsmi_get_clk_freq = lambda _handle, _clk_type: {"frequency": []}

    exception.AmdSmiLibraryException = _FakeLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception

    sys.modules["amdsmi"] = amdsmi_pkg
    sys.modules["amdsmi.amdsmi_interface"] = interface
    sys.modules["amdsmi.amdsmi_exception"] = exception
    sys.modules["amdsmi.amdsmi_wrapper"] = wrapper
    return interface


def _load_set_value_module():
    # set_value.py imports the sibling ``amdsmi_cli_exceptions`` module by bare
    # name, so the CLI dir must be importable before the module is executed.
    if _CLI_DIR and _CLI_DIR not in sys.path:
        sys.path.insert(0, _CLI_DIR)
    spec = importlib.util.spec_from_file_location("set_value_under_test", SET_VALUE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSnapClkLimitToDpm(unittest.TestCase):
    """Unit tests for ``SetValueCommands._snap_clk_limit_to_dpm``."""

    # Example fclk DPM levels (Hz). Deliberately unsorted and seeded with a 0 Hz
    # entry to prove the helper sorts the levels and filters out f <= 0.
    FCLK_DPM_HZ = [1600_000_000, 0, 1200_000_000, 2000_000_000, 1900_000_000]

    _SAVED_MODULE_NAMES = (
        "amdsmi",
        "amdsmi.amdsmi_interface",
        "amdsmi.amdsmi_exception",
        "amdsmi.amdsmi_wrapper",
    )

    @classmethod
    def setUpClass(cls):
        if not SET_VALUE_PATH:
            raise unittest.SkipTest("amd-smi CLI set_value.py not found (source or installed)")
        # Snapshot any real amdsmi already loaded so the stub does not leak into
        # sibling suites sharing the interpreter; restored in tearDownClass.
        cls._saved_modules = {name: sys.modules.get(name) for name in cls._SAVED_MODULE_NAMES}
        cls.interface = _install_fake_amdsmi()
        cls.module = _load_set_value_module()

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def _snap(self, requested_mhz, frequency, min_mhz=None):
        # Shared harness: point the stubbed C entry point at a fixed DPM table
        # (Hz) so every case drives the real helper with controlled input.
        freq_info = {"frequency": list(frequency)}
        self.interface.amdsmi_get_clk_freq = lambda _handle, _clk_type: freq_info
        return self.module.SetValueCommands._snap_clk_limit_to_dpm(
            None, None, requested_mhz, min_mhz
        )

    def test_exact_dpm_levels_unchanged(self):
        # Requests already on a DPM level pass through untouched: only snap when
        # we must, never move an aligned cap.
        for level in (1200, 1600, 1900, 2000):
            with self.subTest(level=level):
                self.assertEqual(self._snap(level, self.FCLK_DPM_HZ), level)

    def test_between_levels_snaps_down(self):
        # Every off-level request must resolve to the largest DPM <= requested.
        expected = {
            1300: 1200,
            1500: 1200,
            1599: 1200,
            1700: 1600,
            1850: 1600,
            1899: 1600,
            1950: 1900,
            1999: 1900,
        }
        for requested, want in expected.items():
            with self.subTest(requested=requested):
                self.assertEqual(self._snap(requested, self.FCLK_DPM_HZ), want)

    def test_above_highest_caps_to_highest_dpm(self):
        # A request above the top DPM level clamps to the highest level -- we
        # never invent a cap the hardware cannot reach.
        self.assertEqual(self._snap(2500, self.FCLK_DPM_HZ), 2000)

    def test_below_lowest_returns_none(self):
        # The caller rejects val < min_clk first; the helper must not invent a
        # level below the lowest reachable DPM.
        self.assertIsNone(self._snap(1000, self.FCLK_DPM_HZ))

    def test_empty_frequency_list_returns_none(self):
        # No DPM levels reported -> no snap; the helper returns None so the
        # caller falls back to the raw request instead of crashing.
        self.assertIsNone(self._snap(1600, []))

    def test_missing_frequency_key_returns_none(self):
        # Malformed driver payload (no "frequency" key) also degrades to None.
        self.interface.amdsmi_get_clk_freq = lambda _handle, _clk_type: {}
        self.assertIsNone(self.module.SetValueCommands._snap_clk_limit_to_dpm(None, None, 1600))

    def test_library_exception_returns_none(self):
        # If the library call itself raises, the helper swallows it and returns
        # None so a set never fails just because the DPM query did.
        def _raise(_handle, _clk_type):
            raise _FakeLibraryException(_STATUS_NOT_SUPPORTED)

        self.interface.amdsmi_get_clk_freq = _raise
        self.assertIsNone(self.module.SetValueCommands._snap_clk_limit_to_dpm(None, None, 1600))

    def test_two_level_dpm_snaps_down(self):
        # Sparse two-level DPM table ({500, 2100} MHz): an unaligned request
        # snaps down to 500. sclk is excluded from snapping at the CLI call site
        # (it honors the exact requested max), so this only covers the generic
        # discrete-table path used by mclk/fclk.
        two_level_hz = [500_000_000, 2100_000_000]
        self.assertEqual(self._snap(1500, two_level_hz), 500)
        self.assertEqual(self._snap(2100, two_level_hz), 2100)

    def test_mclk_levels_snap_down(self):
        # Multi-level domain ({900, 1100, 1200, 1300} MHz).
        mclk_hz = [900_000_000, 1100_000_000, 1200_000_000, 1300_000_000]
        self.assertEqual(self._snap(1250, mclk_hz), 1200)
        self.assertEqual(self._snap(1099, mclk_hz), 900)

    def test_min_between_levels_excludes_lower_dpm(self):
        # min=1000 sits between DPM {900, 1100}; a request of 1050 has no
        # level in [1000, 1050], so the min filter drops 900 and the helper
        # returns None rather than snapping below the reported minimum.
        self.assertIsNone(self._snap(1050, [900_000_000, 1100_000_000], min_mhz=1000))

    def test_min_filter_keeps_level_at_or_above_min(self):
        # A DPM level exactly at min_clk stays eligible; lower ones are dropped.
        levels = [1000_000_000, 1100_000_000, 1200_000_000]
        self.assertEqual(self._snap(1150, levels, min_mhz=1000), 1100)
        self.assertEqual(self._snap(1000, levels, min_mhz=1000), 1000)


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

    def store_multiple_device_output(self):
        pass

    def last_output(self, key):
        for _device, stored_key, value in reversed(self.outputs):
            if stored_key == key:
                return value
        return None


class _StubHelpers:
    """Minimal ``self.helpers`` stub for the GPU ``set`` path."""

    def check_required_groups(self):
        pass

    def handle_gpus(self, args, logger, func):
        # Single device: never recurse, echo the resolved handle back.
        return (False, args.gpu)

    def is_baremetal(self):
        # False routes straight to the universal clk-limit block, skipping the
        # baremetal-only dispatch this suite does not exercise.
        return False

    def get_gpu_id_from_device_handle(self, handle):
        return 0


_ClkLimit = collections.namedtuple("_ClkLimit", ["clk_type", "lim_type", "val"])


class TestSetGpuClkLimitCallSite(unittest.TestCase):
    """Call-site tests for the ``set_gpu`` clk-limit snap/validation branch.

    Complements ``TestSnapClkLimitToDpm`` (snap helper in isolation) by driving
    the real ``set_gpu`` dispatch with the C library stubbed, asserting the
    CLI-only behaviors: sclk is never snapped, the snap-down annotation, and the
    ``N/A`` opposing-bound skip that keeps the requested limit settable.
    """

    _SAVED_MODULE_NAMES = TestSnapClkLimitToDpm._SAVED_MODULE_NAMES

    @classmethod
    def setUpClass(cls):
        if not SET_VALUE_PATH:
            raise unittest.SkipTest("amd-smi CLI set_value.py not found (source or installed)")
        cls._saved_modules = {name: sys.modules.get(name) for name in cls._SAVED_MODULE_NAMES}
        cls.interface = _install_fake_amdsmi()
        # Extra surface the set_gpu clk-limit branch touches beyond the snap
        # helper: the clock-info entry point, the set entry point, the clk-type
        # enum, and the BDF lookup used only to build error strings.
        cls.interface.AmdSmiClkType = types.SimpleNamespace(GFX="GFX", MEM="MEM", DF="DF")
        cls.interface.amdsmi_get_gpu_device_bdf = lambda _handle: "0000:00:00.0"
        cls.interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM = 4
        cls.module = _load_set_value_module()

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def _run_clk_limit(
        self, clk_type, lim_type, val, min_clk, max_clk, dpm_hz=(), freq_raises=False
    ):
        # Wire per-case driver responses, then drive the real set_gpu dispatch.
        self.interface.amdsmi_get_clock_info = lambda _h, _c: {
            "min_clk": min_clk,
            "max_clk": max_clk,
        }
        if freq_raises:
            # Assert the DPM query is not issued on this path (exact-max skip).
            def _freq_boom(_h, _c):
                raise AssertionError("amdsmi_get_clk_freq should not be called")

            self.interface.amdsmi_get_clk_freq = _freq_boom
        else:
            freq_info = {"frequency": list(dpm_hz)}
            self.interface.amdsmi_get_clk_freq = lambda _h, _c: freq_info
        set_calls = []
        self.interface.amdsmi_set_gpu_clk_limit = lambda _h, ct, lt, v: set_calls.append(
            (ct, lt, v)
        )

        logger = _RecordingLogger()
        cmd = self.module.SetValueCommands()
        cmd.logger = logger
        cmd.helpers = _StubHelpers()
        cmd.group_check_printed = True
        cmd.device_handles = ["gpu0"]

        args = types.SimpleNamespace(
            gpu="gpu0",
            power_cap=None,
            process_isolation=None,
            clk_limit=_ClkLimit(clk_type, lim_type, val),
        )
        cmd.set_gpu(args)
        return set_calls, logger.last_output("clk_limit")

    def test_sclk_max_is_never_snapped(self):
        # sclk exposes a continuous range: an off-DPM request is applied exactly
        # and carries no snap annotation.
        set_calls, message = self._run_clk_limit(
            "sclk", "max", 1234, min_clk=500, max_clk=2000, dpm_hz=(500_000_000, 2000_000_000)
        )
        self.assertEqual(set_calls, [("sclk", "max", 1234)])
        self.assertNotIn("snapped", message)

    def test_mclk_max_snaps_down_and_annotates(self):
        # mclk exposes a discrete DPM table: 1250 snaps down to 1200 and the
        # message reports both the request and the enforced level.
        set_calls, message = self._run_clk_limit(
            "mclk",
            "max",
            1250,
            min_clk=900,
            max_clk=1300,
            dpm_hz=(900_000_000, 1100_000_000, 1200_000_000, 1300_000_000),
        )
        self.assertEqual(set_calls, [("mclk", "max", 1200)])
        self.assertIn("1200MHz", message)
        self.assertIn("requested 1250MHz", message)
        self.assertIn("snapped down to nearest reachable DPM level", message)

    def test_fclk_max_snaps_down_and_annotates(self):
        # fclk is the second discrete-DPM clock; a guard accidentally
        # narrowed to only "mclk" would silently drop fclk snapping, so exercise
        # the fclk path at the call site as well, not just in the helper.
        set_calls, message = self._run_clk_limit(
            "fclk",
            "max",
            1850,
            min_clk=1200,
            max_clk=2000,
            dpm_hz=(1200_000_000, 1600_000_000, 1900_000_000, 2000_000_000),
        )
        self.assertEqual(set_calls, [("fclk", "max", 1600)])
        self.assertIn("1600MHz", message)
        self.assertIn("requested 1850MHz", message)
        self.assertIn("snapped down to nearest reachable DPM level", message)

    def test_max_fallthrough_applies_raw_request_when_dpm_unavailable(self):
        # When the DPM query yields no levels the snap helper returns None
        # and the call site must fall open -- the raw request still reaches the
        # driver (never silently dropped) and carries no snap annotation, since
        # nothing was actually aligned.
        set_calls, message = self._run_clk_limit(
            "mclk", "max", 1250, min_clk=900, max_clk=1300, dpm_hz=()
        )
        self.assertEqual(set_calls, [("mclk", "max", 1250)])
        self.assertIn("Successfully changed", message)
        self.assertNotIn("snapped", message)

    def test_max_snap_never_lands_below_min_clk(self):
        # When min_clk sits between DPM levels the snap must not drop
        # the cap below it. min=1000 with DPM {900, 1100} and request 1050 has
        # no level in [1000, 1050]; the raw (>= min) request is applied instead
        # of an invalid 900 cap, and no snap annotation is shown.
        set_calls, message = self._run_clk_limit(
            "mclk", "max", 1050, min_clk=1000, max_clk=1100, dpm_hz=(900_000_000, 1100_000_000)
        )
        self.assertEqual(set_calls, [("mclk", "max", 1050)])
        self.assertNotIn("900MHz", message)
        self.assertNotIn("snapped", message)

    def test_exact_max_skips_dpm_query(self):
        # When the request already equals max_clk there is no snap to do,
        # so the DPM query is skipped entirely (the stubbed query raises if the
        # call site hits it).
        set_calls, message = self._run_clk_limit(
            "mclk", "max", 1300, min_clk=900, max_clk=1300, freq_raises=True
        )
        self.assertEqual(set_calls, [])
        self.assertIn("already set to 1300MHz", message)

    def test_snap_to_current_max_reports_already_set_with_annotation(self):
        # A request above max_clk snaps down to the top DPM level, which
        # equals the current max -- so no driver write happens, but the
        # annotation must still report that the request was reduced to the
        # enforced level.
        set_calls, message = self._run_clk_limit(
            "mclk",
            "max",
            1350,
            min_clk=900,
            max_clk=1300,
            dpm_hz=(900_000_000, 1100_000_000, 1200_000_000, 1300_000_000),
        )
        self.assertEqual(set_calls, [])
        self.assertIn("already set to 1300MHz", message)
        self.assertIn("requested 1350MHz", message)

    def test_exact_max_level_reports_already_set_without_annotation(self):
        # A request already sitting on the top DPM level (== current max)
        # is a no-op with no snap annotation -- nothing was reduced.
        set_calls, message = self._run_clk_limit(
            "mclk",
            "max",
            1300,
            min_clk=900,
            max_clk=1300,
            dpm_hz=(900_000_000, 1100_000_000, 1200_000_000, 1300_000_000),
        )
        self.assertEqual(set_calls, [])
        self.assertIn("already set to 1300MHz", message)
        self.assertNotIn("snapped", message)

    def test_max_set_proceeds_when_min_clk_na(self):
        # An unreadable min bound must not block a max-set; the requested
        # (DPM-aligned) cap is still applied.
        set_calls, message = self._run_clk_limit(
            "mclk", "max", 1200, min_clk="N/A", max_clk=2000, dpm_hz=(1200_000_000,)
        )
        self.assertEqual(set_calls, [("mclk", "max", 1200)])
        self.assertIn("Successfully changed", message)

    def test_min_set_proceeds_when_max_clk_na(self):
        # An unreadable max bound must not block a min-set (symmetric case).
        set_calls, message = self._run_clk_limit("fclk", "min", 1500, min_clk=1000, max_clk="N/A")
        self.assertEqual(set_calls, [("fclk", "min", 1500)])
        self.assertIn("Successfully changed", message)

    def test_max_below_min_is_rejected(self):
        # The in-range validation still fires when the opposing bound is known:
        # a max below the reported min is refused and never reaches the driver.
        set_calls, message = self._run_clk_limit(
            "mclk", "max", 500, min_clk=1000, max_clk=2000, dpm_hz=(1000_000_000, 2000_000_000)
        )
        self.assertEqual(set_calls, [])
        self.assertIn("less than min", message)

    def test_min_above_max_is_rejected(self):
        # Symmetric to the max-below-min case: a min above the reported max is
        # refused and never reaches the driver.
        set_calls, message = self._run_clk_limit("fclk", "min", 2500, min_clk=1000, max_clk=2000)
        self.assertEqual(set_calls, [])
        self.assertIn("greater than max", message)


if __name__ == "__main__":
    unittest.main()
