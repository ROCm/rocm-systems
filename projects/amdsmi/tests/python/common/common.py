#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Core shared infrastructure for the AMD SMI Python test suite.

Owns three responsibilities:
  * amdsmi module discovery -- resolves the install via
    AMDSMI_PATH -> ROCM_HOME -> ROCM_PATH -> /opt/rocm, inserts it on
    sys.path, and guards against a shadowing site-packages copy
    (see ROCM-1552 / PR #6359).
  * the ``Common`` helper base -- enum tables, ``Test_API_*`` drivers,
    ``check_ret`` and friends, held as ``self.common`` by the API tests
    (intentionally NOT a unittest.TestCase).
  * the unittest runner machinery -- ``run_test_dir``,
    ``GTestSummaryRunner``, verbosity/-k/-l parsing and the help/legend
    printers used by the three top-level runners.

Imported by every functional/unit leaf test and by all three runners
(integration_test.py, cli_unit_test.py, unit_tests.py).
"""

import contextlib
import inspect
import io
import json
import os
import pathlib
import sys
import time
import unittest


def _print_path_remediation(script, file):
    """Print the shared 'fix with one of:' remediation block.

    Used by both print_amdsmi_path_help() (-h context) and print_shadow_error()
    (import-error context) to avoid duplicating the step list.
    """
    print(
        "\t 1. Run the tests from the installed amd-smi-lib-tests package location (RECOMMENDED):\n"
        f"\t\t `sudo $ROCM_PATH/share/amd_smi/tests/python_unittest/{script} -v`\n"
        f"\t\t e.g. `sudo /opt/rocm/share/amd_smi/tests/python_unittest/{script} -v`\n\n"
        "\t 2. Set AMDSMI_PATH to point at the correct build:\n"
        "\t\t `export AMDSMI_PATH=$ROCM_PATH/share/amd_smi`\n\n"
        "\t 3. Reinstall the correct amd-smi-lib package (removes the stale Python package too):\n"
        "\t\t `sudo apt remove amd-smi-lib && sudo apt install amd-smi-lib`\n"
        "\t\t or, if amd-smi was installed directly via pip:\n"
        "\t\t `sudo pip uninstall amdsmi`",
        file=file,
    )


def print_amdsmi_path_help(file=sys.stdout):
    """Print env-var documentation, path remediation guidance, and usage examples for -h output."""
    _script = os.path.basename(sys.argv[0]) or "<script>"
    print("\nAdditional options:", file=file)
    print("  -l, --list           List all available tests and exit", file=file)
    print("  -x, --exclude PAT    Skip tests whose id contains PAT (inverse of -k)", file=file)
    print(file=file)
    print("Environment variables:", file=file)
    print("  AMDSMI_PATH   Path to amd_smi share dir (overrides ROCM_HOME/ROCM_PATH)", file=file)
    print("  ROCM_HOME     ROCm install root, used when AMDSMI_PATH is unset", file=file)
    print("  ROCM_PATH     Fallback ROCm install root (default: /opt/rocm)", file=file)
    print(file=file)
    print("If amd-smi loads from the wrong path, fix with one of:", file=file)
    _print_path_remediation(_script, file)
    _full = "sudo " + (sys.argv[0] or _script)
    _base = sys.argv[0] or _script
    _cmds = [
        (_base + " -l", "list all available tests"),
        (_full + " -v", "run all tests, verbose with print statements (RECOMMENDED)"),
        (_full + ' -k "test_name" -v', "run only tests matching substring"),
        (_full + " -x performance -v", "run everything except the performance tests"),
        (_full + " -q", "run all tests, quiet with no print statements"),
    ]
    _w = max(len(cmd) for cmd, _ in _cmds) + 2
    print(file=file)
    print("Examples:", file=file)
    for cmd, desc in _cmds:
        print(f"  {cmd:<{_w}} - {desc}", file=file)


def print_shadow_error(script, loaded_from, expected_path, file=sys.stderr):
    """Print an actionable error when amdsmi was loaded from the wrong path.

    Prints an ERROR header identifying the unexpected source, the shadowing
    diagnosis, the shared remediation steps, and a pointer to -h for more details.
    """
    print(
        f"ERROR: amdsmi loaded from '{loaded_from}' instead of expected path '{expected_path}'.",
        file=file,
    )
    print("A system-installed amdsmi package is shadowing the test target.", file=file)
    print("Fix with one of:", file=file)
    _print_path_remediation(script, file)
    print(f"\nRefer to `{script} -h` for more details.", file=file)


amdsmi_path = os.environ.get("AMDSMI_PATH") or os.path.join(
    os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or "/opt/rocm", "share/amd_smi"
)
if not os.path.exists(amdsmi_path):
    raise FileNotFoundError(
        f'amdsmi path "{amdsmi_path}" does not exist. '
        "Set ROCM_HOME, ROCM_PATH, or AMDSMI_PATH to the correct install location."
    )
sys.path.insert(0, amdsmi_path)
try:
    import amdsmi
except ImportError as e:
    raise ImportError(f'Could not import the "amdsmi" module from "{amdsmi_path}"') from e

# Verify the imported amdsmi package came from amdsmi_path, not a system-installed
# package in dist-packages. A stale system install wins over sys.path.append() because
# dist-packages is already on sys.path at interpreter startup; insert(0,...) above
# prevents this, but error explicitly in case amdsmi was already cached in sys.modules.
#
# Example of how to trigger this error output (for test purposes):
# sudo AMDSMI_PATH=/tmp /opt/rocm/share/amd_smi/tests/python_unittest/cli_unit_test.py -v
_amdsmi_file = getattr(amdsmi, "__file__", None) or ""
if not os.path.realpath(_amdsmi_file).startswith(os.path.realpath(amdsmi_path) + os.sep):
    _script = os.path.basename(sys.argv[0]) or "unit_tests.py"
    print_shadow_error(_script, _amdsmi_file, amdsmi_path)
    # For direct test-script invocations use sys.exit so no Python traceback
    # clutters the remediation output.  For anything else (pytest, IDE runners,
    # ad-hoc imports) raise so the caller gets a clear error with location.
    _known_scripts = ("unit_tests.py", "integration_test.py", "cli_unit_test.py")
    _main_file = getattr(sys.modules.get("__main__"), "__file__", "") or ""
    if os.path.basename(_main_file) in _known_scripts:
        sys.exit(1)
    raise RuntimeError(
        f"amdsmi loaded from wrong path: '{_amdsmi_file}' (expected under '{amdsmi_path}')"
    )

#################################################
# Module level functions, not part of the class #
#################################################

# Canonical status-string sentinels, shared by the Common class and the
# standalone benchmark suites (single source of truth).
PASS = "AMDSMI_STATUS_SUCCESS"
FAIL = "AMDSMI_STATUS_INVAL"

# Status-code -> name map, derived once from the AmdSmiStatus enum so the table
# never drifts from the library. Shared by the Common class and the standalone
# performance benchmark suites (which intentionally do not inherit Common).
ERROR_MAP = {str(member.value): f"AMDSMI_STATUS_{member.name}" for member in amdsmi.AmdSmiStatus}

VERBOSITY_QUIET = 0  # -q / --quiet
VERBOSITY_NORMAL = 1  # default (dot-per-test)
VERBOSITY_VERBOSE = 2  # -v / --verbose (per-test result lines)


def build_type_lists():
    """Enum-derived test-parameter lists, each ``[(name, enum_value, expected)]``.

    Pure: built solely from the amdsmi enums/wrapper with no ``amdsmi_init`` or
    hardware access, so it is safe to call at import or test-discovery time.
    Shared by the ``Common`` class and the standalone performance benchmark
    suites (which intentionally avoid constructing ``Common`` to keep hardware
    init out of collection), so this enum-to-list logic lives in one place.
    """
    status_types = [
        (member.name, amdsmi.AmdSmiStatus(member.value), PASS) for member in amdsmi.AmdSmiStatus
    ]

    clk_types = []
    for member in amdsmi.AmdSmiClkType:
        cond = PASS
        if member.name in ["DCEF", "PCIE"]:
            cond = [PASS, FAIL]
        clk_types.append((member.name, amdsmi.AmdSmiClkType(member.value), cond))

    clk_limit_types = [
        (member.name, amdsmi.AmdSmiClkLimitType(member.value), PASS)
        for member in amdsmi.AmdSmiClkLimitType
    ]

    # amdsmi_get_cpu_current_io_bandwidth(handle, encoding: int, link_name: str) takes a
    # bandwidth-type int and a link-ID string (see amdsmi_interface). The valid values come
    # from the amdsmi_link_id_bw_type_t struct in amdsmi.h: link names P0-P4/G0-G7, and
    # bandwidth types 1 (Aggregate), 2 (Read), 4 (Write). For IO bandwidth only AGG (1) is
    # used; Read/Write (2/4) apply to XGMI bandwidth. Each tuple is
    # (link_name, bw_type, expected): the consumers pass element 0 as link_name and element
    # 1 as the encoding int. This mirrors the public interface contract rather than scraping
    # wrapper *_BW0 symbols (the old scrape passed the symbol name, e.g. "AGG_BW0", as
    # link_name, which is not a valid link ID).
    io_bw_agg_type = 1  # AGG: the bandwidth type used for IO bandwidth
    io_bw_encodings = [
        (link_name, io_bw_agg_type, PASS)
        for link_name in (
            "P0",
            "P1",
            "P2",
            "P3",
            "P4",
            "G0",
            "G1",
            "G2",
            "G3",
            "G4",
            "G5",
            "G6",
            "G7",
        )
    ]

    gpu_blocks = []
    for member in amdsmi.AmdSmiGpuBlock:
        cond = PASS
        if member.name in ["INVALID", "RESERVED"]:
            cond = FAIL
        gpu_blocks.append((member.name, amdsmi.AmdSmiGpuBlock(member.value), cond))

    memory_types = [
        (member.name, amdsmi.AmdSmiMemoryType(member.value), PASS)
        for member in amdsmi.AmdSmiMemoryType
    ]

    reg_types = [
        (member.name, amdsmi.AmdSmiRegType(member.value), PASS) for member in amdsmi.AmdSmiRegType
    ]

    voltage_metrics = [
        (member.name, amdsmi.AmdSmiVoltageMetric(member.value), PASS)
        for member in amdsmi.AmdSmiVoltageMetric
    ]

    voltage_types = []
    for member in amdsmi.AmdSmiVoltageType:
        cond = PASS
        if member.name in ["INVALID"]:
            cond = FAIL
        voltage_types.append((member.name, amdsmi.AmdSmiVoltageType(member.value), cond))

    link_types = []
    for member in amdsmi.AmdSmiLinkType:
        cond = PASS
        if member.name in ["AMDSMI_LINK_TYPE_NOT_APPLICABLE", "AMDSMI_LINK_TYPE_UNKNOWN"]:
            cond = FAIL
        link_types.append((member.name, amdsmi.AmdSmiLinkType(member.value), cond))

    temperature_types = [
        (member.name, amdsmi.AmdSmiTemperatureType(member.value), PASS)
        for member in amdsmi.AmdSmiTemperatureType
    ]

    temperature_metrics = [
        (member.name, amdsmi.AmdSmiTemperatureMetric(member.value), PASS)
        for member in amdsmi.AmdSmiTemperatureMetric
    ]

    utilization_counter_types: list = [
        (member.name, amdsmi.AmdSmiUtilizationCounterType(member.value), PASS)
        for member in amdsmi.AmdSmiUtilizationCounterType
    ]
    # Negative test: integer 100 is out of range for AmdSmiUtilizationCounterType;
    # the API must reject it. Not a real enum member so must be added explicitly.
    utilization_counter_types.append(("UTILIZATION_COUNTER_BAD", 100, FAIL))

    event_groups = []
    for member in amdsmi.AmdSmiEventGroup:
        cond = PASS
        if member.name in ["GRP_INVALID"]:
            cond = FAIL
        event_groups.append((member.name, amdsmi.AmdSmiEventGroup(member.value), cond))

    event_types = [
        (member.name, amdsmi.AmdSmiEventType(member.value), PASS)
        for member in amdsmi.AmdSmiEventType
    ]

    counter_commands = [
        (member.name, amdsmi.AmdSmiCounterCommand(member.value), PASS)
        for member in amdsmi.AmdSmiCounterCommand
    ]

    compute_partition_types = []
    for member in amdsmi.AmdSmiComputePartitionType:
        cond = PASS
        if member.name in ["INVALID"]:
            cond = FAIL
        compute_partition_types.append(
            (member.name, amdsmi.AmdSmiComputePartitionType(member.value), cond)
        )

    memory_partition_types = []
    for member in amdsmi.AmdSmiMemoryPartitionType:
        cond = PASS
        if member.name in ["UNKNOWN"]:
            cond = FAIL
        elif member.name in ["NPS4", "NPS8"]:
            # NPS4/NPS8 are hardware-dependent; accept success or invalid depending on support
            # BTW - no asic supports NPS8...
            cond = [PASS, FAIL]
        memory_partition_types.append(
            (member.name, amdsmi.AmdSmiMemoryPartitionType(member.value), cond)
        )

    compute_partition_mem_alloc_mode_types = []
    for member in amdsmi.AmdSmiComputePartitionMemAllocModeType:
        cond = PASS
        if member.name in ["INVALID"]:
            cond = FAIL
        compute_partition_mem_alloc_mode_types.append(
            (member.name, amdsmi.AmdSmiComputePartitionMemAllocModeType(member.value), cond)
        )

    freq_inds = []
    for member in amdsmi.AmdSmiFreqInd:
        cond = PASS
        if member.name in ["INVALID"]:
            cond = FAIL
        freq_inds.append((member.name, amdsmi.AmdSmiFreqInd(member.value), cond))

    power_profile_preset_masks = []
    for member in amdsmi.AmdSmiPowerProfilePresetMasks:
        cond = PASS
        if member.name in ["INVALID"]:
            # INVALID (0xFFFFFFFFFFFFFFFF) is intentionally tested with a specific expected
            # status rather than the generic FAIL (AMDSMI_STATUS_INVAL). The API returns
            # AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS for this sentinel, not AMDSMI_STATUS_INVAL.
            cond = "AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS"
        power_profile_preset_masks.append(
            (member.name, amdsmi.AmdSmiPowerProfilePresetMasks(member.value), cond)
        )

    processor_types = []
    for member in amdsmi.AmdSmiProcessorType:
        cond = PASS
        if member.name in ["UNKNOWN"]:
            cond = FAIL
        processor_types.append((member.name, amdsmi.AmdSmiProcessorType(member.value), cond))

    dev_perf_levels = []
    for member in amdsmi.AmdSmiDevPerfLevel:
        cond = PASS
        if member.name in ["UNKNOWN"]:
            cond = FAIL
        dev_perf_levels.append((member.name, amdsmi.AmdSmiDevPerfLevel(member.value), cond))

    return {
        "status_types": status_types,
        "clk_types": clk_types,
        "clk_limit_types": clk_limit_types,
        "io_bw_encodings": io_bw_encodings,
        "gpu_blocks": gpu_blocks,
        "memory_types": memory_types,
        "reg_types": reg_types,
        "voltage_metrics": voltage_metrics,
        "voltage_types": voltage_types,
        "link_types": link_types,
        "temperature_types": temperature_types,
        "temperature_metrics": temperature_metrics,
        "utilization_counter_types": utilization_counter_types,
        "event_groups": event_groups,
        "event_types": event_types,
        "counter_commands": counter_commands,
        "compute_partition_types": compute_partition_types,
        "memory_partition_types": memory_partition_types,
        "compute_partition_mem_alloc_mode_types": compute_partition_mem_alloc_mode_types,
        "freq_inds": freq_inds,
        "power_profile_preset_masks": power_profile_preset_masks,
        "processor_types": processor_types,
        "dev_perf_levels": dev_perf_levels,
    }


# Module-level views of the enum-derived parameter lists, computed once at import.
# They resolve natively (e.g. ``common.GPU_BLOCKS``) so the standalone benchmark
# suites can consume them directly instead of via dynamic ``setattr`` a type checker
# can't follow. Read-only test parameters, so sharing a single instance is safe.
_TYPE_LISTS = build_type_lists()
STATUS_TYPES = _TYPE_LISTS["status_types"]
CLK_TYPES = _TYPE_LISTS["clk_types"]
CLK_LIMIT_TYPES = _TYPE_LISTS["clk_limit_types"]
IO_BW_ENCODINGS = _TYPE_LISTS["io_bw_encodings"]
GPU_BLOCKS = _TYPE_LISTS["gpu_blocks"]
MEMORY_TYPES = _TYPE_LISTS["memory_types"]
REG_TYPES = _TYPE_LISTS["reg_types"]
VOLTAGE_METRICS = _TYPE_LISTS["voltage_metrics"]
VOLTAGE_TYPES = _TYPE_LISTS["voltage_types"]
LINK_TYPES = _TYPE_LISTS["link_types"]
TEMPERATURE_TYPES = _TYPE_LISTS["temperature_types"]
TEMPERATURE_METRICS = _TYPE_LISTS["temperature_metrics"]
UTILIZATION_COUNTER_TYPES = _TYPE_LISTS["utilization_counter_types"]
EVENT_GROUPS = _TYPE_LISTS["event_groups"]
EVENT_TYPES = _TYPE_LISTS["event_types"]
COUNTER_COMMANDS = _TYPE_LISTS["counter_commands"]
COMPUTE_PARTITION_TYPES = _TYPE_LISTS["compute_partition_types"]
MEMORY_PARTITION_TYPES = _TYPE_LISTS["memory_partition_types"]
COMPUTE_PARTITION_MEM_ALLOC_MODE_TYPES = _TYPE_LISTS["compute_partition_mem_alloc_mode_types"]
FREQ_INDS = _TYPE_LISTS["freq_inds"]
POWER_PROFILE_PRESET_MASKS = _TYPE_LISTS["power_profile_preset_masks"]
PROCESSOR_TYPES = _TYPE_LISTS["processor_types"]
DEV_PERF_LEVELS = _TYPE_LISTS["dev_perf_levels"]


def _parse_verbose(argv=None):
    """Resolve the verbosity selector from *argv* (defaults to ``sys.argv``).

    ``-q``/``--quiet`` -> QUIET; ``-v``/``-vv``/``--verbose`` -> VERBOSE;
    otherwise NORMAL.  Parsed once at import as the module-level ``verbose`` so
    every runner and test module shares a single source of truth instead of
    re-parsing ``sys.argv`` in each file.
    """
    argv = sys.argv if argv is None else argv
    if "-q" in argv or "--quiet" in argv:
        return VERBOSITY_QUIET
    if any(a in ("-v", "-vv", "--verbose") for a in argv):
        return VERBOSITY_VERBOSE
    return VERBOSITY_NORMAL


# Shared verbosity for the whole process (runners + the test modules they import).
verbose = _parse_verbose()


def _print_test_ids(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            _print_test_ids(test)
        else:
            # Print the full dotted test id (e.g. "cli.test_event.TestEvent.test_command")
            # so -l output matches the GTestSummaryRunner's per-test labels. Strip the
            # "__main__." prefix the same way so module-run and discover-run agree.
            #
            # The -l/--list output is the data the user asked for, so it goes to stdout
            # (lets `... -l | grep xgmi` work without 2>&1).  Test-run chatter (legend,
            # progress, summary, errors, help) stays on stderr.
            test_id = test.id()
            if test_id.startswith("__main__."):
                test_id = test_id[len("__main__.") :]
            print(f"\t{test_id}", file=sys.stdout)
    return


def print_test_ids(suite):
    """Print every test ID in an already-loaded *suite* to stdout.

    Public entry point for the discover()-based runners (cli_unit_test.py,
    integration_test.py, unit_tests.py), which build their suite from a
    directory rather than from a single module.  Emits an "Available tests:"
    header followed by one dotted test id per line.  Output goes to stdout so
    the listing can be captured/piped independently of the normal test-run
    output on stderr (e.g. ``... -l | grep xgmi``).
    """
    print("Available tests:", file=sys.stdout)
    _print_test_ids(suite)
    return


def print_legend():
    """Print the dot-character legend for unittest output to stderr.

    Call this before running tests in non-verbose mode so users know what
    the single-character result indicators (., s, F, E) mean.
    """
    print("=" * 70, file=sys.stderr)
    print("Legend: . = pass, s = skipped, F = fail, E = error", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    return


def print_unittest_help():
    """Print unittest's -h output with its built-in Examples epilog stripped.

    unittest (via argparse) appends its own Examples block; we capture stdout,
    remove everything from the last 'Examples:' onward, and reprint — leaving
    our print_amdsmi_path_help() as the sole Examples section at the bottom.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            unittest.main(argv=[sys.argv[0] or "unit_tests.py", "--help"])
    except SystemExit:
        pass
    output = buf.getvalue()
    examples_idx = output.rfind("\nExamples:")
    if examples_idx != -1:
        output = output[:examples_idx]
    sys.stdout.write(output)
    sys.stdout.flush()


def make_runner_verbosity(verbose):
    """Map our three verbosity levels to the correct unittest.TextTestRunner verbosity.

    In verbose mode, test bodies self-report each result via print_func_name() /
    check_ret(), so we suppress the runner's own per-test lines (VERBOSITY_QUIET)
    to avoid printing each test name twice.  In normal and quiet modes we keep the
    runner at VERBOSITY_NORMAL (dots) so CI has a per-test progress indicator and
    hung tests are detectable.

    verbose value          → runner verbosity
    VERBOSITY_QUIET  (0)  → VERBOSITY_NORMAL (dots only)
    VERBOSITY_NORMAL (1)  → VERBOSITY_NORMAL (dots only)
    VERBOSITY_VERBOSE (2) → VERBOSITY_QUIET  (runner silent; test bodies own output)
    """
    if verbose >= VERBOSITY_VERBOSE:
        return VERBOSITY_QUIET  # test bodies print their own per-test output
    return VERBOSITY_NORMAL  # dots keep CI informed of progress


def _parse_k_pattern(argv=None):
    """Return the ``-k``/``--keyword`` filter substring from *argv*, or ``None``.

    Accepts both the separate ("-k", "pattern") and joined ("-kpattern") forms,
    matching the spellings unittest itself understands.
    """
    argv = sys.argv if argv is None else argv
    for i, arg in enumerate(argv):
        if arg in ("-k", "--keyword") and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("-k") and arg != "-k":
            return arg[2:]
    return None


def _parse_x_pattern(argv=None):
    """Return the ``-x``/``--exclude`` filter substring from *argv*, or ``None``.

    Tests whose id contains this substring are skipped (the inverse of ``-k``).
    Accepts both the separate ("-x", "pattern") and joined ("-xpattern") forms.
    """
    argv = sys.argv if argv is None else argv
    for i, arg in enumerate(argv):
        if arg in ("-x", "--exclude") and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("-x") and arg != "-x":
            return arg[2:]
    return None


def _filter_suite_exclude(suite, pattern):
    """Return a copy of *suite* with every test whose id contains *pattern* removed.

    unittest's ``-k`` can only include matching tests; this provides the inverse so
    callers can run everything except, e.g., the performance suites (``-x performance``).
    """
    filtered = unittest.TestSuite()
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            filtered.addTest(_filter_suite_exclude(test, pattern))
        elif pattern not in test.id():
            filtered.addTest(test)
    return filtered


def run_test_dir(subdir, title, top_level_dir):
    """Discover and run every test under *top_level_dir*/*subdir*, then exit.

    Single implementation of the runner boilerplate shared by the three entry
    scripts (integration_test.py / cli_unit_test.py / unit_tests.py): it handles
    ``-h``/``--help``, ``-k``/``--keyword`` filtering, ``-l``/``--list``, the
    root-privilege check, the legend/title preamble, output buffering
    (``-b``/``--buffer``) and the GTest-style summary runner.  Never returns —
    always calls ``sys.exit()`` (0 on success, 1 on failure).
    """
    argv = sys.argv

    # Skip legend/title/"Running" preamble when the user just wants help text.
    if "-h" in argv or "--help" in argv:
        print_unittest_help()
        print_amdsmi_path_help()
        sys.exit(0)

    loader = unittest.TestLoader()
    # -k does substring matching; wrapping in * lets globs (test_ttm*) match too.
    k_pattern = _parse_k_pattern(argv)
    if k_pattern:
        loader.testNamePatterns = [f"*{k_pattern}*"]

    suite = loader.discover(
        start_dir=os.path.join(top_level_dir, subdir),
        pattern="test_*.py",
        top_level_dir=top_level_dir,
    )

    # -x/--exclude drops any test whose id contains the substring (the inverse of
    # -k), e.g. run everything but the perf suites with `-x performance`.
    x_pattern = _parse_x_pattern(argv)
    if x_pattern:
        suite = _filter_suite_exclude(suite, x_pattern)

    if "-l" in argv or "--list" in argv:
        print_test_ids(suite)
        sys.exit(0)

    # Detect if ran without sudo or root privileges
    if os.geteuid() != 0:
        print(
            "Warning: Some tests may require elevated privileges (sudo/root) to run completely.\n",
            file=sys.stderr,
        )
        print("Please relaunch with elevated privileges.\n", file=sys.stderr)
        sys.exit(1)

    # Only show the dot-character legend when not in verbose mode; in verbose
    # mode each test prints its own result line so the dot legend is irrelevant.
    if verbose < VERBOSITY_VERBOSE:
        print_legend()

    if verbose > VERBOSITY_QUIET:
        print(f"{title}\n")
        print("Running tests...\n")

    runner = GTestSummaryRunner(
        stream=sys.stderr,
        verbosity=make_runner_verbosity(verbose),
        buffer="-b" in argv or "--buffer" in argv,
    )
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


class GTestSummaryRunner(unittest.TextTestRunner):
    """TextTestRunner that appends a GTest-style pass/skip/fail summary.

    After the standard unittest output, prints a colored block like::

        [----------] 40 tests ran.
        [  PASSED  ] 39 tests.
        [  SKIPPED ] 1 test, listed below:
        [  SKIPPED ] TestClass.test_method_name

    Color scheme mirrors GTest:
        cyan   = separator line ([----------])
        green  = PASSED
        yellow = SKIPPED
        red    = FAILED

    Colors are automatically suppressed when the output stream is not a TTY
    (e.g. when piped to a file or CI log capture), so file-based runners such
    as those in perf_tests.py will never receive ANSI escape codes in their logs.

    Example::

        runner = common.GTestSummaryRunner(verbosity=common.make_runner_verbosity(verbose))
        unittest.main(testRunner=runner)

        # To redirect output to stderr (e.g. when stdout is used for data):
        runner = common.GTestSummaryRunner(stream=sys.stderr, verbosity=common.make_runner_verbosity(verbose))
    """

    _CYAN = "\033[36m"
    _GREEN = "\033[32m"
    _YELLOW = "\033[33m"
    _RED = "\033[31m"
    _RESET = "\033[0m"

    @staticmethod
    def _plural(n):
        return "s" if n != 1 else ""

    @staticmethod
    def _test_label(test):
        """Return the GTest-format label for *test*.

        ``TestCase.id()`` is the public API for a test's full dotted name
        (e.g. ``"__main__.ClassName.method_name"``).  When the id starts with
        the standard ``__main__.`` prefix we strip it, yielding
        ``ClassName.method_name`` without risking label collisions between
        tests from different modules that share a class name.
        """
        test_id = test.id()
        if test_id.startswith("__main__."):
            return test_id[len("__main__.") :]
        return test_id

    def _color(self, code, text):
        """Wrap *text* in *code* only when colors are appropriate.

        Colors are suppressed when the ``NO_COLOR`` environment variable is set
        (https://no-color.org/) or when the output stream is not a real TTY.
        """
        if os.environ.get("NO_COLOR"):
            return text
        underlying = getattr(self.stream, "stream", self.stream)
        if hasattr(underlying, "isatty") and underlying.isatty():
            return f"{code}{text}{self._RESET}"
        return text

    def run(self, test):
        start = time.perf_counter()
        result = super().run(test)
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        self._print_gtest_summary(result, elapsed_ms)
        return result

    def _print_gtest_summary(self, result, elapsed_ms):
        """Write the GTest-style pass/skip/fail block to *self.stream*."""
        stream = self.stream  # unittest._WritelnDecorator, supports .writeln()
        skipped = len(result.skipped)
        # unexpectedSuccesses are tests marked @expectedFailure that passed instead.
        # They cause wasSuccessful() to return False, so count them as failures so
        # the summary accurately reflects the overall run status.
        unexpectedSuccesses = len(getattr(result, "unexpectedSuccesses", []))
        failures = len(result.failures) + len(result.errors) + unexpectedSuccesses
        passed = result.testsRun - skipped - failures

        stream.writeln()
        stream.writeln(
            self._color(
                self._CYAN,
                f"[----------] {result.testsRun} test{self._plural(result.testsRun)} ran. ({elapsed_ms} ms total)",
            )
        )
        stream.writeln(
            self._color(self._GREEN, f"[  PASSED  ] {passed} test{self._plural(passed)}.")
        )

        if failures:
            stream.writeln(
                self._color(
                    self._RED,
                    f"[  FAILED  ] {failures} test{self._plural(failures)}, listed below:",
                )
            )
            for t, _ in result.failures:
                stream.writeln(self._color(self._RED, f"[  FAILED  ] {self._test_label(t)}"))
            for t, _ in result.errors:
                stream.writeln(self._color(self._RED, f"[  FAILED  ] {self._test_label(t)}"))
            # unexpectedSuccesses items are bare TestCase instances (not (test, traceback) tuples).
            for t in getattr(result, "unexpectedSuccesses", []):
                stream.writeln(
                    self._color(
                        self._RED, f"[  FAILED  ] {self._test_label(t)} (unexpected success)"
                    )
                )
        if skipped:
            stream.writeln(
                self._color(
                    self._YELLOW,
                    f"[  SKIPPED ] {skipped} test{self._plural(skipped)}, listed below:",
                )
            )
            for t, _ in result.skipped:
                stream.writeln(self._color(self._YELLOW, f"[  SKIPPED ] {self._test_label(t)}"))
        stream.writeln()


def has_gpu_od_interface(bdf):
    """Check if a GPU has the gpu_od sysfs interface.

    Delegates to amdsmi_helpers.AMDSMIHelpers.detect_gpu_od(). Requires the
    AMD-SMI CLI to be installed (amdsmi_helpers is imported on first call).

    Args:
        bdf: PCI Bus/Device/Function string (e.g. '0000:26:00.0')

    Returns:
        bool: True if gpu_od directory exists for this GPU
    """
    # TODO(amdsmi_team): Refactor to create an amdsmi_get_gpu_fan_speed_range() API
    #                    amdsmi_get_gpu_fan_speed_range(
    #                                           amdsmi_processor_handle processor_handle,
    #                                           uint32_t sensor_ind,
    #                                           amdsmi_range_t* fan_speed_range)
    # and begin deprecation of amdsmi_get_gpu_fan_speed_max(). This will allow us
    # to remove the dependency on amdsmi_helpers from this common module.
    # Exposing non-public SYSFS API interfaces in the CLI (and in general)
    # is a bad design pattern and needs to be addressed in the future.
    amdsmi_cli_path = os.path.normpath(
        os.path.join(amdsmi_path, "..", "..", "libexec", "amdsmi_cli")
    )
    if not os.path.exists(amdsmi_cli_path):
        raise FileNotFoundError(
            f'amdsmi_cli path "{amdsmi_cli_path}" does not exist. '
            f"Ensure the AMD-SMI CLI is installed, or set AMDSMI_PATH correctly."
        )
    if amdsmi_cli_path not in sys.path:
        sys.path.append(amdsmi_cli_path)
    try:
        import amdsmi_helpers  # type: ignore
    except ImportError as e:
        raise ImportError(
            f'Could not import the "amdsmi_helpers" module from "{amdsmi_cli_path}"'
        ) from e
    has_gpu_od, _ = amdsmi_helpers.AMDSMIHelpers.detect_gpu_od(bdf)
    return has_gpu_od


class Common:
    VIRTUALIZATION_MODE_MAP = {}
    for member in amdsmi.AmdSmiVirtualizationMode:
        VIRTUALIZATION_MODE_MAP[amdsmi.AmdSmiVirtualizationMode(member.value)] = member.name

    def __init__(self, verbose, *args, **kwargs):
        self.verbose = verbose
        self.max_num_physical_devices = (
            amdsmi.amdsmi_interface.AMDSMI_MAX_NUM_XCP * amdsmi.amdsmi_interface.AMDSMI_MAX_DEVICES
        )
        self.PASS = PASS
        self.FAIL = FAIL
        self.ANY_FAIL = "ANY_FAIL"

        # Tests marked with either of these flags will be skipped
        # and need to be implemented later.
        self.TODO_SKIP_FAIL = True
        self.TODO_SKIP_NOT_COMPLETE = True

        try:
            self.amdsmi_smart_init()

            # Get gpus
            self.processors = amdsmi.amdsmi_get_processor_handles()
            # Set bad gpu: None is rejected by the isinstance check in amdsmi_interface.py,
            # raising AmdSmiParameterException(INVAL). Platform-agnostic and unambiguous.
            self.bad_gpu = None

            self.virt_mode = []
            self.asic_info = []
            self.board_info = []
            for gpu in self.processors:
                # Get virtualization mode info
                try:
                    ret = amdsmi.amdsmi_get_gpu_virtualization_mode(gpu)
                    mode_name = self.VIRTUALIZATION_MODE_MAP.get(ret["mode"], "UNKNOWN")
                    self.virt_mode.append({"mode": mode_name})
                except amdsmi.AmdSmiLibraryException:
                    # Virtualization mode is not supported on every ASIC; record
                    # UNKNOWN silently (it is still surfaced via print_device_header).
                    self.virt_mode.append({"mode": "UNKNOWN"})

                # Get asic info
                self.asic_info.append(amdsmi.amdsmi_get_gpu_asic_info(gpu))
                # Get board info
                self.board_info.append(amdsmi.amdsmi_get_gpu_board_info(gpu))

            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiLibraryException as e:
            print(f"In class Common, Cannot get processor information, {e}")

        self.not_supported_error_codes = [
            ("2", "AMDSMI_STATUS_NOT_SUPPORTED"),
            ("3", "AMDSMI_STATUS_NOT_YET_IMPLEMENTED"),
            ("49", "AMDSMI_STATUS_NO_HSMP_MSG_SUP"),
        ]

        self.error_map = ERROR_MAP
        # All enum-derived parameter lists are built by the shared, hardware-free
        # build_type_lists() so the logic lives in one place (also importable by
        # the standalone benchmark suites).
        for _attr, _value in build_type_lists().items():
            setattr(self, _attr, _value)
        return

    def print(self, msg, data=None):
        if self.verbose > VERBOSITY_QUIET:
            if data is None:
                print(msg, flush=True)
            elif any(data in value for value in self.not_supported_error_codes):
                print(f"{msg} {data}", flush=True)
            else:
                if isinstance(data, str) and data in self.error_map.values():
                    print(msg, end="")
                else:
                    print(msg)
                if isinstance(data, dict) or isinstance(data, list):
                    print(json.dumps(data, sort_keys=False, indent=4), flush=True)
                else:
                    print(data)
        return

    def print_func_name(self, msg=None):
        if self.verbose == VERBOSITY_VERBOSE:
            stk = inspect.stack()
            if stk[1].function == "_callSetUp":
                return
            print(f"\n## {stk[1].function}()", flush=True)
            if msg:
                print(msg, flush=True)
        return

    def print_device_header(self, i):
        # Guard against stale caches: processors may be refreshed in setUp without
        # refreshing these display-only lists (which are populated once at __init__).
        # Print virtualization mode info
        msg = f"virtualization mode(gpu={i})"
        self.print(f"\t{msg}")
        if i < len(self.virt_mode):
            mode = self.virt_mode[i]["mode"]
            self.print(f"\t\tmode : {mode}")
        # Print asic info
        msg = f"asic info(gpu={i})"
        self.print(f"\t{msg}")
        if i < len(self.asic_info):
            for key, value in self.asic_info[i].items():
                self.print(f"\t\t{key} : {value}")
        # Print board info
        msg = f"board info(gpu={i})"
        self.print(f"\t{msg}")
        if i < len(self.board_info):
            for key, value in self.board_info[i].items():
                self.print(f"\t\t{key} : {value}")
        return

    def get_error_code(self, exc):
        error_code = "-1"
        error_code_name = "UNKNOWN_ERROR"
        if hasattr(exc, "get_error_code"):
            error_code = str(exc.get_error_code())
            if error_code in self.error_map:
                error_code_name = self.error_map[error_code]
        return (error_code, error_code_name)

    def get_dict_key_from_value(self, _value, _dict):
        if _value not in _dict.values():
            return None
        for key, value in _dict.items():
            if value == _value:
                return key
        return None

    def get_error_code_from_name(self, error_code_name):
        error_code = self.get_dict_key_from_value(error_code_name, self.error_map)
        if error_code is None:
            error_code = -1
        return error_code

    def check_ret(self, msg, exc, expected_code_name=None, printIt=True):
        # Returns True if the test FAILED (i.e. the result did not match expected).
        # Callers use the pattern: `if self.check_ret(...): raise_exception = e`
        if isinstance(exc, str) and len(exc) == 0:
            error_code_name = expected_code_name
            error_code = "-1"
            if error_code_name in self.error_map.values():
                for key, value in self.error_map.items():
                    if value == error_code_name:
                        error_code = key
                        break
        elif hasattr(exc, "get_error_code"):
            error_code, error_code_name = self.get_error_code(exc)
        else:
            error_code = str(exc).split(":", maxsplit=1)[0]
            error_code_name = "AMDSMI_STATUS_INVAL"

        # Check for when there are multiple passing conditions
        if isinstance(expected_code_name, list):
            for ec_name in expected_code_name:
                if not self.check_ret(msg, exc, ec_name, False):  # check without printing
                    # This expected code matched - print once and return success
                    if self.verbose > VERBOSITY_QUIET and printIt:
                        if msg:
                            print(f"{msg}\n", end="")
                        ec_code = self.get_error_code_from_name(ec_name)
                        print(f"\tTEST SUCCESS, AMDSMI API Returned {ec_code:>2s}, {ec_name}")
                    return False

            # No expected result matched - print failure (respects same guards as single-condition path)
            if self.verbose > VERBOSITY_QUIET and printIt:
                if msg:
                    print(f"{msg}\n", end="")
                status_msg = (
                    f"\tTEST FAILURE, AMDSMI API Returned {error_code:>2s}, {error_code_name}\n"
                )
                for ec_name in expected_code_name:
                    ec_code = self.get_error_code_from_name(ec_name)
                    status_msg += (
                        f"\t              AMDSMI API Expected {ec_code:>2s}, {expected_code_name}"
                    )
            return True

        # Check for single passing condition
        status_msg = ""
        status_ret = False
        if any(error_code in value for value in self.not_supported_error_codes):
            status_msg = f"\tAMDSMI API Returned {error_code:>2s}, {error_code_name}"
        elif error_code_name == expected_code_name:
            status_msg = f"\tTEST SUCCESS, AMDSMI API Returned expected {error_code:>2s}, {expected_code_name}"
        elif error_code_name != self.PASS and expected_code_name == self.ANY_FAIL:
            status_msg = f"\tTEST SUCCESS, AMDSMI API Returned expected fail {error_code:>2s}, {expected_code_name}"
        else:
            expected_error_code = self.get_error_code_from_name(expected_code_name)
            status_msg = (
                f"\tTEST FAILURE, AMDSMI API Returned {error_code:>2s}, {error_code_name}\n"
            )
            status_msg += f"\t              AMDSMI API Expected {expected_error_code:>2s}, {expected_code_name}"
            status_ret = True
        if self.verbose > VERBOSITY_QUIET and printIt:
            if msg:
                print(f"{msg}\n", end="")
            print(f"{status_msg}", flush=True)
        return status_ret

    def _check_amdgpu_driver(self):
        """Returns true if amdgpu is found in the list of initialized modules"""
        amd_gpu_status_file = pathlib.Path("/sys/module/amdgpu/initstate")
        if amd_gpu_status_file.exists():
            try:
                return amd_gpu_status_file.read_text(encoding="ascii").strip() == "live"
            except OSError:
                pass

        # If the driver is loaded either as a module OR built in, this dir will be populated
        drv = pathlib.Path("/sys/bus/pci/drivers/amdgpu")
        if not drv.exists():
            return False

        # Check if a symlink exists that loosely matches PCI BDF format
        # ex: 0000:03:00.0
        for p in drv.iterdir():
            if p.is_symlink() and ":" in p.name and "." in p.name:
                return True
        return False

    def _check_amd_hsmp_driver(self):
        """Returns true if amd_hsmp or hsmp_acpi is found in the list of initialized modules"""
        return pathlib.Path("/dev/hsmp").exists()

    def _init_with_flag(self, init_flag, driver_msg):
        ret = None
        try:
            ret = amdsmi.amdsmi_init(init_flag)
        except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
            # AmdSmiLibraryException exposes get_error_code(); AmdSmiParameterException does not.
            # Accessing .err_code on AmdSmiParameterException raises AttributeError.
            err_code = e.get_error_code() if hasattr(e, "get_error_code") else None
            if err_code in (amdsmi.AmdSmiStatus.NOT_INIT, amdsmi.AmdSmiStatus.DRIVER_NOT_LOADED):
                self.print(driver_msg)
                raise unittest.SkipTest(driver_msg)
            raise
        return ret

    def _skip_if_missing(self, names):
        # All public APIs must be re-exported at the top-level amdsmi namespace.
        # If an API is missing there, it belongs in amdsmi_interface.py, not a sub-module.
        # Note: self.id() is NOT used here — Common is not a TestCase subclass, so self.id()
        # does not exist when called through a Common instance. Instead, walk the call stack to
        # find the name of the enclosing test_* method, which works regardless of calling convention.
        missing = [name for name in names if not hasattr(amdsmi, name)]
        if missing:
            test_name = next(
                (frame.function for frame in inspect.stack() if frame.function.startswith("test_")),
                "<unknown>",
            )
            msg = f"{test_name} | Missing amdsmi API(s) in amdsmi_interface.py: " + ", ".join(
                missing
            )
            # self may be a Common instance (has .verbose directly) or a TestCase instance
            # that stores Common as self.common (when called via common.Common._skip_if_missing(self, ...)).
            verbose = getattr(self, "verbose", None)
            if verbose is None:
                verbose = getattr(getattr(self, "common", None), "verbose", VERBOSITY_NORMAL)
            if verbose > VERBOSITY_QUIET:
                print(msg, file=sys.stderr)
            raise unittest.SkipTest(msg)
        return

    def _build_call_msg(self, func_name, i, j, params):
        msg = f"\t### {func_name}("
        if i is not None:
            msg += f"gpu={i}"
        if j is not None:
            msg += f", gpu={j}"
        for param_name, param_value in params.items():
            if isinstance(param_value, list):
                msg += f", {param_name}={{value}}"
            else:
                msg += f", {param_name}={param_value}"
        msg += ")"
        return msg

    def amdsmi_smart_init(self):
        """Initializes AMDSMI Library based on live drivers found in the system.

        Checks for the presence of the amdgpu, amd_hsmp or hsmp_acpi drivers and initializes the
        AMD SMI library based on the live drivers found. Mirrors the flag-building logic in the
        CLI (amdsmi_init.py): flags are OR-combined additively so that systems with both a
        discrete GPU and CPU power management (non-APU servers) work correctly. INIT_AMD_APUS
        is intentionally not used here; it targets integrated APU configurations only.

        Return:
            tuple: A tuple containing:
                - ret: The return value from amdsmi_init() (typically None on success)
                - init_flag: The combined flag used to initialize the AMD SMI library
                    (bitwise OR of INIT_AMD_GPUS and/or INIT_AMD_CPUS)

        Raises:
            AmdSmiLibraryException: If initialization fails for reasons other than driver not loaded
            AmdSmiParameterException: If invalid parameters are passed to amdsmi_init
            unittest.SkipTest: If no compatible AMD drivers are detected on the system
        """
        # Build init_flag additively from detected drivers (mirrors CLI amdsmi_init.py pattern).
        # Using INIT_AMD_APUS when both GPU and CPU drivers are present is wrong on discrete-GPU
        # servers — use OR-combined flags instead, exactly as the CLI does.
        init_flag = 0
        if self._check_amdgpu_driver():
            init_flag |= amdsmi.AmdSmiInitFlags.INIT_AMD_GPUS
        if self._check_amd_hsmp_driver():
            init_flag |= amdsmi.AmdSmiInitFlags.INIT_AMD_CPUS
        if init_flag == 0:
            msg = "Drivers not loaded (amdgpu, amd_hsmp or hsmp_acpi drivers not found in modules)"
            self.print(msg)
            raise unittest.SkipTest(msg)

        msg = "AMDSMI init failed for detected drivers"
        ret = self._init_with_flag(init_flag, msg)
        return (ret, init_flag)

    def Test_API(self, **kwargs):
        """
        Tests API with zero or more arguments

        Arguments:
            func_name: API to be executed
        Optional:
            param1_name: Name of parameter 1
        """

        params = kwargs
        iterator = iter(params.items())
        func_name, func = next(iterator)
        del params[func_name]

        raise_exception = None
        cond = self.PASS
        msg = self._build_call_msg(func_name, None, None, params)
        try:
            data = func(*[value for value in params.values()])
            self.print(msg, data)
            self.check_ret("", "", cond)
        except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
            if self.check_ret(msg, e, cond):
                raise_exception = e
        if raise_exception:
            raise raise_exception
        return

    def Test_API_Per_GPU(self, **kwargs):
        """
        Tests API per GPU with zero or more arguments

        Arguments:
            func_name: API to be executed
        Optional:
            param1_name: Name of parameter 1
            param2_name: Name of parameter 2
            param3_name: Name of parameter 3
        """

        params = kwargs
        iterator = iter(params.items())
        func_name, func = next(iterator)
        del params[func_name]

        raise_exception = None
        for i in range(len(self.processors) + 1):
            cond = self.PASS
            if i < len(self.processors):
                gpu = self.processors[i]
                self.print_device_header(i)
            else:
                # bad gpu
                gpu = self.bad_gpu
                i = "invalid"
                cond = self.FAIL

            msg = self._build_call_msg(func_name, i, None, params)
            try:
                data = func(gpu, *[value for value in params.values()])
                self.print(msg, data)
                self.check_ret("", "", cond)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.check_ret(msg, e, cond):
                    raise_exception = e
            self.print("")
        if raise_exception:
            raise raise_exception
        return

    def Test_Per_GPU_With_One_Enum(self, **kwargs):
        """
        Tests API per GPU per Enum with zero or more arguments

        Arguments:
            func_name: API to be executed
            value1_name=[(value_name, value, value_cond), ...]
        Optional:
            param1_name: Name of parameter 1
            param2_name: Name of parameter 2
        """

        params = kwargs
        iterator = iter(params.items())
        func_name, func = next(iterator)
        _, values1 = next(iterator)
        del params[func_name]

        raise_exception = None
        for i in range(len(self.processors) + 1):
            if i < len(self.processors):
                gpu = self.processors[i]
                self.print_device_header(i)
            else:
                # bad gpu
                gpu = self.bad_gpu
                i = "invalid"

            for value1_name, value1, value1_cond in values1:
                cond = self.PASS
                if i == "invalid" or value1_cond == self.FAIL:
                    cond = self.FAIL
                msg = self._build_call_msg(func_name, i, None, params)
                msg = msg.replace("{value}", value1_name, 1)
                try:
                    data = func(
                        gpu,
                        *[
                            value if not isinstance(value, list) else value1
                            for value in params.values()
                        ],
                    )
                    self.print(msg, data)
                    self.check_ret("", "", cond)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if value1_cond != self.PASS:
                        if self.check_ret(msg, e, value1_cond):
                            raise_exception = e
                    else:
                        if self.check_ret(msg, e, cond):
                            raise_exception = e
                self.print("")
        if raise_exception:
            raise raise_exception
        return

    def Test_Per_GPU_With_Two_Enums(self, **kwargs):
        """
        Tests API per GPU per 2 Enums with zero or more arguments

        Arguments:
            func_name: API to be executed
            value1_name=[(value_name, value, value_cond), ...]
            value2_name=[(value_name, value, value_cond), ...]
        Optional:
            param1_name: Name of parameter 1
            param2_name: Name of parameter 2
        """

        params = kwargs
        iterator = iter(params.items())
        func_name, func = next(iterator)
        _, values1 = next(iterator)
        _, values2 = next(iterator)
        del params[func_name]

        raise_exception = None
        for i in range(len(self.processors) + 1):
            if i < len(self.processors):
                gpu = self.processors[i]
                self.print_device_header(i)
            else:
                # bad gpu
                gpu = self.bad_gpu
                i = "invalid"

            for value1_name, value1, value1_cond in values1:
                for value2_name, value2, value2_cond in values2:
                    cond = self.PASS
                    if i == "invalid" or value1_cond == self.FAIL or value2_cond == self.FAIL:
                        cond = self.FAIL
                    msg = self._build_call_msg(func_name, i, None, params)
                    msg = msg.replace("{value}", value1_name, 1)
                    msg = msg.replace("{value}", value2_name, 1)
                    try:
                        data = func(
                            gpu,
                            *[
                                value
                                if not isinstance(value, list)
                                else value1
                                if index == 0
                                else value2
                                for index, value in enumerate(params.values())
                            ],
                        )
                        self.print(msg, data)
                        self.check_ret("", "", cond)
                    except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                        if value1_cond != self.PASS:
                            if self.check_ret(msg, e, value1_cond):
                                raise_exception = e
                        elif value2_cond != self.PASS:
                            if self.check_ret(msg, e, value2_cond):
                                raise_exception = e
                        else:
                            if self.check_ret(msg, e, cond):
                                raise_exception = e
                    self.print("")
        if raise_exception:
            raise raise_exception
        return

    def Test_Per_GPU_With_GPU(self, **kwargs):
        """
        Tests API per GPU per GPU with zero or more arguments

        Arguments:
            func_name: API to be executed
        Optional:
            param1_name: Name of parameter 1
            param2_name: Name of parameter 2
        """
        params = kwargs
        iterator = iter(params.items())
        func_name, func = next(iterator)
        del params[func_name]

        raise_exception = None
        for i in range(len(self.processors) + 1):
            if i < len(self.processors):
                gpu_i = self.processors[i]
                self.print_device_header(i)
            else:
                # bad gpu
                gpu_i = self.bad_gpu
                i = "invalid"
            for j in range(len(self.processors) + 1):
                if j < len(self.processors):
                    gpu_j = self.processors[j]
                    self.print_device_header(j)
                else:
                    # bad gpu
                    gpu_j = self.bad_gpu
                    j = "invalid"

                cond = self.PASS
                if i == "invalid" or j == "invalid" or i == j:
                    cond = self.FAIL
                msg = self._build_call_msg(func_name, i, j, params)
                try:
                    data = func(gpu_i, gpu_j, *[value for value in params.values()])
                    self.print(msg, data)
                    self.check_ret("", "", cond)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.check_ret(msg, e, cond):
                        raise_exception = e
                self.print("")
        if raise_exception:
            raise raise_exception
        return
