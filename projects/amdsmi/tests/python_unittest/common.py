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

import inspect
import json
import os
import pathlib
import sys

import unittest

amdsmi_path = os.environ.get("AMDSMI_PATH", "/opt/rocm/share/amd_smi")
if not os.path.exists(amdsmi_path):
    raise FileNotFoundError(
        f'AMDSMI_PATH "{amdsmi_path}" does not exist. Please set the correct path in your environment.'
    )
sys.path.append(amdsmi_path)
try:
    import amdsmi
except ImportError as e:
    raise ImportError(f'Could not import the "amdsmi" module from "{amdsmi_path}"') from e

#################################################
# Module level functions, not part of the class #
#################################################

VERBOSITY_QUIET = 0  # -q / --quiet
VERBOSITY_NORMAL = 1  # default (dot-per-test)
VERBOSITY_VERBOSE = 2  # -v / --verbose (per-test result lines)


def print_test_ids(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            print_test_ids(test)
        else:
            test = str(test).split()[0]
            print(f"\t{test}", file=sys.stderr)
    return


def print_tests(module_name):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[module_name])
    print("=" * 70, file=sys.stderr)
    print("Available tests:", file=sys.stderr)
    print_test_ids(suite)
    return


def print_legend():
    # Provide Legend for test results, otherwise it is not clear what the output means
    print("=" * 70, file=sys.stderr)
    print("Legend: . = pass, s = skipped, F = fail, E = error", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    return


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


def expand_glob_k_arg(caller_globals):
    """Expand a glob pattern in a -k/--keyword argument into individual -k flags.

    Python's unittest -k flag does substring matching only — wildcards like
    'test_ttm*' are treated as literal strings and match nothing.  This function
    detects when a glob pattern is supplied, finds all test method names from
    the caller's globals that match it, and rewrites sys.argv to use one -k per
    match.  unittest treats multiple -k flags as OR, so the result is equivalent
    to the intended glob.

    Call this from the __main__ block of any test file, passing globals():
        common.expand_glob_k_arg(globals())

    Example: -k "test_ttm*"  →  -k test_ttm_info -k test_ttm_set_dry_run
    """
    import fnmatch

    for flag in ("-k", "--keyword"):
        try:
            idx = sys.argv.index(flag)
        except ValueError:
            continue

        pattern = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""
        if "*" not in pattern and "?" not in pattern:
            break  # plain substring — nothing to expand

        # Collect every test method name from all TestCase subclasses in the caller's module
        all_test_names = []
        for obj in list(caller_globals.values()):
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
                all_test_names.extend(m for m in dir(obj) if m.startswith("test"))
        # Deduplicate while preserving order
        all_test_names = list(dict.fromkeys(all_test_names))

        matches = [n for n in all_test_names if fnmatch.fnmatch(n, pattern)]
        if matches:
            # Remove the single "-k glob*" pair and insert one "-k name" per match
            del sys.argv[idx : idx + 2]
            for i, name in enumerate(matches):
                sys.argv.insert(idx + i * 2, flag)
                sys.argv.insert(idx + i * 2 + 1, name)
        break


class Common:
    VIRTUALIZATION_MODE_MAP = {}
    for member in amdsmi.AmdSmiVirtualizationMode:
        VIRTUALIZATION_MODE_MAP[amdsmi.AmdSmiVirtualizationMode(member.value)] = member.name

    def __init__(self, verbose, *args, **kwargs):
        self.verbose = verbose
        self.max_num_physical_devices = (
            amdsmi.amdsmi_interface.AMDSMI_MAX_NUM_XCP * amdsmi.amdsmi_interface.AMDSMI_MAX_DEVICES
        )
        self.PASS = "AMDSMI_STATUS_SUCCESS"
        self.FAIL = "AMDSMI_STATUS_INVAL"
        self.ANY_FAIL = "ANY_FAIL"

        # Tests marked wtih either of these flags will be skipped
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
                except amdsmi.AmdSmiLibraryException as e:
                    if self.verbose > VERBOSITY_QUIET:
                        print(
                            f"In class Common, Cannot get virtualization mode information for gpu {gpu}, {e}"
                        )
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

        self.error_map = {}
        self.status_types = []
        for member in amdsmi.AmdSmiStatus:
            self.error_map[str(member.value)] = f"AMDSMI_STATUS_{member.name}"
            self.status_types.append((member.name, amdsmi.AmdSmiStatus(member.value), self.PASS))

        self.clk_types = []
        for member in amdsmi.AmdSmiClkType:
            cond = self.PASS
            if member.name in ["DCEF", "PCIE"]:
                cond = [self.PASS, self.FAIL]
            self.clk_types.append((member.name, amdsmi.AmdSmiClkType(member.value), cond))

        self.clk_limit_types = []
        for member in amdsmi.AmdSmiClkLimitType:
            self.clk_limit_types.append(
                (member.name, amdsmi.AmdSmiClkLimitType(member.value), self.PASS)
            )

        # AmdSmiIOBWEncoding is not yet part of the public amdsmi interface.
        # Discover the wrapper symbols by convention (AGG_BW0, RD_BW0, WR_BW0).
        # The assertion guards against a silent empty list if symbols are renamed.
        self.io_bw_encodings = [
            (name, getattr(amdsmi.amdsmi_wrapper, name), self.PASS)
            for name in sorted(dir(amdsmi.amdsmi_wrapper))
            if name.endswith("_BW0")
        ]
        assert self.io_bw_encodings, (
            "No *_BW0 symbols found in amdsmi_wrapper — "
            "amdsmi_get_cpu_current_io_bandwidth tests would pass with zero coverage. "
            "Check that AGG_BW0/RD_BW0/WR_BW0 are still exported by the wrapper."
        )

        self.gpu_blocks = []
        for member in amdsmi.AmdSmiGpuBlock:
            cond = self.PASS
            if member.name in ["INVALID", "RESERVED"]:
                cond = self.FAIL
            self.gpu_blocks.append((member.name, amdsmi.AmdSmiGpuBlock(member.value), cond))

        self.memory_types = []
        for member in amdsmi.AmdSmiMemoryType:
            self.memory_types.append(
                (member.name, amdsmi.AmdSmiMemoryType(member.value), self.PASS)
            )

        self.reg_types = []
        for member in amdsmi.AmdSmiRegType:
            self.reg_types.append((member.name, amdsmi.AmdSmiRegType(member.value), self.PASS))

        self.voltage_metrics = []
        for member in amdsmi.AmdSmiVoltageMetric:
            self.voltage_metrics.append(
                (member.name, amdsmi.AmdSmiVoltageMetric(member.value), self.PASS)
            )

        self.voltage_types = []
        for member in amdsmi.AmdSmiVoltageType:
            cond = self.PASS
            if member.name in ["INVALID"]:
                cond = self.FAIL
            self.voltage_types.append((member.name, amdsmi.AmdSmiVoltageType(member.value), cond))

        self.link_types = []
        for member in amdsmi.AmdSmiLinkType:
            cond = self.PASS
            if member.name in ["AMDSMI_LINK_TYPE_NOT_APPLICABLE", "AMDSMI_LINK_TYPE_UNKNOWN"]:
                cond = self.FAIL
            self.link_types.append((member.name, amdsmi.AmdSmiLinkType(member.value), cond))

        self.temperature_types = []
        for member in amdsmi.AmdSmiTemperatureType:
            self.temperature_types.append(
                (member.name, amdsmi.AmdSmiTemperatureType(member.value), self.PASS)
            )

        self.temperature_metrics = []
        for member in amdsmi.AmdSmiTemperatureMetric:
            self.temperature_metrics.append(
                (member.name, amdsmi.AmdSmiTemperatureMetric(member.value), self.PASS)
            )

        self.utilization_counter_types = []
        for member in amdsmi.AmdSmiUtilizationCounterType:
            self.utilization_counter_types.append(
                (member.name, amdsmi.AmdSmiUtilizationCounterType(member.value), self.PASS)
            )
        # Negative test: integer 100 is out of range for AmdSmiUtilizationCounterType;
        # the API must reject it. Not a real enum member so must be added explicitly.
        self.utilization_counter_types.append(("UTILIZATION_COUNTER_BAD", 100, self.FAIL))

        self.event_groups = []
        for member in amdsmi.AmdSmiEventGroup:
            cond = self.PASS
            if member.name in ["GRP_INVALID"]:
                cond = self.FAIL
            self.event_groups.append((member.name, amdsmi.AmdSmiEventGroup(member.value), cond))

        self.event_types = []
        for member in amdsmi.AmdSmiEventType:
            self.event_types.append((member.name, amdsmi.AmdSmiEventType(member.value), self.PASS))

        self.counter_commands = []
        for member in amdsmi.AmdSmiCounterCommand:
            self.counter_commands.append(
                (member.name, amdsmi.AmdSmiCounterCommand(member.value), self.PASS)
            )

        self.compute_partition_types = []
        for member in amdsmi.AmdSmiComputePartitionType:
            cond = self.PASS
            if member.name in ["INVALID"]:
                cond = self.FAIL
            self.compute_partition_types.append(
                (member.name, amdsmi.AmdSmiComputePartitionType(member.value), cond)
            )

        self.memory_partition_types = []
        for member in amdsmi.AmdSmiMemoryPartitionType:
            cond = self.PASS
            if member.name in ["UNKNOWN"]:
                cond = self.FAIL
            elif member.name in ["NPS4", "NPS8"]:
                # NPS4/NPS8 are hardware-dependent; accept success or invalid depending on support
                # BTW - no asic supports NPS8...
                cond = [self.PASS, self.FAIL]
            self.memory_partition_types.append(
                (member.name, amdsmi.AmdSmiMemoryPartitionType(member.value), cond)
            )

        self.freq_inds = []
        for member in amdsmi.AmdSmiFreqInd:
            cond = self.PASS
            if member.name in ["INVALID"]:
                cond = self.FAIL
            self.freq_inds.append((member.name, amdsmi.AmdSmiFreqInd(member.value), cond))

        self.power_profile_preset_masks = []
        for member in amdsmi.AmdSmiPowerProfilePresetMasks:
            cond = self.PASS
            if member.name in ["INVALID"]:
                # INVALID (0xFFFFFFFFFFFFFFFF) is intentionally tested with a specific expected
                # status rather than the generic self.FAIL (AMDSMI_STATUS_INVAL). The API returns
                # AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS for this sentinel, not AMDSMI_STATUS_INVAL.
                cond = "AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS"
            self.power_profile_preset_masks.append(
                (member.name, amdsmi.AmdSmiPowerProfilePresetMasks(member.value), cond)
            )

        self.processor_types = []
        for member in amdsmi.AmdSmiProcessorType:
            cond = self.PASS
            if member.name in ["UNKNOWN"]:
                cond = self.FAIL
            self.processor_types.append(
                (member.name, amdsmi.AmdSmiProcessorType(member.value), cond)
            )

        self.dev_perf_levels = []
        for member in amdsmi.AmdSmiDevPerfLevel:
            cond = self.PASS
            if member.name in ["UNKNOWN"]:
                cond = self.FAIL
            self.dev_perf_levels.append(
                (member.name, amdsmi.AmdSmiDevPerfLevel(member.value), cond)
            )
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

    def check_ret(self, msg, exc, expected_code_name=None, printIt=True):
        # Returns True if the test FAILED (i.e. the result did not match expected).
        # Callers use the pattern: `if self.check_ret(...): raise_exception = e`
        if isinstance(exc, str) and len(exc) == 0:
            error_code_name = expected_code_name
            if error_code_name in self.error_map.values():
                for key, value in self.error_map.items():
                    if value == error_code_name:
                        error_code = key
                        break
            else:
                error_code = "-1"
        elif hasattr(exc, "get_error_code"):
            error_code, error_code_name = self.get_error_code(exc)
        else:
            error_code = str(exc).split(":", maxsplit=1)[0]
            error_code_name = "AMDSMI_STATUS_INVAL"

        # Check for when there are multiple passing conditions
        if isinstance(expected_code_name, list):
            for ec in expected_code_name:
                if not self.check_ret(msg, exc, ec, False):  # check without printing
                    # This expected code matched - print once and return success
                    if self.verbose > VERBOSITY_QUIET and printIt:
                        if msg:
                            print(f"{msg}\n", end="")
                        print(f"\tTest PASSED with expected result {ec}", flush=True)
                    return False

            # No expected result matched - print failure (respects same guards as single-condition path)
            if self.verbose > VERBOSITY_QUIET and printIt:
                if msg:
                    print(f"{msg}\n", end="")
                print(
                    f"\tTest FAILED with expected results {expected_code_name} but received {error_code_name}",
                    flush=True,
                )
            return True

        # Check for single passing condition
        status_msg = ""
        status_ret = False
        if any(error_code in value for value in self.not_supported_error_codes):
            status_msg = f"\tAMDSMI API Returned {error_code_name}"
        elif error_code_name == expected_code_name:
            status_msg = f"\tTest PASSED with expected result {expected_code_name}"
        elif error_code_name != self.PASS and expected_code_name == self.ANY_FAIL:
            status_msg = f"\tTest PASSED with expected result {expected_code_name} and received {error_code_name}"
        else:
            status_msg = f"\tTest FAILED with expected result {expected_code_name} but received {error_code_name}"
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
            if err_code in (
                amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT,
                amdsmi.amdsmi_wrapper.AMDSMI_STATUS_DRIVER_NOT_LOADED,
            ):
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
        func_name, func = next(iterator, (None, None))
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
        func_name, func = next(iterator, (None, None))
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
        func_name, func = next(iterator, (None, None))
        _, values1 = next(iterator, (None, None))
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
        func_name, func = next(iterator, (None, None))
        _, values1 = next(iterator, (None, None))
        _, values2 = next(iterator, (None, None))
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
        func_name, func = next(iterator, (None, None))
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
