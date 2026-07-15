#!/usr/bin/env python3
"""
AMD SMI Complete API Test Script
================================

PURPOSE
-------
Run a capability inventory of the AMD SMI (`amdsmi`) Python module against a
host. Each known API is invoked and classified into one of five buckets, then a
markdown report is emitted.

REQUIREMENTS
------------
* Linux host with ROCm/AMDSMI installed and the `amdgpu` kernel driver loaded
  (`/dev/kfd` must exist).
* Python 3.8+ with the `amdsmi` Python bindings importable. If multiple
  Python interpreters are installed (e.g. `python3` and `python3.9`), this
  script will auto-detect one that has `amdsmi` and re-execute itself with
  that interpreter, so you normally do NOT need to pick the right one
  yourself. Override by invoking the script directly with the interpreter
  you want.
* For full coverage of write/setter APIs, run the script as root:
      sudo python3 test_amdsmi_all_apis.py
  Without root, every API tagged as a "write" call is reported as
  `Skipped (requires root)` rather than failing as `Unsupported`.

STATUS CATEGORIES
-----------------
* Supported       The API exists, was called, and returned successfully.
* Unsupported     The API exists but the underlying driver/hardware rejected
                  the call, OR the installed amdsmi build expects a different
                  call signature than this script provides
                  (`signature mismatch:` prefix in the note).
* Skipped         The API was intentionally not executed. Reasons include:
                  risky write (e.g. GPU reset, partition change), requires
                  root and the script was run unprivileged, a required enum
                  member is not present in this amdsmi build, or no usable
                  prerequisite (e.g. no active GPU process) was available.
* Script Error    A genuine bug in this test script (any non-signature
                  AttributeError/TypeError). Should be 0 on a clean run.
* API Not Found   The symbol is not present in the installed amdsmi module.

USAGE
-----
  python3 test_amdsmi_all_apis.py                          # primary GPU only
  sudo python3 test_amdsmi_all_apis.py                     # include write APIs
  python3 test_amdsmi_all_apis.py --gpu-index 1            # target a specific GPU
  python3 test_amdsmi_all_apis.py --all-gpus               # primary sweep + per-GPU sanity sweep
  python3 test_amdsmi_all_apis.py --output ./report.md     # custom markdown report path
  python3 test_amdsmi_all_apis.py --skip-writes            # never run setters
  python3 test_amdsmi_all_apis.py --no-revert              # do not restore mutated state

OUTPUT
------
A timestamped markdown report is written next to this script
(`test_amdsmi_all_apis_output_YYYYMMDD_HHMMSS.md`) unless `--output` is
supplied. The report contains a summary table, a per-API status row, and the
script version that produced it.
"""

__version__ = "1.1.0"

import argparse
import datetime
import importlib.util
import os
import platform
import shutil
import subprocess
import sys


# ----------------------------------------------------------------------------
# Auto-detect a Python interpreter that has the amdsmi module installed.
# This must run BEFORE `import amdsmi`. If the current interpreter doesn't
# have amdsmi but another candidate does, re-execute this script with that
# interpreter so the user doesn't have to know which one to pick.
# ----------------------------------------------------------------------------
def _maybe_reexec_with_amdsmi():
    if importlib.util.find_spec("amdsmi") is not None:
        return  # current interpreter is fine
    # Allow callers to opt out (avoids accidental infinite re-exec loops).
    if os.environ.get("AMDSMI_NO_REEXEC") == "1":
        return

    candidates = [
        "python3",
        "python3.13",
        "python3.12",
        "python3.11",
        "python3.10",
        "python3.9",
        "python3.8",
        "/opt/rocm/bin/python3",
        "/usr/local/bin/python3",
    ]
    seen = {os.path.realpath(sys.executable)}
    for cand in candidates:
        path = cand if os.path.isabs(cand) else shutil.which(cand)
        if not path or not os.path.exists(path):
            continue
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        try:
            r = subprocess.run([path, "-c", "import amdsmi"], capture_output=True, timeout=5)
        except Exception:
            continue
        if r.returncode == 0:
            print(
                f"[INFO] amdsmi not importable in {sys.executable}; re-executing with {path}",
                flush=True,
            )
            env = dict(os.environ)
            env["AMDSMI_NO_REEXEC"] = "1"
            try:
                os.execve(path, [path] + sys.argv, env)
            except OSError:
                # Fall back to subprocess on platforms where execve is restricted.
                sys.exit(subprocess.call([path] + sys.argv, env=env))


_maybe_reexec_with_amdsmi()

try:
    import amdsmi
except ImportError as e:
    print(f"ERROR: Cannot import amdsmi: {e}")
    print("Tried this interpreter and any candidates found on PATH.")
    print("Install the amdsmi Python bindings (e.g. via the ROCm/amd-smi packages),")
    print("or invoke this script with the interpreter that already has them.")
    sys.exit(1)


def is_root():
    """Cross-platform root/admin check."""
    if hasattr(os, "geteuid"):
        try:
            return os.geteuid() == 0
        except Exception:
            return False
    # Windows: best-effort admin check
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def get_hostname():
    try:
        return platform.node() or "unknown"
    except Exception:
        return "unknown"


class _EnumMissing(Exception):
    """Raised inside a test lambda when a required amdsmi enum is not present."""

    pass


def need(attr):
    """Return the amdsmi attribute or raise _EnumMissing. Used inside test lambdas."""
    if not hasattr(amdsmi, attr):
        raise _EnumMissing(f"amdsmi.{attr} not present in this build")
    return getattr(amdsmi, attr)


# ============================================================================
# GLOBALS
# ============================================================================
results = []
gpu_handle = None
gpu_handles = []
socket_handles = []
cpu_socket_handles = []
cpu_core_handles = []

# Saved values for reverting
saved_values = {}

# Runtime configuration (set in main()).
CONFIG = {
    "gpu_index": 0,
    "output": None,
    "skip_writes": False,
    "no_revert": False,
    "all_gpus": False,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def add_result(api_name, status, output="", note=""):
    """Add a test result to the results list."""
    results.append(
        {
            "api": api_name,
            "status": status,
            "output": str(output)[:120] if output else "",
            "note": note,
        }
    )


def test_api(api_name, test_func, skip_reason=None, requires_root=False):
    """
    Test a single API and record the result.

    Parameters:
        api_name: Name of the API to test
        test_func: Lambda/function to call the API (None if skipped)
        skip_reason: If provided, API is not executed. Existence is still reported.
        requires_root: If True and not running as root, the API is skipped instead of
                       executed (to avoid mislabeling permission errors as Unsupported).
    """
    print(f"Testing: {api_name}... ", end="", flush=True)
    # Allow tagged names like "[gpu1] amdsmi_get_gpu_device_bdf" — strip the
    # tag for the existence check while keeping the tag in the report row.
    lookup_name = (
        api_name.split("] ", 1)[1] if api_name.startswith("[") and "] " in api_name else api_name
    )
    exists = hasattr(amdsmi, lookup_name)

    # Explicit skip path: preserve existence information separately from skip.
    if skip_reason:
        if exists:
            print(f"Skipped ({skip_reason})")
            add_result(api_name, "Skipped", note=f"API exists, skipped: {skip_reason}")
        else:
            print("API Not Found")
            add_result(api_name, "API Not Found", note=f"module amdsmi has no attribute {api_name}")
        return

    if not exists:
        print("API Not Found")
        add_result(api_name, "API Not Found", note=f"module amdsmi has no attribute {api_name}")
        return

    if requires_root:
        if CONFIG.get("skip_writes"):
            print("Skipped (--skip-writes)")
            add_result(api_name, "Skipped", note="write API skipped via --skip-writes")
            return
        if not is_root():
            print("Skipped (requires root)")
            add_result(api_name, "Skipped", note="write/privileged API; skipped when not root")
            return

    # Try to call the API
    try:
        result = test_func()
        print("Supported")
        add_result(api_name, "Supported", output=result)
    except _EnumMissing as e:
        print("Skipped (enum missing)")
        add_result(api_name, "Skipped", note=str(e)[:100])
    except (AttributeError, TypeError) as e:
        msg = str(e)
        # Signature mismatches indicate the installed amdsmi build expects
        # a different argument list than this script uses. That's a build
        # difference, not a script bug, so report as Unsupported.
        sig_markers = (
            "missing",
            "required positional",
            "takes",
            "arguments",
            "positional argument",
        )
        if any(m in msg for m in sig_markers):
            print("Unsupported")
            add_result(api_name, "Unsupported", note=f"signature mismatch: {msg[:90]}")
        else:
            # Genuine script bug.
            print("Script Error")
            add_result(api_name, "Script Error", note=msg[:100])
    except getattr(amdsmi, "AmdSmiLibraryException", Exception) as e:
        print("Unsupported")
        add_result(api_name, "Unsupported", note=str(e)[:100])
    except Exception as e:
        print("Unsupported")
        add_result(api_name, "Unsupported", note=str(e)[:100])


def safe_get(func, default=None):
    """Safely get a value, returning default on any error."""
    try:
        return func()
    except Exception:
        return default


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================


def run_tests():
    global \
        gpu_handle, \
        gpu_handles, \
        socket_handles, \
        cpu_socket_handles, \
        cpu_core_handles, \
        saved_values

    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    print("=" * 70)
    print(f"AMD SMI COMPLETE API TEST   (script v{__version__})")
    print("=" * 70)
    print()
    print("Initializing AMD SMI...")

    amdsmi.amdsmi_init()

    # Get all handles
    gpu_handles = amdsmi.amdsmi_get_processor_handles()
    if not gpu_handles:
        print("ERROR: No GPU handles found!")
        amdsmi.amdsmi_shut_down()
        sys.exit(1)

    gpu_index = CONFIG.get("gpu_index", 0)
    if gpu_index < 0 or gpu_index >= len(gpu_handles):
        print(f"ERROR: --gpu-index {gpu_index} out of range (found {len(gpu_handles)} GPUs).")
        amdsmi.amdsmi_shut_down()
        sys.exit(2)
    gpu_handle = gpu_handles[gpu_index]

    # Get socket handles
    try:
        socket_handles = amdsmi.amdsmi_get_socket_handles()
    except:
        socket_handles = []

    # Get CPU socket handles (for CPU APIs)
    try:
        cpu_socket_handles = amdsmi.amdsmi_get_cpusocket_handles()
    except:
        cpu_socket_handles = []

    # Get CPU core handles
    try:
        cpu_core_handles = amdsmi.amdsmi_get_cpucore_handles()
    except:
        cpu_core_handles = []

    # Get GPU info for display
    asic = amdsmi.amdsmi_get_gpu_asic_info(gpu_handle)
    gpu_name = asic.get("market_name", "Unknown")
    bdf = amdsmi.amdsmi_get_gpu_device_bdf(gpu_handle)

    print(f"GPU: {gpu_name}")
    print(f"BDF: {bdf}")
    print(f"GPUs found: {len(gpu_handles)}")
    print(f"Sockets found: {len(socket_handles)}")
    print(f"CPU Sockets found: {len(cpu_socket_handles)}")
    print(f"CPU Cores found: {len(cpu_core_handles)}")
    print(f"Running as: {'ROOT' if is_root() else 'USER'}")
    print("=" * 70)
    print()

    # ========================================================================
    # SAVE CURRENT VALUES FOR REVERTING
    # ========================================================================
    print("Saving current values for revert...")

    saved_values["power_cap"] = safe_get(
        lambda: amdsmi.amdsmi_get_power_cap_info(gpu_handle).get("power_cap")
    )
    saved_values["perf_level"] = safe_get(lambda: amdsmi.amdsmi_get_gpu_perf_level(gpu_handle))
    saved_values["fan_speed"] = safe_get(lambda: amdsmi.amdsmi_get_gpu_fan_speed(gpu_handle, 0))
    saved_values["process_isolation"] = safe_get(
        lambda: amdsmi.amdsmi_get_gpu_process_isolation(gpu_handle)
    )

    print(f"  Power Cap: {saved_values['power_cap']}")
    print(f"  Perf Level: {saved_values['perf_level']}")
    print(f"  Fan Speed: {saved_values['fan_speed']}")
    print(f"  Process Isolation: {saved_values['process_isolation']}")
    print()

    # ========================================================================
    # SECTION 1: INITIALIZATION & SHUTDOWN (2 APIs)
    # ========================================================================
    print("=" * 70)
    print("SECTION 1: INITIALIZATION & SHUTDOWN (2 APIs)")
    print("=" * 70)
    print()

    # amdsmi_init - already called, just verify it exists
    test_api("amdsmi_init", None, skip_reason="Already called during setup")

    # amdsmi_shut_down - will call at end
    test_api("amdsmi_shut_down", None, skip_reason="Will call at end of tests")

    # ========================================================================
    # SECTION 2: PROCESSOR/DEVICE DISCOVERY (14 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 2: PROCESSOR/DEVICE DISCOVERY (14 APIs)")
    print("=" * 70)
    print()

    test_api("amdsmi_get_processor_type", lambda: amdsmi.amdsmi_get_processor_type(gpu_handle))

    test_api("amdsmi_get_processor_handles", lambda: amdsmi.amdsmi_get_processor_handles())

    test_api(
        "amdsmi_get_processor_handles_by_type",
        lambda: (
            amdsmi.amdsmi_get_processor_handles_by_type(
                socket_handles[0],
                getattr(
                    need("AmdSmiProcessorType"),
                    "AMD_GPU",
                    getattr(need("AmdSmiProcessorType"), "AMDSMI_PROCESSOR_TYPE_AMD_GPU", 0),
                ),
            )
            if socket_handles
            else (_ for _ in ()).throw(_EnumMissing("no socket handles available"))
        ),
    )

    test_api(
        "amdsmi_get_processor_count_from_handles",
        lambda: amdsmi.amdsmi_get_processor_count_from_handles(gpu_handles),
    )

    test_api(
        "amdsmi_get_processor_handle_from_bdf",
        lambda: amdsmi.amdsmi_get_processor_handle_from_bdf(bdf),
    )

    test_api("amdsmi_get_processor_info", lambda: amdsmi.amdsmi_get_processor_info(gpu_handle))

    test_api("amdsmi_get_socket_handles", lambda: amdsmi.amdsmi_get_socket_handles())

    test_api(
        "amdsmi_get_socket_info",
        lambda: amdsmi.amdsmi_get_socket_info(socket_handles[0]) if socket_handles else None,
    )

    test_api("amdsmi_get_node_handle", lambda: amdsmi.amdsmi_get_node_handle(gpu_handle))

    test_api("amdsmi_get_gpu_device_bdf", lambda: amdsmi.amdsmi_get_gpu_device_bdf(gpu_handle))

    test_api("amdsmi_get_gpu_device_uuid", lambda: amdsmi.amdsmi_get_gpu_device_uuid(gpu_handle))

    test_api("amdsmi_get_gpu_bdf_id", lambda: amdsmi.amdsmi_get_gpu_bdf_id(gpu_handle))

    test_api(
        "amdsmi_get_gpu_enumeration_info",
        lambda: amdsmi.amdsmi_get_gpu_enumeration_info(gpu_handle),
    )

    test_api("amdsmi_get_gpu_kfd_info", lambda: amdsmi.amdsmi_get_gpu_kfd_info(gpu_handle))

    # ========================================================================
    # SECTION 3: DRIVER & VERSION INFO (8 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 3: DRIVER & VERSION INFO (8 APIs)")
    print("=" * 70)
    print()

    test_api(
        "amdsmi_get_gpu_driver_version", lambda: amdsmi.amdsmi_get_gpu_driver_version(gpu_handle)
    )

    test_api("amdsmi_get_gpu_driver_info", lambda: amdsmi.amdsmi_get_gpu_driver_info(gpu_handle))

    test_api("amdsmi_get_version", lambda: amdsmi.amdsmi_get_version())

    test_api("amdsmi_get_version_str", lambda: amdsmi.amdsmi_get_version_str())

    test_api("amdsmi_get_lib_version", lambda: amdsmi.amdsmi_get_lib_version())

    test_api("amdsmi_get_rocm_version", lambda: amdsmi.amdsmi_get_rocm_version())

    test_api("amdsmi_status_string", lambda: amdsmi.amdsmi_status_string(0))

    test_api("amdsmi_status_code_to_string", lambda: amdsmi.amdsmi_status_code_to_string(0))

    # ========================================================================
    # SECTION 4: GPU HARDWARE INFO (12 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 4: GPU HARDWARE INFO (12 APIs)")
    print("=" * 70)
    print()

    test_api("amdsmi_get_gpu_asic_info", lambda: amdsmi.amdsmi_get_gpu_asic_info(gpu_handle))

    test_api("amdsmi_get_gpu_vram_info", lambda: amdsmi.amdsmi_get_gpu_vram_info(gpu_handle))

    test_api("amdsmi_get_gpu_board_info", lambda: amdsmi.amdsmi_get_gpu_board_info(gpu_handle))

    test_api("amdsmi_get_gpu_vbios_info", lambda: amdsmi.amdsmi_get_gpu_vbios_info(gpu_handle))

    test_api("amdsmi_get_gpu_cache_info", lambda: amdsmi.amdsmi_get_gpu_cache_info(gpu_handle))

    test_api("amdsmi_get_gpu_revision", lambda: amdsmi.amdsmi_get_gpu_revision(gpu_handle))

    test_api("amdsmi_get_gpu_vendor_name", lambda: amdsmi.amdsmi_get_gpu_vendor_name(gpu_handle))

    test_api("amdsmi_get_gpu_id", lambda: amdsmi.amdsmi_get_gpu_id(gpu_handle))

    test_api("amdsmi_get_gpu_vram_vendor", lambda: amdsmi.amdsmi_get_gpu_vram_vendor(gpu_handle))

    test_api(
        "amdsmi_get_gpu_drm_render_minor",
        lambda: amdsmi.amdsmi_get_gpu_drm_render_minor(gpu_handle),
    )

    test_api("amdsmi_get_gpu_subsystem_id", lambda: amdsmi.amdsmi_get_gpu_subsystem_id(gpu_handle))

    test_api(
        "amdsmi_get_gpu_subsystem_name", lambda: amdsmi.amdsmi_get_gpu_subsystem_name(gpu_handle)
    )

    # ========================================================================
    # SECTION 5: FIRMWARE INFO (2 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 5: FIRMWARE INFO (2 APIs)")
    print("=" * 70)
    print()

    test_api("amdsmi_get_fw_info", lambda: amdsmi.amdsmi_get_fw_info(gpu_handle))

    test_api(
        "amdsmi_get_npm_info",
        lambda: (
            amdsmi.amdsmi_get_npm_info(amdsmi.amdsmi_get_node_handle(gpu_handle))
            if hasattr(amdsmi, "amdsmi_get_node_handle")
            else None
        ),
    )

    # ========================================================================
    # SECTION 6: POWER MANAGEMENT (10 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 6: POWER MANAGEMENT (10 APIs)")
    print("=" * 70)
    print()

    test_api("amdsmi_get_power_cap_info", lambda: amdsmi.amdsmi_get_power_cap_info(gpu_handle))

    test_api(
        "amdsmi_get_supported_power_cap", lambda: amdsmi.amdsmi_get_supported_power_cap(gpu_handle)
    )

    test_api("amdsmi_get_power_info", lambda: amdsmi.amdsmi_get_power_info(gpu_handle))

    # Set power cap - use current value (safe)
    if saved_values["power_cap"]:
        test_api(
            "amdsmi_set_power_cap",
            lambda: amdsmi.amdsmi_set_power_cap(gpu_handle, 0, saved_values["power_cap"]),
            requires_root=True,
        )
    else:
        test_api("amdsmi_set_power_cap", None, skip_reason="Could not get current power cap")

    test_api(
        "amdsmi_get_gpu_power_profile_presets",
        lambda: amdsmi.amdsmi_get_gpu_power_profile_presets(gpu_handle, 0),
    )

    test_api(
        "amdsmi_set_gpu_power_profile",
        lambda: amdsmi.amdsmi_set_gpu_power_profile(
            gpu_handle, 0, need("AmdSmiPowerProfilePresetMasks").BOOTUP_DEFAULT
        ),
        requires_root=True,
    )

    test_api(
        "amdsmi_is_gpu_power_management_enabled",
        lambda: amdsmi.amdsmi_is_gpu_power_management_enabled(gpu_handle),
    )

    test_api("amdsmi_get_energy_count", lambda: amdsmi.amdsmi_get_energy_count(gpu_handle))

    test_api("amdsmi_get_violation_status", lambda: amdsmi.amdsmi_get_violation_status(gpu_handle))

    test_api(
        "amdsmi_gpu_driver_reload",
        None,
        skip_reason="Risky - reloads GPU driver, could crash system",
    )

    # ========================================================================
    # SECTION 7: PCIe INFO (12 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 7: PCIe INFO (12 APIs)")
    print("=" * 70)
    print()

    test_api("amdsmi_get_pcie_info", lambda: amdsmi.amdsmi_get_pcie_info(gpu_handle))

    test_api("amdsmi_get_pcie_link_status", lambda: amdsmi.amdsmi_get_pcie_link_status(gpu_handle))

    test_api("amdsmi_get_pcie_link_caps", lambda: amdsmi.amdsmi_get_pcie_link_caps(gpu_handle))

    test_api("amdsmi_get_gpu_pci_id", lambda: amdsmi.amdsmi_get_gpu_pci_id(gpu_handle))

    test_api(
        "amdsmi_get_gpu_pci_bandwidth", lambda: amdsmi.amdsmi_get_gpu_pci_bandwidth(gpu_handle)
    )

    test_api(
        "amdsmi_set_gpu_pci_bandwidth",
        None,
        skip_reason="Risky - writing arbitrary PCIe bandwidth mask is hardware-dependent",
    )

    test_api(
        "amdsmi_get_gpu_pci_throughput", lambda: amdsmi.amdsmi_get_gpu_pci_throughput(gpu_handle)
    )

    test_api(
        "amdsmi_get_gpu_pci_replay_counter",
        lambda: amdsmi.amdsmi_get_gpu_pci_replay_counter(gpu_handle),
    )

    test_api("amdsmi_get_gpu_ptl_formats", lambda: amdsmi.amdsmi_get_gpu_ptl_formats(gpu_handle))

    test_api("amdsmi_get_gpu_ptl_state", lambda: amdsmi.amdsmi_get_gpu_ptl_state(gpu_handle))

    test_api("amdsmi_set_gpu_ptl_formats", None, skip_reason="Risky - modifies PCIe PTL formats")

    test_api("amdsmi_set_gpu_ptl_state", None, skip_reason="Risky - modifies PCIe PTL state")

    # ========================================================================
    # SECTION 8: ACTIVITY & UTILIZATION (6 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 8: ACTIVITY & UTILIZATION (6 APIs)")
    print("=" * 70)
    print()

    test_api("amdsmi_get_gpu_activity", lambda: amdsmi.amdsmi_get_gpu_activity(gpu_handle))

    test_api("amdsmi_get_busy_percent", lambda: amdsmi.amdsmi_get_busy_percent(gpu_handle))

    test_api("amdsmi_get_gpu_busy_percent", lambda: amdsmi.amdsmi_get_gpu_busy_percent(gpu_handle))

    # Build a portable counter_types list from whatever this build exposes.
    def _utilization_counter_types():
        ut = need("AmdSmiUtilizationCounterType")
        candidates = [
            "COARSE_GRAIN_GFX_ACTIVITY",
            "COARSE_GRAIN_MEM_ACTIVITY",
            "FINE_GRAIN_GFX_ACTIVITY",
            "FINE_GRAIN_MEM_ACTIVITY",
        ]
        picked = [getattr(ut, name) for name in candidates if hasattr(ut, name)]
        if not picked:
            raise _EnumMissing("no AmdSmiUtilizationCounterType members available")
        return picked

    test_api(
        "amdsmi_get_utilization_count",
        lambda: amdsmi.amdsmi_get_utilization_count(gpu_handle, _utilization_counter_types()),
    )

    test_api("amdsmi_get_gpu_xcd_counter", lambda: amdsmi.amdsmi_get_gpu_xcd_counter(gpu_handle))

    test_api("amdsmi_clean_gpu_local_data", None, skip_reason="Risky - could delete GPU local data")

    # ========================================================================
    # SECTION 9: CLOCK & FREQUENCY (15 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 9: CLOCK & FREQUENCY (15 APIs)")
    print("=" * 70)
    print()

    test_api(
        "amdsmi_get_clock_info",
        lambda: amdsmi.amdsmi_get_clock_info(gpu_handle, need("AmdSmiClkType").GFX),
    )

    test_api(
        "amdsmi_get_clk_freq",
        lambda: amdsmi.amdsmi_get_clk_freq(gpu_handle, need("AmdSmiClkType").GFX),
    )

    test_api(
        "amdsmi_set_clk_freq",
        None,
        skip_reason="Risky - writing arbitrary clk freq mask is hardware-dependent",
    )

    test_api(
        "amdsmi_get_gpu_target_frequency_range",
        lambda: amdsmi.amdsmi_get_gpu_target_frequency_range(gpu_handle),
    )

    test_api(
        "amdsmi_set_gpu_clk_range",
        lambda: amdsmi.amdsmi_set_gpu_clk_range(gpu_handle, 500, 2500, need("AmdSmiClkType").GFX),
        requires_root=True,
    )

    # set_gpu_clk_limit takes (handle, clk_type, limit_type, value).
    # Read the current target range and replay max as a no-op-ish write.
    def _set_clk_limit_noop():
        clk_type = need("AmdSmiClkType").GFX
        limit_type_enum = need("AmdSmiClkLimitType")
        # Prefer SOFT_MAX, fall back to MAX or first member.
        limit_type = getattr(
            limit_type_enum, "SOFT_MAX", getattr(limit_type_enum, "MAX", list(limit_type_enum)[0])
        )
        cur = (
            safe_get(lambda: amdsmi.amdsmi_get_gpu_target_frequency_range(gpu_handle), default={})
            or {}
        )
        # target_frequency_range usually returns dict with min/max per clk
        max_clk = None
        if isinstance(cur, dict):
            for k in ("max_clk", "max", "sclk_max", "gfx_max"):
                if k in cur:
                    max_clk = cur[k]
                    break
        if not max_clk:
            max_clk = 1500  # sane fallback in MHz
        return amdsmi.amdsmi_set_gpu_clk_limit(gpu_handle, clk_type, limit_type, int(max_clk))

    test_api("amdsmi_set_gpu_clk_limit", _set_clk_limit_noop, requires_root=True)

    test_api("amdsmi_get_gpu_od_clk_info", lambda: amdsmi.amdsmi_get_gpu_od_clk_info(gpu_handle))

    test_api(
        "amdsmi_set_gpu_od_clk_info",
        lambda: amdsmi.amdsmi_set_gpu_od_clk_info(
            gpu_handle, need("AmdSmiFreqInd").MIN, 500, need("AmdSmiClkType").GFX
        ),
        requires_root=True,
    )

    test_api("amdsmi_get_gpu_od_volt_info", lambda: amdsmi.amdsmi_get_gpu_od_volt_info(gpu_handle))

    test_api(
        "amdsmi_set_gpu_od_volt_info",
        lambda: amdsmi.amdsmi_set_gpu_od_volt_info(gpu_handle, 0, 500, 700),
        requires_root=True,
    )

    # od_volt_curve_regions needs num_regions; pull it from od_volt_info.
    def _od_volt_curve_regions():
        info = safe_get(lambda: amdsmi.amdsmi_get_gpu_od_volt_info(gpu_handle), default={}) or {}
        n = 0
        if isinstance(info, dict):
            n = int(info.get("num_regions") or info.get("num_curve_regions") or 0)
        if n <= 0:
            n = 1  # try a single region; API will error cleanly if unsupported
        return amdsmi.amdsmi_get_gpu_od_volt_curve_regions(gpu_handle, n)

    test_api("amdsmi_get_gpu_od_volt_curve_regions", _od_volt_curve_regions)

    test_api("amdsmi_get_dfc_ctrl", lambda: amdsmi.amdsmi_get_dfc_ctrl(gpu_handle))

    test_api("amdsmi_set_dfc_ctrl", None, skip_reason="Risky - modifies Data Fabric Clock control")

    test_api("amdsmi_get_soc_pstate", lambda: amdsmi.amdsmi_get_soc_pstate(gpu_handle))

    test_api(
        "amdsmi_set_soc_pstate",
        lambda: amdsmi.amdsmi_set_soc_pstate(gpu_handle, 0),
        requires_root=True,
    )

    # ========================================================================
    # SECTION 10: PERFORMANCE LEVEL & OVERDRIVE (8 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 10: PERFORMANCE LEVEL & OVERDRIVE (8 APIs)")
    print("=" * 70)
    print()

    test_api("amdsmi_get_gpu_perf_level", lambda: amdsmi.amdsmi_get_gpu_perf_level(gpu_handle))

    test_api(
        "amdsmi_set_gpu_perf_level",
        lambda: amdsmi.amdsmi_set_gpu_perf_level(gpu_handle, need("AmdSmiDevPerfLevel").AUTO),
        requires_root=True,
    )

    test_api(
        "amdsmi_set_gpu_perf_level_v1",
        lambda: amdsmi.amdsmi_set_gpu_perf_level_v1(gpu_handle, need("AmdSmiDevPerfLevel").AUTO),
        requires_root=True,
    )

    test_api(
        "amdsmi_set_gpu_perf_determinism_mode",
        lambda: amdsmi.amdsmi_set_gpu_perf_determinism_mode(gpu_handle, 1000),
        requires_root=True,
    )

    test_api(
        "amdsmi_get_gpu_overdrive_level", lambda: amdsmi.amdsmi_get_gpu_overdrive_level(gpu_handle)
    )

    test_api(
        "amdsmi_set_gpu_overdrive_level",
        lambda: amdsmi.amdsmi_set_gpu_overdrive_level(gpu_handle, 0),
        requires_root=True,
    )

    test_api(
        "amdsmi_set_gpu_overdrive_level_v1",
        lambda: amdsmi.amdsmi_set_gpu_overdrive_level_v1(gpu_handle, 0),
        requires_root=True,
    )

    test_api(
        "amdsmi_get_gpu_mem_overdrive_level",
        lambda: amdsmi.amdsmi_get_gpu_mem_overdrive_level(gpu_handle),
    )

    # ========================================================================
    # SECTION 11: TEMPERATURE & VOLTAGE & METRICS (7 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 11: TEMPERATURE & VOLTAGE & METRICS (7 APIs)")
    print("=" * 70)
    print()

    test_api(
        "amdsmi_get_temp_metric",
        lambda: amdsmi.amdsmi_get_temp_metric(
            gpu_handle, need("AmdSmiTemperatureType").EDGE, need("AmdSmiTemperatureMetric").CURRENT
        ),
    )

    test_api(
        "amdsmi_get_gpu_volt_metric",
        lambda: amdsmi.amdsmi_get_gpu_volt_metric(
            gpu_handle, need("AmdSmiVoltageType").VDDGFX, need("AmdSmiVoltageMetric").CURRENT
        ),
    )

    test_api("amdsmi_get_gpu_metrics_info", lambda: amdsmi.amdsmi_get_gpu_metrics_info(gpu_handle))

    test_api(
        "amdsmi_get_gpu_metrics_header_info",
        lambda: amdsmi.amdsmi_get_gpu_metrics_header_info(gpu_handle),
    )

    test_api(
        "amdsmi_get_gpu_pm_metrics_info", lambda: amdsmi.amdsmi_get_gpu_pm_metrics_info(gpu_handle)
    )

    test_api(
        "amdsmi_get_gpu_partition_metrics_info",
        lambda: amdsmi.amdsmi_get_gpu_partition_metrics_info(gpu_handle),
    )

    test_api(
        "amdsmi_get_gpu_reg_table_info", lambda: amdsmi.amdsmi_get_gpu_reg_table_info(gpu_handle, 0)
    )

    # ========================================================================
    # SECTION 12: FAN CONTROL (5 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 12: FAN CONTROL (5 APIs)")
    print("=" * 70)
    print()

    test_api("amdsmi_get_gpu_fan_rpms", lambda: amdsmi.amdsmi_get_gpu_fan_rpms(gpu_handle, 0))

    test_api("amdsmi_get_gpu_fan_speed", lambda: amdsmi.amdsmi_get_gpu_fan_speed(gpu_handle, 0))

    test_api(
        "amdsmi_get_gpu_fan_speed_max", lambda: amdsmi.amdsmi_get_gpu_fan_speed_max(gpu_handle, 0)
    )

    test_api(
        "amdsmi_set_gpu_fan_speed",
        lambda: amdsmi.amdsmi_set_gpu_fan_speed(gpu_handle, 0, 128),
        requires_root=True,
    )

    test_api(
        "amdsmi_reset_gpu_fan",
        lambda: amdsmi.amdsmi_reset_gpu_fan(gpu_handle, 0),
        requires_root=True,
    )

    # ========================================================================
    # SECTION 13: MEMORY (8 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 13: MEMORY (8 APIs)")
    print("=" * 70)
    print()

    test_api("amdsmi_get_gpu_vram_usage", lambda: amdsmi.amdsmi_get_gpu_vram_usage(gpu_handle))

    test_api(
        "amdsmi_get_gpu_memory_total",
        lambda: amdsmi.amdsmi_get_gpu_memory_total(gpu_handle, need("AmdSmiMemoryType").VRAM),
    )

    test_api(
        "amdsmi_get_gpu_memory_usage",
        lambda: amdsmi.amdsmi_get_gpu_memory_usage(gpu_handle, need("AmdSmiMemoryType").VRAM),
    )

    test_api(
        "amdsmi_get_gpu_memory_busy_percent",
        lambda: amdsmi.amdsmi_get_gpu_memory_busy_percent(gpu_handle),
    )

    test_api(
        "amdsmi_get_gpu_memory_reserved_pages",
        lambda: amdsmi.amdsmi_get_gpu_memory_reserved_pages(gpu_handle),
    )

    test_api(
        "amdsmi_get_gpu_bad_page_info", lambda: amdsmi.amdsmi_get_gpu_bad_page_info(gpu_handle)
    )

    test_api(
        "amdsmi_get_gpu_bad_page_threshold",
        lambda: amdsmi.amdsmi_get_gpu_bad_page_threshold(gpu_handle),
    )

    # ========================================================================
    # SECTION 14: RAS & ECC (10 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 14: RAS & ECC (10 APIs)")
    print("=" * 70)
    print()

    test_api(
        "amdsmi_get_gpu_ras_block_features_enabled",
        lambda: amdsmi.amdsmi_get_gpu_ras_block_features_enabled(gpu_handle),
    )

    test_api(
        "amdsmi_get_gpu_ras_feature_info",
        lambda: amdsmi.amdsmi_get_gpu_ras_feature_info(gpu_handle),
    )

    test_api(
        "amdsmi_get_gpu_ecc_error_count", lambda: amdsmi.amdsmi_get_gpu_ecc_error_count(gpu_handle)
    )

    test_api(
        "amdsmi_get_gpu_total_ecc_count", lambda: amdsmi.amdsmi_get_gpu_total_ecc_count(gpu_handle)
    )

    test_api(
        "amdsmi_get_gpu_ecc_count",
        lambda: amdsmi.amdsmi_get_gpu_ecc_count(gpu_handle, need("AmdSmiGpuBlock").UMC),
    )

    test_api("amdsmi_get_gpu_ecc_enabled", lambda: amdsmi.amdsmi_get_gpu_ecc_enabled(gpu_handle))

    test_api(
        "amdsmi_get_gpu_ecc_status",
        lambda: amdsmi.amdsmi_get_gpu_ecc_status(gpu_handle, need("AmdSmiGpuBlock").UMC),
    )

    test_api(
        "amdsmi_get_gpu_cper_entries", lambda: amdsmi.amdsmi_get_gpu_cper_entries(gpu_handle, 0)
    )

    test_api("amdsmi_get_afids_from_cper", lambda: amdsmi.amdsmi_get_afids_from_cper(gpu_handle))

    test_api(
        "amdsmi_gpu_validate_ras_eeprom", lambda: amdsmi.amdsmi_gpu_validate_ras_eeprom(gpu_handle)
    )

    # ========================================================================
    # SECTION 15: PROCESS MANAGEMENT (7 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 15: PROCESS MANAGEMENT (7 APIs)")
    print("=" * 70)
    print()

    test_api("amdsmi_get_gpu_process_list", lambda: amdsmi.amdsmi_get_gpu_process_list(gpu_handle))

    # Pick a real PID from the process list if possible; otherwise skip.
    _proc_list = safe_get(lambda: amdsmi.amdsmi_get_gpu_process_list(gpu_handle), default=[])
    _first_pid = None
    if _proc_list:
        _p0 = _proc_list[0]
        _first_pid = _p0.get("pid") if isinstance(_p0, dict) else _p0
    if _first_pid:
        test_api(
            "amdsmi_get_gpu_process_info",
            lambda: amdsmi.amdsmi_get_gpu_process_info(gpu_handle, _first_pid),
        )
    else:
        test_api(
            "amdsmi_get_gpu_process_info", None, skip_reason="No active GPU processes to query"
        )

    test_api(
        "amdsmi_get_gpu_compute_process_info", lambda: amdsmi.amdsmi_get_gpu_compute_process_info()
    )

    test_api(
        "amdsmi_get_gpu_compute_process_info_by_pid",
        lambda: amdsmi.amdsmi_get_gpu_compute_process_info_by_pid(os.getpid()),
    )

    test_api(
        "amdsmi_get_gpu_compute_process_gpus",
        lambda: amdsmi.amdsmi_get_gpu_compute_process_gpus(os.getpid()),
    )

    test_api(
        "amdsmi_get_gpu_process_isolation",
        lambda: amdsmi.amdsmi_get_gpu_process_isolation(gpu_handle),
    )

    # Set process isolation - use saved value
    if saved_values["process_isolation"] is not None:
        test_api(
            "amdsmi_set_gpu_process_isolation",
            lambda: amdsmi.amdsmi_set_gpu_process_isolation(
                gpu_handle, saved_values["process_isolation"]
            ),
            requires_root=True,
        )
    else:
        test_api(
            "amdsmi_set_gpu_process_isolation",
            lambda: amdsmi.amdsmi_set_gpu_process_isolation(gpu_handle, 0),
            requires_root=True,
        )

    # ========================================================================
    # SECTION 16: TOPOLOGY & NUMA (11 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 16: TOPOLOGY & NUMA (11 APIs)")
    print("=" * 70)
    print()

    test_api(
        "amdsmi_get_gpu_topo_numa_affinity",
        lambda: amdsmi.amdsmi_get_gpu_topo_numa_affinity(gpu_handle),
    )

    test_api(
        "amdsmi_topo_get_numa_node_number",
        lambda: amdsmi.amdsmi_topo_get_numa_node_number(gpu_handle),
    )

    test_api(
        "amdsmi_topo_get_link_weight",
        lambda: (
            amdsmi.amdsmi_topo_get_link_weight(gpu_handles[0], gpu_handles[1])
            if len(gpu_handles) >= 2
            else amdsmi.amdsmi_topo_get_link_weight(gpu_handle, gpu_handle)
        ),
    )

    test_api(
        "amdsmi_topo_get_link_type",
        lambda: (
            amdsmi.amdsmi_topo_get_link_type(gpu_handles[0], gpu_handles[1])
            if len(gpu_handles) >= 2
            else amdsmi.amdsmi_topo_get_link_type(gpu_handle, gpu_handle)
        ),
    )

    test_api(
        "amdsmi_topo_get_p2p_status",
        lambda: (
            amdsmi.amdsmi_topo_get_p2p_status(gpu_handles[0], gpu_handles[1])
            if len(gpu_handles) >= 2
            else amdsmi.amdsmi_topo_get_p2p_status(gpu_handle, gpu_handle)
        ),
    )

    test_api(
        "amdsmi_get_P2P_status",
        lambda: (
            amdsmi.amdsmi_get_P2P_status(gpu_handles[0], gpu_handles[1])
            if len(gpu_handles) >= 2
            else amdsmi.amdsmi_get_P2P_status(gpu_handle, gpu_handle)
        ),
    )

    test_api(
        "amdsmi_is_P2P_accessible",
        lambda: (
            amdsmi.amdsmi_is_P2P_accessible(gpu_handles[0], gpu_handles[1])
            if len(gpu_handles) >= 2
            else amdsmi.amdsmi_is_P2P_accessible(gpu_handle, gpu_handle)
        ),
    )

    test_api(
        "amdsmi_get_minmax_bandwidth",
        lambda: (
            amdsmi.amdsmi_get_minmax_bandwidth(gpu_handles[0], gpu_handles[1])
            if len(gpu_handles) >= 2
            else amdsmi.amdsmi_get_minmax_bandwidth(gpu_handle, gpu_handle)
        ),
    )

    test_api(
        "amdsmi_get_minmax_bandwidth_between_processors",
        lambda: (
            amdsmi.amdsmi_get_minmax_bandwidth_between_processors(gpu_handles[0], gpu_handles[1])
            if len(gpu_handles) >= 2
            else amdsmi.amdsmi_get_minmax_bandwidth_between_processors(gpu_handle, gpu_handle)
        ),
    )

    # Some amdsmi builds raise AttributeError internally (e.g. missing
    # 'domain_number' on union_amdsmi_bdf_t). Re-raise as a runtime error
    # so it reports as Unsupported rather than a Script Error.
    def _get_link_metrics():
        try:
            return amdsmi.amdsmi_get_link_metrics(gpu_handle)
        except AttributeError as e:
            raise RuntimeError(f"amdsmi internal: {e}")

    test_api("amdsmi_get_link_metrics", _get_link_metrics)

    # link_topology_nearest needs a link_type; try common members across builds.
    def _link_topology_nearest():
        lt = need("AmdSmiLinkType")
        for name in ("XGMI", "PCIE", "AMDSMI_LINK_TYPE_XGMI", "AMDSMI_LINK_TYPE_PCIE"):
            if hasattr(lt, name):
                return amdsmi.amdsmi_get_link_topology_nearest(gpu_handle, getattr(lt, name))
        # Fall back to first enum value
        first = list(lt)[0]
        return amdsmi.amdsmi_get_link_topology_nearest(gpu_handle, first)

    test_api("amdsmi_get_link_topology_nearest", _link_topology_nearest)

    # ========================================================================
    # SECTION 17: XGMI (6 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 17: XGMI (6 APIs)")
    print("=" * 70)
    print()

    test_api("amdsmi_get_xgmi_info", lambda: amdsmi.amdsmi_get_xgmi_info(gpu_handle))

    test_api("amdsmi_get_xgmi_plpd", lambda: amdsmi.amdsmi_get_xgmi_plpd(gpu_handle))

    test_api(
        "amdsmi_set_xgmi_plpd",
        lambda: amdsmi.amdsmi_set_xgmi_plpd(gpu_handle, 0),
        requires_root=True,
    )

    test_api(
        "amdsmi_get_gpu_xgmi_link_status",
        lambda: amdsmi.amdsmi_get_gpu_xgmi_link_status(gpu_handle),
    )

    test_api(
        "amdsmi_gpu_xgmi_error_status", lambda: amdsmi.amdsmi_gpu_xgmi_error_status(gpu_handle)
    )

    test_api(
        "amdsmi_reset_gpu_xgmi_error",
        lambda: amdsmi.amdsmi_reset_gpu_xgmi_error(gpu_handle),
        requires_root=True,
    )

    # ========================================================================
    # SECTION 18: PARTITIONING (15 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 18: PARTITIONING (15 APIs)")
    print("=" * 70)
    print()

    test_api(
        "amdsmi_get_gpu_compute_partition",
        lambda: amdsmi.amdsmi_get_gpu_compute_partition(gpu_handle),
    )

    test_api(
        "amdsmi_set_gpu_compute_partition",
        None,
        skip_reason="Risky - changes compute partition, may need reboot",
    )

    test_api(
        "amdsmi_dev_compute_partition_get",
        lambda: amdsmi.amdsmi_dev_compute_partition_get(gpu_handle),
    )

    test_api(
        "amdsmi_dev_compute_partition_set",
        None,
        skip_reason="Risky - changes compute partition, may need reboot",
    )

    test_api(
        "amdsmi_dev_compute_partition_reset", None, skip_reason="Risky - resets compute partition"
    )

    test_api(
        "amdsmi_get_gpu_memory_partition",
        lambda: amdsmi.amdsmi_get_gpu_memory_partition(gpu_handle),
    )

    test_api(
        "amdsmi_set_gpu_memory_partition",
        None,
        skip_reason="Risky - changes memory partition, may need reboot",
    )

    test_api(
        "amdsmi_set_gpu_memory_partition_mode",
        None,
        skip_reason="Risky - changes memory partition mode",
    )

    test_api(
        "amdsmi_get_gpu_memory_partition_config",
        lambda: amdsmi.amdsmi_get_gpu_memory_partition_config(gpu_handle),
    )

    test_api(
        "amdsmi_get_gpu_accelerator_partition_profile",
        lambda: amdsmi.amdsmi_get_gpu_accelerator_partition_profile(gpu_handle),
    )

    test_api(
        "amdsmi_set_gpu_accelerator_partition_profile",
        None,
        skip_reason="Risky - changes accelerator partition profile",
    )

    test_api(
        "amdsmi_get_gpu_accelerator_partition_profile_config",
        lambda: amdsmi.amdsmi_get_gpu_accelerator_partition_profile_config(gpu_handle),
    )

    test_api("amdsmi_dev_nps_mode_get", lambda: amdsmi.amdsmi_dev_nps_mode_get(gpu_handle))

    test_api("amdsmi_dev_nps_mode_set", None, skip_reason="Risky - changes NPS mode")

    test_api("amdsmi_dev_nps_mode_reset", None, skip_reason="Risky - resets NPS mode")

    # ========================================================================
    # SECTION 19: VIRTUALIZATION (2 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 19: VIRTUALIZATION (2 APIs)")
    print("=" * 70)
    print()

    test_api(
        "amdsmi_get_gpu_virtualization_mode",
        lambda: amdsmi.amdsmi_get_gpu_virtualization_mode(gpu_handle),
    )

    # cpu_affinity_with_scope needs (gpu_handle, scope).
    def _cpu_affinity_with_scope():
        scope_enum = need("AmdSmiCpuAffinityScope")
        scope = getattr(scope_enum, "NODE", getattr(scope_enum, "SOCKET", list(scope_enum)[0]))
        return amdsmi.amdsmi_get_cpu_affinity_with_scope(gpu_handle, scope)

    test_api("amdsmi_get_cpu_affinity_with_scope", _cpu_affinity_with_scope)

    # ========================================================================
    # SECTION 20: EVENT NOTIFICATION (5 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 20: EVENT NOTIFICATION (5 APIs)")
    print("=" * 70)
    print()

    # AmdSmiEventReader is a class; just verify it is importable.
    if hasattr(amdsmi, "AmdSmiEventReader"):
        add_result("AmdSmiEventReader", "Supported", output="class present")
        print("Testing: AmdSmiEventReader... Supported")
    else:
        add_result(
            "AmdSmiEventReader",
            "API Not Found",
            note="module amdsmi has no attribute AmdSmiEventReader",
        )
        print("Testing: AmdSmiEventReader... API Not Found")

    test_api(
        "amdsmi_init_gpu_event_notification",
        lambda: amdsmi.amdsmi_init_gpu_event_notification(gpu_handle),
    )

    test_api(
        "amdsmi_set_gpu_event_notification_mask",
        None,
        skip_reason="Risky - modifies event notification state",
    )

    test_api(
        "amdsmi_get_gpu_event_notification",
        None,
        skip_reason="Requires an active event session with a configured mask",
    )

    test_api(
        "amdsmi_stop_gpu_event_notification",
        lambda: amdsmi.amdsmi_stop_gpu_event_notification(gpu_handle),
    )

    # ========================================================================
    # SECTION 21: PERFORMANCE COUNTERS (6 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 21: PERFORMANCE COUNTERS (6 APIs)")
    print("=" * 70)
    print()

    test_api(
        "amdsmi_gpu_counter_group_supported",
        lambda: amdsmi.amdsmi_gpu_counter_group_supported(
            gpu_handle, need("AmdSmiEventGroup").XGMI
        ),
    )

    test_api(
        "amdsmi_gpu_create_counter",
        lambda: amdsmi.amdsmi_gpu_create_counter(gpu_handle, need("AmdSmiEventType").XGMI_0_NOP_TX),
    )

    test_api(
        "amdsmi_gpu_destroy_counter", None, skip_reason="Requires valid counter handle from create"
    )

    test_api(
        "amdsmi_gpu_control_counter", None, skip_reason="Requires valid counter handle from create"
    )

    test_api(
        "amdsmi_gpu_read_counter", None, skip_reason="Requires valid counter handle from create"
    )

    test_api(
        "amdsmi_get_gpu_available_counters",
        lambda: amdsmi.amdsmi_get_gpu_available_counters(gpu_handle, need("AmdSmiEventGroup").XGMI),
    )

    # ========================================================================
    # SECTION 22: ITERATOR APIs (5 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 22: ITERATOR APIs (5 APIs)")
    print("=" * 70)
    print()

    test_api(
        "amdsmi_open_supported_func_iterator",
        lambda: amdsmi.amdsmi_open_supported_func_iterator(gpu_handle),
    )

    test_api(
        "amdsmi_open_supported_variant_iterator",
        lambda: amdsmi.amdsmi_open_supported_variant_iterator(gpu_handle),
    )

    test_api(
        "amdsmi_close_supported_func_iterator", None, skip_reason="Requires valid iterator handle"
    )

    test_api("amdsmi_next_func_iter", None, skip_reason="Requires valid iterator handle")

    test_api("amdsmi_get_func_iter_value", None, skip_reason="Requires valid iterator handle")

    # ========================================================================
    # SECTION 23: RESET APIs (1 API)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 23: RESET APIs (1 API)")
    print("=" * 70)
    print()

    test_api("amdsmi_reset_gpu", None, skip_reason="Risky - would reset GPU, could crash system")

    # ========================================================================
    # SECTION 24: CPU/HSMP APIs (74 APIs)
    # ========================================================================
    print()
    print("=" * 70)
    print("SECTION 24: CPU/HSMP APIs (74 APIs)")
    print("=" * 70)
    print()

    # Get CPU socket handle for CPU APIs
    cpu_socket = cpu_socket_handles[0] if cpu_socket_handles else None
    cpu_core = cpu_core_handles[0] if cpu_core_handles else None

    test_api("amdsmi_get_cpusocket_handles", lambda: amdsmi.amdsmi_get_cpusocket_handles())

    test_api("amdsmi_get_cpucore_handles", lambda: amdsmi.amdsmi_get_cpucore_handles())

    test_api(
        "amdsmi_get_cpu_hsmp_proto_ver",
        lambda: amdsmi.amdsmi_get_cpu_hsmp_proto_ver(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_smu_fw_version",
        lambda: amdsmi.amdsmi_get_cpu_smu_fw_version(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_hsmp_driver_version",
        lambda: amdsmi.amdsmi_get_cpu_hsmp_driver_version(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_core_energy",
        lambda: amdsmi.amdsmi_get_cpu_core_energy(cpu_core) if cpu_core else None,
    )

    test_api(
        "amdsmi_get_cpu_core_ccd_power",
        lambda: amdsmi.amdsmi_get_cpu_core_ccd_power(cpu_core) if cpu_core else None,
    )

    test_api(
        "amdsmi_get_cpu_socket_energy",
        lambda: amdsmi.amdsmi_get_cpu_socket_energy(cpu_socket) if cpu_socket else None,
    )

    test_api("amdsmi_get_threads_per_core", lambda: amdsmi.amdsmi_get_threads_per_core())

    test_api(
        "amdsmi_get_cpu_prochot_status",
        lambda: amdsmi.amdsmi_get_cpu_prochot_status(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_fclk_mclk",
        lambda: amdsmi.amdsmi_get_cpu_fclk_mclk(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_cclk_limit",
        lambda: amdsmi.amdsmi_get_cpu_cclk_limit(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_socket_current_active_freq_limit",
        lambda: (
            amdsmi.amdsmi_get_cpu_socket_current_active_freq_limit(cpu_socket)
            if cpu_socket
            else None
        ),
    )

    test_api(
        "amdsmi_get_cpu_socket_freq_range",
        lambda: amdsmi.amdsmi_get_cpu_socket_freq_range(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_core_current_freq_limit",
        lambda: amdsmi.amdsmi_get_cpu_core_current_freq_limit(cpu_core) if cpu_core else None,
    )

    test_api(
        "amdsmi_get_cpu_socket_power",
        lambda: amdsmi.amdsmi_get_cpu_socket_power(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_socket_power_cap",
        lambda: amdsmi.amdsmi_get_cpu_socket_power_cap(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_socket_power_cap_max",
        lambda: amdsmi.amdsmi_get_cpu_socket_power_cap_max(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_pwr_svi_telemetry_all_rails",
        lambda: (
            amdsmi.amdsmi_get_cpu_pwr_svi_telemetry_all_rails(cpu_socket) if cpu_socket else None
        ),
    )

    test_api("amdsmi_set_cpu_socket_power_cap", None, skip_reason="Risky - modifies CPU power cap")

    test_api(
        "amdsmi_set_cpu_pwr_efficiency_mode",
        None,
        skip_reason="Risky - modifies CPU power efficiency mode",
    )

    test_api(
        "amdsmi_get_cpu_pwr_efficiency_mode",
        lambda: amdsmi.amdsmi_get_cpu_pwr_efficiency_mode(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_core_boostlimit",
        lambda: amdsmi.amdsmi_get_cpu_core_boostlimit(cpu_core) if cpu_core else None,
    )

    test_api(
        "amdsmi_get_cpu_socket_c0_residency",
        lambda: amdsmi.amdsmi_get_cpu_socket_c0_residency(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_set_cpu_core_boostlimit", None, skip_reason="Risky - modifies CPU core boost limit"
    )

    test_api(
        "amdsmi_set_cpu_socket_boostlimit",
        None,
        skip_reason="Risky - modifies CPU socket boost limit",
    )

    test_api(
        "amdsmi_get_cpu_ddr_bw",
        lambda: amdsmi.amdsmi_get_cpu_ddr_bw(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_socket_temperature",
        lambda: amdsmi.amdsmi_get_cpu_socket_temperature(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_dimm_temp_range_and_refresh_rate",
        lambda: (
            amdsmi.amdsmi_get_cpu_dimm_temp_range_and_refresh_rate(cpu_socket, 0)
            if cpu_socket
            else None
        ),
    )

    test_api(
        "amdsmi_get_cpu_dimm_power_consumption",
        lambda: amdsmi.amdsmi_get_cpu_dimm_power_consumption(cpu_socket, 0) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_dimm_thermal_sensor",
        lambda: amdsmi.amdsmi_get_cpu_dimm_thermal_sensor(cpu_socket, 0) if cpu_socket else None,
    )

    test_api("amdsmi_set_cpu_xgmi_width", None, skip_reason="Risky - modifies CPU XGMI width")

    test_api(
        "amdsmi_set_cpu_gmi3_link_width_range", None, skip_reason="Risky - modifies GMI3 link width"
    )

    test_api(
        "amdsmi_cpu_apb_enable",
        lambda: amdsmi.amdsmi_cpu_apb_enable(cpu_socket) if cpu_socket else None,
    )

    test_api("amdsmi_cpu_apb_disable", None, skip_reason="Risky - disables CPU APB")

    test_api(
        "amdsmi_set_cpu_socket_lclk_dpm_level", None, skip_reason="Risky - modifies LCLK DPM level"
    )

    test_api(
        "amdsmi_get_cpu_socket_lclk_dpm_level",
        lambda: amdsmi.amdsmi_get_cpu_socket_lclk_dpm_level(cpu_socket, 0) if cpu_socket else None,
    )

    test_api("amdsmi_set_cpu_pcie_link_rate", None, skip_reason="Risky - modifies PCIe link rate")

    test_api(
        "amdsmi_set_cpu_df_pstate_range", None, skip_reason="Risky - modifies DF P-state range"
    )

    test_api(
        "amdsmi_get_cpu_current_io_bandwidth",
        lambda: amdsmi.amdsmi_get_cpu_current_io_bandwidth(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_current_xgmi_bw",
        lambda: amdsmi.amdsmi_get_cpu_current_xgmi_bw(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_hsmp_metrics_table_version",
        lambda: amdsmi.amdsmi_get_hsmp_metrics_table_version(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_set_cpu_xgmi_pstate_range", None, skip_reason="Risky - modifies XGMI P-state range"
    )

    test_api(
        "amdsmi_get_cpu_xgmi_pstate_range",
        lambda: amdsmi.amdsmi_get_cpu_xgmi_pstate_range(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_set_cpu_rail_isofreq_policy",
        None,
        skip_reason="Risky - modifies rail isofreq policy",
    )

    test_api(
        "amdsmi_get_cpu_rail_isofreq_policy",
        lambda: amdsmi.amdsmi_get_cpu_rail_isofreq_policy(cpu_socket) if cpu_socket else None,
    )

    test_api("amdsmi_set_cpu_dfc_ctrl", None, skip_reason="Risky - modifies CPU DFC control")

    test_api(
        "amdsmi_get_cpu_dfc_ctrl",
        lambda: amdsmi.amdsmi_get_cpu_dfc_ctrl(cpu_socket) if cpu_socket else None,
    )

    test_api("amdsmi_set_cpu_pc6_enable", None, skip_reason="Risky - modifies CPU PC6 state")

    test_api(
        "amdsmi_get_cpu_pc6_enable",
        lambda: amdsmi.amdsmi_get_cpu_pc6_enable(cpu_socket) if cpu_socket else None,
    )

    test_api("amdsmi_set_cpu_cc6_enable", None, skip_reason="Risky - modifies CPU CC6 state")

    test_api(
        "amdsmi_get_cpu_cc6_enable",
        lambda: amdsmi.amdsmi_get_cpu_cc6_enable(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_dimm_sb_reg",
        lambda: amdsmi.amdsmi_get_cpu_dimm_sb_reg(cpu_socket, 0, 0) if cpu_socket else None,
    )

    test_api(
        "amdsmi_set_cpu_dimm_sb_reg", None, skip_reason="Risky - modifies DIMM sideband registers"
    )

    test_api(
        "amdsmi_get_cpu_tdelta",
        lambda: amdsmi.amdsmi_get_cpu_tdelta(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_svi3_vr_controller_temp",
        lambda: amdsmi.amdsmi_get_cpu_svi3_vr_controller_temp(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_enabled_commands",
        lambda: amdsmi.amdsmi_get_cpu_enabled_commands(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_core_floor_freq_limit",
        lambda: amdsmi.amdsmi_get_cpu_core_floor_freq_limit(cpu_core) if cpu_core else None,
    )

    test_api(
        "amdsmi_get_cpu_floor_freq_limit",
        lambda: amdsmi.amdsmi_get_cpu_floor_freq_limit(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_cpu_core_eff_floor_freq_limit",
        lambda: amdsmi.amdsmi_get_cpu_core_eff_floor_freq_limit(cpu_core) if cpu_core else None,
    )

    test_api(
        "amdsmi_get_cpu_eff_floor_freq_limit",
        lambda: amdsmi.amdsmi_get_cpu_eff_floor_freq_limit(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_set_cpu_core_floor_freq_limit",
        None,
        skip_reason="Risky - modifies CPU core floor frequency",
    )

    test_api(
        "amdsmi_set_cpu_floor_freq_limit", None, skip_reason="Risky - modifies CPU floor frequency"
    )

    test_api(
        "amdsmi_set_cpu_msr_floor_freq_limit",
        None,
        skip_reason="Risky - modifies CPU MSR floor frequency",
    )

    test_api(
        "amdsmi_set_cpu_core_msr_floor_freq_limit",
        None,
        skip_reason="Risky - modifies CPU core MSR floor frequency",
    )

    test_api(
        "amdsmi_get_cpu_freq_range",
        lambda: amdsmi.amdsmi_get_cpu_freq_range(cpu_socket) if cpu_socket else None,
    )

    test_api("amdsmi_set_cpu_sdps_limit", None, skip_reason="Risky - modifies CPU SDPS limit")

    test_api(
        "amdsmi_get_cpu_sdps_limit",
        lambda: amdsmi.amdsmi_get_cpu_sdps_limit(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_get_hsmp_metrics_table",
        lambda: amdsmi.amdsmi_get_hsmp_metrics_table(cpu_socket) if cpu_socket else None,
    )

    test_api(
        "amdsmi_first_online_core_on_cpu_socket",
        lambda: amdsmi.amdsmi_first_online_core_on_cpu_socket(cpu_socket) if cpu_socket else None,
    )

    test_api("amdsmi_get_cpu_family", lambda: amdsmi.amdsmi_get_cpu_family())

    test_api("amdsmi_get_cpu_model", lambda: amdsmi.amdsmi_get_cpu_model())

    test_api(
        "amdsmi_get_cpu_model_name",
        lambda: amdsmi.amdsmi_get_cpu_model_name(cpu_socket) if cpu_socket else None,
    )

    test_api("amdsmi_get_esmi_err_msg", lambda: amdsmi.amdsmi_get_esmi_err_msg(0))

    # ========================================================================
    # MULTI-GPU SANITY SWEEP (only when --all-gpus is used)
    # ========================================================================
    if CONFIG.get("all_gpus") and len(gpu_handles) > 1:
        print()
        print("=" * 70)
        print(f"MULTI-GPU SANITY SWEEP ({len(gpu_handles) - 1} additional GPU(s))")
        print("=" * 70)
        print()

        sanity_apis = [
            ("amdsmi_get_gpu_device_bdf", lambda h: amdsmi.amdsmi_get_gpu_device_bdf(h)),
            ("amdsmi_get_gpu_device_uuid", lambda h: amdsmi.amdsmi_get_gpu_device_uuid(h)),
            ("amdsmi_get_gpu_asic_info", lambda h: amdsmi.amdsmi_get_gpu_asic_info(h)),
            ("amdsmi_get_gpu_board_info", lambda h: amdsmi.amdsmi_get_gpu_board_info(h)),
            ("amdsmi_get_gpu_vram_usage", lambda h: amdsmi.amdsmi_get_gpu_vram_usage(h)),
            ("amdsmi_get_gpu_activity", lambda h: amdsmi.amdsmi_get_gpu_activity(h)),
            ("amdsmi_get_power_info", lambda h: amdsmi.amdsmi_get_power_info(h)),
            (
                "amdsmi_get_clock_info",
                lambda h: amdsmi.amdsmi_get_clock_info(h, need("AmdSmiClkType").GFX),
            ),
            (
                "amdsmi_get_temp_metric",
                lambda h: amdsmi.amdsmi_get_temp_metric(
                    h, need("AmdSmiTemperatureType").EDGE, need("AmdSmiTemperatureMetric").CURRENT
                ),
            ),
        ]

        primary_index = CONFIG.get("gpu_index", 0)
        for idx, h in enumerate(gpu_handles):
            if idx == primary_index:
                continue
            for api_name, fn in sanity_apis:
                tagged = f"[gpu{idx}] {api_name}"
                test_api(tagged, lambda fn=fn, h=h: fn(h))

    # ========================================================================
    # REVERT SECTION
    # ========================================================================
    print()
    print("=" * 70)
    print("REVERTING WRITE OPERATIONS TO SAFE DEFAULTS")
    print("=" * 70)
    print()

    if CONFIG.get("no_revert"):
        print("  [INFO] --no-revert supplied; leaving any mutated state as-is.")
    elif CONFIG.get("skip_writes"):
        print("  [INFO] --skip-writes supplied; no setters ran, nothing to revert.")
    elif not is_root():
        print("  [INFO] Running as non-root user - write operations were skipped.")
        print("  [INFO] No changes were made, nothing to revert.")
    else:
        # Revert perf level to AUTO. Some builds return a string from the
        # getter that the setter won't accept, so always use the enum here.
        try:
            if hasattr(amdsmi, "AmdSmiDevPerfLevel"):
                amdsmi.amdsmi_set_gpu_perf_level(gpu_handle, amdsmi.AmdSmiDevPerfLevel.AUTO)
                print("  Reverted perf level to AUTO")
        except Exception as e:
            print(f"  Could not revert perf level: {str(e).splitlines()[0]}")

        # Revert power cap to saved value.
        try:
            if saved_values.get("power_cap"):
                amdsmi.amdsmi_set_power_cap(gpu_handle, 0, saved_values["power_cap"])
                print(f"  Reverted power_cap to {saved_values['power_cap']}")
        except Exception as e:
            print(f"  Could not revert power_cap: {str(e).splitlines()[0]}")

        # Revert fan to automatic; ignore NOT_SUPPORTED quietly.
        try:
            amdsmi.amdsmi_reset_gpu_fan(gpu_handle, 0)
            print("  Reverted fan to automatic")
        except Exception as e:
            msg = str(e)
            if "NOT_SUPPORTED" in msg:
                print("  Fan revert skipped (no controllable fan)")
            else:
                print(f"  Could not revert fan: {msg.splitlines()[0]}")

        # Revert process isolation
        if saved_values["process_isolation"] is not None:
            try:
                amdsmi.amdsmi_set_gpu_process_isolation(
                    gpu_handle, saved_values["process_isolation"]
                )
                print(f"  Reverted process_isolation to {saved_values['process_isolation']}")
            except Exception as e:
                print(f"  Could not revert process_isolation: {str(e).splitlines()[0]}")

    # Shutdown
    amdsmi.amdsmi_shut_down()
    print("  AMD SMI shut down successfully")

    # Generate report
    generate_report(gpu_name, bdf)


# ============================================================================
# REPORT GENERATION
# ============================================================================


def generate_report(gpu_name, bdf):
    """Generate a markdown report of all test results."""
    run_as = "ROOT" if is_root() else "USER"
    supported = [r for r in results if r["status"] == "Supported"]
    unsupported = [r for r in results if r["status"] == "Unsupported"]
    not_found = [r for r in results if r["status"] == "API Not Found"]
    skipped = [r for r in results if r["status"] == "Skipped"]
    script_errors = [r for r in results if r["status"] == "Script Error"]

    if CONFIG.get("output"):
        outfile = os.path.abspath(CONFIG["output"])
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"test_amdsmi_all_apis_output_{ts}.md"
        )

    try:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
    except Exception:
        pass

    try:
        f_out = open(outfile, "w", encoding="utf-8")
    except OSError as e:
        print(f"WARNING: Cannot write report to {outfile}: {e}")
        print("Report not written. Final summary below.")
        print(
            f"FINAL SUMMARY: {len(supported)} Supported / "
            f"{len(unsupported)} Unsupported / {len(skipped)} Skipped / "
            f"{len(script_errors)} Script Errors / {len(not_found)} Not Found / "
            f"{len(results)} Total"
        )
        return

    with f_out as f:
        f.write(f"# AMD SMI Complete API Test Results\n\n")
        f.write(f"**Script Version:** {__version__}  \n")
        f.write(f"**Test Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**System:** {gpu_name}  \n")
        f.write(f"**BDF:** {bdf}  \n")
        f.write(f"**Primary GPU Index:** {CONFIG.get('gpu_index', 0)}  \n")
        f.write(f"**GPUs:** {len(gpu_handles)}  \n")
        f.write(f"**Sockets:** {len(socket_handles)}  \n")
        f.write(f"**CPU Sockets:** {len(cpu_socket_handles)}  \n")
        f.write(f"**CPU Cores:** {len(cpu_core_handles)}  \n")
        f.write(f"**Platform:** Bare Metal  \n")
        f.write(f"**Run As:** {run_as}  \n")
        f.write(f"**Python:** {sys.executable} ({platform.python_version()})  \n")
        f.write(f"**amdsmi module:** {getattr(amdsmi, '__file__', 'unknown')}  \n")
        f.write(f"**Hostname:** {get_hostname()}  \n\n")
        f.write("---\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write("| Metric | Count |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total APIs Tested | {len(results)} |\n")
        f.write(f"| Supported | {len(supported)} |\n")
        f.write(f"| Unsupported | {len(unsupported)} |\n")
        f.write(f"| Skipped | {len(skipped)} |\n")
        f.write(f"| Script Error | {len(script_errors)} |\n")
        f.write(f"| API Not Found | {len(not_found)} |\n\n")
        f.write("---\n\n")

        # Complete results table
        f.write("## Complete Results\n\n")
        f.write("| # | API | Status | Output / Note |\n")
        f.write("|---|-----|--------|---------------|\n")
        for i, r in enumerate(results, 1):
            detail = r["output"] if r["output"] else r["note"]
            detail = detail.replace("|", "/").replace("\n", " ")[:100]
            f.write(f"| {i} | `{r['api']}` | {r['status']} | {detail} |\n")

        f.write("\n---\n\n")

        # Unsupported APIs detail
        if unsupported:
            f.write("## ❌ Unsupported APIs Detail\n\n")
            for r in unsupported:
                note = r["note"].replace("\n", " ")
                f.write(f"- **`{r['api']}`**: {note[:120]}\n")
            f.write("\n")

        # APIs Not Found
        if not_found:
            f.write("## ❓ APIs Not Found\n\n")
            for r in not_found:
                note = r["note"].replace("\n", " ")
                f.write(f"- **`{r['api']}`**: {note[:120]}\n")
            f.write("\n")

        # Supported APIs list
        f.write("## ✅ Supported APIs\n\n")
        for r in supported:
            f.write(f"- `{r['api']}`\n")
        f.write("\n")

    print()
    print(f"Report written to: {outfile}")
    print()
    print("=" * 70)
    print(
        f"FINAL SUMMARY: {len(supported)} Supported / "
        f"{len(unsupported)} Unsupported / {len(skipped)} Skipped / "
        f"{len(script_errors)} Script Errors / {len(not_found)} Not Found / "
        f"{len(results)} Total"
    )
    print("=" * 70)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Test AMD SMI Python APIs and emit a markdown report."
    )
    parser.add_argument(
        "--gpu-index", type=int, default=0, help="Index of the GPU to target (default: 0)."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the markdown report (default: timestamped file next to this script).",
    )
    parser.add_argument(
        "--skip-writes",
        action="store_true",
        help="Skip every API marked requires_root (no setters will run).",
    )
    parser.add_argument(
        "--no-revert",
        action="store_true",
        help="Do not attempt to restore mutated state at the end.",
    )
    parser.add_argument(
        "--all-gpus",
        action="store_true",
        help="After the primary sweep, run a small read-only sanity "
        "sweep on every other GPU so multi-GPU systems are also "
        "validated. The primary sweep still runs only on --gpu-index.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    CONFIG["gpu_index"] = args.gpu_index
    CONFIG["output"] = args.output
    CONFIG["skip_writes"] = args.skip_writes
    CONFIG["no_revert"] = args.no_revert
    CONFIG["all_gpus"] = args.all_gpus

    run_tests()


if __name__ == "__main__":
    main()
