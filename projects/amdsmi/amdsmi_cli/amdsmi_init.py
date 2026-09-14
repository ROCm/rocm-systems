#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

### Handle safe initialization for amdsmi

import atexit
import logging
import os
import signal
import sys
import threading
from pathlib import Path

# CLI module resolution order (distinct from `import amdsmi` in a user script):
#   1. this installation's share/amd_smi copy -- the modules the CLI shipped
#      with, always preferred so `amd-smi` uses its own version even on a host
#      with multiple ROCm installs or a pip-installed amdsmi (restores #3082).
#   2. a pip install -- fallback when the share copy is absent; the natural
#      import below finds it (pip's site-packages precedes the system copy on
#      sys.path).
#   3. the system site-packages -- last resort.
# A pip install is meant for Python scripting, not for changing CLI behavior,
# so it must not override the shipped modules; the fallbacks are safety nets.
_share_candidates = []
_rocm = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
if _rocm:
    _share_candidates.append(os.path.join(_rocm, "share", "amd_smi"))
# CLI runs from ROCM_PATH/libexec/amdsmi_cli/, so ../../share/amd_smi is the
# installed copy relative to this file.
_share_candidates.append(str(Path(__file__).resolve().parent.parent.parent / "share" / "amd_smi"))
for _cand in _share_candidates:
    if os.path.isdir(os.path.join(_cand, "amdsmi")):
        sys.path.insert(0, _cand)
        break

try:
    from amdsmi import amdsmi_exception, amdsmi_interface
except ImportError as e:
    print(f"Unhandled import error: {e}")
    print(
        "Failed to import the amdsmi Python library. Install amd-smi-lib (rpm/deb) or pip install the amdsmi wheel."
    )
    sys.exit(1)

# Using basic python logging for user errors and development
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.ERROR)  # User level logging
# This traceback limit only affects this file, once the code hit's the cli portion it gets reset to the user's preference
sys.tracebacklimit = -1  # Disable traceback when raising errors

# On initial import set initialized variable
AMDSMI_INITIALIZED = False
AMDSMI_INIT_FLAG = amdsmi_interface.AmdSmiInitFlags.INIT_ALL_PROCESSORS
AMD_VENDOR_ID = 4098


def check_wsl_dxg():
    """Returns true if running under WSL2 (not Hyper-V) with /dev/dxg present.

    /dev/dxg is created by dxgkrnl for any WDDM GPU, so we additionally require
    the WSL2 kernel signature in /proc/version to avoid false-positives on
    Hyper-V Linux guests or native hosts where dxgkrnl may be loaded.
    Vendor confirmation (AMD 0x1002) is deferred to amdsmi_init(), which calls
    into librocdxg and rejects non-AMD adapters.
    """
    if not Path("/dev/dxg").exists():
        return False
    try:
        osrelease = Path("/proc/sys/kernel/osrelease").read_text(encoding="ascii").lower()
        return "microsoft" in osrelease and "wsl" in osrelease
    except OSError:
        return False


def check_amdgpu_driver():
    """Returns true if amdgpu is found in the list of initialized modules"""
    # WSL2: no native amdgpu module; use /dev/dxg via dxgkrnl instead.
    # check_wsl_dxg() requires the WSL2 kernel string, so this is safe for
    # Hyper-V guests (which lack "wsl" in osrelease) and bare-metal hosts.
    # Vendor confirmation (AMD 0x1002) is done inside librocdxg during amdsmi_init().
    if check_wsl_dxg():
        return True

    amd_gpu_status_file = Path("/sys/module/amdgpu/initstate")
    if amd_gpu_status_file.exists():
        try:
            return amd_gpu_status_file.read_text(encoding="ascii").strip() == "live"
        except OSError:
            pass

    # If the driver is loaded either as a module OR built in, this dir will be populated
    drv = Path("/sys/bus/pci/drivers/amdgpu")
    if not drv.exists():
        return False

    # Check if a symlink exists that loosely matches PCI BDF format
    # ex: 0000:03:00.0
    for p in drv.iterdir():
        if p.is_symlink() and ":" in p.name and "." in p.name:
            return True
    return False


def check_amd_hsmp_driver():
    """Returns true if amd_hsmp or hsmp_acpi is found in the list of initialized modules"""
    amd_cpu_status_file = Path("/dev/hsmp")
    if amd_cpu_status_file.exists():
        return True
    return False


def check_amd_ionic_driver():
    """Returns true if ionic is found in the list of initialized modules"""
    status_file = Path("/sys/module/ionic/initstate")
    if status_file.exists():
        if status_file.read_text(encoding="ascii").strip() == "live":
            return True
    return False


def check_brcm_nic_driver():
    """Returns true if bnxt_en is found in the list of initialized modules"""
    status_file = Path("/sys/module/bnxt_en/initstate")
    try:
        if status_file.exists():
            if status_file.read_text(encoding="ascii").strip() == "live":
                return True
    except OSError:
        return False
    return False


def amdsmi_cli_init():
    """Initializes AMDSMI Library for the CLI

    Probes for the presence of the amdgpu, amd_hsmp/hsmp_acpi, ionic, and bnxt_en
    drivers and initializes the AMD SMI library based on the live drivers found.

    Return:
        init_flag: the flag used to initialize the AMD SMI library without error

    Raises:
        err: AmdSmiLibraryException if not successful in initializing any drivers
    """
    init_flag = 0
    cpu_init_disabled = os.environ.get("AMDSMI_DISABLE_CPU_INIT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if check_amdgpu_driver():
        init_flag |= amdsmi_interface.AmdSmiInitFlags.INIT_AMD_GPUS
        logging.debug("amdgpu driver's initstate is live")
    if cpu_init_disabled:
        logging.debug("CPU/ESMI init disabled via AMDSMI_DISABLE_CPU_INIT")
    # amdsmi_get_cpu_handles has shipped in every supported libamd_smi.so
    # (ROCm 5.6+), so the previous hasattr() guard here was always true; it
    # was removed because the regenerated wrapper binds the symbol directly.
    elif check_amd_hsmp_driver():
        init_flag |= amdsmi_interface.AmdSmiInitFlags.INIT_AMD_CPUS
        logging.debug("hsmp driver's initstate is live")
    if check_amd_ionic_driver():
        logging.debug("ionic driver's initstate is live")
        init_flag |= amdsmi_interface.AmdSmiInitFlags.INIT_AMD_NICS
    if check_brcm_nic_driver():
        logging.debug("bnxt_en driver's initstate is live")
        init_flag |= amdsmi_interface.AmdSmiInitFlags.INIT_AMD_NICS

    _INIT_TIMEOUT_SEC = 60
    init_result = {"exception": None}

    def _run_init():
        try:
            amdsmi_interface.amdsmi_init(init_flag)
        except Exception as e:
            init_result["exception"] = e

    init_thread = threading.Thread(target=_run_init, daemon=True)
    init_thread.start()
    init_thread.join(timeout=_INIT_TIMEOUT_SEC)

    if init_thread.is_alive():
        logging.error(
            "amdsmi_init() timed out after %ds. The GPU driver may be unresponsive.",
            _INIT_TIMEOUT_SEC,
        )
        sys.exit(2)

    if isinstance(
        init_result["exception"],
        (amdsmi_interface.AmdSmiLibraryException, amdsmi_interface.AmdSmiParameterException),
    ):
        e = init_result["exception"]
        # parameter exception thrown if init_flag is 0, but err_code will be set to 0 in that case, so must check if init_flag is 0 too
        if (
            e.err_code
            in (
                amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT,
                amdsmi_interface.amdsmi_wrapper.AMDSMI_STATUS_DRIVER_NOT_LOADED,
            )
            or init_flag == 0
        ):
            logging.error(
                "Drivers not loaded (amdgpu, amd_hsmp, ionic, bnxt_en drivers not found in modules)"
            )
            sys.exit(-1)
        else:
            raise e
    elif init_result["exception"] is not None:
        raise init_result["exception"]

    logging.debug(
        f"AMDSMI initialized with at least one driver successfully | init flag: {init_flag}"
    )

    return init_flag


def amdsmi_cli_shutdown():
    """Shutdown AMDSMI instance

    Raises:
        err: AmdSmiLibraryException if not successful
    """
    try:
        amdsmi_interface.amdsmi_shut_down()
    except amdsmi_exception.AmdSmiLibraryException as e:
        logging.error("Unable to cleanly shut down amd-smi-lib")
        raise e


def signal_handler(sig, frame):
    logging.debug(f"Handling signal: {sig}")
    try:
        sys.exit(0)
    except Exception as e:
        logging.error(
            "Unable to cleanly shut down amd-smi-lib, exception: %s", str(type(e).__name__)
        )
        os._exit(0)


if not AMDSMI_INITIALIZED:
    AMDSMI_INIT_FLAG = amdsmi_cli_init()
    AMDSMI_INITIALIZED = True
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(amdsmi_cli_shutdown)
