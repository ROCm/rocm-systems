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
"""Memory partition driver lifecycle integration test.

Exercises the full set -> driver reload -> verify round-trip that cannot be
covered by the standard functional test suite.

Requirements
------------
- Root privileges  (modprobe requires root)
- kmod installed   (modprobe must be on PATH)
- No GPU workloads running during the test
- Run in isolation -- NOT via the standard unittest/pytest runner

Usage
-----
  sudo python3 tests/python/integration/test_memory_partition_lifecycle.py [--target NPS2]

Exit codes
----------
  0  all phases passed
  1  a phase failed (reason printed to stdout)
  2  precondition not met (no root, no kmod, bad --target)
  3  unexpected exception
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import amdsmi

_MODPROBE_TIMEOUT_SEC = 120
_DRIVER_SETTLE_SEC = 5

_NPS_NAME_TO_TYPE = {
    "NPS1": amdsmi.AmdSmiMemoryPartitionType.NPS1,
    "NPS2": amdsmi.AmdSmiMemoryPartitionType.NPS2,
    "NPS4": amdsmi.AmdSmiMemoryPartitionType.NPS4,
    "NPS8": amdsmi.AmdSmiMemoryPartitionType.NPS8,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _print(msg: str) -> None:
    print(msg, flush=True)


def _fail(msg: str) -> int:
    _print(f"  [FAIL] {msg}")
    return 1


def _check_preconditions(target_mode: str) -> int:
    """Return 0 if all preconditions are satisfied, 2 otherwise."""
    if target_mode not in _NPS_NAME_TO_TYPE:
        _print(
            f"[ERROR] Unknown target mode '{target_mode}'. Choose from: {list(_NPS_NAME_TO_TYPE)}"
        )
        return 2

    if os.geteuid() != 0:
        _print("[ERROR] This test requires root privileges (run with sudo).")
        return 2

    if not shutil.which("modprobe"):
        _print("[ERROR] modprobe not found. Install kmod and retry.")
        return 2

    return 0


def _reload_driver() -> bool:
    """
    Unload then reload the amdgpu kernel module.

    Uses two separate subprocess.run() calls with shell=False so the arguments
    are passed directly to execvp -- no shell is involved and there is no
    injection surface. Returns True on success.
    """
    _print("  Running: modprobe -r amdgpu")
    try:
        subprocess.run(["modprobe", "-r", "amdgpu"], check=True, timeout=_MODPROBE_TIMEOUT_SEC)
    except subprocess.CalledProcessError as e:
        _print(f"  [FAIL] modprobe -r amdgpu exited with code {e.returncode}; check dmesg")
        return False
    except subprocess.TimeoutExpired:
        _print(f"  [FAIL] modprobe -r amdgpu timed out after {_MODPROBE_TIMEOUT_SEC}s")
        return False

    _print("  Running: modprobe amdgpu")
    try:
        subprocess.run(["modprobe", "amdgpu"], check=True, timeout=_MODPROBE_TIMEOUT_SEC)
    except subprocess.CalledProcessError as e:
        _print(f"  [FAIL] modprobe amdgpu exited with code {e.returncode}; check dmesg")
        return False
    except subprocess.TimeoutExpired:
        _print(f"  [FAIL] modprobe amdgpu timed out after {_MODPROBE_TIMEOUT_SEC}s")
        return False

    _print(f"  Driver reloaded. Waiting {_DRIVER_SETTLE_SEC}s for devices to settle...")
    time.sleep(_DRIVER_SETTLE_SEC)
    return True


def _init_and_get_primary() -> tuple:
    """
    Init amdsmi and return (gpu_handle, partition_string) for the first GPU.
    Caller is responsible for calling amdsmi.amdsmi_shut_down() when done.
    Raises RuntimeError if no GPUs are found.
    """
    amdsmi.amdsmi_init()
    gpus = amdsmi.amdsmi_get_processor_handles()
    if not gpus:
        amdsmi.amdsmi_shut_down()
        raise RuntimeError("No GPU handles returned after amdsmi_init()")
    gpu = gpus[0]
    partition = amdsmi.amdsmi_get_gpu_memory_partition(gpu)
    return gpu, partition


# ---------------------------------------------------------------------------
# Test phases
# ---------------------------------------------------------------------------


def _phase1_read_original() -> tuple:
    """Return (gpu, original_partition). Shuts down amdsmi before returning."""
    _print("\n=== Phase 1: Read original partition ===")
    gpu, original_partition = _init_and_get_primary()
    _print(f"  Original partition: {original_partition}")
    amdsmi.amdsmi_shut_down()
    return gpu, original_partition


def _phase2_check_capabilities(target_mode: str) -> bool:
    """
    Return True if target_mode is advertised by hardware capabilities.
    Returns True unconditionally if the config API is unavailable (older kernel).
    Shuts down amdsmi before returning.
    """
    _print("\n=== Phase 2: Check hardware capabilities ===")
    gpu, _ = _init_and_get_primary()
    supported = True
    try:
        config = amdsmi.amdsmi_get_gpu_memory_partition_config(gpu)
        caps = config.get("partition_caps", [])
        _print(f"  Hardware capabilities: {caps}")
        if target_mode not in caps:
            _print(f"  [SKIP] {target_mode} not advertised in partition_caps")
            supported = False
    except amdsmi.AmdSmiLibraryException as e:
        _print(f"  [WARN] amdsmi_get_gpu_memory_partition_config unavailable: {e}")
        _print("  Proceeding without capability pre-check.")
    amdsmi.amdsmi_shut_down()
    return supported


def _phase3_stage_change(target_mode: str) -> int:
    """Stage the partition change. Returns 0 on success, 1 on failure."""
    _print(f"\n=== Phase 3: Stage partition change to {target_mode} ===")
    gpu, _ = _init_and_get_primary()
    try:
        amdsmi.amdsmi_set_gpu_memory_partition(gpu, _NPS_NAME_TO_TYPE[target_mode])
        _print(f"  amdsmi_set_gpu_memory_partition({target_mode}): SUCCESS")
    except amdsmi.AmdSmiLibraryException as e:
        amdsmi.amdsmi_shut_down()
        return _fail(f"amdsmi_set_gpu_memory_partition failed: {e}")
    amdsmi.amdsmi_shut_down()
    return 0


def _phase4_reload() -> int:
    """Reload the driver. Returns 0 on success, 1 on failure."""
    _print("\n=== Phase 4: Reload driver to apply change ===")
    return 0 if _reload_driver() else _fail("Driver reload failed")


def _phase5_verify(expected_mode: str) -> int:
    """Re-init and verify the active partition matches expected_mode. Returns 0 or 1."""
    _print(f"\n=== Phase 5: Re-init and verify partition is {expected_mode} ===")
    _, post_partition = _init_and_get_primary()
    _print(f"  Post-reload partition: {post_partition}")
    amdsmi.amdsmi_shut_down()
    if post_partition != expected_mode:
        return _fail(f"Expected {expected_mode}, got {post_partition}")
    _print(f"  [PASS] Partition is {post_partition}")
    return 0


def _phase6_restore(original_partition: str) -> int:
    """Stage and apply restore to original_partition. Returns 0 or 1."""
    _print(f"\n=== Phase 6: Restore to {original_partition} ===")
    gpu, _ = _init_and_get_primary()
    try:
        amdsmi.amdsmi_set_gpu_memory_partition(gpu, _NPS_NAME_TO_TYPE[original_partition])
        _print(f"  amdsmi_set_gpu_memory_partition({original_partition}): SUCCESS")
    except amdsmi.AmdSmiLibraryException as e:
        amdsmi.amdsmi_shut_down()
        return _fail(f"Restore set failed: {e}")
    amdsmi.amdsmi_shut_down()

    if not _reload_driver():
        return _fail("Driver reload for restore failed")

    _, restored = _init_and_get_primary()
    amdsmi.amdsmi_shut_down()
    _print(f"  Post-restore partition: {restored}")
    if restored != original_partition:
        return _fail(f"Restore failed: expected {original_partition}, got {restored}")
    _print(f"  [PASS] Partition restored to {restored}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(target_mode: str) -> int:
    rc = _check_preconditions(target_mode)
    if rc != 0:
        return rc

    try:
        _, original_partition = _phase1_read_original()

        if original_partition == target_mode:
            _print(
                f"\n[SKIP] Device is already in {target_mode}. "
                "Choose a different --target to exercise a partition change."
            )
            return 0

        if original_partition not in _NPS_NAME_TO_TYPE:
            _print(f"\n[ERROR] Original partition '{original_partition}' is not a known NPS mode.")
            return 1

        if not _phase2_check_capabilities(target_mode):
            return 0  # hardware does not support this mode -- skip, not fail

        rc = _phase3_stage_change(target_mode)
        if rc != 0:
            return rc

        rc = _phase4_reload()
        if rc != 0:
            # Driver did not reload -- attempt restore so the system is not left
            # with a staged-but-unapplied partition change.
            _print("  Attempting restore despite reload failure...")
            _phase6_restore(original_partition)
            return rc

        rc = _phase5_verify(target_mode)
        # Always attempt restore, even if verify failed.
        restore_rc = _phase6_restore(original_partition)

        if rc != 0:
            return rc
        if restore_rc != 0:
            return restore_rc

    except Exception as e:
        _print(f"\n[ERROR] Unexpected exception: {e}")
        try:
            amdsmi.amdsmi_shut_down()
        except Exception:
            pass
        return 3

    _print(
        f"\n[PASS] Full lifecycle test passed: "
        f"{original_partition} -> {target_mode} -> {original_partition}"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Memory partition driver lifecycle integration test."
    )
    parser.add_argument(
        "--target",
        default="NPS2",
        choices=list(_NPS_NAME_TO_TYPE),
        help="Partition mode to change to (default: NPS2)",
    )
    args = parser.parse_args()
    sys.exit(run(args.target))
