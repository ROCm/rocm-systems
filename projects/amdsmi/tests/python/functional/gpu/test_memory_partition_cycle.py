#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cycle a GPU through every NPS mode: set -> driver reload -> verify -> restore.

Laid out in phases, mirroring example/amd_smi_partition_example.py:

  Phase 1  Report the current mode and the modes the device advertises.
  Phase 2  For each mode: stage it, reload amdgpu, re-enumerate, verify.
  Phase 3  Put the device back on the mode the run started with.

Every call is issued against every handle amdsmi enumerates, and a mode counts as
applied only once all handles that can report one agree on it. In a partitioned
layout the sub-partition handles return NOT_SUPPORTED for the memory partition
APIs; those are skipped rather than counted as a mismatch.

The driver is only reloaded when a reload can actually change something -- a
mode the device rejects, or a mode that is already active, is reported and
skipped. Every external wait (modprobe, device re-enumeration) is bounded by a
timeout so a stuck reload fails the test instead of hanging the suite.

NOTE: Partition changes alter device topology -- AMD SMI must re-initialize.
AMD SMI builds its device table at amdsmi_init() and does not update it live, so
every processor handle is valid only inside the session that produced it. Each
phase below therefore opens its own AmdsmiSession rather than caching a handle.

Requirements
------------
- Root privileges  (modprobe requires root)
- kmod installed   (modprobe must be on PATH)
- No GPU workloads running during the run

Usage
-----
  1. Install amd-smi-lib & amd-smi-lib-test packages
  2. sudo /opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py \
         -k test_cycle_memory_partition_modes -v
"""

import os
import shutil
import subprocess
import time
import unittest

import common.common as common
from common.common import amdsmi

# modprobe is slow on large hives and the devices then come back asynchronously,
# so the unload/load call and the re-enumeration poll get separate bounds.
_MODPROBE_TIMEOUT_SEC = 120
_DEVICE_READY_TIMEOUT_SEC = 60
_DEVICE_POLL_SEC = 2
# A modprobe stuck in the driver stays in uninterruptible sleep and ignores
# SIGKILL, so the reap after a timeout needs its own bound.
_REAP_TIMEOUT_SEC = 10

# Debug: reload even when the mode was rejected or already active.
# This allows the test to exercise reloading the driver without memory partition
# support.
_ALWAYS_RELOAD = os.environ.get("AMDSMI_TEST_ALWAYS_RELOAD") == "1"

_BANNER_WIDTH = 70

# name -> (enum, expected status). UNKNOWN is a reporting sentinel, not settable.
# Every NPS mode accepts [PASS, INVAL, NOT_SUPPORTED] from the shared table, since
# which modes a device supports is hardware-dependent.
_NPS_MODES = {
    name: (partition_type, expected_status)
    for name, partition_type, expected_status in common.MEMORY_PARTITION_TYPES
    if name != "UNKNOWN"
}


class AmdsmiSession:
    """Scoped amdsmi_init / amdsmi_shut_down pair yielding the handle list."""

    def __enter__(self):
        amdsmi.amdsmi_init()
        try:
            return amdsmi.amdsmi_get_processor_handles()
        except amdsmi.AmdSmiException:
            # __exit__ never runs when __enter__ raises, so unwind the init here.
            amdsmi.amdsmi_shut_down()
            raise

    def __exit__(self, *_):
        try:
            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiLibraryException:
            pass


class TestGpuMemoryPartitionCycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.geteuid() != 0:
            raise unittest.SkipTest("Memory partition tests need root/admin")
        cls.common = common.Common(common.verbose)
        # HSMP is a CPU-side status; a GPU memory-partition call reporting it is a
        # real bug, so drop it from the statuses check_ret silently tolerates.
        cls.common.not_supported_error_codes = [
            entry for entry in cls.common.not_supported_error_codes if "HSMP" not in entry[1]
        ]

    def setUp(self):
        self.raise_exception = None

    def tearDown(self):
        try:
            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiLibraryException:
            pass

    # -- output -------------------------------------------------------------

    def _separator(self, title):
        header = f"--- {title} "
        self.common.print(f"\n{header}{'-' * (_BANNER_WIDTH - len(header))}")

    # -- queries ------------------------------------------------------------

    def _physical_gpu_count(self):
        """Physical GPUs in the system. Caller must hold an open session.

        A socket is keyed on the BD of the BDF, so every logical partition of a
        GPU folds into one socket. The processor handle list does not: a reload
        resets the accelerator partition to the default for the new NPS mode, so
        the handle count legitimately changes (SPX -> CPX multiplies it).
        """
        return len(amdsmi.amdsmi_get_socket_handles())

    def _for_each_gpu(self, call_name, api, expected=None, params=""):
        """Run *api* on every handle in one session; return ``{index: result}``.

        Only handles that answered appear in the result. In a partitioned layout
        amdsmi enumerates one handle per logical partition, and the sub-partition
        handles report NOT_SUPPORTED for the memory partition APIs -- they drop
        out here rather than counting as a mismatch when modes are compared.
        """
        answered = {}
        with AmdsmiSession() as gpus:
            # Printed before the ### marker so the api_summary parser still sees
            # the status line as the one following it.
            self.common.print(
                f"  >> Device Count: {self._physical_gpu_count()} physical GPUs, "
                f"{len(gpus)} (processor) partition handles"
            )
            self.assertTrue(gpus, "No GPU handles returned after amdsmi_init()")
            for i, gpu in enumerate(gpus):
                msg = f"\t### {call_name}(gpu={i}{params}):"
                try:
                    result = api(gpu)
                    self.common.print(msg, result)
                    self.common.check_ret("", "", self.common.PASS)
                    answered[i] = result
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, expected or self.common.PASS):
                        self.raise_exception = e
        return answered

    def _read_modes(self):
        """``{index: NPS mode}`` for every handle that reports one."""
        return self._for_each_gpu(
            "amdsmi_get_gpu_memory_partition", amdsmi.amdsmi_get_gpu_memory_partition
        )

    @staticmethod
    def _agreed_mode(modes):
        """The mode every readable handle agrees on, or None when they differ."""
        distinct = set(modes.values())
        return distinct.pop() if len(distinct) == 1 else None

    def _read_caps(self):
        """Modes to attempt: the advertised caps, else every known NPS mode.

        Memory partition is hive-wide, so the first handle that answers speaks
        for the system. The caps read is advisory only -- a device that reports
        nothing still gets each mode attempted, since the set call is the real
        authority.
        """
        configs = self._for_each_gpu(
            "amdsmi_get_gpu_memory_partition_config", amdsmi.amdsmi_get_gpu_memory_partition_config
        )
        for config in configs.values():
            caps = [c for c in config.get("partition_caps", []) if c in _NPS_MODES]
            if caps:
                return caps
        return list(_NPS_MODES)

    @staticmethod
    def _profile_of(profile):
        """``(index, name)`` for a handle that owns an accelerator profile, else None.

        This getter reports NOT_SUPPORTED in-band as "N/A" instead of raising, so
        sub-partition handles have to be filtered on the value rather than dropping
        out of _for_each_gpu the way the memory partition getters do.
        """
        partition = profile["partition_profile"]
        index = partition["profile_index"]
        return (index, partition["profile_type"]) if isinstance(index, int) else None

    @staticmethod
    def _describe_profiles(profiles):
        """One-line summary, e.g. ``SPX (index 0) on 8 GPUs``."""
        if not profiles:
            return "none reported"
        distinct = sorted({f"{name} (index {index})" for index, name in profiles})
        return f"{', '.join(distinct)} on {len(profiles)} GPUs"

    def _read_accelerator_profiles(self):
        """``[(index, name), ...]``, one entry per physical GPU, in handle order."""
        profiles = self._for_each_gpu(
            "amdsmi_get_gpu_accelerator_partition_profile",
            amdsmi.amdsmi_get_gpu_accelerator_partition_profile,
        )
        return [p for p in map(self._profile_of, profiles.values()) if p is not None]

    # -- policy -------------------------------------------------------------

    @staticmethod
    def _reload_skip_reason(mode, staged, current):
        """Why a reload would change nothing, or None when one is needed."""
        if _ALWAYS_RELOAD:
            return None
        if not staged:
            return f"{mode} rejected; skipping driver reload"
        if mode == current:
            return f"{mode} already active; skipping driver reload"
        return None

    # -- mutations ----------------------------------------------------------

    def _stage_mode(self, mode):
        """Stage *mode* on every device. False when no device accepted it.

        A hardware-dependent rejection is an expected status rather than a test
        failure -- see _NPS_MODES.
        """
        partition_type, expected = _NPS_MODES[mode]
        accepted = self._for_each_gpu(
            "amdsmi_set_gpu_memory_partition_mode",
            lambda gpu: amdsmi.amdsmi_set_gpu_memory_partition_mode(gpu, partition_type),
            expected,
            f", memory_partition_type={mode}",
        )
        return bool(accepted)

    def _reload_driver(self):
        """Unload then reload amdgpu. shell=False, so there is no injection surface.

        Mandatory for a staged memory partition to take effect. Callers must hold no
        open session, and the baseline count below is read in a scoped one -- amdsmi
        may hold /dev/dri descriptors that would make ``modprobe -r`` fail with
        "module in use".
        """
        self._separator("Reload driver")
        with AmdsmiSession():
            expected = self._physical_gpu_count()
        self.common.print("  Reloading driver, this may take some time...")
        started = time.monotonic()
        for argv in (["modprobe", "-r", "amdgpu"], ["modprobe", "amdgpu"]):
            command = " ".join(argv)  # log text only; never handed to a shell
            self.common.print(f"  {command} (timeout {_MODPROBE_TIMEOUT_SEC}s)")
            leg_started = time.monotonic()
            proc = subprocess.Popen(argv)
            try:
                returncode = proc.wait(timeout=_MODPROBE_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=_REAP_TIMEOUT_SEC)
                except subprocess.TimeoutExpired:
                    self.fail(
                        f"{command} timed out after {_MODPROBE_TIMEOUT_SEC}s and ignored "
                        f"SIGKILL: the module load is stuck inside the driver and cannot be "
                        f"cancelled, so the node needs a reboot. See dmesg and "
                        f"/sys/module/amdgpu/initstate."
                    )
                self.fail(f"{command} timed out after {_MODPROBE_TIMEOUT_SEC}s")
            if returncode != 0:
                self.fail(f"{command} exited {returncode}; check dmesg")
            self.common.print(f"    took {time.monotonic() - leg_started:.1f}s")
        probe_started = time.monotonic()
        self._wait_for_devices(expected)
        self.common.print(
            f"  {expected} physical GPUs re-enumerated in {time.monotonic() - probe_started:.1f}s"
        )
        self.common.print(f"  Driver reloaded in {time.monotonic() - started:.1f}s total")

    def _wait_for_devices(self, expected):
        """Block until *expected* physical GPUs (sockets) re-enumerate after a reload.

        modprobe returns once the module is inserted; the devices then probe
        asynchronously and one at a time, so poll rather than assume a fixed
        settle time. We used socket count as a check, since the partitioned
        devices can fluctuate based on the NPS mode.
        """
        deadline = time.monotonic() + _DEVICE_READY_TIMEOUT_SEC
        found = 0
        while True:
            time.sleep(_DEVICE_POLL_SEC)
            try:
                with AmdsmiSession():
                    found = self._physical_gpu_count()
                    if found >= expected:
                        return
            except amdsmi.AmdSmiException:
                pass
            if time.monotonic() >= deadline:
                self.fail(
                    f"Only {found} of {expected} physical GPUs enumerated within "
                    f"{_DEVICE_READY_TIMEOUT_SEC}s of reloading amdgpu"
                )

    def _restore_memory_partition(self, mode):
        """Put the device back on the memory partition mode the run started with."""
        # Runs as cleanup, after the body raised its own; reset so the check below
        # reports only statuses seen while restoring.
        self.raise_exception = None
        self._separator(f"Restore memory partition ({mode})")
        if not self.common.check_amdgpu_driver():
            self.common.print(f"  amdgpu is not loaded; cannot restore {mode} -- node left as-is")
            return
        if self._agreed_mode(self._read_modes()) == mode and not _ALWAYS_RELOAD:
            self.common.print(f"  already on {mode}; nothing to restore")
        else:
            if not self._stage_mode(mode):
                self.fail(f"could not stage restore back to {mode}")
            self._reload_driver()
            self._assert_all_on(self._read_modes(), mode, "restore failed")
        if self.raise_exception:
            raise self.raise_exception

    def _restore_accelerator_profiles(self, profiles):
        """Re-apply the accelerator profiles captured before the cycle.

        A memory partition reload resets the accelerator profile to the new mode's
        default, and profile_index only means anything against that mode's config
        table -- so this runs after the memory partition is back on the original.
        """
        if not profiles:
            return
        self.raise_exception = None
        self._separator("Restore accelerator partition")
        if not self.common.check_amdgpu_driver():
            self.common.print("  amdgpu is not loaded; cannot restore the accelerator profile")
            return
        self.common.print(f"  Restoring : {self._describe_profiles(profiles)}")
        wanted = iter(profiles)
        with AmdsmiSession() as gpus:
            for i, gpu in enumerate(gpus):
                current = amdsmi.amdsmi_get_gpu_accelerator_partition_profile(gpu)
                if self._profile_of(current) is None:
                    continue
                target = next(wanted, None)
                if target is None:
                    break
                index, name = target
                msg = (
                    "\t### amdsmi_set_gpu_accelerator_partition_profile("
                    f"gpu={i}, profile={name}, index={index}):"
                )
                try:
                    result = amdsmi.amdsmi_set_gpu_accelerator_partition_profile(gpu, index)
                    self.common.print(msg, result)
                    self.common.check_ret("", "", self.common.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, self.common.PASS):
                        self.raise_exception = e
        if self.raise_exception:
            raise self.raise_exception

    def _assert_all_on(self, modes, mode, context):
        """Fail unless every handle that reported a mode reports *mode*."""
        self.assertTrue(modes, f"{context}: no device reported a memory partition mode")
        mismatched = {i: m for i, m in modes.items() if m != mode}
        self.assertFalse(
            mismatched,
            f"{context}: expected all {len(modes)} readable devices on {mode}, "
            f"but {len(mismatched)} differ: {mismatched}",
        )

    # -- tests ---------------------------------------------------------------

    def test_applying_memory_partition_needs_reload(self):
        """Staging a mode must not change the active mode until the driver reloads.

        Documented in docs/conceptual/partition.md: the set call alone does not apply
        the change, and the system keeps the old configuration until the next driver
        load. Nothing here reloads, so nothing should change.
        """
        self.common.print_func_name("")
        self._separator("Set without reload")

        before = self._read_modes()
        original = self._agreed_mode(before)
        # A staged mode stays pending until the next driver load, re-staging the original
        # clears it. Registered before any set so a failure below cannot leave it armed.
        if original in _NPS_MODES:
            self.addCleanup(self._stage_mode, original)

        for mode in _NPS_MODES:
            with self.subTest(mode=mode):
                staged = self._stage_mode(mode)
                self.assertEqual(
                    self._read_modes(),
                    before,
                    f"{mode} was staged without a driver reload; the reported mode must not change",
                )
                if staged:
                    self.common.print(f"  [PASS] {mode} staged; reported mode unchanged")
                else:
                    self.common.print(f"  [INFO] {mode} was rejected; reported mode unchanged")

        if self.raise_exception:
            raise self.raise_exception

    def test_cycle_memory_partition_modes(self):
        self.common.print_func_name("")
        if not shutil.which("modprobe"):
            self.skipTest("Cycling memory partition modes needs modprobe (install kmod)")

        # -- Phase 1: what the device reports today -------------------------
        self._separator("Current memory partition")
        modes = self._read_modes()
        original = self._agreed_mode(modes)
        caps = self._read_caps()
        accelerator = self._read_accelerator_profiles()

        # Every advertised mode in order, then the starting mode once more so the
        # run ends where it began. The leading pass over the starting mode is
        # already active, so it stages without a reload.
        candidates = list(caps)
        if original in caps:
            candidates.append(original)

        self.common.print(f"  Current mode : {original} (on {len(modes)} sockets)")
        self.common.print(f"  Accelerator  : {self._describe_profiles(accelerator)}")
        self.common.print(f"  Attempting   : {', '.join(candidates)}")
        if original is None and modes:
            self.common.print(f"  Devices disagree on the current mode: {modes}")

        if original in _NPS_MODES:
            # Cleanups run LIFO, so registering the accelerator profile first runs
            # it last -- the memory reload resets it to the new mode's default.
            self.addCleanup(self._restore_accelerator_profiles, accelerator)
            # A no-op when nothing changed, so a device that rejects every mode
            # is never reloaded on the way out.
            self.addCleanup(self._restore_memory_partition, original)
        elif original is not None:
            self.common.print(f"  {original!r} is not a settable mode; it cannot be restored")

        # -- Phase 2: stage / reload / verify, once per mode ----------------
        current = original
        applied = []
        for mode in candidates:
            self._separator(f"Set memory partition -> {mode}")
            staged = self._stage_mode(mode)

            skipped = self._reload_skip_reason(mode, staged, current)
            if skipped:
                self.common.print(f"  {skipped}")
                continue

            # Outside the subTest below: subTest swallows failures, and continuing
            # the cycle after a broken reload only buries the real cause.
            self._reload_driver()
            modes = self._read_modes()
            current = self._agreed_mode(modes)

            with self.subTest(mode=mode):
                if not staged:
                    self.common.print(f"  [INFO] {mode} was rejected; active mode is {current}")
                else:
                    self._assert_all_on(modes, mode, f"after reload to {mode}")
                    applied.append(mode)
                    self.common.print(
                        f"  [PASS] {mode} applied and verified on {len(modes)} devices"
                    )

        # -- Phase 3: leave the device as the run found it ------------------
        self._separator("Summary")
        if applied:
            self.common.print(f"  Modes applied : {', '.join(applied)}")
        else:
            self.common.print("  No mode was settable; validated the reported status instead")

        if self.raise_exception:
            raise self.raise_exception
