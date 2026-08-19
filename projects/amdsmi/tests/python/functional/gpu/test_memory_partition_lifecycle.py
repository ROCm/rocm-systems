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
"""GPU memory partition: set -> driver reload -> verify, across every mode.

Requirements
------------
- Root privileges  (modprobe requires root)
- kmod installed   (modprobe must be on PATH)
- No GPU workloads running during the test

Usage
-----
  1. Install amd-smi-lib & amd-smi-lib-test packages
  2. sudo /opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -k "test_cycles_every_supported_memory_partition_mode" -v
"""

import os
import shutil
import subprocess
import time
import unittest


import common.common as common
from common.common import amdsmi

_MODPROBE_TIMEOUT_SEC = 120
_DEVICE_POLL_SEC = 2
_DEVICE_READY_TIMEOUT_SEC = 60

# name -> (enum, expected status). UNKNOWN is a reporting sentinel, not settable.
# The expected status comes from the shared table: NPS1/NPS2 are PASS, NPS4/NPS8
# are [PASS, FAIL] since they are hardware-dependent. check_ret additionally
# treats NOT_SUPPORTED / NOT_YET_IMPLEMENTED as acceptable on any call.
_NPS_MODES = {
    name: (partition_type, expected_status)
    for name, partition_type, expected_status in common.MEMORY_PARTITION_TYPES
    if name != "UNKNOWN"
}


class TestGpuMemoryPartitionLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.geteuid() != 0:
            raise unittest.SkipTest("memory partition lifecycle needs root (modprobe)")
        if not shutil.which("modprobe"):
            raise unittest.SkipTest("memory partition lifecycle needs modprobe (install kmod)")
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

    # -- helpers ------------------------------------------------------------

    def _primary_gpu(self):
        """Init amdsmi and return the first GPU handle.

        Each reload invalidates outstanding handles, so callers re-acquire after
        every reload rather than caching one across the cycle.
        """
        amdsmi.amdsmi_init()
        gpus = amdsmi.amdsmi_get_processor_handles()
        self.assertTrue(gpus, "no GPU handles returned after amdsmi_init()")
        return gpus[0]

    def _read_mode(self):
        """Current mode, or None when the device will not report one."""
        msg = "\t### amdsmi_get_gpu_memory_partition(gpu=0):"
        gpu = self._primary_gpu()
        try:
            mode = amdsmi.amdsmi_get_gpu_memory_partition(gpu)
            self.common.print(msg, mode)
            self.common.check_ret("", "", self.common.PASS)
            return mode
        except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
            if self.common.check_ret(msg, e, self.common.PASS):
                self.raise_exception = e
            return None
        finally:
            amdsmi.amdsmi_shut_down()

    def _candidate_modes(self):
        """Modes to attempt: the advertised caps, else every known NPS mode.

        The caps read is advisory only -- a device that reports nothing still
        gets each mode attempted, since the set call is the real authority.
        """
        msg = "\t### amdsmi_get_gpu_memory_partition_config(gpu=0):"
        gpu = self._primary_gpu()
        try:
            config = amdsmi.amdsmi_get_gpu_memory_partition_config(gpu)
            self.common.print(msg, config)
            self.common.check_ret("", "", self.common.PASS)
            caps = [c for c in config.get("partition_caps", []) if c in _NPS_MODES]
        except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
            if self.common.check_ret(msg, e, self.common.PASS):
                self.raise_exception = e
            caps = []
        finally:
            amdsmi.amdsmi_shut_down()
        return caps or list(_NPS_MODES)

    def _reload_driver(self):
        """Unload then reload amdgpu. shell=False, so there is no injection surface.

        A partition change reshapes device topology, so the library must be torn
        down around it: amdsmi holds /dev/dri descriptors that would make
        ``modprobe -r`` fail with "module in use", and every handle taken before
        the reload is stale afterwards. Callers re-init via ``_primary_gpu``.
        """
        try:
            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiLibraryException:
            pass
        for argv in (["modprobe", "-r", "amdgpu"], ["modprobe", "amdgpu"]):
            self.common.print(f"\tReloading driver: {' '.join(argv)}")
            try:
                subprocess.run(argv, check=True, timeout=_MODPROBE_TIMEOUT_SEC)
            except subprocess.CalledProcessError as e:
                self.fail(f"{' '.join(argv)} exited {e.returncode}; check dmesg")
            except subprocess.TimeoutExpired:
                self.fail(f"{' '.join(argv)} timed out after {_MODPROBE_TIMEOUT_SEC}s")
        self.common.print("\tSuccessfully reloaded driver!")
        self._wait_for_devices()

    def _wait_for_devices(self):
        """Block until the GPUs re-enumerate after a reload.

        Devices come back asynchronously and the new partition changes topology,
        so poll rather than assume a fixed settle time -- otherwise a slow
        re-enumeration reads as a failed partition change.
        """
        deadline = time.monotonic() + _DEVICE_READY_TIMEOUT_SEC
        while True:
            time.sleep(_DEVICE_POLL_SEC)
            try:
                amdsmi.amdsmi_init()
                try:
                    if amdsmi.amdsmi_get_processor_handles():
                        return
                finally:
                    amdsmi.amdsmi_shut_down()
            except amdsmi.AmdSmiLibraryException:
                pass
            if time.monotonic() >= deadline:
                self.fail(
                    f"no GPUs enumerated within {_DEVICE_READY_TIMEOUT_SEC}s of reloading amdgpu"
                )

    def _try_stage(self, mode):
        """Stage *mode*. False when the device rejects it -- caller must not reload.

        Rejections are judged against the shared expected-status table, so a
        hardware-dependent NPS4/NPS8 INVAL passes while an undocumented status
        is recorded for the end of the run.
        """
        partition_type, expected = _NPS_MODES[mode]
        msg = f"\t### amdsmi_set_gpu_memory_partition_mode(gpu=0, memory_partition_type={mode}):"
        gpu = self._primary_gpu()
        try:
            ret = amdsmi.amdsmi_set_gpu_memory_partition_mode(gpu, partition_type)
            self.common.print(msg, ret)
            self.common.check_ret("", "", self.common.PASS)
            return True
        except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
            if self.common.check_ret(msg, e, expected):
                self.raise_exception = e
            return False
        finally:
            amdsmi.amdsmi_shut_down()

    def _restore(self, mode):
        if self._read_mode() == mode:
            return
        self.common.print(f"\trestoring {mode}")
        if not self._try_stage(mode):
            self.fail(f"could not stage restore back to {mode}")
        self._reload_driver()
        restored = self._read_mode()
        self.assertEqual(restored, mode, f"restore failed: expected {mode}, got {restored}")

    # -- test ---------------------------------------------------------------

    def test_cycles_every_supported_memory_partition_mode(self):
        self.common.print_func_name("")

        original = self._read_mode()
        candidates = self._candidate_modes()
        self.common.print(f"\tcurrent mode: {original}")
        self.common.print(f"\tattempting: {candidates}")

        # Finish on the starting mode when it is known, so the device is left as
        # the run found it.
        if original in candidates:
            start = candidates.index(original)
            candidates = candidates[start + 1 :] + candidates[: start + 1]
        elif original is not None and original not in _NPS_MODES:
            self.common.print(f"\tcurrent mode {original!r} is not settable; cannot restore it")

        current = original
        applied = []
        for mode in candidates:
            with self.subTest(mode=mode):
                if not self._try_stage(mode):
                    continue
                if mode == current:
                    # Staging the mode already in effect changes nothing, so there
                    # is nothing for a reload to apply.
                    self.common.print(f"\t{mode} already active; skipping reload")
                    continue
                if not applied and original in _NPS_MODES:
                    # Registered only once a change is real, so a device that
                    # rejects everything is never reloaded on the way out.
                    self.addCleanup(self._restore, original)
                applied.append(mode)
                self._reload_driver()
                active = self._read_mode()
                self.assertEqual(active, mode, f"expected {mode} after reload, got {active}")
                current = active
                self.common.print(f"\t[PASS] {mode} applied and verified")

        if not applied:
            self.common.print("\tno mode was settable; validated the reported status instead")

        if self.raise_exception:
            raise self.raise_exception
