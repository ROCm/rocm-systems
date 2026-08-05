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
"""KFD process-to-GPU attribution unit tests (hardware-free)."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest


def _resolve_process_gpus(proc_dir):
    """Resolve the KFD GPU ids a process runs on from its sysfs proc dir.

    Executable spec mirroring GetProcessGPUs() in rocm_smi/src/rocm_smi_kfd.cc
    (non-container path). A process is attributed to a GPU when it has an active
    queue on it (queues/<qid>/gpuid) or a non-zero VRAM allocation on it
    (vram_<gpuid>). The stats_/counters_/sdma_<gpuid> files are deliberately not
    consulted: opening /dev/kfd creates them (as zero) for every GPU, so using
    them would attribute an idle process to the whole topology.
    """
    gpu_ids = set()
    queues_dir = os.path.join(proc_dir, "queues")
    if os.path.isdir(queues_dir):
        for qid in os.listdir(queues_dir):
            gpuid_file = os.path.join(queues_dir, qid, "gpuid")
            try:
                with open(gpuid_file) as f:
                    gpu_ids.add(int(f.read().strip()))
            except (OSError, ValueError):
                continue

    for name in os.listdir(proc_dir):
        if not name.startswith("vram_"):
            continue
        suffix = name[len("vram_") :]
        if not suffix.isdigit():
            continue
        try:
            with open(os.path.join(proc_dir, name)) as f:
                if int(f.read().strip() or "0") > 0:
                    gpu_ids.add(int(suffix))
        except (OSError, ValueError):
            continue
    return gpu_ids


class TestKfdProcessGpuResolution(unittest.TestCase):
    """Regression coverage for processes being reported on every GPU instead of
    only the GPUs they actually use (active queue or VRAM allocation).

    This pins the queue/VRAM attribution algorithm as an executable spec. The
    other-user visibility fix (IsKfdPidNamespaced skipping inaccessible PIDs)
    lives in a hardcoded-/proc-path C++ path that is validated on hardware
    (see PR test plan), not reproducible in this hardware-free harness.
    """

    ALL_GPUS = (56179, 9367, 45514, 39772)

    def _make_proc_dir(self, queue_gpus, vram_gpus=None):
        """Build a fake KFD proc dir: per-node files on ALL_GPUS (VRAM 0 unless
        listed in vram_gpus), queues on the subset in queue_gpus."""
        vram_gpus = vram_gpus or {}
        proc_dir = tempfile.mkdtemp(prefix="kfd_proc_")
        self.addCleanup(shutil.rmtree, proc_dir, ignore_errors=True)
        for gid in self.ALL_GPUS:
            for prefix in ("stats_", "counters_", "sdma_", "ais_"):
                open(os.path.join(proc_dir, f"{prefix}{gid}"), "w").close()
            with open(os.path.join(proc_dir, f"vram_{gid}"), "w") as f:
                f.write(str(vram_gpus.get(gid, 0)))
        for idx, gid in enumerate(queue_gpus):
            qdir = os.path.join(proc_dir, "queues", str(idx))
            os.makedirs(qdir)
            with open(os.path.join(qdir, "gpuid"), "w") as f:
                f.write(str(gid))
        return proc_dir

    def test_single_queue_gpu_not_reported_on_all(self):
        # Queues on one GPU, per-node files on all four -> only the queue GPU.
        proc_dir = self._make_proc_dir(queue_gpus=[45514])
        self.assertEqual(_resolve_process_gpus(proc_dir), {45514})

    def test_multiple_queue_gpus(self):
        # Queues on two GPUs -> exactly those two, not all four.
        proc_dir = self._make_proc_dir(queue_gpus=[45514, 9367])
        self.assertEqual(_resolve_process_gpus(proc_dir), {45514, 9367})

    def test_no_queues_no_vram_reports_no_gpus(self):
        # No queues and all VRAM zero -> no GPUs, matching rocm-smi --showpids
        # (0). Per-node context files must not inflate this to all GPUs.
        proc_dir = self._make_proc_dir(queue_gpus=[])
        self.assertEqual(_resolve_process_gpus(proc_dir), set())

    def test_vram_allocation_without_queue_is_attributed(self):
        # A GPU with a VRAM allocation but no queue is still the process's GPU.
        proc_dir = self._make_proc_dir(queue_gpus=[], vram_gpus={56179: 271622705152})
        self.assertEqual(_resolve_process_gpus(proc_dir), {56179})

    def test_queue_and_vram_union(self):
        # Queue on one GPU, VRAM on another -> both, not all four.
        proc_dir = self._make_proc_dir(queue_gpus=[45514], vram_gpus={9367: 1024})
        self.assertEqual(_resolve_process_gpus(proc_dir), {45514, 9367})

    def test_unreadable_vram_file_is_skipped(self):
        # A vram_<gpuid> entry that cannot be read must be skipped rather than
        # attributed, mirroring the C++ `ReadSysfsStr(...) != 0` continue. A
        # directory in place of the file makes open() fail on any uid (root
        # included), unlike chmod which root would bypass.
        proc_dir = self._make_proc_dir(queue_gpus=[])
        target = os.path.join(proc_dir, f"vram_{self.ALL_GPUS[0]}")
        os.remove(target)
        os.makedirs(target)
        self.assertEqual(_resolve_process_gpus(proc_dir), set())


if __name__ == "__main__":
    unittest.main()
