#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Unit tests for when hrr_torch_regions.py turns on PyTorch's allocator
history: only when asked, and with a bounded ring. Each case imports the
producer in a fresh interpreter, beside a stand-in torch module that records
the call and an archive that reads as active capture, so the only thing left
to decide whether history starts is the producer's own environment. Run with
pytest or python -m unittest; torch is not needed."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PRODUCER_DIR = Path(__file__).resolve().parent.parent

# Runs in the child. The archive is pid-<pid> with an active marker in it, as
# the capture writer leaves it while it runs, so a started producer goes on to
# enable history straight away.
CHILD = r"""
import json, os, sys, threading, time, types

calls = []
memory = types.SimpleNamespace(
    _record_memory_history=lambda **kw: calls.append(kw),
    _snapshot=lambda: {})
torch = types.ModuleType("torch")
torch.cuda = types.SimpleNamespace(memory=memory)
sys.modules["torch"] = torch

archive = os.path.join(os.environ["HIP_HRR_CAPTURE_OUTPUT"], "pid-%d" % os.getpid())
os.mkdir(archive, 0o700)
fd = os.open(os.path.join(archive, "active"), os.O_CREAT | os.O_WRONLY, 0o600)
os.close(fd)

sys.path.insert(0, sys.argv[1])
import hrr_torch_regions

deadline = time.monotonic() + float(sys.argv[2])
while not calls and time.monotonic() < deadline:
    time.sleep(0.05)
print(json.dumps({
    "calls": calls,
    "thread": any(t.name == "hrr-regions" for t in threading.enumerate()),
}))
"""


class AutostartTest(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="hrr_regions_autostart_"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def import_producer(self, env, wait_s):
        """Import the producer in a child with only `env` set among the
        HRR_REGIONS_* variables; return what it did within `wait_s`."""
        child_env = {
            k: v for k, v in os.environ.items() if not k.startswith("HRR_REGIONS_")
        }
        child_env["HIP_HRR_CAPTURE_OUTPUT"] = str(self.root)
        child_env.update(env)
        proc = subprocess.run(
            [sys.executable, "-c", CHILD, str(PRODUCER_DIR), str(wait_s)],
            env=child_env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_import_alone_records_nothing(self):
        # A started producer enables history within milliseconds here, so a
        # second without it is a clear negative.
        got = self.import_producer({}, wait_s=1.0)
        self.assertFalse(got["thread"])
        self.assertEqual(got["calls"], [])

    def test_autostart_zero_records_nothing(self):
        got = self.import_producer({"HRR_REGIONS_AUTOSTART": "0"}, wait_s=1.0)
        self.assertFalse(got["thread"])
        self.assertEqual(got["calls"], [])

    def test_autostart_enables_bounded_history(self):
        got = self.import_producer({"HRR_REGIONS_AUTOSTART": "1"}, wait_s=20.0)
        self.assertTrue(got["thread"])
        self.assertEqual(len(got["calls"]), 1)
        self.assertEqual(got["calls"][0]["max_entries"], 100000)

    def test_max_entries_override(self):
        got = self.import_producer(
            {"HRR_REGIONS_AUTOSTART": "1", "HRR_REGIONS_MAX_ENTRIES": "5000"},
            wait_s=20.0,
        )
        self.assertEqual([c["max_entries"] for c in got["calls"]], [5000])


if __name__ == "__main__":
    unittest.main()
