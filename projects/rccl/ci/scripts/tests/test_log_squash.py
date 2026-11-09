"""Tests for squash_log_spam() in test_madengine.py.

Run from projects/rccl/ci/scripts:  python3 -m unittest discover -s tests
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
import test_madengine as tm

SPAM = (b"[2026-10-12 21:11:21] node:985:985 [0] api_trace.cc:569 NCCL WARN Please note "
        b"that ncclAllToAllv is deprecated, please use ncclAlltoAllv instead\n")
SPAM2 = SPAM.replace(b"ncclAllToAllv", b"ncclAllToAll").replace(b"ncclAlltoAllv", b"ncclAlltoAll")
BANNER = b"RCCL version : 2.32.3-develop:2b88119\n"
REAL_WARN = b"node:985:985 [0] NCCL WARN NET/IB : Got async event : port error\n"


class SquashLogSpam(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)
        self.logs = self.run_dir / "slurm_output"
        (self.logs / "model" / "node_0").mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_keeps_first_and_counts_the_rest(self):
        log = self.logs / "madengine-x_node_0.out"
        log.write_bytes(BANNER + SPAM * 5 + REAL_WARN + SPAM * 3 + b"tok/s 1500\n")
        saved = tm.squash_log_spam(self.run_dir)
        self.assertEqual(
            log.read_bytes(),
            BANNER + SPAM + REAL_WARN + b"tok/s 1500\n"
            + b"[rccl-ci] dropped 7 repeats of 'NCCL WARN Please note that "
              b"ncclAllToAllv is deprecated'\n",
        )
        self.assertEqual(saved, 7 * len(SPAM) - len(log.read_bytes().splitlines(True)[-1]))
        self.assertEqual(tm.runtime_rccl_versions(self.run_dir), ["2.32.3-develop:2b88119"])

    def test_each_api_counted_separately(self):
        log = self.logs / "model" / "node_0" / "stdout.out"
        log.write_bytes(SPAM + SPAM2 + SPAM + SPAM2 + SPAM2)
        tm.squash_log_spam(self.run_dir)
        self.assertEqual(
            log.read_bytes(),
            SPAM + SPAM2
            + b"[rccl-ci] dropped 1 repeats of 'NCCL WARN Please note that "
              b"ncclAllToAllv is deprecated'\n"
            + b"[rccl-ci] dropped 2 repeats of 'NCCL WARN Please note that "
              b"ncclAllToAll is deprecated'\n",
        )

    def test_file_without_repeats_is_untouched(self):
        log = self.logs / "madengine-x_node_1.out"
        body = BANNER + SPAM + REAL_WARN
        log.write_bytes(body)
        mtime = log.stat().st_mtime_ns
        self.assertEqual(tm.squash_log_spam(self.run_dir), 0)
        self.assertEqual(log.read_bytes(), body)
        self.assertEqual(log.stat().st_mtime_ns, mtime)
        self.assertEqual(sorted(p.name for p in self.logs.iterdir() if p.is_file()),
                         ["madengine-x_node_1.out"])

    def test_missing_slurm_output(self):
        self.assertEqual(tm.squash_log_spam(self.run_dir / "nope"), 0)


if __name__ == "__main__":
    unittest.main()
