#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""Unit tests for test_executor._distinct_host_count().

The runner uses this count to decide whether a multi-node test can run: a test
declaring num_nodes=N is SKIPPED when fewer than N distinct hosts are available.

The regression these guard: an empty mpi_hosts dict (no hostfile, no SLURM
allocation) used to report 0, which the caller reads as "topology unknown" and
so disables the check entirely. On a single-node box that launched every
declared multi-node suite oversubscribed onto the local host, where RCCL
rejects the duplicate GPU assignment ("Multiple Ranks are using the same
GPU/Partition") and the suites reported FAILED instead of SKIPPED. With no host
source, mpirun places all ranks locally, so the correct answer is 1.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from lib.test_executor import _distinct_host_count


class TestDistinctHostCount(unittest.TestCase):
    def test_no_host_source_is_one_local_host(self):
        """No hostfile and no SLURM allocation means mpirun runs on this host only."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_distinct_host_count({}), 1)

    def test_no_host_source_skips_multi_node(self):
        """Caller skips only when 0 < avail < num_nodes. Zero means unknown and does not skip."""
        with patch.dict(os.environ, {}, clear=True):
            avail = _distinct_host_count({})
        num_nodes = 2
        self.assertTrue(avail > 0 and avail < num_nodes)

    def test_other_allocators_without_hosts_stay_unknown(self):
        """PBS/LSF/Flux/Cobalt with no parsed host list must not look like one local host."""
        for key in ("PBS_JOBID", "LSB_JOBID", "FLUX_JOB_ID", "COBALT_JOBID"):
            with self.subTest(key=key):
                with patch.dict(os.environ, {key: "99"}, clear=True):
                    self.assertEqual(_distinct_host_count({}), 0)

    def test_slurm_allocation_without_detected_hosts_is_unknown(self):
        """Do not collapse an active allocation to the local host."""
        with patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}, clear=True):
            self.assertEqual(_distinct_host_count({}), 0)

    def test_slurm_host_list_counts_distinct_entries(self):
        self.assertEqual(_distinct_host_count({"host_list": "node-a,node-b,node-c"}), 3)

    def test_slurm_host_list_ignores_slot_suffix_and_duplicates(self):
        hosts = {"host_list": "node-a:8, node-b:8 ,node-a:8"}
        self.assertEqual(_distinct_host_count(hosts), 2)

    def test_slurm_host_list_ignores_empty_entries(self):
        self.assertEqual(_distinct_host_count({"host_list": "node-a,,node-b,"}), 2)

    def test_hostfile_counts_distinct_hosts(self):
        with tempfile.NamedTemporaryFile("w", suffix=".hostfile", delete=False) as hf:
            hf.write("node-a slots=8\n")
            hf.write("# a comment\n")
            hf.write("\n")
            hf.write("node-b slots=8\n")
            hf.write("node-a slots=8\n")
            path = hf.name
        try:
            self.assertEqual(_distinct_host_count({"hostfile": path}), 2)
        finally:
            os.unlink(path)

    def test_unreadable_hostfile_is_unknown(self):
        """A declared-but-unreadable host source stays 0 so the check is skipped."""
        missing = os.path.join(tempfile.gettempdir(), "rccl-no-such-hostfile")
        self.assertFalse(os.path.exists(missing))
        self.assertEqual(_distinct_host_count({"hostfile": missing}), 0)


if __name__ == "__main__":
    unittest.main()
