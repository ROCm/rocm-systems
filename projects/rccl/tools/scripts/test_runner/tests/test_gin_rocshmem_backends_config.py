#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""Lock the 2-node unset-type suite's launch shape; a skip still reports green (AICOMRCCL-3450)."""

import unittest
from pathlib import Path

from lib.test_config import TestConfigProcessor

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "gin_rocshmem_backends_mellanox.json"
_SUITE = "gin_devmpi_mixed_2node"


class MixedBackendsTwoNodeSuiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proc = TestConfigProcessor(str(_CONFIG))
        cls.suites = [
            s for s in cls.proc.parse_test_suites() if s["suite_details"]["name"].startswith(
                "GIN Device MPI Tests - mixed backends (2 nodes")
        ]

    def test_suite_is_enabled_once(self):
        self.assertEqual(len(self.suites), 1)
        self.assertTrue(self.suites[0]["suite_details"]["enabled"])

    def test_gin_type_is_unset_at_every_level(self):
        suite = self.suites[0]
        self.assertNotIn("NCCL_GIN_TYPE", self.proc.get_env_variables())
        self.assertNotIn("NCCL_GIN_TYPE", suite.get("env_variables", {}))
        for test in suite["tests"]:
            self.assertNotIn("NCCL_GIN_TYPE", test.get("env_variables", {}), test["name"])

    def test_runs_only_the_two_host_case(self):
        tests = self.suites[0]["tests"]
        self.assertEqual([t["test_filter"] for t in tests],
                         ["GinMPIDeviceTests.MixedBackends_TwoNodeUnsetType"])
        test = tests[0]
        self.assertEqual((test["num_ranks"], test["num_nodes"], test["num_gpus"]), (2, 2, 1))
        self.assertEqual(test["binary"], "rccl-UnitTestsMPI")

    def test_build_enables_rocshmem_gin(self):
        self.assertIn("--rocshmem-gin", self.proc.get_build_config()["install_flags"])


class GinFinalizeMicroSuiteTest(unittest.TestCase):
    """Pin where gin_micro_finalize looks for its binary.

    It is the one suite in any config that names a binary from test/host/, whose layout this
    file cannot see: when the microtests moved down into micro/, the path here went stale and
    the suite reported "Binary not found" -- a hard failure, since it sets no skip_if_missing.
    Asserting the directory, rather than that the file exists, keeps this runnable without a
    build while still failing if that tree is rearranged again.

    The suite reaches across because this build arm (ENABLE_ROCSHMEM_GIN) has no binary of its
    own; when it gets one, this goes with it.
    """

    def setUp(self):
        self.proc = TestConfigProcessor(str(_CONFIG))
        self.config = self.proc.config["test_configurations"]["gin_micro_finalize"]

    def test_binary_is_in_the_host_trees_micro_directory(self):
        # test/host/CMakeLists.txt: RCCL_MICRO_SUBDIR, relative to the runner's build/test/.
        self.assertEqual(self.config["binary"], "host/micro/rccl-UnitTestsMicro")

    def test_a_missing_binary_would_fail_the_run(self):
        # The assertion above is the whole guard: nothing downstream would soften a wrong path.
        self.assertFalse(self.config.get("skip_if_missing", False))


if __name__ == "__main__":
    unittest.main()
