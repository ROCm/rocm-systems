#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""Guards for the 2-node mixed-backend suite in gin_rocshmem_backends_mellanox.json.

GinMPIDeviceTests.MixedBackends_TwoNodeUnsetType skips unless NCCL_GIN_TYPE is
unset (an empty value also disables Anvil SDMA) and the run is 2 ranks on 2
hosts. A skip still reports green, so lock the launch shape here (AICOMRCCL-3450).
"""

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


if __name__ == "__main__":
    unittest.main()
