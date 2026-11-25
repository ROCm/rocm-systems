#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""Lock the launch shape of the MI455 pod suites; a skip still reports green (AICOMRCCL-3444)."""

import unittest
from pathlib import Path

from lib.test_config import TestConfigProcessor

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "gin_rocshmem_backends_mi455.json"
_FILTER = "GinMPIDeviceTests.MixedBackends_TwoNodeUnsetType"
_OPT_INS = ("RCCL_GIN_ANVIL_FABRIC_POD", "RCCL_GIN_ANVIL_FABRIC_MEM")


class Mi455PodSuitesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proc = TestConfigProcessor(str(_CONFIG))
        suites = {s["suite_details"]["name"]: s for s in cls.proc.parse_test_suites()}
        cls.no_opt_in = suites["GIN Device MPI Tests - MI450 pod, no opt-ins (2 nodes, type unset)"]
        cls.opt_in = suites["GIN Device MPI Tests - MI450 pod, both opt-ins (2 nodes, type unset)"]

    def env(self, suite, test):
        return {**self.proc.get_env_variables(), **suite.get("env_variables", {}), **test.get("env_variables", {})}

    def test_suites_are_enabled(self):
        for suite in (self.no_opt_in, self.opt_in):
            self.assertTrue(suite["suite_details"]["enabled"], suite["suite_details"]["name"])

    def test_run_only_the_two_host_case(self):
        for suite in (self.no_opt_in, self.opt_in):
            tests = suite["tests"]
            self.assertEqual([t["test_filter"] for t in tests], [_FILTER])
            self.assertEqual((tests[0]["num_ranks"], tests[0]["num_nodes"], tests[0]["num_gpus"]), (2, 2, 1))
            self.assertEqual(tests[0]["binary"], "rccl-UnitTestsMPI")

    def test_gin_type_is_unset_at_every_level(self):
        for suite in (self.no_opt_in, self.opt_in):
            for test in suite["tests"]:
                self.assertNotIn("NCCL_GIN_TYPE", self.env(suite, test), test["name"])

    def test_pod_prerequisites_are_on(self):
        for suite in (self.no_opt_in, self.opt_in):
            for test in suite["tests"]:
                env = self.env(suite, test)
                self.assertEqual(env.get("NCCL_MNNVL_ENABLE"), "1", test["name"])
                self.assertEqual(env.get("NCCL_CUMEM_ENABLE"), "1", test["name"])
                self.assertEqual(env.get("NCCL_GIN_ENABLE"), "1", test["name"])

    def test_opt_ins_split_the_suites(self):
        for test in self.no_opt_in["tests"]:
            env = self.env(self.no_opt_in, test)
            for name in _OPT_INS:
                self.assertNotIn(name, env, test["name"])
        for test in self.opt_in["tests"]:
            env = self.env(self.opt_in, test)
            for name in _OPT_INS:
                self.assertEqual(env.get(name), "1", f"{test['name']}: {name}")

    def test_build_enables_rocshmem_gin_and_mpi_tests(self):
        flags = self.proc.get_build_config()["install_flags"]
        for flag in ("--rocshmem-gin", "--enable-mpi-tests", "--debug"):
            self.assertIn(flag, flags)


if __name__ == "__main__":
    unittest.main()
