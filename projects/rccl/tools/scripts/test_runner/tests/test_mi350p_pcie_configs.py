#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""Guards for the MI350P PCIe test plans.

These checks lock the review fixes: Debug build, gtest lookup, rccl_home,
algorithm cells RCCL will not select, and the smoke versus full channel split.
"""

import json
import unittest
from pathlib import Path

from lib.test_config import TestConfigProcessor

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_FUNC = _CONFIG_DIR / "mi350p_pcie_8gpu_func.json"
_PERF = _CONFIG_DIR / "mi350p_pcie_8gpu_perf.json"

_RING_ONLY = ("AllGather", "ReduceScatter", "Broadcast")


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _tests(config, name):
    return config["test_configurations"][name].get("tests", [])


class Mi350pPcieConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.func = _load(_FUNC)
        cls.perf = _load(_PERF)
        # Schema validation and extends resolution.
        cls.func_proc = TestConfigProcessor(str(_FUNC))
        cls.perf_proc = TestConfigProcessor(str(_PERF))

    def test_debug_build_and_rccl_home(self):
        for label, doc in (("func", self.func), ("perf", self.perf)):
            flags = doc["build_configuration"]["install_flags"]
            self.assertIn("--debug", flags, label)
            rccl_tests = doc["rccl_tests_build_configuration"]
            self.assertNotIn("rccl_home", rccl_tests, label)

    def test_gtest_binary_is_not_taken_from_rccl_tests(self):
        topo = self.func["test_configurations"]["mi350p_topo_unit"]
        self.assertEqual(topo["binary"], "rccl-UnitTestsFixturesDebug")
        self.assertIn("build/debug/test", topo["test_binary_dir"])
        self.assertNotEqual(
            topo["test_binary_dir"],
            self.func["paths"]["test_binary_dir"],
        )

    def test_no_unselectable_tree_cells(self):
        for doc in (self.func, self.perf):
            for name, block in doc["test_configurations"].items():
                for test in block.get("tests", []):
                    env = test.get("env_variables") or {}
                    algo = env.get("NCCL_ALGO")
                    proto = env.get("NCCL_PROTO")
                    self.assertFalse(
                        algo == "Tree" and proto == "Simple",
                        f"{name}/{test['name']} is Tree+Simple, which gfx950 skips",
                    )
                    for prefix in _RING_ONLY:
                        if test["name"].startswith(prefix):
                            self.assertNotEqual(algo, "Tree", test["name"])
                    if test["name"].startswith("AllToAll"):
                        self.assertNotIn("NCCL_ALGO", env, test["name"])

    def test_tree_ll_allreduce_is_kept(self):
        names = [t["name"] for t in _tests(self.func, "allreduce_smoke")]
        self.assertIn("AllReduce_LL_SHM_Tree_ch4", names)
        self.assertIn("AllReduce_LL_IPC_Tree_ch32", names)

    def test_smoke_scope_is_the_short_channel_set(self):
        smoke = {
            suite["config"]
            for suite in self.func["test_suites"]
            if suite.get("smoke")
        }
        self.assertIn("allreduce_smoke", smoke)
        self.assertIn("mi350p_topo_unit", smoke)
        self.assertNotIn("allreduce_extra", smoke)

        smoke_channels = set()
        for test in _tests(self.func, "allreduce_smoke"):
            smoke_channels.add(test["env_variables"]["NCCL_MIN_NCHANNELS"])
        self.assertEqual(smoke_channels, {"1", "4", "32"})

        extra_channels = set()
        for test in _tests(self.func, "allreduce_extra"):
            extra_channels.add(test["env_variables"]["NCCL_MIN_NCHANNELS"])
        self.assertEqual(extra_channels, {"2", "8", "16", "64"})

    def test_ipc_requests_sys_p2p_and_shm_disables_p2p(self):
        for test in _tests(self.func, "allreduce_smoke"):
            env = test["env_variables"]
            if env["NCCL_P2P_DISABLE"] == "0":
                self.assertEqual(env["NCCL_P2P_LEVEL"], "SYS", test["name"])
                self.assertEqual(env["NCCL_SHM_DISABLE"], "1", test["name"])
            else:
                self.assertEqual(env["NCCL_P2P_DISABLE"], "1", test["name"])
                self.assertNotIn("NCCL_P2P_LEVEL", env, test["name"])

    def test_perf_iteration_counts_match(self):
        proc = self.perf_proc
        args = {
            name: proc.combine_configs(name)["command_args"]
            for name in ("allreduce_perf", "all_gather_perf", "reduce_scatter_perf")
        }
        self.assertEqual(len(set(args.values())), 1, args)
        self.assertIn("-n 20", next(iter(args.values())))
        self.assertIn("-w 5", next(iter(args.values())))

    def test_extends_resolves_shared_ranks(self):
        combined = self.func_proc.combine_configs("allreduce_smoke")
        self.assertEqual(combined["num_ranks"], 8)
        self.assertEqual(combined["binary"], "all_reduce_perf")
        self.assertGreaterEqual(len(combined["tests"]), 18)


if __name__ == "__main__":
    unittest.main()
