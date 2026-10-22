#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""Guards for the MI350P PCIe test plans.

These checks lock the review fixes: Debug build, gtest lookup, rccl_home,
algorithm cells RCCL will not select, and the smoke versus full channel split.
"""

import json
import re
import unittest
from fnmatch import fnmatch
from pathlib import Path

from lib.test_config import TestConfigProcessor

_RCCL = Path(__file__).resolve().parents[4]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_FUNC = _CONFIG_DIR / "mi350p_pcie_8gpu_func.json"
_PERF = _CONFIG_DIR / "mi350p_pcie_8gpu_perf.json"
_TOPO_CPP = _RCCL / "test/graph/Mi350pPcieTopoTests.cpp"
_FIXTURES_YAML = _RCCL / "test/test_categories_fixtures_debug.yaml"

_RING_ONLY = ("AllGather", "ReduceScatter", "Broadcast")


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _tests(config, name):
    return config["test_configurations"][name].get("tests", [])


def _merged_env(proc, config_name, test):
    """Group env from extends, then the test's own env. Same order as run_test."""
    group = proc.combine_configs(config_name).get("env_variables") or {}
    return {**group, **(test.get("env_variables") or {})}


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
            self.assertIn("-l", flags, label)
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

    def _plans(self):
        return (
            (self.func, self.func_proc),
            (self.perf, self.perf_proc),
        )

    def test_no_unselectable_tree_cells(self):
        for doc, proc in self._plans():
            for name, block in doc["test_configurations"].items():
                for test in block.get("tests", []):
                    env = _merged_env(proc, name, test)
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
                        self.assertNotIn("NCCL_PROTO", env, test["name"])

    def test_tree_ll_allreduce_is_kept(self):
        names = []
        for suite in self.func["test_suites"]:
            if suite.get("smoke"):
                names.extend(t["name"] for t in _tests(self.func, suite["config"]))
        self.assertIn("AllReduce_LL_SHM_Tree_ch4", names)
        self.assertIn("AllReduce_LL_IPC_Tree_ch32", names)

    def test_smoke_scope_is_the_short_channel_set(self):
        smoke = {
            suite["config"]
            for suite in self.func["test_suites"]
            if suite.get("smoke")
        }
        extra = {
            suite["config"]
            for suite in self.func["test_suites"]
            if not suite.get("smoke")
        }
        self.assertIn("ar_simple_shm_ring_smoke", smoke)
        self.assertIn("mi350p_topo_unit", smoke)
        self.assertIn("ar_simple_shm_ring_extra", extra)
        self.assertTrue(all(not name.endswith("_extra") for name in smoke))

        def channels(configs):
            found = set()
            for name in configs:
                for test in _tests(self.func, name):
                    if not test["name"].startswith("AllReduce_"):
                        continue
                    found.add(_merged_env(self.func_proc, name, test)["NCCL_MIN_NCHANNELS"])
            return found

        self.assertEqual(channels(smoke), {"1", "4", "32"})
        self.assertEqual(channels(extra), {"2", "8", "16", "64"})

    def test_ipc_requests_sys_p2p_and_shm_disables_p2p(self):
        for name, block in self.func["test_configurations"].items():
            if not (name.startswith("ar_") and name.endswith(("_smoke", "_extra"))):
                continue
            for test in block.get("tests", []):
                env = _merged_env(self.func_proc, name, test)
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

    def test_perf_allreduce_cells_share_env_without_pinning_auto(self):
        auto = self.perf_proc.combine_configs("allreduce_perf")
        self.assertEqual(auto["tests"][0]["name"], "AllReduce_auto")
        self.assertEqual(auto["tests"][0].get("env_variables"), {})
        self.assertFalse(auto.get("env_variables"))

        cell_names = [
            name
            for name in self.perf["test_configurations"]
            if name.startswith("ar_perf_")
        ]
        self.assertEqual(len(cell_names), 6)
        self.assertEqual(
            sum(len(self.perf_proc.combine_configs(name)["tests"]) for name in cell_names),
            12,
        )
        for name in cell_names:
            combined = self.perf_proc.combine_configs(name)
            self.assertIn(combined["env_variables"]["NCCL_PROTO"], {"Simple", "LL"})
            self.assertIn(combined["env_variables"]["NCCL_ALGO"], {"Ring", "Tree"})
            for test in combined["tests"]:
                self.assertEqual(
                    set(test["env_variables"]),
                    {"NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS"},
                )

    def test_topo_names_stay_tied(self):
        """A renamed TEST() must not leave the plan green with the suite gone."""
        source = _TOPO_CPP.read_text(encoding="utf-8")
        match = re.search(r"TEST\((\w+),\s*(\w+)\)", source)
        self.assertIsNotNone(match)
        suite, case = match.group(1), match.group(2)
        gtest_name = f"{suite}.{case}"

        topo = self.func["test_configurations"]["mi350p_topo_unit"]
        self.assertEqual(topo["tests"][0]["test_filter"], gtest_name)

        model = re.search(r"topo_expl/models/([\w.]+\.xml)", source)
        self.assertIsNotNone(model)
        model_path = _RCCL / "tools" / "topo_expl" / "models" / model.group(1)
        self.assertTrue(model_path.is_file(), model_path)

        yaml_text = _FIXTURES_YAML.read_text(encoding="utf-8")
        patterns_block = yaml_text.split("test_patterns:", 1)[1].split("exclude_windows:", 1)[0]
        patterns = re.findall(r'"([^"]+)"', patterns_block)
        self.assertTrue(
            any(fnmatch(gtest_name, pattern) for pattern in patterns),
            f"{gtest_name} is not covered by {_FIXTURES_YAML.name}",
        )

    def test_extends_resolves_shared_ranks(self):
        combined = self.func_proc.combine_configs("ar_simple_shm_ring_smoke")
        self.assertEqual(combined["num_ranks"], 8)
        self.assertEqual(combined["binary"], "all_reduce_perf")
        self.assertEqual(len(combined["tests"]), 3)
        self.assertEqual(combined["env_variables"]["NCCL_PROTO"], "Simple")
        self.assertEqual(combined["env_variables"]["NCCL_ALGO"], "Ring")
        merged = _merged_env(self.func_proc, "ar_simple_shm_ring_smoke", combined["tests"][0])
        self.assertEqual(merged["NCCL_MIN_NCHANNELS"], "1")
        self.assertEqual(merged["NCCL_P2P_DISABLE"], "1")

    def test_ring_ch4_is_shared_and_alltoall_does_not_inherit_it(self):
        leaves = [
            name
            for name in self.func["test_configurations"]
            if name.startswith(("ag_", "rs_", "bc_"))
        ]
        self.assertEqual(len(leaves), 12)
        transports = set()
        for name in leaves:
            combined = self.func_proc.combine_configs(name)
            env = combined["env_variables"]
            self.assertEqual(env["NCCL_MIN_NCHANNELS"], "4", name)
            self.assertEqual(env["NCCL_MAX_NCHANNELS"], "4", name)
            self.assertEqual(env["NCCL_ALGO"], "Ring", name)
            self.assertIn(env["NCCL_PROTO"], {"Simple", "LL"}, name)
            transports.add((
                env["NCCL_PROTO"],
                env["NCCL_SHM_DISABLE"],
                env["NCCL_P2P_DISABLE"],
                env.get("NCCL_P2P_LEVEL"),
            ))
            test = combined["tests"][0]
            self.assertNotIn("env_variables", test, test["name"])
            merged = _merged_env(self.func_proc, name, test)
            self.assertEqual(merged["NCCL_ALGO"], "Ring", test["name"])
        self.assertEqual(len(transports), 4)
        alltoall = self.func_proc.combine_configs("alltoall_perf")
        self.assertNotIn("NCCL_ALGO", alltoall.get("env_variables") or {})


if __name__ == "__main__":
    unittest.main()
