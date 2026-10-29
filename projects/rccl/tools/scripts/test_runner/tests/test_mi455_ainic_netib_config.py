#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""Guards for the NET/IB suites of the MI455 functional plans.

NET/IB suites run in the AINIC plan only; the generic MI455 plan keeps them
disabled. These checks lock the AINIC environment and keep the GPU-memory vNIC
tests, which hang an MI450 host, out of every enabled suite.
"""

import re
import unittest
from fnmatch import fnmatch
from pathlib import Path

from lib.test_config import TestConfigProcessor

_RCCL = Path(__file__).resolve().parents[4]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_AINIC = _CONFIG_DIR / "mi455_ainic_roce_func.json"
_GENERIC = _CONFIG_DIR / "mi455_roce_func.json"
_NETIB_SOURCES = _RCCL / "test/transport/NetIbMPI"

_NETIB_PREFIXES = ("NetIb", "GdrFlushTest", "GdrFlushFlag", "OpsFault", "Transport/NetIbGidChangeTest")
_HOST_HANG_TESTS = (
    "NetIbMPITest.SendRecvDifferentMemoryTypes",
    "NetIbMPITest.MultiRecvGPUShuffled",
    "NetIbMPITest.FlushRepeated_VNic",
)
_TEST_RE = re.compile(r"^\s*TEST(_F|_P)?\(\s*(\w+)\s*,\s*(\w+)\s*\)", re.M)


def _positive_patterns(test_filter):
    return [p for p in test_filter.split("-", 1)[0].split(":") if p]


def _netib_suites(proc):
    for suite in proc.parse_test_suites():
        if suite.get("binary") != "rccl-UnitTestsMPI":
            continue
        filters = [t.get("test_filter", "") for t in suite.get("tests", [])]
        if any(f.startswith(_NETIB_PREFIXES) for f in filters):
            yield suite


def _matches(suite, name):
    return any(fnmatch(name, p)
               for t in suite.get("tests", [])
               for p in _positive_patterns(t.get("test_filter", "")))


class Mi455AinicNetIbConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Schema validation and extends resolution.
        cls.ainic = TestConfigProcessor(str(_AINIC))
        cls.generic = TestConfigProcessor(str(_GENERIC))
        cls.suites = list(_netib_suites(cls.ainic))

    def test_plugin_name_is_exact(self):
        # GetPlugin() matches NCCL_NET with strcmp, so "ib-cast" silently runs plain IB.
        for suite in self.suites:
            for test in suite.get("tests", []):
                env = {**suite.get("env_variables", {}), **(test.get("env_variables") or {})}
                self.assertIn(env.get("NCCL_NET", "IB"), ("IB", "IB-CAST"),
                              f"{suite['suite_details']['name']}: {test.get('name')}")

    def test_netib_base_env(self):
        env = self.ainic.combine_configs("net_ib_base")["env_variables"]
        self.assertEqual(env.get("RCCL_CTS_OFFLOAD_ENABLED"), "0")
        self.assertEqual(env.get("NCCL_GDR_FLUSH_DISABLE"), "1")

    def test_gdr_flush_stays_disabled(self):
        # Meta RoCE (ionic) does not support the flush QP's loopback RDMA READ.
        for suite in self.suites:
            for test in suite.get("tests", []):
                env = {**suite.get("env_variables", {}), **(test.get("env_variables") or {})}
                self.assertEqual(env.get("NCCL_GDR_FLUSH_DISABLE"), "1",
                                 f"{suite['suite_details']['name']}: {test.get('name')}")

    def test_host_hang_tests_stay_disabled(self):
        for name in _HOST_HANG_TESTS:
            holders = [s for s in self.suites if _matches(s, name)]
            self.assertTrue(holders, f"{name} is in no suite")
            for suite in holders:
                self.assertFalse(suite["suite_details"]["enabled"],
                                 f"{name} in enabled suite {suite['suite_details']['name']}")

    def test_netib_suites_enabled(self):
        for suite in self.suites:
            if any(_matches(suite, name) for name in _HOST_HANG_TESTS):
                continue
            self.assertTrue(suite["suite_details"]["enabled"], suite["suite_details"]["name"])

    def test_every_netib_test_has_a_suite(self):
        missing = []
        for source in sorted(_NETIB_SOURCES.glob("*.cpp")):
            for kind, fixture, test in _TEST_RE.findall(source.read_text(encoding="utf-8")):
                if test.startswith("DISABLED_"):
                    continue
                name = (f"Transport/{fixture}.{test}/NetIb" if kind == "_P"
                        else f"{fixture}.{test}")
                if not any(_matches(s, name) for s in self.suites):
                    missing.append(name)
        self.assertEqual(missing, [])

    def test_generic_plan_keeps_netib_disabled(self):
        enabled = {s["suite_details"]["name"] for s in _netib_suites(self.generic)
                   if s["suite_details"]["enabled"]}
        self.assertNotIn("NET IB - Data Transfer", enabled)
        self.assertNotIn("NET IB - CAST WRR+Split (QPS=2, splitData=1)", enabled)


if __name__ == "__main__":
    unittest.main()
