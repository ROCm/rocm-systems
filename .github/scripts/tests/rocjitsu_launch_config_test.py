# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if SCRIPTS.name == "tests":
    SCRIPTS = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import rocjitsu_launch_config  # noqa: E402


class LaunchConfigTest(unittest.TestCase):
    def test_parallel_sets_engine_threads_and_cu_quantum(self):
        config = {
            "thread_allocations": [{"num_threads": 1, "cpu_dispatch_threads": 1}],
            "topology": {
                "root": {
                    "type": "soc",
                    "children": [
                        {
                            "type": "compute_unit",
                            "config": [{"key": "num_wf_slots", "value": "32"}],
                        }
                    ],
                }
            },
        }
        rocjitsu_launch_config.apply_overrides(
            config, num_threads=5, functional_quantum=0, cpu_dispatch_threads=170
        )
        self.assertEqual(config["num_threads"], 5)
        self.assertEqual(config["cpu_dispatch_threads"], 170)
        entries = config["topology"]["root"]["children"][0]["config"]
        self.assertEqual(
            entries,
            [
                {"key": "num_wf_slots", "value": "32"},
                {"key": "functional_quantum", "value": "0"},
            ],
        )

    def test_gfx942_preset_accepts_the_requested_overrides(self):
        source = SCRIPTS.parents[1] / "emulation/rocjitsu/configs/gfx942_cdna3.json"
        if not source.is_file():
            self.skipTest("gfx942 config is not in this checkout")
        config = json.loads(source.read_text(encoding="utf-8"))
        rocjitsu_launch_config.apply_overrides(
            config, num_threads=5, functional_quantum=0, cpu_dispatch_threads=170
        )
        units = []

        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "compute_unit":
                    units.append(node)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(config["topology"])
        self.assertGreaterEqual(len(units), 1)
        for unit in units:
            quantum = [item for item in unit["config"] if item["key"] == "functional_quantum"]
            self.assertEqual(quantum, [{"key": "functional_quantum", "value": "0"}])


if __name__ == "__main__":
    unittest.main()
