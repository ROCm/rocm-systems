#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run one missing wait under the core memory-wait checker and the race plugin.

Both checkers must write the shared hazard record for the same register site.
Which workgroup, wave and lane a checker reports is the first occurrence it sees,
and with several dispatch threads that can differ between the two checkers, so
the comparison leaves those fields out.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SITE_FIELDS = (
    "gpu",
    "dispatch",
    "kernel",
    "symbol",
    "process",
    "type",
    "access",
    "reg",
    "producer_pc",
    "consumer_pc",
    "producer_disasm",
    "consumer_disasm",
)


class HazardReportCrosscheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("RJ_ROCJITSU", "RJ_SIM_CONFIG", "RJ_HIP_PROGRAM"):
            if not os.environ.get(name):
                raise unittest.SkipTest(f"{name} is not set")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rj-hazard-crosscheck-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_program(self):
        report = self.root / "hazards.jsonl"
        config = json.loads(Path(os.environ["RJ_SIM_CONFIG"]).read_text())
        config["plugins"] = {"race": {}}
        config["diagnostics"] = {"report_path": str(report)}
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        env = dict(os.environ)
        for key in ("LD_PRELOAD", "ROCJITSU_INVOCATION_DIR", "ROCJITSU_SOCKET_PATH"):
            env.pop(key, None)
        env["ROCJITSU_RUNTIME_DIR"] = str(self.root / "runtime")
        result = subprocess.run(
            [
                os.environ["RJ_ROCJITSU"],
                "--config",
                str(config_path),
                "--",
                os.environ["RJ_HIP_PROGRAM"],
            ],
            env=env,
            text=True,
            capture_output=True,
            timeout=240,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [json.loads(line) for line in report.read_text().splitlines()]
        return result.stderr, records

    def test_both_checkers_report_the_same_site(self):
        stderr, records = self.run_program()
        sites = {
            checker: [
                record
                for record in records
                if record["record"] == "site" and record["checker"] == checker
            ]
            for checker in ("memory-wait", "race")
        }
        self.assertEqual(len(sites["memory-wait"]), 1, records)
        self.assertEqual(len(sites["race"]), 1, records)
        core = sites["memory-wait"][0]
        race = sites["race"][0]
        self.assertEqual(
            {field: core[field] for field in SITE_FIELDS},
            {field: race[field] for field in SITE_FIELDS},
        )
        self.assertEqual(core["kernel"], "missing_wait_kernel")
        self.assertTrue(core["producer_disasm"].startswith("global_load_dword"), core)
        self.assertTrue(core["consumer_disasm"].startswith("v_add_u32"), core)
        self.assertFalse(core["trace_executed"])
        self.assertTrue(race["trace_executed"])
        for record in (core, race):
            self.assertLess(record["wg"][0], 4, record)

        summaries = [
            record
            for record in records
            if record["record"] == "summary" and record["dispatch"] == core["dispatch"]
        ]
        self.assertEqual(len(summaries), 1, records)
        self.assertEqual(summaries[0]["gpu"], core["gpu"])
        self.assertEqual(summaries[0]["sites"], 1)
        self.assertEqual(summaries[0]["occurrences"], 4)
        self.assertIn("HAZARD checker=memory-wait kernel=missing_wait_kernel", stderr)
        self.assertIn("RACE kernel=missing_wait_kernel", stderr)


if __name__ == "__main__":
    unittest.main()
