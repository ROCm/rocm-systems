#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: metric command."""

import json

from cli.base import TestCliBase


class TestMetric(TestCliBase):
    def test_requested_metric_arguments(self) -> None:
        for option, section in (("-c", "clock"), ("-P", "pcie"), ("--voltage", "voltage")):
            with self.subTest(option=option):
                cmd = f"amd-smi metric {option} --json"
                rc, output, stderr = self.util.RunCmdSync(cmd)
                self.assertEqual(rc, self.PASS, f"{cmd}: {stderr}")
                entries = json.loads(output)["gpu_data"]
                self.assertTrue(entries, cmd)
                for entry in entries:
                    self.assertIn(section, entry, cmd)

    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi metric"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "metric",
            "Metric arguments:",
            "Device Arguments:",
            "Command Modifiers:",
            "Watch Arguments:",
        )
        self.RunCmds(cmds)
        return

    def test_csv(self):
        """`amd-smi metric --csv` is well-formed CSV, one row per GPU, agreeing with JSON."""
        self.common.print_func_name("")
        _, csv_rows = self.assert_json_csv_consistent("amd-smi metric")
        gpu_ids = [int(row["gpu"]) for row in csv_rows]
        self.assertEqual(len(gpu_ids), len(set(gpu_ids)), f"GPU rows repeated: {gpu_ids}")
        self.assert_gpu_ids(gpu_ids, "amd-smi metric --csv")
