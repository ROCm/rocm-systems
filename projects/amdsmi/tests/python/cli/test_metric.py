#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: metric command."""

import json

from cli.base import TestCliBase


class TestMetric(TestCliBase):
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

    def test_repeated_gpu_selection(self):
        self.common.print_func_name("")
        if len(self.list_data) < 2:
            self.skipTest("Requires at least two GPUs")
        first, second = (gpu["gpu"] for gpu in self.list_data[:2])
        cmd = f"amd-smi metric -u -g={first} -g={second} -g={first} --json"
        rc, data, std_err = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed: {std_err or data}")
        self.assertEqual([gpu["gpu"] for gpu in json.loads(data)["gpu_data"]], [first, second])
