#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: metric command."""

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
