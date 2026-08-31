#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: monitor command."""

from cli.base import TestCliBase


class TestMonitor(TestCliBase):
    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi monitor"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "monitor",
            "Monitor Arguments:",
            "Device Arguments:",
            "Command Modifiers:",
            "Watch Arguments:",
        )
        self.RunCmds(cmds)
        return
