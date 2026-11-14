#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: monitor command."""

from cli.base import TestCliBase


class TestMonitor(TestCliBase):
    def test_requested_monitor_arguments(self) -> None:
        for arguments in ("", "-p -t -u -m -q -w 1 -i 2"):
            with self.subTest(arguments=arguments):
                cmd = f"amd-smi monitor {arguments}".strip()
                rc, output, stderr = self.util.RunCmdSync(cmd, time_out=30)
                self.assertEqual(rc, self.PASS, f"{cmd}: {stderr}")
                self.assertTrue(output.strip(), cmd)

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
