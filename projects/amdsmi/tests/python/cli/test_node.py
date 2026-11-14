#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: node command."""

import json

from cli.base import TestCliBase


class TestNode(TestCliBase):
    def test_power_management_short_option(self) -> None:
        cmd = "amd-smi node -p --json"
        rc, output, stderr = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"{cmd}: {stderr}")
        entries = json.loads(output)
        self.assertTrue(entries, cmd)
        for entry in entries:
            self.assertIn("power_management", entry["node"], cmd)

    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi node"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "node", "Node arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return
