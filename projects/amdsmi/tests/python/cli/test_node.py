#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: node command."""

from cli.base import TestCliBase


class TestNode(TestCliBase):
    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi node"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "node", "Node arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return
