#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: xgmi command."""

from cli.base import TestCliBase


class TestXgmi(TestCliBase):
    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi xgmi"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "xgmi", "XGMI arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return
