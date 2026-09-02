#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: firmware (and ucode) command."""

from cli.base import TestCliBase


class TestFirmware(TestCliBase):
    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi firmware"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "firmware", "Firmware Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        cmds = self.CreateCmds(
            "ucode", "Firmware Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return
