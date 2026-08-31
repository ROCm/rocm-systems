#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: bad-pages command."""

from cli.base import TestCliBase


class TestBadPages(TestCliBase):
    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi bad-pages"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "bad-pages", "Bad Pages Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return
