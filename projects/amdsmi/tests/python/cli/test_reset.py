#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: reset command."""

from cli.base import TestCliBase


class TestReset(TestCliBase):
    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi reset"
        self.common.print(msg)

        # TODO allow reset commands to be executed
        if not self.PrintCmdsOnly:
            if self.common.TODO_SKIP_FAIL:
                msg = f"{self.tab}Needs Testing, Not Yet Implemented"
                # self.common.print(msg)
                self.skipTest(msg)

        cmds = self.CreateCmds(
            "reset", "Reset Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return
