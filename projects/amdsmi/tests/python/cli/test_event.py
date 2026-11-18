#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: event command."""

from cli.base import TestCliBase


class TestEvent(TestCliBase):
    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi event"
        self.common.print(msg)

        # TODO allow event commands to be executed
        if not self.PrintCmdsOnly:
            if self.common.TODO_SKIP_FAIL:
                msg = f"{self.tab}Needs input"
                self.common.print(msg)
                self.skipTest(msg)

        # Start process with "amd-smi event"
        # In another process create an event with like "amd-smi reset --gpureset"
        cmds = self.CreateCmds(
            "event", "Event Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return
