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

        cmds = self.CreateCmds(
            "reset", "Reset Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        # TODO: remove before committing - skips --gpureset (slow, actually resets
        # the GPU) so local sweeps don't take forever.
        cmds = [c for c in cmds if "gpureset" not in c[0]]
        self.RunCmds(cmds)
        return
