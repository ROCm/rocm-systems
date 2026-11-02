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
        # Registered before the sweep so a failing command still restores: reset
        # --clocks forces AUTO, which is not what the machine started on if the
        # level was anything else.
        self.addCleanup(self._restore_perf_levels)
        self.RunCmds(cmds)
        return

    def _restore_perf_levels(self):
        self.common.print("Restore Starting Values")
        self.RunCmds(self.PerfLevelRestoreCmds())
