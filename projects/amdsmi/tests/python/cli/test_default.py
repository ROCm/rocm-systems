#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: default command (bare ``amd-smi``)."""

from cli.base import TestCliBase


class TestDefault(TestCliBase):
    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi"
        self.common.print(msg)

        cmds = [("amd-smi", self.PASS)]

        self.RunCmds(cmds)
        return
