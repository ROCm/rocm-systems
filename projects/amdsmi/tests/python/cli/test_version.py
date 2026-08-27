#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: version command."""

from cli.base import TestCliBase


class TestVersion(TestCliBase):
    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi version"
        self.common.print(msg)

        cmds = [
            ("amd-smi version", self.PASS),
            ("amd-smi version --cpu_version", self.PASS),
            ("amd-smi version --gpu_version", self.PASS),
        ]

        self.RunCmds(cmds)
        return
