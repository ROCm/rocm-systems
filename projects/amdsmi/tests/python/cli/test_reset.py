#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: reset command."""

import os
import unittest

from common.common import amdsmi

from cli.base import TestCliBase


class TestReset(TestCliBase):
    @unittest.skipUnless(
        os.environ.get("AMDSMI_TEST_GPU_RESET") == "1",
        "GPU reset can hang the host; requires AMDSMI_TEST_GPU_RESET=1",
    )
    def test_gpu_reset_zero(self) -> None:
        self.RunCmds(
            [("amd-smi reset --gpureset -g 0", [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED])]
        )

    @unittest.skipUnless(
        os.environ.get("AMDSMI_TEST_GPU_RESET") == "1",
        "GPU reset can hang the host; requires AMDSMI_TEST_GPU_RESET=1",
    )
    def test_gpu_reset_all(self) -> None:
        self.RunCmds(
            [("amd-smi reset --gpureset -g all", [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED])]
        )

    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi reset"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "reset", "Reset Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return
