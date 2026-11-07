#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: list command."""

import json

from cli.base import AmdSmiExitCode, TestCliBase


class TestList(TestCliBase):
    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi list"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "list", "List Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def _list_json(self, selection):
        cmd = f"amd-smi list {selection} --json"
        rc, data, std_err = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed: {std_err or data}")
        return json.loads(data)

    def test_repeated_gpu_selection(self):
        self.common.print_func_name("")
        if len(self.list_data) < 2:
            self.skipTest("Requires at least two GPUs")
        first, second = self.list_data[:2]
        for field in ("gpu", "bdf", "uuid"):
            for template in ("-g {} -g {}", "--gpu={} --gpu={}"):
                selection = template.format(first[field], second[field])
                with self.subTest(selection=selection):
                    self.assertEqual(self._list_json(selection), [first, second])

    def test_duplicate_gpu_selection(self):
        self.common.print_func_name("")
        if not self.list_data:
            self.skipTest("Requires at least one GPU")
        first = self.list_data[0]
        for selection, expected in (
            (f"-g {first['gpu']} {first['gpu']}", [first]),
            (f"-g {first['gpu']} -g {first['gpu']}", [first]),
            (f"-g {first['gpu']} --gpu {first['bdf']} --gpu {first['uuid']}", [first]),
            (f"-g {first['gpu']} -g all", self.list_data),
        ):
            with self.subTest(selection=selection):
                self.assertEqual(self._list_json(selection), expected)

    def test_invalid_gpu_selection_errors(self):
        self.common.print_func_name("")
        if not self.list_data:
            self.skipTest("Requires at least one GPU")
        for selection, expected_code in (
            (
                f"-g {self.list_data[0]['gpu']} -g {len(self.list_data)}",
                AmdSmiExitCode.DEVICE_NOT_FOUND,
            ),
            ("-g=0,1", AmdSmiExitCode.INVALID_PARAMETER_VALUE),
        ):
            with self.subTest(selection=selection):
                cmd = f"amd-smi list {selection} --json"
                rc, data, std_err = self.util.RunCmdSync(cmd)
                self.assertEqual(rc, expected_code.value, f"Command '{cmd}': {std_err or data}")
                result = json.loads(data)
                self.assertEqual(result["error_type"], expected_code.name)
                self.assertEqual(result["code"], rc)
