#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: list command."""

import json

from cli.base import TestCliBase


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
        status, output, error = self.util.RunCmdSync(f"amd-smi list {selection} --json")
        self.assertEqual(status, 0, error or output)
        return json.loads(output)

    def test_repeated_gpu_selection(self):
        if len(self.list_data) < 2:
            self.skipTest("Requires at least two GPUs")
        first, second = self.list_data[:2]
        for field in ("gpu", "bdf", "uuid"):
            for template in ("-g {} -g {}", "--gpu={} --gpu={}"):
                selection = template.format(first[field], second[field])
                with self.subTest(selection=selection):
                    self.assertEqual(self._list_json(selection), [first, second])
        for selection, expected in (
            (f"-g {first['gpu']} -g {first['gpu']}", [first]),
            (f"-g {first['gpu']} -g all", self.list_data),
        ):
            with self.subTest(selection=selection):
                self.assertEqual(self._list_json(selection), expected)

    def test_invalid_gpu_selection_errors(self):
        for selection, error_type in (
            (f"-g 0 -g {len(self.list_data)}", "DEVICE_NOT_FOUND"),
            ("-g=0,1", "INVALID_PARAMETER_VALUE"),
        ):
            with self.subTest(selection=selection):
                status, output, error = self.util.RunCmdSync(f"amd-smi list {selection} --json")
                self.assertNotEqual(status, 0, error or output)
                result = json.loads(output)
                self.assertEqual(result["error_type"], error_type)
                self.assertEqual(result["code"], status)
