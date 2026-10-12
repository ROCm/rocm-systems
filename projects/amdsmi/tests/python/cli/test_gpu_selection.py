#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only CLI checks for repeated ``--gpu`` selections."""

import json
import unittest

from common.runcmd import Util


class TestCliRepeatedGpuSelection(unittest.TestCase):
    def setUp(self):
        self.util = Util()
        self.devices = self._run_json("amd-smi list --json")
        if len(self.devices) < 2:
            self.skipTest("Requires at least two GPUs")

    def _run_json(self, command):
        status, output, error = self.util.RunCmdSync(command)
        self.assertEqual(status, 0, error or output)
        return json.loads(output)

    def test_repeated_gpu_options_match_one_list(self):
        first, second = self.devices[:2]
        expected = self._run_json(f"amd-smi list --gpu {first['gpu']} {second['gpu']} --json")
        self.assertEqual(expected, self.devices[:2])
        for field in ("gpu", "bdf", "uuid"):
            for template in ("-g {} -g {}", "--gpu={} --gpu={}"):
                selection = template.format(first[field], second[field])
                with self.subTest(selection=selection):
                    self.assertEqual(self._run_json(f"amd-smi list {selection} --json"), expected)

    def test_metric_reports_each_repeated_gpu(self):
        selected = [gpu["gpu"] for gpu in self.devices[:2]]
        data = self._run_json(f"amd-smi metric -u -g={selected[0]} -g={selected[1]} --json")
        self.assertEqual([gpu["gpu"] for gpu in data["gpu_data"]], selected)

    def test_invalid_gpu_selections_return_json_errors(self):
        for selection, error_type in (
            (f"-g 0 -g {len(self.devices)}", "DEVICE_NOT_FOUND"),
            ("-g=0,1", "INVALID_PARAMETER_VALUE"),
        ):
            with self.subTest(selection=selection):
                status, output, error = self.util.RunCmdSync(f"amd-smi list {selection} --json")
                self.assertNotEqual(status, 0, error or output)
                result = json.loads(output)
                self.assertEqual(result["error_type"], error_type)
                self.assertEqual(result["code"], status)
