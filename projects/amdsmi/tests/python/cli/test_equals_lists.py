# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only CLI checks for equals-form GPU lists."""

import json
import unittest

from common.runcmd import Util


class TestCliGpuEqualsLists(unittest.TestCase):
    def setUp(self) -> None:
        self.util = Util()

    def _list_gpus(self, selection: str) -> list:
        status, output, error = self.util.RunCmdSync(f"amd-smi list {selection} --json")
        self.assertEqual(status, 0, error or output)
        return json.loads(output)

    def test_gpu_equals_lists_match_space_lists(self) -> None:
        devices = self._list_gpus("")
        if len(devices) < 2:
            self.skipTest("Requires at least two GPUs")
        selected = devices[:2]
        expected = self._list_gpus("--gpu " + " ".join(str(gpu["gpu"]) for gpu in selected))
        for flag in ("--gpu", "-g"):
            for field in ("gpu", "bdf", "uuid"):
                with self.subTest(flag=flag, field=field):
                    values = ",".join(str(gpu[field]) for gpu in selected)
                    self.assertEqual(self._list_gpus(f"{flag}={values}"), expected)

    def test_metric_equals_list_reports_both_gpus(self) -> None:
        devices = self._list_gpus("")
        if len(devices) < 2:
            self.skipTest("Requires at least two GPUs")
        selected = [gpu["gpu"] for gpu in devices[:2]]
        values = ",".join(str(gpu) for gpu in selected)
        status, output, error = self.util.RunCmdSync(f"amd-smi metric -u -g={values} --json")
        self.assertEqual(status, 0, error or output)
        self.assertEqual([gpu["gpu"] for gpu in json.loads(output)["gpu_data"]], selected)

    def test_invalid_equals_lists_return_json_errors(self) -> None:
        for selection, error_type in (
            ("--gpu=0,", "INVALID_PARAMETER_VALUE"),
            ("--gpu=,0", "INVALID_PARAMETER_VALUE"),
            ("--gpu=0,,1", "INVALID_PARAMETER_VALUE"),
            ("--gpu=0,--json", "INVALID_PARAMETER_VALUE"),
            ("--gpu=0,1 all", "INVALID_PARAMETER"),
        ):
            with self.subTest(selection=selection):
                status, output, error = self.util.RunCmdSync(f"amd-smi list {selection} --json")
                self.assertNotEqual(status, 0, error or output)
                result = json.loads(output)
                self.assertEqual(result["error_type"], error_type)
                self.assertEqual(result["code"], status)
