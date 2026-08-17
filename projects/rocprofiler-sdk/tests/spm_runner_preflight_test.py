#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import unittest
from unittest import mock

import spm_runner_preflight


class SpmRunnerPreflightTest(unittest.TestCase):
    def test_parse_dot_version(self):
        self.assertEqual(
            spm_runner_preflight.parse_dot_version("6.19.14.31400000"),
            (6, 19, 14, 31400000),
        )

    def test_version_ge(self):
        self.assertTrue(
            spm_runner_preflight.version_ge(
                "6.19.14.31400000", spm_runner_preflight.SPM_MIN_AMDGPU_DRIVER_VERSION
            )
        )
        self.assertFalse(
            spm_runner_preflight.version_ge(
                "6.18.0.0", spm_runner_preflight.SPM_MIN_AMDGPU_DRIVER_VERSION
            )
        )

    def test_check_passes_on_supported_driver(self):
        with mock.patch.object(
            spm_runner_preflight,
            "read_amdgpu_driver_version",
            return_value="6.19.14.31400000",
        ):
            spm_runner_preflight.check_spm_runner_requirements()

    def test_check_fails_on_unsupported_driver(self):
        with mock.patch.object(
            spm_runner_preflight,
            "read_amdgpu_driver_version",
            return_value="6.18.0.0",
        ):
            with self.assertRaises(SystemExit):
                spm_runner_preflight.check_spm_runner_requirements()


if __name__ == "__main__":
    unittest.main()
