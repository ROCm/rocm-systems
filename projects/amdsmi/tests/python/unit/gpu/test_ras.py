#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""GPU RAS, ECC and CPER APIs."""

import unittest

import common.common as common


class TestGpuRas(common.ApiTestCase):
    def test_get_gpu_ecc_count(self):
        self.both("amdsmi_get_gpu_ecc_count", self.handle, common.enum("block", common.GPU_BLOCKS))

    def test_get_gpu_ecc_status(self):
        self.both("amdsmi_get_gpu_ecc_status", self.handle, common.enum("block", common.GPU_BLOCKS))

    def test_get_gpu_ecc_enabled(self):
        self.both("amdsmi_get_gpu_ecc_enabled", self.handle)

    def test_get_gpu_total_ecc_count(self):
        self.both("amdsmi_get_gpu_total_ecc_count", self.handle)

    def test_get_gpu_ras_feature_info(self):
        self.both("amdsmi_get_gpu_ras_feature_info", self.handle)

    def test_get_gpu_ras_block_features_enabled(self):
        self.both("amdsmi_get_gpu_ras_block_features_enabled", self.handle)

    def test_get_violation_status(self):
        self.both("amdsmi_get_violation_status", self.handle)

    def test_gpu_validate_ras_eeprom(self):
        # Reads the RAS EEPROM; left to the functional tier to drive positively.
        self.reject_only("amdsmi_gpu_validate_ras_eeprom", self.handle)

    def test_get_gpu_cper_entries(self):
        # A positive read needs a populated CPER ring and cursor bookkeeping.
        self.reject_only(
            "amdsmi_get_gpu_cper_entries",
            self.handle,
            common.integer("severity_mask", 0),
            common.integer("buffer_size", 4096),
            common.integer("cursor", 0),
        )

    def test_get_afids_from_cper(self):
        # A positive decode needs a real CPER record to feed in.
        self.reject_only(
            "amdsmi_get_afids_from_cper",
            common.Param("cper_afid_data", ("b''", b""), [("bad-type", common.BAD_BYTES)]),
        )


if __name__ == "__main__":
    unittest.main()
