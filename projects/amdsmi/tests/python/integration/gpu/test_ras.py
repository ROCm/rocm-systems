#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU RAS, ECC and CPER APIs."""

import unittest

import common.api_test as api
import common.common as common


class TestGpuRas(api.ApiTestCase):
    def test_get_gpu_ecc_count(self):
        self.both("amdsmi_get_gpu_ecc_count", self.handle, api.enum("block", common.GPU_BLOCKS))

    def test_get_gpu_ecc_status(self):
        self.both("amdsmi_get_gpu_ecc_status", self.handle, api.enum("block", common.GPU_BLOCKS))

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
            api.integer("severity_mask", 0),
            api.integer("buffer_size", 4096),
            api.integer("cursor", 0),
        )

    def test_get_afids_from_cper(self):
        # A positive decode needs a real CPER record to feed in.
        self.reject_only(
            "amdsmi_get_afids_from_cper",
            api.Param("cper_afid_data", ("b''", b""), [("bad-type", api.BAD_BYTES)]),
        )


if __name__ == "__main__":
    unittest.main()
