#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the ppod/vpod/station fabric config setters (hardware-free).

Covers argument validation for amdsmi_set_gpu_fabric_{ppod,vpod,station}_config
and the shared _populate_fabric_config_data() packing helper, stubbing the C
entry point so nothing here touches real hardware or sysfs.
"""

import unittest
from unittest import mock

from common.common import amdsmi

_BAD_HANDLE = None
_GOOD_HANDLE = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()


class TestFabricConfigSetterArgumentValidation(unittest.TestCase):
    """Every setter must reject the same shapes of bad input the same way."""

    def test_ppod_rejects_bad_processor_handle(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_gpu_fabric_ppod_config(_BAD_HANDLE, {"accelerator_id": 1})

    def test_vpod_rejects_bad_processor_handle(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_gpu_fabric_vpod_config(_BAD_HANDLE, {"vpod_id": 1})

    def test_station_rejects_bad_processor_handle(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_gpu_fabric_station_config(_BAD_HANDLE, {"station_flags": 1})

    def test_ppod_rejects_non_dict_data(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_gpu_fabric_ppod_config(_GOOD_HANDLE, "not a dict")

    def test_vpod_rejects_non_dict_data(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_gpu_fabric_vpod_config(_GOOD_HANDLE, ["not", "a", "dict"])

    def test_station_rejects_non_dict_data(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_gpu_fabric_station_config(_GOOD_HANDLE, 5)

    def test_ppod_rejects_non_bool_commit(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_gpu_fabric_ppod_config(_GOOD_HANDLE, {"accelerator_id": 1}, commit=1)

    def test_vpod_rejects_non_bool_commit(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_gpu_fabric_vpod_config(_GOOD_HANDLE, {"vpod_id": 1}, commit="yes")

    def test_station_rejects_non_bool_commit(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_gpu_fabric_station_config(
                _GOOD_HANDLE, {"station_flags": 1}, commit=None
            )

    def test_ppod_rejects_unknown_field_name(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_gpu_fabric_ppod_config(_GOOD_HANDLE, {"not_a_real_field": 1})

    def test_vpod_rejects_unknown_field_name(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_gpu_fabric_vpod_config(_GOOD_HANDLE, {"num_stations": 1})

    def test_station_rejects_unknown_field_name(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_gpu_fabric_station_config(_GOOD_HANDLE, {"vpod_id": 1})


class TestPopulateFabricConfigData(unittest.TestCase):
    """The shared packing helper: mask derivation and per-field writes."""

    def test_scalar_fields_written_and_masked(self):
        config = amdsmi.amdsmi_wrapper.amdsmi_fabric_ppod_config_t()
        mask = amdsmi.amdsmi_interface._populate_fabric_config_data(
            config,
            {"accelerator_id": 42, "bandwidth": 100},
            amdsmi.amdsmi_interface._FABRIC_PPOD_CONFIG_FIELD_MASKS,
        )
        self.assertEqual(config.data.accelerator_id, 42)
        self.assertEqual(config.data.bandwidth, 100)
        expected_mask = (
            amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID
            | amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_PPOD_FIELD_BANDWIDTH
        )
        self.assertEqual(mask, expected_mask)

    def test_unsupplied_fields_left_untouched(self):
        # A zero-initialized struct's other fields are not written just because
        # one field was requested; only the mask bits for supplied keys are set.
        config = amdsmi.amdsmi_wrapper.amdsmi_fabric_ppod_config_t()
        amdsmi.amdsmi_interface._populate_fabric_config_data(
            config, {"latency": 7}, amdsmi.amdsmi_interface._FABRIC_PPOD_CONFIG_FIELD_MASKS
        )
        self.assertEqual(config.data.bandwidth, 0)
        self.assertEqual(config.data.accelerator_id, 0)

    def test_array_field_written_via_slice_assignment(self):
        config = amdsmi.amdsmi_wrapper.amdsmi_fabric_ppod_config_t()
        values = list(range(16))
        amdsmi.amdsmi_interface._populate_fabric_config_data(
            config,
            {"local_accelerators": values},
            amdsmi.amdsmi_interface._FABRIC_PPOD_CONFIG_FIELD_MASKS,
        )
        self.assertEqual(list(config.data.local_accelerators), values)

    def test_unknown_field_raises_before_any_write(self):
        config = amdsmi.amdsmi_wrapper.amdsmi_fabric_vpod_config_t()
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_interface._populate_fabric_config_data(
                config,
                {"vpod_id": 3, "bogus_field": 1},
                amdsmi.amdsmi_interface._FABRIC_VPOD_CONFIG_FIELD_MASKS,
            )


class TestFabricSetterCallSite(unittest.TestCase):
    """Drives each setter through the (stubbed) C entry point end to end."""

    def test_ppod_config_reaches_the_c_api_with_version_and_mask(self):
        captured = {}

        def _stub(_handle, config_ptr):
            cfg = config_ptr._obj
            captured["version"] = cfg.version
            captured["mask"] = cfg.mask
            captured["commit"] = cfg.commit
            captured["accelerator_id"] = cfg.data.accelerator_id
            return 0

        with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_set_gpu_fabric_ppod_config", _stub):
            amdsmi.amdsmi_set_gpu_fabric_ppod_config(
                _GOOD_HANDLE, {"accelerator_id": 7}, commit=False
            )

        self.assertEqual(captured["version"], amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_PPOD_CONFIG_V1)
        self.assertEqual(captured["mask"], amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID)
        self.assertFalse(captured["commit"])
        self.assertEqual(captured["accelerator_id"], 7)

    def test_vpod_config_reaches_the_c_api(self):
        captured = {}

        def _stub(_handle, config_ptr):
            cfg = config_ptr._obj
            captured["vpod_id"] = cfg.data.vpod_id
            captured["mask"] = cfg.mask
            return 0

        with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_set_gpu_fabric_vpod_config", _stub):
            amdsmi.amdsmi_set_gpu_fabric_vpod_config(_GOOD_HANDLE, {"vpod_id": 9})

        self.assertEqual(captured["vpod_id"], 9)
        self.assertEqual(captured["mask"], amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID)

    def test_station_config_reaches_the_c_api(self):
        captured = {}

        def _stub(_handle, config_ptr):
            cfg = config_ptr._obj
            captured["num_stations"] = cfg.data.num_stations
            captured["mask"] = cfg.mask
            return 0

        with mock.patch.object(
            amdsmi.amdsmi_wrapper, "amdsmi_set_gpu_fabric_station_config", _stub
        ):
            amdsmi.amdsmi_set_gpu_fabric_station_config(_GOOD_HANDLE, {"num_stations": 3})

        self.assertEqual(captured["num_stations"], 3)
        self.assertEqual(
            captured["mask"], amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_DF_FIELD_NUM_STATIONS
        )

    def test_library_error_raised_as_amdsmi_exception(self):
        with mock.patch.object(
            amdsmi.amdsmi_wrapper,
            "amdsmi_set_gpu_fabric_ppod_config",
            lambda _h, _c: amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL,
        ):
            with self.assertRaises(amdsmi.AmdSmiLibraryException):
                amdsmi.amdsmi_set_gpu_fabric_ppod_config(_GOOD_HANDLE, {"accelerator_id": 1})


if __name__ == "__main__":
    unittest.main()
