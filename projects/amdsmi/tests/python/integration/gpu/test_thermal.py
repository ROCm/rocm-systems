#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU temperature and fan APIs."""

import unittest

import common.api_test as api
import common.common as common


class TestGpuThermal(api.ApiTestCase):
    @classmethod
    def _present_sensors(cls):
        """The sensor types this ASIC answers for.

        Reading the whole 66 x 14 grid is thousands of device reads on a
        multi-GPU box, nearly all of them not-supported.
        """
        if not cls.has_device:
            return []
        metric = common.TEMPERATURE_METRICS[0][1]
        present = []
        for name, value, cond in common.TEMPERATURE_TYPES:
            if cond != common.PASS:
                continue
            try:
                common.amdsmi.amdsmi_get_temp_metric(cls.common.processors[0], value, metric)
            except common.amdsmi.AmdSmiLibraryException as e:
                # Drop only the sensors this board does not have. Any other
                # status is a defect and must still be driven.
                if str(e.get_error_code()) in {
                    code for code, _ in cls.common.not_supported_error_codes
                }:
                    continue
            present.append((name, value))
        return present

    def test_get_temp_metric(self):
        every_type = [(name, value) for name, value, _ in common.TEMPERATURE_TYPES]
        present = self._present_sensors()
        sensor = api.Param(
            "sensor_type",
            (present or every_type)[0],
            [("bad-type", api.BAD_ENUM)],
            sweep=every_type,
            accepted=present,
        )
        self.both(
            "amdsmi_get_temp_metric",
            self.handle,
            sensor,
            api.enum("metric", common.TEMPERATURE_METRICS),
        )

    def test_get_gpu_fan_rpms(self):
        self.both("amdsmi_get_gpu_fan_rpms", self.handle, api.integer("sensor_idx", 0, bounds=True))

    def test_get_gpu_fan_speed(self):
        self.both(
            "amdsmi_get_gpu_fan_speed", self.handle, api.integer("sensor_idx", 0, bounds=True)
        )

    def test_get_gpu_fan_speed_max(self):
        self.both(
            "amdsmi_get_gpu_fan_speed_max", self.handle, api.integer("sensor_idx", 0, bounds=True)
        )

    def test_set_gpu_fan_speed(self):
        self.reject_only(
            "amdsmi_set_gpu_fan_speed",
            self.handle,
            api.integer("sensor_idx", 0),
            api.integer("fan_speed", 0),
        )

    def test_reset_gpu_fan(self):
        self.reject_only("amdsmi_reset_gpu_fan", self.handle, api.integer("sensor_idx", 0))


if __name__ == "__main__":
    unittest.main()
