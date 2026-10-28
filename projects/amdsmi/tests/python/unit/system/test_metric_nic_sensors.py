#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""metric --nic --sensors tests: records are grouped into one entry per sensor with its
thresholds, shown as C/V/W in text and as value plus unit in JSON, and a NIC with no sensors
says so plainly instead of raising.

These load subcommands/metric.py standalone with a stubbed ``amdsmi`` package so the control flow
is exercised without a built libamd_smi.so or NIC hardware.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

from common.common import amdsmi_path, cli_search_order, find_cli_dir

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
_MODULE_PATH = os.path.join(_CLI_DIR, "subcommands", "metric.py") if _CLI_DIR else None

if not _MODULE_PATH or not os.path.isfile(_MODULE_PATH):
    raise unittest.SkipTest(
        f"amd-smi CLI metric.py not found (looked in {_CLI_DIR or amdsmi_path})"
    )


def _record(category, attribute, unit, channel, instance, label, value):
    return {
        "category": category,
        "attribute": attribute,
        "unit": unit,
        "channel": channel,
        "instance": instance,
        "label": label,
        "value": value,
    }


_IONIC_RECORDS = [
    _record("TEMPERATURE", "INPUT", "MILLIDEGREE_CELSIUS", 1, 3, "board temperature", 39000),
    _record("TEMPERATURE", "CRIT", "MILLIDEGREE_CELSIUS", 1, 3, "board temperature", 80000),
    _record("POWER", "INPUT", "MICROWATT", 1, 3, "total power", 37700000),
    _record("POWER", "MAX", "MICROWATT", 1, 3, "total power", 40000000),
    _record("VOLTAGE", "INPUT", "MILLIVOLT", 1, 3, "core voltage", 744),
]


def _load_metric():
    """Import subcommands/metric.py standalone with a stubbed amdsmi."""
    amdsmi_stub = types.ModuleType("amdsmi")
    amdsmi_stub.amdsmi_exception = types.ModuleType("amdsmi.amdsmi_exception")

    class _AmdSmiLibraryException(Exception):
        def get_error_info(self):
            return "stub error"

    amdsmi_stub.amdsmi_exception.AmdSmiLibraryException = _AmdSmiLibraryException

    interface_stub = types.ModuleType("amdsmi.amdsmi_interface")
    interface_stub.AMDSMI_MAX_RAIL_INDEX = 0xFFFFFFFF
    wrapper_stub = types.ModuleType("amdsmi.amdsmi_wrapper")
    wrapper_stub.AMDSMI_NIC_STAT_SCOPE_DEFAULT = 0
    wrapper_stub.AMDSMI_NIC_STAT_SCOPE_EXTENDED = 1
    interface_stub.amdsmi_wrapper = wrapper_stub
    interface_stub.amdsmi_get_nic_type_label = mock.Mock(return_value="AINIC")
    interface_stub.amdsmi_get_nic_telemetry = mock.Mock(return_value={})
    amdsmi_stub.amdsmi_interface = interface_stub

    with mock.patch.dict(
        sys.modules,
        {
            "amdsmi": amdsmi_stub,
            "amdsmi.amdsmi_exception": amdsmi_stub.amdsmi_exception,
            "amdsmi.amdsmi_interface": interface_stub,
        },
    ):
        spec = importlib.util.spec_from_file_location("metric_under_test", _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, amdsmi_stub


class TestMetricNicSensors(unittest.TestCase):
    def _make_command(self, module, *, is_json=False, is_csv=False):
        class _Command(module.MetricCommands):
            def __init__(self):
                self.device_handles_ainics = [mock.Mock()]
                self.helpers = mock.Mock()
                self.helpers.get_ainic_id_from_device_handle.return_value = 0
                self.helpers.handle_ainics.return_value = (False, self.device_handles_ainics[0])
                self.logger = mock.Mock()
                self.logger.is_json_format.return_value = is_json
                self.logger.is_csv_format.return_value = is_csv
                self.logger.is_human_readable_format.return_value = not (is_json or is_csv)

        return _Command()

    def _run(self, records, *, sensors=True, is_json=False, is_csv=False, error=False):
        module, amdsmi_stub = _load_metric()
        command = self._make_command(module, is_json=is_json, is_csv=is_csv)
        get_sensors = mock.Mock(return_value=records)
        if error:
            get_sensors.side_effect = amdsmi_stub.amdsmi_exception.AmdSmiLibraryException()
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_hwmon_sensors = get_sensors

        command.metric_nic(types.SimpleNamespace(nic=None, sensors=sensors))

        stored = command.logger.store_ainic_output.call_args[0]
        return stored[2], get_sensors

    def test_text_numbers_one_block_per_sensor_in_c_v_w(self):
        values, _ = self._run(_IONIC_RECORDS)
        self.assertEqual(
            values["SENSORS"],
            {
                "SENSOR_0": {
                    "LABEL": "board temperature",
                    "CATEGORY": "TEMPERATURE",
                    "INSTANCE": 3,
                    "CHANNEL": 1,
                    "VALUE": "39.0 \N{DEGREE SIGN}C",
                    "CRIT": "80.0 \N{DEGREE SIGN}C",
                },
                "SENSOR_1": {
                    "LABEL": "total power",
                    "CATEGORY": "POWER",
                    "INSTANCE": 3,
                    "CHANNEL": 1,
                    "VALUE": "37.7 W",
                    "MAX": "40.0 W",
                },
                "SENSOR_2": {
                    "LABEL": "core voltage",
                    "CATEGORY": "VOLTAGE",
                    "INSTANCE": 3,
                    "CHANNEL": 1,
                    "VALUE": "0.744 V",
                },
            },
        )
        self.assertEqual(list(values["SENSORS"]), ["SENSOR_0", "SENSOR_1", "SENSOR_2"])

    def test_shutdown_prints_as_a_threshold_of_its_sensor(self):
        celsius = "MILLIDEGREE_CELSIUS"
        records = [
            _record("TEMPERATURE", "INPUT", celsius, 1, 0, "", 37000),
            _record("TEMPERATURE", "MAX", celsius, 1, 0, "", 95000),
            _record("TEMPERATURE", "SHUTDOWN", celsius, 1, 0, "", 115000),
        ]
        values, _ = self._run(records)
        self.assertEqual(values["SENSORS"]["SENSOR_0"]["SHUTDOWN"], "115.0 \N{DEGREE SIGN}C")
        self.assertEqual(list(values["SENSORS"]), ["SENSOR_0"])

    def test_json_carries_a_value_and_a_unit(self):
        values, _ = self._run(_IONIC_RECORDS[:2], is_json=True)
        self.assertEqual(
            values["SENSORS"]["SENSOR_0"],
            {
                "LABEL": "board temperature",
                "CATEGORY": "TEMPERATURE",
                "INSTANCE": 3,
                "CHANNEL": 1,
                "VALUE": {"value": 39.0, "unit": "C"},
                "CRIT": {"value": 80.0, "unit": "C"},
            },
        )

    def test_json_maps_power_to_w_voltage_to_v_and_keeps_a_negative_temperature(self):
        records = [
            _record("TEMPERATURE", "INPUT", "MILLIDEGREE_CELSIUS", 2, 3, "die", -5500),
            *_IONIC_RECORDS[2:],
        ]
        values, _ = self._run(records, is_json=True)
        sensors = values["SENSORS"]
        self.assertEqual(sensors["SENSOR_0"]["VALUE"], {"value": -5.5, "unit": "C"})
        self.assertEqual(sensors["SENSOR_1"]["VALUE"], {"value": 37.7, "unit": "W"})
        self.assertEqual(sensors["SENSOR_1"]["MAX"], {"value": 40.0, "unit": "W"})
        self.assertEqual(sensors["SENSOR_2"]["VALUE"], {"value": 0.744, "unit": "V"})

    def test_csv_flattens_sensor_blocks_with_plain_numbers(self):
        values, _ = self._run(_IONIC_RECORDS[:2], is_csv=True)
        self.assertEqual(values["SENSOR_0_LABEL"], "board temperature")
        self.assertEqual(values["SENSOR_0_CATEGORY"], "TEMPERATURE")
        self.assertEqual(values["SENSOR_0_INSTANCE"], 3)
        self.assertEqual(values["SENSOR_0_CHANNEL"], 1)
        self.assertEqual(values["SENSOR_0_VALUE"], 39.0)
        self.assertEqual(values["SENSOR_0_CRIT"], 80.0)
        self.assertNotIn("SENSORS", values)

    def test_no_sensors_says_so_in_text_and_is_empty_in_json(self):
        text_values, _ = self._run([])
        self.assertEqual(text_values["SENSORS"], "No sensors found for this NIC")
        json_values, _ = self._run([], is_json=True)
        self.assertEqual(json_values["SENSORS"], {})

    def test_a_library_error_reads_as_no_sensors_in_text_and_json(self):
        text_values, _ = self._run([], error=True)
        self.assertEqual(text_values["SENSORS"], "No sensors found for this NIC")
        json_values, _ = self._run([], error=True, is_json=True)
        self.assertEqual(json_values["SENSORS"], {})

    def test_the_default_output_does_not_read_sensors(self):
        values, get_sensors = self._run(_IONIC_RECORDS, sensors=False)
        self.assertNotIn("SENSORS", values)
        get_sensors.assert_not_called()

    def test_an_unlabelled_record_is_na_and_keeps_its_identity(self):
        record = _record("TEMPERATURE", "INPUT", "MILLIDEGREE_CELSIUS", 2, 4, "", 46000)
        values, _ = self._run([record])
        self.assertEqual(
            values["SENSORS"],
            {
                "SENSOR_0": {
                    "LABEL": "N/A",
                    "CATEGORY": "TEMPERATURE",
                    "INSTANCE": 4,
                    "CHANNEL": 2,
                    "VALUE": "46.0 \N{DEGREE SIGN}C",
                }
            },
        )

    def _assert_blocks(self, sensors, expected):
        self.assertEqual(len(sensors), len(expected))
        for index, (label, channel, instance) in enumerate(expected):
            block = sensors[f"SENSOR_{index}"]
            self.assertEqual(
                (block["LABEL"], block["CHANNEL"], block["INSTANCE"]), (label, channel, instance)
            )

    def test_the_same_label_on_two_channels_gives_two_blocks(self):
        records = [
            _record("TEMPERATURE", "INPUT", "MILLIDEGREE_CELSIUS", 1, 1, "core", 30000),
            _record("TEMPERATURE", "INPUT", "MILLIDEGREE_CELSIUS", 2, 1, "core", 40000),
        ]
        values, _ = self._run(records)
        self._assert_blocks(values["SENSORS"], [("core", 1, 1), ("core", 2, 1)])
        self.assertEqual(values["SENSORS"]["SENSOR_0"]["VALUE"], "30.0 \N{DEGREE SIGN}C")
        self.assertEqual(values["SENSORS"]["SENSOR_1"]["VALUE"], "40.0 \N{DEGREE SIGN}C")

    def test_the_same_label_on_two_hwmon_instances_gives_two_blocks(self):
        records = [
            _record("TEMPERATURE", "INPUT", "MILLIDEGREE_CELSIUS", 1, 1, "core", 30000),
            _record("TEMPERATURE", "INPUT", "MILLIDEGREE_CELSIUS", 1, 2, "core", 40000),
        ]
        values, _ = self._run(records)
        self._assert_blocks(values["SENSORS"], [("core", 1, 1), ("core", 1, 2)])
        self.assertEqual(values["SENSORS"]["SENSOR_0"]["VALUE"], "30.0 \N{DEGREE SIGN}C")
        self.assertEqual(values["SENSORS"]["SENSOR_1"]["VALUE"], "40.0 \N{DEGREE SIGN}C")
