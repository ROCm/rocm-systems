#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""NIC hwmon sensors: ``amdsmi_get_nic_hwmon_sensors`` makes a count call, then a fetch call.

The shipped ``amdsmi_interface.py`` needs a built ``libamd_smi.so`` to import, so the function is
compiled from the real source and run against a fake wrapper, as in test_nic_telemetry_reporter.py.
"""

import ast
import ctypes
import types
import unittest
from enum import IntEnum
from typing import Any, Dict, List, Tuple

from common.common import amdsmi_interface_path

_INTERFACE_PATH = amdsmi_interface_path(__file__)

_STATUS_SUCCESS = 0
_STATUS_INSUFFICIENT_SIZE = 41
_LABEL_BYTES = 256

_CATEGORY_TEMPERATURE = 3
_CATEGORY_POWER = 4
_CATEGORY_VOLTAGE = 14
_UNIT_MILLIVOLT = 5
_UNIT_MILLIDEGREE_CELSIUS = 15
_UNIT_MICROWATT = 16
_NAME_UNKNOWN = 0
_NAME_INPUT = 1
_NAME_MIN = 2
_NAME_MAX = 3
_NAME_LCRIT = 4
_NAME_CRIT = 5
_NAME_EMERGENCY = 6
_NAME_SHUTDOWN = 7
_NAME_CAP = 8
_UNDEFINED_ENUM_VALUE = 99

# (category, name, unit, channel, instance, label, value)
_Record = Tuple[int, int, int, int, int, bytes, int]
_BOARD_INPUT: _Record = (
    _CATEGORY_TEMPERATURE,
    _NAME_INPUT,
    _UNIT_MILLIDEGREE_CELSIUS,
    1,
    3,
    b"board temperature",
    39000,
)
_BOARD_CRIT: _Record = (
    _CATEGORY_TEMPERATURE,
    _NAME_CRIT,
    _UNIT_MILLIDEGREE_CELSIUS,
    1,
    3,
    b"board temperature",
    80000,
)


class _FakeParameterException(Exception):
    """Stands in for AmdSmiParameterException."""


class _FakeHandle:
    """Stands in for the ctypes amdsmi_processor_handle type."""


class _Sensor(ctypes.Structure):
    _fields_ = [
        ("category", ctypes.c_int),
        ("name", ctypes.c_int),
        ("unit", ctypes.c_int),
        ("channel", ctypes.c_uint32),
        ("instance", ctypes.c_uint32),
        ("label", ctypes.c_char * _LABEL_BYTES),
        ("value", ctypes.c_int64),
    ]


def _fake_check_res(status: int) -> None:
    if status != _STATUS_SUCCESS:
        raise AssertionError(f"unexpected status {status}")


class _FakeLibrary:
    """Stands in for libamd_smi.so: holds the records and notes how each call was made."""

    def __init__(self, records: List[_Record]) -> None:
        self.records = records
        self.count_calls = 0
        self.fetches: List[Tuple[int, int]] = []  # (declared capacity, array length)

    def get_nic_hwmon_sensors(self, handle: object, num_sensors_ref: Any, sensors: Any) -> int:
        num_sensors = num_sensors_ref._obj
        if sensors is None:
            self.count_calls += 1
            num_sensors.value = len(self.records)
            return _STATUS_SUCCESS

        capacity = num_sensors.value
        self.fetches.append((capacity, len(sensors)))
        for index, record in enumerate(self.records[:capacity]):
            sensor = sensors[index]
            (
                sensor.category,
                sensor.name,
                sensor.unit,
                sensor.channel,
                sensor.instance,
                sensor.label,
                sensor.value,
            ) = record
        is_truncated = len(self.records) > capacity
        num_sensors.value = len(self.records)
        return _STATUS_INSUFFICIENT_SIZE if is_truncated else _STATUS_SUCCESS


def _definition_name(node: ast.stmt) -> str:
    """Name bound by a top-level function, class or single-target assignment; "" otherwise."""
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return ""


def _load_definitions(names: List[str], namespace: Dict[str, Any]) -> Dict[str, Any]:
    """Compile the named top-level definitions from the shipped interface source."""
    tree = ast.parse(_INTERFACE_PATH.read_text(encoding="utf-8"))
    wanted = [node for node in tree.body if _definition_name(node) in names]
    missing = set(names) - {_definition_name(node) for node in wanted}
    if missing:
        raise AssertionError(f"not found in amdsmi_interface.py: {sorted(missing)}")
    exec(compile(ast.Module(wanted, []), str(_INTERFACE_PATH), "exec"), namespace)
    return namespace


def _run(
    records: List[_Record], handle: object = None
) -> Tuple[List[Dict[str, Any]], _FakeLibrary]:
    """Run the shipped amdsmi_get_nic_hwmon_sensors against a fake library holding ``records``."""
    library = _FakeLibrary(records)
    wrapper = types.SimpleNamespace(
        AMDSMI_METRIC_CATEGORY_TEMPERATURE=_CATEGORY_TEMPERATURE,
        AMDSMI_METRIC_CATEGORY_POWER=_CATEGORY_POWER,
        AMDSMI_METRIC_CATEGORY_VOLTAGE=_CATEGORY_VOLTAGE,
        AMDSMI_METRIC_UNIT_MILLIVOLT=_UNIT_MILLIVOLT,
        AMDSMI_METRIC_UNIT_MILLIDEGREE_CELSIUS=_UNIT_MILLIDEGREE_CELSIUS,
        AMDSMI_METRIC_UNIT_MICROWATT=_UNIT_MICROWATT,
        AMDSMI_NIC_SENSOR_NAME_UNKNOWN=_NAME_UNKNOWN,
        AMDSMI_NIC_SENSOR_NAME_INPUT=_NAME_INPUT,
        AMDSMI_NIC_SENSOR_NAME_MIN=_NAME_MIN,
        AMDSMI_NIC_SENSOR_NAME_MAX=_NAME_MAX,
        AMDSMI_NIC_SENSOR_NAME_LCRIT=_NAME_LCRIT,
        AMDSMI_NIC_SENSOR_NAME_CRIT=_NAME_CRIT,
        AMDSMI_NIC_SENSOR_NAME_EMERGENCY=_NAME_EMERGENCY,
        AMDSMI_NIC_SENSOR_NAME_SHUTDOWN=_NAME_SHUTDOWN,
        AMDSMI_NIC_SENSOR_NAME_CAP=_NAME_CAP,
        amdsmi_processor_handle=_FakeHandle,
        amdsmi_nic_sensor_t=_Sensor,
        amdsmi_get_nic_hwmon_sensors=library.get_nic_hwmon_sensors,
    )
    namespace = _load_definitions(
        [
            "AmdSmiMetricCategory",
            "AmdSmiMetricUnit",
            "AmdSmiNicSensorName",
            "_nic_sensor_enum_name",
            "amdsmi_get_nic_hwmon_sensors",
        ],
        {
            "amdsmi_wrapper": wrapper,
            "ctypes": ctypes,
            "IntEnum": IntEnum,
            "Dict": Dict,
            "List": List,
            "Any": Any,
            "AmdSmiParameterException": _FakeParameterException,
            "_check_res": _fake_check_res,
        },
    )
    result = namespace["amdsmi_get_nic_hwmon_sensors"](_FakeHandle() if handle is None else handle)
    return result, library


class TestNicHwmonSensors(unittest.TestCase):
    def test_a_nic_without_sensors_returns_an_empty_list_after_the_count_call_only(self) -> None:
        result, library = _run([])
        self.assertEqual(result, [])
        self.assertEqual(library.count_calls, 1)
        self.assertEqual(library.fetches, [])

    def test_the_fetch_array_is_sized_from_the_count(self) -> None:
        _, library = _run([_BOARD_INPUT, _BOARD_CRIT])
        self.assertEqual(library.fetches, [(2, 2)])

    def test_a_record_decodes_to_names_and_native_units(self) -> None:
        result, _ = _run([_BOARD_INPUT])
        self.assertEqual(
            result,
            [
                {
                    "category": "TEMPERATURE",
                    "attribute": "INPUT",
                    "unit": "MILLIDEGREE_CELSIUS",
                    "channel": 1,
                    "instance": 3,
                    "label": "board temperature",
                    "value": 39000,
                }
            ],
        )

    def test_threshold_and_cap_attributes_decode(self) -> None:
        cap: _Record = (_CATEGORY_POWER, _NAME_CAP, _UNIT_MICROWATT, 1, 0, b"", 1400000000)
        result, _ = _run([_BOARD_CRIT, cap])
        self.assertEqual([record["attribute"] for record in result], ["CRIT", "CAP"])
        self.assertEqual(result[1]["unit"], "MICROWATT")

    def test_shutdown_threshold_decodes(self) -> None:
        shutdown: _Record = (
            _CATEGORY_TEMPERATURE,
            _NAME_SHUTDOWN,
            _UNIT_MILLIDEGREE_CELSIUS,
            1,
            0,
            b"",
            115000,
        )
        result, _ = _run([shutdown])
        self.assertEqual(result[0]["attribute"], "SHUTDOWN")

    def test_a_label_with_a_stray_byte_is_replaced_not_raised(self) -> None:
        record: _Record = (_CATEGORY_VOLTAGE, _NAME_INPUT, _UNIT_MILLIVOLT, 0, 0, b"ab\xffcd", 744)
        result, _ = _run([record])
        self.assertEqual(result[0]["label"], "ab\N{REPLACEMENT CHARACTER}cd")

    def test_an_enum_value_this_module_does_not_know_reads_unknown(self) -> None:
        record: _Record = (_UNDEFINED_ENUM_VALUE, _NAME_INPUT, _UNIT_MILLIVOLT, 0, 0, b"", 1)
        result, _ = _run([record])
        self.assertEqual(result[0]["category"], "UNKNOWN")

    def test_an_object_that_is_not_a_handle_is_rejected(self) -> None:
        with self.assertRaises(_FakeParameterException):
            _run([], handle=object())


if __name__ == "__main__":
    unittest.main()
