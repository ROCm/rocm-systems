#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``amd-smi monitor`` fields that gpu_metrics leaves unset.

Some firmware leaves current_gfxclk, current_uclk or the temperatures unset in
gpu_metrics while pp_dpm_* and hwmon still report them, so monitor falls back to
those. Fields with no source stay N/A but must keep their CSV column.
"""

import argparse
import os
import types
import unittest

from common.common import cli_search_order, fake_module, find_cli_dir, load_cli_module, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
MONITOR_PATH = os.path.join(_CLI_DIR, "subcommands", "monitor.py") if _CLI_DIR else None

_UINT32_MAX = 0xFFFFFFFF


class _FakeLibraryException(Exception):
    def get_error_info(self):
        return "AMDSMI_STATUS_NOT_SUPPORTED"


_CLK = types.SimpleNamespace(
    GFX=types.SimpleNamespace(name="GFX"), MEM=types.SimpleNamespace(name="MEM")
)
_TEMP = types.SimpleNamespace(
    HOTSPOT=types.SimpleNamespace(name="HOTSPOT"), VRAM=types.SimpleNamespace(name="VRAM")
)


class _FakeHelpersClass:
    SI_Unit = types.SimpleNamespace(MICRO=0.000001)


class _FakeHelpers:
    def is_brcm_nic_initialized(self):
        return False

    def is_brcm_switch_initialized(self):
        return False

    def is_virtual_os(self):
        return False

    def check_required_groups(self):
        pass

    def get_gpu_id_from_device_handle(self, _handle):
        return 0

    def _get_metric_version_and_partition_info(self, *_args):
        return {"partition_id": 0, "num_partition": "N/A"}

    def convert_SI_unit(self, val, unit_in):
        return int(float(val) * unit_in)


class _FakeLogger:
    destination = "stdout"

    def __init__(self):
        self.table_header = ""
        self.values = None

    def is_human_readable_format(self):
        return False

    def is_json_format(self):
        return False

    def is_csv_format(self):
        return True

    def store_output(self, _gpu, key, value):
        if key == "values":
            self.values = value

    def print_output(self, *_args, **_kwargs):
        pass


def _metrics(**overrides):
    metrics = {
        "current_socket_power": "N/A",
        "average_socket_power": "N/A",
        "temperature_hotspot": "N/A",
        "temperature_mem": "N/A",
        "current_gfxclk": "N/A",
        "current_uclk": "N/A",
        "average_gfx_activity": 0,
        "average_umc_activity": 0,
    }
    metrics.update(overrides)
    return metrics


def _raise(*_args):
    raise _FakeLibraryException()


class TestMonitorFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        interface = fake_module(
            "amdsmi.amdsmi_interface",
            AmdSmiClkType=_CLK,
            AmdSmiTemperatureType=_TEMP,
            AmdSmiTemperatureMetric=types.SimpleNamespace(CURRENT="CURRENT"),
            _NA_amdsmi_get_gpu_metrics_info=_metrics,
        )
        exception = fake_module(
            "amdsmi.amdsmi_exception", AmdSmiLibraryException=_FakeLibraryException
        )
        stub_modules(
            cls,
            {
                "amdsmi": fake_module(
                    "amdsmi", amdsmi_interface=interface, amdsmi_exception=exception
                ),
                "amdsmi.amdsmi_interface": interface,
                "amdsmi.amdsmi_exception": exception,
                "amdsmi_helpers": fake_module("amdsmi_helpers", AMDSMIHelpers=_FakeHelpersClass),
            },
        )
        cls.interface = interface
        cls.module = load_cli_module("monitor_under_test", MONITOR_PATH)

    def setUp(self):
        self.calls = []
        self._set(
            amdsmi_get_gpu_metrics_info=lambda _h: _metrics(),
            amdsmi_get_clk_freq=self._clk_freq,
            amdsmi_get_temp_metric=self._temp,
        )

    def _set(self, **attrs):
        for name, value in attrs.items():
            setattr(self.interface, name, value)

    def _clk_freq(self, _h, clk_type):
        self.calls.append(clk_type.name)
        if clk_type.name == "GFX":
            return {"current": 1, "frequency": [500000000, 1000000000, 3000000000]}
        return {"current": 1, "frequency": [234000000, 500000000, 1125000000]}

    def _temp(self, _h, sensor, _metric):
        self.calls.append(sensor.name)
        return {"HOTSPOT": 22, "VRAM": 26}[sensor.name]

    def _run(self):
        args = argparse.Namespace(
            gpu=object(),
            watch=False,
            watch_time=None,
            iterations=None,
            loglevel="INFO",
            power_usage=False,
            temperature=True,
            base_board_temps=False,
            gpu_board_temps=False,
            gfx=True,
            mem=True,
            encoder=False,
            decoder=False,
            ecc=False,
            vram_usage=False,
            pcie=False,
            process=False,
            violation=False,
        )
        commands = object.__new__(self.module.MonitorCommands)
        commands.helpers = _FakeHelpers()
        commands.logger = _FakeLogger()
        commands.group_check_printed = True
        commands.monitor(args)
        return commands.logger.values

    def test_unset_metrics_fall_back_to_dpm_and_hwmon(self):
        values = self._run()
        self.assertEqual(values["gfx_clk"], 1000)
        self.assertEqual(values["mem_clock"], 500)
        self.assertEqual(values["hotspot_temperature"], 22)
        self.assertEqual(values["memory_temperature"], 26)

    def test_metrics_values_win_over_fallbacks(self):
        self._set(
            amdsmi_get_gpu_metrics_info=lambda _h: _metrics(
                current_gfxclk=2100, current_uclk=900, temperature_hotspot=40, temperature_mem=35
            )
        )
        values = self._run()
        self.assertEqual(
            (values["gfx_clk"], values["mem_clock"]), (2100, 900), "gpu_metrics clocks replaced"
        )
        self.assertEqual((values["hotspot_temperature"], values["memory_temperature"]), (40, 35))
        self.assertEqual(self.calls, [])

    def test_unavailable_fallbacks_stay_na(self):
        self._set(amdsmi_get_clk_freq=_raise, amdsmi_get_temp_metric=_raise)
        values = self._run()
        for key in ("gfx_clk", "mem_clock", "hotspot_temperature", "memory_temperature"):
            self.assertEqual(values[key], "N/A", key)

    def test_na_utilization_keeps_its_csv_column(self):
        self._set(
            amdsmi_get_gpu_metrics_info=lambda _h: _metrics(
                average_gfx_activity="N/A", average_umc_activity="N/A"
            )
        )
        values = self._run()
        self.assertEqual((values.get("gfx"), values.get("mem")), ("N/A", "N/A"))

    def test_no_marked_dpm_level_is_na(self):
        self._set(
            amdsmi_get_clk_freq=lambda _h, _t: {"current": _UINT32_MAX, "frequency": [500000000]}
        )
        values = self._run()
        self.assertEqual((values["gfx_clk"], values["mem_clock"]), ("N/A", "N/A"))
