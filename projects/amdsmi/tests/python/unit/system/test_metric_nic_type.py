#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""metric --nic leads each NIC's block with its type.

A UALoE endpoint has no telemetry source, so its block is all N/A; the type line says why, so the
block is not read as a failure. ``metric.py`` is loaded in isolation with a stubbed ``amdsmi``.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import Dict, Optional
from unittest import mock

from common.common import cli_search_order, find_cli_dir

# cli_search_order() decides whether the install or this checkout wins; None means no CLI found.
_CLI_DIR = find_cli_dir(*cli_search_order(str(Path(__file__).resolve().parent)))
_MODULE_PATH = Path(_CLI_DIR) / "subcommands" / "metric.py" if _CLI_DIR else None

if _MODULE_PATH is None or not _MODULE_PATH.is_file():
    raise unittest.SkipTest(f"amd-smi CLI metric.py not found (looked in {_CLI_DIR})")


class _LibraryError(Exception):
    def get_error_info(self) -> str:
        return "stub error"


def _load_metric() -> types.SimpleNamespace:
    """Import subcommands/metric.py standalone; return the module and the stubbed amdsmi."""
    amdsmi_stub = types.ModuleType("amdsmi")
    amdsmi_stub.amdsmi_exception = types.ModuleType("amdsmi.amdsmi_exception")
    amdsmi_stub.amdsmi_exception.AmdSmiLibraryException = _LibraryError

    interface_stub = types.ModuleType("amdsmi.amdsmi_interface")
    # metric.py imports this name at module scope.
    interface_stub.AMDSMI_MAX_RAIL_INDEX = 0xFFFFFFFF
    amdsmi_stub.amdsmi_interface = interface_stub

    stubs = {
        "amdsmi": amdsmi_stub,
        "amdsmi.amdsmi_exception": amdsmi_stub.amdsmi_exception,
        "amdsmi.amdsmi_interface": interface_stub,
    }
    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location("metric_type_under_test", _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return types.SimpleNamespace(module=module, interface=interface_stub)


class TestMetricNicType(unittest.TestCase):
    def _run_metric_nic(self, label: Optional[str], telemetry: Dict[str, object]) -> Dict:
        """Run metric_nic and return the values dict it stored; ``None`` fails the type read."""
        loaded = _load_metric()
        self._interface = loaded.interface
        handle = mock.Mock()

        class _Command(loaded.module.MetricCommands):
            def __init__(self) -> None:
                self.device_handles_ainics = [handle]
                self.helpers = mock.Mock()
                self.helpers.get_ainic_id_from_device_handle.return_value = 0
                self.helpers.handle_ainics.return_value = (False, handle)
                self.logger = mock.Mock()
                self.logger.is_json_format.return_value = False

        loaded.interface.amdsmi_get_nic_telemetry = mock.Mock(return_value=telemetry)
        if label is None:
            loaded.interface.amdsmi_get_nic_type_label = mock.Mock(side_effect=_LibraryError())
        else:
            loaded.interface.amdsmi_get_nic_type_label = mock.Mock(return_value=label)
        # The six-getter summary must stay out of the metric path.
        loaded.interface.amdsmi_get_ainic_info = mock.Mock()

        command = _Command()
        command.metric_nic(types.SimpleNamespace(nic=None, port=None))
        return command.logger.store_ainic_output.call_args[0][2]

    def test_type_is_stored_with_the_telemetry(self) -> None:
        stored = self._run_metric_nic("UALoE", {"health": {"state": "UNSUPPORTED"}})
        self.assertEqual(stored["type"], "UALoE")
        self.assertEqual(stored["health"], {"state": "UNSUPPORTED"})

    def test_type_leads_the_block(self) -> None:
        stored = self._run_metric_nic("AINIC", {"temperature": {}, "health": {}})
        self.assertEqual(list(stored)[0], "type")

    def test_a_type_read_failure_reports_na_and_keeps_the_telemetry(self) -> None:
        stored = self._run_metric_nic(None, {"temperature": {"asic_temp_c": 31}})
        self.assertEqual(stored["type"], "N/A")
        self.assertEqual(stored["temperature"], {"asic_temp_c": 31})

    def test_the_six_getter_summary_is_not_read(self) -> None:
        # An unrelated getter failing there (port, rdma) must not turn the type into N/A.
        self._run_metric_nic("AINIC", {})
        self._interface.amdsmi_get_ainic_info.assert_not_called()
