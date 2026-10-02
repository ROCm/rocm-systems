#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""metric --nic --port [--extended] tests: text output trims to the default
scope, JSON always requests the extended scope regardless of --extended, and
a NIC with no ports reports that plainly instead of raising.

These load subcommands/metric.py standalone with a stubbed ``amdsmi`` package
so the control flow is exercised without a built libamd_smi.so or NIC hardware.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

_CLI_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "amdsmi_cli")
)
_MODULE_PATH = os.path.join(_CLI_DIR, "subcommands", "metric.py")


def _load_metric():
    """Import subcommands/metric.py standalone with a stubbed amdsmi.

    Loading by file path skips subcommands/__init__.py (which pulls in every
    sibling command); stubbing amdsmi removes the built-library dependency.
    """
    amdsmi_stub = types.ModuleType("amdsmi")
    amdsmi_stub.amdsmi_exception = types.ModuleType("amdsmi.amdsmi_exception")

    class _AmdSmiLibraryException(Exception):
        def get_error_info(self):
            return "stub error"

    amdsmi_stub.amdsmi_exception.AmdSmiLibraryException = _AmdSmiLibraryException

    interface_stub = types.ModuleType("amdsmi.amdsmi_interface")
    # metric.py does `from amdsmi.amdsmi_interface import AMDSMI_MAX_RAIL_INDEX`
    # at module scope; the stub needs the attribute or exec_module raises
    # ImportError before any test body runs.
    interface_stub.AMDSMI_MAX_RAIL_INDEX = 0xFFFFFFFF
    wrapper_stub = types.ModuleType("amdsmi.amdsmi_wrapper")
    wrapper_stub.AMDSMI_NIC_STAT_SCOPE_DEFAULT = 0
    wrapper_stub.AMDSMI_NIC_STAT_SCOPE_EXTENDED = 1
    interface_stub.amdsmi_wrapper = wrapper_stub
    # metric_nic reads the NIC type label; tests that care override it.
    interface_stub.amdsmi_get_nic_type_label = mock.Mock(return_value="AINIC")
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


class TestMetricNicPortStats(unittest.TestCase):
    def _make_command(self, module, amdsmi_stub, *, is_json=False):
        class _Command(module.MetricCommands):
            def __init__(self):
                self.device_handles_ainics = [mock.Mock()]
                self.helpers = mock.Mock()
                self.helpers.get_ainic_id_from_device_handle.return_value = 0
                self.helpers.handle_ainics.return_value = (False, self.device_handles_ainics[0])
                self.logger = mock.Mock()
                self.logger.is_json_format.return_value = is_json

        command = _Command()
        command._amdsmi_stub = amdsmi_stub
        return command

    def test_text_output_requests_default_scope(self):
        module, amdsmi_stub = _load_metric()
        command = self._make_command(module, amdsmi_stub, is_json=False)
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_telemetry = mock.Mock(return_value={})
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_port_info = mock.Mock(
            return_value={"num_ports": 1, "ports": [{"netdev": "enp1s0"}]}
        )
        get_vendor_stats = mock.Mock(return_value={"tx_packets": 1})
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_vendor_statistics = get_vendor_stats
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_port_statistics = mock.Mock(return_value={})

        args = types.SimpleNamespace(nic=None, port=-1, extended=False)
        command.metric_nic(args)

        get_vendor_stats.assert_called_once()
        _, called_args, _ = get_vendor_stats.mock_calls[0]
        self.assertEqual(called_args[2], 0)  # AMDSMI_NIC_STAT_SCOPE_DEFAULT

    def test_json_always_requests_extended_scope_even_without_flag(self):
        module, amdsmi_stub = _load_metric()
        command = self._make_command(module, amdsmi_stub, is_json=True)
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_telemetry = mock.Mock(return_value={})
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_port_info = mock.Mock(
            return_value={"num_ports": 1, "ports": [{"netdev": "enp1s0"}]}
        )
        get_vendor_stats = mock.Mock(return_value={})
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_vendor_statistics = get_vendor_stats
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_port_statistics = mock.Mock(return_value={})

        args = types.SimpleNamespace(nic=None, port=-1, extended=False)  # no --extended
        command.metric_nic(args)

        _, called_args, _ = get_vendor_stats.mock_calls[0]
        self.assertEqual(called_args[2], 1)  # AMDSMI_NIC_STAT_SCOPE_EXTENDED, forced by JSON

    def test_no_ports_reports_plainly(self):
        module, amdsmi_stub = _load_metric()
        command = self._make_command(module, amdsmi_stub, is_json=False)
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_telemetry = mock.Mock(return_value={})
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_port_info = mock.Mock(
            return_value={"num_ports": 0, "ports": []}
        )

        args = types.SimpleNamespace(nic=None, port=-1, extended=False)
        command.metric_nic(args)

        stored = command.logger.store_ainic_output.call_args[0][2]
        self.assertEqual(stored["PORTS"], "No ports found for this NIC")

    def test_stat_names_are_uppercased(self):
        module, amdsmi_stub = _load_metric()
        command = self._make_command(module, amdsmi_stub, is_json=False)
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_telemetry = mock.Mock(return_value={})
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_port_info = mock.Mock(
            return_value={"num_ports": 1, "ports": [{"netdev": "enp1s0"}]}
        )
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_vendor_statistics = mock.Mock(
            return_value={"tx_packets": 1}
        )
        amdsmi_stub.amdsmi_interface.amdsmi_get_nic_port_statistics = mock.Mock(
            return_value={"rx_bytes": 2}
        )

        args = types.SimpleNamespace(nic=None, port=0, extended=False)
        command.metric_nic(args)

        stored = command.logger.store_ainic_output.call_args[0][2]
        port0 = stored["PORTS"]["PORT_0"]
        self.assertIn("TX_PACKETS", port0["VENDOR_STATISTICS"])
        self.assertIn("RX_BYTES", port0["STATISTICS"])
        self.assertEqual(port0["NETDEV"], "enp1s0")


if __name__ == "__main__":
    unittest.main()
