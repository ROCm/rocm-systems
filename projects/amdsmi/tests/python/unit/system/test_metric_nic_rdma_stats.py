#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""metric --nic --rdma prints RDMA hardware counters per RDMA port.

The port block names the RDMA device and netdev so an index maps to ``rdma statistic show link
DEV/PORT``. ``metric.py`` is loaded in isolation with a stubbed ``amdsmi``.
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


def _ainic_info(devices):
    """RDMA_DEVICES as amdsmi_get_ainic_info builds it: devices in order, each with PORT_<k>."""
    rdma_devices = {}
    for index, (name, netdev, num_ports) in enumerate(devices):
        entry = {"NAME": name, "NODE_GUID": "g", "NODE_TYPE": "ca", "FW_VER": "1.0"}
        for port in range(num_ports):
            entry[f"PORT_{port}"] = {"NETDEV": netdev, "PORT_NUM": port + 1, "STATE": "ACTIVE"}
        rdma_devices[f"RDMA_DEVICE_{index}"] = entry
    return {"RDMA_DEVICES": rdma_devices}


class TestMetricNicRdma(unittest.TestCase):
    def _command(self, *, is_json=False, devices=(("ionic_0", "enP1p3s0f3", 1),)):
        module, amdsmi_stub = _load_metric()
        interface = amdsmi_stub.amdsmi_interface

        class _Command(module.MetricCommands):
            def __init__(self):
                self.device_handles_ainics = [mock.Mock()]
                self.helpers = mock.Mock()
                self.helpers.get_ainic_id_from_device_handle.return_value = 0
                self.helpers.handle_ainics.return_value = (False, self.device_handles_ainics[0])
                self.logger = mock.Mock()
                self.logger.is_json_format.return_value = is_json

        interface.amdsmi_get_nic_telemetry = mock.Mock(return_value={})
        # Like the real function: RDMA_DEVICES is only in the detail view.
        interface.amdsmi_get_ainic_info = mock.Mock(
            side_effect=lambda handle, detail=False: (
                _ainic_info(devices) if detail else {"bdf": "0000:01:00.0"}
            )
        )
        interface.amdsmi_get_nic_rdma_port_statistics = mock.Mock(
            side_effect=lambda handle, index: {"rx_rdma_ucast_pkts": index + 10}
        )
        command = _Command()
        # The exception class metric.py caught at import: the stub's, not the installed one.
        command.library_error = amdsmi_stub.amdsmi_exception.AmdSmiLibraryException
        return command, interface

    @staticmethod
    def _stored(command):
        return command.logger.store_ainic_output.call_args[0][2]

    def test_bare_rdma_lists_every_port_with_device_netdev_and_uppercased_counters(self):
        command, _ = self._command(
            devices=(("ionic_0", "enP1p3s0f3", 1), ("ionic_1", "enP2p3s0f3", 1))
        )
        command.metric_nic(types.SimpleNamespace(nic=None, port=None, extended=False, rdma=-1))

        self.assertEqual(
            self._stored(command)["RDMA_PORTS"],
            {
                "PORT_0": {
                    "RDMA_DEVICE": "ionic_0",
                    "PORT_NUM": 1,
                    "NETDEV": "enP1p3s0f3",
                    "STATISTICS": {"RX_RDMA_UCAST_PKTS": 10},
                },
                "PORT_1": {
                    "RDMA_DEVICE": "ionic_1",
                    "PORT_NUM": 1,
                    "NETDEV": "enP2p3s0f3",
                    "STATISTICS": {"RX_RDMA_UCAST_PKTS": 11},
                },
            },
        )

    def test_index_selects_only_that_port(self):
        command, interface = self._command(
            devices=(("ionic_0", "enP1p3s0f3", 1), ("ionic_1", "enP2p3s0f3", 1))
        )
        command.metric_nic(types.SimpleNamespace(nic=None, port=None, extended=False, rdma=1))

        self.assertEqual(list(self._stored(command)["RDMA_PORTS"]), ["PORT_1"])
        self.assertEqual(interface.amdsmi_get_nic_rdma_port_statistics.call_count, 1)

    def test_index_counts_ports_across_devices(self):
        command, _ = self._command(devices=(("ionic_0", "enP1p3s0f3", 2),))
        command.metric_nic(types.SimpleNamespace(nic=None, port=None, extended=False, rdma=1))

        port = self._stored(command)["RDMA_PORTS"]["PORT_1"]
        self.assertEqual((port["RDMA_DEVICE"], port["PORT_NUM"]), ("ionic_0", 2))

    def test_out_of_range_index_reports_no_ports(self):
        command, _ = self._command()
        command.metric_nic(types.SimpleNamespace(nic=None, port=None, extended=False, rdma=7))

        self.assertEqual(self._stored(command)["RDMA_PORTS"], "No RDMA ports found for this NIC")

    def test_negative_index_other_than_the_bare_marker_does_not_wrap_around(self):
        command, interface = self._command(
            devices=(("ionic_0", "enP1p3s0f3", 1), ("ionic_1", "enP2p3s0f3", 1))
        )
        command.metric_nic(types.SimpleNamespace(nic=None, port=None, extended=False, rdma=-2))

        self.assertEqual(self._stored(command)["RDMA_PORTS"], "No RDMA ports found for this NIC")
        interface.amdsmi_get_nic_rdma_port_statistics.assert_not_called()

    def test_nic_without_rdma_devices_reports_plainly_in_text(self):
        command, _ = self._command(devices=())
        command.metric_nic(types.SimpleNamespace(nic=None, port=None, extended=False, rdma=-1))

        self.assertEqual(self._stored(command)["RDMA_PORTS"], "No RDMA ports found for this NIC")

    def test_nic_without_rdma_devices_is_an_empty_object_in_json(self):
        command, _ = self._command(is_json=True, devices=())
        command.metric_nic(types.SimpleNamespace(nic=None, port=None, extended=False, rdma=-1))

        self.assertEqual(self._stored(command)["RDMA_PORTS"], {})

    def test_rdma_does_not_require_port_and_adds_no_ports_block(self):
        command, _ = self._command()
        command.metric_nic(types.SimpleNamespace(nic=None, port=None, extended=False, rdma=-1))

        self.assertNotIn("PORTS", self._stored(command))

    def test_without_rdma_flag_no_rdma_block_and_no_library_call(self):
        command, interface = self._command()
        command.metric_nic(types.SimpleNamespace(nic=None, port=None, extended=False))

        self.assertNotIn("RDMA_PORTS", self._stored(command))
        interface.amdsmi_get_nic_rdma_port_statistics.assert_not_called()

    def test_library_error_keeps_the_port_with_empty_counters(self):
        command, interface = self._command()
        interface.amdsmi_get_nic_rdma_port_statistics.side_effect = command.library_error("boom")
        command.metric_nic(types.SimpleNamespace(nic=None, port=None, extended=False, rdma=-1))

        port = self._stored(command)["RDMA_PORTS"]["PORT_0"]
        self.assertEqual(port["STATISTICS"], {})
        self.assertEqual(port["RDMA_DEVICE"], "ionic_0")


if __name__ == "__main__":
    unittest.main()
