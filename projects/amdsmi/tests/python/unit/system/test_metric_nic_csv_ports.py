#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""``metric --nic --port --csv`` keeps every port's counters.

The CSV logger flattens nested dicts without a parent prefix, so two ports that report the same
counter name overwrote each other and the row carried only the last port's value. The command and
the real logger run end to end here; stdout is parsed as CSV.
"""

import contextlib
import csv
import io
import sys
import types
import unittest
from pathlib import Path
from typing import Dict
from unittest import mock

from common.common import cli_search_order, fake_module, find_cli_dir, load_cli_module

_CLI_DIR = find_cli_dir(*cli_search_order(str(Path(__file__).resolve().parent)))


class _LibraryError(Exception):
    def get_error_info(self) -> str:
        return "stub error"


def _load_modules() -> types.SimpleNamespace:
    """Load the real ``metric.py`` and ``amdsmi_logger.py`` against a stubbed ``amdsmi``."""
    wrapper = fake_module(
        "amdsmi.amdsmi_wrapper", AMDSMI_NIC_STAT_SCOPE_DEFAULT=0, AMDSMI_NIC_STAT_SCOPE_EXTENDED=1
    )
    interface = fake_module(
        "amdsmi.amdsmi_interface",
        AMDSMI_MAX_RAIL_INDEX=0xFFFFFFFF,
        amdsmi_wrapper=wrapper,
        amdsmi_get_nic_type_label=mock.Mock(return_value="AINIC"),
        amdsmi_get_nic_telemetry=mock.Mock(return_value={}),
    )
    exception = fake_module("amdsmi.amdsmi_exception", AmdSmiLibraryException=_LibraryError)
    package = fake_module("amdsmi", amdsmi_exception=exception, amdsmi_interface=interface)
    stubs = {
        "amdsmi": package,
        "amdsmi.amdsmi_exception": exception,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi_helpers": fake_module(
            "amdsmi_helpers", AMDSMIHelpers=type("AMDSMIHelpers", (), {})
        ),
    }
    with mock.patch.dict(sys.modules, stubs):
        metric = load_cli_module(
            "csv_ports_metric", str(Path(_CLI_DIR) / "subcommands" / "metric.py")
        )
        logger = load_cli_module("csv_ports_logger", str(Path(_CLI_DIR) / "amdsmi_logger.py"))
    return types.SimpleNamespace(metric=metric, logger=logger, interface=interface)


def _csv_row(args: types.SimpleNamespace, interface_setup: Dict[str, object]) -> Dict[str, str]:
    """Run ``metric_nic`` with the real logger in CSV mode and return the one printed row."""
    if not _CLI_DIR:
        raise unittest.SkipTest("amd-smi CLI directory not found")
    loaded = _load_modules()
    for name, value in interface_setup.items():
        setattr(loaded.interface, name, value)

    logger = loaded.logger.AMDSMILogger.__new__(loaded.logger.AMDSMILogger)
    logger.format = "csv"
    logger.destination = "stdout"
    logger.output = {}
    logger.multiple_device_output = []
    logger.helpers = mock.Mock()
    logger.helpers.get_ainic_id_from_device_handle.return_value = 0

    handle = mock.Mock()

    class _Command(loaded.metric.MetricCommands):
        def __init__(self) -> None:
            self.device_handles_ainics = [handle]
            self.helpers = mock.Mock()
            self.helpers.get_ainic_id_from_device_handle.return_value = 0
            self.helpers.handle_ainics.return_value = (False, handle)
            self.logger = logger

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _Command().metric_nic(args)
    rows = list(csv.DictReader(io.StringIO(out.getvalue())))
    if len(rows) != 1:
        raise AssertionError(f"expected one CSV row, got {rows!r}")
    return rows[0]


class TestMetricNicCsvPorts(unittest.TestCase):
    def test_two_ports_with_the_same_counter_name_keep_both_values(self) -> None:
        stats_by_port = {0: {"rx_packets": 10}, 1: {"rx_packets": 20}}
        row = _csv_row(
            types.SimpleNamespace(nic=None, port=-1, extended=False),
            {
                "amdsmi_get_nic_port_info": mock.Mock(
                    return_value={"num_ports": 2, "ports": [{"netdev": "eth0"}, {"netdev": "eth1"}]}
                ),
                "amdsmi_get_nic_vendor_statistics": mock.Mock(
                    side_effect=lambda handle, index, scope: stats_by_port[index]
                ),
                "amdsmi_get_nic_port_statistics": mock.Mock(return_value={}),
            },
        )
        self.assertEqual(row["PORT_0_VENDOR_STATISTICS_RX_PACKETS"], "10")
        self.assertEqual(row["PORT_1_VENDOR_STATISTICS_RX_PACKETS"], "20")
        self.assertEqual(row["PORT_0_NETDEV"], "eth0")
        self.assertEqual(row["PORT_1_NETDEV"], "eth1")

    def test_a_vendor_and_a_standard_counter_of_one_port_stay_apart(self) -> None:
        row = _csv_row(
            types.SimpleNamespace(nic=None, port=-1, extended=False),
            {
                "amdsmi_get_nic_port_info": mock.Mock(
                    return_value={"num_ports": 1, "ports": [{"netdev": "eth0"}]}
                ),
                "amdsmi_get_nic_vendor_statistics": mock.Mock(return_value={"rx_packets": 1}),
                "amdsmi_get_nic_port_statistics": mock.Mock(return_value={"rx_packets": 2}),
            },
        )
        self.assertEqual(row["PORT_0_VENDOR_STATISTICS_RX_PACKETS"], "1")
        self.assertEqual(row["PORT_0_STATISTICS_RX_PACKETS"], "2")

    def test_rdma_ports_do_not_collide_with_the_netdev_ports(self) -> None:
        rdma_devices = {
            "RDMA_DEVICE_0": {
                "NAME": "ionic_0",
                "PORT_0": {"NETDEV": "eth0", "PORT_NUM": 1},
                "PORT_1": {"NETDEV": "eth1", "PORT_NUM": 1},
            }
        }
        row = _csv_row(
            types.SimpleNamespace(nic=None, port=-1, rdma=-1, extended=False),
            {
                "amdsmi_get_nic_port_info": mock.Mock(
                    return_value={"num_ports": 1, "ports": [{"netdev": "eth0"}]}
                ),
                "amdsmi_get_nic_vendor_statistics": mock.Mock(return_value={}),
                "amdsmi_get_nic_port_statistics": mock.Mock(return_value={"rx_packets": 5}),
                "amdsmi_get_ainic_info": mock.Mock(return_value={"RDMA_DEVICES": rdma_devices}),
                "amdsmi_get_nic_rdma_port_statistics": mock.Mock(
                    side_effect=lambda handle, index: {"tx_pkts": 100 + index}
                ),
            },
        )
        self.assertEqual(row["PORT_0_STATISTICS_RX_PACKETS"], "5")
        self.assertEqual(row["RDMA_PORT_0_STATISTICS_TX_PKTS"], "100")
        self.assertEqual(row["RDMA_PORT_1_STATISTICS_TX_PKTS"], "101")
