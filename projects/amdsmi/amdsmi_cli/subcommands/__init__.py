#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from subcommands.bad_pages import BadPagesCommands
from subcommands.default import DefaultCommands
from subcommands.event import EventCommands
from subcommands.fabric import FabricCommands
from subcommands.firmware import FirmwareCommands
from subcommands.list_devices import ListDevicesCommands
from subcommands.metric import MetricCommands
from subcommands.monitor import MonitorCommands
from subcommands.node import NodeCommands
from subcommands.partition import PartitionCommands
from subcommands.process import ProcessCommands
from subcommands.ras import RasCommands
from subcommands.reset import ResetCommands
from subcommands.set_value import SetValueCommands
from subcommands.static import StaticCommands
from subcommands.topology import TopologyCommands
from subcommands.version import VersionCommands
from subcommands.xgmi import XgmiCommands

__all__ = [
    "VersionCommands",
    "ListDevicesCommands",
    "StaticCommands",
    "FirmwareCommands",
    "BadPagesCommands",
    "MetricCommands",
    "ProcessCommands",
    "EventCommands",
    "FabricCommands",
    "TopologyCommands",
    "SetValueCommands",
    "ResetCommands",
    "MonitorCommands",
    "XgmiCommands",
    "PartitionCommands",
    "RasCommands",
    "NodeCommands",
    "DefaultCommands",
]
