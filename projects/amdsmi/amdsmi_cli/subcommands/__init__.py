#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
