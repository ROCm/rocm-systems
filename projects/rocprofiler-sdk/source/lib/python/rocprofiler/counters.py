# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Counter discovery and enumeration.

This module provides functions for discovering available hardware
counters and GPU devices on the system.
"""

from dataclasses import dataclass
from typing import List, Optional
from . import libpyrocprofiler


@dataclass
class CounterInfo:
    """
    Information about an available hardware counter.

    Attributes
    ----------
    id : int
        Counter identifier.
    name : str
        Counter name (e.g., "SQ_WAVES").
    description : str
        Human-readable description of what the counter measures.
    block : str
        Hardware block the counter belongs to.
    expression : str
        Expression for derived counters.
    is_constant : bool
        True if the counter value is constant.
    is_derived : bool
        True if the counter is derived from other counters.
    """

    id: int
    name: str
    description: str
    block: str
    expression: str
    is_constant: bool
    is_derived: bool

    @classmethod
    def _from_native(cls, native) -> "CounterInfo":
        """Convert from native C++ CounterInfo."""
        return cls(
            id=native.id,
            name=native.name,
            description=native.description,
            block=native.block,
            expression=native.expression,
            is_constant=native.is_constant,
            is_derived=native.is_derived,
        )

    def __repr__(self) -> str:
        return f"CounterInfo(name='{self.name}')"


@dataclass
class GPUAgent:
    """
    Information about a GPU device.

    Attributes
    ----------
    id : int
        Agent identifier.
    name : str
        Agent name.
    product_name : str
        Product name (e.g., "AMD Instinct MI300X").
    device_index : int
        Logical device index.
    gfx_version : int
        GFX version (e.g., 90a for gfx90a).
    """

    id: int
    name: str
    product_name: str
    device_index: int
    gfx_version: int

    @classmethod
    def _from_native(cls, native) -> "GPUAgent":
        """Convert from native C++ AgentInfo."""
        return cls(
            id=native.id,
            name=native.name,
            product_name=native.product_name,
            device_index=native.device_index,
            gfx_version=native.gfx_version,
        )

    def __repr__(self) -> str:
        return f"GPUAgent(name='{self.name}', device_index={self.device_index})"


def available_counters(device_id: Optional[int] = None) -> List[CounterInfo]:
    """
    Get list of available hardware counters.

    Parameters
    ----------
    device_id : int, optional
        Device ID to query. If None, returns counters for all devices.

    Returns
    -------
    list of CounterInfo
        List of CounterInfo objects describing available counters.

    Example
    -------
    >>> counters = rocprofiler.available_counters()
    >>> for c in counters:
    ...     print(f"{c.name}: {c.description}")
    """
    native_counters = libpyrocprofiler.get_available_counters(device_id)
    return [CounterInfo._from_native(c) for c in native_counters]


def gpu_agents() -> List[GPUAgent]:
    """
    Get list of available GPU agents (devices).

    Returns
    -------
    list of GPUAgent
        List of GPUAgent objects describing available GPU devices.

    Example
    -------
    >>> agents = rocprofiler.gpu_agents()
    >>> for agent in agents:
    ...     print(f"Device {agent.device_index}: {agent.product_name}")
    """
    native_agents = libpyrocprofiler.get_gpu_agents()
    return [GPUAgent._from_native(a) for a in native_agents]
