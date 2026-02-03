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
Data classes for counter records.

This module provides Python data classes for representing hardware
counter measurements collected during profiling.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CounterRecord:
    """
    A single hardware counter measurement.

    Attributes
    ----------
    dispatch_id : int
        Unique identifier for the kernel dispatch.
    counter_id : int
        Hardware counter identifier.
    counter_name : str
        Human-readable counter name.
    kernel_name : str
        Name of the kernel this record is associated with.
    value : float
        Counter value (typically a double).
    agent_id : int
        GPU agent (device) identifier.
    dimensions : list of tuple
        List of (dimension_name, position) tuples for multi-dimensional
        counters.
    """

    dispatch_id: int
    counter_id: int
    counter_name: str
    kernel_name: str
    value: float
    agent_id: int
    dimensions: List[Tuple[str, int]]

    @classmethod
    def _from_native(cls, native) -> "CounterRecord":
        """
        Convert from native C++ CounterRecord.

        Parameters
        ----------
        native : libpyrocprofiler.CounterRecord
            Native C++ counter record object.

        Returns
        -------
        CounterRecord
            Python CounterRecord dataclass instance.
        """
        return cls(
            dispatch_id=native.dispatch_id,
            counter_id=native.counter_id,
            counter_name=native.counter_name,
            kernel_name=native.kernel_name,
            value=native.value,
            agent_id=native.agent_id,
            dimensions=list(native.dimensions),
        )

    def __repr__(self) -> str:
        return (
            f"CounterRecord(kernel='{self.kernel_name}', "
            f"counter='{self.counter_name}', value={self.value})"
        )
