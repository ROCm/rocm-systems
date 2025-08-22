#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################


import math
from .importer import RocpdImportData

__all__ = ["StdDevSamp", "create", "supported_functions"]


def supported_functions():
    # tuple of supported functions
    return ("STDDEV_SAMP",)


# Aggregate class for use with sqlite3
class StdDevSamp:

    def __init__(self):
        self.values = []

    def step(self, value):
        if value is not None:
            self.values.append(value)

    def finalize(self):
        return StdDevSamp.stddev_samp(self.values)

    # Standard deviation function (sample stddev, i.e., n-1 in denominator)
    @staticmethod
    def stddev_samp(values):
        n = len(values)
        if n < 2:
            return None
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        return math.sqrt(variance)


def create(connection: RocpdImportData, functions=list(supported_functions())):

    unsupported = [itr for itr in functions if itr not in supported_functions()]

    if unsupported:
        raise RuntimeError(
            f"rocpd does not support creating SQLite3 functions for {unsupported}"
        )

    if "STDDEV_SAMP" in functions:
        connection.create_aggregate("STDDEV_SAMP", 1, StdDevSamp)
