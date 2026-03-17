# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_base_gfx9.py
#
# Benchmarking base class for all gfx9-based products.
#
# -----------------------------------------------------------------------------

from .. import benchmark_base


class Bench_gfx9(benchmark_base.Bench_base):
    def __init__(self, device_ids: list) -> None:
        super().__init__(device_ids)
