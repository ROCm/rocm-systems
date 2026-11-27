# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_gfx1200.py
#
# Benchmarking class for gfx1200 products: Navi 44 (RDNA4).
#
# -----------------------------------------------------------------------------

from . import benchmark_gfx1201


# =============================================================================
# Bench_gfx1200 Class
# =============================================================================
class Bench_gfx1200(benchmark_gfx1201.Bench_gfx1201):
    """Navi 44 runs the same RDNA4 benchmark kernels as Navi 48."""
