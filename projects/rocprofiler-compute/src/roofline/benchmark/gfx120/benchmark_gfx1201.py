# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -----------------------------------------------------------------------------
# benchmark_gfx1201.py
#
# Benchmarking class for gfx1201 products: Navi 48 (RDNA4).
# -----------------------------------------------------------------------------

from functools import partial

from . import benchmark_gfx120_base
from .benchmark_gfx120_base import (
    WMMA_ACC,
    WMMA_FLOPS_PER_INSTRUCTION,
    ResidentReadRoof,
    dram_pin_bandwidth_gbps,
    percent_of_ceiling,
)

# Infinity Cache size used only when AMD-SMI reports no cache level 3. 64 MB
# is the largest Navi 48 Infinity Cache. Half of it, the streamed set, still
# fits a 48 MB Infinity Cache.
INFINITY_CACHE_BYTES = 64 * 1024 * 1024

# Half the reported Infinity Cache. The stream is large enough to miss GL2 and
# hit the Infinity Cache.
MALL_WORKING_SET_DIVISOR = 2

# Navi 48 cache pipes from the gfx1201 cache-hierarchy spec.
# GL0 is 32 KB per CU, and its read return is 128 B per clock. HIP counts
# each two-CU WGP as one multiprocessor, so a board has twice
# multiProcessorCount GL0 instances. L2_WORKING_SET_BYTES is 4 MB, which
# still fits a 6 MB GL2. The 32 banks and 128 B/clock are full-chip
# ceilings used by pipe_bandwidth_gbps. GL1 is a buffer with no tags and no
# capacity, so gfx1201 has no L1 roof.
GL0_PER_MULTIPROCESSOR = 2
GL0_READ_BYTES_PER_CLOCK = 128
GL0_CAPACITY_BYTES = 32 * 1024
GL2_BANKS = 32
GL2_HIT_RETURN_BYTES_PER_CLOCK = 128

# L0 is one whole TCP. The 4 MB L2 set misses GL0 and still fits GL2,
# including a 6 MB GL2.
L0_WORKING_SET_BYTES = 32 * 1024
L2_WORKING_SET_BYTES = 4 * 1024 * 1024
# The counts were chosen so a GL2-resident read approaches the GL2 return
# and a GL0-resident read stays under the 128 B return.
CACHE_WORKGROUPS_PER_CU = 8
CACHE_LOADS_IN_FLIGHT = 4
L0_WORKGROUPS_PER_CU = 128
L0_LOADS_IN_FLIGHT = 8

L0_ROOF = ResidentReadRoof(
    L0_WORKING_SET_BYTES,
    GL0_PER_MULTIPROCESSOR,
    GL0_READ_BYTES_PER_CLOCK,
    "GL0 read return",
    L0_WORKGROUPS_PER_CU,
    L0_LOADS_IN_FLIGHT,
    "cache_l0_bw",
    instances_per_multiprocessor=True,
)
L2_ROOF = ResidentReadRoof(
    L2_WORKING_SET_BYTES,
    GL2_BANKS,
    GL2_HIT_RETURN_BYTES_PER_CLOCK,
    "GL2 hit return",
    CACHE_WORKGROUPS_PER_CU,
    CACHE_LOADS_IN_FLIGHT,
    "cache_resident_bw",
)

# Full 256-bit Navi 48 reference. The stored VRAMBw value is the measured read.
DRAM_BUS_BITS = 256
DRAM_GBPS_PER_PIN = 20


# =============================================================================
# Bench_gfx1201 Class
# =============================================================================
class Bench_gfx1201(benchmark_gfx120_base.Bench_gfx120):
    def __init__(self, device_id: int, cache_sizes: dict, gpu_arch: str) -> None:
        super().__init__(device_id, cache_sizes)

        # VRAM runs the DRAM_bw kernel and returns the measured bandwidth. The
        # printed line is the full Navi 48 pin-rate reference, not every board.
        # Infinity Cache uses the MALL column. mi_gpu_spec.yaml lists no
        # F4/F6/F32/F64/I8 matrix path, so those tests are marked unsupported
        # by apply_roofline_spec. VRAM keeps the first slot, so VRAMBw stays the
        # first roofline.csv column.
        self.tests["VRAM"] = partial(
            self.dram_pin_rated_bw_benchmark,
            kernel_name="DRAM_bw",
            label="VRAM",
            bus_bits=DRAM_BUS_BITS,
            gbps_per_pin=DRAM_GBPS_PER_PIN,
        )
        # A set larger than GL2 misses into the Infinity Cache. L2_ROOF stays at
        # 4 MB.
        self.tests["L0"] = partial(self.cache_resident_bw, label="L0", roof=L0_ROOF)
        self.tests["L2"] = partial(self.cache_resident_bw, label="L2", roof=L2_ROOF)
        # The Infinity Cache holds a set that L2 does not.
        self.tests["MALL"] = partial(
            self.cache_stream_bw_bench,
            level="MALL",
            default_capacity_bytes=INFINITY_CACHE_BYTES,
            working_set_divisor=MALL_WORKING_SET_DIVISOR,
        )
        self.apply_roofline_spec(gpu_arch)
        # GL1 is a buffer, not a cache, so L1Bw is written as 0 and the plot
        # draws no L1 roof.
        self.unsupported_data_types += ("L1",)

        # FLOPs per wave per inner iteration, accounting for the WMMA_ACC chains.
        # BF8 is V_WMMA_F32_16X16X16_BF8_BF8: the same 16x16x16 MAC count as FP8.
        self.matrix_ops = {
            "F8": WMMA_FLOPS_PER_INSTRUCTION * WMMA_ACC,
            "BF8": WMMA_FLOPS_PER_INSTRUCTION * WMMA_ACC,
            "F16": WMMA_FLOPS_PER_INSTRUCTION * WMMA_ACC,
            "BF16": WMMA_FLOPS_PER_INSTRUCTION * WMMA_ACC,
        }

    # -----------------------------------------------------------------------------
    # Device-memory bandwidth benchmark
    # -----------------------------------------------------------------------------
    def dram_pin_rated_bw_benchmark(
        self,
        device: int,
        kernel_name: str,
        label: str,
        bus_bits: int,
        gbps_per_pin: float,
    ) -> float:
        """Measure device-memory read bandwidth.

        The returned peak is the measured bandwidth. The printed line is the
        full Navi 48 pin-rate reference, not the peak of every board.
        """
        peak = self.device_memory_bw_benchmark(device, kernel_name, label)
        ceiling = dram_pin_bandwidth_gbps(bus_bits, gbps_per_pin)
        print(
            "Full Navi 48 pin-rate reference (not every board): "
            f"{bus_bits}-bit x {gbps_per_pin} Gbps "
            f"= {ceiling:.1f} GB/sec ({percent_of_ceiling(peak, ceiling)})"
        )
        return peak
