# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
from typing import Optional

from rocprof_compute_soc.soc_base import OmniSoC_Base
from utils.logger import demarcate
from utils.mi_gpu_spec import mi_gpu_specs
from utils.specs import MachineSpecs


class gfx1151_soc(OmniSoC_Base):
    """SoC class for gfx1151 (Strix Halo, RDNA 3.5 APU).

    Key characteristics:
    - APU: no compute/memory partitioning, no HBM, no XCDs
    - Memory: LPDDR5X with a 256-bit bus (8 memory channels)
    - 8 L2 cache banks (2048 KiB total L2)
    - 4 CU per Shader Array (num_gl1c = cu_per_gpu // 4)
    - 2 pipes per GPU
    """

    # Number of LPDDR5X memory channels (256-bit bus / 32-bit per channel)
    _NUM_MEMORY_CHANNELS: int = 8

    def __init__(self, args: argparse.Namespace, mspec: MachineSpecs) -> None:
        super().__init__(args, mspec)
        self.set_arch("gfx1151")
        self.set_compatible_profilers([
            "rocprofv3",
            "rocprofiler-sdk",
        ])
        # Per IP block max number of simultaneous counters. GFX IP Blocks
        self.set_perfmon_config(mi_gpu_specs.get_perfmon_config("gfx1151"))

        # Set arch-specific specs
        # Strix Halo has 8 L2 cache banks (8 x 256 KiB = 2048 KiB L2)
        self._mspec.l2_banks = 8
        self._mspec.lds_banks_per_cu = 32
        self._mspec.pipes_per_gpu = 2

        # LPDDR5X memory channels: 256-bit bus / 32-bit per channel = 8 channels
        self._mspec.num_memory_channels = str(self._NUM_MEMORY_CHANNELS)

        # GL1 cache count: RDNA 3.5 has 4 CU per Shader Array.
        # num_gl1c = total_SA_count = cu_per_gpu / 4.
        # This is derived dynamically after populate_mspec() has read cu_per_gpu
        # from rocminfo, so we compute it here after super().__init__() returns.
        if self._mspec.cu_per_gpu is not None:
            try:
                self._mspec.num_gl1c = str(int(self._mspec.cu_per_gpu) // 4)
            except (ValueError, TypeError):
                pass

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def profiling_setup(self) -> Optional[list[str]]:
        """Perform any SoC-specific setup prior to profiling."""
        super().profiling_setup()
        # Performance counter filtering
        filter_blocks = self.perfmon_filter()
        return filter_blocks

    @demarcate
    def post_profiling(self) -> None:
        """Perform any SoC-specific post profiling activities."""
        super().post_profiling()
