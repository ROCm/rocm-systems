# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
from typing import List, Optional

from rocprof_compute_soc.soc_base import OmniSoC_Base
from utils.logger import demarcate
from utils.mi_gpu_spec import mi_gpu_specs
from utils.specs import MachineSpecs

_WMMA_COLLECTION_HINT = (
    "Check SDK/aqlprofile support for gfx1201 SQ event 233 and the "
    "configured SqcCounterBlockMaxEvent limit."
)


class gfx1201_soc(OmniSoC_Base):
    counter_collection_hints = {
        "SQ_WMMA_VALU_INSTS": _WMMA_COLLECTION_HINT,
        "SQ_WMMA_VALU_INSTS_sum": _WMMA_COLLECTION_HINT,
    }

    def __init__(self, args: argparse.Namespace, mspec: MachineSpecs) -> None:
        super().__init__(args, mspec)
        self.set_arch("gfx1201")
        self.set_perfmon_config(mi_gpu_specs.get_perfmon_config("gfx1201"))

        # Ported topology values remain subject to Navi48 SKU qualification.
        self._mspec.l2_banks = 32
        self._mspec.lds_banks_per_cu = 32
        self._mspec.pipes_per_gpu = 2

    @demarcate
    def profiling_setup(self) -> Optional[List[str]]:
        """Perform any SoC-specific setup prior to profiling."""
        super().profiling_setup()
        # Performance counter filtering
        filter_blocks = self.perfmon_filter()
        return filter_blocks

    @demarcate
    def post_profiling(self) -> None:
        """Perform any SoC-specific post profiling activities."""
        super().post_profiling()
