# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Navi48 (gfx1201) topology and profiling setup."""

import argparse
from typing import List, Optional, Set

from rocprof_compute_soc.soc_base import OmniSoC_Base
from utils.logger import console_warning, demarcate
from utils.mi_gpu_spec import mi_gpu_specs
from utils.specs import MachineSpecs

_WMMA_COUNTERS = frozenset({"SQ_WMMA_VALU_INSTS", "SQ_WMMA_VALU_INSTS_sum"})


class gfx1201_soc(OmniSoC_Base):
    """SoC handler for Navi48 RDNA 4 GPUs, including Radeon RX 9070."""

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
        """Select and group counters through the shared profiling pipeline."""
        super().profiling_setup()
        return self.perfmon_filter()

    def validate_counter_collection(
        self, requested: Set[str], collected: Set[str]
    ) -> None:
        """Diagnose a missing WMMA counter only when its pass collected other data."""
        if (
            not _WMMA_COUNTERS.intersection(requested)
            or not requested.intersection(collected)
            or _WMMA_COUNTERS.intersection(collected)
        ):
            return
        console_warning(
            "SQ_WMMA_VALU_INSTS was requested but is absent from a pass that "
            "collected other counters. Check SDK/aqlprofile support for gfx1201 "
            "SQ event 233 and the configured SqcCounterBlockMaxEvent limit."
        )

    @demarcate
    def post_profiling(self) -> None:
        """Run shared post processing."""
        super().post_profiling()
