# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import csv
from pathlib import Path
from typing import Optional

from rocprof_compute_soc.soc_base import OmniSoC_Base
from utils import csv_compression
from utils.logger import console_error, demarcate
from utils.mi_gpu_spec import mi_gpu_specs
from utils.specs import MachineSpecs
from utils.utils_common import parse_pmc_perf


class gfx1201_soc(OmniSoC_Base):
    """SoC handler for gfx1201 (RDNA4 Navi48 dGPU, e.g. Radeon AI PRO R9700).

    RDNA4 key differences vs gfx1151 (RDNA3.5 APU):
    - Discrete GPU with GDDR6 memory (not unified LPDDR5X)
    - 64 CUs (4 SEs × 2 SAs × 8 CUs)
    - 4 MCD chiplets × 8 GL2C banks = 32 GL2C banks total
    - Wave32 ISA, 4 SIMDs per CU — same as gfx1151
    - New WMMA unit: F16→F32 16×16×16 in 16 cycles, event IDs 229/233/256
    - Requires patched aqlprofile (SqcCounterBlockMaxEvent=511) for events >232
    """

    def __init__(self, args: argparse.Namespace, mspec: MachineSpecs) -> None:
        super().__init__(args, mspec)
        self.set_arch("gfx1201")
        self.set_compatible_profilers(["rocprofv3", "rocprofiler-sdk"])
        self.set_perfmon_config(mi_gpu_specs.get_perfmon_config("gfx1201"))

        # Navi48 dGPU: 4 MCD chiplets × 8 GL2C banks = 32 banks.
        # Navi48 has 2 render backends per SA; 4 SEs × 2 SAs = 8 SA pairs → 32 banks.
        self._mspec.l2_banks = 32
        # 32 lanes per LDS bank (wave32), same as RDNA3
        self._mspec.lds_banks_per_cu = 32
        # 2 geometry pipes (same as RDNA3.5)
        self._mspec.pipes_per_gpu = 2

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def profiling_setup(self) -> Optional[list[str]]:
        """Perform any SoC-specific setup prior to profiling."""
        super().profiling_setup()
        filter_blocks = self.perfmon_filter()
        return filter_blocks

    @demarcate
    def post_profiling(self) -> None:
        """Perform any SoC-specific post profiling activities."""
        super().post_profiling()

        workload_dir = Path(self.get_args().output_directory)
        wmma_counters = {"SQ_WMMA_VALU_INSTS", "SQ_WMMA_VALU_INSTS_sum"}
        requested = any(
            wmma_counters.intersection(parse_pmc_perf(str(pmc_file)))
            for pmc_file in (workload_dir / "perfmon").glob("pmc_perf_*.yaml")
        )
        if not requested:
            return

        collected: set[str] = set()
        try:
            for result_file in csv_compression.find_csvs(
                workload_dir, "results_*.csv"
            ):
                with csv_compression.open_csv_read(result_file) as stream:
                    collected.update(
                        row["Counter_Name"]
                        for row in csv.DictReader(stream)
                        if row.get("Counter_Name")
                    )
        except (OSError, UnicodeError, csv.Error, KeyError) as error:
            console_error(
                "Unable to validate SQ_WMMA_VALU_INSTS profiling output: "
                f"{error}"
            )

        if not wmma_counters.intersection(collected):
            console_error(
                "SQ_WMMA_VALU_INSTS was requested but is absent from the "
                "profiling output. The installed aqlprofile may reject gfx1201 "
                "SQ event 233 because SqcCounterBlockMaxEvent is below 511."
            )
