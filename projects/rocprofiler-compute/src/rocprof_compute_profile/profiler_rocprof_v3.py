# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse

from orchestrator.rocprofv3 import Rocprofv3ProfileOrchestrator
from rocprof_compute_profile.profiler_base import RocProfCompute_Base
from rocprof_compute_soc.soc_base import OmniSoC_Base
from utils.logger import console_log, demarcate


class rocprof_v3_profiler(RocProfCompute_Base):
    def __init__(
        self,
        profiling_args: argparse.Namespace,
        profiler_mode: str,
        soc: OmniSoC_Base,
    ) -> None:
        super().__init__(profiling_args, profiler_mode, soc)

    def get_profiler_options(self) -> list[str]:
        return Rocprofv3ProfileOrchestrator().build_profiler_options(self.get_args())

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def pre_processing(self) -> None:
        """Perform any pre-processing steps prior to profiling."""
        super().pre_processing()

    @demarcate
    def run_profiling(self, version: str, prog: str) -> None:
        """Run profiling."""
        if self.get_args().roof_only:
            console_log("roofline", "Profiling roofline counters only.")

        # Log profiling options and setup filtering
        super().run_profiling(version, prog)

    @demarcate
    def post_processing(self) -> None:
        """Perform any post-processing steps prior to profiling."""
        super().post_processing()
