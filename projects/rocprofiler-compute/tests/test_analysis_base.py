# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_analyze.analysis_base.OmniAnalyze_Base."""

from argparse import Namespace
from unittest.mock import MagicMock

import pandas as pd

from rocprof_compute_analyze.analysis_cli import cli_analysis


class TestJoinProf:
    """Tests for OmniAnalyze_Base.join_prof."""

    def test_renames_sq_accum_prev_hires_to_bucket_target(self, tmp_path) -> None:
        """ACCUM bucket files rename the generic SQ_ACCUM_PREV_HIRES column.

        results_pmc_perf_<bucket>_ACCUM.csv files whose body uses rocprof's
        generic SQ_ACCUM_PREV_HIRES column must have that column renamed to the
        bucket target derived from the file stem before the merge.
        """
        (tmp_path / "profiling_config.yaml").write_text(
            "format_rocprof_output: csv\njoin_type: kernel\n"
        )

        header = (
            "GPU_ID,Kernel_Name,Dispatch_ID,Grid_Size,Workgroup_Size,"
            "LDS_Per_Workgroup,Scratch_Per_Workitem,SGPR,vgpr,{counter}\n"
        )
        acc = tmp_path / "results_pmc_perf_SQ_LEVEL_WAVES_ACCUM.csv"
        acc.write_text(
            header.format(counter="SQ_ACCUM_PREV_HIRES")
            + "0,kernel_a,0,1024,64,32,0,8,4,100\n"
            + "0,kernel_a,1,1024,64,32,0,8,4,200\n"
        )
        other = tmp_path / "results_pmc_perf_0.csv"
        other.write_text(
            header.format(counter="SQ_WAVES")
            + "0,kernel_a,0,1024,64,32,0,8,4,10\n"
            + "0,kernel_a,1,1024,64,32,0,8,4,20\n"
        )

        inst = cli_analysis.__new__(cli_analysis)
        args = Namespace(
            path=[[str(tmp_path)]],
            join_type="kernel",
            kokkos_trace=False,
        )
        inst.get_args = MagicMock(return_value=args)
        inst._arch_configs = {}
        inst._runs = {}

        inst.join_prof(tmp_path, out=str(tmp_path / "pmc_perf.csv"))
        merged = pd.read_csv(tmp_path / "pmc_perf.csv")

        assert "SQ_LEVEL_WAVES_ACCUM" in merged.columns
        assert "SQ_ACCUM_PREV_HIRES" not in merged.columns
        assert set(merged["SQ_LEVEL_WAVES_ACCUM"].tolist()) == {100, 200}
        assert "SQ_WAVES" in merged.columns
