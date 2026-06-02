# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared metric-evaluation helpers used by evaluation_pipeline and analysis_db."""

import pandas as pd

from utils.logger import console_warning


class ValuDualIssueDetector:
    """Per-workload detector for VALU metrics exceeding theoretical peak.

    - check() emits a dual-issue warning above peak and a counter-issue
      warning above 2x peak on any arch.
    - On gfx950 a non-zero SQ_ACTIVE_INST_VALU2 sum suffixes the dual-issue
      message with hardware confirmation.
    """

    candidate_metrics = {"VALU Utilization", "VALU FLOPs (F64)"}
    faq_url = (
        "https://rocm.docs.amd.com/projects/"
        "rocprofiler-compute/en/latest/reference/"
        "faq.html#why-does-valu-utilization-exceed-"
        "the-theoretical-peak"
    )

    def __init__(self, gpu_arch: str, raw_pmc_df: pd.DataFrame) -> None:
        self._dual_issue_confirmed = (
            gpu_arch == "gfx950"
            and "SQ_ACTIVE_INST_VALU2" in raw_pmc_df.columns
            and float(raw_pmc_df["SQ_ACTIVE_INST_VALU2"].sum()) > 0
        )

    def check(self, metric_name: str, value: float, peak: float) -> None:
        """Emit dual-issue warning above peak; counter-issue warning above 2x peak."""
        if metric_name not in self.candidate_metrics:
            return
        if not (peak > 0 and value > peak):
            return
        if value > 2 * peak:
            console_warning(self._build_counter_issue_warning(metric_name))
        else:
            console_warning(self._build_warning(metric_name))

    def _build_counter_issue_warning(self, metric_name: str) -> str:
        if metric_name == "VALU Utilization":
            return (
                "VALU Utilization exceeds twice the theoretical peak, "
                "indicating an issue in raw performance counters."
            )
        return (
            "VALU FLOPs exceeds twice the theoretical peak, "
            "indicating an issue in raw performance counters."
        )

    def _build_warning(self, metric_name: str) -> str:
        if metric_name == "VALU Utilization":
            msg = (
                "VALU Utilization can go up to 200% "
                "because CU can dual-issue instructions. "
                f"See {self.faq_url} for more information."
            )
        else:
            msg = (
                "VALU FLOPs can exceed the peak value "
                "because these instructions can be "
                "dual-issued in specific circumstances. "
                f"See {self.faq_url} for more information."
            )
        if self._dual_issue_confirmed:
            msg += " (Dual-issue activity detected via SQ_ACTIVE_INST_VALU2 counter)"
        return msg
