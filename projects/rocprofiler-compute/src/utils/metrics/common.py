# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared metric-evaluation helpers used by evaluation_pipeline and analysis_db."""

from typing import Optional

import pandas as pd

from utils.logger import console_warning


class ValuDualIssueDetector:
    """Per-workload detector for VALU metrics exceeding theoretical peak.

    - Configured with gpu_arch and an optional SQ_ACTIVE_INST_VALU2 series.
    - check() emits the dual-issue console_warning when a candidate metric's
      value exceeds peak; on gfx950 a non-zero VALU2 counter is appended.
    """

    valu_utilization_metrics = ("VALU Utilization",)
    valu_flops_metrics = ("VALU FLOPs (F64)",)
    metrics = valu_utilization_metrics + valu_flops_metrics
    valu2_counter = "SQ_ACTIVE_INST_VALU2"
    gfx950 = "gfx950"
    faq_url = (
        "https://rocm.docs.amd.com/projects/"
        "rocprofiler-compute/en/latest/reference/"
        "faq.html#why-does-valu-utilization-exceed-"
        "the-theoretical-peak"
    )

    def __init__(
        self,
        gpu_arch: str,
        valu2_series: Optional[pd.Series] = None,
    ) -> None:
        self._gpu_arch = gpu_arch
        self._dual_issue_confirmed = self._compute_dual_issue_confirmed(valu2_series)

    def check(self, metric_name: str, value: float, peak: float) -> None:
        """Emit a dual-issue warning if metric is a candidate and value > peak."""
        if metric_name not in self.metrics:
            return
        if not (peak > 0 and value > peak):
            return
        console_warning(self._build_warning(metric_name))

    def _compute_dual_issue_confirmed(
        self,
        valu2_series: Optional[pd.Series],
    ) -> bool:
        if self._gpu_arch != self.gfx950:
            return False
        if valu2_series is None:
            return False
        return float(valu2_series.sum()) > 0

    def _build_warning(self, metric_name: str) -> str:
        if metric_name in self.valu_utilization_metrics:
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
        if self._gpu_arch == self.gfx950 and self._dual_issue_confirmed:
            msg += " (Dual-issue activity detected via SQ_ACTIVE_INST_VALU2 counter)"
        return msg
