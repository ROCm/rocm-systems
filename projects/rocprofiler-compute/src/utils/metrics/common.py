# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared metric-evaluation helpers used by both eval_pipeline and db_analysis."""

from __future__ import annotations

import pandas as pd

from utils.logger import console_warning


class ValuDualIssueDetector:
    """Detects VALU metrics exceeding theoretical peak due to dual-issue.

    Configured per workload with the workload's gpu_arch and an optional
    SQ_ACTIVE_INST_VALU2 counter Series. Each call to check() evaluates a
    (metric_name, value, peak) triple and emits a console_warning if the
    value exceeds peak and the metric is one of the dual-issue candidates.
    For gfx950, a non-zero SQ_ACTIVE_INST_VALU2 sum confirms dual-issue
    activity and is appended to the warning.
    """

    VALU_UTILIZATION_METRICS = ("VALU Utilization",)
    VALU_FLOPS_METRICS = ("VALU FLOPs (F64)",)
    METRICS = VALU_UTILIZATION_METRICS + VALU_FLOPS_METRICS
    VALU2_COUNTER = "SQ_ACTIVE_INST_VALU2"
    GFX950 = "gfx950"
    FAQ_URL = (
        "https://rocm.docs.amd.com/projects/"
        "rocprofiler-compute/en/latest/reference/"
        "faq.html#why-does-valu-utilization-exceed-"
        "the-theoretical-peak"
    )

    def __init__(
        self,
        gpu_arch: str,
        valu2_series: pd.Series | None = None,
    ) -> None:
        self._gpu_arch = gpu_arch
        self._dual_issue_confirmed = self._compute_dual_issue_confirmed(valu2_series)

    def check(self, metric_name: str, value: float, peak: float) -> None:
        """Emit a dual-issue warning if metric is a candidate and value > peak."""
        if metric_name not in self.METRICS:
            return
        if not (peak > 0 and value > peak):
            return
        console_warning(self._build_warning(metric_name))

    def _compute_dual_issue_confirmed(
        self,
        valu2_series: pd.Series | None,
    ) -> bool:
        if self._gpu_arch != self.GFX950:
            return False
        if valu2_series is None:
            return False
        return float(valu2_series.sum()) > 0

    def _build_warning(self, metric_name: str) -> str:
        if metric_name in self.VALU_UTILIZATION_METRICS:
            msg = (
                "VALU Utilization can go up to 200% "
                "because CU can dual-issue instructions. "
                f"See {self.FAQ_URL} for more information."
            )
        else:
            msg = (
                "VALU FLOPs can exceed the peak value "
                "because these instructions can be "
                "dual-issued in specific circumstances. "
                f"See {self.FAQ_URL} for more information."
            )
        if self._gpu_arch == self.GFX950 and self._dual_issue_confirmed:
            msg += " (Dual-issue activity detected via SQ_ACTIVE_INST_VALU2 counter)"
        return msg
