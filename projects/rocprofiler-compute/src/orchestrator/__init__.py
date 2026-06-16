# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Backend-specific profile and analysis orchestration."""

from __future__ import annotations

from typing import Any

from orchestrator.common import ProfileDataAnalysisOrchestrator


def create_profile_analysis_orchestrator(
    profiling_config: dict[str, Any],
) -> ProfileDataAnalysisOrchestrator:
    """Create the analysis orchestrator for profiled data."""
    profiler = profiling_config.get("profiler", "rocprofiler-sdk")
    if profiler == "rocprofv3":
        from orchestrator.rocprofv3 import Rocprofv3AnalysisOrchestrator

        return Rocprofv3AnalysisOrchestrator()

    from orchestrator.rocprofiler_sdk import RocprofilerSdkAnalysisOrchestrator

    return RocprofilerSdkAnalysisOrchestrator()
