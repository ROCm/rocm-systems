# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end tests for ROCm SPM Perfetto output."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Optional

import pytest

from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.spm,
    pytest.mark.gpu,
    pytest.mark.rocm,
    pytest.mark.transpose,
]


def _find_rocprofv3_avail(rocm_path: Optional[Path]) -> Optional[Path]:
    """Return rocprofv3-avail from ROCm or PATH."""
    if rocm_path is not None:
        candidate = rocm_path / "bin" / "rocprofv3-avail"
        if candidate.exists() and candidate.is_file():
            return candidate

    found = shutil.which("rocprofv3-avail")
    return Path(found) if found else None


def _spm_sq_waves_available(rocm_path: Optional[Path]) -> tuple[bool, str]:
    """Check whether the SDK reports SQ_WAVES as an SPM-capable counter."""
    rocprofv3_avail = _find_rocprofv3_avail(rocm_path)
    if rocprofv3_avail is None:
        return False, "rocprofv3-avail not found"

    env = os.environ.copy()
    if rocm_path is not None:
        env["ROCM_PATH"] = str(rocm_path)
        env["PATH"] = f"{rocm_path / 'bin'}:{env.get('PATH', '')}"

    try:
        result = subprocess.run(
            [str(rocprofv3_avail), "list", "--spm"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"failed to run rocprofv3-avail: {exc}"

    if result.returncode != 0:
        return False, f"rocprofv3-avail failed: {result.stdout}"

    if "SQ_WAVES" not in result.stdout:
        return False, "SQ_WAVES was not reported as SPM-capable"

    return True, ""


def _load_trace_processor(trace_path: Path) -> Any:
    """Load a Perfetto trace using the optional configured trace processor shell."""
    trace_processor = pytest.importorskip(
        "perfetto.trace_processor",
        reason="Perfetto trace processor Python API not available",
    )

    trace_processor_shell = os.environ.get("ROCPROFSYS_TRACE_PROC_SHELL")
    if trace_processor_shell:
        config = trace_processor.TraceProcessorConfig(bin_path=trace_processor_shell)
        return trace_processor.TraceProcessor(trace=str(trace_path), config=config)

    return trace_processor.TraceProcessor(trace=str(trace_path))


def _spm_counter_summary(trace_path: Path) -> dict[str, Any]:
    """Return aggregate SPM counter-track statistics from a Perfetto trace."""
    tp = _load_trace_processor(trace_path)
    try:
        rows = list(tp.query("""
                SELECT
                    COUNT(DISTINCT t.id) AS track_count,
                    COUNT(c.id) AS sample_count,
                    MIN(c.value) AS min_value,
                    MAX(c.value) AS max_value,
                    SUM(c.value) AS total_value
                FROM counter c
                JOIN counter_track t ON c.track_id = t.id
                WHERE t.name LIKE 'GPU SPM SQ_WAVES%'
                """))
    finally:
        close = getattr(tp, "close", None)
        if close is not None:
            close()

    if not rows:
        return {
            "track_count": 0,
            "sample_count": 0,
            "min_value": None,
            "max_value": None,
            "total_value": None,
        }

    row = rows[0]
    return {
        "track_count": row.track_count or 0,
        "sample_count": row.sample_count or 0,
        "min_value": row.min_value,
        "max_value": row.max_value,
        "total_value": row.total_value,
    }


@pytest.fixture
def spm_perfetto_env() -> dict[str, str]:
    """Environment for a bounded SPM Perfetto validation run."""
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "OFF",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
        "ROCPROFSYS_USE_KOKKOSP": "OFF",
        "ROCPROFSYS_ROCM_SPM_EVENTS": "SQ_WAVES",
        "ROCPROFSYS_ROCM_SPM_SAMPLE_INTERVAL": "32768",
        "ROCPROFSYS_ROCM_SPM_SAMPLE_INTERVAL_UNIT": "sclk_cycles",
    }


@pytest.mark.timeout(240)
@pytest.mark.class_name("spm-perfetto")
class TestSPMPerfetto(RocprofsysTest):
    """Validate that SPM emits SQ_WAVES samples to Perfetto when supported."""

    def test_sq_waves_trace(self, rocprof_config, spm_perfetto_env):
        available, reason = _spm_sq_waves_available(rocprof_config.rocm_path)
        if not available:
            pytest.skip(reason)

        result = self.run_test(
            "sys_run",
            "transpose",
            env=spm_perfetto_env,
            check_target_arch=True,
        )
        self.assert_regex(result)

        perfetto_file = result.perfetto_file
        assert perfetto_file is not None, "Perfetto trace file was not generated"

        summary = _spm_counter_summary(perfetto_file)
        assert (
            summary["track_count"] > 0
        ), "No GPU SPM SQ_WAVES counter tracks were found in Perfetto trace"
        assert (
            summary["sample_count"] > 0
        ), "GPU SPM SQ_WAVES tracks did not contain any samples"
        assert (
            summary["max_value"] is not None and summary["max_value"] > 0
        ), f"GPU SPM SQ_WAVES samples did not contain positive values: {summary}"
        assert (
            summary["total_value"] is not None and summary["total_value"] > 0
        ), f"GPU SPM SQ_WAVES sample total is not positive: {summary}"
