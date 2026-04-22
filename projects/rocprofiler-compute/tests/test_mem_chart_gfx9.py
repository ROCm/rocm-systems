# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests for mem_chart_gfx9.py - CDNA Memory Architecture Visualization.

Regression coverage for commit 43ad1138d1, which corrected the InstrDispatch
metric lookup from the legacy "Matrix Ops" key to the panel-YAML key "MFMA".
The tests exercise the public surface (plot_mem_chart) and assert the YAML
contract that the renderer relies on.
"""

import re
from pathlib import Path

import pytest
import yaml

from utils import mem_chart_gfx9

ANSI_ESCAPE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")

# Repo root: tests/test_mem_chart_gfx9.py -> projects/rocprofiler-compute/
REPO_ROOT = Path(__file__).resolve().parent.parent

GFX9_SOC_DIRS = ["gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"]


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


# =============================================================================
# Regression: InstrDispatch MFMA lookup uses the YAML key "MFMA"
# =============================================================================


class TestPlotMemChartMfma:
    """Regression for the gfx9 MFMA lookup-key fix (commit 43ad1138d1)."""

    def test_mfma_value_renders(self):
        """MFMA value from metric_dict is shown in the InstrDispatch column."""
        metric_dict = {"MFMA": 4242}
        out = strip_ansi(
            mem_chart_gfx9.plot_mem_chart("gfx942", "per_kernel", metric_dict)
        )
        assert "MFMA" in out
        assert "4242" in out

    def test_legacy_matrix_ops_key_ignored(self):
        """A legacy 'Matrix Ops' key must NOT be picked up by the renderer.

        Prior to commit 43ad1138d1 the renderer looked up `metric_dict["Matrix Ops"]`,
        so a legacy key would have rendered through. After the fix the renderer
        only honors the YAML key 'MFMA', and legacy values must not leak in.
        """
        metric_dict = {"Matrix Ops": 9999}
        out = strip_ansi(
            mem_chart_gfx9.plot_mem_chart("gfx942", "per_kernel", metric_dict)
        )
        assert "MFMA" in out
        assert "9999" not in out
        assert "N/A" in out

    def test_missing_mfma_renders_na(self):
        """When MFMA is absent, the row label still renders with N/A value."""
        out = strip_ansi(mem_chart_gfx9.plot_mem_chart("gfx942", "per_kernel", {}))
        assert "MFMA" in out
        assert "N/A" in out


# =============================================================================
# InstrDispatch row: every instruction label maps to a YAML key
# =============================================================================


class TestPlotMemChartInstrDispatch:
    """Guards every InstrDispatch row against a similar key-mismatch bug."""

    INSTR_KEYS_AND_VALUES = {
        "SALU": 1101,
        "SMEM": 2202,
        "VALU": 3303,
        "MFMA": 4404,
        "VMEM": 5505,
        "LDS": 6606,
        "GWS": 7707,
        "BR": 8808,
    }

    def test_all_instr_labels_render(self):
        out = strip_ansi(
            mem_chart_gfx9.plot_mem_chart(
                "gfx942", "per_kernel", self.INSTR_KEYS_AND_VALUES
            )
        )
        # The "BR" YAML key is rendered as the "BRANCH" label by the gfx9
        # renderer (see mem_chart_gfx9.py InstrDispatch population).
        for label in ("SALU", "SMEM", "VALU", "MFMA", "VMEM", "LDS", "GWS", "BRANCH"):
            assert label in out, f"missing label {label!r} in chart output"
        for value in self.INSTR_KEYS_AND_VALUES.values():
            assert str(value) in out, f"missing value {value} in chart output"


# =============================================================================
# Smoke: full sample dict covering every gfx9 Memory Chart panel-YAML key
# =============================================================================


def _full_sample_metric_dict() -> dict[str, int]:
    """Representative gfx9 Memory Chart fixture covering every panel-YAML key.

    Values are sequential integers (1..53) chosen only to be visually
    distinguishable in the rendered chart; they have no physical meaning.
    """
    return {
        "Wavefront Occupancy": 1,
        "Wave Life": 2,
        "SALU": 3,
        "SMEM": 4,
        "VALU": 5,
        "MFMA": 6,
        "VMEM": 7,
        "LDS": 8,
        "GWS": 9,
        "BR": 10,
        "Active CUs": 11,
        "Num CUs": 12,
        "VGPR": 13,
        "SGPR": 14,
        "LDS Allocation": 15,
        "Scratch Allocation": 16,
        "Wavefronts": 17,
        "Workgroups": 18,
        "LDS Req": 19,
        "LDS Util": 20,
        "LDS Latency": 21,
        "VL1 Rd": 22,
        "VL1 Wr": 23,
        "VL1 Atomic": 24,
        "VL1 Hit": 25,
        "VL1 Lat": 26,
        "VL1 Coalesce": 27,
        "VL1 Stall": 28,
        "sL1D Rd": 29,
        "sL1D Hit": 30,
        "sL1D Lat": 31,
        "IL1 Fetch": 32,
        "IL1 Hit": 33,
        "IL1 Lat": 34,
        "IL1_L2 Rd": 35,
        "VL1_L2 Rd": 36,
        "VL1_L2 Wr": 37,
        "VL1_L2 Atomic": 38,
        "sL1D_L2 Rd": 39,
        "sL1D_L2 Wr": 40,
        "sL1D_L2 Atomic": 41,
        "L2 Rd": 42,
        "L2 Wr": 43,
        "L2 Atomic": 44,
        "L2 Hit": 45,
        "Fabric_L2 Rd": 46,
        "Fabric_L2 Wr": 47,
        "Fabric_L2 Atomic": 48,
        "Fabric Rd Lat": 49,
        "Fabric Wr Lat": 50,
        "Fabric Atomic Lat": 51,
        "HBM Rd": 52,
        "HBM Wr": 53,
    }


class TestPlotMemChartSmoke:
    """End-to-end smoke test: full sample dict on every gfx9 arch."""

    @pytest.mark.parametrize("arch", GFX9_SOC_DIRS)
    def test_full_sample_renders(self, arch: str):
        out = mem_chart_gfx9.plot_mem_chart(
            arch, "per_kernel", _full_sample_metric_dict()
        )
        assert isinstance(out, str)
        assert len(out) > 100
        clean = strip_ansi(out)
        assert "Normalization: per_kernel" in clean


# =============================================================================
# YAML contract: every gfx9 panel must define MFMA (and not Matrix Ops)
# =============================================================================


class TestGfx9MemChartYamlContract:
    """Guards against renaming/dropping MFMA in any gfx9 panel YAML.

    The panel YAML is the source of truth that mem_chart_gfx9.py and
    gui_components/memchart.py look up by literal key. A YAML rename without
    a corresponding renderer update was the original bug (commit 43ad1138d1).
    """

    @pytest.mark.parametrize("soc", GFX9_SOC_DIRS)
    def test_panel_yaml_defines_mfma(self, soc: str):
        path = (
            REPO_ROOT
            / "src"
            / "rocprof_compute_soc"
            / "analysis_configs"
            / soc
            / "0300_memory_chart.yaml"
        )
        assert path.exists(), f"missing panel YAML for {soc}: {path}"

        with path.open() as f:
            doc = yaml.safe_load(f)

        metrics = doc["Panel Config"]["data source"][0]["metric_table"]["metric"]
        assert "MFMA" in metrics, (
            f"{soc}/0300_memory_chart.yaml lost the 'MFMA' metric, which "
            f"mem_chart_gfx9.py and gui_components/memchart.py rely on"
        )
        assert "Matrix Ops" not in metrics, (
            f"{soc}/0300_memory_chart.yaml re-introduced the legacy "
            f"'Matrix Ops' key; renderers look up 'MFMA'"
        )
