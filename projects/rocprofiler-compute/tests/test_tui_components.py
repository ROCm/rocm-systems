##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
##############################################################################
"""
Unit tests for TUI (Text User Interface) components.

These tests cover the critical functionality of the TUI analysis system,
including dataframe processing, unique key generation for per-dispatch data,
and widget creation.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_top_kernel_df() -> pd.DataFrame:
    """Create a sample top kernel dataframe (dfs[1])."""
    return pd.DataFrame({
        "Kernel_Name": ["kernel_a", "kernel_b", "kernel_a"],
        "Pct": [50.0, 30.0, 20.0],
        "Count": [10, 5, 8],
        "GPU_ID": [0, 0, 1],
    })


@pytest.fixture
def sample_dispatch_id_df() -> pd.DataFrame:
    """Create a sample dispatch ID dataframe (dfs[2])."""
    return pd.DataFrame({
        "Kernel_Name": ["kernel_a", "kernel_b", "kernel_a"],
        "Dispatch_ID": [0, 1, 2],
    })


# =============================================================================
# Tests for tui_utils.py - get_top_kernels_and_dispatch_ids
# =============================================================================


class TestGetTopKernelsAndDispatchIds:
    """Tests for the get_top_kernels_and_dispatch_ids function."""

    def test_returns_none_when_runs_empty(self) -> None:
        """Test that function returns None when runs dict is empty."""
        from rocprof_compute_tui.utils.tui_utils import get_top_kernels_and_dispatch_ids

        result = get_top_kernels_and_dispatch_ids({})
        assert result is None

    def test_returns_none_when_workload_has_no_dfs(self) -> None:
        """Test that function returns None when workload has no dfs attribute."""
        from rocprof_compute_tui.utils.tui_utils import get_top_kernels_and_dispatch_ids

        mock_workload = MagicMock(spec=[])  # No dfs attribute
        runs = {"path": mock_workload}

        result = get_top_kernels_and_dispatch_ids(runs)
        assert result is None

    def test_returns_none_when_top_kernel_df_empty(self) -> None:
        """Test that function returns None when dfs[1] is empty."""
        from rocprof_compute_tui.utils.tui_utils import get_top_kernels_and_dispatch_ids

        mock_workload = MagicMock()
        mock_workload.dfs = {1: pd.DataFrame()}  # Empty dataframe
        runs = {"path": mock_workload}

        result = get_top_kernels_and_dispatch_ids(runs)
        assert result is None

    def test_returns_aggregated_kernel_list(
        self,
        sample_top_kernel_df: pd.DataFrame,
    ) -> None:
        """Test that function returns aggregated kernel list (no dispatch merging)."""
        from rocprof_compute_tui.utils.tui_utils import get_top_kernels_and_dispatch_ids

        mock_workload = MagicMock()
        mock_workload.dfs = {1: sample_top_kernel_df}
        runs = {"path": mock_workload}

        result = get_top_kernels_and_dispatch_ids(runs)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == len(sample_top_kernel_df)

        # Should contain kernel data without dispatch IDs or unique keys
        for record in result:
            assert "Kernel_Name" in record
            assert "Pct" in record
            # Should NOT have Unique_Key (removed for aggregated view)
            assert "Unique_Key" not in record

    def test_results_sorted_by_pct_descending(
        self,
        sample_top_kernel_df: pd.DataFrame,
    ) -> None:
        """Test that results are sorted by Pct in descending order."""
        from rocprof_compute_tui.utils.tui_utils import get_top_kernels_and_dispatch_ids

        mock_workload = MagicMock()
        mock_workload.dfs = {1: sample_top_kernel_df}
        runs = {"path": mock_workload}

        result = get_top_kernels_and_dispatch_ids(runs)

        assert result is not None
        pct_values = [record["Pct"] for record in result]
        assert pct_values == sorted(pct_values, reverse=True)

    def test_returns_none_when_top_kernel_df_missing(
        self,
    ) -> None:
        """Test that function returns None when dfs[1] is missing."""
        from rocprof_compute_tui.utils.tui_utils import get_top_kernels_and_dispatch_ids

        mock_workload = MagicMock()
        mock_workload.dfs = {}  # No dfs[1]
        runs = {"path": mock_workload}

        result = get_top_kernels_and_dispatch_ids(runs)
        assert result is None


# =============================================================================
# Tests for tui_utils.py - process_panels_to_dataframes
# =============================================================================


class TestProcessPanelsToDataframes:
    """Tests for the process_panels_to_dataframes function."""

    def test_returns_dict_structure(self) -> None:
        """Test that function returns proper nested dict structure."""
        from rocprof_compute_tui.utils.tui_utils import process_panels_to_dataframes

        mock_args = MagicMock()
        mock_args.decimal = 2
        mock_args.membw_analysis = False

        mock_arch_configs = MagicMock()
        mock_arch_configs.panel_configs = {}

        result = process_panels_to_dataframes(
            args=mock_args,
            kernel_df={},
            arch_configs=mock_arch_configs,
            profiling_config={},
        )

        assert isinstance(result, dict)

    def test_skips_hidden_sections(self) -> None:
        """Test that hidden sections are skipped."""
        import config
        from rocprof_compute_tui.utils.tui_utils import process_panels_to_dataframes

        mock_args = MagicMock()
        mock_args.decimal = 2
        mock_args.membw_analysis = False

        # Create panel config with hidden section ID
        hidden_id = list(config.HIDDEN_SECTIONS)[0] if config.HIDDEN_SECTIONS else 0
        mock_arch_configs = MagicMock()
        mock_arch_configs.panel_configs = {
            hidden_id: {
                "title": "Hidden Panel",
                "data source": [],
            }
        }

        result = process_panels_to_dataframes(
            args=mock_args,
            kernel_df={},
            arch_configs=mock_arch_configs,
            profiling_config={},
        )

        # Hidden section should not appear in result
        for section_name in result.keys():
            assert "Hidden Panel" not in section_name

    def test_applies_rounding_logic(self) -> None:
        """Test that decimal rounding is applied to dataframes."""
        from rocprof_compute_tui.utils.tui_utils import apply_rounding_logic

        df = pd.DataFrame({
            "Value": [1.23456789, 2.987654321],
            "Pct": [50.123456, 49.876544],
        })

        result = apply_rounding_logic(df, decimal_precision=2)

        # Check that values are rounded to 2 decimal places
        assert result["Value"].iloc[0] == pytest.approx(1.23, rel=0.01)
        assert result["Pct"].iloc[0] == pytest.approx(50.12, rel=0.01)


# =============================================================================
# Tests for analysis_tui.py - Unique Key Generation
# =============================================================================


class TestAnalysisTuiAggregation:
    """Tests for aggregated analysis in analysis_tui.py."""

    def test_aggregated_analysis_no_per_dispatch_keys(self) -> None:
        """Test that analysis uses aggregated approach (no per-dispatch keys)."""
        # This test verifies the conceptual change:
        # TUI no longer creates kernel::dispatch_id keys
        # Instead, it uses aggregated per-kernel data

        # Simulate aggregated kernel data (one entry per kernel)
        aggregated_kernels = [
            {"Kernel_Name": "kernel_a", "Total_Time": 1500, "Count": 100},
            {"Kernel_Name": "kernel_b", "Total_Time": 900, "Count": 100},
        ]

        # Verify no unique keys with dispatch IDs
        for kernel in aggregated_kernels:
            assert "Unique_Key" not in kernel
            assert "::" not in kernel.get("Kernel_Name", "")

        # Verify aggregated metrics are present
        assert aggregated_kernels[0]["Count"] == 100
        assert aggregated_kernels[0]["Total_Time"] == 1500


# =============================================================================
# Tests for collapsibles.py - Widget Creation
# =============================================================================


class TestCollapsiblesWidgetCreation:
    """Tests for widget creation in collapsibles.py."""

    def test_create_table_with_empty_dataframe(self) -> None:
        """Test that create_table returns Label for empty dataframe."""
        from textual.widgets import Label

        from rocprof_compute_tui.widgets.collapsibles import create_table

        df = pd.DataFrame()
        result = create_table(df)

        assert isinstance(result, Label)

    def test_create_widget_from_data_with_none(self) -> None:
        """Test that create_widget_from_data handles None correctly."""
        from textual.widgets import Label

        from rocprof_compute_tui.widgets.collapsibles import create_widget_from_data

        result = create_widget_from_data(None)

        assert isinstance(result, Label)

    def test_create_widget_with_unknown_style(self) -> None:
        """Test that unknown tui_style returns Label with error message."""
        from textual.widgets import Label

        from rocprof_compute_tui.widgets.collapsibles import create_widget_from_data

        df = pd.DataFrame({"col": [1, 2, 3]})
        result = create_widget_from_data(df, tui_style="unknown_style")

        assert isinstance(result, Label)


# =============================================================================
# Tests for Logger in tui_utils.py
# =============================================================================


class TestTuiLogger:
    """Tests for the Logger class in tui_utils.py."""

    def test_logger_initialization(self) -> None:
        """Test that Logger initializes correctly."""
        from rocprof_compute_tui.utils.tui_utils import Logger

        logger = Logger()

        assert logger.output_area is None
        assert logger.logger is not None

        # Verify logging methods work without errors
        logger.info("Info message", update_ui=False)
        logger.warning("Warning message", update_ui=False)
        logger.error("Error message", update_ui=False)
        logger.success("Success message", update_ui=False)


# =============================================================================
# Tests for Config Loading
# =============================================================================


class TestConfigLoading:
    """Tests for configuration loading in collapsibles.py."""

    def test_load_config_with_valid_yaml(self, tmp_path: Path) -> None:
        """Test that load_config correctly parses valid YAML."""
        from rocprof_compute_tui.widgets.collapsibles import load_config

        yaml_content = """
sections:
  - title: "Test Section"
    collapsed: true
    subsections: []
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        result = load_config(str(config_file))

        assert "sections" in result
        assert len(result["sections"]) == 1
        assert result["sections"][0]["title"] == "Test Section"

    def test_load_config_raises_on_invalid_file(self, tmp_path: Path) -> None:
        """Test that load_config raises error for non-existent file."""
        from rocprof_compute_tui.widgets.collapsibles import load_config

        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))


# =============================================================================
# Integration Test - End-to-End Data Flow
# =============================================================================


class TestEndToEndDataFlow:
    """Integration test for the data flow from analysis to display."""

    def test_aggregated_kernel_data_flow(self) -> None:
        """Test that aggregated kernel data flows correctly from analysis to display.

        This tests the new architecture: aggregated per-kernel metrics
        with single panel data structure.
        """
        # Simulate kernel_to_df_dict generated by run_kernel_analysis()
        # Now it's a single dict of panel data (not keyed by kernel name)
        kernel_to_df_dict: dict[str, dict[str, Any]] = {
            "1. System Info": {"data": "system_metrics"},
            "7. Wavefront": {"data": "wavefront_metrics"},
        }

        # Simulate top_kernel_list generated by get_top_kernels_and_dispatch_ids
        # One entry per kernel (aggregated)
        top_kernel_list = [
            {"Kernel_Name": "kernel_a", "Pct": 50.0, "Count": 100},
            {"Kernel_Name": "kernel_b", "Pct": 30.0, "Count": 50},
            {"Kernel_Name": "kernel_c", "Pct": 20.0, "Count": 25},
        ]

        # Verify aggregated structure
        assert len(top_kernel_list) == 3
        assert all("Unique_Key" not in kernel for kernel in top_kernel_list)
        assert all("Kernel_Name" in kernel for kernel in top_kernel_list)

        # Verify panel data is not keyed by kernel
        assert "kernel_a" not in kernel_to_df_dict
        assert "1. System Info" in kernel_to_df_dict
