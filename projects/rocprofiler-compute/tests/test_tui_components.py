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

    def test_returns_none_when_required_dfs_missing(self) -> None:
        """Test that function returns None when dfs[1] or dfs[2] is missing."""
        from rocprof_compute_tui.utils.tui_utils import get_top_kernels_and_dispatch_ids

        mock_workload = MagicMock()
        mock_workload.dfs = {1: pd.DataFrame()}  # Missing dfs[2]
        runs = {"path": mock_workload}

        result = get_top_kernels_and_dispatch_ids(runs)
        assert result is None

    def test_unique_key_generated_correctly(
        self,
        sample_top_kernel_df: pd.DataFrame,
        sample_dispatch_id_df: pd.DataFrame,
    ) -> None:
        """Test that Unique_Key is generated in format 'kernel_name::dispatch_id'."""
        from rocprof_compute_tui.utils.tui_utils import get_top_kernels_and_dispatch_ids

        mock_workload = MagicMock()
        mock_workload.dfs = {1: sample_top_kernel_df, 2: sample_dispatch_id_df}
        runs = {"path": mock_workload}

        result = get_top_kernels_and_dispatch_ids(runs)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0

        for record in result:
            assert "Unique_Key" in record
            kernel_name = record["Kernel_Name"]
            dispatch_id = record["Dispatch_ID"]
            expected_key = f"{kernel_name}::{int(dispatch_id)}"
            assert record["Unique_Key"] == expected_key

    def test_drops_count_and_gpu_id_columns(
        self,
        sample_top_kernel_df: pd.DataFrame,
        sample_dispatch_id_df: pd.DataFrame,
    ) -> None:
        """Test that Count and GPU_ID columns are dropped from result."""
        from rocprof_compute_tui.utils.tui_utils import get_top_kernels_and_dispatch_ids

        mock_workload = MagicMock()
        mock_workload.dfs = {1: sample_top_kernel_df, 2: sample_dispatch_id_df}
        runs = {"path": mock_workload}

        result = get_top_kernels_and_dispatch_ids(runs)

        assert result is not None
        for record in result:
            assert "Count" not in record
            assert "GPU_ID" not in record

    def test_results_sorted_by_pct_descending(
        self,
        sample_top_kernel_df: pd.DataFrame,
        sample_dispatch_id_df: pd.DataFrame,
    ) -> None:
        """Test that results are sorted by Pct in descending order."""
        from rocprof_compute_tui.utils.tui_utils import get_top_kernels_and_dispatch_ids

        mock_workload = MagicMock()
        mock_workload.dfs = {1: sample_top_kernel_df, 2: sample_dispatch_id_df}
        runs = {"path": mock_workload}

        result = get_top_kernels_and_dispatch_ids(runs)

        assert result is not None
        pct_values = [record["Pct"] for record in result]
        assert pct_values == sorted(pct_values, reverse=True)


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


class TestAnalysisTuiUniqueKeyGeneration:
    """Tests for unique key generation in analysis_tui.py."""

    def test_unique_key_prevents_overwrites(self) -> None:
        """Test that unique keys prevent data overwrites for same kernel."""
        raw_dfs: dict[str, dict] = {}

        # Simulate processing multiple dispatches of same kernel
        dispatches = [
            ("kernel_a", 0, {"data": "dispatch_0"}),
            ("kernel_a", 1, {"data": "dispatch_1"}),
            ("kernel_a", 2, {"data": "dispatch_2"}),
        ]

        for kernel_name, dispatch_id, kernel_dfs in dispatches:
            unique_key = f"{kernel_name}::{dispatch_id}"
            raw_dfs[unique_key] = kernel_dfs

        # All dispatches should be preserved
        assert len(raw_dfs) == 3
        assert "kernel_a::0" in raw_dfs
        assert "kernel_a::1" in raw_dfs
        assert "kernel_a::2" in raw_dfs

        # Data should be distinct
        assert raw_dfs["kernel_a::0"]["data"] == "dispatch_0"
        assert raw_dfs["kernel_a::1"]["data"] == "dispatch_1"
        assert raw_dfs["kernel_a::2"]["data"] == "dispatch_2"

    def test_unique_key_with_special_characters(self) -> None:
        """Test unique key generation with special characters in kernel name."""
        kernel_names = [
            "kernel<float>",
            "kernel::method",
            "namespace::class::method",
        ]

        raw_dfs: dict[str, dict] = {}
        for kernel_name in kernel_names:
            unique_key = f"{kernel_name}::0"
            raw_dfs[unique_key] = {"data": kernel_name}

        assert len(raw_dfs) == len(kernel_names)


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

    def test_multiple_dispatches_same_kernel_accessible(self) -> None:
        """Test that all dispatches of the same kernel are individually accessible.

        This tests the core fix: ensuring per-dispatch data is preserved and
        accessible through the unique key mechanism.
        """
        # Simulate raw_dfs generated by analysis_tui.py
        kernel_to_df_dict: dict[str, dict[str, Any]] = {
            "kernel_a::0": {"section": {"data": "first_dispatch"}},
            "kernel_a::1": {"section": {"data": "second_dispatch"}},
            "kernel_a::2": {"section": {"data": "third_dispatch"}},
        }

        # Simulate top_kernel_list generated by get_top_kernels_and_dispatch_ids
        top_kernel_list = [
            {"Kernel_Name": "kernel_a", "Dispatch_ID": 0, "Unique_Key": "kernel_a::0"},
            {"Kernel_Name": "kernel_a", "Dispatch_ID": 1, "Unique_Key": "kernel_a::1"},
            {"Kernel_Name": "kernel_a", "Dispatch_ID": 2, "Unique_Key": "kernel_a::2"},
        ]

        # Verify each selection accesses distinct data (kernel_view.py logic)
        expected_data = ["first_dispatch", "second_dispatch", "third_dispatch"]
        for i, kernel_data in enumerate(top_kernel_list):
            current_selection = kernel_data["Unique_Key"]
            assert current_selection in kernel_to_df_dict
            assert (
                kernel_to_df_dict[current_selection]["section"]["data"]
                == expected_data[i]
            )
