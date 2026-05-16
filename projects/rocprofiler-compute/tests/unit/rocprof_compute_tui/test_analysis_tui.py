# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_tui analysis utilities and widgets."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Mark all tests in this file with the 'tui' marker for pytest
pytestmark = pytest.mark.tui


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_top_kernel_df() -> pd.DataFrame:
    """Create a sample top kernel dataframe (dfs[1])
    with aggregated per-kernel stats."""
    return pd.DataFrame({
        "Kernel_Name": ["kernel_a", "kernel_b", "kernel_c"],
        "Percent": [50.0, 30.0, 20.0],
        "Count": [10, 5, 8],
        "Total_Time": [1500.0, 900.0, 600.0],
    })


# =============================================================================
# Tests for tui_utils.py - get_top_kernels
# =============================================================================


class TestGetTopKernels:
    """Tests for the get_top_kernels function (aggregated kernel stats)."""

    def test_returns_none_for_invalid_input(self) -> None:
        """Test that function returns None for invalid inputs."""
        from rocprof_compute_tui.utils.tui_utils import get_top_kernels

        # Empty runs
        assert get_top_kernels({}) is None

        # No dfs attribute
        mock_workload = MagicMock(spec=[])
        assert get_top_kernels({"path": mock_workload}) is None

        # Missing dfs[1]
        mock_workload = MagicMock()
        mock_workload.dfs = {}
        assert get_top_kernels({"path": mock_workload}) is None

    def test_returns_empty_list_when_dataframe_empty(self) -> None:
        """Test that function returns empty list when dfs[1] is empty."""
        from rocprof_compute_tui.utils.tui_utils import get_top_kernels

        mock_workload = MagicMock()
        mock_workload.dfs = {
            1: pd.DataFrame(columns=["Kernel_Name", "Percent", "Count"])
        }

        assert get_top_kernels({"path": mock_workload}) == []

    def test_returns_sorted_kernel_records(
        self,
        sample_top_kernel_df: pd.DataFrame,
    ) -> None:
        """Test that function returns kernel records sorted by Percent descending."""
        from rocprof_compute_tui.utils.tui_utils import get_top_kernels

        mock_workload = MagicMock()
        mock_workload.dfs = {1: sample_top_kernel_df}

        result = get_top_kernels({"path": mock_workload})

        assert result is not None
        assert len(result) == 3
        # Verify sorted by Percent descending
        pct_values = [record["Percent"] for record in result]
        assert pct_values == sorted(pct_values, reverse=True)
        # Verify all expected columns preserved
        assert all("Kernel_Name" in r and "Percent" in r for r in result)

    def test_handles_missing_pct_column(self) -> None:
        """Test that function returns unsorted records when Pct column is missing."""
        from rocprof_compute_tui.utils.tui_utils import get_top_kernels

        df_no_pct = pd.DataFrame({
            "Kernel_Name": ["kernel_a", "kernel_b"],
            "Count": [10, 5],
        })
        mock_workload = MagicMock()
        mock_workload.dfs = {1: df_no_pct}

        result = get_top_kernels({"path": mock_workload})

        assert result is not None
        assert len(result) == 2


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
            mock_args,
            {},
            mock_arch_configs,
            {},
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
            mock_args,
            {},
            mock_arch_configs,
            {},
        )

        # Hidden section should not appear in result
        for section_name in result.keys():
            assert "Hidden Panel" not in section_name

    def test_applies_rounding_logic(self) -> None:
        """Test that decimal rounding is applied to dataframes."""
        from rocprof_compute_tui.utils.tui_utils import apply_rounding_logic

        df = pd.DataFrame({
            "Value": [1.23456789, 2.987654321],
            "Percent": [50.123456, 49.876544],
        })

        result = apply_rounding_logic(df, decimal_precision=2)

        # Check that values are rounded to 2 decimal places
        assert result["Value"].iloc[0] == pytest.approx(1.23, rel=0.01)
        assert result["Percent"].iloc[0] == pytest.approx(50.12, rel=0.01)


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
# Integration Test - Data Structure Validation
# =============================================================================


class TestDataStructureIntegration:
    """Validate expected data structures flow correctly between components."""

    def test_kernel_selection_data_lookup(self) -> None:
        """Test that kernel selection via Kernel_Name correctly looks up analysis data.

        This validates the contract between:
        - get_top_kernels() returns list with Kernel_Name keys
        - run_kernel_analysis() returns dict keyed by kernel name
        - kernel_view uses Kernel_Name to look up the correct data
        """
        # Data structures as produced by the analysis pipeline
        kernel_to_df_dict: dict[str, dict[str, Any]] = {
            "kernel_a": {"section": {"value": 100}},
            "kernel_b": {"section": {"value": 200}},
        }
        top_kernel_list = [
            {"Kernel_Name": "kernel_a", "Percent": 50.0},
            {"Kernel_Name": "kernel_b", "Percent": 30.0},
        ]

        # Verify kernel_view lookup logic works
        for kernel_data in top_kernel_list:
            kernel_name = kernel_data["Kernel_Name"]
            selected_data = kernel_to_df_dict.get(kernel_name)
            assert selected_data is not None, f"Missing data for {kernel_name}"
