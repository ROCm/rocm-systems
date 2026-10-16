# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_tui/widgets/collapsibles.py."""

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.tui

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
