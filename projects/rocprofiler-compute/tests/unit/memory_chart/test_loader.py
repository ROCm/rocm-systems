# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for the memory chart layout loader."""

import pytest

from memory_chart.loader import (
    list_architectures,
    list_layout_files,
    load_layout,
)


class TestLoadLayout:
    def test_load_gfx908_resolves_to_gfx90x(self) -> None:
        layout = load_layout("gfx908")
        assert layout["arch"] == "gfx90x"

    def test_load_gfx90a_resolves_to_gfx90x(self) -> None:
        layout = load_layout("gfx90a")
        assert layout["arch"] == "gfx90x"

    def test_load_gfx941_resolves_to_gfx94x(self) -> None:
        layout = load_layout("gfx941")
        assert layout["arch"] == "gfx94x"

    def test_load_gfx950_returns_gfx950(self) -> None:
        layout = load_layout("gfx950")
        assert layout["arch"] == "gfx950"

    def test_load_gfx115x_returns_gfx115x(self) -> None:
        layout = load_layout("gfx115x")
        assert layout["arch"] == "gfx115x"

    def test_load_gfx1250_returns_gfx1250(self) -> None:
        layout = load_layout("gfx1250")
        assert layout["arch"] == "gfx1250"

    def test_unknown_arch_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown architecture"):
            load_layout("gfx_nonexistent")

    def test_returned_layout_has_blocks_and_arrows(self) -> None:
        layout = load_layout("gfx942")
        assert isinstance(layout["blocks"], list)
        assert isinstance(layout["arrows"], list)
        assert len(layout["blocks"]) > 0
        assert len(layout["arrows"]) > 0


class TestListArchitectures:
    def test_returns_sorted_list(self) -> None:
        archs = list_architectures()
        assert archs == sorted(archs)

    def test_includes_expected_archs(self) -> None:
        archs = list_architectures()
        for expected in (
            "gfx908",
            "gfx90a",
            "gfx940",
            "gfx942",
            "gfx950",
            "gfx115x",
            "gfx1250",
        ):
            assert expected in archs


class TestListLayoutFiles:
    def test_returns_five_unique_files(self) -> None:
        files = list_layout_files()
        assert len(files) == 5

    def test_excludes_manifest(self) -> None:
        files = list_layout_files()
        names = {f.name for f in files}
        assert "manifest.json" not in names
