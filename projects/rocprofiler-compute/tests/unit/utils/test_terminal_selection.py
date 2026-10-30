# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/terminal_selection.py."""

import pytest

from utils.terminal_selection import (
    TerminalSelection,
    build_terminal_selection,
    table_id_string,
)

WAVEFRONT_PANEL_ID = 700
WAVEFRONT_TABLE_ID = 701
LAUNCH_STATS_TABLE_ID = 702


def use_alias_map(monkeypatch, alias_map):
    """Resolve block aliases from alias_map instead of the arch configs."""
    monkeypatch.setattr(
        "utils.utils_common.get_arch_alias_to_panel_id", lambda arch: alias_map
    )


def test_table_id_string():
    assert table_id_string(1101) == "11.1"


@pytest.mark.parametrize("panel_id", [0, 100, 200])
def test_speed_of_light_and_earlier_panels_are_always_shown(panel_id):
    selection = TerminalSelection()

    assert selection.shows_panel(panel_id)
    assert selection.shows_table(panel_id, panel_id + 1)
    assert selection.shows_metric(panel_id, panel_id + 1, f"{panel_id // 100}.1.0")


def test_later_panels_are_hidden_without_block_tokens():
    selection = TerminalSelection()

    assert not selection.shows_panel(300)
    assert not selection.shows_panel(WAVEFRONT_PANEL_ID)


def test_block_token_shows_every_table_and_metric_of_the_block():
    selection = TerminalSelection(metric_ids=frozenset({"7"}))

    assert selection.shows_panel(WAVEFRONT_PANEL_ID)
    assert selection.shows_table(WAVEFRONT_PANEL_ID, LAUNCH_STATS_TABLE_ID)
    assert selection.shows_metric(WAVEFRONT_PANEL_ID, WAVEFRONT_TABLE_ID, "7.1.4")
    assert not selection.shows_panel(1000)


def test_table_token_shows_only_that_table():
    selection = TerminalSelection(metric_ids=frozenset({"7.1"}))

    assert selection.shows_panel(WAVEFRONT_PANEL_ID)
    assert selection.shows_table(WAVEFRONT_PANEL_ID, WAVEFRONT_TABLE_ID)
    assert not selection.shows_table(WAVEFRONT_PANEL_ID, LAUNCH_STATS_TABLE_ID)
    assert selection.shows_metric(WAVEFRONT_PANEL_ID, WAVEFRONT_TABLE_ID, "7.1.4")


def test_metric_token_shows_its_row_and_hides_siblings():
    selection = TerminalSelection(metric_ids=frozenset({"7.1.5"}))

    assert selection.shows_panel(WAVEFRONT_PANEL_ID)
    assert selection.shows_table(WAVEFRONT_PANEL_ID, WAVEFRONT_TABLE_ID)
    assert not selection.shows_table(WAVEFRONT_PANEL_ID, LAUNCH_STATS_TABLE_ID)
    assert selection.shows_metric(WAVEFRONT_PANEL_ID, WAVEFRONT_TABLE_ID, "7.1.5")
    assert not selection.shows_metric(WAVEFRONT_PANEL_ID, WAVEFRONT_TABLE_ID, "7.1.4")


def test_metric_token_does_not_match_a_longer_id_with_the_same_prefix():
    selection = TerminalSelection(metric_ids=frozenset({"7.1.1"}))

    assert not selection.shows_metric(WAVEFRONT_PANEL_ID, WAVEFRONT_TABLE_ID, "7.1.10")
    assert not selection.shows_table(WAVEFRONT_PANEL_ID, 711)


def test_alias_resolves_to_the_whole_panel(monkeypatch):
    use_alias_map(monkeypatch, {"wavefront": "7"})

    selection = build_terminal_selection(["wavefront"], "gfx942")

    assert selection == TerminalSelection(panel_ids=frozenset({WAVEFRONT_PANEL_ID}))
    assert selection.shows_metric(WAVEFRONT_PANEL_ID, LAUNCH_STATS_TABLE_ID, "7.2.0")


def test_mixed_numeric_and_alias_tokens(monkeypatch):
    use_alias_map(monkeypatch, {"lds": "12"})

    selection = build_terminal_selection(["7.1.5", "lds"], "gfx942")

    assert selection == TerminalSelection(
        metric_ids=frozenset({"7.1.5"}), panel_ids=frozenset({1200})
    )


def test_no_tokens_build_an_empty_selection():
    assert build_terminal_selection(None, "gfx942") == TerminalSelection()


def test_unknown_alias_errors(monkeypatch):
    use_alias_map(monkeypatch, {"lds": "12"})

    with pytest.raises(SystemExit):
        build_terminal_selection(["bogus"], "gfx942")
