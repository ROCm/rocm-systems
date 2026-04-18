"""Tests for `perfxpert analyze` CLI argument propagation.

Regression for E2E bug 1: `--format` and `--llm` were silently dropped
because argparse emitted them as `format=` / `llm=` while
`_execute_agentic` reads `output_format=` / `llm_provider=`.

This test suite asserts that the CLI surface maps `--format` to
`output_format`, `--llm` to `llm_provider`, and that the derived
`enable_llm` flag flips to True when `--llm <provider>` is passed.
"""

from __future__ import annotations

import argparse

import pytest

from perfxpert import analyze


def _build_parsed_args(argv):
    """Build a parsed argparse.Namespace from a minimal analyze parser."""
    parser = argparse.ArgumentParser()
    process_args = analyze.add_args(parser)
    # `add_args` does not register -i/--input; the top-level wrappers do.
    # For arg-propagation tests we don't need -i, just the analysis flags.
    ns = parser.parse_args(argv)
    return process_args, ns


class _FakeConn:
    """Stand-in for a RocpdImportData-style object; unused by propagation checks."""


def test_format_flag_propagates():
    """`--format json` must surface as `output_format="json"` in process_args."""
    process_args, ns = _build_parsed_args(["--format", "json"])
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs.get("output_format") == "json", (
        f"--format must map to output_format kwarg; got {kwargs}"
    )
    # Defensive: the legacy name must NOT leak through; _execute_agentic
    # looks for `output_format`, not `format`.
    assert "format" not in kwargs


@pytest.mark.parametrize("fmt", ["text", "json", "markdown", "webview"])
def test_format_flag_all_choices_propagate(fmt):
    process_args, ns = _build_parsed_args(["--format", fmt])
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs["output_format"] == fmt


def test_llm_flag_propagates():
    """`--llm openai` must surface as `llm_provider="openai"` AND set
    `enable_llm=True` so `_execute_agentic` activates the live path."""
    process_args, ns = _build_parsed_args(["--llm", "openai"])
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs.get("llm_provider") == "openai", (
        f"--llm must map to llm_provider kwarg; got {kwargs}"
    )
    assert kwargs.get("enable_llm") is True, (
        "passing --llm must flip enable_llm so agentic runtime uses the provider"
    )
    # Defensive: the pre-rename name must NOT leak through.
    assert "llm" not in kwargs


@pytest.mark.parametrize("provider", ["anthropic", "openai", "claude-code"])
def test_llm_flag_all_choices_propagate(provider):
    process_args, ns = _build_parsed_args(["--llm", provider])
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs["llm_provider"] == provider
    assert kwargs["enable_llm"] is True


def test_llm_flag_absent_does_not_set_enable_llm():
    """When `--llm` is not passed, enable_llm should not appear in kwargs
    (the caller falls back to its default, which is False)."""
    process_args, ns = _build_parsed_args([])
    kwargs = process_args(_FakeConn(), ns)
    assert "llm_provider" not in kwargs
    assert "enable_llm" not in kwargs


def test_format_and_llm_flags_compose():
    """Both flags set in the same invocation must both propagate."""
    process_args, ns = _build_parsed_args(
        ["--format", "markdown", "--llm", "anthropic"]
    )
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs["output_format"] == "markdown"
    assert kwargs["llm_provider"] == "anthropic"
    assert kwargs["enable_llm"] is True
