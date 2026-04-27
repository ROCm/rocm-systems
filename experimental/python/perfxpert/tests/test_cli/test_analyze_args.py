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
from types import SimpleNamespace

import pytest

from perfxpert import analyze
from perfxpert import output_config


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


@pytest.mark.parametrize(
    "provider",
    ["anthropic", "openai", "ollama", "private", "opencode"],
)
def test_llm_flag_all_supported_providers_propagate(provider):
    process_args, ns = _build_parsed_args(["--llm", provider])
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs["llm_provider"] == provider
    assert kwargs["enable_llm"] is True


def test_llm_flag_accepts_claude_code_compatibility_alias():
    process_args, ns = _build_parsed_args(["--llm", "claude-code"])
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs["llm_provider"] == "claude-code"
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


# ---------------------------------------------------------------------------
# Per-flag propagation regression guards (cycle-2 I-1). Every user-facing
# flag in analyze.add_args() gets its own test below, so future flag
# additions have to add a parallel test or trip the warning in
# `_execute_agentic` at runtime.
# ---------------------------------------------------------------------------


def test_prompt_flag_propagates():
    """`--prompt "text"` must surface as `prompt` kwarg (not `custom_prompt`)."""
    process_args, ns = _build_parsed_args(["--prompt", "Why is my matmul slow?"])
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs.get("prompt") == "Why is my matmul slow?"


def test_top_kernels_flag_propagates():
    """`--top-kernels 20` must surface under `top_kernels` kwarg."""
    process_args, ns = _build_parsed_args(["--top-kernels", "20"])
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs.get("top_kernels") == 20


def test_top_kernels_default_suppressed():
    """When `--top-kernels` is not passed, kwargs must NOT carry the default
    (otherwise the downstream agentic code cannot tell user-intent apart
    from argparse-default noise)."""
    process_args, ns = _build_parsed_args([])
    kwargs = process_args(_FakeConn(), ns)
    assert "top_kernels" not in kwargs


def test_att_dir_flag_propagates(tmp_path):
    process_args, ns = _build_parsed_args(["--att-dir", str(tmp_path)])
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs.get("att_dir") == str(tmp_path)


def test_min_duration_flag_propagates_and_converts_to_ns():
    """`--min-duration 5.0` (µs) must become 5000 (ns) in kwargs."""
    process_args, ns = _build_parsed_args(["--min-duration", "5.0"])
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs.get("min_duration") == 5000.0


def test_source_dir_flag_propagates(tmp_path):
    process_args, ns = _build_parsed_args(["--source-dir", str(tmp_path)])
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs.get("source_dir") == str(tmp_path)


def test_llm_api_key_flag_propagates():
    process_args, ns = _build_parsed_args(["--llm-api-key", "sk-test-123"])
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs.get("llm_api_key") == "sk-test-123"


def test_llm_model_flag_propagates():
    process_args, ns = _build_parsed_args(["--llm-model", "claude-opus-4-6"])
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs.get("llm_model") == "claude-opus-4-6"


def test_llm_thinking_flag_propagates():
    process_args, ns = _build_parsed_args(["--llm-thinking", "8000"])
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs.get("llm_thinking") == 8000


def test_llm_local_flag_propagates():
    process_args, ns = _build_parsed_args(["--llm-local", "ollama"])
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs.get("llm_local") == "ollama"


def test_llm_local_model_flag_propagates():
    process_args, ns = _build_parsed_args(["--llm-local-model", "codellama:13b"])
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs.get("llm_local_model") == "codellama:13b"


def test_verbose_flag_propagates_when_set():
    process_args, ns = _build_parsed_args(["--verbose"])
    kwargs = process_args(_FakeConn(), ns)
    assert kwargs.get("verbose") is True


def test_verbose_flag_absent_does_not_leak():
    """Argparse store_true defaults to False; kwargs must not leak that."""
    process_args, ns = _build_parsed_args([])
    kwargs = process_args(_FakeConn(), ns)
    assert "verbose" not in kwargs


def test_execute_agentic_warns_on_unknown_kwargs(tmp_path, monkeypatch):
    """Future-proof: a stray kwarg that isn't wired must emit a WARNING,
    not silently disappear (cycle-2 I-1 root cause)."""
    import warnings
    from perfxpert import analyze as analyze_mod

    # Stub the agent runtime so we don't actually call an LLM.
    class _StubSession:
        session_id = "stub"

        def run_root(self, _payload):
            class _Out:
                narrative = ""
                recommendations = []
                primary_bottleneck = "mixed"
                warnings = []
                metadata: dict = {}
            return _Out()

    def _build_stub(**_kwargs):
        return _StubSession()

    import perfxpert.agents.runtime as runtime_mod
    monkeypatch.setattr(runtime_mod, "build_session", _build_stub)

    # Build a minimal RocpdImportData-shaped object: None is accepted.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyze_mod._execute_agentic(
            None,
            config=None,
            some_future_flag="bogus",
        )
    assert any(
        issubclass(w.category, RuntimeWarning)
        and "some_future_flag" in str(w.message)
        for w in caught
    ), (
        f"expected RuntimeWarning mentioning 'some_future_flag'; got "
        f"{[str(w.message) for w in caught]}"
    )


def _stub_agentic_render(monkeypatch, rendered: str = "REPORT") -> None:
    from perfxpert import api as api_mod
    from perfxpert.analysis import payload as payload_mod
    from perfxpert import analyze as analyze_mod

    monkeypatch.setattr(
        api_mod,
        "agent_root",
        lambda **_kwargs: SimpleNamespace(
            narrative="",
            recommendations=[],
            primary_bottleneck="mixed",
            warnings=[],
            metadata={},
        ),
    )
    monkeypatch.setattr(payload_mod, "build_analysis_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(analyze_mod, "_format_agentic_output", lambda *_args, **_kwargs: rendered)


@pytest.mark.parametrize(
    ("fmt", "ext"),
    [("json", ".json"), ("markdown", ".md"), ("webview", ".html")],
)
def test_non_text_formats_default_to_file_output_without_output_flags(
    tmp_path,
    monkeypatch,
    capsys,
    fmt,
    ext,
):
    from perfxpert import analyze as analyze_mod

    _stub_agentic_render(monkeypatch)
    monkeypatch.chdir(tmp_path)

    cfg = output_config.output_config(output_file=None, output_path=None)
    analyze_mod._execute_agentic(None, config=cfg, output_format=fmt)

    out = tmp_path / f"analysis{ext}"
    assert out.read_text() == "REPORT"
    assert f"Analysis written to: ./{out.name}" in capsys.readouterr().out


def test_non_text_format_with_output_name_defaults_to_current_directory(
    tmp_path,
    monkeypatch,
):
    from perfxpert import analyze as analyze_mod

    _stub_agentic_render(monkeypatch, rendered="{}")
    monkeypatch.chdir(tmp_path)

    cfg = output_config.output_config(output_file="custom-report", output_path=None)
    analyze_mod._execute_agentic(None, config=cfg, output_format="json")

    assert (tmp_path / "custom-report.json").read_text() == "{}"


def test_text_format_without_output_flags_stays_on_stdout(
    tmp_path,
    monkeypatch,
    capsys,
):
    from perfxpert import analyze as analyze_mod

    _stub_agentic_render(monkeypatch)
    monkeypatch.chdir(tmp_path)

    cfg = output_config.output_config(output_file=None, output_path=None)
    analyze_mod._execute_agentic(None, config=cfg, output_format="text")

    assert capsys.readouterr().out == "REPORT\n"
    assert list(tmp_path.iterdir()) == []
