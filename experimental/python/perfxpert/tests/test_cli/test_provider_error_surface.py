"""Focused regression tests for analyze.py provider taxonomy handling."""

from __future__ import annotations

from unittest import mock

import pytest

from perfxpert import analyze as analyze_mod
from perfxpert.providers._exceptions import (
    FatalError,
    ProviderChainExhausted,
    QuotaExceededError,
    RateLimitError,
    TimeoutError as ProviderTimeoutError,
    TransientError,
)


@pytest.mark.parametrize(
    ("exc", "fragments"),
    [
        (
            QuotaExceededError("openai", model="gpt-4o", message="quota reached"),
            ["quota exhausted", "openai", "gpt-4o", "quota reached"],
        ),
        (
            ProviderTimeoutError("ollama", timeout_seconds=12.5, message="slow response"),
            ["timed out after 12.5s", "ollama"],
        ),
        (
            TransientError("anthropic", kind="api_unavailable", message="try later"),
            ["transient error", "anthropic", "api_unavailable"],
        ),
        (
            FatalError("private", "schema mismatch"),
            ["provider private failed", "schema mismatch"],
        ),
    ],
)
def test_render_cli_error_handles_typed_provider_exceptions(exc, fragments, capsys):
    assert analyze_mod._render_cli_error(exc) == 1
    captured = capsys.readouterr()
    for fragment in fragments:
        assert fragment in captured.err


def test_render_cli_error_handles_provider_chain_exhaustion(capsys):
    exc = ProviderChainExhausted(
        ["anthropic", "openai"],
        last_error=RateLimitError("openai", retry_after=30.0),
    )

    assert analyze_mod._render_cli_error(exc) == 1
    captured = capsys.readouterr()
    assert "fallback chain exhausted" in captured.err
    assert "anthropic -> openai" in captured.err
    assert "rate limited" in captured.err


def test_execute_agentic_normalizes_claude_code_provider_alias(capsys):
    fake_input = mock.Mock()
    fake_input._paths = ["/tmp/fake.db"]

    with mock.patch("perfxpert.api.agent_root", return_value={}) as agent_root:
        with mock.patch(
            "perfxpert.analysis.payload.build_analysis_payload",
            return_value={},
        ):
            with mock.patch(
                "perfxpert.analyze._format_agentic_output",
                return_value="rendered report",
            ):
                analyze_mod._execute_agentic(
                    input=fake_input,
                    output_format="text",
                    llm_provider="claude-code",
                    enable_llm=True,
                )

    captured = capsys.readouterr()
    assert captured.out.strip() == "rendered report"
    assert agent_root.call_args.kwargs["provider"] == "opencode"
