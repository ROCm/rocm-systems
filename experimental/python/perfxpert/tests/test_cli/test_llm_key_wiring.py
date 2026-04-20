"""Tests for `--llm-api-key` CLI flag wiring through the agentic runtime
and the pre-flight auth check that precedes every live LLM call.

These tests exercise the Bug 1 (flag wiring) and Bug 3 (pre-flight
credential validation) fixes landed in Phase 8. Empty-response and
end-to-end output-file-cleanup tests live in companion commits.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from perfxpert import analyze
from perfxpert.providers._exceptions import AuthError


# ---------------------------------------------------------------------------
# 1 — --llm-api-key flag overrides env and reaches the provider layer.
# ---------------------------------------------------------------------------


def test_cli_api_key_flag_overrides_env(monkeypatch, tmp_path):
    """When the user passes ``--llm-api-key`` on the CLI AND
    ``ANTHROPIC_API_KEY`` is set to a different value, the flag wins:
    ``_cascade`` wraps every provider attempt in
    ``_override_provider_env`` so the provider layer sees the flag value
    in ``ANTHROPIC_API_KEY``, then the env var is restored on exit.

    This test drives the mechanism at the session level (the one
    ``agent_root`` uses) so we can observe the real env swap without
    a live LLM call.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-wrong-from-env")

    from perfxpert.agents import runtime as runtime_mod
    from perfxpert.agents import schemas

    seen: dict = {}

    def _capture_root(payload, provider="anthropic"):
        # Inside the cascade the env override is active; record what
        # the provider layer would read.
        seen["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY")
        return schemas.RootOutput(
            narrative="ok",
            recommendations=[],
            primary_bottleneck="mixed",
            warnings=[],
            metadata={},
        )

    session = runtime_mod.build_session(
        provider="anthropic", api_key="sk-right-one", airgap=False
    )
    with patch("perfxpert.agents.root.run_root", side_effect=_capture_root):
        session.run_root(schemas.RootInput(user_query="?"))

    # The flag value must be the one visible to the provider layer.
    assert seen["ANTHROPIC_API_KEY"] == "sk-right-one"
    # And the env var must be restored after the call.
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-wrong-from-env"


def test_execute_agentic_forwards_api_key_to_agent_root(
    monkeypatch, tmp_path
):
    """``_execute_agentic`` must pass ``llm_api_key`` through to the
    ``perfxpert.api.agent_root`` call as ``api_key=``. This is the wiring
    that used to silently drop the flag."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-wrong-from-env")
    seen: dict = {}

    def _capture(*args, **kwargs):
        seen.update(kwargs)
        return {
            "narrative": "ok",
            "recommendations": [],
            "primary_bottleneck": "mixed",
            "warnings": [],
            "metadata": {},
        }

    with patch("perfxpert.api.agent_root", side_effect=_capture):
        analyze._execute_agentic(
            None,
            config=None,
            source_dir=str(tmp_path),
            output_format="text",
            enable_llm=True,
            llm_provider="anthropic",
            llm_api_key="sk-right-one",
        )

    assert seen.get("api_key") == "sk-right-one", (
        f"analyze._execute_agentic must forward llm_api_key as api_key; "
        f"got kwargs={seen!r}"
    )


def test_cli_api_key_flag_warns_when_differs_from_env(
    monkeypatch, tmp_path, capsys
):
    """When flag and env differ, we emit a one-line stderr WARNING naming
    the env var that got overridden."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-wrong-from-env")
    with patch(
        "perfxpert.api.agent_root",
        return_value={
            "narrative": "ok",
            "recommendations": [],
            "primary_bottleneck": "mixed",
            "warnings": [],
            "metadata": {},
        },
    ):
        analyze._execute_agentic(
            None,
            config=None,
            source_dir=str(tmp_path),
            output_format="text",
            enable_llm=True,
            llm_provider="anthropic",
            llm_api_key="sk-right-one",
        )
    captured = capsys.readouterr()
    # Warning must mention the env var being overridden.
    assert "ANTHROPIC_API_KEY" in captured.err
    assert "--llm-api-key" in captured.err


def test_cli_api_key_flag_no_warning_when_matches_env(
    monkeypatch, tmp_path, capsys
):
    """When the flag equals the env value, no warning is emitted."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-same")
    with patch(
        "perfxpert.api.agent_root",
        return_value={
            "narrative": "ok",
            "recommendations": [],
            "primary_bottleneck": "mixed",
            "warnings": [],
            "metadata": {},
        },
    ):
        analyze._execute_agentic(
            None,
            config=None,
            source_dir=str(tmp_path),
            output_format="text",
            enable_llm=True,
            llm_provider="anthropic",
            llm_api_key="sk-same",
        )
    captured = capsys.readouterr()
    assert "--llm-api-key overrides" not in captured.err


# ---------------------------------------------------------------------------
# 2 — Pre-flight auth check raises AuthError before any HTTP call.
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_auth_error_preflight(monkeypatch, tmp_path):
    """With no env var AND no flag, ``_execute_agentic`` must raise
    :class:`AuthError` BEFORE any provider call."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PERFXPERT_LLM_ANTHROPIC_KEY", raising=False)

    # agent_root must NOT be reached.
    with patch("perfxpert.api.agent_root") as mock_root:
        with pytest.raises(AuthError) as excinfo:
            analyze._execute_agentic(
                None,
                config=None,
                source_dir=str(tmp_path),
                output_format="webview",
                enable_llm=True,
                llm_provider="anthropic",
            )
        mock_root.assert_not_called()
    # The error names the provider + tells the user how to fix it.
    msg = str(excinfo.value)
    assert "anthropic" in msg
    assert "--llm-api-key" in msg or "ANTHROPIC_API_KEY" in msg


# ---------------------------------------------------------------------------
# 3 — Build session passes api_key through to session.
# ---------------------------------------------------------------------------


def test_build_session_accepts_api_key():
    """``build_session`` stores the api_key on the session so ``_cascade``
    can wrap provider calls in the env-override context."""
    from perfxpert.agents import runtime as runtime_mod

    session = runtime_mod.build_session(
        provider="anthropic", api_key="sk-test", airgap=False
    )
    assert session.api_key == "sk-test"


def test_build_session_airgap_discards_api_key():
    """Airgap sessions must never hold an api_key — there is no live call
    that could use it."""
    from perfxpert.agents import runtime as runtime_mod

    session = runtime_mod.build_session(
        provider="anthropic", api_key="sk-test", airgap=True
    )
    assert session.api_key is None
