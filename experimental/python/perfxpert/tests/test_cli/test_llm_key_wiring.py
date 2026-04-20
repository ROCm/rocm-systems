"""Tests for `--llm-api-key` CLI flag wiring through the agentic runtime,
the pre-flight auth check that precedes every live LLM call, and the
empty-response ``FatalError`` guard that prevents the formatter from
writing a blank HTML / markdown report.

These tests exercise the Bug 1 (flag wiring), Bug 2 (empty-response
guard), and Bug 3 (pre-flight credential validation) fixes landed in
Phase 8.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from perfxpert import analyze
from perfxpert.providers._exceptions import AuthError, FatalError


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

    def _capture_root(payload, provider="anthropic", **_kw):
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


# ---------------------------------------------------------------------------
# 4 — Empty LLM response raises FatalError instead of silently writing
#     an empty HTML / markdown file.
# ---------------------------------------------------------------------------


def test_empty_llm_response_raises_fatal(monkeypatch, tmp_path):
    """When ``agent_root`` returns an empty narrative, ``run_root`` raises
    ``FatalError`` so the CLI doesn't write a blank report."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-dummy")

    # Bypass the live network entirely — exercise run_root's guard via
    # session build + a stub that returns an empty-narrative RootOutput.
    from perfxpert.agents import runtime as runtime_mod
    from perfxpert.agents import schemas

    def _empty_root(payload, provider="anthropic", **_kw):
        return schemas.RootOutput(
            narrative="",
            recommendations=[],
            primary_bottleneck="mixed",
            warnings=[],
            metadata={},
        )

    session = runtime_mod.build_session(provider="anthropic", airgap=False)
    with patch("perfxpert.agents.root.run_root", side_effect=_empty_root):
        with pytest.raises(FatalError) as excinfo:
            session.run_root(schemas.RootInput(user_query="?"))
    assert "empty response" in str(excinfo.value).lower()


def test_airgap_empty_narrative_does_not_raise():
    """The airgap path must not trigger the empty-response guard (its
    template narrative is deterministic and always populated; guard
    is off in airgap mode)."""
    from perfxpert.agents import runtime as runtime_mod
    from perfxpert.agents import schemas

    session = runtime_mod.build_session(airgap=True)
    # Airgap returns a real narrative; no exception expected.
    out = session.run_root(schemas.RootInput(user_query="?"))
    assert out is not None


# ---------------------------------------------------------------------------
# 5 — End-to-end: CLI exits rc=2 on ProviderError and leaves no empty
#     output file behind.
# ---------------------------------------------------------------------------


def test_preflight_auth_error_main_returns_rc_2(monkeypatch, tmp_path):
    """The outer ``analyze.main()`` wraps the AuthError as rc=2 (distinct
    from the generic rc=1 bucket) and writes no HTML file."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PERFXPERT_LLM_ANTHROPIC_KEY", raising=False)

    # Need an input arg for the argparser; use an empty placeholder.
    db = tmp_path / "empty.db"
    db.write_bytes(b"")
    outdir = tmp_path / "out"
    outdir.mkdir()
    rc = analyze.main(
        [
            "-i",
            str(db),
            "--llm",
            "anthropic",
            "--format",
            "webview",
            "-d",
            str(outdir),
        ]
    )
    assert rc == 2, f"expected rc=2 for AuthError, got {rc}"
    # No output file written.
    assert not any(p.name.endswith(".html") for p in outdir.iterdir()), (
        f"auth pre-flight failure must not produce HTML; dir={list(outdir.iterdir())}"
    )


def test_webview_no_empty_file_on_failure(tmp_path):
    """Full subprocess CLI invocation: patch ``agent_root`` to raise
    ``FatalError`` and verify ``perfxpert analyze`` exits rc=2 with a
    clean stderr one-liner and no empty *.html file."""
    db = tmp_path / "fake.db"
    db.write_bytes(b"")
    outdir = tmp_path / "out"
    outdir.mkdir()

    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = "sk-clearly-bogus"
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(__file__).resolve().parents[2]),
            env.get("PYTHONPATH", ""),
        ]
    )

    driver = (
        "import sys\n"
        "from unittest.mock import patch\n"
        "from perfxpert.providers._exceptions import FatalError\n"
        "from perfxpert import analyze\n"
        "def raise_fatal(*a, **kw):\n"
        "    raise FatalError('anthropic', 'bogus key rejected')\n"
        "with patch('perfxpert.api.agent_root', side_effect=raise_fatal):\n"
        f"    rc = analyze.main(['-i', {str(db)!r}, '--llm', 'anthropic', "
        f"'--format', 'webview', '-d', {str(outdir)!r}])\n"
        "sys.exit(rc)\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", driver],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    # rc=2 for ProviderError.
    assert res.returncode == 2, (
        f"expected rc=2, got {res.returncode}\nstdout={res.stdout!r}\n"
        f"stderr={res.stderr!r}"
    )
    # stderr has a one-liner naming the provider.
    assert "anthropic" in res.stderr.lower()
    # No empty HTML file left behind.
    leftover = [p for p in outdir.iterdir() if p.suffix == ".html"]
    assert not leftover, (
        f"CLI must not produce *.html on FatalError; found {leftover}"
    )
