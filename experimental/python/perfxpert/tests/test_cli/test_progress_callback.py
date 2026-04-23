"""Tests for airgap-aware progress/preflight behavior in ``perfxpert analyze``.

Covers:
  * env-forced airgap skips provider-auth preflight even when
    ``enable_llm=True`` is requested for progress UX,
  * CLI non-TTY behaviour (plain ``[perfxpert]`` status lines on stderr,
    real output on stdout — no ANSI escape codes),
  * ``--no-progress`` silences the feature entirely.

All tests use airgap mode so nothing hits an LLM provider.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from perfxpert import analyze as analyze_mod


def test_execute_agentic_env_airgap_skips_auth_preflight(monkeypatch):
    """``PERFXPERT_AIRGAP=1`` must bypass provider auth preflight and clear
    provider credentials before calling the agentic root."""
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PERFXPERT_LLM_ANTHROPIC_KEY", raising=False)

    with patch("perfxpert.analyze._preflight_provider_auth") as mock_preflight, patch(
        "perfxpert.analysis.payload.build_analysis_payload",
        return_value={},
    ), patch(
        "perfxpert.analyze._format_agentic_output",
        return_value="ok",
    ), patch(
        "perfxpert.api.agent_root",
        return_value={
            "narrative": "ok",
            "recommendations": [],
            "primary_bottleneck": "mixed",
            "warnings": [],
            "metadata": {},
        },
    ) as mock_root:
        analyze_mod._execute_agentic(
            None,
            config=None,
            output_format="text",
            enable_llm=True,
            llm_provider="anthropic",
        )

    mock_preflight.assert_not_called()
    assert mock_root.call_args.kwargs["airgap"] is True
    assert mock_root.call_args.kwargs["provider"] is None
    assert mock_root.call_args.kwargs["api_key"] is None
    assert mock_root.call_args.kwargs["progress_callback"] is None


# --- CLI non-TTY: plain status lines on stderr ---------------------------


_CLI_PRELUDE = (
    # Run analyze.main() directly inside a subprocess so we get a real
    # (non-TTY) stderr pipe. We skip the argparse `-i` requirement by
    # driving _execute_agentic through the process_args path.
    "import sys\n"
    "from unittest.mock import patch\n"
    "from perfxpert import analyze\n"
    "def _fake_agent_root(**kwargs):\n"
    "    cb = kwargs.get('progress_callback')\n"
    "    if cb is not None:\n"
    "        cb('Routing your query (Root agent)')\n"
    "        cb('Root agent done')\n"
    "    return {{\n"
    "        'narrative': 'ok',\n"
    "        'recommendations': [],\n"
    "        'primary_bottleneck': 'mixed',\n"
    "        'warnings': [],\n"
    "        'metadata': {{}},\n"
    "    }}\n"
    "with patch('perfxpert.analyze._preflight_provider_auth'), \\\n"
    "     patch('perfxpert.analysis.payload.build_analysis_payload', return_value={{}}), \\\n"
    "     patch('perfxpert.analyze._format_agentic_output', return_value='{{\"status\":\"ok\"}}'), \\\n"
    "     patch('perfxpert.api.agent_root', side_effect=_fake_agent_root):\n"
    "    analyze._execute_agentic(\n"
    "        None,\n"
    "        config=None,\n"
    "        source_dir='.',\n"
    "        output_format='json',\n"
    "        enable_llm=True,\n"
    "        llm_provider='anthropic',\n"
    "        no_progress={no_progress},\n"
    "    )\n"
)


def _ansi_free(text: str) -> bool:
    """True if the string contains no CSI / OSC escape sequences."""
    return re.search(r"\x1b\[[0-9;]*[A-Za-z]", text) is None


@pytest.fixture
def _airgap_env(monkeypatch, tmp_path):
    env = os.environ.copy()
    env["PERFXPERT_AIRGAP"] = "1"
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("PERFXPERT_LLM_ANTHROPIC_KEY", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH", "")]
    )
    return env


def test_analyze_cli_non_tty_prints_status_lines(_airgap_env):
    """Subprocess run of ``_execute_agentic`` with live LLM mode and a piped
    stderr emits the ``[perfxpert]`` status prefix on stderr and the
    JSON result on stdout. No ANSI escapes in either stream.
    """
    env = dict(_airgap_env)
    env.pop("PERFXPERT_AIRGAP", None)
    code = _CLI_PRELUDE.format(no_progress="False")
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert res.returncode == 0, (
        f"exit={res.returncode}\nstderr={res.stderr!r}\nstdout={res.stdout!r}"
    )
    # stdout should be JSON-ish (the formatted agentic output).
    assert res.stdout.strip(), "stdout must contain the analysis output"
    assert _ansi_free(res.stdout), "stdout must be ANSI-free when piped"
    # stderr has at least one [perfxpert] status line.
    assert "[perfxpert]" in res.stderr, (
        f"expected plain status lines on stderr; got {res.stderr!r}"
    )
    # No spinner escape codes on a piped stream.
    assert _ansi_free(res.stderr), (
        f"stderr escape codes leaked on non-TTY: {res.stderr!r}"
    )


def test_analyze_cli_env_airgap_suppresses_llm_progress(_airgap_env):
    """Env-forced airgap must suppress LLM-only progress UX even when
    ``enable_llm=True`` was requested by the caller.
    """
    code = _CLI_PRELUDE.format(no_progress="False")
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=_airgap_env,
        timeout=30,
    )
    assert res.returncode == 0, (
        f"exit={res.returncode}\nstderr={res.stderr!r}\nstdout={res.stdout!r}"
    )
    assert res.stdout.strip(), "stdout must contain the analysis output"
    assert _ansi_free(res.stdout), "stdout must be ANSI-free when piped"
    assert "[perfxpert]" not in res.stderr, (
        "PERFXPERT_AIRGAP=1 must disable LLM-only progress UX; "
        f"got stderr={res.stderr!r}"
    )
    assert _ansi_free(res.stderr), "stderr must remain ANSI-free when piped"


def test_analyze_cli_no_progress_flag_silent(_airgap_env):
    """When ``--no-progress`` is set, the CLI suppresses the
    ``[perfxpert]`` status prefix even in LLM mode. stdout is unchanged.
    """
    env = dict(_airgap_env)
    env.pop("PERFXPERT_AIRGAP", None)
    code = _CLI_PRELUDE.format(no_progress="True")
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert res.returncode == 0, (
        f"exit={res.returncode}\nstderr={res.stderr!r}\nstdout={res.stdout!r}"
    )
    assert res.stdout.strip(), "stdout must still contain the analysis output"
    assert "[perfxpert]" not in res.stderr, (
        f"--no-progress must silence status lines; got stderr={res.stderr!r}"
    )
