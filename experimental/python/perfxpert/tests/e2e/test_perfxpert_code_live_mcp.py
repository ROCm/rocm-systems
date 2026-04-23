"""Live E2E: perfxpert-code → opencode → MCP → Python brain → response.

This test proves the full stack works:
1. perfxpert-code launcher spawns opencode with our bundled config
2. opencode loads the MCP server (perfxpert-mcp)
3. LLM invokes the perfxpert_arch_lookup_peaks tool via MCP
4. Python brain executes and returns MI300X peak specs
5. opencode returns the result to the user

Gated on:
- opencode binary available (path via PERFXPERT_OPENCODE_PATH or shutil.which)
- LLM API key (OPENAI_API_KEY or ANTHROPIC_API_KEY)
- Reasonable timeout (180s for network latency)
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]


def _opencode_available():
    """Check if opencode binary is available."""
    if os.environ.get("PERFXPERT_OPENCODE_PATH"):
        return os.path.isfile(os.environ["PERFXPERT_OPENCODE_PATH"])
    return shutil.which("opencode") is not None


def _llm_key_set():
    """Check if at least one LLM API key is set."""
    return bool(os.environ.get("OPENAI_API_KEY")) or bool(
        os.environ.get("ANTHROPIC_API_KEY")
    )


def _perfxpert_code_cmd() -> list[str]:
    """Use the installed console script, but import perfxpert from this checkout."""
    path = shutil.which("perfxpert-code")
    if path is None:
        pytest.skip("perfxpert-code console script not available on PATH")
    return [path]


def _checkout_python_env() -> dict[str, str]:
    """Run the subprocess against this checkout's perfxpert package."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [str(_CHECKOUT_ROOT)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


@pytest.mark.skipif(
    not _opencode_available(),
    reason="opencode binary not available",
)
@pytest.mark.skipif(
    not _llm_key_set(),
    reason="no LLM API key (OPENAI_API_KEY / ANTHROPIC_API_KEY)",
)
def test_perfxpert_code_calls_mcp_and_returns_expected_value(
    monkeypatch, tmp_path: Path
):
    """Real LLM call through perfxpert-code → opencode → perfxpert-mcp → arch.lookup_peaks."""
    monkeypatch.setenv("PERFXPERT_CODE_NO_BANNER", "1")

    # Pick LLM model based on available key
    if os.environ.get("OPENAI_API_KEY"):
        model = "openai/gpt-4o-mini"
    else:
        model = "anthropic/claude-3-5-haiku-20241022"

    xdg_root = tmp_path / "opencode-xdg"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_root / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg_root / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_root / "cache"))

    # This prompt asks the LLM to use the perfxpert_arch_lookup_peaks tool
    # for MI300X (gfx942) and return only the peak_fp64_tflops value
    prompt = (
        "Use the perfxpert_arch_lookup_peaks tool for gfx_id gfx942, "
        "then tell me only the peak_fp64_tflops value as a number. "
        "Do not include units or other text."
    )

    result = subprocess.run(
        [*_perfxpert_code_cmd(), "run", "-m", model, prompt],
        capture_output=True,
        text=True,
        timeout=180,
        env=_checkout_python_env(),
    )

    # Skip on LLM-provider quota/rate-limit errors — environmental, not a
    # code defect. Mirrors the phase-7 skip-on-429 guard on the integration
    # LLM test (commit 6b390bd88e). Kept deliberately minimal: we string-
    # match stderr for the well-known markers rather than parse SDK error
    # types, since this test goes through the opencode subprocess and the
    # underlying SDK exception is not directly raised here.
    combined = (result.stderr or "") + "\n" + (result.stdout or "")
    low = combined.lower()
    quota_markers = (
        "429",
        "insufficient_quota",
        "rate_limit",
        "rate limit",
        "quota exceeded",  # opencode-rendered form of the OpenAI 429
    )
    if any(m in low for m in quota_markers):
        pytest.skip(
            f"LLM provider returned a rate-limit / quota error "
            f"(environmental, not a code defect): {result.stderr[:500]}"
        )

    if "pragma wal_checkpoint(passive)" in low:
        pytest.skip(
            "upstream opencode local database state is unhealthy "
            "(environmental, not a perfxpert MCP defect)"
        )

    # Check exit code
    assert result.returncode == 0, (
        f"perfxpert-code failed with rc={result.returncode}\n"
        f"stderr: {result.stderr[:500]}"
    )

    # Our MCP tool-call marker appears in stderr (opencode prints progress there)
    assert "perfxpert_arch_lookup_peaks" in result.stderr, (
        f"no MCP tool call marker in stderr\n"
        f"stdout: {result.stdout[:500]}\n"
        f"stderr: {result.stderr[:500]}"
    )

    # The actual number: MI300X peak FP64 TFLOPS = 81.7
    # (not the incorrect 163.4 from some datasheets due to OI counting)
    # This appears in stdout as the final LLM response
    assert "81.7" in result.stdout, (
        f"expected 81.7 in stdout\n"
        f"stdout: {result.stdout[:500]}"
    )
