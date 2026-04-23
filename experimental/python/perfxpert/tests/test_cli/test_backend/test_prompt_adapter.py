"""Tests for `perfxpert.cli._backend._prompt_adapter` (Task 4a).

Covers each shared helper:

* render_prompt — substitution, backend-block stripping, rejection
  stanza toggle.
* emit_marker_block — sentinel format + cache-hash.
* is_git_tracked — tracked vs untracked vs no-git cases.
* atomic_write — .bak retention + no partial write on crash.
* stage_cache_file — returns the cache hash.
* retry_mcp_handshake — exponential backoff, budget early-exit,
  success propagation.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from perfxpert.cli._backend import _prompt_adapter as pa


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "prompts"
SOURCE = FIXTURES_DIR / "agents_source.md"
TOOLS: tuple[str, ...] = ("intent_classify", "next_step", "report")


# ---------------------------------------------------------------------------
# render_prompt.
# ---------------------------------------------------------------------------


def test_render_substitutes_tool_names_claude() -> None:
    out = pa.render_prompt(
        SOURCE,
        backend="claude",
        tool_name_template="mcp__perfxpert__{tool}",
        known_tools=TOOLS,
        reject_language=False,
    )
    assert "mcp__perfxpert__intent_classify" in out
    assert "mcp__perfxpert__next_step" in out
    assert "mcp__perfxpert__report" in out
    # Original tokens must be gone.
    assert "perfxpert_intent_classify" not in out
    assert "perfxpert_next_step" not in out
    assert "perfxpert_report" not in out


def test_render_substitutes_tool_names_gemini() -> None:
    out = pa.render_prompt(
        SOURCE,
        backend="gemini",
        tool_name_template="mcp_perfxpert_{tool}",
        known_tools=TOOLS,
        reject_language=False,
    )
    assert "mcp_perfxpert_intent_classify" in out


def test_render_strips_non_target_backend_blocks() -> None:
    out = pa.render_prompt(
        SOURCE,
        backend="claude",
        tool_name_template="mcp__perfxpert__{tool}",
        known_tools=TOOLS,
        reject_language=False,
    )
    # Claude block preserved (body kept; markers removed).
    assert "Claude-specific" in out
    # Gemini + Codex blocks stripped entirely.
    assert "Gemini-specific" not in out
    assert "Codex-specific" not in out
    # Marker lines themselves are gone.
    assert "<!--backend:claude-->" not in out
    assert "<!--backend:gemini-->" not in out


def test_render_includes_rejection_stanza_when_true() -> None:
    out = pa.render_prompt(
        SOURCE,
        backend="claude",
        tool_name_template="mcp__perfxpert__{tool}",
        known_tools=TOOLS,
        reject_language=True,
    )
    assert "WILL BE REJECTED" in out
    # Rejection stanza references the rendered classify tool name.
    assert "mcp__perfxpert__intent_classify" in out


def test_render_omits_rejection_stanza_when_false() -> None:
    out = pa.render_prompt(
        SOURCE,
        backend="claude",
        tool_name_template="mcp__perfxpert__{tool}",
        known_tools=TOOLS,
        reject_language=False,
    )
    assert "WILL BE REJECTED" not in out


def test_render_accepts_inline_string_source() -> None:
    src = "Call perfxpert_intent_classify and then perfxpert_report."
    out = pa.render_prompt(
        src,
        backend="claude",
        tool_name_template="mcp__perfxpert__{tool}",
        known_tools=TOOLS,
        reject_language=False,
    )
    assert "mcp__perfxpert__intent_classify" in out
    assert "mcp__perfxpert__report" in out


# ---------------------------------------------------------------------------
# emit_marker_block.
# ---------------------------------------------------------------------------


def test_marker_block_format() -> None:
    begin, end = pa.emit_marker_block("hello world")
    assert begin.startswith("<!-- BEGIN perfxpert-managed v1 cache=")
    assert begin.endswith(" -->")
    assert end == "<!-- END perfxpert-managed v1 -->"


def test_marker_block_cache_hash_is_content_derived() -> None:
    b1, _ = pa.emit_marker_block("content A")
    b2, _ = pa.emit_marker_block("content B")
    assert b1 != b2, "different content must produce different cache hashes"


def test_marker_block_cache_hash_is_deterministic() -> None:
    b1, _ = pa.emit_marker_block("same content")
    b2, _ = pa.emit_marker_block("same content")
    assert b1 == b2


# ---------------------------------------------------------------------------
# is_git_tracked.
# ---------------------------------------------------------------------------


def _init_repo(tmp: Path) -> None:
    subprocess.run(
        ["git", "init", "-q", "--initial-branch=main"],
        cwd=str(tmp),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(tmp),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=str(tmp),
        check=True,
        capture_output=True,
    )


def test_is_git_tracked_true(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    f = tmp_path / "tracked.md"
    f.write_text("hi")
    subprocess.run(["git", "add", "tracked.md"], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=str(tmp_path), check=True
    )
    assert pa.is_git_tracked(f, tmp_path) is True


def test_is_git_tracked_false_for_untracked(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    f = tmp_path / "untracked.md"
    f.write_text("hi")
    assert pa.is_git_tracked(f, tmp_path) is False


def test_is_git_tracked_false_when_not_a_repo(tmp_path: Path) -> None:
    f = tmp_path / "anywhere.md"
    f.write_text("hi")
    assert pa.is_git_tracked(f, tmp_path) is False


# ---------------------------------------------------------------------------
# atomic_write.
# ---------------------------------------------------------------------------


def test_atomic_write_new_file(tmp_path: Path) -> None:
    target = tmp_path / "x.md"
    pa.atomic_write(target, "v1")
    assert target.read_text() == "v1"
    assert not (tmp_path / "x.md.bak").exists()
    assert not (tmp_path / "x.md.tmp").exists()


def test_atomic_write_backup_on_rewrite(tmp_path: Path) -> None:
    target = tmp_path / "x.md"
    pa.atomic_write(target, "v1")
    pa.atomic_write(target, "v2")
    assert target.read_text() == "v2"
    bak = tmp_path / "x.md.bak"
    assert bak.exists()
    assert bak.read_text() == "v1"


def test_atomic_write_backup_disabled(tmp_path: Path) -> None:
    target = tmp_path / "x.md"
    pa.atomic_write(target, "v1")
    pa.atomic_write(target, "v2", backup=False)
    assert target.read_text() == "v2"
    assert not (tmp_path / "x.md.bak").exists()


def test_atomic_write_creates_parent_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deep" / "x.md"
    pa.atomic_write(target, "v1")
    assert target.read_text() == "v1"


# ---------------------------------------------------------------------------
# stage_cache_file.
# ---------------------------------------------------------------------------


def test_stage_cache_file_returns_hash(tmp_path: Path) -> None:
    src = tmp_path / "src.md"
    src.write_text("ignored — just for mtime")
    dst = tmp_path / "dst.md"
    h = pa.stage_cache_file(src, dst, "rendered content")
    assert len(h) == 8
    assert dst.read_text() == "rendered content"


def test_stage_cache_file_deterministic_hash(tmp_path: Path) -> None:
    src = tmp_path / "src.md"
    src.write_text("x")
    dst1 = tmp_path / "d1.md"
    dst2 = tmp_path / "d2.md"
    h1 = pa.stage_cache_file(src, dst1, "same")
    h2 = pa.stage_cache_file(src, dst2, "same")
    assert h1 == h2


# ---------------------------------------------------------------------------
# retry_mcp_handshake.
# ---------------------------------------------------------------------------


def test_retry_success_on_first_attempt() -> None:
    calls: list[int] = []

    def fn() -> str:
        calls.append(1)
        return "ok"

    assert pa.retry_mcp_handshake(fn, sleep=lambda s: None) == "ok"
    assert len(calls) == 1


def test_retry_exponential_backoff_2_4_8() -> None:
    """I-N5: first retry waits 2s, second waits 4s, third waits 8s."""
    sleeps: list[float] = []
    attempts = {"n": 0}

    def fn() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError(f"attempt {attempts['n']} failed")
        return "ok"

    result = pa.retry_mcp_handshake(
        fn,
        attempts=3,
        backoff_s=(2.0, 4.0, 8.0),
        budget_s=100.0,  # far above sum so no early-exit
        sleep=sleeps.append,
    )
    assert result == "ok"
    # Two retries happened → two sleeps at 2.0 and 4.0.
    assert sleeps == [2.0, 4.0]


def test_retry_all_attempts_fail_raises() -> None:
    attempts = {"n": 0}

    def fn() -> str:
        attempts["n"] += 1
        raise RuntimeError(f"fail {attempts['n']}")

    with pytest.raises(RuntimeError, match="fail 3"):
        pa.retry_mcp_handshake(
            fn,
            attempts=3,
            backoff_s=(2.0, 4.0, 8.0),
            budget_s=100.0,
            sleep=lambda s: None,
        )
    assert attempts["n"] == 3


def test_retry_budget_exhausted_early_exit() -> None:
    """Budget exhausted before the next sleep would complete → bail
    early, don't sleep more."""
    sleeps: list[float] = []
    attempts = {"n": 0}

    # Fake clock that advances by the sleep amount + a bit.
    fake_time = {"t": 0.0}

    def fake_clock() -> float:
        return fake_time["t"]

    def fake_sleep(s: float) -> None:
        sleeps.append(s)
        fake_time["t"] += s

    def fn() -> str:
        attempts["n"] += 1
        raise RuntimeError(f"attempt {attempts['n']}")

    # Budget 5s; first backoff is 2s (allowed, elapsed=0+2=2 <= 5),
    # next backoff is 4s but elapsed 2+4=6 > 5 → bail before attempt 3.
    with pytest.raises(RuntimeError):
        pa.retry_mcp_handshake(
            fn,
            attempts=3,
            backoff_s=(2.0, 4.0, 8.0),
            budget_s=5.0,
            sleep=fake_sleep,
            clock=fake_clock,
        )
    # Expect exactly 1 sleep (before attempt 2); attempt 3 skipped.
    assert sleeps == [2.0]
    assert attempts["n"] == 2


def test_retry_reads_budget_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(pa.MCP_RETRY_BUDGET_ENV, "3")
    sleeps: list[float] = []
    fake_time = {"t": 0.0}

    def fake_clock() -> float:
        return fake_time["t"]

    def fake_sleep(s: float) -> None:
        sleeps.append(s)
        fake_time["t"] += s

    def fn() -> str:
        raise RuntimeError("always")

    with pytest.raises(RuntimeError):
        pa.retry_mcp_handshake(
            fn,
            attempts=3,
            backoff_s=(2.0, 4.0, 8.0),
            sleep=fake_sleep,
            clock=fake_clock,
        )
    # With 3s budget: first backoff 2s allowed (elapsed becomes 2 <= 3),
    # next backoff 4s pushes elapsed to 6 > 3 → stop before attempt 3.
    assert sleeps == [2.0]
