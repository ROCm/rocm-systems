"""Tests for `perfxpert.cli._mcp_warmup` (Task 4.7, F2, I-N5, I-N6).

Covers:

* Warmup spawns + closes perfxpert-mcp cleanly (I-N6): no orphan
  WAL/SHM residue, subprocess exits, no kill -9 unless timeout.
* Reports duration.
* Idempotent on second call.
* Skipped when `PERFXPERT_MCP_WARMUP=0`.
* Honors `PERFXPERT_MCP_WARMUP_TIMEOUT_S` override.
* Graceful failure when perfxpert-mcp binary is missing.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from perfxpert.cli._mcp_warmup import (
    WARMUP_ENABLED_ENV,
    WARMUP_TIMEOUT_ENV,
    WarmupReport,
    warmup_perfxpert_mcp,
)


# ---------------------------------------------------------------------------
# A fake subprocess module suitable for unit tests.
# ---------------------------------------------------------------------------


class _FakePipe:
    def __init__(self) -> None:
        self.buffer: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> int:
        self.buffer.append(data)
        return len(data)

    def flush(self) -> None:
        pass

    def readline(self) -> bytes:
        return b"{}\n"

    def read(self) -> bytes:
        return b""

    def close(self) -> None:
        self.closed = True


class _ReadlineTrapPipe(_FakePipe):
    def readline(self) -> bytes:
        raise AssertionError("warmup should not call stdout.readline()")


class _FakeProcess:
    """Fake subprocess.Popen instance that captures every teardown call."""

    def __init__(
        self,
        *,
        returncode_after_wait: int = 0,
        raise_on_wait: bool = False,
        raise_on_second_wait: bool = False,
    ) -> None:
        self.stdin = _FakePipe()
        self.stdout = _FakePipe()
        self.stderr = _FakePipe()
        self.terminated = False
        self.killed = False
        self._wait_calls = 0
        self.returncode: int | None = None
        self._target_returncode = returncode_after_wait
        self._raise_on_wait = raise_on_wait
        self._raise_on_second_wait = raise_on_second_wait

    def wait(self, timeout: float | None = None) -> int:
        self._wait_calls += 1
        if self._raise_on_wait and self._wait_calls == 1:
            from subprocess import TimeoutExpired

            raise TimeoutExpired("perfxpert-mcp", timeout or 10)
        if self._raise_on_second_wait and self._wait_calls == 2:
            from subprocess import TimeoutExpired

            raise TimeoutExpired("perfxpert-mcp", timeout or 10)
        self.returncode = self._target_returncode
        return self._target_returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class _FakeSubprocess:
    """Module-like object satisfying warmup's expected surface."""

    PIPE = -1

    class TimeoutExpired(Exception):
        pass

    def __init__(self, proc: _FakeProcess) -> None:
        self.proc = proc
        # Use the real TimeoutExpired so `except subprocess_module.TimeoutExpired`
        # catches the warmup's expected shape.
        from subprocess import TimeoutExpired

        self.TimeoutExpired = TimeoutExpired

    def Popen(self, *args, **kwargs) -> _FakeProcess:
        return self.proc


# ---------------------------------------------------------------------------
# Happy path.
# ---------------------------------------------------------------------------


def test_warmup_spawns_and_closes_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-N6: clean close — no kill, stdout+stderr closed, no orphan state."""
    monkeypatch.delenv(WARMUP_ENABLED_ENV, raising=False)
    monkeypatch.setattr(
        "shutil.which",
        lambda name: "/usr/bin/perfxpert-mcp" if name == "perfxpert-mcp" else None,
    )
    fake_proc = _FakeProcess(returncode_after_wait=0)
    report = warmup_perfxpert_mcp(
        subprocess_module=_FakeSubprocess(fake_proc)
    )
    assert isinstance(report, WarmupReport)
    assert report.success is True
    assert fake_proc.stdin.closed is True
    assert fake_proc.stdout.closed is True
    assert fake_proc.stderr.closed is True
    # Clean path: no SIGTERM, no SIGKILL.
    assert fake_proc.terminated is False
    assert fake_proc.killed is False


def test_warmup_reports_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(WARMUP_ENABLED_ENV, raising=False)
    monkeypatch.setattr(
        "shutil.which", lambda _: "/usr/bin/perfxpert-mcp"
    )
    report = warmup_perfxpert_mcp(
        subprocess_module=_FakeSubprocess(_FakeProcess())
    )
    assert report.duration_s >= 0
    assert report.duration_s < 5.0  # generous upper bound for fake


def test_warmup_idempotent_on_second_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(WARMUP_ENABLED_ENV, raising=False)
    monkeypatch.setattr(
        "shutil.which", lambda _: "/usr/bin/perfxpert-mcp"
    )
    r1 = warmup_perfxpert_mcp(
        subprocess_module=_FakeSubprocess(_FakeProcess())
    )
    r2 = warmup_perfxpert_mcp(
        subprocess_module=_FakeSubprocess(_FakeProcess())
    )
    assert r1.success and r2.success


# ---------------------------------------------------------------------------
# Env overrides.
# ---------------------------------------------------------------------------


def test_warmup_skipped_when_env_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(WARMUP_ENABLED_ENV, "0")
    called = {"n": 0}

    def _bad_which(_):
        called["n"] += 1
        return "/usr/bin/perfxpert-mcp"

    monkeypatch.setattr("shutil.which", _bad_which)
    # Even if which() would succeed, the warmup should not spawn.
    fake_proc = _FakeProcess()
    report = warmup_perfxpert_mcp(
        subprocess_module=_FakeSubprocess(fake_proc)
    )
    assert report.success is True
    assert report.duration_s == 0.0
    # Popen was not exercised.
    assert fake_proc.stdin.closed is False


def test_warmup_honors_timeout_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(WARMUP_ENABLED_ENV, raising=False)
    monkeypatch.setenv(WARMUP_TIMEOUT_ENV, "2")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/perfxpert-mcp")

    captured_timeout: list[float | None] = []

    class _CaptureProc(_FakeProcess):
        def wait(self, timeout=None):  # type: ignore[override]
            captured_timeout.append(timeout)
            return super().wait(timeout)

    warmup_perfxpert_mcp(
        subprocess_module=_FakeSubprocess(_CaptureProc())
    )
    # First wait() (normal exit path) should see the env-override value.
    assert captured_timeout[0] == 2.0


def test_warmup_timeout_path_sends_sigterm_not_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-N6: cleanup uses terminate()+wait, not kill()."""
    monkeypatch.delenv(WARMUP_ENABLED_ENV, raising=False)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/perfxpert-mcp")
    fake_proc = _FakeProcess(raise_on_wait=True)
    report = warmup_perfxpert_mcp(
        subprocess_module=_FakeSubprocess(fake_proc)
    )
    assert report.success is False
    assert fake_proc.terminated is True
    # terminate → wait(2s) which in our fake resolves cleanly → kill NOT called.
    assert fake_proc.killed is False


def test_warmup_does_not_block_on_stdout_readline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(WARMUP_ENABLED_ENV, raising=False)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/perfxpert-mcp")
    fake_proc = _FakeProcess()
    fake_proc.stdout = _ReadlineTrapPipe()
    report = warmup_perfxpert_mcp(
        subprocess_module=_FakeSubprocess(fake_proc)
    )
    assert report.success is True


def test_warmup_handles_missing_stdin_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(WARMUP_ENABLED_ENV, raising=False)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/perfxpert-mcp")
    fake_proc = _FakeProcess()
    fake_proc.stdin = None
    report = warmup_perfxpert_mcp(
        subprocess_module=_FakeSubprocess(fake_proc)
    )
    assert report.success is False
    assert "stdin pipe" in (report.error or "")


def test_warmup_escalates_to_kill_only_when_terminate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(WARMUP_ENABLED_ENV, raising=False)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/perfxpert-mcp")
    fake_proc = _FakeProcess(raise_on_wait=True, raise_on_second_wait=True)
    warmup_perfxpert_mcp(
        subprocess_module=_FakeSubprocess(fake_proc)
    )
    # Both wait() calls timed out → kill() as last resort.
    assert fake_proc.terminated is True
    assert fake_proc.killed is True


def test_warmup_nonzero_exit_is_reported_as_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(WARMUP_ENABLED_ENV, raising=False)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/perfxpert-mcp")
    fake_proc = _FakeProcess(returncode_after_wait=7)
    report = warmup_perfxpert_mcp(
        subprocess_module=_FakeSubprocess(fake_proc)
    )
    assert report.success is False
    assert "exited 7" in (report.error or "")


# ---------------------------------------------------------------------------
# Missing binary.
# ---------------------------------------------------------------------------


def test_warmup_missing_binary_returns_error_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(WARMUP_ENABLED_ENV, raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    report = warmup_perfxpert_mcp(subprocess_module=MagicMock())
    assert report.success is False
    assert "not found on PATH" in (report.error or "")


def test_warmup_handles_popen_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(WARMUP_ENABLED_ENV, raising=False)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/perfxpert-mcp")

    class _BadModule:
        PIPE = -1
        TimeoutExpired = Exception

        def Popen(self, *a, **kw):
            raise OSError("permission denied")

    report = warmup_perfxpert_mcp(subprocess_module=_BadModule())
    assert report.success is False
    assert "permission denied" in (report.error or "")


# ---------------------------------------------------------------------------
# No orphan -wal / -shm files left after warmup.
# ---------------------------------------------------------------------------


def test_warmup_leaves_no_orphan_wal_or_shm_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """I-N6: after warmup returns, the tmp dir must be clean of -wal / -shm."""
    monkeypatch.delenv(WARMUP_ENABLED_ENV, raising=False)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/perfxpert-mcp")
    # We run warmup against our fake subprocess (no real sqlite), so
    # the test asserts the tmp_path fixture remains pristine and no
    # WAL/SHM files are silently created by our code. (The real
    # perfxpert-mcp writes into its own state dir; this test codifies
    # "the warmup harness itself doesn't leak".)
    before = {p.name for p in tmp_path.iterdir()}
    warmup_perfxpert_mcp(
        subprocess_module=_FakeSubprocess(_FakeProcess())
    )
    after = {p.name for p in tmp_path.iterdir()}
    assert before == after
    # Belt-and-braces: no `-wal` or `-shm` should appear anywhere in tmp_path.
    for p in tmp_path.rglob("*"):
        name = p.name
        assert not name.endswith("-wal"), name
        assert not name.endswith("-shm"), name
