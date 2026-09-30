"""Warmup + retry scaffolding for `perfxpert-mcp` on backend install (Task 4.7).

Problem (F2, brainstorm §13): the first `perfxpert-code run` after a
stale server state occasionally exits 124 — a one-line log, no
traceback. Root cause is the MCP handshake OR sqlite init race
inside `perfxpert-mcp`. Opencode hit it first, but every backend
will hit it the same way once they spawn the MCP server from stdio.

Fix: warm `perfxpert-mcp` once during `install()` so subsequent
backend-originated spawns hit cached state. Bounded by
`PERFXPERT_MCP_WARMUP_TIMEOUT_S` (default 10s) and gated by
`PERFXPERT_MCP_WARMUP` (default 1, set to 0 to disable).

Cleanup invariant (I-N6): the warmup subprocess MUST be torn down
via a clean close+wait handshake so sqlite checkpoints before exit.
No orphan `<db>-wal` / `<db>-shm` files may remain after warmup
returns.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass


__all__ = [
    "warmup_perfxpert_mcp",
    "WarmupReport",
    "WARMUP_ENABLED_ENV",
    "WARMUP_TIMEOUT_ENV",
]


WARMUP_ENABLED_ENV = "PERFXPERT_MCP_WARMUP"
WARMUP_TIMEOUT_ENV = "PERFXPERT_MCP_WARMUP_TIMEOUT_S"


_LOG = logging.getLogger("perfxpert.backend.warmup")


@dataclass(frozen=True)
class WarmupReport:
    success: bool
    duration_s: float
    error: str | None = None


def warmup_perfxpert_mcp(
    *,
    timeout: float | None = None,
    binary: str | None = None,
    subprocess_module=subprocess,
) -> WarmupReport:
    """Spawn `perfxpert-mcp`, exchange one MCP initialize handshake,
    then close cleanly.

    Behavior:

    * Honors `PERFXPERT_MCP_WARMUP=0` — returns a skipped-success
      report without spawning anything.
    * `timeout` defaults to `PERFXPERT_MCP_WARMUP_TIMEOUT_S` env
      (10s if unset).
    * Sends an `initialize` JSON-RPC request, waits for a response,
      then sends `shutdown`/EOF so sqlite can checkpoint.
    * Cleanup: `proc.stdin.close()` + `proc.wait()`; on timeout
      `proc.terminate()` (SIGTERM — permits cleanup) then
      `proc.wait(grace=2s)`. No `kill -9` unless the backend
      genuinely ignores SIGTERM.
    * Idempotent — safe to call multiple times; each call is a
      fresh short spawn.

    Returns a `WarmupReport`. Never raises on the normal failure
    paths — the adapter treats warmup failure as a non-fatal
    optimization miss.
    """
    start = time.monotonic()
    enabled = os.environ.get(WARMUP_ENABLED_ENV, "1").strip().lower()
    if enabled in {"0", "false", "no"}:
        return WarmupReport(success=True, duration_s=0.0, error=None)

    if timeout is None:
        try:
            timeout = float(os.environ.get(WARMUP_TIMEOUT_ENV, "10"))
        except ValueError:
            timeout = 10.0

    binary = binary or shutil.which("perfxpert-mcp")
    if binary is None:
        return WarmupReport(
            success=False,
            duration_s=time.monotonic() - start,
            error="perfxpert-mcp not found on PATH",
        )

    # Spawn.
    try:
        proc = subprocess_module.Popen(
            [binary],
            stdin=subprocess_module.PIPE,
            stdout=subprocess_module.PIPE,
            stderr=subprocess_module.PIPE,
        )
    except OSError as exc:
        return WarmupReport(
            success=False,
            duration_s=time.monotonic() - start,
            error=f"failed to spawn perfxpert-mcp: {exc}",
        )

    # Minimal MCP initialize handshake (JSON-RPC 2.0).
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "perfxpert-warmup", "version": "1.0"},
            "capabilities": {},
        },
    }

    error: str | None = None
    stdin = proc.stdin
    try:
        if stdin is None:
            error = "warmup subprocess missing stdin pipe"
        else:
            stdin.write((json.dumps(init_req) + "\n").encode("utf-8"))
            stdin.flush()
    except (BrokenPipeError, OSError) as exc:
        error = f"write handshake failed: {exc}"

    # Do NOT read stdout here. A blocking readline() defeats the whole
    # timeout budget when the child never emits a response. The bounded
    # wait() below is the only timing gate for the warmup.

    # Clean shutdown: close stdin (EOF signals server to exit),
    # then wait with timeout.
    try:
        if stdin is not None:
            stdin.close()
    except OSError:
        pass

    try:
        proc.wait(timeout=timeout)
    except subprocess_module.TimeoutExpired:
        # Send SIGTERM (NOT kill -9) so sqlite can checkpoint.
        try:
            proc.terminate()
        except OSError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess_module.TimeoutExpired:
            # Very last resort — kill hard to avoid hanging CI.
            try:
                proc.kill()
                proc.wait(timeout=1)
            except (OSError, subprocess_module.TimeoutExpired):
                pass
            error = error or "warmup subprocess did not exit within timeout"
        else:
            error = error or f"warmup exceeded {timeout:.1f}s timeout"

    if error is None and proc.returncode not in (None, 0):
        error = f"warmup subprocess exited {proc.returncode}"

    # Consume any remaining stderr so the buffer doesn't orphan fds.
    if proc.stderr is not None:
        try:
            proc.stderr.read()
            proc.stderr.close()
        except OSError:
            pass
    if proc.stdout is not None:
        try:
            proc.stdout.close()
        except OSError:
            pass

    success = proc.returncode == 0 and error is None
    return WarmupReport(
        success=success,
        duration_s=time.monotonic() - start,
        error=error,
    )
