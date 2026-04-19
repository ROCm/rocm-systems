"""Shared helpers for backend prompt rendering + file staging (Task 4a).

Cross-backend utilities used by every `BackendAdapter.install()`:

* `render_prompt()` — loads a source AGENTS.md-like file, substitutes
  `perfxpert_<tool>` tokens into the backend's wire format, strips
  non-target `<!--backend:NAME-->...<!--/backend:NAME-->` blocks, and
  optionally injects a rejection-language stanza (Task 4a / F1).
* `emit_marker_block()` — sentinel sentinels for managed block
  detection (I6: `<!-- BEGIN perfxpert-managed v1 cache=<sha8> -->`).
* `is_git_tracked()` — `git ls-files --error-unmatch` check (I3).
* `atomic_write()` — tmp-write + fsync + rename with `.bak` rotation.
* `stage_cache_file()` — convenience wrapper returning the cache hash.
* `retry_mcp_handshake()` — exponential-backoff retry helper used by
  every adapter's `verify_mcp_live()` (F2, I-N5).

All file writes go through `atomic_write` so a mid-write crash never
leaves a half-written config.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, TypeVar


__all__ = [
    "BEGIN_MARKER_FMT",
    "END_MARKER_FMT",
    "render_prompt",
    "emit_marker_block",
    "is_git_tracked",
    "is_absolute_path_git_tracked",
    "atomic_write",
    "stage_cache_file",
    "retry_mcp_handshake",
    "REJECTION_LANGUAGE_STANZA",
    "MCP_RETRY_BUDGET_ENV",
]


# Marker format — deliberately distinctive so we can scan for drift
# in Task 8 uninstall without matching user-written "perfxpert" mentions.
BEGIN_MARKER_FMT = "<!-- BEGIN perfxpert-managed v1 cache={cache} -->"
END_MARKER_FMT = "<!-- END perfxpert-managed v1 -->"

# Env var — Task 4.7 retry budget (seconds). Default 6s accommodates
# the common "attempts 1 + 2 at 2s/4s backoff" race; attempt 3 runs
# only if elapsed time so far permits.
MCP_RETRY_BUDGET_ENV = "PERFXPERT_MCP_RETRY_BUDGET_S"


# Rejection-language stanza (F1 — see Task 4.6 gate hook). Kept as a
# constant so each backend's rendered prompt AND the gate hook's
# `permissionDecisionReason` / `retryWith` message reference the
# SAME text. (Avoids drift between prompt-layer and server-side
# enforcement messaging.)
REJECTION_LANGUAGE_STANZA = """\
# Tool-priority gate (NON-NEGOTIABLE)

Your response WILL BE REJECTED if you call any tool other than
`{classify_tool}` before `{classify_tool}` has been observed to
return in this session. This is mechanically enforced by a
pre-tool-call hook — the rejection is not advisory.

After `{classify_tool}` returns, any tool is permitted for the
rest of the session.
"""


# ---------------------------------------------------------------------------
# Prompt rendering.
# ---------------------------------------------------------------------------


_BACKEND_BLOCK_RE = re.compile(
    r"<!--backend:(?P<name>[a-zA-Z0-9_-]+)-->"
    r"(?P<body>.*?)"
    r"<!--/backend:(?P=name)-->",
    re.DOTALL,
)


def _strip_non_target_backend_blocks(src: str, target_backend: str) -> str:
    """Remove every `<!--backend:X-->...<!--/backend:X-->` block where X != target."""

    def _replace(m: re.Match[str]) -> str:
        return m.group("body") if m.group("name") == target_backend else ""

    return _BACKEND_BLOCK_RE.sub(_replace, src)


def _substitute_tool_names(
    src: str, tool_name_template: str, known_tools: tuple[str, ...]
) -> str:
    """Replace every `perfxpert_<tool>` token with the rendered backend form.

    `tool_name_template` is expected to contain exactly one `{tool}`
    placeholder. For each name in `known_tools`, we rewrite
    `perfxpert_<name>` → `<template with tool=<name>>`. Only
    word-boundary matches are rewritten so user-written comments
    mentioning `perfxpert_intent_classify_legacy` etc. aren't
    mangled unless they exactly match.
    """
    out = src
    for tool in known_tools:
        src_token = f"perfxpert_{tool}"
        rendered = tool_name_template.format(tool=tool)
        out = re.sub(rf"\b{re.escape(src_token)}\b", rendered, out)
    return out


def render_prompt(
    source: Path | str,
    *,
    backend: str,
    tool_name_template: str,
    known_tools: tuple[str, ...],
    reject_language: bool = True,
    classify_tool: str = "intent_classify",
) -> str:
    """Render a source prompt file into the backend-specific form.

    Steps:

    1. Load the file contents (if `source` is a Path).
    2. Strip `<!--backend:X-->...<!--/backend:X-->` blocks for all X
       except the target `backend`.
    3. Substitute every `perfxpert_<name>` in `known_tools` with the
       backend's wire format.
    4. If `reject_language=True`, prepend the rejection-language
       stanza (F1 — see Task 4.6 gate hook).

    The `classify_tool` kwarg lets the caller pass a pre-rendered
    tool name (e.g. `mcp__perfxpert__intent_classify`) so the stanza
    refers to the exact name the backend exposes.
    """
    if isinstance(source, Path):
        raw = source.read_text(encoding="utf-8")
    else:
        raw = source

    stripped = _strip_non_target_backend_blocks(raw, backend)
    substituted = _substitute_tool_names(stripped, tool_name_template, known_tools)

    if reject_language:
        rendered_classify = tool_name_template.format(tool=classify_tool)
        stanza = REJECTION_LANGUAGE_STANZA.format(classify_tool=rendered_classify)
        return stanza + "\n" + substituted
    return substituted


# ---------------------------------------------------------------------------
# Marker block (I6).
# ---------------------------------------------------------------------------


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def emit_marker_block(content: str) -> tuple[str, str]:
    """Return `(begin_marker, end_marker)` sentinels for `content`.

    The begin marker includes the SHA-8 of the content so Task 8
    uninstall can detect drift (user edits inside our managed block
    change the hash).
    """
    cache = _sha8(content)
    return (BEGIN_MARKER_FMT.format(cache=cache), END_MARKER_FMT)


# ---------------------------------------------------------------------------
# Git-tracked detection (I3).
# ---------------------------------------------------------------------------


def is_git_tracked(path: Path, cwd: Path) -> bool:
    """Return True iff `path` is tracked in the git repo rooted at `cwd`.

    Uses `git ls-files --error-unmatch` (exit 0 iff tracked). Falls
    back to False on any git error (not-a-repo, git missing, etc.).
    The critical case is "tracked == True": we must refuse to write
    the file in that branch without explicit user opt-in.
    """
    if not shutil.which("git"):
        return False
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=str(cwd),
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def is_absolute_path_git_tracked(path: Path) -> bool:
    """Return True iff the absolute `path` is tracked in any git repo.

    Unlike :func:`is_git_tracked` (which requires a caller-known repo
    root as `cwd`), this helper walks upward from `path.parent`
    looking for a `.git` directory, then asks git whether `path` is
    tracked relative to that repo root. Used for absolute writes to
    user-scope files (e.g. ``~/.codex/config.toml``) that may live
    inside a dotfiles-style repo the caller doesn't know about.

    Returns False on any of: git missing, no enclosing repo, git
    errors. The critical case is "tracked == True" — we must refuse
    to overwrite such a file.
    """
    if not shutil.which("git"):
        return False
    path = Path(path).expanduser()
    # Absolute-path operands are required for the upward walk to be
    # well-defined. A non-existent path is fine; we still walk its
    # parents looking for the enclosing repo root.
    if not path.is_absolute():
        try:
            path = path.resolve()
        except OSError:
            return False

    # Walk upward looking for `.git` (file or dir — supports worktrees).
    probe = path.parent if not path.is_dir() else path
    repo_root: Path | None = None
    while True:
        if (probe / ".git").exists():
            repo_root = probe
            break
        if probe.parent == probe:  # reached filesystem root
            break
        probe = probe.parent
    if repo_root is None:
        return False

    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=str(repo_root),
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Atomic write + `.bak` rotation.
# ---------------------------------------------------------------------------


def atomic_write(path: Path, content: str, *, backup: bool = True) -> None:
    """Write `content` to `path` atomically.

    Algorithm:

    1. If `path` exists and `backup` is True, copy it to `path.bak`.
    2. Write `content` to `path.tmp`.
    3. `fsync` on the tmp file.
    4. `os.replace(tmp, path)` — atomic on POSIX.

    Does NOT guarantee `.bak` rotation beyond one generation; that
    is Task 8's job. The focus here is "never leave a partial file".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if backup and path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)

    tmp = path.with_suffix(path.suffix + ".tmp")
    # Write + fsync to survive a mid-write crash.
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(content)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            # Some filesystems (tmpfs) refuse fsync; treat as best-effort.
            pass
    os.replace(tmp, path)


def stage_cache_file(source: Path, destination: Path, rendered: str) -> str:
    """Write `rendered` to `destination` atomically and return the SHA-8 hash.

    `source` is metadata-only (its mtime is copied to the destination)
    so a re-render with identical content doesn't invalidate downstream
    cache checks.
    """
    atomic_write(destination, rendered)
    # Best-effort: propagate the source mtime so downstream tools can cheaply
    # detect "did my source change?". Ignored on filesystems that don't
    # support it.
    try:
        st = Path(source).stat()
        os.utime(destination, (st.st_atime, st.st_mtime))
    except (FileNotFoundError, OSError):
        pass
    return _sha8(rendered)


# ---------------------------------------------------------------------------
# MCP handshake retry helper (F2, I-N5).
# ---------------------------------------------------------------------------


T = TypeVar("T")


def retry_mcp_handshake(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    backoff_s: tuple[float, ...] = (2.0, 4.0, 8.0),
    budget_s: float | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> T:
    """Call `fn` up to `attempts` times with exponential backoff.

    Behavior:

    * Each attempt that raises is retried until attempts exhausted
      OR the total elapsed time exceeds `budget_s`.
    * Backoff before attempt N = `backoff_s[N-1]` (so the first
      retry sleeps `backoff_s[0]`).
    * `budget_s` defaults to `PERFXPERT_MCP_RETRY_BUDGET_S` from env
      (6.0s if unset).

    The `sleep` + `clock` kwargs exist purely so tests can assert
    the exact backoff sequence without actually waiting 14 seconds.
    """
    if budget_s is None:
        budget_s = float(os.environ.get(MCP_RETRY_BUDGET_ENV, "6"))
    # Resolve sleep + clock lazily so monkeypatching `time.sleep` /
    # `time.monotonic` in tests takes effect per-call.
    if sleep is None:
        sleep = time.sleep
    if clock is None:
        clock = time.monotonic

    start = clock()
    last_exc: Exception | None = None

    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # intentionally broad — caller decides what's retryable.
            last_exc = exc

        # Decide whether to sleep + retry OR bail now.
        if i + 1 >= attempts:
            break
        wait = backoff_s[i] if i < len(backoff_s) else backoff_s[-1]
        elapsed = clock() - start
        if elapsed + wait > budget_s:
            # Budget would be exhausted; exit early.
            break
        sleep(wait)

    assert last_exc is not None  # unreachable unless attempts == 0
    raise last_exc
