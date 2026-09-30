"""_safety — §5.8 threat-model helpers shared by all EXECUTION tools.

Every EXECUTION tool in perfxpert.tools.* MUST funnel untrusted input
through these helpers. Test coverage is in tests/test_tools/test_safety.py
and tests/test_red_team/ (adversarial).

See docs/superpowers/specs/2026-04-17-multi-agent-perfxpert-design.md §5.8
for the complete attack-vector table.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Set, Tuple


class SafetyError(Exception):
    """Base class for §5.8 rejections."""


class PathConfinementError(SafetyError):
    """Resolved path is outside the project root."""


class ShellMetacharError(SafetyError):
    """String contains shell metacharacters."""


class DangerousCommandError(SafetyError):
    """String matches a denylisted destructive pattern."""


# -- path confinement -------------------------------------------------------

def confine_to_project_root(project_root: Path, user_path: str) -> Path:
    """Resolve `user_path` under `project_root`; reject if it escapes.

    Rejects:
    - Absolute paths outside project_root
    - Relative paths containing `..` that resolve outside project_root
    - Symlinks whose target lies outside project_root

    Returns the fully-resolved canonical path (with symlinks followed).
    """
    root = Path(project_root).resolve(strict=True)
    candidate = (root / user_path) if not Path(user_path).is_absolute() else Path(user_path)
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise PathConfinementError(f"cannot resolve {user_path!r}: {e}") from e

    # is_relative_to is Python 3.9+; use explicit check for portability
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PathConfinementError(
            f"path {user_path!r} resolves to {resolved} which is outside project root {root}"
        )
    return resolved


# -- shell-metachar denial --------------------------------------------------

# Disallowed anywhere in a string that may reach a shell or argument list.
# Newlines and NULs are always rejected.
_SHELL_METACHARS = re.compile(r"[;&|`$()<>\n\r\0]|\\\\|\\\"|\\'")


def reject_shell_metachars(s: str) -> None:
    """Raise ShellMetacharError if `s` contains shell metachars.

    Applied to every string a tool receives from LLM output or trace metadata
    BEFORE it flows into subprocess or patch operations.
    """
    if _SHELL_METACHARS.search(s):
        raise ShellMetacharError(
            f"string contains shell metacharacter: {s!r}"
        )


# -- destructive-command denylist -------------------------------------------

_DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\bcurl\b.*\|\s*sh\b", re.IGNORECASE),
    re.compile(r"\bwget\b.*\|\s*sh\b", re.IGNORECASE),
    re.compile(r"\bwget\b\s+http", re.IGNORECASE),
    re.compile(r"\bmv\s+/\s+"),
    re.compile(r":\(\)\s*\{.*\};:"),  # classic fork-bomb shape
    re.compile(r">\s*/dev/sd[a-z]"),  # writing to raw disk
    re.compile(r"\bdd\s+.*\bof=/dev/"),
]


def strip_dangerous_patterns(s: str) -> str:
    """Raise DangerousCommandError if `s` matches any denylist pattern.

    Name 'strip' is historical; we REJECT rather than sanitize in-place
    (per spec §5.8: "any match rejected"). Kept for compatibility with spec
    wording.
    """
    for pat in _DANGEROUS_PATTERNS:
        if pat.search(s):
            raise DangerousCommandError(
                f"string matches denylisted destructive pattern {pat.pattern!r}: {s!r}"
            )
    return s


# -- flag allowlist helper --------------------------------------------------

def filter_by_allowlist(
    flags: Iterable[str],
    allowed: Set[str],
) -> Tuple[List[str], List[str]]:
    """Partition `flags` into (accepted, rejected) against `allowed`.

    Used by compile_runner and profile_runner to enforce §5.8 allowlists
    derived from knowledge/compiler_flags.yaml and the rocprofv3 flag set.
    """
    accepted, rejected = [], []
    for f in flags:
        # Some flags take values as separate tokens; accept only the flag key
        # by comparing the leading token. Value validation is the caller's job.
        key = f.split("=", 1)[0]
        if key in allowed:
            accepted.append(f)
        else:
            rejected.append(f)
    return accepted, rejected


# -- subprocess env whitelist -----------------------------------------------

_ENV_WHITELIST = (
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    # ROCm-specific
    "ROCM_PATH",
    "HIP_PATH",
    "HSA_OVERRIDE_GFX_VERSION",
    # rocprofv3 envs (prefix-match below)
)

_ENV_PREFIX_WHITELIST = (
    "ROCPROFV3_",
    "ROCPROFILER_",
)


def build_safe_env(extra: dict | None = None) -> dict:
    """Construct a minimal subprocess env containing only whitelisted keys.

    - API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, …) are NEVER forwarded.
    - Adds anything in `extra` last (caller responsibility for those values).
    """
    safe = {}
    for k, v in os.environ.items():
        if k in _ENV_WHITELIST or any(k.startswith(p) for p in _ENV_PREFIX_WHITELIST):
            safe[k] = v
    if extra:
        safe.update(extra)
    return safe
