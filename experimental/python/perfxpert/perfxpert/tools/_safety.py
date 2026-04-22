"""_safety — §5.8 threat-model helpers shared by all EXECUTION tools.

Every EXECUTION tool in perfxpert.tools.* MUST funnel untrusted input
through these helpers. Test coverage is in tests/test_tools/test_safety.py
and tests/test_red_team/ (adversarial).
"""

from __future__ import annotations

import os
import re
import stat
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
# Includes brace/history expansion markers from the interactive threat model.
# Newlines and NULs are always rejected.
_SHELL_METACHARS = re.compile(r"[;&|`$()<>{}!\n\r\0]|\\\\|\\\"|\\'")


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

_ENV_DENYLIST_EXACT = {
    "BASH_ENV",
    "ENV",
    "GCONV_PATH",
    "IFS",
    "NODE_OPTIONS",
    "PERL5OPT",
    "PYTHONHOME",
    "PYTHONPATH",
    "RUBYOPT",
}

_ENV_DENYLIST_PREFIXES = (
    "LD_",
    "DYLD_",
)

_SAFE_PATH_DIRS = (
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/local/sbin",
    "/usr/sbin",
    "/sbin",
    "/opt/rocm/bin",
)

_SAFE_PATH_PREFIXES = tuple(Path(p).resolve() for p in _SAFE_PATH_DIRS)


def _is_denied_env_key(key: str) -> bool:
    return key in _ENV_DENYLIST_EXACT or any(
        key.startswith(prefix) for prefix in _ENV_DENYLIST_PREFIXES
    )


def _collect_allowed_env(extra: dict | None = None) -> dict:
    safe = {}
    for source in (os.environ, extra or {}):
        for k, v in source.items():
            if k == "PATH" or _is_denied_env_key(k):
                continue
            if k in _ENV_WHITELIST or any(
                k.startswith(p) for p in _ENV_PREFIX_WHITELIST
            ):
                safe[k] = v
    return safe


def _validated_path_dir(entry: str) -> Path | None:
    """Return a canonical safe PATH directory or None if the entry is unusable."""
    if not entry:
        return None
    if os.pathsep in entry or not os.path.isabs(entry):
        return None
    try:
        path = Path(entry).resolve(strict=True)
    except OSError:
        return None
    if not path.is_dir():
        return None
    return path


def _is_safe_path_dir(path: Path) -> bool:
    for prefix in _SAFE_PATH_PREFIXES:
        if path == prefix or prefix in path.parents:
            return True

    try:
        home = Path.home().resolve(strict=True)
    except OSError:
        return False
    return path == home or home in path.parents


def _safe_path_entries(env: dict) -> List[str]:
    entries: List[str] = []

    inherited_path = env.get("PATH", "") or os.environ.get("PATH", "")
    for raw in inherited_path.split(os.pathsep):
        path = _validated_path_dir(raw)
        if path and _is_safe_path_dir(path):
            entries.append(str(path))

    for root_var in ("ROCM_PATH", "HIP_PATH"):
        root = env.get(root_var)
        if not root:
            continue
        root_path = _validated_path_dir(root)
        if root_path is None:
            continue
        bin_dir = _validated_path_dir(str(root_path / "bin"))
        if bin_dir and _is_safe_path_dir(bin_dir):
            entries.append(str(bin_dir))

    entries.extend(str(prefix) for prefix in _SAFE_PATH_PREFIXES)
    return entries


def _build_safe_path(env: dict) -> str:
    unique_entries = []
    seen = set()
    for entry in _safe_path_entries(env):
        if entry not in seen:
            unique_entries.append(entry)
            seen.add(entry)
    return os.pathsep.join(unique_entries)


def build_safe_env(extra: dict | None = None) -> dict:
    """Construct a minimal subprocess env containing only whitelisted keys.

    - API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, …) are NEVER forwarded.
    - Loader/interpreter injection vars (LD_*, DYLD_*, PYTHONPATH, …) are
      explicitly denied even if supplied via `extra`.
    - PATH is deterministic and does not inherit the caller's PATH.
    """
    safe = _collect_allowed_env(extra=extra)
    safe["PATH"] = _build_safe_path(safe)
    return safe
