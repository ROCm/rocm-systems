"""Consent cache for backend installation actions (Task 3, I10).

Tracks whether the user has consented to install perfxpert
configuration (MCP entry + prompt + gate hook) for a given
`(backend, cwd, file-set)` triple. Re-prompts whenever any element
of the triple changes:

* `backend` — switching from `claude` to `gemini` requires a fresh
  consent (different files touched).
* `cwd-hash` — switching project directories requires a fresh
  consent (different repo).
* `file-set-hash` — if the set of files the adapter plans to touch
  changes (e.g. the user just started tracking `CLAUDE.md`), the
  impact of saying "yes" changed, so re-prompt.

Storage: `$XDG_CONFIG_HOME/perfxpert/consent.yaml` (default
`~/.config/perfxpert/consent.yaml`). YAML so it is human-auditable.

Env overrides:
  * `PERFXPERT_ASSUME_CONSENT=1` — auto-grant, no prompt (CI).
  * If not a TTY and no env override → refuse with guidance.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


__all__ = [
    "CONSENT_ASSUME_ENV",
    "ConsentRecord",
    "consent_path",
    "cwd_hash",
    "file_set_hash",
    "has_consent",
    "grant_consent",
    "revoke_consent",
    "prompt_consent_interactive",
]


CONSENT_ASSUME_ENV = "PERFXPERT_ASSUME_CONSENT"


# ---------------------------------------------------------------------------
# File location (XDG-compliant).
# ---------------------------------------------------------------------------


def _xdg_config_home() -> Path:
    override = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".config"


def consent_path() -> Path:
    """Absolute path to the consent cache YAML file."""
    return _xdg_config_home() / "perfxpert" / "consent.yaml"


# ---------------------------------------------------------------------------
# Hashing helpers (short, human-readable SHA-8 prefixes).
# ---------------------------------------------------------------------------


def _sha8(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def cwd_hash(cwd: Path) -> str:
    """Stable 8-char hash of the absolute resolved cwd."""
    return _sha8(str(Path(cwd).expanduser().resolve()))


def file_set_hash(files: Iterable[tuple[Path, bool, bool]]) -> str:
    """Stable 8-char hash of a `(path, exists, tracked)` tuple set.

    The adapter builds this set in `plan()` so consent invalidates
    when the target file set changes (e.g. the user newly tracks
    `CLAUDE.md` in git — consent-at-untracked is no longer safe).
    Input tuples are sorted so the hash is order-independent.
    """
    normalized = sorted(
        (str(Path(p).expanduser().resolve()), bool(exists), bool(tracked))
        for p, exists, tracked in files
    )
    return _sha8(repr(normalized))


# ---------------------------------------------------------------------------
# Persistent record + I/O.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsentRecord:
    backend: str
    cwd_hash: str
    file_set_hash: str


def _load_cache() -> dict:
    p = consent_path()
    if not p.is_file():
        return {}
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _save_cache(data: dict) -> None:
    p = consent_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=True))
    os.replace(tmp, p)


def _key(record: ConsentRecord) -> str:
    return f"{record.backend}:{record.cwd_hash}:{record.file_set_hash}"


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def has_consent(backend: str, cwd: Path, fset_hash: str) -> bool:
    """True iff the user has previously granted consent for this triple."""
    data = _load_cache()
    record = ConsentRecord(
        backend=backend, cwd_hash=cwd_hash(cwd), file_set_hash=fset_hash
    )
    return bool(data.get(_key(record)))


def grant_consent(backend: str, cwd: Path, fset_hash: str) -> None:
    data = _load_cache()
    record = ConsentRecord(
        backend=backend, cwd_hash=cwd_hash(cwd), file_set_hash=fset_hash
    )
    data[_key(record)] = {
        "backend": record.backend,
        "cwd_hash": record.cwd_hash,
        "file_set_hash": record.file_set_hash,
    }
    _save_cache(data)


def revoke_consent(backend: str, cwd: Path) -> None:
    """Remove every consent record matching (backend, cwd), regardless
    of file-set-hash.

    Called by `perfxpert-code uninstall <backend>` (Task 8) so the next
    install prompts again.
    """
    data = _load_cache()
    target_cwd = cwd_hash(cwd)
    to_drop = [
        k for k, v in data.items()
        if isinstance(v, dict)
        and v.get("backend") == backend
        and v.get("cwd_hash") == target_cwd
    ]
    for k in to_drop:
        data.pop(k, None)
    _save_cache(data)


def prompt_consent_interactive(
    backend: str,
    cwd: Path,
    plan_lines: list[str],
    *,
    stream=None,
) -> bool:
    """Display `plan_lines` and ask `[y/N]`. Return True on explicit "yes".

    Behavior:

    * If `PERFXPERT_ASSUME_CONSENT=1`, auto-grants (no prompt).
    * If stdin is not a TTY AND env is unset, refuse with guidance
      (non-interactive install paths must set the env explicitly).
    * Otherwise: read one line from stdin, accept `y` / `yes`
      (case-insensitive); anything else declines.
    """
    out = stream if stream is not None else sys.stderr

    if os.environ.get(CONSENT_ASSUME_ENV, "").strip().lower() in {"1", "true", "yes"}:
        out.write(
            f"[consent] {CONSENT_ASSUME_ENV}=1 → auto-grant for {backend} in {cwd}\n"
        )
        return True

    if not sys.stdin.isatty():
        out.write(
            f"[consent] non-interactive stdin detected and {CONSENT_ASSUME_ENV} is not set.\n"
            f"  To install perfxpert into {backend} non-interactively, export\n"
            f"  {CONSENT_ASSUME_ENV}=1 and re-run.\n"
        )
        return False

    out.write(f"\nperfxpert-code: consent required for backend {backend!r}\n")
    out.write(f"  cwd: {Path(cwd).expanduser().resolve()}\n")
    out.write("  planned actions:\n")
    for line in plan_lines:
        out.write(f"    - {line}\n")
    out.write("\nProceed? [y/N] ")
    out.flush()

    try:
        answer = input().strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}
