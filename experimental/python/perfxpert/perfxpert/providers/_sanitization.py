"""Data sanitization helpers for LLM prompts.

All 5 provider adapters import from here.

These helpers are pure, deterministic, and unit-testable.
"""

from __future__ import annotations

import dataclasses
import os
import re
from typing import Any, Dict, List, Optional

_PATH_PATTERN = re.compile(
    r'(/home/[^\s,"\';>]+|/opt/[^\s,"\';>]+|/root/[^\s,"\';>]+|'
    r'/tmp/[^\s,"\';>]+|/var/[^\s,"\';>]+|[A-Za-z]:\\[^\s,"\';>]+)'
)

# Regex to match relative path traversal (e.g. ../../secret/file)
_RELATIVE_PATH_PATTERN = re.compile(r'(?:\.\./)+\S+')

_PATH_FIELD_NAMES = {
    "att_dir",
    "baseline_db",
    "database_path",
    "db_path",
    "new_db",
    "project_root",
    "source_dir",
}


def redact_paths(value: str) -> str:
    """Replace file system paths in a string with [REDACTED].

    Handles both absolute Unix/Windows paths and relative path traversals.
    """
    result = _PATH_PATTERN.sub("[REDACTED]", value)
    result = _RELATIVE_PATH_PATTERN.sub("[REDACTED_PATH]", result)
    return result


def _is_path_field(key: Optional[str]) -> bool:
    if key is None:
        return False
    normalized = key.lower()
    return (
        normalized in _PATH_FIELD_NAMES
        or normalized.endswith("_path")
        or normalized.endswith("_dir")
    )


def _path_redaction_value(
    key: Optional[str],
    value: str,
    token_to_raw_path: Optional[Dict[str, str]],
) -> str:
    """Return a provider-safe redaction, optionally recording token -> raw path."""
    if token_to_raw_path is None:
        return "[REDACTED_PATH]"

    token_name = re.sub(r"[^a-z0-9_]+", "_", (key or "path").lower()).strip("_")
    token_name = token_name or "path"
    token = f"[REDACTED_PATH:{token_name}]"
    if token_to_raw_path.get(token) == value:
        return token

    suffix = 2
    while token in token_to_raw_path:
        token = f"[REDACTED_PATH:{token_name}_{suffix}]"
        if token_to_raw_path.get(token) == value:
            return token
        suffix += 1

    token_to_raw_path[token] = value
    return token


def sanitize_value(
    value: Any,
    *,
    key: Optional[str] = None,
    token_to_raw_path: Optional[Dict[str, str]] = None,
) -> Any:
    """Recursively redact path-like strings in provider-bound payloads."""
    if hasattr(value, "model_dump"):
        return sanitize_value(value.model_dump(), key=key, token_to_raw_path=token_to_raw_path)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sanitize_value(dataclasses.asdict(value), key=key, token_to_raw_path=token_to_raw_path)
    if isinstance(value, os.PathLike):
        return _path_redaction_value(key, os.fspath(value), token_to_raw_path)
    if isinstance(value, str):
        if _is_path_field(key) and value:
            return _path_redaction_value(key, value, token_to_raw_path)
        return redact_paths(value)
    if isinstance(value, (list, tuple)):
        return [sanitize_value(item, key=key, token_to_raw_path=token_to_raw_path) for item in value]
    if isinstance(value, (set, frozenset)):
        return [
            sanitize_value(item, key=key, token_to_raw_path=token_to_raw_path)
            for item in sorted(value, key=repr)
        ]
    if isinstance(value, dict):
        return {
            item_key: sanitize_value(item, key=str(item_key), token_to_raw_path=token_to_raw_path)
            for item_key, item in value.items()
        }
    return value


def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a sanitized copy of an OpenAI-style messages list."""
    return sanitize_value(messages)


# Alias for backward compatibility with existing code
_redact_paths = redact_paths
