"""Data sanitization helpers for LLM prompts.

All 5 provider adapters import from here.

These helpers are pure, deterministic, and unit-testable.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_PATH_PATTERN = re.compile(
    r'(/home/[^\s,"\';>]+|/opt/[^\s,"\';>]+|/root/[^\s,"\';>]+|'
    r'/tmp/[^\s,"\';>]+|/var/[^\s,"\';>]+|[A-Za-z]:\\[^\s,"\';>]+)'
)

# Regex to match relative path traversal (e.g. ../../secret/file)
_RELATIVE_PATH_PATTERN = re.compile(r'(?:\.\./)+\S+')


def redact_paths(value: str) -> str:
    """Replace file system paths in a string with [REDACTED].

    Handles both absolute Unix/Windows paths and relative path traversals.
    """
    result = _PATH_PATTERN.sub("[REDACTED]", value)
    result = _RELATIVE_PATH_PATTERN.sub("[REDACTED_PATH]", result)
    return result


def sanitize_value(value: Any) -> Any:
    """Recursively redact path-like strings in provider-bound payloads."""
    if isinstance(value, str):
        return redact_paths(value)
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_value(item) for key, item in value.items()}
    return value


def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a sanitized copy of an OpenAI-style messages list."""
    return sanitize_value(messages)


# Alias for backward compatibility with existing code
_redact_paths = redact_paths
