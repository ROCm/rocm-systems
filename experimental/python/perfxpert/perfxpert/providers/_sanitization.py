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


# Alias for backward compatibility with existing code
_redact_paths = redact_paths
