"""trace_fingerprint — canonical fingerprint for rocprofv3 commands.

Used by Recommendation to detect "this command would collect identical
data to what we already have" — avoids wasted re-profile cycles.

Fingerprint includes: named trace flags (--sys-trace, --hip-trace, etc.)
and --pmc counter names. Ignores output paths (-d, -o) and app args.

Tool class: READ_ONLY.
"""

import re
import shlex
from typing import FrozenSet

from perfxpert.tools._class import ToolClass, tool_class


_NAMED_TRACE_FLAGS = frozenset({
    "--sys-trace", "--hip-trace", "--kernel-trace", "--memory-copy-trace",
    "--hsa-trace", "--stats", "--att", "--pc-sampling",
})

_OUTPUT_FLAGS = {"-d", "--output-dir", "-o", "--output"}


@tool_class(ToolClass.READ_ONLY)
def fingerprint(cmd: str) -> FrozenSet[str]:
    """Compute the canonical set-of-flags fingerprint for a rocprofv3 command.

    Args:
        cmd: full rocprofv3 invocation string.

    Returns:
        Frozen set of canonical tokens. Equal fingerprints = equivalent collection.
    """
    tokens = shlex.split(cmd)
    fp = set()
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok in _NAMED_TRACE_FLAGS:
            fp.add(tok)
            i += 1
        elif tok == "--pmc":
            # Next tokens until next flag or "--" are counter names
            i += 1
            while i < len(tokens) and not tokens[i].startswith("-"):
                fp.add(f"pmc:{tokens[i]}")
                i += 1
        elif tok in _OUTPUT_FLAGS:
            i += 2  # skip flag + value
        elif tok == "--":
            break  # everything after "--" is app args
        else:
            i += 1

    return frozenset(fp)
