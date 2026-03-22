#!/usr/bin/env python3
##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################
"""Claude Code PreToolUse hook: block a small set of destructive Bash patterns."""

from __future__ import annotations

import json
import re
import sys

_DENY_REASONS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\brm\s+(-[a-zA-Z0-9-]+\s+)*-rf\s+/(?:\*(?:\s|$|;|&&|\|\|)|\s*(?:$|;|&&|\|\|))"
        ),
        "rm -rf against filesystem root or / *",
    ),
    (
        re.compile(
            r"\brm\s+(-[a-zA-Z0-9-]+\s+)*-fr\s+/(?:\*(?:\s|$|;|&&|\|\|)|\s*(?:$|;|&&|\|\|))"
        ),
        "rm -fr against filesystem root or / *",
    ),
    (re.compile(r"\bdd\b.*\bof=/dev/(sd|nvme|vd|hd|mmcblk)"), "dd output to a block device"),
    (re.compile(r"\bmkfs\."), "mkfs on a device"),
    (re.compile(r":\(\)\s*\{\s*:\s*\|:\s*&\s*\}\s*;:"), "fork bomb"),
]


def _deny(reason: str) -> None:
    out = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }
    print(json.dumps(out), flush=True)


def main() -> int:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return 0
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return 0

    if data.get("tool_name") != "Bash":
        return 0

    cmd = data.get("tool_input", {}).get("command", "")
    if not isinstance(cmd, str):
        return 0

    for pattern, reason in _DENY_REASONS:
        if pattern.search(cmd):
            _deny(reason)
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
