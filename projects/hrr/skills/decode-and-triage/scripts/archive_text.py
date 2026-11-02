#!/usr/bin/env python3
"""The one place archive-authored text is made safe to print.

Whoever recorded an archive chose the strings in its manifest (runtime and
compiler versions, GPU architecture) and the kernel names that a replay or
`hrr-playback --info` reports back. This skill prints them into markdown that
an agent reads and acts on, so a value with a newline and a heading in it
would otherwise appear as a heading of the report, and a backtick would close
the code span it is printed in. Every such string goes through `sanitize`
before it is printed, from Python and, through the filter mode below, from the
shell script.

`sanitize` keeps the text on one line, escapes what could change how it is
read, and caps its length. It escapes rather than drops, so the reader can
still see that something odd was there: `\\x0a` for a newline, `\\x60` for a
backtick.

As a command, `archive_text.py --text STRING` prints one sanitized string, for
the shell script's status lines. With no argument it filters stdin to stdout
one line at a time and marks each line as replay output, so no line of a replay
log can begin a heading or a list item of the report it is printed in.
"""

from __future__ import annotations

import sys
import unicodedata
from typing import Any

# A kernel name is the longest legitimate value (a mangled template symbol runs
# to a few hundred characters); a version or an architecture is far shorter.
MAX_LEN = 256
TRUNCATED = "...[truncated]"

# Prefix of every line the filter mode writes. A line that starts with it cannot
# start a markdown heading, list item, quote or fence.
STREAM_PREFIX = "[data] "


def _escape(ch: str) -> str:
    code = ord(ch)
    if code <= 0xFF:
        return f"\\x{code:02x}"
    return f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}"


def sanitize(value: Any, limit: int = MAX_LEN) -> str:
    """Return `value` as one printable line: no backticks, newlines or controls.

    Control characters, line and paragraph separators and format characters
    (bidirectional overrides among them) are escaped, as is the backslash that
    would make an escape ambiguous. Anything longer than `limit` is cut.
    """
    text = str(value)
    out: list[str] = []
    size = 0
    for ch in text:
        if ch == "`" or ch == "\\" or unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp", "Cs", "Co", "Cn"):
            piece = _escape(ch)
        else:
            piece = ch
        if size + len(piece) > limit:
            out.append(TRUNCATED)
            break
        out.append(piece)
        size += len(piece)
    return "".join(out)


def sanitize_list(values: Any, limit: int = MAX_LEN) -> list[str]:
    return [sanitize(v, limit) for v in values]


def filter_stream(src, dst, limit: int = 1024) -> None:
    """Copy `src` to `dst` a line at a time, sanitized and marked as data."""
    for raw in src:
        dst.write(f"{STREAM_PREFIX}{sanitize(raw.rstrip(chr(10)), limit)}\n")
        dst.flush()


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--text":
        print(sanitize(argv[1]))
        return 0
    if argv:
        print("usage: archive_text.py [--text STRING]", file=sys.stderr)
        return 2
    stdin = open(sys.stdin.fileno(), encoding="utf-8", errors="replace", closefd=False)
    stdout = open(sys.stdout.fileno(), "w", encoding="utf-8", closefd=False)
    try:
        filter_stream(stdin, stdout)
    except BrokenPipeError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
