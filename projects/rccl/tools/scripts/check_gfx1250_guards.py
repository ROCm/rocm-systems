#!/usr/bin/env python3
"""Fail if a preprocessor guard tests __gfx1250__ without also testing __gfx1250_strict__.

gfx1250 strict builds define __gfx1250_strict__ instead of __gfx1250__, so a guard that
names only __gfx1250__ silently compiles different code for the two builds.

Checks every #if/#elif/#ifdef/#ifndef in C/C++ sources (backslash continuations joined,
comments ignored) and every emitted `defined(__gfx1250__)` in Python code generators.

Usage: check_gfx1250_guards.py [PATH ...]   (default: projects/rccl/src and projects/rccl/test)
"""

import pathlib
import re
import sys

CPP_SUFFIXES = {".h", ".hpp", ".cc", ".cpp", ".cu", ".cuh", ".inc"}
DIRECTIVE = re.compile(r"^\s*#\s*(?:if|elif|ifdef|ifndef)\b")
COMMENT = re.compile(r"//.*|/\*.*?\*/")
CPP_PLAIN = re.compile(r"\b__gfx1250__\b")
CPP_STRICT = re.compile(r"\b__gfx1250_strict__\b")
PY_PLAIN = re.compile(r"(?:defined\s*\(?\s*|#\s*ifn?def\s+)__gfx1250__\b")
PY_STRICT = re.compile(r"(?:defined\s*\(?\s*|#\s*ifn?def\s+)__gfx1250_strict__\b")


def cpp_directives(text):
    start, buf = None, []
    for lineno, line in enumerate(text.splitlines(), 1):
        start = start or lineno
        if line.endswith("\\"):
            buf.append(line[:-1])
            continue
        buf.append(line)
        joined = " ".join(buf)
        if DIRECTIVE.match(joined):
            yield start, COMMENT.sub("", joined)
        start, buf = None, []


def check(path):
    text = path.read_text(errors="ignore")
    if "__gfx1250__" not in text:
        return []
    if path.suffix == ".py":
        lines, plain, strict = enumerate(text.splitlines(), 1), PY_PLAIN, PY_STRICT
    else:
        lines, plain, strict = cpp_directives(text), CPP_PLAIN, CPP_STRICT
    return [
        f"{path}:{lineno}: __gfx1250__ guard without matching __gfx1250_strict__: {line.strip()}"
        for lineno, line in lines
        if len(plain.findall(line)) > len(strict.findall(line))
    ]


def iter_files(paths):
    for root in paths:
        if root.is_file():
            yield root
            continue
        for path in root.rglob("*"):
            if path.suffix in CPP_SUFFIXES or path.suffix == ".py":
                yield path


def main(argv):
    rccl = pathlib.Path(__file__).resolve().parents[2]
    roots = [pathlib.Path(arg) for arg in argv] or [rccl / "src", rccl / "test"]
    errors = [err for path in iter_files(roots) for err in check(path)]
    print("\n".join(errors) if errors else "gfx1250 guards OK")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
