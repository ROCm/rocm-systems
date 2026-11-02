#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A ``/*`` inside a block comment of amdsmi.h triggers -Wcomment in every includer."""

import unittest
from pathlib import Path

_HEADER_REL = Path("include") / "amd_smi" / "amdsmi.h"

_BLOCK_OPEN = "/*"
_BLOCK_CLOSE = "*/"
_LINE_OPEN = "//"
_DELIM_LEN = len(_BLOCK_OPEN)
_QUOTES = "\"'"

_CODE, _BLOCK, _LINE, _LITERAL = range(4)


def find_nested_comment_openers(text: str) -> list:
    """Return the 1-based line of every ``/*`` that appears inside a block comment."""
    hits = []
    state = _CODE
    quote = ""
    pos = 0
    line = 1
    while pos < len(text):
        pair = text[pos : pos + _DELIM_LEN]
        char = text[pos]
        if state == _CODE:
            if pair in (_BLOCK_OPEN, _LINE_OPEN):
                state = _BLOCK if pair == _BLOCK_OPEN else _LINE
                pos += _DELIM_LEN
                continue
            if char in _QUOTES:
                state, quote = _LITERAL, char
        elif state == _BLOCK:
            if pair == _BLOCK_CLOSE:
                state = _CODE
                pos += _DELIM_LEN
                continue
            if pair == _BLOCK_OPEN:
                hits.append(line)
        elif state == _LINE:
            if char == "\n" and text[pos - 1] != "\\":
                state = _CODE
        elif char == "\\":
            pos += _DELIM_LEN
            continue
        elif char in (quote, "\n"):
            state = _CODE
        if char == "\n":
            line += 1
        pos += 1
    return hits


def find_header() -> Path:
    """Locate amdsmi.h by walking up from this file.

    The same walk resolves ``<repo>/include/...`` in the source tree and
    ``<rocm>/include/...`` when the tests run from ``share/amd_smi/tests``.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / _HEADER_REL
        if candidate.is_file():
            return candidate
    raise unittest.SkipTest("amdsmi.h not found in the source tree or the install prefix")


class NestedCommentOpenersTest(unittest.TestCase):
    def test_reports_line_of_nested_opener(self):
        text = "/* a\n * b/* c\n */\n"
        self.assertEqual(find_nested_comment_openers(text), [2])

    def test_ignores_opener_in_line_comment(self):
        text = "// path/* here\n/* a\n */\n"
        self.assertEqual(find_nested_comment_openers(text), [])

    def test_ignores_opener_in_string_literal(self):
        text = 'const char *s = "/*";\n/* a\n */\n'
        self.assertEqual(find_nested_comment_openers(text), [])

    def test_clean_block_reports_nothing(self):
        self.assertEqual(find_nested_comment_openers("/* a\n * b\n */\n"), [])


class AmdsmiHeaderNestedCommentTest(unittest.TestCase):
    def test_header_has_no_nested_comment_opener(self):
        header = find_header()
        text = header.read_text(encoding="utf-8", errors="replace")
        self.assertEqual(find_nested_comment_openers(text), [], msg=str(header))
