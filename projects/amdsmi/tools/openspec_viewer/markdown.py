# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The small markdown subset OpenSpec bodies actually use.

Paragraphs, bullet and ordered lists, pipe tables, inline code, bold, italic
(both ``*`` and ``_``), backslash escapes, links, and bare
``[capability-id]`` cross references. Deliberately not a general markdown
implementation.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Dict, List, Sequence, Tuple

from .model import RE_BULLET, RE_FENCE, RE_XREF

# --------------------------------------------------------------------------
# markdown -> html (only what OpenSpec bodies actually use)
# --------------------------------------------------------------------------

RE_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|[\s:|-]*$")
RE_NORMATIVE = re.compile(r"\b(SHALL NOT|SHALL|MUST NOT|MUST|SHOULD NOT|SHOULD|MAY NOT|MAY)\b")
# Backslash escapes and code spans resolved in one left-to-right pass, which is
# what gives them CommonMark's mutual precedence: ``\``` cannot open a span
# (the escape alternative consumes it first), while a backslash inside a span
# stays literal (the span alternative consumes the whole run). The escapable
# set is the 32 ASCII punctuation characters.
RE_ESCAPE_OR_CODE = re.compile(r"\\([!-/:-@\[-`{-~])|`([^`]+)`")
RE_BOLD = re.compile(r"\*\*(.+?)\*\*")
RE_ITAL = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?!\*)")
RE_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
RE_OL = re.compile(r"^\s*\d+[.)]\s+(.*)$")
# Only a list numbered 1 may interrupt a paragraph, per CommonMark. The specs
# wrap at 80 columns, so a sentence can put "84." at the start of a line; that
# is prose, not a list.
RE_OL_ONE = re.compile(r"^\s*1[.)]\s+")
RE_QUOTE = re.compile(r"^ {0,3}>[ \t]?(.*)$")

# Marks a stashed code span or escaped character. Emphasis treats it as
# punctuation, which is how CommonMark sees the backtick it stands in for.
STASH = "\x00"


class Inline:
    """Inline markdown renderer. Knows the capability ids, so ``[an-id]``
    becomes a real link and any other bracketed text is left alone."""

    def __init__(self, anchors: Dict[str, str]):
        self.anchors = anchors

    def __call__(self, text: str, normative: bool = False) -> str:
        keep: List[str] = []

        def stash(m: re.Match) -> str:
            esc, code = m.group(1), m.group(2)
            keep.append(html.escape(esc) if code is None else f"<code>{html.escape(code)}</code>")
            return f"{STASH}{len(keep) - 1}{STASH}"

        out = RE_ESCAPE_OR_CODE.sub(stash, text)
        out = html.escape(out)
        # Before the link rule, so underscores in an href are never delimiters.
        out = _emph_underscore(out)
        out = RE_LINK.sub(
            lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>', out
        )
        out = RE_XREF.sub(self._xref, out)
        out = RE_BOLD.sub(r"<strong>\1</strong>", out)
        out = RE_ITAL.sub(r"<em>\1</em>", out)
        if normative:
            out = RE_NORMATIVE.sub(r'<b class="kw">\1</b>', out)
        return re.sub(rf"{STASH}(\d+){STASH}", lambda m: keep[int(m.group(1))], out)

    def _xref(self, m: re.Match) -> str:
        cid = m.group(1)
        anchor = self.anchors.get(cid)
        if not anchor:
            return m.group(0)
        return f'<a class="xref" href="#{anchor}" data-cap="{cid}">{cid}</a>'


def _punct(c: str) -> bool:
    return c == STASH or unicodedata.category(c)[0] in "PS"


def _delim(text: str, i: int) -> Tuple[bool, bool]:
    """``(can_open, can_close)`` for the lone ``_`` at ``text[i]``.

    CommonMark's flanking rules. An absent neighbour counts as whitespace. The
    extra clause each way -- an opener must not also be right-flanking unless
    it follows punctuation, and vice versa -- is precisely what forbids
    intraword ``_``, and so what keeps ``amd_smi``, ``x86_64`` and
    ``pcie_nak_sent_count_acc`` out of the emphasis machinery.
    """
    before = text[i - 1] if i else " "
    after = text[i + 1] if i + 1 < len(text) else " "
    b_ws, a_ws = before.isspace(), after.isspace()
    b_p, a_p = _punct(before), _punct(after)
    left = not a_ws and (not a_p or b_ws or b_p)
    right = not b_ws and (not b_p or a_ws or a_p)
    return left and (not right or b_p), right and (not left or a_p)


def _emph_underscore(text: str) -> str:
    """``_emphasis_`` -> ``<em>``. Runs of two or more underscores are left
    alone, so ``__init__`` survives even outside a code span."""
    if "_" not in text:
        return text
    opens: List[int] = []
    tags: Dict[int, str] = {}
    i, n = 0, len(text)
    while i < n:
        if text[i] != "_":
            i += 1
            continue
        j = i
        while j < n and text[j] == "_":
            j += 1
        if j - i == 1:
            can_open, can_close = _delim(text, i)
            if can_close and opens:
                tags[opens.pop()] = "<em>"
                tags[i] = "</em>"
            elif can_open:
                opens.append(i)
        i = j
    if not tags:
        return text
    return "".join(tags.get(k, c) for k, c in enumerate(text))


def render_md(lines: Sequence[str], inline: Inline, normative: bool = False) -> str:
    """Block-level markdown: paragraphs, lists, GFM tables, fenced code."""
    out: List[str] = []
    i, n = 0, len(lines)
    while i < n:
        raw = lines[i]
        if not raw.strip():
            i += 1
            continue
        if RE_FENCE.match(raw):
            i += 1
            buf = []
            while i < n and not RE_FENCE.match(lines[i]):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            continue
        if RE_QUOTE.match(raw):
            buf = []
            while i < n and (q := RE_QUOTE.match(lines[i])):
                buf.append(q.group(1))
                i += 1
            out.append("<blockquote>" + render_md(buf, inline, normative) + "</blockquote>")
            continue
        if "|" in raw and i + 1 < n and RE_TABLE_SEP.match(lines[i + 1]):
            i, tbl = _table(lines, i, inline)
            out.append(tbl)
            continue
        if RE_BULLET.match(raw) or RE_OL.match(raw):
            i, lst = _list(lines, i, inline, normative)
            out.append(lst)
            continue
        buf = []
        while i < n and lines[i].strip() and not _block_start(lines, i):
            buf.append(lines[i].strip())
            i += 1
        out.append("<p>" + inline(" ".join(buf), normative) + "</p>")
    return "\n".join(out)


def _block_start(lines: Sequence[str], i: int) -> bool:
    if i == 0:
        return False
    ln = lines[i]
    return bool(
        RE_BULLET.match(ln)
        or RE_OL_ONE.match(ln)
        or RE_FENCE.match(ln)
        or RE_QUOTE.match(ln)
        or ("|" in ln and i + 1 < len(lines) and RE_TABLE_SEP.match(lines[i + 1]))
    )


def _cells(row: str) -> List[str]:
    """Split a table row on its cell separators -- a ``|`` that is neither
    backslash-escaped nor inside a code span, GFM's two ways of putting a
    literal pipe in a cell."""
    row = row.strip()
    guard = bytearray(len(row))
    for m in RE_ESCAPE_OR_CODE.finditer(row):
        guard[m.start() : m.end()] = b"\x01" * (m.end() - m.start())
    parts, last = [], 0
    for k, c in enumerate(row):
        if c == "|" and not guard[k]:
            parts.append(row[last:k])
            last = k + 1
    parts.append(row[last:])
    if parts and not parts[0]:
        parts.pop(0)
    if parts and not parts[-1]:
        parts.pop()
    return [c.strip() for c in parts]


def _table(lines: Sequence[str], i: int, inline: Inline) -> Tuple[int, str]:
    head = _cells(lines[i])
    i += 2
    body: List[List[str]] = []
    while i < len(lines) and "|" in lines[i] and lines[i].strip():
        body.append(_cells(lines[i]))
        i += 1
    th = "".join(f"<th>{inline(c)}</th>" for c in head)
    rows = []
    for r in body:
        r = (r + [""] * len(head))[: len(head)]
        rows.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
    return i, (
        f'<div class="tw"><table><thead><tr>{th}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _list(lines: Sequence[str], i: int, inline: Inline, normative: bool) -> Tuple[int, str]:
    ordered = bool(RE_OL.match(lines[i]))
    items: List[str] = []
    while i < len(lines):
        m = RE_OL.match(lines[i]) if ordered else RE_BULLET.match(lines[i])
        if m:
            items.append(m.group(1).strip())
            i += 1
        elif lines[i].strip() and lines[i][:1] in " \t" and items:
            items[-1] += " " + lines[i].strip()
            i += 1
        else:
            break
    tag = "ol" if ordered else "ul"
    body = "".join(f"<li>{inline(x, normative)}</li>" for x in items)
    return i, f"<{tag}>{body}</{tag}>"
