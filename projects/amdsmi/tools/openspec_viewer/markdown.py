"""The small markdown subset OpenSpec bodies actually use.

Paragraphs, bullet and ordered lists, pipe tables, inline code, bold, italic,
links, and bare ``[capability-id]`` cross references. Deliberately not a
general markdown implementation.
"""

from __future__ import annotations

import html
import re
from typing import Dict, List, Sequence, Tuple

from .model import RE_BULLET, RE_FENCE, RE_XREF

# --------------------------------------------------------------------------
# markdown -> html (only what OpenSpec bodies actually use)
# --------------------------------------------------------------------------

RE_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|[\s:|-]*$")
RE_NORMATIVE = re.compile(r"\b(SHALL NOT|SHALL|MUST NOT|MUST|SHOULD NOT|SHOULD|MAY NOT|MAY)\b")
RE_CODE = re.compile(r"`([^`]+)`")
RE_BOLD = re.compile(r"\*\*(.+?)\*\*")
RE_ITAL = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?!\*)")
RE_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
RE_OL = re.compile(r"^\s*\d+[.)]\s+(.*)$")


class Inline:
    """Inline markdown renderer. Knows the capability ids, so ``[an-id]``
    becomes a real link and any other bracketed text is left alone."""

    def __init__(self, anchors: Dict[str, str]):
        self.anchors = anchors

    def __call__(self, text: str, normative: bool = False) -> str:
        keep: List[str] = []

        def stash(m: re.Match) -> str:
            keep.append(f"<code>{html.escape(m.group(1))}</code>")
            return f"\x00{len(keep) - 1}\x00"

        out = RE_CODE.sub(stash, text)
        out = html.escape(out)
        out = RE_LINK.sub(
            lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>', out
        )
        out = RE_XREF.sub(self._xref, out)
        out = RE_BOLD.sub(r"<strong>\1</strong>", out)
        out = RE_ITAL.sub(r"<em>\1</em>", out)
        if normative:
            out = RE_NORMATIVE.sub(r'<b class="kw">\1</b>', out)
        return re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], out)

    def _xref(self, m: re.Match) -> str:
        cid = m.group(1)
        anchor = self.anchors.get(cid)
        if not anchor:
            return m.group(0)
        return f'<a class="xref" href="#{anchor}" data-cap="{cid}">{cid}</a>'


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
        or RE_OL.match(ln)
        or RE_FENCE.match(ln)
        or ("|" in ln and i + 1 < len(lines) and RE_TABLE_SEP.match(lines[i + 1]))
    )


def _cells(row: str) -> List[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


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
