# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Assemble the self-contained HTML page from a parsed :class:`Site`.

One page holds one or more projects. Each project contributes a pane with its
own masthead, capability list, reference diagram, specs and changes; a switcher
in the rail picks which pane is on screen. A site of one project renders
without any switcher chrome at all.

Two things carry meaning through colour here, and only two:

* the **spec surface** -- capabilities, requirements, cross references -- in
  violet, and the two halves of a scenario in blue (WHEN, the condition) and
  green (THEN, the asserted outcome);
* a **proposed change to that surface** -- ADDED in green, MODIFIED in amber,
  REMOVED in red, each also carrying a ``+`` / ``~`` / ``-`` sigil and its own
  border treatment so the distinction survives greyscale, print and colour
  blindness.

Nothing else is coloured.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .assets import CSS, JS, favicon, logo
from .graph import (
    Graph,
    capability_groups,
    flow_legend,
    graph_prefix,
    svg_delivery,
    svg_flow,
    svg_graph,
)
from .markdown import Inline, render_md
from .model import (
    RE_FENCE,
    RE_XREF,
    Capability,
    Change,
    Project,
    Requirement,
    Scenario,
    Site,
    slugify,
    source_org,
)

E = html.escape

#: display order and the non-colour cue for each delta kind
DELTA_ORDER = ("ADDED", "MODIFIED", "REMOVED", "RENAMED")
SIGIL = {"ADDED": "+", "MODIFIED": "~", "REMOVED": "\u2212", "RENAMED": "\u2192"}

#: a phase list longer than this starts collapsed
PHASE_OPEN_MAX = 10


# --------------------------------------------------------------------------
# anchors: one id space for the whole page, across every project
# --------------------------------------------------------------------------


class Anchors:
    """Allocates a unique element id per model object.

    Slugs are only deduplicated within a single spec file, and the same
    capability can be delta'd by two changes, so uniqueness has to be settled
    once for the whole page rather than per project.
    """

    def __init__(self) -> None:
        self._taken: Set[str] = set()
        self._by_obj: Dict[int, str] = {}
        self._pin: List[object] = []  # keep ids from being recycled

    def take(self, obj: object, *parts: str) -> str:
        base = slugify(*parts) or "x"
        out, n = base, 2
        while out in self._taken:
            out, n = f"{base}-{n}", n + 1
        self._taken.add(out)
        self._by_obj[id(obj)] = out
        self._pin.append(obj)
        return out

    def of(self, obj: object) -> str:
        return self._by_obj.get(id(obj), "")


# --------------------------------------------------------------------------
# small shared pieces
# --------------------------------------------------------------------------

RE_DOC_HEAD = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
RE_KW_NOT = re.compile(r'<b class="kw">([A-Z]+ NOT)</b>')


def _neg(markup: str) -> str:
    """Split the normative keywords: a prohibition reads as a prohibition."""
    return RE_KW_NOT.sub(r'<b class="kw no">\1</b>', markup)


def md(lines: Sequence[str], inline: Inline, normative: bool = False) -> str:
    return _neg(render_md(lines, inline, normative))


def il(inline: Inline, text: str, normative: bool = False) -> str:
    return _neg(inline(text, normative))


def first_sentence(lines: Sequence[str], limit: int = 170) -> str:
    """A one-line gist: first real sentence, headings and markup stripped."""
    words: List[str] = []
    fenced = False
    for raw in lines:
        if RE_FENCE.match(raw):
            fenced = not fenced
            continue
        if fenced or RE_DOC_HEAD.match(raw) or raw.lstrip().startswith(("-", "*", "|", ">")):
            if words:
                break
            continue
        if not raw.strip():
            if words:
                break
            continue
        words.append(raw.strip())
    text = re.sub(r"[`*\[\]]", "", " ".join(words))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    m = re.search(r"^(.{40,%d}?[.;])\s" % limit, text + " ")
    if m:
        return m.group(1).strip()
    return text[:limit].rsplit(" ", 1)[0] + "\u2026" if len(text) > limit else text


def split_doc(body: Sequence[str]) -> List[Tuple[int, str, List[str]]]:
    """``[(heading level, heading text, body lines)]``; level 0 is a preamble."""
    chunks: List[Tuple[int, str, List[str]]] = []
    level, label, buf = 0, "", []  # type: int, str, List[str]
    fenced = False
    for line in body:
        if RE_FENCE.match(line):
            fenced = not fenced
        m = None if fenced else RE_DOC_HEAD.match(line)
        if m:
            if label or any(x.strip() for x in buf):
                chunks.append((level, label, buf))
            level, label, buf = len(m.group(1)), m.group(2), []
        else:
            buf.append(line)
    if label or any(x.strip() for x in buf):
        chunks.append((level, label, buf))
    return chunks


def render_doc(body: Sequence[str], inline: Inline) -> str:
    """A change document as label-in-the-gutter sections.

    Proposals are structured arguments -- Why, What Changes, Impact -- so the
    top-level headings become a fixed left column and never compete with the
    page's own heading scale.
    """
    out: List[str] = []
    is_open = False

    def close() -> None:
        if out and is_open:
            out.append("</div></div>")

    def start(text: str) -> None:
        nonlocal is_open
        close()
        lab = f'<div class="dlab">{il(inline, text)}</div>' if text else '<div class="dlab"></div>'
        out.append(f'<div class="dsec">{lab}<div class="prose">')
        is_open = True

    for level, label, lines in split_doc(body):
        if level <= 2:
            start(label)
        else:
            if not is_open:
                start("")
            out.append(f'<h5 class="dh">{il(inline, label)}</h5>')
        if any(x.strip() for x in lines):
            out.append(md(lines, inline))
    close()
    return "".join(out)


def doc_outline(body: Sequence[str], limit: int = 5) -> str:
    heads = [lab for lvl, lab, _ in split_doc(body) if 0 < lvl <= 2 and lab]
    if not heads:
        return ""
    shown = heads[:limit]
    more = f" +{len(heads) - limit}" if len(heads) > limit else ""
    return " \u00b7 ".join(shown) + more


# --------------------------------------------------------------------------
# delta vocabulary and progress
# --------------------------------------------------------------------------


def delta_pill(kind: str) -> str:
    if not kind:
        return ""
    k = kind.lower()
    return f'<span class="delta {k}"><b>{SIGIL.get(kind, "*")}</b>{E(k)}</span>'


def delta_ledger(counts: Dict[str, int]) -> str:
    """``+7 ~2 -1`` -- sigil first so the reading survives without colour."""
    chips = [
        f'<span class="dl {k.lower()}" title="{counts[k]} {k.lower()}">'
        f"<b>{SIGIL[k]}</b>{counts[k]}</span>"
        for k in DELTA_ORDER
        if counts.get(k)
    ]
    return f'<span class="ledger">{"".join(chips)}</span>' if chips else ""


#: The smallest cell that still reads as a cell: a 4px box -- 1px border, 2px
#: interior, 1px border -- plus a 1.5px gap, so two neighbours do not merge
#: their borders into one line. Measured against a 30-cell strip drawn at box
#: widths 2/3/3.5/4/5/6/8px: below this a *hollow* cell collapses to a hairline
#: and the tape reads as a solid smear with pale patches, which is exactly the
#: lie being fixed. Seeing the unchecked cells is the whole point of cells.
MIN_CELL_PITCH = 5.5

#: Width in px of each tape variant's cell track, measured in the browser on a
#: 1280px page: the fluid masthead gauge, the fluid change bar, and the fixed
#: 150px ``sm`` tape shared by the change index and the per-phase summary.
TAPE_TRACK = {"wide": 600.0, "": 689.0, "sm": 113.0}


def tape_max_cells(cls: str = "") -> int:
    """How many cells this tape variant can draw and still be countable."""
    return int(TAPE_TRACK.get(cls, TAPE_TRACK[""]) // MIN_CELL_PITCH)


def _group_title(name: str, flags: Sequence[bool]) -> str:
    return f"{name or 'ungrouped'} \u2014 {sum(1 for d in flags if d)}/{len(flags)} done"


def _cells(groups: Sequence[Tuple[str, List[bool]]]) -> str:
    """One cell per task: filled is done, hollow is not."""
    return "".join(
        f'<span class="seg" style="flex:{len(f)}" title="{E(_group_title(n, f))}">'
        + "".join('<i class="on"></i>' if d else "<i></i>" for d in f)
        + "</span>"
        for n, f in groups
    )


def _bars(groups: Sequence[Tuple[str, List[bool]]]) -> str:
    """One proportional segment per group, filled in proportion to its progress.

    Segment widths are ``flex:len(group)`` exactly as in :func:`_cells`, and the
    fill is that group's completion, so the green covers ``done/total`` of the
    track in both drawings and the two modes cannot disagree.
    """
    out = []
    for n, f in groups:
        pct = 100.0 * sum(1 for d in f if d) / len(f)
        out.append(
            f'<span class="seg" style="flex:{len(f)}" title="{E(_group_title(n, f))}">'
            f'<i class="on" style="width:{pct:.4g}%"></i></span>'
        )
    return "".join(out)


def tape(groups: Sequence[Tuple[str, Sequence[bool]]], done: int, total: int, cls: str = "") -> str:
    """The page's ruler: how much of a change's work is done, by group.

    One cell per task while cells stay distinguishable -- filled is done, hollow
    is not, so it reads in greyscale and in print. Past :func:`tape_max_cells`
    that drawing is a lie: the cells are narrower than their own borders and the
    tape smears. There it becomes a proportional bar of one segment per group,
    which answers the better question anyway -- not "which of 155 identical
    ticks is unchecked" but "which phase is lagging".
    """
    kept = [(n, [bool(d) for d in f]) for n, f in groups if len(f)]
    if not kept:
        return ""
    cells = sum(len(f) for _, f in kept)
    bar = cells > tape_max_cells(cls)
    pct = int(round(100 * done / total)) if total else 0
    return (
        f'<span class="prog {cls}">'
        f'<span class="tape{" bar" if bar else ""}" role="img" '
        f'aria-label="{done} of {total} tasks done, {pct}%">'
        f"{_bars(kept) if bar else _cells(kept)}</span>"
        f'<span class="pnum"><b>{done}</b>/{total}</span></span>'
    )


def change_tape(chg: Change, cls: str = "") -> str:
    """A change's own tape, grouped by phase."""
    return tape(
        [(p.name, [t.done for t in p.tasks]) for p in chg.phases],
        chg.task_done_count,
        chg.task_count,
        cls,
    )


def change_counts(chg: Change) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for cap in chg.deltas:
        for k, v in cap.delta_counts.items():
            out[k] = out.get(k, 0) + v
    return out


def delta_rule(counts: Dict[str, int]) -> str:
    """The rule above a change, split in proportion to what it does."""
    total = sum(counts.values())
    if not total:
        return '<div class="drule"><span class="none"></span></div>'
    bits = "".join(
        f'<span class="{k.lower()}" style="flex:{counts[k]}"></span>'
        for k in DELTA_ORDER
        if counts.get(k)
    )
    return f'<div class="drule" aria-hidden="true">{bits}</div>'


# --------------------------------------------------------------------------
# spec units
# --------------------------------------------------------------------------


def render_steps(scn: Scenario, inline: Inline) -> str:
    rows = []
    for st in scn.steps:
        cls = {"WHEN": "when", "THEN": "then"}.get(st.kind, "other" if st.kind else "plain")
        rows.append(
            f'<li class="step {cls}"><span class="k">{E(st.kind or "-")}</span>'
            f'<div class="t">{il(inline, st.text, st.kind == "THEN")}</div></li>'
        )
    return f'<ul class="steps">{"".join(rows)}</ul>' if rows else ""


def render_requirement(req: Requirement, ord_txt: str, inline: Inline, A: Anchors) -> str:
    slug = A.of(req)
    cls = f"req d-{req.delta.lower()}" if req.delta else "req"
    scns = []
    for j, scn in enumerate(req.scenarios, 1):
        scns.append(
            f'<article class="scn" id="{A.of(scn)}" data-leaf>'
            f'<div class="scn-head"><span class="ord">{ord_txt[1:]}.{j}</span>'
            f"<h4>{il(inline, scn.title)}</h4>"
            f'<a class="anchor" href="#{A.of(scn)}" aria-label="Link to this scenario">#</a>'
            f"</div>{render_steps(scn, inline)}</article>"
        )
    body = md(req.body, inline, normative=True) if req.body else ""
    return (
        f'<section class="{cls}" id="{slug}" data-item'
        f"{f' data-delta={req.delta}' if req.delta else ''}>"
        f'<div class="req-head"><span class="ord">{E(ord_txt)}</span>'
        f"<h3>{il(inline, req.title)}</h3>{delta_pill(req.delta)}"
        f'<a class="anchor" href="#{slug}" aria-label="Link to this requirement">#</a></div>'
        f'<div class="prose">{body}</div>'
        f"{f'<div class=scns>{chr(10).join(scns)}</div>' if scns else ''}"
        "</section>"
    )


def ref_row(label: str, ids: Sequence[str], anchors: Dict[str, str]) -> str:
    """One labelled row of capability chips; the label sits in its own gutter."""
    chips = "".join(f'<a href="#{anchors[i]}">{E(i)}</a>' for i in ids if i in anchors)
    if not chips:
        return ""
    return f'<div class="rrow"><span>{E(label)}</span><div class="rchips">{chips}</div></div>'


def render_capability(
    cap: Capability,
    inline: Inline,
    out_refs: Sequence[str],
    in_refs: Sequence[str],
    anchors: Dict[str, str],
    A: Anchors,
) -> str:
    reqs = "".join(
        render_requirement(r, f"R{i}", inline, A) for i, r in enumerate(cap.requirements, 1)
    )
    rows = [
        ref_row(label, ids, anchors)
        for label, ids in (("references", out_refs), ("referenced by", in_refs))
    ]
    ref_html = f'<div class="reflist">{"".join(rows)}</div>' if any(rows) else ""
    return (
        f'<section class="cap unit" id="{A.of(cap)}" data-unit>'
        f'<div class="cap-head"><h2>{E(cap.title)}</h2>'
        f'<a class="anchor" href="#{A.of(cap)}" aria-label="Link to this capability">#</a>'
        f'<span class="meta">{len(cap.requirements)} req &middot; '
        f"{cap.scenario_count} scn</span></div>"
        f'<div class="purpose">{md(cap.purpose, inline)}</div>'
        f"{ref_html}{reqs}</section>"
    )


# --------------------------------------------------------------------------
# changes
# --------------------------------------------------------------------------


def render_tasks(chg: Change, A: Anchors, inline: Inline) -> str:
    if not chg.phases:
        return ""
    start_open = chg.task_count <= PHASE_OPEN_MAX
    rows = []
    for ph in chg.phases:
        items = "".join(
            f'<li class="task{" done" if t.done else ""}" data-leaf '
            f'style="--d:{min(t.depth, 4)}">'
            f'<span class="box" aria-hidden="true"></span>'
            f'<span class="tt">{il(inline, t.text)}</span></li>'
            for t in ph.tasks
        )
        head = (
            f'<summary><span class="pn">{E(ph.name or "ungrouped")}</span>'
            f"{tape([(ph.name, [t.done for t in ph.tasks])], ph.done_count, len(ph.tasks), 'sm')}"
            "</summary>"
        )
        rows.append(
            f'<details class="phase" id="{A.of(ph)}" data-item'
            f"{' open' if start_open else ''}>{head}"
            f'<ul class="tasks">{items}</ul></details>'
        )
    return (
        '<div class="tasks-block"><div class="blab">'
        f'tasks<a class="tall" href="#" data-all>expand all</a></div>'
        f"{''.join(rows)}</div>"
    )


def render_change(chg: Change, inline: Inline, A: Anchors, gmap: Dict[str, str], site: Site) -> str:
    counts = change_counts(chg)
    slug = A.of(chg)

    docs = []
    for name, body in chg.docs:
        primary = name.lower() in ("proposal", "why")
        outline = doc_outline(body)
        docs.append(
            f'<details class="doc" data-item{" open" if primary else ""}>'
            f'<summary><span class="dn">{E(name)}</span>'
            f'<span class="dsum">{E(outline)}</span></summary>'
            f'<div class="dbody">{render_doc(body, inline)}</div></details>'
        )

    deltas = []
    for cap in chg.deltas:
        reqs = "".join(
            render_requirement(r, f"R{i}", inline, A) for i, r in enumerate(cap.requirements, 1)
        )
        owner = _baseline_link(cap.cid, site, gmap)
        deltas.append(
            f'<section class="dspec" id="{A.of(cap)}" data-box>'
            f'<div class="dspec-head"><h4>{E(cap.cid)}</h4>'
            f"{delta_ledger(cap.delta_counts)}{owner}"
            f'<a class="anchor" href="#{A.of(cap)}" aria-label="Link to this delta">#</a></div>'
            f'<div class="purpose">{md(cap.purpose, inline)}</div>{reqs}</section>'
        )
    deltas_html = (
        f'<div class="delta-block"><div class="blab">delta specs '
        f"<em>{len(chg.deltas)}</em></div>{''.join(deltas)}</div>"
        if deltas
        else ""
    )

    # the title is only worth printing when it says something the id does not
    sub = E(chg.title) if chg.title and slugify(chg.title) != slugify(chg.cid) else ""
    refs = [r for r in getattr(chg, "refs", None) or [] if r in gmap]
    row = ref_row("references", refs, gmap)
    ref_html = f'<div class="reflist chg-refs">{row}</div>' if row else ""
    return (
        f'<section class="chg unit" id="{slug}" data-unit>'
        f"{delta_rule(counts)}"
        f'<div class="chg-head">'
        f'<div class="chg-id"><h2>{E(chg.cid)}</h2>'
        f'<a class="anchor" href="#{slug}" aria-label="Link to this change">#</a></div>'
        f"{f'<p class=chg-sub>{sub}</p>' if sub else ''}"
        f'<div class="chg-bar">{delta_ledger(counts)}'
        f'<span class="cmeta">{len(chg.deltas)} delta spec'
        f"{'s' if len(chg.deltas) != 1 else ''}</span>"
        f"{change_tape(chg)}</div>{ref_html}</div>"
        f"{''.join(docs)}{render_tasks(chg, A, inline)}{deltas_html}"
        "</section>"
    )


def _baseline_link(cid: str, site: Site, anchors: Dict[str, str]) -> str:
    """Point a delta at the baseline capability it revises, if there is one."""
    target = site.resolve(cid)
    if not target or target == cid:
        return ""
    owner = site.capability_owner(target)
    if not owner or target not in anchors:
        return ""
    return f'<a class="owner" href="#{anchors[target]}">revises {E(target)}</a>'


def render_change_index(changes: Sequence[Change], A: Anchors) -> str:
    rows = []
    for chg in changes:
        gist = ""
        for name, body in chg.docs:
            if name.lower() in ("proposal", "why"):
                gist = first_sentence(body)
                break
        gist = gist or chg.title
        rows.append(
            f'<a href="#{A.of(chg)}"><span class="nm">{E(chg.cid)}</span>'
            f'<span class="ds">{E(gist)}</span>'
            f"{delta_ledger(change_counts(chg))}{change_tape(chg, 'sm')}</a>"
        )
    return f'<div class="cidx">{"".join(rows)}</div>'


# --------------------------------------------------------------------------
# project pane
# --------------------------------------------------------------------------


def render_context(context: str, root: str) -> str:
    if not context.strip():
        return (
            '<div class="empty"><code>openspec/config.yaml</code> has no '
            "<code>context:</code> block.</div>"
        )
    body = E(context)
    body = re.sub(r"`([^`\n]+)`", r"<b>\1</b>", body)
    _ = root
    return f'<div class="ctx"><pre>{body}</pre></div>'


def render_delivery(doc: Dict[str, Any], anchors: Dict[str, str], sec_id: str) -> str:
    """The hand-written build-and-delivery diagram, titled by the file itself."""
    svg = svg_delivery(doc, anchors)
    if not svg:
        return ""
    title = str(doc.get("title") or "build and delivery")
    subtitle = str(doc.get("subtitle") or "")
    legend = (
        '<div class="maplegend">'
        "<span>a node links to the capability that specifies it</span>"
        "<span>hand-written, not derived from the specs</span></div>"
    )
    return (
        f'<section class="block oview" id="{sec_id}">'
        f'<h2 class="sec">{E(title)}'
        f"{f'<em>{E(subtitle)}</em>' if subtitle else ''}</h2>"
        f'<div class="dlvwrap">{svg}{legend}</div></section>'
    )


def _prime_anchors(project: Project, A: Anchors, multi: bool) -> None:
    p = project.slug if multi else ""
    for cap in project.capabilities:
        A.take(cap, p, cap.slug)
        for req in cap.requirements:
            A.take(req, p, req.slug)
            for scn in req.scenarios:
                A.take(scn, p, scn.slug)
    for chg in project.changes:
        A.take(chg, p, chg.slug)
        for ph in chg.phases:
            A.take(ph, p, chg.slug, ph.slug or ph.name or "phase")
        for cap in chg.deltas:
            A.take(cap, p, chg.slug, cap.slug)
            for req in cap.requirements:
                A.take(req, p, chg.slug, req.slug)
                for scn in req.scenarios:
                    A.take(scn, p, chg.slug, scn.slug)


def _global_anchors(projects: Sequence[Project], A: Anchors) -> Dict[str, str]:
    """``capability id -> element id`` across every pane on the page.

    A baseline capability wins over a delta of the same id, so a reference
    lands on the specification rather than on somebody's proposed edit to it.
    """
    out: Dict[str, str] = {}
    for p in projects:
        for chg in p.changes:
            for cap in chg.deltas:
                out.setdefault(cap.cid, A.of(cap))
    for p in projects:
        for cap in p.capabilities:
            out[cap.cid] = A.of(cap)
    for p in projects:  # ``chg:`` keyed so svg_flow can link a proposal node
        for chg in p.changes:
            out[f"chg:{chg.cid}"] = A.of(chg)
    return out


def _flow_anchors(gmap: Dict[str, str]) -> Dict[str, str]:
    """``svg_flow`` prefers ``cap:<id>`` / ``chg:<id>`` keys; give it both forms."""
    out = dict(gmap)
    for cid, anchor in gmap.items():
        out.setdefault(f"cap:{cid}", anchor)
    return out


def load_delivery(project: Project) -> Optional[Dict[str, Any]]:
    """``<root>/diagrams/delivery.json``, or ``None`` when the corpus has none.

    :class:`Project` has no field for this document, so it is read here. A
    malformed file raises rather than disappearing: only its absence is
    ordinary.
    """
    path = project.root / "diagrams" / "delivery.json"
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not doc.get("nodes"):
        return None
    return doc


def _cap_refs(cap: Capability, known: Set[str]) -> List[str]:
    refs = list(getattr(cap, "refs", None) or [])
    if not refs:
        refs = RE_XREF.findall(cap.raw)
    return sorted({r for r in refs if r in known and r != cap.cid})


def render_project(
    project: Project, site: Site, A: Anchors, stamp: str, gmap: Dict[str, str]
) -> Tuple[str, str]:
    """``(rail html, main html)`` for one project pane."""
    caps = project.capabilities
    changes = project.changes
    multi = site.multi
    pre = f"{project.slug}--" if multi else ""
    anchors = {c.cid: A.of(c) for c in caps}  # this project only: the reference diagram
    inline = Inline(gmap)  # every project: a reference may cross a pane

    known = set(anchors)
    edges: Set[Tuple[str, str]] = set()
    for cap in caps:
        for ref in _cap_refs(cap, known):
            edges.add((cap.cid, ref))
    out_refs = {c.cid: sorted(v for u, v in edges if u == c.cid) for c in caps}
    in_refs = {c.cid: sorted(u for u, v in edges if v == c.cid) for c in caps}

    # -- overview -----------------------------------------------------------
    blocks: List[str] = []
    blocks.append(
        f'<section class="block oview" id="{pre}ctx"><h2 class="sec">context</h2>'
        f"{render_context(project.context, str(project.root))}</section>"
    )

    # packaging is what this corpus is mostly about, so the delivery diagram
    # sits directly under the context, above the reference diagram
    delivery = load_delivery(project)
    if delivery:
        blocks.append(render_delivery(delivery, gmap, f"{pre}dlv"))

    # a diagram of unconnected boxes says nothing the list does not, and
    # Graph() currently raises KeyError on an edgeless node set (reported to B)
    if caps and edges:
        cids = [c.cid for c in caps]
        graph = Graph(cids, edges, groups=capability_groups(project.context, cids))
        titles = {
            c.cid: f"{c.cid} - {len(c.requirements)} requirements, "
            f"{c.scenario_count} scenarios, referenced by {len(in_refs[c.cid])}"
            for c in caps
        }
        prefix = graph_prefix([c.cid for c in caps])
        legend = (
            '<div class="maplegend">'
            f"<span>{E(prefix) + '*' if prefix else 'capability'} = capability</span>"
            '<span class="o"><i></i>references</span>'
            '<span class="i"><i></i>referenced by</span>'
            "<span>arrow points at the referenced capability</span>"
            "<span>click holds a node, double-click opens it, Esc releases</span></div>"
        )
        blocks.append(
            f'<section class="block oview" id="{pre}map">'
            '<h2 class="sec">capability references</h2>'
            f'<div class="mapwrap">{svg_graph(graph, anchors, titles)}{legend}</div>'
            "</section>"
        )

    if caps:
        idx = "".join(
            f'<a href="#{A.of(c)}"><span class="nm">{E(c.cid)}</span>'
            f'<span class="ds">{E(first_sentence(c.purpose))}</span>'
            f"{strip_svg([len(r.scenarios) for r in c.requirements])}"
            f'<span class="ct"><b>{len(c.requirements)}</b> req<br>'
            f"<b>{c.scenario_count}</b> scn</span></a>"
            for c in caps
        )
        blocks.append(
            f'<section class="block oview" id="{pre}idx">'
            f'<h2 class="sec">capabilities <em>{len(caps)}</em></h2>'
            f'<div class="idx">{idx}</div></section>'
        )

    if changes:
        blocks.append(
            f'<section class="block oview" id="{pre}cidx">'
            f'<h2 class="sec">changes <em>{len(changes)}</em>'
            f'<span class="key">{delta_key()}</span></h2>'
            f"{render_change_index(changes, A)}</section>"
        )
        # one diagram per project, not per change: the thing worth seeing is
        # several proposals converging on the same capability
        flow = svg_flow(site.projects, _flow_anchors(gmap), focus=project)
        if flow:
            blocks.append(
                f'<section class="block oview" id="{pre}flow">'
                '<h2 class="sec">changes and capabilities</h2>'
                f'<div class="flowwrap">{flow}{flow_legend()}</div></section>'
            )

    blocks.append('<div class="nores">No matches.</div>')

    if caps:
        blocks.append(
            '<section class="block" id="%scaps"><h2 class="sec">capability specs</h2></section>'
            % pre
        )
        blocks.append(
            "".join(
                render_capability(c, inline, out_refs[c.cid], in_refs[c.cid], anchors, A)
                for c in caps
            )
        )
    if changes:
        blocks.append(
            f'<section class="block chg-lead" id="{pre}changes">'
            '<h2 class="sec">change proposals</h2></section>'
        )
        blocks.append("".join(render_change(c, inline, A, gmap, site) for c in changes))

    blocks.append(render_provenance(project))

    numbers, gauges = render_stats(project, len(edges))
    main = (
        f'<header class="mast" id="{pre}top">'
        f'<p class="sub">openspec &middot; {E(_kind_of(project))}</p>'
        f"<h1>{E(project.name)}<i>/openspec</i></h1>"
        f'<div class="scale" aria-hidden="true">{"<i></i>" * 46}</div>'
        f'<div class="stats">{numbers}</div>{gauges}'
        "</header>" + "".join(blocks)
    )

    # -- rail ---------------------------------------------------------------
    over = [f'<a href="#{pre}ctx"><span class="nm">context</span></a>']
    if delivery:
        over.append(
            f'<a href="#{pre}dlv"><span class="nm">'
            f"{E(str(delivery.get('title') or 'build and delivery').lower())}</span></a>"
        )
    if caps and edges:
        over.append(f'<a href="#{pre}map"><span class="nm">capability references</span></a>')
    if caps:
        over.append(
            f'<a href="#{pre}idx"><span class="nm">capabilities</span>'
            f'<span class="ct">{len(caps)}</span></a>'
        )
    if changes:
        over.append(
            f'<a href="#{pre}cidx"><span class="nm">changes</span>'
            f'<span class="ct">{len(changes)}</span></a>'
        )
        if flow:  # svg_flow returns "" when there is nothing to draw
            over.append(
                f'<a href="#{pre}flow"><span class="nm">changes and capabilities</span></a>'
            )
    rail = [f"<h2>overview</h2><nav>{''.join(over)}</nav>"]
    if caps:
        rail.append(
            "<h2>capabilities</h2><nav>"
            + "".join(
                f'<a href="#{A.of(c)}"><span class="nm">{E(c.cid)}</span>'
                f'<span class="ct">{len(c.requirements)}&#8202;/&#8202;{c.scenario_count}</span>'
                "</a>"
                for c in caps
            )
            + "</nav>"
        )
    if changes:
        rail.append(
            '<h2>changes</h2><nav class="cnav">'
            + "".join(
                f'<a href="#{A.of(c)}"><span class="nm">{E(c.cid)}</span>'
                f"{delta_ledger(change_counts(c))}"
                f'<span class="ct">{c.task_done_count}&#8202;/&#8202;{c.task_count}</span></a>'
                for c in changes
            )
            + "</nav>"
        )
    rail.append(f'<div class="railfoot">{E(str(project.root))}<br>generated {stamp}</div>')
    return "".join(rail), main


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

#: The format these documents are written in, and the CLI that maintains them.
OPENSPEC_URL = "https://github.com/Fission-AI/OpenSpec"

#: The one piece of vocabulary this otherwise generic tool special-cases: a
#: GitHub organisation whose bare slug is not how the organisation writes its
#: own name. Any org not listed renders as its own segment, unchanged.
ORG_LABEL = {"ROCm": "AMD ROCm"}


def render_provenance(project: Project) -> str:
    """Where these specs came from: the code, the format, the owner.

    A provenance line, not a banner. Every part is derived -- from the corpus's
    own ``source:`` key or from the git checkout around it -- so a corpus that
    answers neither simply names the format and stops.
    """
    rows: List[str] = []

    def item(label: str, value: str, href: str = "") -> None:
        tag = f'<a class="pv" href="{E(href)}">' if href else '<span class="pv">'
        rows.append(
            f'<span class="pi"><span class="pl">{E(label)}</span>'
            f"{tag}{E(value)}</{'a' if href else 'span'}></span>"
        )

    url = getattr(project, "source", "") or ""
    if url:
        item("source", re.sub(r"^https?://", "", url), url)
    item("format", "OpenSpec", OPENSPEC_URL)
    org = source_org(url)
    if org:
        item("maintained by", ORG_LABEL.get(org, org))
    return f'<footer class="prov">{"".join(rows)}</footer>'


def _kind_of(project: Project) -> str:
    """The openspec subdirectories this corpus actually has."""
    if project.capabilities and project.changes:
        return "specs and changes"
    if project.changes:
        return "changes"
    return "specs"


def delta_key() -> str:
    return "".join(
        f'<span class="dl {k.lower()}"><b>{SIGIL[k]}</b>{k.lower()}</span>'
        for k in ("ADDED", "MODIFIED", "REMOVED")
    )


def render_stats(project: Project, n_edges: int) -> Tuple[str, str]:
    """``(headline numbers, gauge strip)``.

    Every number is one plain cell. Anything that needs width -- the delta
    ledger, the task tape -- goes in the strip underneath, never inside a cell.
    """
    caps, changes = project.capabilities, project.changes
    cells: List[str] = []
    gauges: List[str] = []

    def num(value: int, label: str, cls: str = "") -> None:
        cells.append(f'<div class="{cls}"><b>{value}</b><span>{label}</span></div>')

    def gauge(label: str, body: str) -> None:
        gauges.append(f'<div class="gauge"><span class="glab">{label}</span>{body}</div>')

    if caps:
        num(len(caps), "capabilities", "c1")
        num(project.requirement_count, "requirements")
        num(project.scenario_count, "scenarios")
        if n_edges:
            num(n_edges, "references")
    if changes:
        counts: Dict[str, int] = {}
        n_delta_req = n_delta_scn = 0
        for chg in changes:
            for k, v in change_counts(chg).items():
                counts[k] = counts.get(k, 0) + v
            for cap in chg.deltas:
                n_delta_req += len(cap.requirements)
                n_delta_scn += cap.scenario_count
        num(len(changes), "changes")
        num(sum(len(c.deltas) for c in changes), "delta specs")
        num(n_delta_req, "delta requirements")
        if not caps:
            num(n_delta_scn, "scenarios")
        if counts:
            gauge("delta kinds", delta_ledger(counts))
        done = sum(c.task_done_count for c in changes)
        total = sum(c.task_count for c in changes)
        if total:
            # the site's grouping is by change, not by phase: phase names repeat
            # across changes and mean different things in each
            gauge(
                "tasks",
                tape(
                    [(c.cid, [t.done for p in c.phases for t in p.tasks]) for c in changes],
                    done,
                    total,
                    "wide",
                ),
            )
    strip = f'<div class="gauges">{"".join(gauges)}</div>' if gauges else ""
    return "".join(cells), strip


def strip_svg(per_req: Sequence[int]) -> str:
    """A sparkline of scenarios per requirement: how evenly a spec is covered."""
    if not per_req:
        return ""
    w, h, gap = 128, 20, 1.6
    top = max(per_req + [1])
    bw = max(1.4, min(7.0, (w + gap) / max(len(per_req), 1) - gap))
    bars = []
    for i, n in enumerate(per_req):
        x = i * (bw + gap)
        if x > w:
            break
        bh = max(2.5, h * (n / top))
        bars.append(
            f'<rect x="{x:.1f}" y="{h - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="1"/>'
        )
    return (
        f'<svg class="strip" viewBox="0 0 {w} {h}" aria-hidden="true" '
        f'preserveAspectRatio="xMinYMax meet">{"".join(bars)}</svg>'
    )


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------


def _site_of(arg: object) -> Site:
    """Accept a Site, or a bare Project while the CLI is still catching up."""
    if isinstance(arg, Project):
        return Site([arg])
    return arg  # type: ignore[return-value]


def render(site: Site) -> str:
    site = _site_of(site)
    projects = list(site.projects)
    used: Set[str] = set()
    for p in projects:  # load_project leaves slug empty; make one that is unique
        base = p.slug or slugify(p.name) or "project"
        slug, n = base, 2
        while slug in used:
            slug, n = f"{base}-{n}", n + 1
        used.add(slug)
        p.slug = slug

    multi = len(projects) > 1
    A = Anchors()
    for p in projects:
        _prime_anchors(p, A, multi)
    gmap = _global_anchors(projects, A)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    panes, rails = [], []
    for p in projects:
        rail, main = render_project(p, site, A, stamp, gmap)
        panes.append(f'<div class="proj" data-p="{E(p.slug)}">{main}</div>')
        rails.append(f'<div class="railproj" data-p="{E(p.slug)}">{rail}</div>')

    switch = ""
    if multi:
        switch = (
            '<div class="switch" role="tablist" aria-label="Project">'
            + "".join(
                f'<button type="button" role="tab" data-go="{E(p.slug)}">'
                f'<span class="pn">{E(p.name)}</span>'
                f'<span class="pc">{_switch_count(p)}</span></button>'
                for p in projects
            )
            + "</div>"
        )

    title = projects[0].name if not multi else " + ".join(p.name for p in projects)
    home = f"#{projects[0].slug}--top" if multi else "#top"

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{E(title)} openspec</title>
<link rel="icon" href="{favicon()}">
<style>{CSS}</style>
</head><body{' class="multi"' if multi else ""}>
<div class="shell">
<aside class="rail">
  <a class="brand" href="{home}">{logo()}<span class="txt"><b>openspec</b><span>viewer</span></span></a>
  {switch}
  <div class="search">
    <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/>
      <path d="M20 20l-4-4"/></svg>
    <input id="q" type="search" placeholder="search" aria-label="Search this page"
      autocomplete="off" spellcheck="false"><kbd>/</kbd>
  </div>
  <div class="hits" id="hits" role="status" aria-live="polite"></div>
  <details open>
  <summary>contents</summary>
  {"".join(rails)}
  </details>
  <div class="railtail"><button class="tbtn" id="theme" type="button">light / dark</button></div>
</aside>
<main>{"".join(panes)}</main>
</div>
<script>{JS}</script>
</body></html>
"""


def _switch_count(p: Project) -> str:
    if p.capabilities and p.changes:
        return f"{len(p.capabilities)} cap &middot; {len(p.changes)} chg"
    if p.changes:
        return f"{len(p.changes)} change{'s' if len(p.changes) != 1 else ''}"
    return f"{len(p.capabilities)} capabilities"


__all__ = ["render", "Anchors"]
