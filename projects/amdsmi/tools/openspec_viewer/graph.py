# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Graph layout and SVG emission, computed here rather than by JS.

Three diagrams live here.  Two share one layout engine:

``svg_graph``
    the capability *reference* schematic - who relies on whom.
``svg_flow``
    the *change flow* - which proposal creates, edits or deletes which
    capability, and which existing capabilities feed into it.

The engine is Sugiyama-shaped: break cycles, assign layers, optionally
compress the layer count, order within layers to reduce crossings, place,
then route orthogonally.

The third does not use it:

``svg_delivery``
    a left-to-right staged flow read from a hand-authored document.  Its
    columns are *given* - the author assigned every node to a stage - so
    there is nothing to layer, and forcing it through the vertical engine
    would throw away the one piece of structure that matters.

Everything is hand-emitted SVG; no marker elements, no JS is needed to draw.

A dense cross-reference graph has no readable global drawing: when most of
the corpus sits in one strongly connected component, every node is a hub and
no layout untangles it.  ``svg_graph`` therefore answers a smaller question -
*what does this one capability rely on, and who relies on it* - by shipping a
calm default state plus the attributes a focus mode needs.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import graph acyclic
    from .model import Capability, Change, Project

# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

NODE_H = 34
LAYER_GAP = 76
X_GAP = 26
PAD = 16
CHAR_W = 7.05
MIN_NODE_W = 92.0
PAD_W = 26.0
MAX_LABEL = 38
#: wrap an id onto a second line past this many characters
WRAP_AT = 15
LINE_H = 13.0
#: approximate advance width of the small caption font
CAPTION_CH = 7.0
#: half-width of an arrowhead, which lands on the centre of a node's top edge
HEAD_W = 5.0
#: approximate advance width of the small caption font
#: how far below a row the first same-layer ("flat") wire runs
FLAT_BASE = 14.0
FLAT_STEP = 9.0
#: vertical spacing between horizontal wire lanes inside one layer gap
LANE_STEP = 9.0
#: width divided by height we aim for when compressing the layer count
ASPECT = 1.55
#: width the diagram has to stay inside: the content column at the narrowest
#: window the page is designed for.  Overflow is not a cosmetic problem - the
#: wrapper scrolls, the scrollbar is easy to miss, and a reader concludes the
#: part they can see is the whole graph.
FIT_W = 916.0


@dataclass
class GNode:
    key: str
    label: str
    w: float
    x: float = 0.0
    layer: int = 0
    dummy: bool = False
    lines: Sequence[str] = ()


@dataclass
class GEdge:
    """One drawn wire, in its *true* direction: ``a`` references/feeds ``b``."""

    a: str
    b: str
    pts: List[Tuple[float, float]] = field(default_factory=list)
    #: the reverse edge exists too; drawn once with a head at either end
    bidir: bool = False
    #: both ends sit on the same layer, so the wire dips below the row
    flat: bool = False


class Graph:
    """Layered layout: break cycles, layer, compress, order, place, route."""

    def __init__(
        self,
        ids: Sequence[str],
        edges: Iterable[Tuple[str, str]],
        *,
        compress: bool = True,
        labels: Optional[Mapping[str, str]] = None,
        extra_w: Optional[Mapping[str, float]] = None,
        groups: Optional[Mapping[str, str]] = None,
    ):
        self.ids = list(dict.fromkeys(ids))
        known = set(self.ids)
        self.edges = sorted({(u, v) for u, v in edges if u != v and u in known and v in known})
        self.labels: Dict[str, str] = dict(labels) if labels else _short_labels(self.ids)
        self.labels = {i: _clip(self.labels.get(i) or i) for i in self.ids}
        self._extra_w = dict(extra_w or {})
        self.groups: Dict[str, str] = {k: v for k, v in (groups or {}).items() if k in known and v}
        self.group_ix: Dict[str, int] = {}
        for cid in self.ids:  # index by first appearance, so it is stable
            g = self.groups.get(cid)
            if g and g not in self.group_ix:
                self.group_ix[g] = len(self.group_ix)
        self.nodes: Dict[str, GNode] = {}
        self.layers: List[List[str]] = []
        self.chains: Dict[Tuple[str, str], List[str]] = {}
        self._gap: List[float] = []
        self._rung: Dict[Tuple[str, str], int] = {}
        self._lane_ix: Dict[Tuple[int, Tuple[str, str, int]], int] = {}
        self._flat_block: Dict[int, float] = {}
        self.nbr_up: Dict[str, List[str]] = {}
        self.nbr_dn: Dict[str, List[str]] = {}
        self.width = 0.0
        self.height = 0.0
        self.node_h = float(NODE_H)
        self.paths: List[GEdge] = []
        self.compress = compress
        if self.ids:
            self._build()

    # -- pipeline ----------------------------------------------------------
    def _build(self) -> None:
        arcs, bidir = self._merge_mutual()
        keep, back = self._break_cycles(arcs)
        dag = list(dict.fromkeys([(u, v) for u, v in keep] + [(v, u) for (u, v) in back]))
        lines = {cid: _wrap(self.labels[cid]) for cid in self.ids}
        widths = {
            cid: max(
                MIN_NODE_W,
                max(len(x) for x in lines[cid]) * CHAR_W + PAD_W + self._extra_w.get(cid, 0.0),
            )
            for cid in self.ids
        }
        self.node_h = NODE_H + (LINE_H if any(len(v) > 1 for v in lines.values()) else 0.0)
        self._lines = lines

        depth = self._layer(dag)
        linked = {k for e in dag for k in e}
        loose = [i for i in self.ids if i not in linked]

        best: Optional[Tuple[float, Dict[str, object]]] = None
        for cand in self._candidates(depth, linked, widths):
            cand = self._stack_loose(cand, linked, loose, widths)
            self.nodes = {
                k: GNode(k, self.labels[k], widths[k], layer=cand[k], lines=self._lines[k])
                for k in self.ids
            }
            self.paths = []
            nlayers = max(cand.values()) + 1
            spans = [(u, v) for u, v in dag if cand[u] != cand[v]]
            flats = [(u, v) for u, v in dag if cand[u] == cand[v]]
            self._order(spans, flats, cand, nlayers)
            self._place()
            self._plan_y(spans, flats, nlayers)
            self._route(spans, flats, set(back), bidir)
            # overflow is close to disqualifying.  A taller, more tangled
            # drawing that fits still shows every node; a tidier one that
            # does not fit hides some of them behind a scrollbar, and this
            # graph is a thicket either way - the reader is going to pick a
            # node and follow its two hops, not trace the whole picture.
            tangle = (
                self._crossings(self.layers, self.nbr_dn)
                + self._flat_span(self.layers, flats)
                + 2 * self._gmix(self.layers)
            )
            score = (
                25.0 * max(0.0, self.width - FIT_W) + self.width + 1.3 * self.height + 7.0 * tangle
            )
            if best is None or score < best[0]:
                best = (score, self._snapshot())
        if best is not None:
            self._restore(best[1])

    def _snapshot(self) -> "Dict[str, object]":
        return {
            "nodes": self.nodes,
            "layers": self.layers,
            "paths": self.paths,
            "chains": self.chains,
            "gap": self._gap,
            "w": self.width,
            "h": self.height,
        }

    def _restore(self, s: "Dict[str, object]") -> None:
        self.nodes = s["nodes"]  # type: ignore[assignment]
        self.layers = s["layers"]  # type: ignore[assignment]
        self.paths = s["paths"]  # type: ignore[assignment]
        self.chains = s["chains"]  # type: ignore[assignment]
        self._gap = s["gap"]  # type: ignore[assignment]
        self.width = s["w"]  # type: ignore[assignment]
        self.height = s["h"]  # type: ignore[assignment]

    def _candidates(
        self, depth: Dict[str, int], linked: Set[str], widths: Mapping[str, float]
    ) -> "List[Dict[str, int]]":
        """Layer assignments worth trying, best guess first."""
        if not self.compress:
            return [depth]
        return [self._rescale(depth, t) for t in self._targets(depth, linked, widths)]

    # -- edges -------------------------------------------------------------
    def _merge_mutual(self) -> Tuple[List[Tuple[str, str]], Set[Tuple[str, str]]]:
        """Collapse ``a->b`` plus ``b->a`` into one arc flagged bidirectional.

        Drawing a mutual reference as two near-parallel wires doubles the
        clutter and hides the very thing worth seeing, that the two rely on
        each other.  It also used to hand the same arc to the layout twice.
        """
        seen = set(self.edges)
        arcs: List[Tuple[str, str]] = []
        bidir: Set[Tuple[str, str]] = set()
        for u, v in self.edges:
            if (v, u) in seen:
                if u > v:
                    continue
                bidir.add((u, v))
            arcs.append((u, v))
        return arcs, bidir

    def _break_cycles(
        self, arcs: Sequence[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Greedy feedback-arc-set (Eades-Lin-Smyth) rather than DFS back-edges.

        DFS picks whichever arcs happen to close a cycle first, which depends
        on nothing but id order: the same blob of mutually referencing
        capabilities could come out pointing any which way.  Ranking by
        out-degree minus in-degree instead means the heavy referrers come out
        near the top and the things everything relies on sink to the bottom,
        so "the arrow points down at what you depend on" holds for most of
        the picture instead of half of it.  It also leaves fewer arcs to
        reverse.
        """
        alive = set(self.ids)
        out: Dict[str, Set[str]] = {i: set() for i in self.ids}
        inn: Dict[str, Set[str]] = {i: set() for i in self.ids}
        for u, v in arcs:
            out[u].add(v)
            inn[v].add(u)

        def drop(k: str) -> None:
            alive.discard(k)
            for w in out[k]:
                inn[w].discard(k)
            for w in inn[k]:
                out[w].discard(k)

        head: List[str] = []
        tail: List[str] = []
        while alive:
            moved = True
            while moved:
                moved = False
                for k in sorted(alive):
                    if not out[k]:
                        tail.append(k)
                        drop(k)
                        moved = True
                for k in sorted(alive):
                    if k in alive and not inn[k]:
                        head.append(k)
                        drop(k)
                        moved = True
            if alive:
                pick = max(sorted(alive), key=lambda k: len(out[k]) - len(inn[k]))
                head.append(pick)
                drop(pick)
        pos = {k: i for i, k in enumerate(head + tail[::-1])}
        keep = [e for e in arcs if pos[e[0]] < pos[e[1]]]
        back = sorted(e for e in arcs if pos[e[0]] > pos[e[1]])
        return keep, back

    def _layer(self, dag: Sequence[Tuple[str, str]]) -> Dict[str, int]:
        """Longest-path (as-soon-as-possible) layering over the acyclic arcs."""
        preds: Dict[str, List[str]] = {i: [] for i in self.ids}
        succ: Dict[str, List[str]] = {i: [] for i in self.ids}
        for u, v in dag:
            preds[v].append(u)
            succ[u].append(v)
        depth = {i: 0 for i in self.ids}
        indeg = {i: len(preds[i]) for i in self.ids}
        ready = [i for i in self.ids if not indeg[i]]
        seen = 0
        while ready:
            u = ready.pop()
            seen += 1
            for v in succ[u]:
                depth[v] = max(depth[v], depth[u] + 1)
                indeg[v] -= 1
                if not indeg[v]:
                    ready.append(v)
        if seen < len(self.ids):  # defensive: a cycle survived, flatten the rest
            for i in self.ids:
                depth.setdefault(i, 0)
        return depth

    def _targets(
        self, depth: Dict[str, int], linked: Set[str], widths: Mapping[str, float]
    ) -> List[int]:
        """Candidate row counts, best aspect first, never taller than natural.

        A densely cross-referencing spec set is usually one big strongly
        connected blob: every acyclic orientation of it has a long path, so
        plain longest-path layering yields one node per row and a page of
        wire.  Rescaling the depths monotonically keeps "u is drawn above v"
        wherever that was true and turns the rest into same-row wires, which
        is honest about there being no real ordering to show.
        """
        rows = [d for k, d in depth.items() if k in linked]
        n = len(linked)
        if not rows or n < 3:
            return [max(rows, default=0) + 1]
        nlayers = max(rows) + 1
        if nlayers < 3:
            return [nlayers]
        cell = sum(widths[k] for k in linked) / n + X_GAP
        row_h = self.node_h + LAYER_GAP
        ranked = sorted(
            range(1, nlayers + 1),
            key=lambda t: abs(_log((-(-n // t) * cell) / (t * row_h)) - _log(ASPECT)),
        )
        return ranked[:4]

    def _rescale(self, depth: Dict[str, int], target: int) -> Dict[str, int]:
        nlayers = max(depth.values()) + 1
        if target >= nlayers or nlayers < 2:
            return dict(depth)
        k = (target - 1) / (nlayers - 1)
        return {i: int(round(d * k)) for i, d in depth.items()}

    def _stack_loose(
        self,
        depth: Dict[str, int],
        linked: Set[str],
        loose: Sequence[str],
        widths: Mapping[str, float],
    ) -> Dict[str, int]:
        """Wrap edgeless nodes into rows of their own instead of one long line."""
        out = dict(depth)
        if not loose:
            return out
        base = (max((d for k, d in depth.items() if k in linked), default=-1)) + 1
        span = 0.0
        for row in range(base):
            span = max(span, sum(widths[k] + X_GAP for k in linked if depth[k] == row))
        span = span or FIT_W
        wide = sum(widths[k] + X_GAP for k in loose) / len(loose)
        per = max(1, min(int(span // wide) or 1, len(loose)))
        for i, k in enumerate(loose):
            out[k] = base + i // per
        return out

    # -- ordering ----------------------------------------------------------
    def _order(
        self,
        spans: Sequence[Tuple[str, str]],
        flats: Sequence[Tuple[str, str]],
        depth: Dict[str, int],
        nlayers: int,
    ) -> None:
        """Order each layer, inserting dummies so a long arc gets its own lane."""
        seed: List[List[str]] = [[] for _ in range(nlayers)]
        for cid in self.ids:
            seed[depth[cid]].append(cid)

        self.chains = {}
        for u, v in spans:
            chain: List[str] = []
            for step in range(1, depth[v] - depth[u]):
                key = f"~{u}>{v}#{step}"
                self.nodes[key] = GNode(key, "", 1.0, layer=depth[u] + step, dummy=True)
                seed[depth[u] + step].append(key)
                chain.append(key)
            self.chains[(u, v)] = chain

        nbr_up: Dict[str, List[str]] = {k: [] for k in self.nodes}
        nbr_dn: Dict[str, List[str]] = {k: [] for k in self.nodes}
        for (u, v), chain in self.chains.items():
            seq = [u] + chain + [v]
            for a, b in zip(seq, seq[1:]):
                nbr_dn[a].append(b)
                nbr_up[b].append(a)
        self.nbr_up, self.nbr_dn = nbr_up, nbr_dn

        deg = {k: len(nbr_up[k]) + len(nbr_dn[k]) for k in self.nodes}
        gix = self._gix
        starts = (
            seed,
            [la[::-1] for la in seed],
            [sorted(la, key=lambda k: (-deg[k], k)) for la in seed],
            [sorted(la, key=lambda k: (deg[k], k)) for la in seed],
            [sorted(la, key=lambda k: (gix(k), -deg[k], k)) for la in seed],
        )
        best: Optional[List[List[str]]] = None
        best_x = -1
        for start in starts:  # a few deterministic restarts; keep the tidiest
            layers = [list(la) for la in start]
            for sweep in range(8):
                down = sweep % 2 == 0
                rng = range(1, nlayers) if down else range(nlayers - 2, -1, -1)
                ref = nbr_up if down else nbr_dn
                for li in rng:
                    prev = layers[li - 1] if down else layers[li + 1]
                    pos = {k: i for i, k in enumerate(prev)}
                    cur = layers[li]
                    base = {k: i for i, k in enumerate(cur)}
                    layers[li] = sorted(
                        cur,
                        key=lambda k: (
                            _mean([pos[n] for n in ref[k] if n in pos], base[k]),
                            base[k],
                        ),
                    )
                self._transpose(layers, nbr_up, nbr_dn)
            self._polish(layers, nbr_up, nbr_dn, flats)
            score = (
                self._crossings(layers, nbr_dn)
                + self._flat_span(layers, flats)
                + 2 * self._gmix(layers)
            )
            if best is None or score < best_x:
                best, best_x = layers, score
        self.layers = best if best is not None else seed

    def _polish(
        self,
        layers: List[List[str]],
        nbr_up: Dict[str, List[str]],
        nbr_dn: Dict[str, List[str]],
        flats: Sequence[Tuple[str, str]],
    ) -> None:
        """Adjacent swaps against crossings *and* same-layer wire length.

        A same-layer wire dips into the gap under its row and runs the whole
        way across it, so it crosses every descending wire in between.
        Pulling mutually referencing peers next to each other is worth as
        much as removing a crossing, and it also makes the relation legible:
        a short hop between neighbours instead of a bar across the diagram.
        """
        mates: Dict[str, List[str]] = {}
        for u, v in flats:
            mates.setdefault(u, []).append(v)
            mates.setdefault(v, []).append(u)
        if not mates and not layers:
            return
        for _ in range(3):
            moved = False
            for li, layer in enumerate(layers):
                up = {k: i for i, k in enumerate(layers[li - 1])} if li else {}
                dn = {k: i for i, k in enumerate(layers[li + 1])} if li + 1 < len(layers) else {}
                for i in range(len(layer) - 1):
                    a, b = layer[i], layer[i + 1]
                    now = _cross(a, b, up, nbr_up) + _cross(a, b, dn, nbr_dn) + _span(layer, mates)
                    layer[i], layer[i + 1] = b, a
                    alt = _cross(b, a, up, nbr_up) + _cross(b, a, dn, nbr_dn) + _span(layer, mates)
                    if alt < now:
                        moved = True
                    else:
                        layer[i], layer[i + 1] = a, b
            if not moved:
                return

    def _gix(self, key: str) -> int:
        """Sort index of a node's group; ungrouped and dummies sort last."""
        return self.group_ix.get(self.groups.get(key, ""), len(self.group_ix))

    def _gmix(self, layers: Sequence[Sequence[str]]) -> int:
        """Adjacent pairs in a row from different groups.

        Grouping is authored data, not something inferred from the wires, so
        it is worth as much as a crossing: two capabilities in the same
        family sitting next to each other says something a reader can use,
        and the wire between them gets shorter as a bonus.
        """
        if not self.group_ix:
            return 0
        return sum(
            1
            for layer in layers
            for a, b in zip(layer, layer[1:])
            if not self.nodes[a].dummy and not self.nodes[b].dummy and self._gix(a) != self._gix(b)
        )

    @staticmethod
    def _flat_span(layers: Sequence[Sequence[str]], flats: Sequence[Tuple[str, str]]) -> int:
        """Wires a same-layer edge has to hop over: every slot it passes."""
        pos = {k: i for la in layers for i, k in enumerate(la)}
        return sum(max(0, abs(pos[u] - pos[v]) - 1) for u, v in flats if u in pos and v in pos)

    def _crossings(
        self, layers: Sequence[Sequence[str]], nbr_dn: Mapping[str, Sequence[str]]
    ) -> int:
        total = 0
        for li in range(len(layers) - 1):
            pos = {k: i for i, k in enumerate(layers[li + 1])}
            arcs = [
                (i, pos[n]) for i, k in enumerate(layers[li]) for n in nbr_dn.get(k, ()) if n in pos
            ]
            total += sum(
                1
                for i in range(len(arcs))
                for j in range(i + 1, len(arcs))
                if (arcs[i][0] - arcs[j][0]) * (arcs[i][1] - arcs[j][1]) < 0
            )
        return total

    def _transpose(
        self, layers: List[List[str]], nbr_up: Dict[str, List[str]], nbr_dn: Dict[str, List[str]]
    ) -> None:
        """Swap adjacent pairs while that removes crossings (Sugiyama phase 2b)."""
        improved = True
        guard = 0
        while improved and guard < 4:
            improved = False
            guard += 1
            for li, layer in enumerate(layers):
                up = {k: i for i, k in enumerate(layers[li - 1])} if li else {}
                dn = {k: i for i, k in enumerate(layers[li + 1])} if li + 1 < len(layers) else {}
                for i in range(len(layer) - 1):
                    a, b = layer[i], layer[i + 1]
                    if _cross(a, b, up, nbr_up) + _cross(a, b, dn, nbr_dn) > _cross(
                        b, a, up, nbr_up
                    ) + _cross(b, a, dn, nbr_dn):
                        layer[i], layer[i + 1] = b, a
                        improved = True

    # -- placement ---------------------------------------------------------
    def _place(self) -> None:
        layers = self.layers
        for layer in layers:
            x = 0.0
            for k in layer:
                self.nodes[k].x = x + self.nodes[k].w / 2
                x += self.nodes[k].w + X_GAP

        def recenter() -> float:
            spans = [
                (
                    self.nodes[la[0]].x - self.nodes[la[0]].w / 2,
                    self.nodes[la[-1]].x + self.nodes[la[-1]].w / 2,
                )
                for la in layers
                if la
            ]
            total = max((hi - lo for lo, hi in spans), default=0.0)
            for layer in layers:
                if not layer:
                    continue
                lo = self.nodes[layer[0]].x - self.nodes[layer[0]].w / 2
                hi = self.nodes[layer[-1]].x + self.nodes[layer[-1]].w / 2
                shift = (total - (hi - lo)) / 2 - lo + PAD
                for k in layer:
                    self.nodes[k].x += shift
            return total + PAD * 2

        self.width = recenter()
        for sweep in range(6):  # straighten: pull each node toward its neighbours
            down = sweep % 2 == 0
            rng = range(1, len(layers)) if down else range(len(layers) - 1, -1, -1)
            ref = self.nbr_up if down else self.nbr_dn
            for li in rng:
                layer = layers[li]
                want = {k: _mean([self.nodes[n].x for n in ref[k]], self.nodes[k].x) for k in layer}
                left = PAD
                for k in layer:
                    n = self.nodes[k]
                    n.x = max(want[k], left + n.w / 2)
                    left = n.x + n.w / 2 + X_GAP
                right = self.width - PAD
                for k in reversed(layer):
                    n = self.nodes[k]
                    n.x = min(n.x, right - n.w / 2)
                    right = n.x - n.w / 2 - X_GAP
        self.width = recenter()

    def _y(self, layer: int) -> float:
        return PAD + layer * self.node_h + sum(self._gap[:layer])

    # -- vertical plan -----------------------------------------------------
    def _plan_y(
        self, spans: Sequence[Tuple[str, str]], flats: Sequence[Tuple[str, str]], nlayers: int
    ) -> None:
        """Size each layer gap to the wires that have to cross it.

        A fixed gap forces every horizontal run in a busy band onto lanes a
        few pixels apart, and a thicket of near-parallel lines a few pixels
        apart is exactly the thing you cannot trace with your eye.  Gaps grow
        where the traffic is instead.
        """
        self._rung = self._pack_flat(flats)
        self._lane_ix = self._pack_spans(spans)
        nflat: Dict[int, int] = {}
        for (u, _), i in self._rung.items():
            nflat[self.nodes[u].layer] = max(nflat.get(self.nodes[u].layer, 0), i + 1)
        nlane: Dict[int, int] = {}
        for (gap, _), i in self._lane_ix.items():
            nlane[gap] = max(nlane.get(gap, 0), i + 1)

        self._flat_block = {
            g: (FLAT_BASE + (c - 1) * FLAT_STEP if c else 0.0) for g, c in nflat.items()
        }
        self._gap = []
        for g in range(nlayers - 1):
            below = self._flat_block.get(g, 0.0)
            need = (below + 16 if below else 18) + max(0, nlane.get(g, 0) - 1) * LANE_STEP + 18
            self._gap.append(max(float(LAYER_GAP), need))
        tail = self._flat_block.get(nlayers - 1, 0.0)
        self.height = PAD * 2 + nlayers * self.node_h + sum(self._gap) + max(0.0, tail + 9 - PAD)

    # -- routing -----------------------------------------------------------
    def _route(
        self,
        spans: Sequence[Tuple[str, str]],
        flats: Sequence[Tuple[str, str]],
        back: Set[Tuple[str, str]],
        bidir: Set[Tuple[str, str]],
    ) -> None:
        """Give every horizontal run its own lane inside the gap it crosses.

        Sending each dogleg down the middle of the gap made unrelated edges
        overlap exactly, so two wires read as one.  Runs are packed per gap:
        those whose x-intervals do not overlap share a lane, the rest stack.
        """
        lane = self._lane_y()

        for u, v in spans:
            stops = [u] + self.chains[(u, v)] + [v]
            pts: List[Tuple[float, float]] = []
            for i, k in enumerate(stops[:-1]):
                n, m = self.nodes[k], self.nodes[stops[i + 1]]
                bot = self._y(n.layer) + (0 if n.dummy else self.node_h)
                pts.append((n.x, bot))
                if abs(m.x - n.x) > 0.6:
                    y = lane[(u, v, i)]
                    pts.append((n.x, y))
                    pts.append((m.x, y))
                pts.append((m.x, self._y(m.layer)))
            flipped = (v, u) in back
            a, b = (v, u) if flipped else (u, v)
            two = (u, v) in bidir or (v, u) in bidir
            self.paths.append(GEdge(a, b, _dedupe(pts[::-1] if flipped else pts), bidir=two))

        for u, v in flats:
            n, m = self.nodes[u], self.nodes[v]
            y = self._y(n.layer) + self.node_h
            dy = FLAT_BASE + self._rung[(u, v)] * FLAT_STEP
            pts = [(n.x, y), (n.x, y + dy), (m.x, y + dy), (m.x, y)]
            flipped = (u, v) in back
            a, b = (v, u) if flipped else (u, v)
            two = (u, v) in bidir or (v, u) in bidir
            self.paths.append(GEdge(a, b, pts[::-1] if flipped else pts, bidir=two, flat=True))

    def _pack_flat(self, flats: Sequence[Tuple[str, str]]) -> Dict[Tuple[str, str], int]:
        """Rung index below the row for each same-layer wire, shortest first."""
        out: Dict[Tuple[str, str], int] = {}
        by_layer: Dict[int, List[Tuple[str, str]]] = {}
        for u, v in flats:
            by_layer.setdefault(self.nodes[u].layer, []).append((u, v))
        for group in by_layer.values():
            out.update(
                _lanes(
                    sorted(group, key=lambda e: abs(self.nodes[e[0]].x - self.nodes[e[1]].x)),
                    lambda e: _range(self.nodes[e[0]].x, self.nodes[e[1]].x),
                )
            )
        return out

    def _pack_spans(
        self, spans: Sequence[Tuple[str, str]]
    ) -> Dict[Tuple[int, Tuple[str, str, int]], int]:
        """Lane index for each horizontal run, keyed by the gap it sits in."""
        runs: Dict[int, List[Tuple[str, str, int]]] = {}
        self._extent: Dict[Tuple[str, str, int], Tuple[float, float]] = {}
        for u, v in spans:
            stops = [u] + self.chains[(u, v)] + [v]
            for i, k in enumerate(stops[:-1]):
                n, m = self.nodes[k], self.nodes[stops[i + 1]]
                if abs(m.x - n.x) > 0.6:
                    runs.setdefault(n.layer, []).append((u, v, i))
                    self._extent[(u, v, i)] = _range(n.x, m.x)
        out: Dict[Tuple[int, Tuple[str, str, int]], int] = {}
        for gap, group in runs.items():
            slots = _lanes(
                sorted(group, key=lambda r: self._extent[r][1] - self._extent[r][0]),
                lambda r: self._extent[r],
            )
            for r, i in slots.items():
                out[(gap, r)] = i
        return out

    def _lane_y(self) -> Dict[Tuple[str, str, int], float]:
        """Turn lane indices into absolute y, centred in the gap below the flats."""
        per_gap: Dict[int, int] = {}
        for (gap, _), i in self._lane_ix.items():
            per_gap[gap] = max(per_gap.get(gap, 0), i + 1)
        out: Dict[Tuple[str, str, int], float] = {}
        for (gap, run), i in self._lane_ix.items():
            below = self._flat_block.get(gap, 0.0)
            top = self._y(gap) + self.node_h + (below + 14 if below else 16)
            bottom = self._y(gap + 1) - 12
            count = per_gap[gap]
            base = max(top, (top + bottom) / 2 - (count - 1) * LANE_STEP / 2)
            out[run] = base + i * LANE_STEP
        return out


def _range(a: float, b: float) -> Tuple[float, float]:
    return (a, b) if a <= b else (b, a)


def _dedupe(pts: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for p in pts:
        if not out or abs(p[0] - out[-1][0]) > 0.4 or abs(p[1] - out[-1][1]) > 0.4:
            out.append(p)
    return out


def _lanes(items, extent, pad: float = 8.0):  # type: ignore[no-untyped-def]
    """Greedy interval packing: reuse a lane when the x-ranges do not overlap."""
    taken: List[List[Tuple[float, float]]] = []
    out = {}
    for it in items:
        lo, hi = extent(it)
        idx = len(taken)
        for i, used in enumerate(taken):
            if all(hi < a - pad or lo > b + pad for a, b in used):
                idx = i
                break
        if idx == len(taken):
            taken.append([])
        taken[idx].append((lo, hi))
        out[it] = idx
    return out


def _span(layer: Sequence[str], mates: Mapping[str, Sequence[str]]) -> int:
    """Total same-layer wire length, in slots, inside one layer."""
    pos = {k: i for i, k in enumerate(layer)}
    return sum(
        abs(pos[k] - pos[m])
        for k, ms in mates.items()
        if k in pos
        for m in ms
        if m in pos and m > k
    )


def _cross(a: str, b: str, pos: Mapping[str, int], nbr: Mapping[str, Sequence[str]]) -> int:
    """Crossings between the wires of ``a`` and ``b`` when drawn in that order."""
    pa = sorted(pos[n] for n in nbr.get(a, ()) if n in pos)
    pb = sorted(pos[n] for n in nbr.get(b, ()) if n in pos)
    return sum(1 for i in pa for j in pb if j < i)


def _mean(vals: Sequence[float], fallback: float) -> float:
    return sum(vals) / len(vals) if vals else fallback


def _log(x: float) -> float:
    n = 0.0
    while x > 2.0:
        x /= 2.0
        n += 1.0
    while x < 0.5:
        x *= 2.0
        n -= 1.0
    return n + (x - 1.0)  # good enough for comparing aspect ratios


_TOKEN = re.compile(r"[^-/]+[-/]?")


def _wrap(label: str, at: int = WRAP_AT) -> List[str]:
    """Break a long id onto two lines at the most balanced ``-`` or ``/``.

    A node sized for the whole of ``rocm-python-distribution`` on one line is
    195px wide; five of those overflow the column and the browser slices the
    label mid-word.  Two lines of thirteen characters is 111px and still
    perfectly readable at natural size.
    """
    if len(label) <= at:
        return [label]
    toks = _TOKEN.findall(label) or [label]
    if len(toks) < 2:
        return [label]
    best, cost = 1, None
    for i in range(1, len(toks)):
        a, b = "".join(toks[:i]), "".join(toks[i:])
        c = max(len(a), len(b))
        if cost is None or c < cost:
            best, cost = i, c
    return ["".join(toks[:best]), "".join(toks[best:])]


def _lines_svg(lines: Sequence[str], x: float, y: float, cls: str) -> str:
    """One ``<text>``; a second line becomes a ``<tspan>`` on the next baseline."""
    if len(lines) < 2:
        return (
            f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="middle">'
            f"{E(lines[0] if lines else '')}</text>"
        )
    return (
        f'<text class="{cls}" x="{x:.1f}" y="{y - LINE_H / 2:.1f}" text-anchor="middle">'
        f'{E(lines[0])}<tspan x="{x:.1f}" dy="{LINE_H:.0f}">{E(lines[1])}</tspan></text>'
    )


def _clip(label: str, limit: int = MAX_LABEL) -> str:
    """Middle-elide an over-long label; the full id stays in the tooltip."""
    if len(label) <= limit:
        return label
    head = limit // 2
    return label[:head].rstrip("-/") + "\u2026" + label[len(label) - (limit - head - 1) :]


def _short_labels(ids: Sequence[str]) -> Dict[str, str]:
    """Strip the shared ``<project>-`` / ``<project>/`` prefix from every id.

    Computed at runtime from the ids themselves, so it works for
    ``amdsmi-python-loader`` and for a nested delta id such as
    ``cuid/identifier-format``.  Never strips the final token, so a label is
    never empty.
    """
    ids = list(ids)
    if len(ids) < 2:
        return {i: i for i in ids}
    parts = [_TOKEN.findall(i) or [i] for i in ids]
    n = 0
    while all(len(p) > n + 1 and p[n] == parts[0][n] for p in parts):
        n += 1
    # back off if the shared part is doing all the work: ids that differ only
    # in a trailing digit would otherwise render as bare "0", "1", "2"
    while n and min(len("".join(p[n:])) for p in parts) < 3:
        n -= 1
    return {i: ("".join(p[n:]) or i) for i, p in zip(ids, parts)}


#: "Delivery, ten capabilities:" - a paragraph that names a family and then
#: lists it.  Deliberately narrow: this must not fire on ordinary prose.
_GROUP_HEAD = re.compile(r"^([A-Z][A-Za-z][A-Za-z ]{1,22}?),\s+[\w-]+\s+capabilit(?:y|ies)\s*:")


def capability_groups(context: str, ids: Sequence[str]) -> Dict[str, str]:
    """Read a capability grouping out of the project's authored context.

    The families in this corpus are not a guess and not something to infer
    from the wires - ``config.yaml`` states them, in the same paragraph that
    gives their sizes::

        Delivery, ten capabilities: amdsmi-build-configuration (...),
        amdsmi-install-layout (...), amdsmi-python-* (loader, wheel, ...)

        Behavior, five capabilities: amdsmi-c-api-abi (...), amdsmi-cli (...)

    So this reads them rather than hard-coding a list that would rot the
    moment a capability moved.  Ids may be named exactly or by an
    ``amdsmi-python-*`` prefix glob, and an exact mention always beats a
    glob - which is the whole reason ``amdsmi-python-api`` lands in Behavior
    even though ``amdsmi-python-*`` appears under Delivery.

    Returns ``{}`` for a corpus whose context says nothing of the kind, and
    ``{}`` rather than a partial answer if only one family is found: one
    group is not a grouping.
    """
    ids = list(ids)
    if not context or not ids:
        return {}
    exact: Dict[str, str] = {}
    globbed: Dict[str, str] = {}
    for para in re.split(r"\n\s*\n", context):
        head = _GROUP_HEAD.match(" ".join(para.split()))
        if not head:
            continue
        name = head.group(1).strip()
        for token in re.findall(r"[a-z0-9][\w./-]*\*?", para):
            if token.endswith("*"):
                stem = token[:-1]
                for cid in ids:
                    if cid.startswith(stem) and cid != stem:
                        globbed.setdefault(cid, name)
            elif token in ids:
                exact[token] = name
    out = dict(globbed)
    out.update(exact)
    return out if len(set(out.values())) > 1 else {}


def graph_prefix(ids: Sequence[str]) -> str:
    labels = _short_labels(ids)
    for i in ids:
        if labels[i] != i:
            return i[: len(i) - len(labels[i])]
    return ""


# --------------------------------------------------------------------------
# shared svg helpers
# --------------------------------------------------------------------------

E = html.escape


def esc_attr(s: str) -> str:
    return html.escape(s, quote=True)


def _ortho(pts: Sequence[Tuple[float, float]], r: float = 8.0) -> str:
    """Emit an already right-angled polyline with the corners rounded off."""
    if len(pts) < 2:
        return ""
    d = [f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"]
    for i in range(1, len(pts) - 1):
        x0, y0 = pts[i - 1]
        cx, cy = pts[i]
        x1, y1 = pts[i + 1]
        rr = min(r, _hyp(x0 - cx, y0 - cy) / 2, _hyp(x1 - cx, y1 - cy) / 2)
        ax, ay = _toward(cx, cy, x0, y0, rr)
        bx, by = _toward(cx, cy, x1, y1, rr)
        d.append(f"L{ax:.1f},{ay:.1f}")
        d.append(f"Q{cx:.1f},{cy:.1f} {bx:.1f},{by:.1f}")
    d.append(f"L{pts[-1][0]:.1f},{pts[-1][1]:.1f}")
    return " ".join(d)


def _hyp(dx: float, dy: float) -> float:
    return (dx * dx + dy * dy) ** 0.5


def _toward(cx: float, cy: float, x: float, y: float, r: float) -> Tuple[float, float]:
    d = _hyp(x - cx, y - cy)
    return (cx, cy) if d < 1e-6 else (cx + (x - cx) * r / d, cy + (y - cy) * r / d)


def _path_d(e: GEdge) -> str:
    return _ortho(e.pts)


def _arrow(prev: Tuple[float, float], tip: Tuple[float, float], s: float = 4.6) -> str:
    """Filled triangle whose point sits exactly on the target's boundary."""
    x, y = tip
    dy = -s * 1.7 if tip[1] >= prev[1] else s * 1.7
    return f"M{x:.1f},{y:.1f} L{x - s:.1f},{y + dy:.1f} L{x + s:.1f},{y + dy:.1f} Z"


def _swallow(prev: Tuple[float, float], tip: Tuple[float, float], s: float = 5.0) -> str:
    """Notched arrowhead - reads as different from a plain triangle in mono."""
    x, y = tip
    dy = -s * 1.9 if tip[1] >= prev[1] else s * 1.9
    return (
        f"M{x:.1f},{y:.1f} L{x - s:.1f},{y + dy:.1f} L{x:.1f},{y + dy * 0.45:.1f} "
        f"L{x + s:.1f},{y + dy:.1f} Z"
    )


def _chevron(prev: Tuple[float, float], tip: Tuple[float, float], s: float = 4.4) -> str:
    """Open V head, for the weaker 'feeds into' relation."""
    x, y = tip
    dy = -s * 1.8 if tip[1] >= prev[1] else s * 1.8
    return f"M{x - s:.1f},{y + dy:.1f} L{x:.1f},{y:.1f} L{x + s:.1f},{y + dy:.1f}"


def _along(pts: Sequence[Tuple[float, float]], frac: float) -> Tuple[float, float]:
    """Point ``frac`` of the way along the polyline, by arc length."""
    segs = [_hyp(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts))]
    want = sum(segs) * frac
    run = 0.0
    for i, ln in enumerate(segs):
        if run + ln >= want and ln:
            t = (want - run) / ln
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        run += ln
    return pts[len(pts) // 2]


def _rrect(x: float, y: float, w: float, h: float, r: float = 5.0) -> str:
    return (
        f"M{x + r:.1f},{y:.1f} H{x + w - r:.1f} A{r},{r} 0 0 1 {x + w:.1f},{y + r:.1f} "
        f"V{y + h - r:.1f} A{r},{r} 0 0 1 {x + w - r:.1f},{y + h:.1f} H{x + r:.1f} "
        f"A{r},{r} 0 0 1 {x:.1f},{y + h - r:.1f} V{y + r:.1f} "
        f"A{r},{r} 0 0 1 {x + r:.1f},{y:.1f} Z"
    )


def _tag(x: float, y: float, w: float, h: float, cut: float = 9.0, r: float = 4.0) -> str:
    """Rounded box with a chamfered top-left corner: the 'change' shape."""
    return (
        f"M{x + cut:.1f},{y:.1f} H{x + w - r:.1f} A{r},{r} 0 0 1 {x + w:.1f},{y + r:.1f} "
        f"V{y + h - r:.1f} A{r},{r} 0 0 1 {x + w - r:.1f},{y + h:.1f} H{x + r:.1f} "
        f"A{r},{r} 0 0 1 {x:.1f},{y + h - r:.1f} V{y + cut:.1f} Z"
    )


# --------------------------------------------------------------------------
# diagram 1: capability reference schematic
# --------------------------------------------------------------------------


#: styling the schematic needs in order to have a *default* state at all.
#: Class-level, so the stylesheet overrides any of it with one more class.
MAP_STYLE = (
    "<style>"
    # the calm default: the wires are texture, the nodes are the content
    "svg.map.calm .mapedges{opacity:.42}"
    "svg.map.calm .wire{stroke-width:1}"
    # ...until a node is picked, when the two hops that matter come back to
    # full strength and everything else drops away
    "svg.map.calm.act .mapedges,svg.map.calm.held .mapedges{opacity:1}"
    "svg.map .ndegbg{fill:var(--panel);stroke:var(--line)}"
    "svg.map .ndegn{font-size:9px;fill:var(--faint);letter-spacing:0}"
    "svg.map .hub .ndegbg{stroke:var(--cap);stroke-opacity:.7}"
    "svg.map .hub .ndegn{fill:var(--cap)}"
    "svg.map a:hover .ndegbg,svg.map .self .ndegbg{fill:var(--cap-bg);stroke:var(--cap)}"
    "svg.map a:hover .ndegn,svg.map .self .ndegn{fill:var(--cap)}"
    "</style>"
)


def svg_graph(graph: Graph, anchors: Mapping[str, str], titles: Mapping[str, str]) -> str:
    """Emit the ``A references B`` schematic. Arrow points at the dependency.

    A spec set that cross-references itself densely has no readable global
    drawing.  This one is fifteen capabilities and forty-three edges with
    twelve of them in a single strongly connected component: every node is a
    hub, so every layout of it is a hairball and no amount of routing work
    changes that.  The diagram therefore answers a smaller question.

    The default state is deliberately quiet - thin wires at low contrast,
    so the picture reads as *this set is densely interconnected* rather than
    inviting anyone to trace a line.  What it does offer without a click is
    the degree of every node, so a reader can see which capabilities are
    load bearing before touching anything.  Picking a node (hover, keyboard
    focus, or a click to park it) is what makes it precise: its incoming and
    outgoing wires come to full strength and the rest fades, which turns the
    hairball into the two-hop diagram that was actually wanted.

    Emitted for that: ``data-cap``, ``data-in``, ``data-out``, ``data-deg``
    and ``data-group`` on every node, ``data-from`` / ``data-to`` /
    ``data-bi`` on every wire, and ``.mapedges`` / ``.mapnodes`` layers.
    """
    if not graph.ids:
        return ""
    w, h = round(graph.width), round(graph.height)
    indeg = {cid: 0 for cid in graph.ids}
    outdeg = {cid: 0 for cid in graph.ids}
    for u, v in graph.edges:
        outdeg[u] += 1
        indeg[v] += 1
    deg = {cid: indeg[cid] + outdeg[cid] for cid in graph.ids}
    # "hub" is relative to this corpus: no fixed threshold survives a set of
    # five capabilities and a set of fifty
    top = sorted(deg.values(), reverse=True)
    cut = top[max(0, len(top) // 4 - 1)] if top else 0
    out = [
        f'<svg class="map calm" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="Capability reference graph" '
        f'data-nodes="{len(graph.ids)}" data-edges="{len(graph.edges)}">',
        MAP_STYLE,
        '<g class="mapedges">',
    ]
    for e in graph.paths:
        attrs = f'data-from="{esc_attr(e.a)}" data-to="{esc_attr(e.b)}"'
        if e.bidir:
            attrs += ' data-bi="1"'
        cls = "edge flat" if e.flat else "edge"
        out.append(f'<g class="{cls}" {attrs}>')
        out.append(f'<path class="wire" d="{_path_d(e)}"/>')
        out.append(f'<path class="head" d="{_arrow(e.pts[-2], e.pts[-1])}"/>')
        if e.bidir and len(e.pts) > 1:
            out.append(f'<path class="head" d="{_arrow(e.pts[1], e.pts[0])}"/>')
        out.append("</g>")
    out.append('</g><g class="mapnodes">')
    for cid in graph.ids:
        n = graph.nodes[cid]
        x, y = n.x - n.w / 2, graph._y(n.layer)
        href = anchors.get(cid)
        group = graph.groups.get(cid, "")
        cls = "n" + (" hub" if deg[cid] >= cut and deg[cid] else "")
        if group:
            cls += f" g{graph.group_ix.get(group, 0)}"
        attrs = (
            f'class="{cls}" data-cap="{esc_attr(cid)}" data-in="{indeg[cid]}" '
            f'data-out="{outdeg[cid]}" data-deg="{deg[cid]}"'
        )
        if group:
            attrs += f' data-group="{esc_attr(group)}"'
        open_t = f'<a href="#{href}" {attrs}>' if href else f"<g {attrs}>"
        badge = ""
        if deg[cid]:
            # a notification-style badge on the corner rather than anything
            # inside the box: the box is sized for its label, and covering
            # half a capability id to report a number is a bad trade
            bx, by = x + n.w - 3.0, y + 2.0
            badge = (
                f'<circle class="ndegbg" cx="{bx:.1f}" cy="{by:.1f}" r="8"/>'
                f'<text class="ndegn" x="{bx:.1f}" y="{by + 3.2:.1f}" '
                f'text-anchor="middle">{deg[cid]}</text>'
            )
        out.append(
            f"{open_t}<title>{E(titles.get(cid, cid))}</title>"
            f'<rect class="nbox" x="{x:.1f}" y="{y:.1f}" width="{n.w:.1f}" '
            f'height="{graph.node_h:.0f}"/>'
            f'<rect class="ntab" x="{x:.1f}" y="{y:.1f}" width="{n.w:.1f}" height="2.5" rx="1.2"/>'
            + _lines_svg(n.lines, n.x, y + graph.node_h / 2 + 4.4, "nlabel")
            + badge
            + ("</a>" if href else "</g>")
        )
    out.append("</g>")
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------
# diagram 2: change flow
# --------------------------------------------------------------------------

#: how a delta edge is drawn, keyed by ``Requirement.delta``
_DELTA_STYLE = {
    "ADDED": ("added", "+", "", _arrow),
    "MODIFIED": ("modified", "~", "7 3.5", _arrow),
    "REMOVED": ("removed", "\u2212", "1.6 3.4", _swallow),
    "RENAMED": ("renamed", "\u00bb", "10 3 2 3", _arrow),
}
_FEED_STYLE = ("feeds", "", "9 3 2 3", _chevron)
#: precedence when one delta spec mixes several kinds
_DELTA_ORDER = ("REMOVED", "ADDED", "MODIFIED", "RENAMED")

#: bracketed ``[cap-id]`` or backticked ``\u0060cap-id\u0060`` reference
_REF = re.compile(r"\[([a-z0-9][\w./-]*)\](?!\()|`([a-z0-9][\w./-]*)`")


@dataclass
class _FNode:
    key: str
    cid: str
    kind: str  # "change" | "capability"
    owner: str  # project slug that owns it, "" when unknown
    exists: bool  # True when a baseline spec defines this capability
    cross: bool  # owned by a project other than the one being drawn
    label: str
    caption: str = ""
    owners: int = 0  # number of changes that create/edit/delete it


@dataclass
class _FEdge:
    a: str
    b: str
    rel: str  # "owns" | "feeds"
    kind: str  # ADDED / MODIFIED / REMOVED / RENAMED / FEEDS
    cross: bool
    detail: str


def _text_refs(text: str, known: Iterable[str]) -> Set[str]:
    """Ids mentioned in prose, bracketed or backticked, matched against ``known``.

    Also accepts the bare last path segment, so a change can point at
    ``sysfs-interface`` and reach ``cuid/sysfs-interface``.
    """
    index: Dict[str, str] = {}
    for cid in known:
        index[cid] = cid
        index.setdefault(cid.rsplit("/", 1)[-1], cid)
    hits: Set[str] = set()
    for m in _REF.finditer(text):
        tok = m.group(1) or m.group(2)
        if tok in index:
            hits.add(index[tok])
    return hits


def _cap_refs(cap: "Capability", known: Iterable[str]) -> Set[str]:
    got = getattr(cap, "refs", None)
    if got:
        return {r for r in got if r != cap.cid}
    return _text_refs(cap.raw, known) - {cap.cid}


def _change_refs(chg: "Change", known: Iterable[str]) -> Set[str]:
    got = getattr(chg, "refs", None)
    if got:
        return set(got)
    text = "\n".join("\n".join(body) for _, body in chg.docs)
    hits = _text_refs(text, known)
    for delta in chg.deltas:
        hits |= _cap_refs(delta, known)
    return hits


def _dominant(counts: Mapping[str, int]) -> str:
    if not counts:
        return "MODIFIED"
    top = max(counts.values())
    for kind in _DELTA_ORDER:
        if counts.get(kind, 0) == top:
            return kind
    return "MODIFIED"


def build_flow(
    projects: "Sequence[Project]", focus: "Optional[Project]" = None
) -> "Tuple[Dict[str, _FNode], List[_FEdge]]":
    """Resolve changes and capabilities into flow nodes and typed edges."""
    projects = list(projects)
    drawn = [focus] if focus is not None else projects
    changes = [(p, c) for p in drawn for c in p.changes]
    if not changes:
        return {}, []

    baseline: Dict[str, str] = {}  # capability id -> owning project slug
    alias: Dict[str, str] = {}  # last path segment -> capability id
    for p in projects:
        for cap in p.capabilities:
            baseline[cap.cid] = p.slug or p.name
            alias.setdefault(cap.cid.rsplit("/", 1)[-1], cap.cid)
    for _, chg in [(p, c) for p in projects for c in p.changes]:
        for delta in chg.deltas:
            alias.setdefault(delta.cid.rsplit("/", 1)[-1], delta.cid)

    def resolve(cid: str) -> str:
        if cid in baseline:
            return cid
        return alias.get(cid.rsplit("/", 1)[-1], cid)

    nodes: Dict[str, _FNode] = {}
    edges: List[_FEdge] = []

    def cap_node(cid: str, home: str) -> str:
        cid = resolve(cid)
        key = f"cap:{cid}"
        if key not in nodes:
            owner = baseline.get(cid, "")
            if not owner:
                for p in projects:
                    if any(cid == d.cid for c in p.changes for d in c.deltas):
                        owner = p.slug or p.name
                        break
            cross = bool(owner) and owner != home
            nodes[key] = _FNode(
                key=key,
                cid=cid,
                kind="capability",
                owner=owner,
                exists=cid in baseline,
                cross=cross,
                label=cid,
                caption=(f"in {owner}" if cross else ""),
            )
        elif nodes[key].cross and nodes[key].owner == home:
            nodes[key].cross = False
            nodes[key].caption = ""
        return key

    for proj, chg in changes:
        home = proj.slug or proj.name
        ckey = f"chg:{chg.cid}"
        nodes[ckey] = _FNode(
            key=ckey,
            cid=chg.cid,
            kind="change",
            owner=home,
            exists=True,
            cross=False,
            label=chg.cid,
        )
        owned: Set[str] = set()
        for delta in chg.deltas:
            tkey = cap_node(delta.cid, home)
            owned.add(nodes[tkey].cid)
            nodes[tkey].owners += 1
            counts = delta.delta_counts
            kind = _dominant(counts)
            detail = (
                ", ".join(f"{n} {k.lower()}" for k, n in sorted(counts.items())) or kind.lower()
            )
            edges.append(
                _FEdge(ckey, tkey, "owns", kind, nodes[tkey].owner not in ("", home), detail)
            )

        known = set(baseline) | {n.cid for n in nodes.values() if n.kind == "capability"}
        known |= {d.cid for _, c in changes for d in c.deltas}
        for ref in sorted(_change_refs(chg, known)):
            rid = resolve(ref)
            if rid in owned or rid == chg.cid:
                continue
            skey = cap_node(rid, home)
            edges.append(
                _FEdge(skey, ckey, "feeds", "FEEDS", nodes[skey].owner not in ("", home), "")
            )

    seen: Set[Tuple[str, str, str]] = set()
    uniq: List[_FEdge] = []
    for e in edges:
        sig = (e.a, e.b, e.rel)
        if sig not in seen:
            seen.add(sig)
            uniq.append(e)
    return nodes, uniq


def _flow_labels(nodes: Mapping[str, _FNode]) -> Dict[str, str]:
    """Shorten ids per group, so a change id is never clipped against a cap id."""
    out: Dict[str, str] = {}
    groups: Dict[Tuple[str, str], List[str]] = {}
    for key, n in nodes.items():
        groups.setdefault((n.kind, n.owner), []).append(key)
    for (keys,) in ((v,) for v in groups.values()):
        ids = [nodes[k].cid for k in keys]
        short = _short_labels(ids)
        for k, cid in zip(keys, ids):
            out[k] = short[cid]
    return out


def svg_flow(
    projects: "Sequence[Project]",
    anchors: Mapping[str, str],
    *,
    focus: "Optional[Project]" = None,
    label: str = "Change flow diagram",
) -> str:
    """Emit the change-flow diagram, or ``""`` when there is nothing to draw.

    ``projects`` is every project on the page (needed to tell whether a
    capability already exists and who owns it); ``focus`` narrows the drawn
    changes to one project.  ``anchors`` maps a change id or capability id to
    its html anchor; ``chg:<id>`` / ``cap:<id>`` keys win when present.
    """
    nodes, edges = build_flow(projects, focus)
    if not nodes or not edges:
        return ""

    labels = _flow_labels(nodes)
    extra = {}
    for k, n in nodes.items():
        pad = (16.0 if n.kind == "change" else 0.0) + (18.0 if n.owners > 1 else 0.0)
        if n.caption:
            # the caption is right-aligned above the node and the arrowhead
            # lands on its centre, so the box has to be wide enough for the
            # two never to sit on top of each other
            body = max(len(x) for x in _wrap(labels[k])) * CHAR_W + PAD_W
            clear = 2 * (len(n.caption) * CAPTION_CH + HEAD_W + 4) + 8
            pad = max(pad, clear - body)
        extra[k] = pad
    order = sorted(nodes, key=lambda k: (nodes[k].kind != "change", k))
    graph = Graph(order, [(e.a, e.b) for e in edges], compress=False, labels=labels, extra_w=extra)
    if not graph.paths and len(nodes) < 2:
        return ""

    by_pair = {(e.a, e.b): e for e in edges}
    w, h = round(graph.width), round(graph.height)
    out = [
        f'<svg class="flow" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="{esc_attr(label)}">'
    ]

    for ge in graph.paths:
        fe = by_pair.get((ge.a, ge.b)) or by_pair.get((ge.b, ge.a))
        if fe is None:
            continue
        cls, glyph, dash, head = (
            _FEED_STYLE
            if fe.rel == "feeds"
            else _DELTA_STYLE.get(fe.kind, _DELTA_STYLE["MODIFIED"])
        )
        src, dst = nodes[fe.a], nodes[fe.b]
        attrs = (
            f'data-from="{esc_attr(fe.a)}" data-to="{esc_attr(fe.b)}" '
            f'data-rel="{fe.rel}" data-kind="{esc_attr(fe.kind)}" '
            f'data-from-owner="{esc_attr(src.owner)}" data-to-owner="{esc_attr(dst.owner)}"'
        )
        if fe.cross:
            attrs += ' data-cross="1"'
        klass = f"fedge e-{cls}" + (" cross" if fe.cross else "") + (" flat" if ge.flat else "")
        out.append(f'<g class="{klass}" {attrs}>')
        title = (
            f"{src.cid} \u2192 {dst.cid}"
            if fe.rel == "feeds"
            else f"{src.cid} {fe.detail} in {dst.cid}"
        )
        out.append(f"<title>{E(title)}</title>")
        dash_a = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<path class="fwire" d="{_path_d(ge)}" fill="none"{dash_a}/>')
        out.append(f'<path class="fhead" d="{head(ge.pts[-2], ge.pts[-1])}"/>')
        if glyph:
            mx, my = _along(ge.pts, 0.5)
            out.append(f'<circle class="fglyphbg" cx="{mx:.1f}" cy="{my:.1f}" r="7.2"/>')
            out.append(
                f'<text class="fglyph" x="{mx:.1f}" y="{my + 3.9:.1f}" '
                f'text-anchor="middle">{E(glyph)}</text>'
            )
        if fe.cross:  # schematic "leaves this project" break mark, near the target
            bx, by = _along(ge.pts, 0.78 if glyph else 0.5)
            out.append(
                f'<path class="fbreak" d="M{bx - 5:.1f},{by + 4.5:.1f} L{bx + 1:.1f},{by - 4.5:.1f} '
                f'M{bx - 1:.1f},{by + 4.5:.1f} L{bx + 5:.1f},{by - 4.5:.1f}" fill="none"/>'
            )
        out.append("</g>")

    indeg: Dict[str, int] = {k: 0 for k in nodes}
    outdeg: Dict[str, int] = {k: 0 for k in nodes}
    for e in edges:
        outdeg[e.a] += 1
        indeg[e.b] += 1

    for key in graph.ids:
        n = nodes[key]
        gn = graph.nodes[key]
        x, y = gn.x - gn.w / 2, graph._y(gn.layer)
        if n.kind == "change":
            shape, cls, dash = _tag(x, y, gn.w, graph.node_h), "k-change", ""
        elif n.cross:
            shape, cls, dash = _rrect(x, y, gn.w, graph.node_h), "k-ghost", "3 3"
        elif n.exists:
            shape, cls, dash = _rrect(x, y, gn.w, graph.node_h), "k-spec", ""
        else:
            shape, cls, dash = _rrect(x, y, gn.w, graph.node_h), "k-new", "7 4"
        href = (
            anchors.get(f"{key.split(':', 1)[0]}:{n.cid}") or anchors.get(key) or anchors.get(n.cid)
        )
        attrs = (
            f'class="fnode {cls}" data-node="{esc_attr(key)}" data-kind="{n.kind}" '
            f'data-cid="{esc_attr(n.cid)}" data-owner="{esc_attr(n.owner)}" '
            f'data-in="{indeg[key]}" data-out="{outdeg[key]}"'
        )
        attrs += f' data-{"chg" if n.kind == "change" else "cap"}="{esc_attr(n.cid)}"'
        if n.cross:
            attrs += ' data-cross="1"'
        if n.exists and n.kind == "capability":
            attrs += ' data-exists="1"'
        if not n.exists and n.kind == "capability":
            attrs += ' data-new="1"'
        if n.owners > 1:
            attrs += f' data-owners="{n.owners}"'
        open_t = f'<a href="#{href}" {attrs}>' if href else f"<g {attrs}>"
        tip = _flow_title(n, indeg[key], outdeg[key])
        body = [f"{open_t}<title>{E(tip)}</title>"]
        dash_a = f' stroke-dasharray="{dash}"' if dash else ""
        if n.cross:
            body.append(
                f'<path class="fbox fshadow" d="'
                f'{_rrect(x + 4, y + 4, gn.w, graph.node_h)}" fill="none"/>'
            )
        body.append(f'<path class="fbox" d="{shape}"{dash_a}/>')
        body.append(
            f'<rect class="frail" x="{x:.1f}" y="{y + 5:.1f}" width="2.6" '
            f'height="{graph.node_h - 10:.0f}" rx="1.3"/>'
        )
        body.append(_lines_svg(gn.lines, gn.x, y + graph.node_h / 2 + 4.3, "flabel"))
        if n.caption:
            # textLength pins the caption to the width the node was sized
            # for, so it cannot grow into the arrowhead if the stylesheet
            # picks a bigger caption font than this module assumed.  It also
            # sits above the band the arrowhead occupies rather than merely
            # beside it, clamped so a layer-0 node keeps it on the canvas.
            body.append(
                f'<text class="fcaption" x="{x + gn.w - 2:.1f}" y="{max(y - 15, 10.0):.1f}" '
                f'text-anchor="end" textLength="{len(n.caption) * CAPTION_CH:.1f}" '
                f'lengthAdjust="spacingAndGlyphs">{E(n.caption)}</text>'
            )
        if n.owners > 1:  # convergence: several proposals land on this capability
            bx = x + gn.w - 15
            body.append(f'<circle class="fconv" cx="{bx:.1f}" cy="{y + 11:.1f}" r="8.4"/>')
            body.append(
                f'<text class="fconvn" x="{bx:.1f}" y="{y + 14.6:.1f}" '
                f'text-anchor="middle">{n.owners}</text>'
            )
        body.append("</a>" if href else "</g>")
        out.append("".join(body))
    out.append("</svg>")
    return "".join(out)


def _flow_title(n: _FNode, indeg: int, outdeg: int) -> str:
    if n.kind == "change":
        return f"proposal {n.cid} - touches {outdeg}, builds on {indeg}"
    what = "existing capability" if n.exists else "capability this proposal would create"
    if n.cross:
        what = f"{what}, owned by {n.owner}"
    tail = f" - changed by {n.owners}" if n.owners > 1 else ""
    return f"{n.cid} - {what}{tail}"


FLOW_LEGEND_ITEMS: Sequence[Tuple[str, str]] = (
    ("k-change", "proposal"),
    ("k-spec", "existing capability"),
    ("k-new", "capability the proposal would create"),
    ("k-ghost", "owned by another project"),
    ("e-added", "adds requirements"),
    ("e-modified", "modifies requirements"),
    ("e-removed", "removes requirements"),
    ("e-feeds", "existing capability it builds on"),
)


def flow_legend() -> str:
    """Legend markup mirroring ``.maplegend``; every class is stylable by C."""
    cells = "".join(
        f'<span class="{cls}"><i></i>{E(text)}</span>' for cls, text in FLOW_LEGEND_ITEMS
    )
    return f'<div class="flowlegend">{cells}</div>'


# --------------------------------------------------------------------------
# diagram 3: delivery flow - a hand-authored staged document
# --------------------------------------------------------------------------
#
# The reference schematic and the change flow are both *derived*: they show
# what the corpus says about itself.  Neither can answer "how does this thing
# reach a user", because a cross reference is not a production step - a
# capability that mentions another is not built from it.  That story has to
# be written down, and ``diagrams/delivery.json`` writes it down: ordered
# stages, a node per artifact carrying the capability that specifies it, and
# an edge per production step.
#
# The stage of every node is given, so there is no layering pass here and the
# vertical engine above is not used.  Two stages' worth of freedom are left:
# the order inside a column, and where the wires run.  One wrinkle - the
# author may put an edge *inside* a stage (three artifacts aggregate into one
# installable tree, all four of them "Artifact") - so a stage is split into
# as many sub-columns as its internal edges need, and the stage header spans
# them.  Dropping those edges instead would lose the aggregation, which is
# the most load-bearing step in the whole picture.

#: outer margin
D_PAD = 10.0
#: vertical space between two boxes in the same column
D_ROW_GAP = 20.0
#: narrowest gap between two columns; widened for labels and wire lanes
D_GAP_MIN = 28.0
#: narrowest a column may be, so a one-word column still reads as a column
D_COL_MIN = 97.0
#: lines a node label may use, and lines an edge label may use
D_LABEL_LINES = 2
D_ELAB_LINES = 3
#: node label line height and advance width.  Wider than ``CHAR_W`` on
#: purpose: a box whose text overflows by half a glyph reads as broken
#: rather than as tight, so the estimate errs generous here.
D_LINE = 15.0
D_CHAR = 7.4
#: small-text line height and advance width (edge labels, stage heads)
D_SMALL_LINE = 12.0
D_SMALL_CH = 6.2
#: height of the stage header strip above the first row
D_HEAD_H = 26.0
#: box padding
D_BOX_TOP = 8.0
D_BOX_BOT = 8.0
D_BOX_SIDE = 10.0
#: how far a wire runs straight out of a box before it may turn
D_STUB = 10.0
#: horizontal distance between two wires sharing one inter-column gap
D_LANE = 10.0
#: advance width of the capability chip along the bottom of a box (10px mono)
D_SPEC_CH = 6.1
#: room reserved at the bottom right of a box for the channel number
D_NUM_W = 12.0
#: the diagram tries to stay inside this, which is the content column at the
#: narrowest window the page is designed for.  Overflowing is not a cosmetic
#: problem here: the last stage is the whole point of the picture, and a
#: reader who cannot see it concludes the four columns they can see are the
#: whole story.
D_FIT = 916.0

#: one dash pattern per delivery channel, cycled.  Deliberately not one hue
#: per channel: this page spends its five hues on meanings already and a
#: sixth would read as "some kind of requirement" to anyone scanning fast.
D_LANE_DASH: Sequence[str] = ("", "7 3", "2 3", "10 3 2 3", "1 3")


@dataclass
class _DNode:
    key: str
    stage: int  # index into the declared stages
    col: int = 0  # flattened column, a stage may occupy several
    label: Sequence[str] = ()
    chip: Sequence[str] = ()
    raw_label: str = ""
    raw_note: str = ""
    spec: str = ""
    w: float = 0.0
    h: float = 0.0
    x: float = 0.0  # left edge
    y: float = 0.0  # top edge
    lane: int = -1  # delivery channel this node belongs to, -1 upstream of any

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass(eq=False)
class _DEdge:
    a: str
    b: str
    label: Sequence[str] = ()
    spec: str = ""
    lane: int = -1
    gap: int = 0  # inter-column gap the vertical run sits in
    pts: List[Tuple[float, float]] = field(default_factory=list)
    lx: float = 0.0  # label centre
    ly: float = 0.0
    lw: float = 0.0


#: a word too long for its column breaks at a path-ish separator.  Not at a
#: hyphen: ``site-packages`` and ``rocm-sdk-core`` are single names, and
#: splitting them reads as damage rather than as a line break.
_D_BREAK = re.compile(r"[^._/]*[._/]|[^._/]+")


def _wrap_words(text: str, width: int) -> List[str]:
    """Greedy word wrap that can also break inside an over-long token.

    ``libamd_smi.so.MAJOR`` is one word of nineteen characters, and a column
    sized for it is half again as wide as one sized for everything else in
    the same stage, so it breaks at its own punctuation instead.
    """
    out: List[str] = []
    cur = ""
    for word in text.split():
        for piece in [word] if len(word) <= width else (_D_BREAK.findall(word) or [word]):
            joint = "" if (not cur or cur.endswith(("-", "/", ".", "_"))) else " "
            cand = cur + joint + piece if cur else piece
            if cur and len(cand) > width:
                out.append(cur)
                cur = piece
            else:
                cur = cand
    if cur:
        out.append(cur)
    return out or [""]


def _wrap_fit(text: str, lines: int) -> List[str]:
    """Narrowest wrap of ``text`` that still fits within ``lines`` lines.

    Authored text is never clipped or elided in this diagram, so the only
    lever on a column's width is where the lines break.  Wrapping greedily
    at a fixed width sizes every column for whichever label happens to be
    longest; searching for the narrowest width that still fits gives the
    *demand* each label makes, and a column is then as wide as the largest
    demand in it.

    This is only the measurement.  Laying the label out again at the width
    the column actually got is what stops a short label being split for no
    reason - ``/opt/rocm`` demands five characters but is never drawn on two
    lines, because it fits on one in the column it ends up in.
    """
    text = " ".join(text.split())
    if not text:
        return [""]
    for width in range(2, len(text) + 1):
        got = _wrap_words(text, width)
        if len(got) <= lines:
            return got
    return [text]


@dataclass
class _Deliv:
    """The authored document, validated and laid out."""

    stages: List[Tuple[str, str]] = field(default_factory=list)
    nodes: Dict[str, _DNode] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)
    edges: List[_DEdge] = field(default_factory=list)
    cols: List[List[str]] = field(default_factory=list)
    #: flattened column index -> declared stage index
    col_stage: List[int] = field(default_factory=list)
    colw: List[float] = field(default_factory=list)
    colx: List[float] = field(default_factory=list)
    gaps: List[float] = field(default_factory=list)
    lanes: int = 0
    width: float = 0.0
    height: float = 0.0


def _deliv_read(doc: Mapping[str, object]) -> "Optional[_Deliv]":
    """Validate and normalise the authored document.

    Everything optional is optional: a node needs an id and a known stage, an
    edge needs two ends that exist.  Anything else is dropped rather than
    raised on - this file is content, and content drifts.
    """
    raw_stages, raw_nodes = doc.get("stages"), doc.get("nodes")
    if not isinstance(raw_stages, list) or not isinstance(raw_nodes, list):
        return None
    stages = [
        (str(s["id"]), str(s.get("label") or s["id"]))
        for s in raw_stages
        if isinstance(s, dict) and s.get("id")
    ]
    if not stages:
        return None
    sidx = {sid: i for i, (sid, _) in enumerate(stages)}

    nodes: Dict[str, _DNode] = {}
    order: List[str] = []
    for n in raw_nodes:
        if not isinstance(n, dict):
            continue
        key, stage = str(n.get("id") or ""), str(n.get("stage") or "")
        if not key or key in nodes or stage not in sidx:
            continue
        nodes[key] = _DNode(
            key=key,
            stage=sidx[stage],
            label=_wrap_fit(str(n.get("label") or key), D_LABEL_LINES),
            raw_label=" ".join(str(n.get("label") or key).split()),
            raw_note=str(n.get("note") or ""),
            spec=str(n.get("spec") or ""),
        )
        order.append(key)
    if not nodes:
        return None

    pairs: List[Tuple[str, str, Sequence[str], str]] = []
    seen: Set[Tuple[str, str]] = set()
    raw_edges = doc.get("edges")
    for e in raw_edges if isinstance(raw_edges, list) else []:
        if not isinstance(e, dict):
            continue
        a, b = str(e.get("from") or ""), str(e.get("to") or "")
        if a not in nodes or b not in nodes or a == b or (a, b) in seen:
            continue
        if nodes[b].stage < nodes[a].stage:  # a staged flow only runs forward
            continue
        seen.add((a, b))
        text = str(e.get("label") or "")
        pairs.append(
            (a, b, _wrap_fit(text, D_ELAB_LINES) if text else (), str(e.get("spec") or ""))
        )

    _deliv_columns(stages, nodes, order, [(a, b) for a, b, _, _ in pairs])
    d = _Deliv(stages=stages, nodes=nodes, order=order)
    d.edges = [
        _DEdge(a, b, label, spec) for a, b, label, spec in pairs if nodes[b].col > nodes[a].col
    ]
    ncols = max(n.col for n in nodes.values()) + 1
    d.cols = [[k for k in order if nodes[k].col == c] for c in range(ncols)]
    d.col_stage = [nodes[col[0]].stage for col in d.cols]
    return d


def _deliv_columns(
    stages: Sequence[Tuple[str, str]],
    nodes: Dict[str, _DNode],
    order: Sequence[str],
    pairs: Sequence[Tuple[str, str]],
) -> None:
    """Split each stage into sub-columns so its internal edges still run right.

    Longest-path over the edges that stay inside one stage.  A cycle inside a
    stage cannot be ranked at all, so the relaxation is bounded and whatever
    ranks it lands on are used as they are; the arcs that then point backwards
    are dropped by the caller, and the picture stays honest either way.

    Only the ranks that something actually sits on become columns, so a
    saturated cycle cannot leave an empty column behind.
    """
    rank = {k: 0 for k in order}
    for si in range(len(stages)):
        inside = [(a, b) for a, b in pairs if nodes[a].stage == si == nodes[b].stage]
        for _ in range(len(order)):
            moved = False
            for a, b in inside:
                if rank[b] < rank[a] + 1:
                    rank[b] = rank[a] + 1
                    moved = True
            if not moved:
                break
    used = sorted({(nodes[k].stage, rank[k]) for k in order})
    col = {cell: i for i, cell in enumerate(used)}
    for k in order:
        nodes[k].col = col[(nodes[k].stage, rank[k])]


def _deliv_lanes(d: _Deliv) -> None:
    """Number the delivery channels and mark everything downstream of one.

    A *channel* is a node in the stage before the last: the last stage is
    where things land, so the one before it is the last point at which a
    route is still a choice.  Everything reachable from a channel inherits
    its number, which is what makes "this one channel lands in two places"
    something you see rather than something you trace.
    """
    if len(d.stages) < 2:
        return
    roots = [k for k in d.order if d.nodes[k].stage == len(d.stages) - 2]
    if not roots:
        return
    succ: Dict[str, List[str]] = {}
    for e in d.edges:
        succ.setdefault(e.a, []).append(e.b)
    for lane, root in enumerate(roots):
        stack = [root]
        while stack:
            k = stack.pop()
            if d.nodes[k].lane >= 0:
                continue
            d.nodes[k].lane = lane
            stack.extend(succ.get(k, ()))
    for e in d.edges:
        e.lane = d.nodes[e.a].lane if d.nodes[e.a].lane >= 0 else d.nodes[e.b].lane
    d.lanes = len(roots)


def _deliv_order(d: _Deliv) -> None:
    """Barycentre sweeps inside each column; the columns themselves never move."""
    pred: Dict[str, List[str]] = {}
    succ: Dict[str, List[str]] = {}
    for e in d.edges:
        succ.setdefault(e.a, []).append(e.b)
        pred.setdefault(e.b, []).append(e.a)

    def sweep(i: int, ref: Mapping[str, List[str]], other: Sequence[str]) -> None:
        pos = {k: j for j, k in enumerate(other)}
        base = {k: j for j, k in enumerate(d.cols[i])}
        d.cols[i].sort(
            key=lambda k: (_mean([pos[p] for p in ref.get(k, ()) if p in pos], base[k]), base[k])
        )

    for _ in range(5):
        for i in range(1, len(d.cols)):
            sweep(i, pred, d.cols[i - 1])
        for i in range(len(d.cols) - 2, -1, -1):
            sweep(i, succ, d.cols[i + 1])


def _deliv_size(d: _Deliv) -> None:
    """Column widths from the labels alone; the notes are not drawn in the box.

    A note is a full sentence, and a sentence squeezed into a box eleven
    characters wide is three ellipses and no information.  They live in the
    tooltip and, spelled out in full, in the legend under the diagram.
    """
    d.colw = []
    for col in d.cols:
        chars = max((max(len(x) for x in d.nodes[k].label) for k in col), default=0)
        for k in col:  # re-wrap to the width the column actually got
            d.nodes[k].label = _wrap_words(d.nodes[k].raw_label, chars)
            d.nodes[k].chip = _wrap(_spec_chip(d.nodes[k].spec)) if d.nodes[k].spec else ()
        # the chip is a capability id, so it wraps at its own hyphens; the
        # column has to hold whichever of the two is wider
        numbered = any(d.nodes[k].lane >= 0 for k in col) and d.lanes > 1
        chip = max(
            (max(len(x) for x in d.nodes[k].chip) for k in col if d.nodes[k].chip), default=0
        )
        d.colw.append(
            max(
                D_COL_MIN,
                chars * D_CHAR + D_BOX_SIDE * 2,
                chip * D_SPEC_CH + D_BOX_SIDE * 2 + (D_NUM_W if numbered else 0.0),
            )
        )
    for ci, col in enumerate(d.cols):
        for k in col:
            n = d.nodes[k]
            n.w = d.colw[ci]
            numbered = n.lane >= 0 and d.lanes > 1
            n.h = (
                D_BOX_TOP
                + len(n.label) * D_LINE
                + (5.0 + len(n.chip) * D_SMALL_LINE if n.chip else 0.0)
                # the digit shares the chip's row, or gets one to itself: it
                # is painted last, so anything it lands on top of loses a
                # character, and a label is the one thing that must not
                + (D_SMALL_LINE if numbered and not n.chip else 0.0)
                + D_BOX_BOT
            )


def _deliv_y(d: _Deliv) -> None:
    """Stack each column, centre the columns, then pull nodes toward their mates."""
    tall = max(
        (sum(d.nodes[k].h for k in col) + D_ROW_GAP * max(0, len(col) - 1) for col in d.cols),
        default=0.0,
    )
    top = D_PAD + D_HEAD_H
    for col in d.cols:
        run = sum(d.nodes[k].h for k in col) + D_ROW_GAP * max(0, len(col) - 1)
        y = top + (tall - run) / 2
        for k in col:
            d.nodes[k].y = y
            y += d.nodes[k].h + D_ROW_GAP

    pred: Dict[str, List[str]] = {}
    succ: Dict[str, List[str]] = {}
    for e in d.edges:
        succ.setdefault(e.a, []).append(e.b)
        pred.setdefault(e.b, []).append(e.a)
    for n in range(8):  # straighten wires without reordering or overlapping
        down = n % 2 == 0
        rng = range(1, len(d.cols)) if down else range(len(d.cols) - 2, -1, -1)
        ref = pred if down else succ
        for ci in rng:
            col = d.cols[ci]
            want = [
                _mean([d.nodes[m].cy for m in ref.get(k, ())], d.nodes[k].cy) - d.nodes[k].h / 2
                for k in col
            ]
            floor = top
            for i, k in enumerate(col):
                d.nodes[k].y = max(want[i], floor)
                floor = d.nodes[k].y + d.nodes[k].h + D_ROW_GAP
            ceil = top + tall
            for i in range(len(col) - 1, -1, -1):
                node = d.nodes[col[i]]
                node.y = min(node.y, ceil - node.h)
                ceil = node.y - D_ROW_GAP
            floor = top
            for k in col:
                d.nodes[k].y = max(d.nodes[k].y, floor)
                floor = d.nodes[k].y + d.nodes[k].h + D_ROW_GAP
    d.height = top + tall + D_PAD


def _deliv_x(d: _Deliv, lanes_in: Mapping[int, int]) -> None:
    """Size every gap for the labels and wire lanes it carries, then place columns."""
    # the floor is what a gap needs for its edge labels: a label narrower
    # than its gap can always be centred clear of both columns, and one
    # wider than its gap cannot, so this is what keeps text off the boxes
    floor = [D_GAP_MIN] * max(0, len(d.cols) - 1)
    for e in d.edges:
        g = d.nodes[e.b].col - 1
        if e.label and 0 <= g < len(floor):
            floor[g] = max(floor[g], max(len(x) for x in e.label) * D_SMALL_CH + 14.0)
    d.gaps = list(floor)
    for g, count in lanes_in.items():  # room for the wires is negotiable
        if 0 <= g < len(d.gaps):
            d.gaps[g] = max(d.gaps[g], D_STUB * 2 + max(0, count - 1) * D_LANE + 8.0)
    # a diagram wider than the content column hides its last stage behind a
    # scrollbar nobody notices, so the slack in the gaps is spent on fitting
    # before it is spent on breathing room
    over = sum(d.colw) + sum(d.gaps) + D_PAD * 2 - D_FIT
    slack = sum(g - f for g, f in zip(d.gaps, floor))
    if over > 0 and slack > 0:
        take = min(1.0, over / slack)
        d.gaps = [g - (g - f) * take for g, f in zip(d.gaps, floor)]
    d.colx, x = [], D_PAD
    for ci, w in enumerate(d.colw):
        d.colx.append(x)
        x += w + (d.gaps[ci] if ci < len(d.gaps) else 0.0)
    d.width = x + D_PAD
    for ci, col in enumerate(d.cols):
        for k in col:
            d.nodes[k].x = d.colx[ci]


def _deliv_hits(d: _Deliv, y: float, xa: float, xb: float, skip: Tuple[str, str]) -> int:
    """Boxes a horizontal run at ``y`` would pass through."""
    lo, hi = _range(xa, xb)
    return sum(
        1
        for k, n in d.nodes.items()
        if k not in skip and n.y - 4 < y < n.y + n.h + 4 and n.x < hi - 3 and n.x + n.w > lo + 3
    )


def _deliv_pick(d: _Deliv) -> None:
    """Choose the gap each wire turns in: the one that runs over fewest boxes.

    Most edges span a single gap and have no choice.  The ones that skip a
    column - a Python module that becomes a wheel without passing through the
    installable tree - do, and picking badly draws a line straight through an
    unrelated box.
    """
    for e in d.edges:
        src, dst = d.nodes[e.a], d.nodes[e.b]
        lo, hi = src.col, dst.col - 1
        if hi <= lo or not d.colx:
            e.gap = lo
            continue
        best, score = lo, None
        for g in range(lo, hi + 1):
            vx = d.colx[g] + d.colw[g] + d.gaps[g] / 2
            cost = _deliv_hits(d, src.cy, src.x + src.w, vx, (e.a, e.b)) * 10
            cost += _deliv_hits(d, dst.cy, vx, dst.x, (e.a, e.b)) * 10
            cost += (hi - g) * 0.5  # all else equal, turn late: fan-in merges
            if score is None or cost < score:
                best, score = g, cost
        e.gap = best


def _deliv_route(d: _Deliv) -> Dict[int, int]:
    """One orthogonal dogleg per wire, each with its own vertical in the gap.

    Wires whose vertical spans do not overlap share a lane, so a fan-out of
    three does not become three lines a pixel apart.
    """
    by_gap: Dict[int, List[_DEdge]] = {}
    for e in d.edges:
        by_gap.setdefault(e.gap, []).append(e)
    counts: Dict[int, int] = {}
    for g, group in by_gap.items():
        span = {
            e: _range(d.nodes[e.a].cy, d.nodes[e.b].cy)
            for e in group
            if abs(d.nodes[e.a].cy - d.nodes[e.b].cy) > 1.0
        }
        idx = _lanes(sorted(span, key=lambda e: span[e][1] - span[e][0]), lambda e: span[e], 6.0)
        counts[g] = max(idx.values(), default=-1) + 1
        if not d.colx:
            continue
        room = d.gaps[g] - D_STUB * 2
        step = min(D_LANE, room / counts[g]) if counts[g] else 0.0
        left = d.colx[g] + d.colw[g] + D_STUB
        mid = left + max(0.0, (room - step * max(0, counts[g] - 1)) / 2)
        for e in group:
            src, dst = d.nodes[e.a], d.nodes[e.b]
            x0, y0, x1, y1 = src.x + src.w, src.cy, dst.x, dst.cy
            if e in idx:
                vx = min(max(mid + idx[e] * step, x0 + D_STUB), x1 - D_STUB)
                e.pts = [(x0, y0), (vx, y0), (vx, y1), (x1, y1)]
            else:
                e.pts = [(x0, y0), (x1, y1)]
            e.lw = (max(len(x) for x in e.label) if e.label else 0) * D_SMALL_CH
            lg = dst.col - 1  # the label rides the run into its target
            if 0 <= lg < len(d.gaps):
                # centre it in that gap, then clamp so its background box
                # cannot reach either column; the gap floor guarantees room
                lo = d.colx[lg] + d.colw[lg]
                half = e.lw / 2 + 6
                e.lx = min(max(lo + d.gaps[lg] / 2, lo + half), lo + d.gaps[lg] - half)
            else:
                e.lx = x1 - e.lw / 2 - 8
            e.ly = y1 - 6.0
    return counts


def _deliv_layout(doc: Mapping[str, object]) -> "Optional[_Deliv]":
    d = _deliv_read(doc)
    if d is None or not d.cols:
        return None
    _deliv_lanes(d)
    _deliv_order(d)
    _deliv_size(d)
    _deliv_y(d)
    counts: Dict[int, int] = {}
    for _ in range(3):  # gap width depends on the lanes, which depend on x
        _deliv_x(d, counts)
        _deliv_pick(d)
        counts = _deliv_route(d)
    return d


def _deliv_style(lanes: int) -> str:
    """Defaults, so the diagram is legible before anybody styles it.

    Class-level rather than id-level: the stylesheet beats any of this with
    one extra class.
    """
    css = [
        "svg.dlv .caplink{font-size:10px;fill:var(--cap);opacity:.85}",
        "svg.dlv .dsep{stroke:var(--line-2);fill:none}",
        "svg.dlv .elabelbg{fill:var(--panel)}",
        "svg.dlv .lanerail{stroke:var(--faint);stroke-width:2.4;fill:none;opacity:.7}",
        "svg.dlv .lanen{font-size:10px;fill:var(--faint)}",
        "svg.dlv .dedges{opacity:.85}",
        "svg.dlv.act .dedges{opacity:1}",
        "svg.dlv a:hover .lanerail,svg.dlv .self .lanerail{stroke:var(--cap);opacity:1}",
        "svg.dlv a:hover .lanen,svg.dlv .self .lanen{fill:var(--cap)}",
    ]
    for i in range(lanes):
        dash = D_LANE_DASH[i % len(D_LANE_DASH)]
        if dash:
            css.append(f"svg.dlv .ln{i} .wire{{stroke-dasharray:{dash}}}")
    return "<style>" + "".join(css) + "</style>"


def svg_delivery(
    doc: Mapping[str, object],
    anchors: Mapping[str, str],
    *,
    label: str = "Build and delivery diagram",
) -> str:
    """Emit the staged delivery flow, or ``""`` when there is nothing to draw.

    ``doc`` is ``diagrams/delivery.json`` already parsed; ``anchors`` maps a
    capability id to its html anchor, so every node can link to the
    capability that specifies it.  A corpus without the file never calls
    this, and a malformed one gets an empty string rather than a traceback.
    """
    d = _deliv_layout(doc)
    if d is None:
        return ""
    w, h = round(d.width), round(d.height)
    out = [
        f'<svg class="dlv" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="{esc_attr(label)}" data-lanes="{d.lanes}">',
        _deliv_style(d.lanes),
        '<g class="dstages">',
    ]
    for si, (sid, slabel) in enumerate(d.stages):
        cols = [c for c, s in enumerate(d.col_stage) if s == si]
        if not cols:
            continue
        x0 = d.colx[cols[0]] - 8
        x1 = d.colx[cols[-1]] + d.colw[cols[-1]] + 8
        band = (
            f'<rect class="stage" x="{x0:.1f}" y="{D_PAD + 14:.1f}" opacity=".55" '
            f'width="{x1 - x0:.1f}" height="{d.height - D_PAD * 2 - 14:.1f}" rx="8"/>'
            if si % 2
            else ""
        )
        out.append(
            f'<g class="dstagehead" data-stage="{esc_attr(sid)}">{band}'
            f'<text class="stagelab" x="{(x0 + x1) / 2:.1f}" y="{D_PAD + 8:.1f}" '
            f'text-anchor="middle">{E(slabel.upper())}</text></g>'
        )
    out.append('</g><g class="dedges">')
    for e in d.edges:
        cls = "edge" + (f" ln{e.lane}" if e.lane >= 0 else "")
        attrs = f'data-from="{esc_attr(e.a)}" data-to="{esc_attr(e.b)}"'
        if e.lane >= 0:
            attrs += f' data-lane="{e.lane}"'
        if e.spec:
            attrs += f' data-spec="{esc_attr(e.spec)}"'
        step = " ".join(e.label) if e.label else "\u2192"
        tip = f"{' '.join(d.nodes[e.a].label)} \u2192 {' '.join(d.nodes[e.b].label)}"
        if e.label:
            tip += f" ({step})"
        if e.spec:
            tip += f"\nspecified by {e.spec}"
        out.append(
            f'<g class="{cls}" {attrs}><title>{E(tip)}</title>'
            f'<path class="wire" d="{_ortho(e.pts, 7.0)}" fill="none"/>'
            f'<path class="head" d="{_arrow_r(e.pts[-1])}"/></g>'
        )
    out.append('</g><g class="dlabels">')
    for e in d.edges:
        if not e.label:
            continue
        top = e.ly - (len(e.label) - 1) * D_SMALL_LINE
        out.append(
            f'<rect class="elabelbg" x="{e.lx - e.lw / 2 - 4:.1f}" '
            f'y="{top - 10:.1f}" width="{e.lw + 8:.1f}" '
            f'height="{len(e.label) * D_SMALL_LINE + 4:.1f}" rx="3"/>'
        )
        for i, line in enumerate(e.label):
            out.append(
                f'<text class="elabel" x="{e.lx:.1f}" y="{top + i * D_SMALL_LINE:.1f}" '
                f'text-anchor="middle">{E(line)}</text>'
            )
    out.append('</g><g class="dnodes">')
    for key in d.order:
        out.append(_deliv_node(d, d.nodes[key], anchors))
    out.append("</g></svg>")
    return "".join(out)


def _deliv_node(d: _Deliv, n: _DNode, anchors: Mapping[str, str]) -> str:
    href = anchors.get(n.spec) if n.spec else None
    attrs = (
        f'class="dnode" data-node="{esc_attr(n.key)}" data-stage="{esc_attr(d.stages[n.stage][0])}"'
    )
    if n.spec:
        attrs += f' data-spec="{esc_attr(n.spec)}" data-cap="{esc_attr(n.spec)}"'
    if n.lane >= 0:
        attrs += f' data-lane="{n.lane}"'
    tip = " \u2014 ".join(x for x in (" ".join(n.label), n.raw_note) if x)
    if n.spec:
        tip += f"\nspecified by {n.spec}"
    body = [
        f'<a href="#{href}" {attrs}>' if href else f"<g {attrs}>",
        f"<title>{E(tip)}</title>",
        f'<rect class="nbox" x="{n.x:.1f}" y="{n.y:.1f}" width="{n.w:.1f}" height="{n.h:.1f}"/>',
    ]
    if n.spec:  # the same accent the schematic uses for "this box is a link"
        body.append(
            f'<rect class="ntab" x="{n.x:.1f}" y="{n.y:.1f}" '
            f'width="{n.w:.1f}" height="2.5" rx="1.2"/>'
        )
    if n.lane >= 0 and d.lanes > 1:
        # the wire into this box is dashed per channel; the box wears the
        # same dash down its left edge, so the family is legible standing
        # still.  It costs no width - it sits inside the box padding.
        dash = D_LANE_DASH[n.lane % len(D_LANE_DASH)]
        body.append(
            f'<path class="lanerail" d="M{n.x + 3:.1f},{n.y + 6:.1f} '
            f'V{n.y + n.h - 6:.1f}"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>"
        )
    y = n.y + D_BOX_TOP + 11.0
    for line in n.label:
        body.append(f'<text class="nlabel" x="{n.x + D_BOX_SIDE:.1f}" y="{y:.1f}">{E(line)}</text>')
        y += D_LINE
    if n.chip:
        y += 3.0
        body.append(
            f'<path class="dsep" d="M{n.x + D_BOX_SIDE:.1f},{y - 9:.1f} '
            f'H{n.x + n.w - D_BOX_SIDE:.1f}"/>'
        )
        for line in n.chip:
            body.append(
                f'<text class="caplink" x="{n.x + D_BOX_SIDE:.1f}" y="{y + 3:.1f}">{E(line)}</text>'
            )
            y += D_SMALL_LINE
    if n.lane >= 0 and d.lanes > 1:
        # bottom right, clear of the label: a digit over the label was the
        # one place in this diagram where text was actually being covered
        body.append(
            f'<text class="lanen" x="{n.x + n.w - 5:.1f}" '
            f'y="{n.y + n.h - D_BOX_BOT + 1:.1f}" text-anchor="end">{n.lane + 1}</text>'
        )
    body.append("</a>" if href else "</g>")
    return "".join(body)


def _spec_chip(spec: str) -> str:
    """Tail of a capability id: ``amdsmi-python-wheel`` -> ``python-wheel``."""
    tail = spec.rsplit("/", 1)[-1]
    head, _, rest = tail.partition("-")
    return rest if rest and len(head) > 2 else tail


def _arrow_r(tip: Tuple[float, float], s: float = 4.4) -> str:
    """Filled triangle pointing right; this diagram only ever arrives leftward."""
    x, y = tip
    return f"M{x:.1f},{y:.1f} L{x - s * 1.8:.1f},{y - s:.1f} L{x - s * 1.8:.1f},{y + s:.1f} Z"
