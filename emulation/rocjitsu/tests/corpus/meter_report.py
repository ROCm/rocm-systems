#!/usr/bin/env python3
"""Render one or more scored meter comparisons as a report a reader can act on.

``meter_score.py`` decides *what* is compared and emits it with ``--json``.
This script decides only how to present it, and adds nothing to the comparison
itself.  The scoring rule is that script's, restated here so a reader of the
report does not have to go and find it: the model's opinion is the ``event_us``
bracket of the emulated run, and the measurement it is scored against is
``kernel_us`` from the same corpus recorded on hardware.

The headline number is the **spread** of ``log2(model / measured)``.  Not the
bias: a bias is a median, so a model that is 4x fast on half the corpus and 4x
slow on the other half has a bias of zero and an unusable clock.  Not a
correlation either: it is nearly free on a corpus whose kernel times span four
orders of magnitude, and published GPU models reach 0.97 of it at 42-75% mean
absolute error.  Absolute time is what a guest reads off the modelled clock, so
absolute time is what the report scores.

Coverage comes before accuracy on purpose.  The scorer drops cases the emulated
run never produced and cases with no kernel time on one side, and dropping them
flatters whatever survives, so the count of what was not scored belongs beside
the accuracy claim rather than in a footnote under it.

    meter_report.py --scored des=des.json --scored leaky=leaky.json \
        --out reports/ --title "gfx950 timing" --format both

Standard library only, no timestamps: the same inputs render byte-identical
output, and ``--generated-at`` is the only way a date reaches the page.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# Bin widths tried in order, coarsening until the joint range of every run fits
# in MAX_BINS.  All are exact in binary, so a bin edge lands on 0.0 whatever
# the data, which is what lets the histogram show the 1.00x line as an edge
# rather than somewhere inside a bar.
BIN_LADDER = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MAX_BINS = 48

# Case ids are dotted, ``gemm.eager.float32.512x512x512``, so the tokens that
# name a dtype or an implementation can be recovered without asking rocm-meter.
# A token that matches neither contributes no facet rather than a wrong one.
DTYPE_TOKENS = frozenset(
    {
        "float64",
        "float32",
        "float16",
        "bfloat16",
        "float8",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "int8",
        "int32",
        "uint8",
    }
)
IMPL_TOKENS = frozenset({"eager", "compiled", "triton", "inductor"})

# Labels a --wall pair may use for the run with no timing model loaded.  The
# multiplier is meaningless without one, and guessing which of several labels
# is the baseline is worse than saying it was not given.
BASELINE_WALL_LABELS = ("none", "off", "baseline", "untimed", "no-timing", "notiming")

# Categorical slots 1-3 of the validated palette, which clear the all-pairs
# colour-vision floors in both modes.  A fourth model reuses slot 1: each
# histogram is a single-series chart titled with its model, so identity never
# rests on the hue.
SERIES_SLOTS = 3


# --------------------------------------------------------------------------
# Loading


@dataclass(frozen=True)
class Case:
    """One case scored in both views, straight out of the scorer's JSON."""

    case_id: str
    category: str
    measured_us: float
    model_us: float
    ratio: float
    abs_pct: float
    launches: int
    real_event_us: float | None
    model_event_us: float | None

    @property
    def log2_ratio(self) -> float:
        """How many octaves the model is away from the measurement."""
        return math.log2(self.ratio)


@dataclass
class Run:
    """One scored payload, under the label the caller gave it."""

    label: str
    path: str
    tolerance_pct: float
    cases: list[Case]
    unscorable: list[str]
    missing: list[str]
    by_id: dict[str, Case] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_id = {c.case_id: c for c in self.cases}

    @property
    def reference_cases(self) -> int:
        """Cases the reference run offered, scorable or not."""
        return len(self.cases) + len(self.unscorable) + len(self.missing)


def load_run(label: str, path: str) -> Run:
    """Read one ``meter_score.py --json`` payload."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cases: list[Case] = []
    for entry in payload.get("cases", ()):
        measured = float(entry["real_kernel_us"])
        model = float(entry["model_us"])
        if measured <= 0.0 or model <= 0.0:
            raise SystemExit(
                f"{path}: case {entry.get('case_id')!r} has a non-positive "
                f"time ({measured} vs {model}); the scorer should not emit it"
            )
        cases.append(
            Case(
                case_id=str(entry["case_id"]),
                category=str(entry.get("category") or "?"),
                measured_us=measured,
                model_us=model,
                ratio=float(entry.get("ratio", model / measured)),
                abs_pct=float(
                    entry.get("abs_pct", abs(model - measured) / measured * 100.0)
                ),
                launches=int(entry.get("launches") or 0),
                real_event_us=_optional_float(entry.get("real_event_us")),
                model_event_us=_optional_float(entry.get("model_event_us")),
            )
        )
    cases.sort(key=lambda c: c.case_id)
    return Run(
        label=label,
        path=path,
        tolerance_pct=float(payload.get("tolerance_pct", 20.0)),
        cases=cases,
        unscorable=sorted(str(x) for x in payload.get("unscorable", ())),
        missing=sorted(str(x) for x in payload.get("missing", ())),
    )


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


# --------------------------------------------------------------------------
# Statistics


def quantile(values: Sequence[float], q: float) -> float:
    """Linearly interpolated quantile, defined exactly as in meter_score.py.

    The two scripts report the same percentiles from the same data, so they are
    deliberately the same estimator rather than two defensible ones.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def stats(cases: Sequence[Case], tolerance_pct: float) -> dict[str, float]:
    """Every figure the accuracy and spread sections quote, for one group."""
    if not cases:
        return {"n": 0}
    ratios = [c.ratio for c in cases]
    logs = [c.log2_ratio for c in cases]
    abs_logs = [abs(v) for v in logs]
    pcts = [c.abs_pct for c in cases]
    p05, p95 = quantile(logs, 0.05), quantile(logs, 0.95)
    return {
        "n": len(cases),
        "median_ratio": statistics.median(ratios),
        "median_abs_pct": statistics.median(pcts),
        "p90_abs_pct": quantile(pcts, 0.90),
        "max_abs_pct": max(pcts),
        "within_tolerance": sum(1 for c in cases if c.abs_pct <= tolerance_pct),
        "within_2x": sum(1 for c in cases if 0.5 <= c.ratio <= 2.0),
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
        "median_abs_log2": statistics.median(abs_logs),
        "p90_abs_log2": quantile(abs_logs, 0.90),
        "bias_log2": statistics.median(logs),
        "p05_log2": p05,
        "p95_log2": p95,
        "spread_log2": p95 - p05,
        "stdev_log2": statistics.pstdev(logs) if len(logs) > 1 else 0.0,
    }


def tolerance_met(summary: dict[str, float], tolerance_pct: float) -> bool:
    """The criterion the report states: median absolute error inside the band.

    A per-case tolerance needs an aggregate rule before it can be met or
    missed, and the median is the one the scorer already headlines.
    """
    return bool(summary["n"]) and summary["median_abs_pct"] <= tolerance_pct


def by_category(cases: Sequence[Case]) -> dict[str, list[Case]]:
    grouped: dict[str, list[Case]] = {}
    for case in cases:
        grouped.setdefault(case.category, []).append(case)
    return {k: grouped[k] for k in sorted(grouped)}


def facets(case: Case) -> list[tuple[str, str]]:
    """Properties the worst cases might turn out to share.

    Deliberately coarse.  The point of the grouping is to say "the model is bad
    at small kernels" rather than to name twelve unrelated cases, so a facet
    that is unique per case would be noise.
    """
    found: list[tuple[str, str]] = [("category", case.category)]
    tokens = case.case_id.split(".")
    # The leading token repeats the category for some families; reporting both
    # would spend two lines of a short list saying one thing.
    if tokens and tokens[0] != case.category:
        found.append(("kernel family", tokens[0]))
    for token in tokens[1:]:
        if token in DTYPE_TOKENS:
            found.append(("dtype", token))
        elif token in IMPL_TOKENS:
            found.append(("implementation", token))
    found.append(
        ("direction", "model too slow" if case.ratio > 1.0 else "model too fast")
    )
    found.append(
        ("launches", "one kernel" if case.launches <= 1 else "several kernels")
    )
    if case.measured_us < 10.0:
        found.append(("measured time", "under 10us"))
    elif case.measured_us < 100.0:
        found.append(("measured time", "10-100us"))
    else:
        found.append(("measured time", "100us and over"))
    return found


def shared_traits(
    worst: Sequence[Case], everything: Sequence[Case]
) -> list[tuple[str, str, int, float]]:
    """Facet values over-represented among the worst cases.

    Returns (facet, value, count in the worst set, the facet's percentage
    share of the whole scored corpus).

    A facet needs to be over-represented, not merely common: if the worst
    cases are 90% GEMM on a corpus that is 90% GEMM, the corpus explains it
    and the model has said nothing.
    """
    if not worst or not everything:
        return []
    corpus_share: dict[tuple[str, str], float] = {}
    for case in everything:
        for key in facets(case):
            corpus_share[key] = corpus_share.get(key, 0.0) + 1.0
    for key in corpus_share:
        corpus_share[key] /= len(everything)
    worst_count: dict[tuple[str, str], int] = {}
    for case in worst:
        for key in facets(case):
            worst_count[key] = worst_count.get(key, 0) + 1
    # Lift rides along as a fifth element purely to order the result.
    out: list[tuple[str, str, int, float, float]] = []
    for (facet, value), count in worst_count.items():
        share = count / len(worst)
        base = corpus_share.get((facet, value), 0.0)
        lift = share / base if base > 0.0 else float("inf")
        if count < 2 or lift < 1.25:
            continue
        out.append((facet, value, count, base * 100.0, lift))
    out.sort(key=lambda row: (-row[2], -row[4], row[0], row[1]))
    return [row[:4] for row in out[:8]]


# --------------------------------------------------------------------------
# Histogram


def choose_bins(values: Iterable[float]) -> tuple[float, float, int]:
    """One set of log2 bin edges shared by every run, so the shapes compare.

    The range always contains 0.0 as an edge, so the 1.00x line is a boundary
    between bars rather than a point buried inside one.
    """
    finite = [v for v in values if math.isfinite(v)]
    low = min(finite + [0.0])
    high = max(finite + [0.0])
    for width in BIN_LADDER:
        lo_edge = math.floor(low / width) * width
        hi_edge = math.ceil(high / width) * width
        if hi_edge <= lo_edge:
            hi_edge = lo_edge + width
        count = int(round((hi_edge - lo_edge) / width))
        if count <= MAX_BINS:
            return width, lo_edge, count
    width = BIN_LADDER[-1]
    while (
        int(
            round(
                (math.ceil(high / width) * width - math.floor(low / width) * width)
                / width
            )
        )
        > MAX_BINS
    ):
        width *= 2.0
    lo_edge = math.floor(low / width) * width
    count = int(round((math.ceil(high / width) * width - lo_edge) / width))
    return width, lo_edge, max(1, count)


def bin_counts(
    cases: Sequence[Case], width: float, lo_edge: float, count: int
) -> list[int]:
    counts = [0] * count
    for case in cases:
        index = int(math.floor((case.log2_ratio - lo_edge) / width))
        counts[min(count - 1, max(0, index))] += 1
    return counts


def text_histogram(
    cases: Sequence[Case],
    width: float,
    lo_edge: float,
    count: int,
    tolerance_pct: float,
) -> str:
    """The histogram as characters, which is the form both outputs carry."""
    if not cases:
        return "no cases scored"
    counts = bin_counts(cases, width, lo_edge, count)
    peak = max(counts)
    lines = [f"log2(model / measured), {len(cases)} cases, " f"bin width {width:g}", ""]
    for index, value in enumerate(counts):
        lo = lo_edge + index * width
        hi = lo + width
        bar = "#" * int(round(32 * value / peak)) if peak and value else ""
        marker = "  <- 1.00x" if abs(lo) < 1e-9 else ""
        lines.append(f"  [{lo:6.2f},{hi:6.2f})  {value:5d}  " f"{bar}{marker}".rstrip())
    lines.append("")
    lines.append(
        f"  the {tolerance_pct:g}% tolerance band is "
        f"{tolerance_band(tolerance_pct)[0]:+.3f} .. "
        f"{tolerance_band(tolerance_pct)[1]:+.3f} on this axis"
    )
    return "\n".join(lines)


def tolerance_band(tolerance_pct: float) -> tuple[float, float]:
    """The tolerance expressed on the log2 axis the histogram plots."""
    high = math.log2(1.0 + tolerance_pct / 100.0)
    fraction = 1.0 - tolerance_pct / 100.0
    low = math.log2(fraction) if fraction > 0.0 else float("-inf")
    return low, high


def nice_step(peak: int) -> int:
    """A whole-number y tick that lands on at most four gridlines.

    Counts are integers, so a gridline labelled 1.5 would be a label for a
    number of cases that cannot occur.
    """
    exponent = 1
    while True:
        for mantissa in (1, 2, 5):
            step = mantissa * exponent
            if peak / step <= 4:
                return step
        exponent *= 10


def svg_histogram(
    cases: Sequence[Case],
    width: float,
    lo_edge: float,
    count: int,
    peak: int,
    tolerance_pct: float,
    label: str,
    series: str,
) -> str:
    """The same distribution as inline SVG, sharing both scales across runs."""
    w, h = 780.0, 250.0
    left, right, top, bottom = 48.0, 14.0, 26.0, 46.0
    plot_w = w - left - right
    plot_h = h - top - bottom
    base_y = top + plot_h
    span = width * count
    counts = bin_counts(cases, width, lo_edge, count)
    step = nice_step(max(peak, 1))
    y_max = max(step, int(math.ceil(max(peak, 1) / step)) * step)

    def x_of(value: float) -> float:
        return left + (value - lo_edge) / span * plot_w

    def y_of(value: int) -> float:
        return base_y - (value / y_max) * plot_h

    parts: list[str] = [
        # No xmlns: the element is inline in an HTML5 document, where the SVG
        # namespace is implied, and the page must reference no external host.
        f'<svg viewBox="0 0 {w:.0f} {h:.0f}" role="img" '
        f'aria-label="log2 ratio distribution for {esc_attr(label)}">',
        f"<title>{esc(label)}: distribution of log2(model / measured)</title>",
    ]

    band_lo, band_hi = tolerance_band(tolerance_pct)
    band_x0 = max(left, x_of(band_lo) if math.isfinite(band_lo) else left)
    band_x1 = min(left + plot_w, x_of(band_hi))
    if band_x1 > band_x0:
        parts.append(
            f'<rect x="{band_x0:.2f}" y="{top:.2f}" '
            f'width="{band_x1 - band_x0:.2f}" height="{plot_h:.2f}" '
            f'fill="var(--band)"/>'
        )

    for value in range(step, y_max + 1, step):
        y = y_of(value)
        parts.append(
            f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{left + plot_w:.2f}" '
            f'y2="{y:.2f}" stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8:.2f}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="11" fill="var(--muted)">{value}</text>'
        )
    parts.append(
        f'<line x1="{left:.2f}" y1="{base_y:.2f}" x2="{left + plot_w:.2f}" '
        f'y2="{base_y:.2f}" stroke="var(--axis)" stroke-width="1"/>'
    )

    for index, value in enumerate(counts):
        if not value:
            continue
        bar_w = max(1.0, plot_w / count - 2.0)
        bar_x = x_of(lo_edge + index * width) + 1.0
        # A bin holding one case out of hundreds is under a pixel tall, and a
        # tail that renders as nothing is exactly the tail worth seeing.
        bar_h = max(2.0, base_y - y_of(value))
        bar_y = base_y - bar_h
        radius = min(4.0, bar_w / 2.0, bar_h)
        parts.append(
            f'<path d="M{bar_x:.2f},{base_y:.2f} '
            f'V{bar_y + radius:.2f} '
            f'Q{bar_x:.2f},{bar_y:.2f} {bar_x + radius:.2f},{bar_y:.2f} '
            f'H{bar_x + bar_w - radius:.2f} '
            f'Q{bar_x + bar_w:.2f},{bar_y:.2f} '
            f'{bar_x + bar_w:.2f},{bar_y + radius:.2f} '
            f'V{base_y:.2f} Z" fill="{series}"/>'
        )
        if value == max(counts):
            parts.append(
                f'<text x="{bar_x + bar_w / 2:.2f}" y="{bar_y - 5:.2f}" '
                f'text-anchor="middle" font-size="11" '
                f'fill="var(--ink-2)">{value}</text>'
            )

    if lo_edge <= 0.0 <= lo_edge + span:
        zero_x = x_of(0.0)
        parts.append(
            f'<line x1="{zero_x:.2f}" y1="{top:.2f}" x2="{zero_x:.2f}" '
            f'y2="{base_y:.2f}" stroke="var(--axis)" stroke-width="2"/>'
        )

    if cases:
        median_x = x_of(statistics.median([c.log2_ratio for c in cases]))
        parts.append(
            f'<line x1="{median_x:.2f}" y1="{top:.2f}" x2="{median_x:.2f}" '
            f'y2="{base_y:.2f}" stroke="var(--muted)" stroke-width="2"/>'
        )
        anchor = "start" if median_x < left + plot_w * 0.75 else "end"
        offset = 5.0 if anchor == "start" else -5.0
        parts.append(
            f'<text x="{median_x + offset:.2f}" y="{top - 8:.2f}" '
            f'text-anchor="{anchor}" font-size="11" '
            f'fill="var(--muted)">median</text>'
        )

    tick = max(1, int(math.ceil(span / 10.0)))
    value = math.ceil(lo_edge / tick) * tick
    while value <= lo_edge + span + 1e-9:
        x = x_of(value)
        parts.append(
            f'<line x1="{x:.2f}" y1="{base_y:.2f}" x2="{x:.2f}" '
            f'y2="{base_y + 4:.2f}" stroke="var(--axis)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{base_y + 18:.2f}" text-anchor="middle" '
            f'font-size="11" fill="var(--muted)">'
            f'{2.0 ** value:g}x</text>'
        )
        value += tick
    parts.append(
        f'<text x="{left + plot_w / 2:.2f}" y="{h - 6:.2f}" '
        f'text-anchor="middle" font-size="11" fill="var(--muted)">'
        f'model / measured, log2 axis; shaded band is '
        f'{tolerance_pct:g}% tolerance</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Document model
#
# Both renderers walk the same node list, so the prose exists once and the two
# outputs cannot drift into saying different things about the same numbers.


class Table:
    """A rendered table: cells arrive already formatted."""

    def __init__(
        self,
        headers: Sequence[str],
        aligns: Sequence[str] | None = None,
        tall: bool = False,
    ) -> None:
        self.headers = list(headers)
        self.aligns = list(aligns) if aligns else ["l"] * len(self.headers)
        self.rows: list[list[str]] = []
        self.tall = tall

    def row(self, cells: Sequence[Any]) -> None:
        self.rows.append([str(c) for c in cells])


Node = tuple


def h(level: int, text: str) -> Node:
    return ("h", level, text)


def p(text: str) -> Node:
    return ("p", text)


def ul(items: Sequence[str]) -> Node:
    return ("ul", list(items))


def pre(text: str) -> Node:
    return ("pre", text)


def table(value: Table) -> Node:
    return ("table", value)


def callout(kind: str, tag: str, text: str) -> Node:
    return ("callout", kind, tag, text)


def chart(svg: str, fallback: str) -> Node:
    return ("chart", svg, fallback)


def details(summary: str, children: Sequence[Node]) -> Node:
    return ("details", summary, list(children))


# --------------------------------------------------------------------------
# Formatting helpers


def fmt_ratio(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.3f}x"


def fmt_pct(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.1f}%"


def fmt_log(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:+.3f}"


def fmt_us(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.2f}"


def fmt_share(part: int, whole: int) -> str:
    if not whole:
        return "0/0"
    return f"{part}/{whole} ({100.0 * part / whole:.1f}%)"


def slug(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


def esc(text: str) -> str:
    return html.escape(str(text), quote=False)


def esc_attr(text: str) -> str:
    return html.escape(str(text), quote=True)


def inline_html(text: str) -> str:
    """Escape, then honour the two inline markers the prose uses."""
    out = html.escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    return out


# --------------------------------------------------------------------------
# Sections


def section_verdict(runs: Sequence[Run]) -> list[Node]:
    nodes: list[Node] = [h(2, "Verdict")]
    for run in runs:
        summary = stats(run.cases, run.tolerance_pct)
        if not summary["n"]:
            nodes.append(
                callout(
                    "bad",
                    "NO DATA",
                    f"{run.label}: the payload scored no cases at all.",
                )
            )
            continue
        met = tolerance_met(summary, run.tolerance_pct)
        factor = 2.0 ** (summary["spread_log2"] / 2.0)
        nodes.append(
            callout(
                "good" if met else "bad",
                "TOLERANCE MET" if met else "TOLERANCE NOT MET",
                f"{run.label}: median absolute error "
                f"{fmt_pct(summary['median_abs_pct'])} against a stated tolerance "
                f"of {run.tolerance_pct:g}%, "
                f"{'inside' if met else 'outside'} it. "
                f"{fmt_share(int(summary['within_tolerance']), int(summary['n']))}"
                f" of scored cases are individually inside the band. "
                f"The middle 90% of cases span "
                f"{summary['spread_log2']:.3f} octaves of log2 ratio, a factor of "
                f"{factor:.2f}x either side of centre. "
                f"{fmt_share(int(summary['n']), run.reference_cases)} of the "
                f"reference cases were scored at all.",
            )
        )
    nodes.append(
        p(
            "The tolerance is met when the median absolute error is at or inside "
            "it. That is an aggregate criterion over a per-case band, so the "
            "fraction of cases individually inside the band is stated beside it "
            "rather than instead of it."
        )
    )
    return nodes


def section_coverage(runs: Sequence[Run]) -> list[Node]:
    nodes: list[Node] = [h(2, "Coverage")]
    nodes.append(
        p(
            "A suite that quietly drops cases flatters whatever is left, so this "
            "comes before the accuracy figures rather than after them. Every "
            "number in the rest of this report describes the scored column only."
        )
    )
    summary = Table(
        [
            "model",
            "reference cases",
            "scored",
            "missing from emulated run",
            "no kernel time on one side",
        ],
        ["l", "r", "r", "r", "r"],
    )
    for run in runs:
        summary.row(
            [
                run.label,
                run.reference_cases,
                fmt_share(len(run.cases), run.reference_cases),
                len(run.missing),
                len(run.unscorable),
            ]
        )
    nodes.append(table(summary))
    nodes.append(
        p(
            "`missing from emulated run` is a case the reference run produced and "
            "the emulated run did not: it timed out, crashed, or was never "
            "launched. `no kernel time on one side` is a case both runs produced "
            "where one of them reported no device work, which the scorer cannot "
            "turn into a ratio. Neither is evidence about the model, and neither "
            "is evidence that the model is fine."
        )
    )
    for run in runs:
        if not run.missing and not run.unscorable:
            nodes.append(p(f"{run.label}: every reference case was scored."))
            continue
        children: list[Node] = []
        if run.missing:
            children.append(
                p(f"Missing from the emulated run " f"({len(run.missing)}):")
            )
            children.append(ul(run.missing))
        if run.unscorable:
            children.append(
                p(f"No kernel time on one side " f"({len(run.unscorable)}):")
            )
            children.append(ul(run.unscorable))
        nodes.append(
            details(
                f"{run.label}: the {len(run.missing) + len(run.unscorable)} cases "
                f"that were not scored",
                children,
            )
        )
    return nodes


def accuracy_row(label: str, cases: Sequence[Case], tolerance_pct: float) -> list[str]:
    summary = stats(cases, tolerance_pct)
    if not summary["n"]:
        return [label] + ["n/a"] * 8
    return [
        label,
        str(int(summary["n"])),
        fmt_ratio(summary["median_ratio"]),
        fmt_pct(summary["median_abs_pct"]),
        fmt_pct(summary["p90_abs_pct"]),
        fmt_pct(summary["max_abs_pct"]),
        fmt_share(int(summary["within_tolerance"]), int(summary["n"])),
        fmt_share(int(summary["within_2x"]), int(summary["n"])),
        f"{fmt_ratio(summary['min_ratio'])} .. " f"{fmt_ratio(summary['max_ratio'])}",
    ]


# No header may contain a pipe: it would end the cell in the markdown
# table.  "abs err" throughout rather than "|err|" for that reason.
ACCURACY_HEADERS = [
    "",
    "cases",
    "median ratio",
    "median abs err",
    "p90 abs err",
    "max abs err",
    "within tolerance",
    "within 2x",
    "ratio range",
]
ACCURACY_ALIGNS = ["l", "r", "r", "r", "r", "r", "r", "r", "r"]


def section_accuracy(runs: Sequence[Run]) -> list[Node]:
    nodes: list[Node] = [h(2, "Accuracy")]
    overall = Table(["model"] + ACCURACY_HEADERS[1:], ACCURACY_ALIGNS)
    for run in runs:
        overall.row(accuracy_row(run.label, run.cases, run.tolerance_pct))
    nodes.append(table(overall))
    nodes.append(
        p(
            "`ratio` is model over measured, so below 1.00x the model reads fast. "
            "The error columns are absolute, never signed: a mean of signed "
            "errors cancels, and a model 2x fast on half the corpus and 2x slow "
            "on the other half would read as perfect."
        )
    )
    for run in runs:
        nodes.append(h(3, f"{run.label} by category"))
        per_category = Table(["category"] + ACCURACY_HEADERS[1:], ACCURACY_ALIGNS)
        for category, cases in by_category(run.cases).items():
            per_category.row(accuracy_row(category, cases, run.tolerance_pct))
        nodes.append(table(per_category))
        met = [
            c
            for c, cs in by_category(run.cases).items()
            if tolerance_met(stats(cs, run.tolerance_pct), run.tolerance_pct)
        ]
        missed = [c for c in by_category(run.cases) if c not in met]
        if met and missed:
            nodes.append(
                p(
                    f"Inside the {run.tolerance_pct:g}% tolerance on median "
                    f"error: "
                    + ", ".join(met)
                    + ". Outside it: "
                    + ", ".join(missed)
                    + "."
                )
            )
        elif met:
            nodes.append(
                p(
                    f"Every category is inside the {run.tolerance_pct:g}% "
                    f"tolerance on median error."
                )
            )
        elif missed:
            nodes.append(
                p(
                    f"No category is inside the {run.tolerance_pct:g}% tolerance "
                    f"on median error."
                )
            )
    return nodes


def section_spread(
    runs: Sequence[Run], width: float, lo_edge: float, count: int
) -> list[Node]:
    nodes: list[Node] = [h(2, "Spread of log2 ratio")]
    nodes.append(
        p(
            "This is the score. The spread is the width of the band holding the "
            "middle 90% of cases, measured in octaves of `log2(model / "
            "measured)`; one octave is a factor of two. Bias is reported beside "
            "it and is **not** the score: it is a median, so errors large in both "
            "directions cancel into a bias near zero. Correlation is not "
            "reported at all, because it is nearly free on a corpus spanning four "
            "orders of magnitude and published GPU models reach 0.97 of it at "
            "42-75% mean absolute error."
        )
    )
    spread = Table(
        [
            "model",
            "cases",
            "spread p05..p95",
            "as a factor",
            "bias (not the score)",
            "stdev log2",
            "median abs log2",
            "p90 abs log2",
        ],
        ["l", "r", "r", "r", "r", "r", "r", "r"],
    )
    for run in runs:
        summary = stats(run.cases, run.tolerance_pct)
        if not summary["n"]:
            spread.row([run.label] + ["n/a"] * 7)
            continue
        spread.row(
            [
                run.label,
                int(summary["n"]),
                f"{summary['spread_log2']:.3f} oct "
                f"({fmt_log(summary['p05_log2'])} .. "
                f"{fmt_log(summary['p95_log2'])})",
                f"{2.0 ** (summary['spread_log2'] / 2.0):.2f}x",
                f"{fmt_log(summary['bias_log2'])} "
                f"({2.0 ** summary['bias_log2']:.3f}x)",
                f"{summary['stdev_log2']:.3f}",
                f"{summary['median_abs_log2']:.3f}",
                f"{summary['p90_abs_log2']:.3f}",
            ]
        )
    nodes.append(table(spread))
    peak = max(
        [1]
        + [
            max(bin_counts(r.cases, width, lo_edge, count) or [0])
            for r in runs
            if r.cases
        ]
    )
    for index, run in enumerate(runs):
        nodes.append(h(3, f"{run.label}: distribution"))
        series = f"var(--series-{index % SERIES_SLOTS + 1})"
        fallback = text_histogram(run.cases, width, lo_edge, count, run.tolerance_pct)
        if run.cases:
            nodes.append(
                chart(
                    svg_histogram(
                        run.cases,
                        width,
                        lo_edge,
                        count,
                        peak,
                        run.tolerance_pct,
                        run.label,
                        series,
                    ),
                    fallback,
                )
            )
        else:
            nodes.append(pre(fallback))
    if len(runs) > 1:
        nodes.append(
            p(
                "The histograms share both axes, so the bars are directly "
                "comparable between models."
            )
        )
    return nodes


def section_wrong(runs: Sequence[Run], worst_n: int) -> list[Node]:
    nodes: list[Node] = [h(2, "What the model gets wrong")]
    for run in runs:
        if not run.cases:
            continue
        nodes.append(h(3, run.label))
        ordered = [
            c
            for c in sorted(run.cases, key=lambda c: (-c.abs_pct, c.case_id))
            if c.abs_pct > run.tolerance_pct
        ]
        worst = ordered[:worst_n]
        if not ordered:
            nodes.append(
                p(
                    f"No case is outside the {run.tolerance_pct:g}% tolerance, so "
                    f"there is nothing here to group."
                )
            )
            continue
        if not worst:
            nodes.append(
                p(
                    f"{len(ordered)} cases are outside the "
                    f"{run.tolerance_pct:g}% tolerance; `--worst 0` suppressed "
                    f"the list."
                )
            )
            continue
        slow = sum(1 for c in worst if c.ratio > 1.0)
        nodes.append(
            p(
                f"The {len(worst)} worst of the {len(ordered)} cases outside the "
                f"{run.tolerance_pct:g}% tolerance, by absolute error. {slow} of "
                f"them read slow and {len(worst) - slow} read fast. An effect a "
                f"model does not know about is charged zero, which always reads "
                f"fast, so a fast tail is the shape to look at first."
            )
        )
        traits = shared_traits(worst, run.cases)
        if traits:
            nodes.append(p("What they have in common:"))
            nodes.append(
                ul(
                    [
                        f"{facet} = {value}: {count} of the {len(worst)} worst "
                        f"({100.0 * count / len(worst):.0f}%), against "
                        f"{corpus_pct:.0f}% of the scored corpus"
                        for facet, value, count, corpus_pct in traits
                    ]
                )
            )
        else:
            nodes.append(
                p(
                    "Nothing groups them: no category, dtype, implementation, "
                    "direction or size band is over-represented among the worst "
                    "cases, which points at per-case error rather than a missing "
                    "effect."
                )
            )
        worst_table = Table(
            [
                "case",
                "category",
                "measured us",
                "model us",
                "ratio",
                "abs err",
                "launches",
            ],
            ["l", "l", "r", "r", "r", "r", "r"],
        )
        for case in worst:
            worst_table.row(
                [
                    case.case_id,
                    case.category,
                    fmt_us(case.measured_us),
                    fmt_us(case.model_us),
                    fmt_ratio(case.ratio),
                    fmt_pct(case.abs_pct),
                    case.launches,
                ]
            )
        nodes.append(table(worst_table))
    return nodes


def section_comparison(runs: Sequence[Run], worst_n: int) -> list[Node]:
    nodes: list[Node] = [h(2, "Model comparison")]
    shared = sorted(set.intersection(*[set(r.by_id) for r in runs]))
    nodes.append(
        p(
            f"{len(shared)} cases were scored by every model and only those are "
            f"compared here; a model that dropped a case cannot win or lose it."
        )
    )
    if not shared:
        return nodes

    categories = sorted({runs[0].by_id[c].category for c in shared})
    per_category = Table(
        ["category", "cases"] + [f"{r.label} median abs err" for r in runs] + ["best"],
        ["l", "r"] + ["r"] * len(runs) + ["l"],
    )
    for category in categories:
        cases = [c for c in shared if runs[0].by_id[c].category == category]
        medians = [
            stats([r.by_id[c] for c in cases], r.tolerance_pct)["median_abs_pct"]
            for r in runs
        ]
        best = min(medians)
        winners = [r.label for r, m in zip(runs, medians) if m == best]
        per_category.row(
            [category, len(cases)]
            + [fmt_pct(m) for m in medians]
            + ["tie" if len(winners) > 1 else winners[0]]
        )
    nodes.append(table(per_category))

    wins = {r.label: 0 for r in runs}
    ties = 0
    for case_id in shared:
        errors = [r.by_id[case_id].abs_pct for r in runs]
        best = min(errors)
        winners = [r.label for r, e in zip(runs, errors) if e == best]
        if len(winners) > 1:
            ties += 1
        else:
            wins[winners[0]] += 1
    win_table = Table(["model", "cases with the smallest absolute error"], ["l", "r"])
    for run in runs:
        win_table.row([run.label, fmt_share(wins[run.label], len(shared))])
    win_table.row(["tied", fmt_share(ties, len(shared))])
    nodes.append(table(win_table))

    def disagreement(case_id: str) -> float:
        logs = [r.by_id[case_id].log2_ratio for r in runs]
        return max(logs) - min(logs)

    if worst_n <= 0:
        return nodes
    ordered = sorted(shared, key=lambda c: (-disagreement(c), c))[:worst_n]
    disagree = Table(
        ["case", "measured us"]
        + [f"{r.label} us" for r in runs]
        + [f"{r.label} ratio" for r in runs]
        + ["gap (octaves)"],
        ["l", "r"] + ["r"] * (2 * len(runs)) + ["r"],
    )
    for case_id in ordered:
        row = [case_id, fmt_us(runs[0].by_id[case_id].measured_us)]
        row += [fmt_us(r.by_id[case_id].model_us) for r in runs]
        row += [fmt_ratio(r.by_id[case_id].ratio) for r in runs]
        row.append(f"{disagreement(case_id):.3f}")
        disagree.row(row)
    nodes.append(
        p(
            "Where the models disagree most, in octaves between the highest and "
            "lowest ratio. These are the cases a model change actually moved, so "
            "they are where to look for the mechanism behind a change in the "
            "aggregate."
        )
    )
    nodes.append(table(disagree))
    return nodes


def section_slowdown(
    runs: Sequence[Run], walls: dict[str, float], baseline: str | None
) -> list[Node]:
    nodes: list[Node] = [h(2, "Cost of running the model")]
    if not walls:
        nodes.append(
            callout(
                "warn",
                "NOT MEASURED",
                "No `--wall LABEL=SECONDS` pairs were given, so the wall-clock "
                "cost of running the timing model over this corpus is unknown. It "
                "is not zero. This section is present and empty rather than "
                "omitted, so an unmeasured cost does not read as a free one.",
            )
        )
        return nodes
    if baseline is None:
        nodes.append(
            callout(
                "warn",
                "NO BASELINE",
                "Wall clocks were given but none of them is the run with no "
                "timing model loaded, so no multiplier can be computed. Pass the "
                "untimed corpus wall clock as `--wall none=SECONDS`, or name the "
                "baseline with `--wall-baseline LABEL`.",
            )
        )
    cases_by_label = {r.label: len(r.cases) for r in runs}
    wall_table = Table(
        ["run", "wall clock", "vs baseline", "per scored case"], ["l", "r", "r", "r"]
    )
    for label in sorted(walls):
        seconds = walls[label]
        if baseline is not None and label == baseline:
            multiplier = "baseline"
        elif baseline is not None and walls[baseline] > 0.0:
            multiplier = f"{seconds / walls[baseline]:.2f}x"
        else:
            multiplier = "n/a"
        scored = cases_by_label.get(label)
        per_case = f"{seconds / scored:.2f} s" if scored else "n/a"
        wall_table.row(
            [
                label,
                f"{seconds:.0f} s " f"({seconds / 60.0:.1f} min)",
                multiplier,
                per_case,
            ]
        )
    nodes.append(table(wall_table))
    nodes.append(
        p(
            "Wall clock is the whole emulated corpus run, not the model in "
            "isolation: it includes functional simulation, which dominates. The "
            "multiplier is therefore what the model costs a user of the emulator, "
            "which is the number worth knowing, and not the model's own share of "
            "the time."
        )
    )
    return nodes


def section_cases(runs: Sequence[Run], max_cases: int) -> list[Node]:
    nodes: list[Node] = [h(2, "Every scored case")]
    nodes.append(
        p(
            "Sorted by absolute error, worst first. `real event us` is the "
            "hardware run's `torch.cuda.Event` bracket, which carries the host's "
            "enqueue gap and is reported only so the gap against `measured us` "
            "stays visible; it is not what the model is scored against."
        )
    )
    for run in runs:
        nodes.append(h(3, run.label))
        ordered = sorted(run.cases, key=lambda c: (-c.abs_pct, c.case_id))
        shown = ordered[:max_cases] if max_cases > 0 else ordered
        case_table = Table(
            [
                "case",
                "category",
                "measured us",
                "model us",
                "ratio",
                "abs err",
                "log2 ratio",
                "launches",
                "real event us",
            ],
            ["l", "l", "r", "r", "r", "r", "r", "r", "r"],
            tall=True,
        )
        for case in shown:
            case_table.row(
                [
                    case.case_id,
                    case.category,
                    fmt_us(case.measured_us),
                    fmt_us(case.model_us),
                    fmt_ratio(case.ratio),
                    fmt_pct(case.abs_pct),
                    fmt_log(case.log2_ratio),
                    case.launches,
                    fmt_us(case.real_event_us),
                ]
            )
        nodes.append(table(case_table))
        if len(shown) < len(ordered):
            nodes.append(
                p(
                    f"{len(ordered) - len(shown)} further cases were "
                    f"cut by `--max-cases`."
                )
            )
    return nodes


def section_method(runs: Sequence[Run], generated_at: str | None) -> list[Node]:
    nodes: list[Node] = [h(2, "How these numbers are defined")]
    nodes.append(
        p(
            "The model's opinion is `event_us` from the emulated run, the "
            "`torch.cuda.Event` bracket around the call. Under emulation the "
            "modelled clock does not advance while the device is idle, so that "
            "bracket contains the modelled kernels and nothing else."
        )
    )
    nodes.append(
        p(
            "The measurement it is scored against is `kernel_us` from the run "
            "recorded on hardware: the sum of `self_device_time_us` over the real "
            "device kernels of the case, divided by the profiler's iteration "
            "count. Most cases launch more than one kernel, so it is a sum and "
            "not a maximum."
        )
    )
    nodes.append(
        p(
            "The emulated bracket is deliberately not compared against the "
            "hardware bracket, which also contains the host's python enqueue gap "
            "and would score the model on the measurement harness. See the module "
            "docstring of `meter_score.py` for the full argument; this report "
            "recomputes its statistics from the same payload and adds no data."
        )
    )
    provenance = Table(["model", "tolerance", "scored payload"], ["l", "r", "l"])
    for run in runs:
        provenance.row([run.label, f"{run.tolerance_pct:g}%", run.path])
    nodes.append(table(provenance))
    if generated_at:
        nodes.append(p(f"Generated at: {generated_at}."))
    return nodes


def build_document(
    runs: Sequence[Run],
    walls: dict[str, float],
    baseline: str | None,
    worst_n: int,
    max_cases: int,
    generated_at: str | None,
) -> list[Node]:
    width, lo_edge, count = choose_bins([c.log2_ratio for r in runs for c in r.cases])
    nodes: list[Node] = []
    nodes += section_verdict(runs)
    nodes += section_coverage(runs)
    nodes += section_accuracy(runs)
    nodes += section_spread(runs, width, lo_edge, count)
    if len(runs) > 1:
        nodes += section_comparison(runs, worst_n)
    nodes += section_wrong(runs, worst_n)
    nodes += section_slowdown(runs, walls, baseline)
    nodes += section_cases(runs, max_cases)
    nodes += section_method(runs, generated_at)
    return nodes


# --------------------------------------------------------------------------
# Renderers


def md_table(value: Table) -> list[str]:
    """A pipe table.  Cells are escaped first, then padded to that width."""
    value = escaped_table(value)
    widths = [len(header) for header in value.headers]
    for row in value.rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(cells: Sequence[str]) -> str:
        padded = [
            cell.rjust(widths[i]) if value.aligns[i] == "r" else cell.ljust(widths[i])
            for i, cell in enumerate(cells)
        ]
        return "| " + " | ".join(padded) + " |"

    out = [line(value.headers)]
    out.append(
        "|"
        + "|".join(
            (
                ("-" * (widths[i] + 1) + ":")
                if value.aligns[i] == "r"
                else (":" + "-" * (widths[i] + 1))
            )
            for i in range(len(value.headers))
        )
        + "|"
    )
    out += [line(row) for row in value.rows]
    return out


def escaped_table(value: Table) -> Table:
    """A copy whose cells cannot end their own column."""
    out = Table(
        [h.replace("|", r"\|") for h in value.headers], value.aligns, value.tall
    )
    for row in value.rows:
        out.row([cell.replace("|", r"\|") for cell in row])
    return out


def render_markdown(title: str, nodes: Sequence[Node]) -> str:
    out: list[str] = [f"# {title}", ""]
    for section in nodes:
        if section[0] == "h":
            out += [f"{'#' * section[1]} {section[2]}", ""]
        elif section[0] == "p":
            out += [section[1], ""]
        elif section[0] == "ul":
            out += [f"- {item}" for item in section[1]] + [""]
        elif section[0] == "pre":
            out += ["```", section[1], "```", ""]
        elif section[0] == "table":
            out += md_table(section[1]) + [""]
        elif section[0] == "callout":
            out += [f"> **{section[2]}** {section[3]}", ""]
        elif section[0] == "chart":
            out += ["```", section[2], "```", ""]
        elif section[0] == "details":
            out += [f"**{section[1]}**", ""]
            body = render_markdown("", section[2]).split("\n")
            out += [l for l in body[2:]]
        else:
            raise AssertionError(f"unhandled node {section[0]}")
    return "\n".join(out).rstrip("\n") + "\n"


CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11, 11, 11, 0.10);
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --band: rgba(42, 120, 214, 0.10);
  --good: #0ca30c;
  --warning: #fab219;
  --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface: #1a1a19;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255, 255, 255, 0.10);
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --band: rgba(57, 135, 229, 0.16);
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 62rem; margin: 0 auto; padding: 32px 20px 96px; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 {
  font-size: 20px;
  margin: 44px 0 12px;
  padding-top: 14px;
  border-top: 1px solid var(--border);
}
h3 { font-size: 15px; margin: 26px 0 8px; color: var(--ink-2); }
p { margin: 10px 0; max-width: 46rem; }
ul { margin: 8px 0; padding-left: 20px; max-width: 46rem; }
li { margin: 2px 0; }
code {
  font: 0.9em ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0 4px;
}
nav ul {
  list-style: none;
  padding: 0;
  max-width: none;
  display: flex;
  flex-wrap: wrap;
  gap: 4px 18px;
  font-size: 13px;
}
nav a { color: var(--series-1); text-decoration: none; }
nav a:hover { text-decoration: underline; }
.scroll {
  overflow-x: auto;
  margin: 12px 0;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface);
}
.scroll.tall { max-height: 32rem; overflow: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td {
  padding: 6px 10px;
  text-align: left;
  white-space: nowrap;
  border-bottom: 1px solid var(--grid);
}
th {
  position: sticky;
  top: 0;
  background: var(--surface);
  color: var(--ink-2);
  font-weight: 600;
}
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr:last-child td { border-bottom: none; }
pre {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  overflow-x: auto;
  font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
figure {
  margin: 12px 0;
  padding: 8px 8px 0;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface);
}
svg { display: block; width: 100%; height: auto; }
.callout {
  margin: 14px 0;
  padding: 10px 14px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 4px solid var(--muted);
  border-radius: 0 8px 8px 0;
  max-width: 46rem;
}
.callout .tag {
  font-weight: 700;
  font-size: 12px;
  letter-spacing: 0.04em;
  margin-right: 8px;
}
.callout.good { border-left-color: var(--good); }
.callout.good .tag { color: var(--good); }
.callout.warn { border-left-color: var(--warning); }
.callout.bad { border-left-color: var(--critical); }
.callout.bad .tag { color: var(--critical); }
details { margin: 12px 0; }
summary { cursor: pointer; color: var(--ink-2); font-size: 14px; }
.sub { color: var(--ink-2); font-size: 13px; margin: 0 0 18px; }
"""


def html_table(value: Table) -> list[str]:
    classes = "scroll tall" if value.tall else "scroll"
    out = [f'<div class="{classes}"><table><thead><tr>']
    for index, header in enumerate(value.headers):
        css = ' class="num"' if value.aligns[index] == "r" else ""
        out.append(f"<th{css}>{esc(header)}</th>")
    out.append("</tr></thead><tbody>")
    for row in value.rows:
        out.append("<tr>")
        for index, cell in enumerate(row):
            css = ' class="num"' if value.aligns[index] == "r" else ""
            out.append(f"<td{css}>{esc(cell)}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return out


def render_html_nodes(nodes: Sequence[Node]) -> list[str]:
    out: list[str] = []
    for section in nodes:
        if section[0] == "h":
            level = section[1]
            anchor = f' id="{esc_attr(slug(section[2]))}"' if level == 2 else ""
            out.append(f"<h{level}{anchor}>{esc(section[2])}</h{level}>")
        elif section[0] == "p":
            out.append(f"<p>{inline_html(section[1])}</p>")
        elif section[0] == "ul":
            out.append("<ul>")
            out += [f"<li>{inline_html(item)}</li>" for item in section[1]]
            out.append("</ul>")
        elif section[0] == "pre":
            out.append(f"<pre>{esc(section[1])}</pre>")
        elif section[0] == "table":
            out += html_table(section[1])
        elif section[0] == "callout":
            out.append(
                f'<div class="callout {esc_attr(section[1])}">'
                f'<span class="tag">{esc(section[2])}</span>'
                f"{inline_html(section[3])}</div>"
            )
        elif section[0] == "chart":
            out.append(f"<figure>{section[1]}</figure>")
            out.append("<details><summary>the same distribution as text" "</summary>")
            out.append(f"<pre>{esc(section[2])}</pre></details>")
        elif section[0] == "details":
            out.append(f"<details><summary>{esc(section[1])}</summary>")
            out += render_html_nodes(section[2])
            out.append("</details>")
        else:
            raise AssertionError(f"unhandled node {section[0]}")
    return out


def render_html(title: str, subtitle: str, nodes: Sequence[Node]) -> str:
    headings = [n[2] for n in nodes if n[0] == "h" and n[1] == 2]
    nav = "".join(
        f'<li><a href="#{esc_attr(slug(text))}">{esc(text)}</a></li>'
        for text in headings
    )
    body = "\n".join(render_html_nodes(nodes))
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" '
        'content="width=device-width, initial-scale=1">\n'
        f"<title>{esc(title)}</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n<main>\n"
        f"<h1>{esc(title)}</h1>\n"
        f'<p class="sub">{esc(subtitle)}</p>\n'
        f"<nav><ul>{nav}</ul></nav>\n"
        f"{body}\n"
        "</main>\n</body>\n</html>\n"
    )


# --------------------------------------------------------------------------
# Command line


def parse_pair(text: str, what: str) -> tuple[str, str]:
    """Split a ``LABEL=VALUE`` argument, keeping ``=`` legal inside a path."""
    label, separator, value = text.partition("=")
    if not separator or not label or not value:
        raise SystemExit(f"{what} wants LABEL=VALUE, got {text!r}")
    return label, value


def pick_baseline(walls: dict[str, float], requested: str | None) -> str | None:
    """The untimed run the slowdown multiplier is taken against."""
    if requested is not None:
        if requested not in walls:
            raise SystemExit(f"--wall-baseline {requested!r} has no --wall pair")
        return requested
    for candidate in BASELINE_WALL_LABELS:
        if candidate in walls:
            return candidate
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scored",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="a meter_score.py --json payload, under the label to report it "
        "as; repeat for a model or branch comparison",
    )
    parser.add_argument(
        "--out", metavar="DIR", help="directory the report is written to"
    )
    parser.add_argument("--title", default="rocjitsu timing model report")
    parser.add_argument(
        "--basename",
        default="meter-report",
        help="output file stem (default: meter-report)",
    )
    parser.add_argument(
        "--format", dest="formats", default="both", choices=("md", "html", "both")
    )
    parser.add_argument(
        "--wall",
        action="append",
        default=[],
        metavar="LABEL=SECONDS",
        help="emulator wall clock over the same corpus, in seconds; pass the "
        "run with no timing model as none=SECONDS to get a multiplier",
    )
    parser.add_argument(
        "--wall-baseline", metavar="LABEL", help="which --wall pair is the untimed run"
    )
    parser.add_argument(
        "--worst", type=int, default=12, help="how many worst cases to group and list"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="truncate the per-case table; 0 lists every case",
    )
    parser.add_argument(
        "--generated-at",
        metavar="TEXT",
        help="stamped verbatim into the report; without it "
        "the output carries no date and is reproducible",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="render a synthetic payload and check the result",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.scored:
        raise SystemExit("--scored LABEL=PATH is required")
    if not args.out:
        raise SystemExit("--out DIR is required")

    runs: list[Run] = []
    seen: set[str] = set()
    for spec in args.scored:
        label, path = parse_pair(spec, "--scored")
        if label in seen:
            raise SystemExit(f"--scored label {label!r} used twice")
        seen.add(label)
        runs.append(load_run(label, path))

    walls: dict[str, float] = {}
    for spec in args.wall:
        label, value = parse_pair(spec, "--wall")
        try:
            walls[label] = float(value)
        except ValueError:
            raise SystemExit(f"--wall {spec!r}: {value!r} is not a number")
    baseline = pick_baseline(walls, args.wall_baseline)

    nodes = build_document(
        runs,
        walls,
        baseline,
        max(0, args.worst),
        max(0, args.max_cases),
        args.generated_at,
    )
    scored = sum(len(r.cases) for r in runs)
    reference = max(r.reference_cases for r in runs)
    subtitle = (
        f"{len(runs)} model{'s' if len(runs) != 1 else ''}, "
        f"{scored} scored case{'s' if scored != 1 else ''} of "
        f"{reference} in the reference run; scored on the spread of "
        f"log2(model / measured)"
    )

    os.makedirs(args.out, exist_ok=True)
    written: list[str] = []
    if args.formats in ("md", "both"):
        path = os.path.join(args.out, args.basename + ".md")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(render_markdown(args.title, nodes))
        written.append(path)
    if args.formats in ("html", "both"):
        path = os.path.join(args.out, args.basename + ".html")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(render_html(args.title, subtitle, nodes))
        written.append(path)
    for path in written:
        print(path)
    return 0


# --------------------------------------------------------------------------
# Self test


def synthetic_payload(scale: float) -> dict[str, Any]:
    """A payload whose statistics can be worked out by hand.

    The eight ratios are 0.5, 0.5, 1, 1, 1, 2, 2, 4, so the median ratio is
    exactly 1.000x while the median absolute error is 50%: the case the report
    exists to make, which is that a centred model can still be badly wrong.
    """
    ratios = [0.5, 0.5, 1.0, 1.0, 1.0, 2.0, 2.0, 4.0]
    names = [
        "gemm.eager.float32.256x256x256",
        "gemm.compiled.float32.512x512x512",
        "triad.eager.float32.nhwc.1048576",
        "copy.eager.float16.nhwc.1048576",
        "rms_norm.plain.eager.bfloat16.4096x8192",
        "attention_qk.eager.float16.2048x128",
        "attention_pv.triton.float16.2048x128",
        "swiglu.triton.bfloat16.4096x8192",
    ]
    categories = [
        "gemm",
        "gemm",
        "memory",
        "memory",
        "normalization",
        "attention",
        "attention",
        "activation",
    ]
    cases = []
    for index, (name, category, ratio) in enumerate(zip(names, categories, ratios)):
        measured = 10.0 * (index + 1)
        model = measured * ratio * scale
        cases.append(
            {
                "case_id": name,
                "category": category,
                "real_kernel_us": measured,
                "model_us": model,
                "ratio": model / measured,
                "abs_pct": abs(model - measured) / measured * 100.0,
                "launches": 1 + index % 3,
                "real_event_us": measured + 9.0,
                "model_event_us": model,
            }
        )
    return {
        "summary": {"cases": len(cases)},
        "tolerance_pct": 20.0,
        "cases": cases,
        "unscorable": ["edge.empty_tensor.float32"],
        "missing": [
            "conv2d.eager.float32.nchw.1x3x224x224",
            "concurrent_gemm.streams2.float16.1024",
        ],
    }


def selftest() -> int:
    """Render both formats from a synthetic payload and check the result."""
    import contextlib
    import io
    import shutil
    import tempfile

    root = tempfile.mkdtemp(prefix="meter-report-selftest-")
    try:
        des = os.path.join(root, "des.json")
        leaky = os.path.join(root, "leaky.json")
        with open(des, "w", encoding="utf-8") as handle:
            json.dump(synthetic_payload(1.0), handle)
        with open(leaky, "w", encoding="utf-8") as handle:
            json.dump(synthetic_payload(1.5), handle)

        first = os.path.join(root, "a")
        second = os.path.join(root, "b")
        argv = [
            "--scored",
            f"des={des}",
            "--scored",
            f"leaky={leaky}",
            "--wall",
            "none=600",
            "--wall",
            "des=2400",
            "--wall",
            "leaky=900",
            "--title",
            "selftest report",
            "--format",
            "both",
        ]
        # The two renders go to different directories, so the paths main()
        # prints differ; only the file contents are supposed to match.
        with contextlib.redirect_stdout(io.StringIO()):
            assert main(argv + ["--out", first]) == 0
            assert main(argv + ["--out", second]) == 0

        with open(os.path.join(first, "meter-report.md"), encoding="utf-8") as handle:
            md = handle.read()
        with open(os.path.join(first, "meter-report.html"), encoding="utf-8") as handle:
            page = handle.read()

        checks: list[tuple[str, str]] = [
            # The hand-computed statistics of synthetic_payload.
            ("median ratio 1.000x", "1.000x"),
            ("median absolute error 50.0%", "50.0%"),
            ("worst case error 300.0%", "300.0%"),
            ("within tolerance 3/8", "3/8 (37.5%)"),
            ("within 2x 7/8", "7/8 (87.5%)"),
            # p95 - p05 of [-1,-1,0,0,0,1,1,2] under the shared estimator.
            ("log2 spread 2.650 octaves", "2.650 oct"),
            ("bias +0.000", "+0.000"),
            ("tolerance verdict", "TOLERANCE NOT MET"),
            ("coverage: 8 of 11 reference cases", "8/11 (72.7%)"),
            ("the unscored case is named", "edge.empty_tensor.float32"),
            ("the missing case is named", "conv2d.eager.float32.nchw.1x3x224x224"),
            ("slowdown multiplier 4.00x", "4.00x"),
            ("text histogram present", "<- 1.00x"),
            ("worst case named", "swiglu.triton.bfloat16.4096x8192"),
            ("model comparison present", "smallest absolute error"),
        ]
        for what, needle in checks:
            assert needle in md, f"markdown is missing {what}: {needle!r}"
            assert (
                needle in page or esc(needle) in page
            ), f"html is missing {what}: {needle!r}"

        # The HTML has to stand alone, and has to work in both themes.
        for needle in (
            "<svg",
            ":root {",
            "@media (prefers-color-scheme: dark)",
            "overflow-x: auto",
            "background: var(--page)",
        ):
            assert needle in page, f"html is missing {needle!r}"
        for forbidden in (
            "<script",
            "<link",
            "http://",
            "https://fonts",
            "src=",
            "@import",
        ):
            assert forbidden not in page, (
                f"html is not self-contained: " f"{forbidden!r}"
            )

        for name in ("meter-report.md", "meter-report.html"):
            with open(os.path.join(first, name), "rb") as handle:
                left = handle.read()
            with open(os.path.join(second, name), "rb") as handle:
                right = handle.read()
            assert left == right, f"{name} is not deterministic"

        print(
            f"selftest: {len(checks)} value checks, "
            f"6 self-containment checks, 2 byte-identity checks"
        )
        print(f"selftest: markdown {len(md)} bytes, html {len(page)} bytes")
        print("selftest: ok")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
