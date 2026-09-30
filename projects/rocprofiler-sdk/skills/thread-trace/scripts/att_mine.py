#!/usr/bin/env python3
"""Read a thread trace capture through the rocprof-trace-decoder Python API.

usage: att_mine.py <capture_dir> <command> [--top N] [--json]
       att_mine.py <capture_dir> compare <other_capture_dir> [--top N]

commands:
  stats      the stats_*.csv rocprofv3 writes, ranked by instruction and by source line
             (reads only the CSV; no decoder needed)
  summary    decoder warnings, waves, dispatch resources, wave-state split, active waves
  hotspots   instructions ranked by non-hidden latency, with per-hit p50/p90
  lines      the same ranking grouped by source line
  pipes      issue and stall cycles by instruction class (VALU, transcendental, matrix, LDS,
             memory, waits, ...) and how much of the SIMDs' time vector instructions issue
  barriers   how each workgroup's waves split their time around s_barrier: which wave the
             others waited for, their barrier shares, and the lines that wave ran
  lifetime   wave lifetime against s_wait latency, VALU latency, other latency, and idle time
  occupancy  active waves and SGPR/VGPR allocation over time
  compare    cost per source line in this capture vs another

stats, summary, pipes, hotspots, and lines end with a "next:" line naming the page of the
skill's resources/ that covers what the trace shows.

Every number comes from the decoder's records (waves with their instructions and timeline
states, occupancy, dispatches), its CodeIndex, or its hidden-latency analysis. hotspots,
lifetime, and occupancy compute what the decoder's hidden_latency_hotspots,
plot_wave_lifetime, and plot_occupancy_resources samples compute.

Import it to write your own queries: Capture(path) exposes .waves (WaveTrace objects with
.insts, each an Inst with time, duration, stall, latency, idle, category, text, source).

Needs Python 3.10 or later with pyelftools, PYTHONPATH=$DECODER_ROOT/python, and
ROCPROF_TRACE_DECODER_LIB=<library file>.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass, field
from pathlib import Path

try:
    from rocprof_trace_decoder import Pc
except ImportError:  # stats and the analyses over already-decoded waves need no decoder
    Pc = namedtuple("Pc", "address code_object_id")


@dataclass
class Inst:
    time: int
    duration: int
    stall: int
    idle: int
    category: str
    text: str
    source: str
    pc: Pc

    @property
    def latency(self) -> int:
        """Issue plus stall cycles, as the decoder's CodeIndex counts them."""
        return max(self.stall, self.duration)

    @property
    def cost(self) -> int:
        return self.latency + self.idle


@dataclass
class WaveTrace:
    se: int
    cu: int
    simd: int
    wave_id: int
    workgroup: int
    begin: int
    end: int
    contexts: int
    insts: list[Inst] = field(default_factory=list)
    states: Counter = field(default_factory=Counter)

    @property
    def lifetime(self) -> int:
        return max(self.end - self.begin, 0)


def pct(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(q * len(ordered)))])


GPU_FIELDS = (
    "cu_count",
    "simd_per_cu",
    "wave_front_size",
    "max_waves_per_simd",
    "lds_size_in_kb",
    "max_engine_clk_fcompute",
    "num_xcc",
)


def gpu_properties(path: Path) -> dict:
    """The traced GPU as rocprofv3 records it: rocpd_info_agent in *_results.db, or the
    *_agent_info.csv that --output-format csv writes."""
    for db_path in sorted(path.rglob("*_results.db")):
        import sqlite3

        try:
            with sqlite3.connect(db_path) as db:
                rows = db.execute(
                    "SELECT product_name, name, extdata FROM rocpd_info_agent "
                    "WHERE type = 'GPU'"
                ).fetchall()
        except sqlite3.Error:
            continue
        if rows:
            product, name, extdata = rows[0]
            info = json.loads(extdata or "{}")
            return {
                "product": product,
                "name": name,
                **{k: info[k] for k in GPU_FIELDS if k in info},
            }
    for csv_path in sorted(path.rglob("*agent_info.csv")):
        with csv_path.open(newline="") as handle:
            for r in csv.DictReader(handle):
                if r.get("Agent_Type") == "GPU":
                    fields = {k.lower(): v for k, v in r.items()}
                    return {
                        "product": r.get("Product_Name"),
                        "name": r.get("Name"),
                        **{
                            k: int(fields[k])
                            for k in GPU_FIELDS
                            if fields.get(k, "").isdigit()
                        },
                    }
    return {}


def short_source(source: str) -> str:
    """file:line without directories; for inlined code, the call site in your file first."""
    if not source:
        return "?"
    chain = [part.rsplit("/", 1)[-1] for part in source.split(" -> ")]
    return chain[-1] if len(chain) == 1 else f"{chain[-1]} (inlined {chain[0]})"


RESOURCES = Path(__file__).resolve().parents[1] / "resources"
MATRIX_OPS = ("v_mfma", "v_smfmac", "v_wmma", "v_swmmac")
TRANSCENDENTAL = re.compile(r"v_(exp|log|rcp|rsq|sqrt|sin|cos)_")
FLAT_PROFILE_SHARE = 5.0  # hotspots: no instruction holds more than this % of the cost
LAUNCH_SLACK = 64  # cycles within which the waves of one workgroup are launched


def inst_class(inst: Inst) -> str:
    """The instruction's decoder category, with VALU split into matrix, transcendental, and
    other VALU instructions, and the IMMED waits and barriers named."""
    op = inst.text.split(" ", 1)[0]
    if inst.category == "VALU":
        if op.startswith(MATRIX_OPS):
            return "VALU matrix"
        return "VALU transcendental" if TRANSCENDENTAL.match(op) else "VALU"
    if op.startswith(("s_waitcnt", "s_wait_")):
        return "wait (s_waitcnt, s_wait_*)"
    if op.startswith("s_barrier"):
        return "barrier (s_barrier)"
    return inst.category


def next_read(states: dict, pipes: dict | None = None) -> str:
    """Which page of resources/ covers what the trace shows, and why."""
    if not states:
        return ""
    state = max(states, key=states.get)
    held = {c["class"]: c["stall_share"] for c in (pipes or {}).get("classes", [])}
    if state == "WAIT":
        page, why = "latency.md", "WAIT is the largest wave state"
        if held.get("barrier (s_barrier)", 0) > held.get("wait (s_waitcnt, s_wait_*)", 0):
            page, why = (
                "synchronization.md",
                "WAIT is the largest wave state and s_barrier holds it; "
                "`barriers` shows which wave the others wait for",
            )
    elif state == "IDLE":
        page, why = "stalls.md#idle-cycles", "IDLE is the largest wave state"
    else:
        page = "compute.md"
        why = f"{state} is the largest wave state"
        busy = [c for c in held if not c.startswith(("wait", "barrier"))]
        if state == "STALL" and busy:
            why += (
                f"; {max(busy, key=held.get)} instructions hold most of the stall (a busy pipe, "
                "or waits for earlier results)"
            )
        if pipes and pipes.get("valu_issue_share", 0) >= 80:
            why += f"; vector instructions issue {pipes['valu_issue_share']}% of the SIMDs' time"
            page = "compute.md#ceiling-check"
    name, _, anchor = page.partition("#")
    where = f"{RESOURCES / name}" + (
        f" (section {anchor.replace('-', ' ').capitalize()!r})" if anchor else ""
    )
    return (
        f"{why}. Before changing the kernel, read {where}: it lists the changes "
        "that address this, roughly in order of what they gain, and how to confirm each one "
        "in the next capture."
    )


def next_read_from_stats(stats: dict) -> str:
    """The page for what the stats CSV shows: its costliest instruction kind decides."""
    rows = stats.get("instructions") or []
    if not rows:
        return ""
    op = rows[0]["text"].split(" ", 1)[0]
    if op.startswith("s_barrier"):
        page, why = (
            "synchronization.md",
            "s_barrier holds the most time; `barriers` shows which wave of each "
            "workgroup the others wait for, which the CSV cannot",
        )
    elif op.startswith(("s_waitcnt", "s_wait_")):
        page, why = "latency.md", "wait instructions hold the most time"
    else:
        page, why = (
            "compute.md",
            f"{op} holds the most time; `pipes` shows which instruction class it belongs to",
        )
    return (
        f"{why}. Before changing the kernel, read {RESOURCES / page}: it lists the changes that address "
        "this, roughly in order of what they gain, and how to confirm each one in the next capture."
    )


def flat_profile_note(rows: list[dict], what: str) -> str:
    """A note for a hotspots or lines ranking whose top row holds a small share of the cost."""
    if not rows or rows[0]["share"] >= FLAT_PROFILE_SHARE:
        return ""
    return (
        f"no {what} holds more than {rows[0]['share']}% of the non-hidden cost; the cost is "
        "spread out. `pipes` shows which instruction classes it goes to."
    )


class Capture:
    def __init__(self, path: str | Path):
        try:
            from rocprof_trace_decoder import (
                CodeObject,
                Decoder,
                HiddenLatency,
                analyze_hidden_latency,
                generate_code_artifacts,
            )
        except ImportError as err:
            if err.name == "elftools":
                raise SystemExit(
                    "the decoder's Python package needs pyelftools: "
                    "pip install pyelftools"
                ) from None
            raise SystemExit(
                f"cannot import rocprof_trace_decoder ({err}); set PYTHONPATH to "
                "the decoder's python directory and ROCPROF_TRACE_DECODER_LIB to "
                "librocprof-trace-decoder.so"
            ) from None
        self.path = Path(path)
        att_files = sorted(self.path.rglob("*.att"))
        if not att_files:
            raise SystemExit(f"no .att files under {self.path}")
        objects = [
            CodeObject(p, int(re.search(r"_code_object_id_(\d+)", p.name).group(1)))
            for p in sorted(self.path.rglob("*_code_object_id_*.out"))
        ]
        if not objects:
            raise SystemExit(f"no *_code_object_id_*.out code objects under {self.path}")
        self.code_index = generate_code_artifacts(objects).code_index
        self.gpu = gpu_properties(self.path)
        self.waves: list[WaveTrace] = []
        self.dispatches: list = []
        self.dispatch_records: list[tuple[int, object]] = []
        self.occupancy: list[tuple[int, object]] = []
        self.warnings: set[str] = set()
        self.unresolved = 0
        self.hidden_by_pc = {}
        try:
            decoder = Decoder()
        except OSError as err:
            raise SystemExit(
                f"cannot load the decoder library ({err}); set "
                "ROCPROF_TRACE_DECODER_LIB to librocprof-trace-decoder.so (see the "
                "skill's resources/capture.md)"
            ) from None
        with decoder:
            for index, att in enumerate(att_files):
                match = re.search(r"_shader_engine_(\d+)_", att.name)
                se = int(match.group(1)) if match else index
                records = decoder.parse_file(att, isa=self.code_index)
                self.warnings |= {decoder.info_string(i) for i in records.info}
                self.dispatches += records.dispatches
                self.dispatch_records += [(se, d) for d in records.dispatches]
                self.occupancy += [(se, o) for o in records.occupancy]
                hidden = analyze_hidden_latency({se: records}, code_index=self.code_index)
                for pc, h in hidden.by_pc.items():
                    self.hidden_by_pc.setdefault(pc, HiddenLatency())
                    self.hidden_by_pc[pc] += h
                for w in records.waves:
                    self.waves.append(self._wave(se, w))

    @classmethod
    def from_waves(
        cls,
        waves: list[WaveTrace],
        dispatches: list | None = None,
        occupancy: list | None = None,
    ) -> "Capture":
        """A capture built from already-decoded records, for tests and custom pipelines.

        dispatches and occupancy are decoder Dispatch and Occupancy records, or (se, record)
        pairs."""

        def with_se(items):
            return [
                item if isinstance(item, tuple) else (0, item) for item in items or []
            ]

        cap = cls.__new__(cls)
        cap.path = None
        cap.code_index = None
        cap.gpu = {}
        cap.waves = list(waves)
        cap.dispatch_records = with_se(dispatches)
        cap.dispatches = [d for _, d in cap.dispatch_records]
        cap.occupancy = with_se(occupancy)
        cap.warnings = set()
        cap.unresolved = 0
        cap.hidden_by_pc = {}
        return cap

    def _wave(self, se: int, w) -> WaveTrace:
        from rocprof_trace_decoder import InstCategory, WaveStateType

        trace = WaveTrace(
            se,
            w.cu,
            w.simd,
            w.wave_id,
            w.workgroup_id,
            w.begin_time,
            w.end_time,
            w.contexts,
        )
        prev_end = w.begin_time
        for i in w.instructions:
            # Unresolved instructions are skipped the way CodeIndex.accumulate_wave skips them.
            if i.pc.code_object_id == 0 and i.pc.address == 0:
                self.unresolved += 1
                continue
            entry = self.code_index.entries.get(i.pc)
            inst = Inst(
                i.time,
                i.duration,
                i.stall,
                max(i.time - prev_end, 0),
                InstCategory(i.category).name,
                entry.inst if entry else "",
                entry.source if entry else "",
                i.pc,
            )
            prev_end = max(prev_end, i.time + inst.latency)
            trace.insts.append(inst)
        for state in w.timeline:
            trace.states[WaveStateType(state.type).name] += state.duration
        return trace

    # ------------------------------------------------------------------ summary

    def summary(self) -> dict:
        states = Counter()
        for w in self.waves:
            states += w.states
        total_state = sum(states.values()) or 1
        dispatch = self.dispatches[0] if self.dispatches else None
        rows = self.occupancy_rows()
        return {
            "waves": len(self.waves),
            "waves_with_context_switch": sum(w.contexts != 0 for w in self.waves),
            "warnings": sorted(self.warnings),
            "unresolved_instructions": self.unresolved,
            "wave_lifetime_median": (
                statistics.median(w.lifetime for w in self.waves) if self.waves else 0
            ),
            "wave_states": {
                k: round(100 * v / total_state, 1) for k, v in states.most_common()
            },
            "dispatch": (
                None
                if dispatch is None
                else {
                    "vgprs": dispatch.vgprs,
                    "sgprs": dispatch.sgprs,
                    "lds_bytes": dispatch.lds_size,
                    "workgroup": [
                        dispatch.thread_dim_x,
                        dispatch.thread_dim_y,
                        dispatch.thread_dim_z,
                    ],
                }
            ),
            "occupancy_simds": len({(se, o.cu, o.simd) for se, o in self.occupancy}),
            "peak_active_waves": max((r["active_waves"] for r in rows), default=0),
            "gpu": self.gpu,
        }

    # ------------------------------------------------------------------ hotspots

    def hotspots(self, top: int = 15, by_line: bool = False) -> list[dict]:
        """Instructions (or source lines) by non-hidden latency, as the decoder's
        hidden_latency_hotspots sample ranks them."""
        per_pc: dict[Pc, list[Inst]] = defaultdict(list)
        for w in self.waves:
            for inst in w.insts:
                per_pc[inst.pc].append(inst)
        rows = []
        for pc, insts in per_pc.items():
            h = self.hidden_by_pc.get(pc)
            latency = sum(i.latency for i in insts)
            idle = sum(i.idle for i in insts)
            hidden = h.stall + h.issue + min(h.idle, idle) if h else 0
            rows.append(
                {
                    "pc": pc,
                    "text": insts[0].text,
                    "source": short_source(insts[0].source),
                    "category": insts[0].category,
                    "hits": len(insts),
                    "cost": latency + idle,
                    "stall": sum(i.stall for i in insts),
                    "idle": idle,
                    "non_hidden": latency + idle - hidden,
                    "p50": pct([i.cost for i in insts], 0.5),
                    "p90": pct([i.cost for i in insts], 0.9),
                }
            )
        if by_line:
            grouped: dict[str, dict] = {}
            for r in rows:
                g = grouped.setdefault(
                    r["source"],
                    {
                        "line": r["source"],
                        "cost": 0,
                        "non_hidden": 0,
                        "hits": 0,
                        "stall": 0,
                        "idle": 0,
                        "top_inst": Counter(),
                    },
                )
                for k in ("cost", "non_hidden", "hits", "stall", "idle"):
                    g[k] += r[k]
                g["top_inst"][r["text"].split(" ", 1)[0]] += r["non_hidden"]
            rows = list(grouped.values())
            for g in rows:
                g["top_inst"] = ", ".join(f"{k}" for k, _ in g["top_inst"].most_common(3))
        total = sum(r["non_hidden"] for r in rows) or 1
        for r in rows:
            r["share"] = round(100 * r["non_hidden"] / total, 1)
        return sorted(rows, key=lambda r: -r["non_hidden"])[:top]

    # ------------------------------------------------------------------ pipes

    def pipes(self) -> dict:
        """Issue and stall cycles by instruction class, against the time the traced SIMDs had
        waves resident. Issue cycles are an instruction's duration minus its stall: the cycles
        it occupied its pipe's issue port once the pipe took it."""
        span: dict[tuple, list[int]] = {}
        issue, stall, count = Counter(), Counter(), Counter()
        for w in self.waves:
            s = span.setdefault((w.se, w.cu, w.simd), [w.begin, w.end])
            s[0], s[1] = min(s[0], w.begin), max(s[1], w.end)
            for i in w.insts:
                c = inst_class(i)
                count[c] += 1
                issue[c] += max(i.duration - i.stall, 0)
                stall[c] += i.stall
        simd_time = sum(max(end - begin, 1) for begin, end in span.values()) or 1
        total_stall = sum(stall.values()) or 1
        waves = len(self.waves) or 1
        classes = [
            {
                "class": c,
                "per_wave": round(count[c] / waves, 1),
                "cycles_each": round(issue[c] / count[c], 1),
                "issue_share": round(100 * issue[c] / simd_time, 1),
                "stall_share": round(100 * stall[c] / total_stall, 1),
            }
            for c in count
        ]
        classes.sort(key=lambda r: -(r["issue_share"] + r["stall_share"]))
        valu = sum(issue[c] for c in issue if c.startswith("VALU"))
        return {
            "waves": len(self.waves),
            "simds": len(span),
            "simd_cycles": simd_time,
            "valu_issue_share": round(100 * valu / simd_time, 1),
            "classes": classes,
        }

    # ------------------------------------------------------------------ barriers

    def barriers(self, top: int = 3) -> dict:
        """How the waves of a workgroup split their time around s_barrier. A workgroup's waves are
        launched together, so on each compute unit consecutive waves that start within LAUNCH_SLACK
        cycles of each other, up to the dispatch's waves per workgroup, form one workgroup; rank 0 is
        its first wave. In each workgroup the wave with the smallest share of its lifetime at
        s_barrier is the one the others waited for: the report gives which rank that is, its
        barrier share against the others', the lines it spent its time on, and the barriers the
        others waited at."""
        dispatch = self.dispatches[0] if self.dispatches else None
        wave_size = int(self.gpu.get("wave_front_size") or 64)
        per_wg = (
            max(
                1,
                -(
                    -dispatch.thread_dim_x
                    * dispatch.thread_dim_y
                    * dispatch.thread_dim_z
                    // wave_size
                ),
            )
            if dispatch
            else 0
        )
        by_cu: dict[tuple, list[WaveTrace]] = defaultdict(list)
        for w in self.waves:
            if any(i.text.startswith("s_barrier") for i in w.insts):
                by_cu[(w.se, w.cu)].append(w)
        groups: list[list[WaveTrace]] = []
        for waves in by_cu.values():
            waves.sort(key=lambda w: (w.begin, w.simd, w.wave_id))
            for w in waves:
                g = groups[-1] if groups else None
                if (
                    g
                    and (g[0].se, g[0].cu) == (w.se, w.cu)
                    and w.begin - g[0].begin <= LAUNCH_SLACK
                    and (not per_wg or len(g) < per_wg)
                ):
                    g.append(w)
                else:
                    groups.append([w])
        groups = [g for g in groups if len(g) > 1]

        def barrier_share(w: WaveTrace) -> float:
            return (
                100
                * sum(i.cost for i in w.insts if i.text.startswith("s_barrier"))
                / max(w.lifetime, 1)
            )

        ranks, least, others = Counter(), [], []
        ran, waited_at = Counter(), Counter()
        for g in groups:
            shares = [barrier_share(w) for w in g]
            r = min(range(len(g)), key=lambda k: shares[k])
            ranks[r] += 1
            least.append(shares[r])
            others += [x for k, x in enumerate(shares) if k != r]
            for i in g[r].insts:
                if not i.text.startswith("s_barrier"):
                    ran[short_source(i.source)] += i.cost
            for k, w in enumerate(g):
                if k != r:
                    for i in w.insts:
                        if i.text.startswith("s_barrier"):
                            waited_at[short_source(i.source)] += i.cost
        n = len(groups) or 1
        ran_total, wait_total = sum(ran.values()) or 1, sum(waited_at.values()) or 1
        return {
            "workgroups": len(groups),
            "waves_per_workgroup": per_wg or None,
            "least_waiting_rank": {
                f"rank {r}": f"{100 * c / n:.0f}%" for r, c in ranks.most_common(3)
            },
            "least_waiting_barrier_share_p50": round(pct(least, 0.5), 1),
            "others_barrier_share_p50": round(pct(others, 0.5), 1),
            "least_waiting_wave_ran": [
                {"line": k, "share": round(100 * v / ran_total, 1)}
                for k, v in ran.most_common(top)
            ],
            "others_waited_at": [
                {"line": k, "share": round(100 * v / wait_total, 1)}
                for k, v in waited_at.most_common(top)
            ],
        }

    # ------------------------------------------------------------------ lifetime

    def lifetime(self) -> dict:
        """Wave lifetime against s_wait, VALU, and other latency and idle time, as the decoder's
        plot_wave_lifetime sample computes them, with that sample's linear fit per component.
        """
        names = ("s_wait_latency", "valu_latency", "non_valu_latency", "idle_time")
        rows = []
        for w in self.waves:
            parts = dict.fromkeys(names, 0)
            prev = w.begin
            for i in w.insts:
                if i.text.strip().startswith("s_wait"):
                    parts["s_wait_latency"] += i.duration
                elif i.category == "VALU":
                    parts["valu_latency"] += i.duration
                else:
                    parts["non_valu_latency"] += i.duration
                parts["idle_time"] += max(i.time - prev, 0)
                prev = max(prev, i.time + i.duration)
            rows.append((w.lifetime, parts))
        lifetimes = [life for life, _ in rows]
        total = sum(lifetimes) or 1
        components = []
        for name in names:
            xs = [parts[name] for _, parts in rows]
            row = {
                "component": name,
                "share_of_lifetime": round(100 * sum(xs) / total, 1),
                "per_wave_p50": pct(xs, 0.5),
            }
            if len(set(xs)) > 1 and len(set(lifetimes)) > 1:
                row["lifetime_slope"] = round(
                    statistics.linear_regression(xs, lifetimes).slope, 2
                )
                row["r"] = round(statistics.correlation(xs, lifetimes), 2)
            components.append(row)
        return {
            "waves": len(rows),
            "lifetime_median": statistics.median(lifetimes) if rows else 0,
            "components": components,
        }

    # ------------------------------------------------------------------ occupancy

    def occupancy_rows(self) -> list[dict]:
        """Active waves and SGPR/VGPR allocation after each occupancy record, as the decoder's
        plot_occupancy_resources sample computes them."""
        events = [(d.time, 0, se, d) for se, d in self.dispatch_records]
        events += [(o.time, 1, se, o) for se, o in self.occupancy]
        events.sort(key=lambda e: (e[0], e[1]))
        resources: dict[tuple, tuple[int, int]] = {}
        active: dict[tuple, tuple[int, int]] = {}
        sgprs = vgprs = 0
        rows = []
        for time, kind, se, record in events:
            if kind == 0:
                resources[(se, record.me_id, record.pipe_id)] = (
                    record.sgprs,
                    record.vgprs,
                )
                continue
            key = (
                se,
                record.cu,
                record.simd,
                record.wave_id,
                record.me_id,
                record.pipe_id,
                record.workgroup_id,
            )
            wave_sgprs, wave_vgprs = resources.get(
                (se, record.me_id, record.pipe_id), (0, 0)
            )
            if record.start:
                if key not in active:
                    active[key] = (wave_sgprs, wave_vgprs)
                    sgprs += wave_sgprs
                    vgprs += wave_vgprs
            elif key in active:
                old_sgprs, old_vgprs = active.pop(key)
                sgprs -= old_sgprs
                vgprs -= old_vgprs
            rows.append(
                {
                    "time": int(time),
                    "active_waves": len(active),
                    "active_sgprs": sgprs,
                    "active_vgprs": vgprs,
                }
            )
        return rows

    def occupancy_report(self, bins: int = 20) -> dict:
        rows = self.occupancy_rows()
        simds = len({(se, o.cu, o.simd) for se, o in self.occupancy})
        if len(rows) < 2:
            return {"occupancy_records": len(self.occupancy), "occupancy_simds": simds}
        times = [r["time"] for r in rows]
        start, end = times[0], times[-1]
        span = max(end - start, 1)
        mean_waves = (
            sum(r["active_waves"] * (t1 - r["time"]) for r, t1 in zip(rows, times[1:]))
            / span
        )

        def sampled(key: str) -> list[int]:
            out = []
            for b in range(bins):
                t = start + (b + 0.5) * span / bins
                out.append(rows[max(bisect.bisect_right(times, t) - 1, 0)][key])
            return out

        return {
            "occupancy_records": len(self.occupancy),
            "occupancy_simds": simds,
            "peak_active_waves": max(r["active_waves"] for r in rows),
            "mean_active_waves": round(mean_waves, 2),
            "mean_active_waves_per_simd": round(mean_waves / max(simds, 1), 2),
            "active_waves_over_time": sampled("active_waves"),
            "active_vgprs_over_time": sampled("active_vgprs"),
            "active_sgprs_over_time": sampled("active_sgprs"),
        }

    # ------------------------------------------------------------------ compare

    def cost_by_line(self) -> dict[str, float]:
        per_line: dict[str, float] = defaultdict(float)
        for w in self.waves:
            for inst in w.insts:
                per_line[short_source(inst.source)] += inst.cost
        n = len(self.waves) or 1
        return {k: v / n for k, v in per_line.items()}


def read_stats(path: str | Path, top: int = 15) -> dict:
    """Rank the per-instruction stats_*.csv rocprofv3 writes next to a capture.

    Each row sums over the traced waves: Hitcount executions, Latency cycles (issue plus
    stall), Stall cycles, and Idle cycles before the instruction. Cost is Latency + Idle.
    """
    files = sorted(Path(path).rglob("stats_*.csv"))
    if not files:
        raise SystemExit(f"no stats_*.csv under {path}")
    insts = []
    for f in files:
        with f.open(newline="") as handle:
            for r in csv.DictReader(handle):
                text = r.get("Instruction", "")
                hits = int(float(r.get("Hitcount") or 0))
                if not hits or text.startswith(";"):
                    continue
                latency, stall, idle = (
                    int(float(r.get(k) or 0)) for k in ("Latency", "Stall", "Idle")
                )
                insts.append(
                    {
                        "cost": latency + idle,
                        "hits": hits,
                        "latency": latency,
                        "stall": stall,
                        "idle": idle,
                        "text": text,
                        "source": short_source(r.get("Source", "")),
                    }
                )
    total = sum(i["cost"] for i in insts) or 1
    lines: dict[str, dict] = {}
    for i in insts:
        i["share"] = round(100 * i["cost"] / total, 1)
        g = lines.setdefault(
            i["source"],
            {"line": i["source"], "cost": 0, "stall": 0, "idle": 0, "insts": Counter()},
        )
        g["cost"] += i["cost"]
        g["stall"] += i["stall"]
        g["idle"] += i["idle"]
        g["insts"][i["text"].split(" ", 1)[0]] += i["cost"]
    by_line = [
        {
            "share": round(100 * g["cost"] / total, 1),
            "cost": g["cost"],
            "stall": g["stall"],
            "idle": g["idle"],
            "line": g["line"],
            "top_inst": ", ".join(k for k, _ in g["insts"].most_common(2)),
        }
        for g in lines.values()
    ]
    return {
        "files": [f.name for f in files],
        "instructions": sorted(insts, key=lambda i: -i["cost"])[:top],
        "lines": sorted(by_line, key=lambda g: -g["cost"])[:top],
    }


def print_rows(rows: list[dict], columns: list[str]) -> None:
    if not rows:
        print("(none)")
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    widths = {c: min(w, 60) for c, w in widths.items()}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for r in rows:
        print("  ".join(str(r.get(c, ""))[: widths[c]].ljust(widths[c]) for c in columns))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("capture")
    p.add_argument(
        "command",
        choices=[
            "stats",
            "summary",
            "pipes",
            "hotspots",
            "lines",
            "barriers",
            "lifetime",
            "occupancy",
            "compare",
        ],
    )
    p.add_argument("other", nargs="?")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if args.command == "stats":
        result = read_stats(args.capture, args.top)
        if args.json:
            print(json.dumps(result, indent=1))
            return 0
        print("files:", ", ".join(result["files"]))
        print("\nInstructions by cost (latency + idle, summed over traced waves):")
        print_rows(
            result["instructions"],
            ["share", "cost", "hits", "latency", "stall", "idle", "text", "source"],
        )
        print("\nSource lines by cost:")
        print_rows(
            result["lines"], ["share", "cost", "stall", "idle", "line", "top_inst"]
        )
        hint = next_read_from_stats(result)
        if hint:
            print(f"\nnext: {hint}")
        return 0

    cap = Capture(args.capture)
    hint = ""
    if args.command in ("summary", "pipes", "hotspots", "lines"):
        hint = next_read(cap.summary()["wave_states"], cap.pipes())
    if args.command == "summary":
        result = cap.summary()
        result["next"] = hint
    elif args.command == "pipes":
        result = cap.pipes()
        result["next"] = hint
    elif args.command in ("hotspots", "lines"):
        result = cap.hotspots(args.top, by_line=args.command == "lines")
    elif args.command == "barriers":
        result = cap.barriers(args.top)
    elif args.command == "lifetime":
        result = cap.lifetime()
    elif args.command == "occupancy":
        result = cap.occupancy_report()
    else:
        if not args.other:
            p.error("compare needs a second capture directory")
        before, after = cap.cost_by_line(), Capture(args.other).cost_by_line()
        rows = [
            {
                "line": k,
                "before": round(before.get(k, 0)),
                "after": round(after.get(k, 0)),
                "delta": round(after.get(k, 0) - before.get(k, 0)),
            }
            for k in set(before) | set(after)
        ]
        result = sorted(rows, key=lambda r: -abs(r["delta"]))[: args.top]
        result.insert(
            0,
            {
                "line": "(per-wave total)",
                "before": round(sum(before.values())),
                "after": round(sum(after.values())),
                "delta": round(sum(after.values()) - sum(before.values())),
            },
        )

    if args.json:
        print(json.dumps(result, indent=1, default=str))
        return 0
    if args.command in ("summary", "occupancy"):
        for k, v in result.items():
            if k != "next":
                print(f"{k}: {v}")
    elif args.command == "pipes":
        print(
            f"waves: {result['waves']}  SIMDs: {result['simds']}  SIMD cycles with waves resident: "
            f"{result['simd_cycles']}"
        )
        print_rows(
            result["classes"],
            ["class", "per_wave", "cycles_each", "issue_share", "stall_share"],
        )
        print(
            f"\nvector instructions (all VALU classes) issue {result['valu_issue_share']}% of the "
            "SIMDs' time. Near or above 100%, the SIMDs issue a vector instruction almost every "
            "cycle: only issuing fewer or cheaper vector instructions shortens the kernel. Stall "
            "on a class is a busy pipe or a wait for an earlier result of the same kind (matrix "
            "and transcendental instructions take many cycles); with few waves resident it is "
            "usually the wait. Matrix instructions issue in a few cycles, so compare the "
            "kernel's FLOP/s with the GPU's matrix peak to see how busy the matrix core is."
        )
    elif args.command == "barriers":
        for k, v in result.items():
            if not isinstance(v, list):
                print(f"{k}: {v}")
        print("\nThe wave the others waited for spent its time on:")
        print_rows(result["least_waiting_wave_ran"], ["share", "line"])
        print("\nThe other waves waited at:")
        print_rows(result["others_waited_at"], ["share", "line"])
    elif args.command == "lifetime":
        print(
            f"waves: {result['waves']}  lifetime median: {result['lifetime_median']} cycles"
        )
        print_rows(
            result["components"],
            ["component", "share_of_lifetime", "per_wave_p50", "lifetime_slope", "r"],
        )
    elif args.command == "hotspots":
        print_rows(
            result,
            [
                "share",
                "non_hidden",
                "cost",
                "hits",
                "p50",
                "p90",
                "stall",
                "idle",
                "category",
                "text",
                "source",
            ],
        )
    elif args.command == "lines":
        print_rows(
            result,
            ["share", "non_hidden", "cost", "hits", "stall", "idle", "line", "top_inst"],
        )
    else:
        print_rows(result, ["line", "before", "after", "delta"])
    note = (
        flat_profile_note(
            result, "instruction" if args.command == "hotspots" else "source line"
        )
        if args.command in ("hotspots", "lines")
        else ""
    )
    if note:
        print(f"\nnote: {note}")
    if hint:
        print(f"\nnext: {hint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
