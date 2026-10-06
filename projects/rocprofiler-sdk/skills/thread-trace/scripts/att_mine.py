#!/usr/bin/env python3
"""Read a thread trace capture through the rocprof-trace-decoder Python API.

usage: att_mine.py <capture_dir> <command> [--top N] [--json]
       att_mine.py <capture_dir> compare <other_capture_dir> [--top N]

commands:
  stats      the stats_*.csv rocprofv3 writes, ranked by instruction and by source line
             (reads only the CSV; no decoder needed; the CSV names only the innermost
             source line of inlined code)
  summary    decoder warnings, waves, the traced dispatch's resources, wave-state split, idle
             between instructions, active waves
  hotspots   instructions ranked by non-hidden cost (latency plus idle, minus hidden), with
             per-hit p50/p90 of cost
  lines      the same ranking grouped by source line, with the number of distinct
             instructions each line ran
  pipes      per instruction class (VALU, matrix, LDS, memory, waits, ...): issue cycles per
             instruction (execution cycles on gfx10 and later), share of resident time issuing, and shares of stall and wait
             cycles; the opcodes that stalled most; and how much of the SIMDs' resident time
             VALU instructions (matrix included) were issuing
  barriers   how each workgroup's waves split their time around s_barrier: which wave
             waited least (often, not always, the one the others waited for), the
             barrier shares, and the lines that wave ran
  lifetime   wave lifetime against s_wait latency, VALU latency, other latency, and idle time
  occupancy  active waves and SGPR/VGPR allocation over time
  compare    cost per wave by source line in this capture vs another

stats, summary, pipes, hotspots, and lines end with a "next:" line naming the pages of the
skill's resources/ to read (for a flat stats profile it points to summary and pipes
instead; a capture with no waves gets none; with --json, only summary and pipes carry it,
as a "next" key). In hotspots and lines, the time wait, barrier, and other immediate
instructions waited is in "wait", not "stall" (in stats, for s_waitcnt, s_wait_*, and
s_barrier). barriers estimates workgroups from launch times. For your own queries, see the
skill's resources/python-api.md.

Needs Python 3.10 or later, pyelftools, llvm-objdump, and the decoder's Python package and
library, set up as in the skill's resources/capture.md (Setup).
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
        """Stall plus issue cycles: the decoder's duration, which includes the stall."""
        return self.duration

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
    dispatch: object = None  # the decoder Dispatch record the wave belongs to, if known
    # Identifies the dispatch across files: rocprofv3 writes one .att file per shader engine,
    # each with its own record of the same dispatch.
    dispatch_key: object = None

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
    "num_xcc",
)


def gpu_properties(path: Path) -> dict:
    """The traced GPU as rocprofv3 records it: rocpd_info_agent in *_results.db, or the
    *_agent_info.csv that --output-format csv writes. The .att file names carry the traced
    agent's handle (<pid>_<agent>_shader_engine_<se>_<n>.att), which selects its row."""
    handles = {
        int(m.group(1))
        for f in path.rglob("*.att")
        if (m := re.search(r"(\d+)_shader_engine_", f.name))
    }
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
            parsed = [(prod, nm, json.loads(ext or "{}")) for prod, nm, ext in rows]
            traced = [
                r for r in parsed if (r[2].get("id") or {}).get("handle") in handles
            ]
            product, name, info = (traced or parsed)[0]
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


def union_length(intervals: list) -> int:
    """Cycles covered by at least one of the (begin, end) intervals."""
    total, end = 0, None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += max(b - a, 0)
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def assign_dispatches(waves: list, dispatches: list, file_key: object = None) -> None:
    """Give each wave the last dispatch that started before it (dispatches: decoder records of
    the same trace file, in any order). With `file_key`, the files of one dispatch on several
    shader engines share a key, so the wave's dispatch_key is that key plus the dispatch's
    position in its file (consecutive kernels share a file)."""
    ordered = sorted(dispatches, key=lambda d: d.time)
    times = [d.time for d in ordered]
    for w in waves:
        k = bisect.bisect_right(times, w.begin) - 1
        w.dispatch = ordered[k] if k >= 0 else None
        if w.dispatch is not None:
            w.dispatch_key = (file_key, k) if file_key is not None else id(w.dispatch)


def waves_per_workgroup(dispatch, wave_size: int) -> int:
    if dispatch is None:
        return 0
    threads = dispatch.thread_dim_x * dispatch.thread_dim_y * dispatch.thread_dim_z
    return max(1, -(-threads // wave_size))


def short_source(source: str) -> str:
    """file:line without directories; for inlined code, the call site in your file first."""
    if not source:
        return "?"
    chain = [part.rsplit("/", 1)[-1] for part in source.split(" -> ")]
    return chain[-1] if len(chain) == 1 else f"{chain[-1]} (inlined {chain[0]})"


RESOURCES = Path(__file__).resolve().parents[1] / "resources"
# The prefixes the decoder's hidden-latency analysis treats as matrix instructions.
MATRIX_OPS = ("v_mfma", "v_smfma", "v_wmma", "v_swmma")
# The decoder's category past InstCategory.LAST: the record that ends a wave cut off.
WAVE_NOT_FINISHED = 14
FLAT_PROFILE_SHARE = 5.0  # heuristic: below this share, no single row stands out
WAIT_CLASSES = ("wait (s_waitcnt, s_wait_*)", "barrier (s_barrier)", "IMMED")
VALU_BUSY_ROUTE = 80.0  # heuristic: route to the ceiling check at this vector-busy share
GUARD = (
    "The trace shows where the time went in the code that ran, not which change will help; "
    "it does not show how the work would run with other instructions or another algorithm. "
    "Time a candidate change rather than ruling it out from the trace."
)
ALSO_READ_SHARE = 20.0  # heuristic: a wave state this large also gets its page named
FEW_WAVES_FRACTION = 0.5  # heuristic: below this fraction of the maximum waves per SIMD
LAUNCH_SLACK = 8  # cycles per wave within which the waves of one workgroup are launched


def inst_class(inst: Inst) -> str:
    """The instruction's decoder category, with matrix instructions split from other VALU
    instructions (as the decoder's hidden-latency analysis does) and the waits and barriers
    named."""
    op = inst.text.split(" ", 1)[0]
    if inst.category == "VALU":
        return "VALU matrix" if op.startswith(MATRIX_OPS) else "VALU"
    if op.startswith(("s_waitcnt", "s_wait_")):
        return "wait (s_waitcnt, s_wait_*)"
    if op.startswith("s_barrier"):
        return "barrier (s_barrier)"
    return inst.category


def is_wait_op(text: str) -> bool:
    """Wait and barrier instructions, whose latency is the time they waited. The stats CSV
    has no instruction category, so they are told apart by opcode."""
    return text.startswith(("s_waitcnt", "s_wait_", "s_barrier"))


def page_ref(page: str) -> str:
    name, _, anchor = page.partition("#")
    return f"{RESOURCES / name}" + (
        f" (section {anchor.replace('-', ' ').capitalize()!r})" if anchor else ""
    )


def page_lists(page: str) -> str:
    """What the page holds; a page named with a section is read from that section."""
    start = ", starting there; the page" if "#" in page else ": it"
    return (
        f"{start} lists possible causes, what each looks like in the trace, suggested "
        "changes, and how to confirm them."
    )


def wait_page(pipes: dict | None) -> tuple[str, str]:
    """latency.md for waits on earlier instructions, synchronization.md when s_barrier holds
    more of the waiting than the wait instructions."""
    waited = {
        c["class"]: c.get("wait_share", c.get("stall_share", 0))
        for c in (pipes or {}).get("classes", [])
    }
    if waited.get("barrier (s_barrier)", 0) > waited.get("wait (s_waitcnt, s_wait_*)", 0):
        return (
            "synchronization.md",
            "s_barrier waited longer than the wait instructions; `barriers` shows which wave "
            "waited least at s_barrier",
        )
    return "latency.md", ""


def next_read(
    states: dict,
    pipes: dict | None = None,
    idle_share: float = 0.0,
    waves: dict | None = None,
) -> str:
    """Which page of resources/ covers what the trace shows, and why. The decoder counts the
    idle cycles between a wave's instructions as EXEC; `idle_share` (from `summary`) moves them
    to IDLE, so that idle before instructions decides as much as the other states. The page for
    the largest state comes first; WAIT or idle time of at least ALSO_READ_SHARE, and few
    resident waves (`waves`, from `summary`), add their pages."""
    if not states:
        return ""
    raw_wait = states.get("WAIT", 0)
    raw_state = max(states, key=states.get)
    if idle_share:
        states = dict(states)
        states["EXEC"] = max(states.get("EXEC", 0) - idle_share, 0)
        states["IDLE"] = states.get("IDLE", 0) + idle_share
    state = max(states, key=states.get)
    # summary prints the states before idle is moved out of EXEC; say so when that changed
    # which state is largest.
    basis = (
        " (with idle between instructions counted apart from EXEC)"
        if state != raw_state
        else ""
    )
    classes = (pipes or {}).get("classes", [])
    held = {c["class"]: c.get("stall_share", 0) for c in classes}
    if state == "WAIT":
        page, detail = wait_page(pipes)
        why = f"WAIT is the largest wave state{basis}" + (
            f" and {detail}" if detail else ""
        )
    elif state == "IDLE":
        page, why = (
            "stalls.md#idle-cycles",
            "idle time before instructions holds the most wave time",
        )
    elif state == "EXEC":
        page, why = (
            "compute.md#ceiling-check",
            f"EXEC is the largest wave state{basis}: the waves spend more time issuing than in "
            "any other state",
        )
    else:
        page = "compute.md"
        why = f"{state} is the largest wave state{basis}"
        busy = [c for c in held if c not in WAIT_CLASSES and c != "MESSAGE" and held[c]]
        if state == "STALL" and busy:
            top = max(busy, key=held.get)
            ops = next(
                (c.get("top_stalled", "") for c in pipes["classes"] if c["class"] == top),
                "",
            )
            why += (
                f"; {top} instructions hold the largest share of the stall"
                + (f" (most stalled: {ops})" if ops else "")
                + ": their pipe did not accept them, usually because the unit was busy or its "
                "queue full"
            )
    if (
        pipes
        and pipes.get("valu_busy", 0) >= VALU_BUSY_ROUTE
        and state in ("STALL", "EXEC")
    ):
        why += (
            f"; VALU instructions were issuing during {pipes['valu_busy']}% of the SIMDs' "
            "resident time"
        )
        page = "compute.md#ceiling-check"
    also = []
    if raw_wait >= ALSO_READ_SHARE and state != "WAIT":
        also.append((wait_page(pipes)[0], f"WAIT is {raw_wait}% of wave time"))
    if idle_share >= ALSO_READ_SHARE and state != "IDLE":
        also.append(
            (
                "stalls.md#idle-cycles",
                f"idle between instructions is {idle_share}% of wave time",
            )
        )
    if (
        waves
        and waves.get("max")
        and waves.get("peak", 0) < FEW_WAVES_FRACTION * waves["max"]
    ):
        also.append(
            (
                "latency.md#few-waves",
                f"{waves['peak']} waves per SIMD on average at the busiest moment, of "
                f"{waves['max']}",
            )
        )
    if (
        pipes
        and pipes.get("valu_busy", 0) >= VALU_BUSY_ROUTE
        and state not in ("STALL", "EXEC")
    ):
        also.append(
            (
                "compute.md#ceiling-check",
                f"VALU instructions were issuing during {pipes['valu_busy']}% of the SIMDs' "
                "resident time",
            )
        )
    also = [(p, w) for p, w in also if p != page]
    text = f"{why}. Before changing the kernel, read {page_ref(page)}{page_lists(page)}"
    same = [(p, w) for p, w in also if p.partition("#")[0] == page.partition("#")[0]]
    other = [(p, w) for p, w in also if (p, w) not in same]
    for p, w in same:
        section = p.partition("#")[2].replace("-", " ").capitalize()
        text += f" See its section {section!r} too ({w})."
    if other:
        text += " Also read " + "; ".join(f"{page_ref(p)} ({w})" for p, w in other) + "."
    return f"{text} {GUARD}"


def next_read_from_stats(stats: dict) -> str:
    """The page for what the stats CSV shows: its costliest instruction kind decides."""
    rows = stats.get("instructions") or []
    if not rows:
        return ""
    op = rows[0]["text"].split(" ", 1)[0]
    if rows[0].get("share", 100) < FLAT_PROFILE_SHARE:
        return (
            f"no instruction holds more than {rows[0]['share']}% of the cost in the CSV; "
            f"`summary` and `pipes` show where the time goes. {GUARD}"
        )
    if op.startswith("s_endpgm"):
        # The decoder counts the time the wave takes to complete as the idle time of s_endpgm.
        return (
            f"{op} holds the most time, as the idle time of the wave completing, which is not "
            f"a stall; `summary` and `pipes` show where the time goes. {GUARD}"
        )
    if rows[0].get("idle", 0) > rows[0].get("latency", 0):
        page, why = (
            "stalls.md#idle-cycles",
            f"{op} holds the most time, mostly as idle cycles before it",
        )
    elif op.startswith("s_barrier"):
        page, why = (
            "synchronization.md",
            "s_barrier holds the most time; `barriers` shows which wave, by launch order, "
            "waited least at s_barrier, and in what share of the workgroups, which the CSV "
            "cannot",
        )
    elif op.startswith(("s_waitcnt", "s_wait_")):
        page, why = "latency.md", "wait instructions hold the most time"
    else:
        page, why = (
            "compute.md",
            f"{op} holds the most time; `pipes` shows how busy its class was",
        )
    return f"{why}. Before changing the kernel, read {page_ref(page)}{page_lists(page)} {GUARD}"


def mean_and_peak_per_simd(rows: list[dict], simds: int) -> dict:
    """Mean (over the span of the occupancy records) and peak active waves per SIMD."""
    if len(rows) < 2 or not simds:
        return {"mean": 0, "peak": 0}
    times = [r["time"] for r in rows]
    span = max(times[-1] - times[0], 1)
    mean = (
        sum(r["active_waves"] * (t1 - r["time"]) for r, t1 in zip(rows, times[1:])) / span
    )
    return {
        "mean": round(mean / simds, 1),
        "peak": round(max(r["active_waves"] for r in rows) / simds, 1),
    }


def flat_profile_note(rows: list[dict], what: str) -> str:
    """A note for a hotspots or lines ranking whose top row holds a small share of the cost."""
    if not rows or rows[0]["share"] >= FLAT_PROFILE_SHARE:
        return ""
    return (
        f"no {what} holds more than {rows[0]['share']}% of the non-hidden cost; `pipes` shows "
        "which instruction classes the issue, stall, and wait cycles go to."
    )


class Capture:
    def __init__(self, path: str | Path):
        try:
            from rocprof_trace_decoder import CodeObject, Decoder, generate_code_artifacts
        except ImportError as err:
            if err.name == "elftools":  # packages that import pyelftools eagerly
                raise SystemExit(
                    "the decoder's Python package needs pyelftools: pip install pyelftools"
                ) from None
            raise SystemExit(
                f"cannot import rocprof_trace_decoder ({err}); set PYTHONPATH to the "
                "decoder's python directory (see the skill's resources/capture.md)"
            ) from None
        try:
            from rocprof_trace_decoder import HiddenLatency, analyze_hidden_latency
        except ImportError:
            raise SystemExit(
                "this rocprof_trace_decoder package has no analyze_hidden_latency; use a "
                "release of the decoder's Python package that includes it"
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
        try:
            self.code_index = generate_code_artifacts(objects).code_index
        except ImportError as err:
            if "elftools" in str(err) or err.name == "elftools":
                raise SystemExit(
                    "the decoder's Python package needs pyelftools: pip install pyelftools"
                ) from None
            raise
        except ValueError as err:
            if "is used by both" in str(err):
                raise SystemExit(
                    f"{err} Each run numbers its code objects from 1: capture each run "
                    "into a new directory."
                ) from None
            raise
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
                file_waves = [self._wave(se, w) for w in records.waves]
                # <agent>_shader_engine_<se>_<n>.att: the same agent and n on every engine
                named = re.search(r"(\d+)_shader_engine_\d+_(\d+)\.att$", att.name)
                file_key = named.groups() if named else att.name
                assign_dispatches(file_waves, records.dispatches, file_key)
                self.waves += file_waves

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
        assign_dispatches(cap.waves, cap.dispatches)
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
            # Records with no pc are skipped the way CodeIndex.accumulate_wave skips them; trap
            # and context records (category CONTEXT) have none either, nor has the record that
            # ends a wave still running when the trace ended (category WAVE_NOT_FINISHED, which
            # InstCategory does not name); none of them is unresolved.
            if i.pc.code_object_id == 0 and i.pc.address == 0:
                self.unresolved += i.category not in (
                    InstCategory.CONTEXT,
                    WAVE_NOT_FINISHED,
                )
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
        traced = Counter(w.dispatch_key for w in self.waves if w.dispatch is not None)
        main_key = traced.most_common(1)[0][0] if traced else None
        dispatch = next(
            (w.dispatch for w in self.waves if w.dispatch_key == main_key),
            self.dispatches[0] if self.dispatches else None,
        )
        warnings = set(self.warnings)
        if len(traced) > 1:
            warnings.add(
                f"the traced waves belong to {len(traced)} dispatches (from a kernel regex that matches "
                "several kernels or no regex, --att-consecutive-kernels, an iteration range, "
                "--selected-regions, or several GPUs (limit them with --att-gpu-index)); these "
                "figures mix them (dispatch: the one with the most waves). For one kernel, "
                "capture into a new directory"
            )
        rows = self.occupancy_rows()
        simds = len({(se, o.cu, o.simd) for se, o in self.occupancy})
        return {
            "waves": len(self.waves),
            "waves_with_context_switch": sum(w.contexts != 0 for w in self.waves),
            # Every .att file under the directory is decoded together.
            "dispatches_with_traced_waves": len(traced),
            # Idle cycles between instructions, which the decoder's wave states count as EXEC
            # (the idle before a wave's first instruction is its IDLE state).
            "idle_between_instructions": round(
                100 * sum(i.idle for w in self.waves for i in w.insts[1:]) / total_state,
                1,
            ),
            "warnings": sorted(warnings),
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
            "occupancy_simds": simds,
            "peak_active_waves": max((r["active_waves"] for r in rows), default=0),
            # Active waves per SIMD that held a wave, against the most a SIMD can hold.
            "waves_per_simd": {
                **mean_and_peak_per_simd(rows, simds),
                "max": self.gpu.get("max_waves_per_simd"),
            },
            **self.scratch(),
            "gpu": self.gpu,
        }

    def scratch(self) -> dict:
        """scratch_* instructions (scratch memory: register spills or private arrays), if any
        ran: instructions per wave and their share of the waves' instruction cost."""
        insts = [i for w in self.waves for i in w.insts if i.text.startswith("scratch_")]
        if not insts:
            return {}
        total = sum(i.cost for w in self.waves for i in w.insts) or 1
        return {
            "scratch_instructions": {
                "per_wave": round(len(insts) / (len(self.waves) or 1), 1),
                "share_of_cost": round(100 * sum(i.cost for i in insts) / total, 1),
                "opcodes": ", ".join(
                    op
                    for op, _ in Counter(
                        i.text.split(" ", 1)[0] for i in insts
                    ).most_common(3)
                ),
            }
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
            stall = sum(i.stall for i in insts)
            waits = inst_class(insts[0]) in WAIT_CLASSES
            rows.append(
                {
                    "pc": pc,
                    "text": insts[0].text,
                    "source": short_source(insts[0].source),
                    "category": max(insts, key=lambda i: i.cost).category,
                    "hits": len(insts),
                    "cost": latency + idle,
                    # A wait or barrier instruction's stall is the time it waited.
                    "stall": 0 if waits else stall,
                    # A wait, barrier, or other immediate instruction's latency is the time it
                    # waited (on some architectures that wait is not in its stall).
                    "wait": latency if waits else 0,
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
                        "wait": 0,
                        "idle": 0,
                        "insts": 0,
                        "top_inst": Counter(),
                    },
                )
                for k in ("cost", "non_hidden", "hits", "stall", "wait", "idle"):
                    g[k] += r[k]
                g["insts"] += 1  # rows are per instruction (pc): distinct instructions
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
        """Per instruction class: instructions per wave, issue cycles per instruction, the share
        of the SIMDs' resident time during which the class was issuing, the share of stall
        cycles, and the opcodes that stalled most. Issue cycles are an instruction's duration
        minus its stall. Instructions of different waves can issue in the same cycles, so busy
        time is the union of their issue intervals on each SIMD, counted once; resident time is
        the union of the traced waves' lifetimes on each SIMD."""
        lifetimes: dict[tuple, list] = defaultdict(list)
        issuing: dict[tuple, list] = defaultdict(list)
        issue, stall, waited, count = Counter(), Counter(), Counter(), Counter()
        stalled_ops: dict[str, Counter] = defaultdict(Counter)
        for w in self.waves:
            simd = (w.se, w.cu, w.simd)
            lifetimes[simd].append((w.begin, w.end))
            for i in w.insts:
                c = inst_class(i)
                count[c] += 1
                issue[c] += max(i.duration - i.stall, 0)
                stall[c] += i.stall
                if c in WAIT_CLASSES:
                    waited[c] += i.latency
                stalled_ops[c][i.text.split(" ", 1)[0]] += i.stall
                issuing[(c, simd)].append((i.time + i.stall, i.time + i.duration))
        resident = sum(union_length(v) for v in lifetimes.values())
        busy = Counter()
        valu: dict[tuple, list] = defaultdict(list)
        for (c, simd), spans in issuing.items():
            busy[c] += union_length(spans)
            if c.startswith("VALU"):
                valu[simd] += spans
        # Wait, barrier, and other immediate instructions spend their latency waiting; keep it
        # apart from the stall of instructions their pipe did not accept.
        total_stall = sum(v for c, v in stall.items() if c not in WAIT_CLASSES) or 1
        total_wait = sum(waited.values()) or 1
        waves = len(self.waves) or 1
        classes = [
            {
                "class": c,
                "per_wave": round(count[c] / waves, 1),
                "cycles_each": round(issue[c] / count[c], 1),
                "busy_share": round(100 * busy[c] / (resident or 1), 1),
                "stall_share": (
                    0.0 if c in WAIT_CLASSES else round(100 * stall[c] / total_stall, 1)
                ),
                "wait_share": (
                    round(100 * waited[c] / total_wait, 1) if c in WAIT_CLASSES else 0.0
                ),
                "top_stalled": (
                    ""
                    if c in WAIT_CLASSES
                    else ", ".join(op for op, n in stalled_ops[c].most_common(3) if n)
                ),
            }
            for c in count
        ]
        classes.sort(
            key=lambda r: -(r["busy_share"] + r["stall_share"] + r["wait_share"])
        )
        return {
            "waves": len(self.waves),
            "simds": len(lifetimes),
            "resident_cycles": resident,
            "valu_busy": round(
                100 * sum(union_length(v) for v in valu.values()) / (resident or 1), 1
            ),
            "classes": classes,
            **self.scratch(),
        }

    # ------------------------------------------------------------------ barriers

    def barriers(self, top: int = 3) -> dict:
        """How the waves of a workgroup split their time around s_barrier. A workgroup's
        waves are launched together, so on each compute unit and dispatch, consecutive
        waves that start within LAUNCH_SLACK cycles per wave (at least 8 waves' worth)
        of the workgroup's first wave, up to the dispatch's waves per workgroup, form
        one workgroup (an estimate: the trace's workgroup ids are not reliable). Ranks
        are launch order, waves launched in the same cycle ordered by SIMD. In each
        workgroup the wave with the smallest share of its lifetime at s_barrier (often,
        not always, the one the others waited for) is reported: which rank that is, its barrier
        share against the others', the lines it spent its time on, and the barriers the
        others waited at."""
        wave_size = int(self.gpu.get("wave_front_size") or 64)
        fallback = self.dispatches[0] if self.dispatches else None

        def per_wg(w: WaveTrace) -> int:
            return waves_per_workgroup(w.dispatch or fallback, wave_size)

        by_cu: dict[tuple, list[WaveTrace]] = defaultdict(list)
        for w in self.waves:
            if any(i.text.startswith("s_barrier") for i in w.insts):
                by_cu[(w.se, w.cu, id(w.dispatch))].append(w)
        groups: list[list[WaveTrace]] = []
        for waves in by_cu.values():
            waves.sort(key=lambda w: (w.begin, w.simd, w.wave_id))
            for w in waves:
                g = groups[-1] if groups else None
                if (
                    g
                    and (g[0].se, g[0].cu, id(g[0].dispatch))
                    == (w.se, w.cu, id(w.dispatch))
                    and w.begin - g[0].begin <= LAUNCH_SLACK * max(per_wg(w), 8)
                    and (not per_wg(w) or len(g) < per_wg(w))
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
            "waves_per_workgroup": (
                Counter(len(g) for g in groups).most_common(1)[0][0] if groups else None
            ),
            "ranks": "launch order within the workgroup (estimated from launch times)",
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
        plot_occupancy_resources sample computes them, except that a wave's start and end are
        matched by its slot alone: workgroup_id is not reliable on gfx9, and a mismatch left
        waves active for good."""
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
                waits = is_wait_op(text)
                insts.append(
                    {
                        "cost": latency + idle,
                        "hits": hits,
                        "latency": latency,
                        "stall": 0 if waits else stall,
                        "wait": latency if waits else 0,
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
            {
                "line": i["source"],
                "cost": 0,
                "stall": 0,
                "wait": 0,
                "idle": 0,
                "count": 0,
                "insts": Counter(),
            },
        )
        for k in ("cost", "stall", "wait", "idle"):
            g[k] += i[k]
        g["count"] += 1
        g["insts"][i["text"].split(" ", 1)[0]] += i["cost"]
    by_line = [
        {
            "share": round(100 * g["cost"] / total, 1),
            "cost": g["cost"],
            "stall": g["stall"],
            "wait": g["wait"],
            "idle": g["idle"],
            "insts": g["count"],
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
            [
                "share",
                "cost",
                "hits",
                "latency",
                "stall",
                "wait",
                "idle",
                "text",
                "source",
            ],
        )
        print("\nSource lines by cost:")
        print_rows(
            result["lines"],
            ["share", "cost", "stall", "wait", "idle", "insts", "line", "top_inst"],
        )
        hint = next_read_from_stats(result)
        if hint:
            print(f"\nnext: {hint}")
        return 0

    cap = Capture(args.capture)
    hint = ""
    if args.command in ("summary", "pipes", "hotspots", "lines"):
        summ = cap.summary()
        hint = next_read(
            summ["wave_states"],
            cap.pipes(),
            summ["idle_between_instructions"],
            summ["waves_per_simd"],
        )
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

        def source_files(lines: dict) -> set:
            return {k.split(" (inlined", 1)[0].rsplit(":", 1)[0] for k in lines}

        if not source_files(before) & source_files(after):
            print("warning: the two captures share no source files", file=sys.stderr)
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
            f"waves: {result['waves']}  SIMDs: {result['simds']}  resident cycles (time with a "
            f"traced wave on the SIMD, summed over SIMDs): {result['resident_cycles']}"
        )
        print_rows(
            result["classes"],
            [
                "class",
                "per_wave",
                "cycles_each",
                "busy_share",
                "stall_share",
                "wait_share",
                "top_stalled",
            ],
        )
        print(
            "\nbusy_share: share of the SIMDs' resident time the class was issuing (overlapping "
            "issue counted once); cycles_each: mean cycles per instruction after its stall (issue "
            "cycles on gfx9; on gfx10 and later execution cycles, mostly a fixed number by "
            "instruction type rather than measured); stall_share: share of the "
            "stall cycles of all classes but wait, barrier, and IMMED, the cycles the pipe did not accept the "
            "class's instructions, usually "
            "because the unit was busy or its queue full; wait_share: share of the latency of "
            "wait, barrier, and other immediate instructions."
            + (
                ""  # the next: line below already gives it, or there are no waves
                if "VALU instructions were issuing" in hint or not result["waves"]
                else " VALU instructions (matrix included) were issuing during "
                f"{result['valu_busy']}% of the resident time "
                f"({page_ref('compute.md#ceiling-check')})."
            )
            + f" Measures: {page_ref('stalls.md')}."
        )
        if result.get("scratch_instructions"):
            sc = result["scratch_instructions"]
            print(
                f"scratch_* instructions (scratch memory: register spills or private arrays) "
                f"ran: {sc['per_wave']} per wave ({sc['opcodes']}), {sc['share_of_cost']}% of "
                "the instruction cost (the waits for scratch loads count with the wait "
                "instructions)."
            )
    elif args.command == "barriers":
        for k, v in result.items():
            if not isinstance(v, list):
                print(f"{k}: {v}")
        print("\nThe wave that waited least at s_barrier spent its time on:")
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
                "wait",
                "idle",
                "category",
                "text",
                "source",
            ],
        )
    elif args.command == "lines":
        print_rows(
            result,
            [
                "share",
                "non_hidden",
                "cost",
                "hits",
                "stall",
                "wait",
                "idle",
                "insts",
                "line",
                "top_inst",
            ],
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
