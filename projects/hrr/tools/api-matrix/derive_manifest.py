#!/usr/bin/env python3
"""
derive_manifest.py — machine-derive the HRR API classification table.

HRR's code generator (`projects/hrr/tools/gen_hrr_api_args.py`) is the single
source of truth for how every HIP API behaves at replay: it holds the
MANUAL_PLAYBACK_APIS / NOOP_PLAYBACK_APIS / ERROR_STUB_PLAYBACK_APIS /
CUSTOM_PLAYBACK_BODIES sets, and it parses the API list out of
`hip_api_trace.hpp`.  Rather than transcribe those lists into a test fixture —
which would silently rot the first time upstream moves an API between sets —
this script imports the generator as a module and asks it directly.

Output is `api_classes.json`: one row per API with its capture class, its
replay class, and the mechanically-detected payload-loss shape.

Baseline counts are asserted so that an upstream change to the generator
surfaces here as a hard failure with a diff, not as a silently shrinking test
matrix.  Refresh them deliberately with --update-baseline.

Runs anywhere Python 3.8+ is available; needs no ROCm, no GPU and no build.

Usage:
    ./derive_manifest.py [--rocm-systems DIR] [-o api_classes.json]
    ./derive_manifest.py --update-baseline     # after a deliberate upstream change
    ./derive_manifest.py --print-payload-loss  # show the payload-loss candidates
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Baseline the matrix was authored against.
#
# From HRR-Use-Case-Priorities.md: "551 APIs: 542 runtime + 9 compiler; 273
# classified faithfully replayed, 254 NOOP_PLAYBACK_APIS, 24
# ERROR_STUB_PLAYBACK_APIS" at develop @ 2d68481fc8f.  The numbers below are
# what the *current* tree actually produces; where they differ from the
# document, the tree wins and the delta is worth reading before you refresh it.
# ---------------------------------------------------------------------------
BASELINE_PATH_NAME = "api_classes_baseline.json"

# This script ships inside rocm-systems, so the checkout it describes is by
# default the one containing it: projects/hrr/tools/api-matrix -> repo root.
DEFAULT_ROCM_SYSTEMS = Path(__file__).resolve().parents[4]

# Replay classes, in the priority order generate_playback_shim() applies them.
REPLAY_UNREPLAYABLE = "UNREPLAYABLE"
REPLAY_ERROR_STUB = "ERROR_STUB"
REPLAY_NOOP = "NOOP"
REPLAY_CUSTOM = "CUSTOM"
REPLAY_MANUAL = "MANUAL"
REPLAY_COMPILER = "COMPILER_NOOP"
REPLAY_GENERATED = "GENERATED"
# Handled by is_special() in hrr_playback.cpp, ahead of the dispatch table, so
# whatever the generator emitted for them is dead code.
REPLAY_SPECIAL = "SPECIAL"

# What an observer of a replay can actually distinguish.  NOOP, ERROR_STUB and
# UNREPLAYABLE each emit a one-time stderr warning of their own; everything else
# is silent, so from the outside CUSTOM / MANUAL / GENERATED / COMPILER_NOOP are
# one class.
OBSERVABLE = {
    REPLAY_UNREPLAYABLE: "UNREPLAYABLE",
    REPLAY_ERROR_STUB: "ERROR_STUB",
    REPLAY_NOOP: "NOOP",
    REPLAY_CUSTOM: "REAL",
    REPLAY_MANUAL: "REAL",
    REPLAY_COMPILER: "REAL",
    REPLAY_GENERATED: "REAL",
    REPLAY_SPECIAL: "REAL",
}

CAPTURE_MANUAL = "MANUAL"
CAPTURE_PASSTHROUGH = "PASSTHROUGH_ONLY"
CAPTURE_GENERATED = "GENERATED"

# Pointer-to-scalar types that are genuinely serialisable or genuinely opaque,
# and so are not evidence of the CONST_STRUCT_PTR_IN payload-loss shape.
_SCALARISH_POINTEES = {
    "void", "char", "int", "unsigned", "float", "double", "size_t",
    "uint32_t", "uint64_t", "int32_t", "int64_t", "uint8_t", "int8_t",
    "uint16_t", "int16_t", "hipDeviceptr_t", "hipStream_t", "hipEvent_t",
    "hipFunction_t", "hipModule_t", "hipGraph_t", "hipGraphExec_t",
    "hipGraphNode_t", "hipArray_t", "hipMemPool_t",
}

# normalise_field_type() emits `uint64_t /* <type> */` for anything it does not
# recognise, which is mostly by-value structs but also catches spellings of
# plain scalars and enums it has no entry for (`unsigned char`, `enum hipLimit_t`).
# Those lose nothing — they fit in the uint64_t. Only a genuine composite does.
_SCALAR_KEYWORDS = re.compile(
    r"\b(char|short|int|long|float|double|signed|unsigned|bool|enum|_Bool)\b")


def load_generator(tools_dir: Path):
    """Import gen_hrr_api_args.py as a module.

    Everything interesting is module-level; main() is guarded by
    __name__ == "__main__", so importing has no side effects.
    """
    gen_path = tools_dir / "gen_hrr_api_args.py"
    if not gen_path.is_file():
        sys.exit(f"error: generator not found: {gen_path}")
    spec = importlib.util.spec_from_file_location("hrr_gen", gen_path)
    if spec is None or spec.loader is None:
        sys.exit(f"error: cannot load {gen_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_IS_SPECIAL_RE = re.compile(
    r"static\s+bool\s+is_special\s*\([^)]*\)\s*\{(.*?)\n\}", re.S)
_CASE_RE = re.compile(r"case\s+(HRR_API_[A-Z0-9_]+)\s*:")


def load_special_apis(hrr_dir: Path, api_names: List[str]) -> set:
    """APIs hrr_playback.cpp handles before the generated dispatch table.

    is_special() short-circuits dispatch_event(), so for these the generated
    handler never runs. That matters because some of them are NOOP in the
    generator and REAL in practice: hipModuleUnload has a NOOP handler that
    would warn, but the special path removes the module from the map and
    returns, so nothing is printed and the API looks faithful. Reading the
    switch is what keeps the manifest describing the binary rather than the
    generator's intent.
    """
    src = hrr_dir / "playback" / "hrr_playback.cpp"
    if not src.is_file():
        sys.exit(f"error: playback driver not found: {src}")
    body = _IS_SPECIAL_RE.search(src.read_text())
    if body is None:
        sys.exit(f"error: could not find is_special() in {src}; the manifest "
                 "cannot tell which APIs bypass the dispatch table")

    by_enum = {f"HRR_API_{name.upper()}": name for name in api_names}
    specials = set()
    unknown = []
    for enum in _CASE_RE.findall(body.group(1)):
        if enum in by_enum:
            specials.add(by_enum[enum])
        else:
            unknown.append(enum)
    if unknown:
        sys.exit(f"error: is_special() names {', '.join(sorted(unknown))}, "
                 "which do not map to any API in the trace table")
    if not specials:
        sys.exit("error: is_special() parsed but matched no APIs")
    return specials


def classify_replay(gen, entry, specials: set) -> str:
    """Mirror generate_playback_shim()'s dispatch order exactly."""
    if entry.name in specials:
        return REPLAY_SPECIAL
    if entry.name in getattr(gen, "UNREPLAYABLE_PLAYBACK_APIS", {}):
        return REPLAY_UNREPLAYABLE
    if entry.name in gen.ERROR_STUB_PLAYBACK_APIS:
        return REPLAY_ERROR_STUB
    if entry.name in gen.NOOP_PLAYBACK_APIS:
        return REPLAY_NOOP
    if entry.name in gen.CUSTOM_PLAYBACK_BODIES:
        return REPLAY_CUSTOM
    if entry.name in gen.MANUAL_PLAYBACK_APIS:
        return REPLAY_MANUAL
    if entry.table == "compiler":
        return REPLAY_COMPILER
    return REPLAY_GENERATED


def classify_capture(gen, entry) -> str:
    if entry.name in gen.MANUAL_CAPTURE_APIS:
        return CAPTURE_MANUAL
    if entry.name in gen.PASSTHROUGH_ONLY:
        return CAPTURE_PASSTHROUGH
    return CAPTURE_GENERATED


def _pointee_base(raw_type: str) -> str:
    t = re.sub(r"\b(const|volatile|restrict)\b", " ", raw_type)
    t = t.replace("*", " ")
    parts = t.split()
    return parts[0] if parts else ""


def detect_payload_loss(gen, entry, capture_class: str) -> List[Dict[str, str]]:
    """Detect the two payload-loss shapes described in section 8.3.

    CONST_STRUCT_PTR_IN — a `const Struct*` input argument.  The generator
      lowers every pointer to a bare uint64_t, so what lands in the archive is
      a capture-time host virtual address; the struct's bytes are never
      serialised and the address is meaningless at replay.

    BIG_STRUCT_BY_VALUE — a struct larger than 8 bytes passed by value, lowered
      to a single uint64_t field.  normalise_field_type() marks exactly this
      case by emitting `uint64_t /* <type> */`.

    Both are properties of the *generated* capture shim.  An API in
    MANUAL_CAPTURE_APIS has a hand-written shim that exists precisely to
    serialise what the generator would drop, so it is excluded — that is why
    hipMemcpy3D and hipArrayCreate, which take `const Struct*`, are not losses.

    Two generator mechanisms cancel a loss, and both are consulted here rather
    than assumed: a parameter listed in DEREF_FIELDS has its pointee copied into
    the event, and a by-value type listed in BY_VALUE_STRUCTS is carried as
    inline bytes (normalise_field_type returns the "__BYVAL__" sentinel for it
    instead of the `uint64_t /* T */` that marks a dropped struct).

    These are *candidates*.  The authoritative per-API marking lives in
    api_matrix.yaml, because whether a loss is consequential depends on how the
    argument is used, which no amount of type inspection reveals.
    """
    if capture_class != CAPTURE_GENERATED:
        return []

    carried = getattr(gen, "deref_covered_params", lambda _a: set())(entry.name)

    found: List[Dict[str, str]] = []
    for p in entry.params:
        raw = p.raw_type.strip()
        if p.name and p.name in carried:
            continue
        norm = gen.normalise_field_type(raw)
        if norm == "__BYVAL__":
            continue

        if norm.startswith("uint64_t /*"):
            if _SCALAR_KEYWORDS.search(raw):
                continue  # unrecognised scalar spelling, not a composite
            found.append({
                "shape": "BIG_STRUCT_BY_VALUE",
                "param": p.name or "(unnamed)",
                "type": raw,
            })
            continue

        if "*" not in raw or "const" not in raw:
            continue
        # A pointer-to-pointer is an output handle slot, not an input struct.
        if raw.count("*") > 1:
            continue
        base = _pointee_base(raw)
        if base in _SCALARISH_POINTEES:
            continue
        found.append({
            "shape": "CONST_STRUCT_PTR_IN",
            "param": p.name or "(unnamed)",
            "type": raw,
        })
    return found


def build_rows(gen, entries, specials: set) -> List[Dict[str, Any]]:
    rows = []
    for e in entries:
        replay = classify_replay(gen, e, specials)
        capture = classify_capture(gen, e)
        payload_loss = detect_payload_loss(gen, e, capture)
        rows.append({
            "api": e.name,
            "table": e.table,
            "ret_type": e.ret_type,
            "params": [{"type": p.raw_type, "name": p.name} for p in e.params],
            "capture_class": capture,
            "replay_class": replay,
            "observable_class": OBSERVABLE[replay],
            "payload_loss": payload_loss,
        })
    rows.sort(key=lambda r: r["api"])
    return rows


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {"total": len(rows)}
    for key in ("table", "capture_class", "replay_class", "observable_class"):
        for r in rows:
            counts[f"{key}.{r[key]}"] = counts.get(f"{key}.{r[key]}", 0) + 1
    counts["payload_loss.any"] = sum(1 for r in rows if r["payload_loss"])
    for shape in ("CONST_STRUCT_PTR_IN", "BIG_STRUCT_BY_VALUE"):
        counts[f"payload_loss.{shape}"] = sum(
            1 for r in rows if any(p["shape"] == shape for p in r["payload_loss"]))
    # The decision-relevant number from section 8.3: APIs that count toward the
    # "faithfully replayed" headline yet cannot actually be replayed.
    counts["payload_loss.in_faithful"] = sum(
        1 for r in rows if r["payload_loss"] and r["observable_class"] == "REAL")
    return counts


def compare_baseline(counts: Dict[str, int],
                     baseline: Optional[Dict[str, int]]) -> List[str]:
    if baseline is None:
        return []
    problems = []
    for key in sorted(set(counts) | set(baseline)):
        want = baseline.get(key)
        got = counts.get(key)
        if want != got:
            problems.append(f"  {key}: baseline={want} current={got}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rocm-systems", type=Path,
                    default=DEFAULT_ROCM_SYSTEMS,
                    help="rocm-systems checkout "
                         f"(default: {DEFAULT_ROCM_SYSTEMS})")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output JSON (default: api_classes.json beside this script)")
    ap.add_argument("--update-baseline", action="store_true",
                    help="rewrite the baseline counts from the current tree")
    ap.add_argument("--print-payload-loss", action="store_true",
                    help="list the payload-loss candidates and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    out_path = args.output or (here / "api_classes.json")
    baseline_path = here / BASELINE_PATH_NAME

    # The generator lives under tools/ and the playback driver under playback/,
    # so this is the project root rather than one directory holding both.
    hrr_dir = args.rocm_systems / "projects" / "hrr"
    trace_hpp = (args.rocm_systems / "projects" / "clr" / "hipamd" / "include" /
                 "hip" / "amd_detail" / "hip_api_trace.hpp")
    if not trace_hpp.is_file():
        sys.exit(f"error: hip_api_trace.hpp not found: {trace_hpp}")

    gen = load_generator(hrr_dir / "tools")
    entries = gen.parse_hip_api_trace(trace_hpp)
    specials = load_special_apis(hrr_dir, [e.name for e in entries])
    rows = build_rows(gen, entries, specials)
    counts = summarise(rows)

    if args.print_payload_loss:
        for r in rows:
            for pl in r["payload_loss"]:
                print(f"{pl['shape']:22s} {r['api']:48s} {pl['type']} {pl['param']}")
        return 0

    if args.update_baseline:
        baseline_path.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")
        print(f"baseline written: {baseline_path}")

    baseline = None
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text())

    manifest = {
        "source": {
            "rocm_systems": str(args.rocm_systems),
            "generator": str(hrr_dir / "tools" / "gen_hrr_api_args.py"),
            "hip_api_trace": str(trace_hpp),
        },
        "counts": counts,
        "apis": rows,
    }
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if not args.quiet:
        print(f"wrote {out_path}  ({counts['total']} APIs)")
        for key in sorted(counts):
            if key != "total":
                print(f"  {key:44s} {counts[key]}")

    problems = compare_baseline(counts, baseline)
    if problems:
        print("\nERROR: HRR API classification drifted from the recorded baseline.\n"
              "The matrix in api_matrix.yaml was authored against the baseline, so\n"
              "these APIs may now have an expected verdict that no longer matches\n"
              "reality. Review the delta, update api_matrix.yaml, then re-run with\n"
              "--update-baseline.\n", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1

    if baseline is None:
        print("\nnote: no baseline recorded yet; run --update-baseline to pin these counts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
