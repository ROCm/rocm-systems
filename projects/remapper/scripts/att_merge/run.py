#!/usr/bin/env python3
"""Extract, align, merge and audit verified multi-SIMD ATT replays (Python 3 + numpy)."""

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def assemble_summary(
    alignment, partner, audit, barriers, phase, rcv_dirs, sequence_check=True
):
    same = audit["same_work"]
    merged = audit["merged"]
    other_checks = (
        same["se_simd_pairs"] > 0
        and same["waves"] > 0
        and same["unidentified_waves"] == 0
        and same["same_wave_identities_every_sweep"] == same["se_simd_pairs"]
        and bool(merged)
        and all(
            m["unidentified_waves"] == 0 and m["waves_with_timestamp_problems"] == 0
            for m in merged
        )
    )
    sequences_match = (
        same["identical_instruction_sequence_every_sweep"] == same["waves"]
    )
    valid = other_checks and (sequences_match or not sequence_check)
    reason = None
    if other_checks and not sequences_match and sequence_check:
        reason = (
            f'{same["waves"] - same["identical_instruction_sequence_every_sweep"]} of {same["waves"]} traced '
            "waves executed different instruction sequences between sweeps (wave identities and every other "
            "check passed; spin-wait loops do this). Rerun with --no-sequence-check to accept the merge"
        )
    elif not other_checks:
        reason = "ATT audit failed: wave identities, coverage or timestamps differ"
    pair = partner["methods"].get("same sweep | occupancy + barriers", {})
    return dict(
        valid=valid,
        reason=reason,
        sequence_check=(
            "passed"
            if sequences_match
            else (
                "failed"
                if sequence_check
                else "failed, accepted by --no-sequence-check"
            )
        ),
        alignment=alignment["methods"].get("occupancy + barriers", {}),
        partner_simd=dict(lds=pair.get("category 5"), vmem=pair.get("category 3")),
        same_work=same,
        partner_stream_coverage=partner["checks"],
        merged=merged,
        same_role_barriers=barriers,
        phase=phase,
        rcv_dirs=rcv_dirs,
        note="Composite of verified replays; alignment errors use held-out events, in shader clocks. Partner errors score equal-length streams only; coverage lists exclusions, including variable prologue leader paths.",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--traces", required=True, type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument(
        "--original",
        type=Path,
        help="ATT traces of the unmodified kernel (no identity markers)",
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--resources", type=Path)
    ap.add_argument(
        "--no-sequence-check",
        action="store_true",
        help="accept traced waves whose instruction sequences differ between sweeps",
    )
    ap.add_argument(
        "--sparse",
        action="store_true",
        help="sparse launch: only shader engines whose traced WGP held a real workgroup are merged",
    )
    ap.add_argument(
        "--admission",
        action="store_true",
        help="admission launch: parked workgroups (A2 marker) on the traced WGPs are left out",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = read(args.manifest)
    env = dict(
        os.environ,
        REMAPPER_MERGE_WAVES=str(manifest["waves_per_workgroup"]),
        REMAPPER_MERGE_TARGET_CU=str(
            (manifest.get("cluster") or {}).get("traced_wgp", 0)
        ),
        REMAPPER_MERGE_PROLOGUE_VADDR=str(manifest["rewrite"]["prologue_vaddr"]),
        REMAPPER_MERGE_PROLOGUE_BYTES=str(manifest["rewrite"]["prologue_bytes"]),
    )
    # Infer the target from the single traced CU directory for non-cluster manifests.
    cells = sorted(args.traces.glob("sweep-*/cu-*-simd-0-attempt-0"))
    if not cells:
        raise ValueError("no ATT cells")
    env["REMAPPER_MERGE_TARGET_CU"] = cells[0].name.split("-")[1]
    if args.resources:
        env["RCV_BROWSER_RESOURCES"] = str(args.resources)
    expected = None
    if args.sparse:
        expected = sorted({loc // 32 for loc in manifest["launch"]["traced_locations"]})
        if not expected:
            raise ValueError("sparse manifest has no traced locations")
        env["REMAPPER_MERGE_ENGINES"] = ",".join(map(str, expected))
    if args.admission:
        env["REMAPPER_MERGE_DROP_FAKES"] = "1"

    def run(script, *values):
        with (args.out / f"{script}.log").open("a") as log:
            subprocess.run(
                [sys.executable, str(HERE / f"{script}.py"), *map(str, values)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    extract = args.out / "table-extract"
    run("extract", args.traces, extract)
    # A missing engine, SIMD, sweep, or detailed identity must never silently reduce coverage.
    matrix = {}
    engines = None
    extraction_problems = 0
    for p in sorted(extract.glob("*.pkl")):
        c = pickle.loads(p.read_bytes())
        current = set(c["engines"])
        if engines is None:
            engines = current
        if not current or current != engines:
            raise ValueError("shader-engine coverage differs between ATT cells")
        if expected is not None:
            if sorted(current) != expected:
                raise ValueError(
                    f"sparse launch: traced shader engines {sorted(current)}, expected {expected}"
                )
            if any(
                x["not_fake"] or not x["detailed"]
                for x in c["excluded_engines"].values()
            ):
                raise ValueError(
                    "sparse launch: an excluded shader engine's traced WGP ran a real workgroup"
                )
        matrix.setdefault(c["sweep"], set()).add(c["traced_simd"])
        extraction_problems += sum(
            sum(e["problems"].values()) for e in c["engines"].values()
        )
        if any(
            not e["detailed"] or any(w["uid"] is None for w in e["detailed"])
            for e in c["engines"].values()
        ):
            raise ValueError(
                "missing or unidentified detailed waves; check --target-cu to WGP mapping"
            )
    if extraction_problems or any(v != {0, 1, 2, 3} for v in matrix.values()):
        raise ValueError("incomplete ATT coverage or invalid wave markers")
    for sweep in sorted(matrix):
        run("merge_rcv", args.traces, extract, sweep, args.out / f"rcv-merged-{sweep}")
    run("align", extract, args.out / "alignment.json")
    run("validate_other_simd", args.traces, extract, args.out / "partner-simd.json")
    run(
        "audit",
        "--traces",
        args.traces,
        "--extract",
        extract,
        "--merged-glob",
        args.out / "rcv-merged-sweep-*/ui_output_*",
        "--out",
        args.out / "audit.json",
    )
    dirs = sorted(args.out.glob("rcv-merged-sweep-*/ui_output_*"))
    run(
        "barrier_sync",
        *dirs,
        "--waves-per-workgroup",
        manifest["waves_per_workgroup"],
        "--out",
        args.out / "same-role-barriers.json",
    )
    phase_args = [f"table={args.traces}"]
    if args.baseline:
        run("extract", args.baseline, args.out / "natural-extract")
        phase_args.append(f"natural={args.baseline}")
    if args.original:
        phase_args.append(f"original={args.original}")
    run("phase", args.manifest, args.out / "phase.json", *phase_args)
    summary = assemble_summary(
        read(args.out / "alignment.json"),
        read(args.out / "partner-simd.json"),
        read(args.out / "audit.json"),
        read(args.out / "same-role-barriers.json"),
        read(args.out / "phase.json"),
        list(map(str, dirs)),
        not args.no_sequence_check,
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    if not summary["valid"]:
        raise ValueError(summary["reason"])
    print(
        f'ATT audit: body sequence check {summary["sequence_check"]} over {summary["same_work"]["waves"]} waves; '
        f"{len(dirs)} merged RCV views"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Preserve a machine-readable failure even when extraction or a subprocess fails.
        if "--out" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            summary_path = out / "summary.json"
            summary = read(summary_path) if summary_path.exists() else {}
            summary.update(valid=False, reason=str(exc))
            summary_path.write_text(json.dumps(summary, indent=2))
        raise
