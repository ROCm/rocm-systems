#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Mutation analysis over rj_isa_diff_pr results: how strong is a pass?

Every function an rj_isa_diff_pr run reports as a plain pass (`NO_MISMATCH_FOUND`,
complete coverage, no don't-care masking) has a case directory. For each, `rj_isa_diff
mutants` flips single bits of the new side of each passing region (up to --max mutants
per region, spread over operator classes) and tests each mutant like the region. A
mutant is a would-be miscompile, so:

- killed       the test finds it at the given budget (by the boundary sweep or the
               random phase);
- killed_late  only a re-run with --confirm-cases random cases finds it (the budget
               is too small for it);
- survived     never found. Each survivor is also run against the new side it came
               from: no_effect (identical final states: an ignored modifier, a masked
               shift amount), dead_effect (only locations not live at the exit differ),
               undefined_effect (a live-out location undefined at the entry, which the
               original passes through, holds another value: any value refines it) or
               observable_effect (a live-out value differs; the comparison with old
               tolerated it, e.g. a NaN payload, or the test is blind to it);
- invalid      the mutant left the model (an effect the classifier did not predict, no
               case that runs on both sides); not counted.

The mutation score is killed / (killed + killed_late + survived); without equivalent
mutants it is killed / (killed + killed_late + survivors with an observable effect),
since no_effect, dead_effect and undefined_effect mutants are not miscompiles. The
report breaks it
down by operator class (opcode, register, constant, modifier), by unit (SALU, VALU,
memory) and by target, and lists survivors so blind spots can be told from equivalent
mutants. All mutants are synthetic; nothing here is an LLVM finding.

usage: rj_isa_diff_mutants RESULTS_DIR... --out DIR [--tool PATH] [--jobs N]
       [--max N] [--cases N] [--confirm-cases N] [--seed N] [--limit N]
"""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

VERDICTS = ("killed", "killed_late", "survived", "invalid")


def plain_passes(roots):
    """(row, case dir) of every plain pass in the summaries under @p roots."""
    seen = set()
    for root in roots:
        for summary in sorted(Path(root).rglob("summary.json")):
            try:
                doc = json.loads(summary.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            for row in doc.get("functions", []):
                case = Path(row.get("dir", "")) / "case"
                if (
                    row.get("status") == "NO_MISMATCH_FOUND"
                    and row.get("coverage") == "COMPLETE_FOR_DECLARED_SCOPE"
                    and not row.get("dont_care_masked")
                    and (case / "case.json").exists()
                    and str(case) not in seen
                ):
                    seen.add(str(case))
                    yield row, case


def run_one(tool, row, case, out, args):
    key = hashlib.sha256(str(case).encode()).hexdigest()[:16]
    dest = out / "mutants" / f"{key}.json"
    argv = [
        str(tool),
        "mutants",
        "--case",
        str(case),
        "--out",
        str(dest),
        "--max",
        str(args.max),
        "--cases",
        str(args.cases),
        "--confirm-cases",
        str(args.confirm_cases),
        "--seed",
        str(args.seed),
        "--effect-cases",
        str(args.effect_cases),
    ]
    start = time.monotonic()
    proc = subprocess.run(argv, capture_output=True, text=True)
    took = time.monotonic() - start
    if proc.returncode != 0 or not dest.exists():
        return row, None, (proc.stderr or proc.stdout).strip()[-300:], took
    return row, json.loads(dest.read_text()), "", took


def score(c):
    n = c["killed"] + c["killed_late"] + c["survived"]
    return f"{100.0 * c['killed'] / n:.1f}%" if n else "-"


def table(title, groups):
    lines = [
        f"### {title}",
        "",
        "| group | killed | killed_late | survived | invalid | score |",
        "|---|---|---|---|---|---|",
    ]
    for name, c in sorted(groups.items(), key=lambda kv: -sum(kv[1].values())):
        lines.append(
            f"| {name} | {c['killed']} | {c['killed_late']} | {c['survived']} | "
            f"{c['invalid']} | {score(c)} |"
        )
    return lines + [""]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("results", nargs="+", help="rj_isa_diff_pr output directories")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--tool", default=str(Path(__file__).resolve().parent / "rj_isa_diff")
    )
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--max", type=int, default=48, help="mutants per region")
    ap.add_argument("--cases", type=int, default=5000, help="random cases per mutant")
    ap.add_argument("--confirm-cases", type=int, default=20000)
    ap.add_argument(
        "--effect-cases",
        type=int,
        default=2000,
        help="random cases comparing a survivor with its original",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="at most N functions (0: all)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "mutants").mkdir(parents=True, exist_ok=True)
    jobs = list(plain_passes(args.results))
    if args.limit and len(jobs) > args.limit:
        step = len(jobs) / args.limit  # an even spread over the runs, not the first N
        jobs = [jobs[int(i * step)] for i in range(args.limit)]
    print(f"{len(jobs)} passing functions", flush=True)

    totals = collections.Counter()
    groups = {
        k: collections.defaultdict(collections.Counter)
        for k in ("class", "unit", "target")
    }
    phases = collections.Counter()
    effects = collections.Counter()
    effects_by_class = collections.defaultdict(collections.Counter)
    survivors, invalid, errors = [], collections.Counter(), []
    regions = baseline_off = 0
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        futs = [
            pool.submit(run_one, args.tool, row, case, out, args) for row, case in jobs
        ]
        for n, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            row, rep, err, took = fut.result()
            if rep is None:
                errors.append(f"{row.get('file')}:{row.get('function')}: {err}")
                continue
            target = f"{row.get('target')} w{row.get('wave_size', 64)}"
            for r in rep["regions"]:
                regions += 1
                if r["baseline"] != "NO_MISMATCH_FOUND" or r["baseline_masked"]:
                    baseline_off += 1
                for m in r["mutants"]:
                    v = m["verdict"]
                    totals[v] += 1
                    groups["class"][m["class"]][v] += 1
                    groups["unit"][m["unit"]][v] += 1
                    groups["target"][target][v] += 1
                    if v in ("killed", "killed_late"):
                        phases[f"{v} in {m.get('phase', '?')}"] += 1
                    elif v == "survived":
                        effects[m.get("effect", "unknown")] += 1
                        effects_by_class[m["class"]][m.get("effect", "unknown")] += 1
                        survivors.append(
                            {
                                "effect": m.get("effect", "unknown"),
                                "where": m.get("effect_detail", ""),
                                "target": target,
                                "file": Path(row.get("file", "")).name,
                                "function": row.get("function"),
                                "region": r["id"],
                                "class": m["class"],
                                "original": m["original"],
                                "mutated": m["mutated"],
                            }
                        )
                    else:
                        invalid[m.get("reason", "?")[:100]] += 1
            if n % 50 == 0:
                print(
                    f"  {n}/{len(jobs)} functions, {time.monotonic() - start:.0f} s",
                    flush=True,
                )

    wall = time.monotonic() - start
    survivors.sort(
        key=lambda x: (x["effect"] != "observable_effect", x["target"], x["file"])
    )
    observable = effects["observable_effect"] + effects["unknown"]
    denom = totals["killed"] + totals["killed_late"] + observable
    adjusted = f"{100.0 * totals['killed'] / denom:.1f}%" if denom else "-"
    summary = {
        "schema_version": "rj_isa_diff.mutation_campaign.v1",
        "options": {
            "max": args.max,
            "cases": args.cases,
            "confirm_cases": args.confirm_cases,
            "seed": args.seed,
        },
        "functions": len(jobs),
        "regions": regions,
        "regions_not_plain_pass": baseline_off,
        "errors": errors,
        "totals": {v: totals[v] for v in VERDICTS},
        "score": score({v: totals[v] for v in VERDICTS}),
        "score_without_equivalent": adjusted,
        "survivor_effects": dict(effects),
        "survivor_effects_by_class": {k: dict(c) for k, c in effects_by_class.items()},
        "phases": dict(phases),
        "by_class": {k: dict(c) for k, c in groups["class"].items()},
        "by_unit": {k: dict(c) for k, c in groups["unit"].items()},
        "by_target": {k: dict(c) for k, c in groups["target"].items()},
        "invalid_reasons": dict(invalid.most_common(20)),
        "survivors": survivors,
        "wall_seconds": round(wall, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    full = lambda g: {k: {v: c.get(v, 0) for v in VERDICTS} for k, c in g.items()}
    lines = [
        "# rj_isa_diff mutation analysis (synthetic mutants)",
        "",
        f"- {len(jobs)} passing functions, {regions} regions "
        f"({baseline_off} not a plain pass at this budget, skipped), {len(errors)} errors, "
        f"{wall:.0f} s",
        f"- budget per mutant: {args.cases} random cases after the boundary sweep; "
        f"survivors re-run with {args.confirm_cases}",
        f"- mutants: " + ", ".join(f"{totals[v]} {v}" for v in VERDICTS),
        f"- **mutation score {summary['score']}** (killed / killed + killed_late + survived); "
        f"**{adjusted} without equivalent mutants** (survivors with no, a dead or an "
        f"undefined-value effect left out)",
        "- survivors: " + ", ".join(f"{n} {k}" for k, n in effects.most_common()),
        "- first kill: " + ", ".join(f"{n} {k}" for k, n in sorted(phases.items())),
        "",
    ]
    lines += table("By operator class", full(groups["class"]))
    lines += table("By unit", full(groups["unit"]))
    lines += table("By target", full(groups["target"]))
    lines += [
        "### Survivors by class and effect",
        "",
        "| class | no_effect | dead_effect | undefined_effect | observable_effect |",
        "|---|---|---|---|---|",
    ]
    for k, c in sorted(effects_by_class.items()):
        lines.append(
            f"| {k} | {c['no_effect']} | {c['dead_effect']} | {c['undefined_effect']} | "
            f"{c['observable_effect']} |"
        )
    lines.append("")
    if invalid:
        lines += ["### Invalid mutants (not counted)", ""]
        lines += [f"- {n} {k}" for k, n in invalid.most_common(10)] + [""]
    if survivors:
        lines += [
            "### Survivors (observable effect first; first 60)",
            "",
            "| effect | target | function | class | original | mutated |",
            "|---|---|---|---|---|---|",
        ]
        for x in survivors[:60]:
            where = f" ({x['where']})" if x.get("where") else ""
            lines.append(
                f"| {x['effect']}{where} | {x['target']} | {x['file']}:{x['function']} "
                f"{x['region']} | {x['class']} | `{x['original']}` | `{x['mutated']}` |"
            )
        lines.append("")
    if errors:
        lines += ["### Errors", ""] + [f"- {e}" for e in errors[:20]] + [""]
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:8]))
    print(f"wrote {out / 'report.md'} and {out / 'summary.json'}")
    return 3 if errors and not totals else 0


if __name__ == "__main__":
    sys.exit(main())
