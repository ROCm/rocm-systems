#!/usr/bin/env python3
"""Plan and sum up the RCCL history sweep (therock-rccl-madengine-history.yml).

  plan     Print candidate=<run ID> and matrix=<JSON> for $GITHUB_OUTPUT: one
           A/B leg per baseline, every one against the same candidate build.
           Each baseline is looked up before any leg takes the cluster.
  summary  Print a Markdown table of every leg's ratios, read from the
           madengine_summary.txt each leg uploaded.

Baselines are comma- or whitespace-separated: a CI run ID (or run:<ID>),
release:<version>, nightly:<version> or url:<url>, as fetch_rccl_build.py
takes them.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path

import fetch_rccl_build
import test_madengine

log = logging.getLogger("rccl_history")

# Every leg holds the same nodes for about two hours, one after another.
MAX_LEGS = 10
STATUS_RE = re.compile(r"^Status:\s+(\w+)")
SIDE_RE = re.compile(r"^(Baseline|Candidate):\s+RCCL (.*) \(([^()]*)\)$")
RATIO_RE = re.compile(r"^\s*(\S+):\s+ratio (\S+)")
VALUE_RE = re.compile(r"^\s+(baseline|candidate)\s+(\S+)")


def _label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")


def plan_leg(spec: str, family: str, repo: str) -> dict:
    """The matrix entry of one baseline; raises if it cannot be fetched.

    ``baseline`` is what the leg passes as baseline_artifact_run_id: a run
    ID, or the spec of a published build.
    """
    run_id = spec[len("run:"):] if spec.startswith("run:") else spec
    if run_id.isdigit():
        if not test_madengine.rccl_artifact_exists(
                run_id, fetch_rccl_build.gpu_target(family), repo):
            raise RuntimeError(f"{spec}: its RCCL is not in the bucket (expired?)")
        return {"baseline": run_id, "label": f"run-{run_id}"}
    plan = fetch_rccl_build.resolve(spec, family)
    kind = spec.partition(":")[0]
    label = f"{kind}-{plan['version']}"
    if kind == "url":
        label += "-" + hashlib.sha256(spec.encode()).hexdigest()[:8]
    return {"baseline": spec, "label": _label(label)}


def plan(candidate: str, baselines: str, family: str, env: dict) -> tuple[str, list[dict]]:
    """The candidate build and one leg per baseline; raises listing every bad one."""
    repo = env["GITHUB_REPOSITORY"]
    gpu = fetch_rccl_build.gpu_target(family)
    errors = []
    candidate = candidate.strip()
    if not candidate:
        candidate = test_madengine.previous_scheduled_build("", "nightly", gpu, env) or ""
        if not candidate:
            errors.append("no scheduled build has its RCCL in the bucket")
        else:
            log.info("Candidate: %s, the newest scheduled build", candidate)
    elif not candidate.isdigit():
        errors.append(f"candidate_artifact_run_id must be a run ID, got {candidate!r}")
    elif not test_madengine.rccl_artifact_exists(candidate, gpu, repo):
        errors.append(f"candidate {candidate}: its RCCL is not in the bucket (expired?)")

    specs = [s for s in re.split(r"[\s,]+", baselines) if s]
    if not specs:
        errors.append("no baselines given")
    if len(specs) > MAX_LEGS:
        errors.append(f"{len(specs)} baselines; at most {MAX_LEGS} legs per sweep")
    legs = []
    for spec in specs:
        try:
            leg = plan_leg(spec, family, repo)
        except (ValueError, RuntimeError, OSError) as exc:
            errors.append(str(exc))
            continue
        if any(leg["label"] == other["label"] for other in legs):
            errors.append(f"{spec}: given twice")
            continue
        log.info("Leg %s: %s", leg["label"], spec)
        legs.append(leg)
    if errors:
        raise RuntimeError("; ".join(errors))
    return candidate, legs


def parse_summary(text: str) -> dict:
    """Status, both sides' RCCL and build, and per precision the ratio and values."""
    result = {"status": "", "rccl": {}, "build": {}, "precisions": []}
    for line in text.splitlines():
        if m := STATUS_RE.match(line):
            result["status"] = m.group(1)
        elif m := SIDE_RE.match(line):
            result["rccl"][m.group(1).lower()] = m.group(2)
            result["build"][m.group(1).lower()] = m.group(3)
        elif m := RATIO_RE.match(line):
            result["precisions"].append({"precision": m.group(1), "ratio": m.group(2)})
        elif (m := VALUE_RE.match(line)) and result["precisions"]:
            result["precisions"][-1][m.group(1)] = m.group(2)
    return result


def summary(artifacts_dir: Path, legs: list[dict], artifact_prefix: str) -> str:
    """A Markdown table of every leg, in plan order, a missing one included."""
    lines = ["| Baseline | Baseline RCCL | Precision | Baseline | Candidate | Ratio | Status |",
             "|---|---|---|---|---|---|---|"]
    candidate_rccl = set()
    for leg in legs:
        path = artifacts_dir / f"{artifact_prefix}-{leg['label']}" / "madengine_summary.txt"
        if not path.is_file():
            lines.append(f"| `{leg['baseline']}` | | | | | | no results |")
            continue
        result = parse_summary(path.read_text(errors="replace"))
        if result["rccl"].get("candidate"):
            candidate_rccl.add(result["rccl"]["candidate"])
        for p in result["precisions"] or [{"precision": "", "ratio": ""}]:
            lines.append(
                f"| `{leg['baseline']}` | {result['rccl'].get('baseline', '')} "
                f"| {p['precision']} | {p.get('baseline', '')} | {p.get('candidate', '')} "
                f"| {p['ratio']} | {result['status']} |"
            )
    head = ["## RCCL history sweep", ""]
    if candidate_rccl:
        head += [f"Candidate RCCL: {', '.join(sorted(candidate_rccl))}", ""]
    head += ["Ratio is candidate over baseline throughput, both measured on the same "
             "nodes in one allocation.", ""]
    return "\n".join(head + lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="print candidate= and matrix= for $GITHUB_OUTPUT")
    p.add_argument("--amdgpu-family", required=True)
    s = sub.add_parser("summary", help="print the sweep's Markdown table")
    s.add_argument("--artifacts-dir", type=Path, required=True)
    s.add_argument("--matrix", required=True, help="the plan's matrix JSON")
    s.add_argument("--artifact-prefix", required=True,
                   help="the legs' result artifact name, without the run label")
    args = parser.parse_args()

    if args.command == "plan":
        try:
            candidate, legs = plan(os.environ.get("CANDIDATE_ARTIFACT_RUN_ID", ""),
                                   os.environ.get("BASELINES", ""),
                                   args.amdgpu_family, os.environ)
        except (RuntimeError, ValueError, OSError, KeyError) as exc:
            log.error("Cannot plan the sweep: %s", exc)
            sys.exit(1)
        print(f"candidate={candidate}")
        print(f"matrix={json.dumps({'include': legs}, separators=(',', ':'))}")
    else:
        legs = json.loads(args.matrix)["include"]
        sys.stdout.write(summary(args.artifacts_dir, legs, args.artifact_prefix))


if __name__ == "__main__":
    main()
