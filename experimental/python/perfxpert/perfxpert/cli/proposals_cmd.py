###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################
"""``perfxpert proposals`` — review exploratory proposals and scaffold promotion.

A proposal is a specialist's hypothesis about one run. It is not advice, and
the catalog it might eventually join is the thing that makes PerfXpert's
recommendations trustworthy: every entry there cites a measured before/after
pair. So this command deliberately cannot finish a promotion. It renders
proposals for review and emits a catalog skeleton with the measured fields
left blank, because filling them in means running the experiment — which is a
human's job, not a CLI flag.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["add_args", "run_proposals", "load_proposals", "promotion_skeleton"]


# Fields a promoted entry must carry that a proposal cannot supply, because
# each one only exists once someone has actually run the experiment. Kept
# explicit so the gap between "proposed" and "proven" stays visible in one
# place rather than implied by the schema.
#
# These are emitted commented-out rather than as placeholder values. A
# placeholder that is merely *wrong* — `[1.0, 1.0]`, `"TODO"` — still satisfies
# the catalog schema, so a skeleton pasted in unedited would validate and
# unmeasured advice would reach the catalog. Omitting the keys makes the entry
# fail its required-property check instead, and the validator then names the
# exact field the human still owes.
UNMEASURED_FIELDS = (
    "measured_speedup_range",
    "source_citation",
    "preconditions",
    "fixture_pair",
)

_BOTTLENECK_BY_SPECIALIST = {
    "compute": "compute",
    "memory": "memory_transfer",
    "latency": "latency",
}


def add_args(parser: argparse.ArgumentParser) -> None:
    """Register flags for ``perfxpert proposals``."""
    sub = parser.add_subparsers(dest="proposals_action", required=True)

    list_p = sub.add_parser(
        "list", help="Summarise exploratory proposals in a saved agent result"
    )
    list_p.add_argument("result_json", metavar="RESULT.JSON", help="Saved agent output")
    list_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of text"
    )

    show_p = sub.add_parser("show", help="Print one proposal in full, with its evidence")
    show_p.add_argument("result_json", metavar="RESULT.JSON", help="Saved agent output")
    show_p.add_argument("proposal_id", metavar="PROPOSAL_ID", help="e.g. pxp-exp-1a2b...")

    promote_p = sub.add_parser(
        "promote",
        help="Emit a catalog entry skeleton for a proposal (does NOT add it)",
        description=(
            "Writes a proven_optimizations entry with the measured fields left\n"
            "as TODO. It is intentionally incomplete: promotion requires a\n"
            "before/after fixture pair and a measured speedup, which only a\n"
            "human running the experiment can supply."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    promote_p.add_argument("result_json", metavar="RESULT.JSON", help="Saved agent output")
    promote_p.add_argument("proposal_id", metavar="PROPOSAL_ID", help="Proposal to promote")
    promote_p.add_argument(
        "--promoted-by", default=None, metavar="NAME", help="Who is vouching for this"
    )
    promote_p.add_argument(
        "-o", "--output", default=None, metavar="FILE", help="Write here instead of stdout"
    )


# -- Loading ---------------------------------------------------------------


def load_proposals(payload: Any) -> List[Dict[str, Any]]:
    """Collect proposals from an agent result, wherever they were carried.

    Specialists surface them top-level; Diff nests them under `kernel_deltas`
    because its output was already at the field cap. Both shapes reach a user's
    saved JSON, so both are read here rather than making the user know which
    agent produced the file.
    """
    found: List[Dict[str, Any]] = []
    seen = set()

    def _collect(container: Any) -> None:
        if not isinstance(container, dict):
            return
        for entry in container.get("exploratory_proposals") or []:
            if not isinstance(entry, dict):
                continue
            pid = entry.get("proposal_id")
            if pid in seen:
                continue
            seen.add(pid)
            found.append(entry)

    _collect(payload)
    if isinstance(payload, dict):
        _collect(payload.get("kernel_deltas"))
        for value in payload.values():
            if isinstance(value, dict):
                _collect(value)

    return found


def _read(result_json: str) -> Any:
    return json.loads(Path(result_json).read_text(encoding="utf-8"))


def _find(proposals: List[Dict[str, Any]], proposal_id: str) -> Optional[Dict[str, Any]]:
    for proposal in proposals:
        if proposal.get("proposal_id") == proposal_id:
            return proposal
    return None


# -- Rendering -------------------------------------------------------------


def _render_list(proposals: List[Dict[str, Any]]) -> str:
    if not proposals:
        return (
            "No exploratory proposals in this result.\n"
            "Proposals require a live session with agent_creativity=exploratory;\n"
            "air-gapped runs never produce them.\n"
        )

    lines = [
        "Exploratory proposals — hypotheses, not recommendations.",
        "None of these has been measured. Review the evidence before acting.",
        "",
    ]
    for proposal in proposals:
        lines.append(f"  {proposal.get('proposal_id', '?')}  [{proposal.get('specialist', '?')}]")
        lines.append(f"    {proposal.get('title', '(untitled)')}")
        kernel = proposal.get("target_kernel")
        detail = f"    confidence {float(proposal.get('confidence', 0.0)):.2f}"
        if kernel:
            detail += f"  ·  target {kernel}"
        lines.append(detail)
        lines.append("")
    lines.append(f"{len(proposals)} proposal(s). Use `proposals show <id>` for detail.")
    lines.append("")
    return "\n".join(lines)


def _bullets(title: str, items: Any) -> List[str]:
    items = [item for item in (items or []) if item]
    if not items:
        return []
    out = [f"{title}:"]
    out.extend(f"  - {item}" for item in items)
    out.append("")
    return out


def _render_show(proposal: Dict[str, Any]) -> str:
    lines = [
        "=" * 72,
        f" {proposal.get('title', '(untitled)')}",
        "=" * 72,
        f"  id          : {proposal.get('proposal_id', '?')}",
        f"  status      : {proposal.get('status', '?')}  (unproven — not benchmarked)",
        f"  specialist  : {proposal.get('specialist', '?')}",
        f"  confidence  : {float(proposal.get('confidence', 0.0)):.2f}",
        f"  target      : {proposal.get('target_kernel') or '(no specific kernel)'}",
        "",
        "Hypothesis:",
        f"  {proposal.get('hypothesis', '')}",
        "",
        "Mechanism:",
        f"  {proposal.get('mechanism', '')}",
        "",
    ]

    evidence = proposal.get("evidence") or []
    lines.append("Evidence (bound to this run):")
    if evidence:
        for item in evidence:
            if isinstance(item, dict):
                lines.append(f"  - [{item.get('kind', '?')}] {item.get('ref', '?')}")
                observation = item.get("observation")
                if observation:
                    lines.append(f"      {observation}")
    else:
        lines.append("  (none recorded)")
    lines.append("")

    for label, key in (
        ("Expected effects", "expected_effects"),
        ("Assumptions", "assumptions"),
        ("Failure modes", "failure_modes"),
    ):
        values = proposal.get(key) or []
        rendered = [
            value if not isinstance(value, dict) else json.dumps(value, sort_keys=True)
            for value in values
        ]
        lines.extend(_bullets(label, rendered))

    verification = proposal.get("verification")
    if isinstance(verification, dict):
        lines.append("Verification plan:")
        for key, value in sorted(verification.items()):
            lines.append(f"  {key}: {value}")
        lines.append("")

    lines.append("To promote: run the experiment, then `perfxpert proposals promote`.")
    lines.append("")
    return "\n".join(lines)


# -- Promotion scaffold ----------------------------------------------------


def promotion_skeleton(
    proposal: Dict[str, Any], *, promoted_by: Optional[str] = None
) -> str:
    """Render a catalog entry with the measured fields left unfilled.

    Returns YAML rather than appending to the catalog. Writing it in is a
    reviewed change, and the entry is invalid until someone replaces the TODOs
    with real measurements — which is the point.
    """
    specialist = str(proposal.get("specialist", "")).strip()
    bottleneck = _BOTTLENECK_BY_SPECIALIST.get(specialist, "mixed")
    entry_id = _suggest_id(proposal)

    def _yaml_list(items: Any, indent: str) -> List[str]:
        items = [item for item in (items or []) if item]
        if not items:
            return [f"{indent}[]  # TODO: fill in"]
        return [f"{indent}- {json.dumps(str(item))}" for item in items]

    lines = [
        "# Promotion skeleton — NOT a valid catalog entry yet, by design.",
        "#",
        f"# Seeded from exploratory proposal {proposal.get('proposal_id', '?')}.",
        "# That proposal was a hypothesis about one run; nobody measured it. The",
        "# fields it cannot supply are commented out below, so this entry fails",
        "# schema validation until you run the experiment and fill them in:",
    ]
    lines.extend(f"#   - {field}" for field in UNMEASURED_FIELDS)
    lines.extend(
        [
            "#",
            "# Append to perfxpert/knowledge/proven_optimizations.yaml only after",
            "# the fixture pair exists and the gate cascade passes on it.",
            "",
            f"- id: {entry_id}",
            f"  bottleneck_type: {bottleneck}",
            f"  technique: {json.dumps(str(proposal.get('title', '')))}",
            "  description: >",
        ]
    )
    for field in ("hypothesis", "mechanism"):
        text = str(proposal.get(field, "")).strip()
        if text:
            lines.append(f"    {text}")

    lines.append("  failure_modes:")
    lines.extend(_yaml_list(proposal.get("failure_modes"), "    "))

    lines.extend(
        [
            "  origin:",
            "    kind: promoted_proposal",
            f"    proposal_id: {proposal.get('proposal_id', 'TODO')}",
        ]
    )
    if specialist:
        lines.append(f"    specialist: {specialist}")
    lines.append(f"    promoted_by: {json.dumps(promoted_by or 'TODO: your name')}")
    lines.append('    promoted_at: "TODO: YYYY-MM-DD"')

    lines.extend(
        [
            "",
            "  # ---------------- TODO: uncomment and fill in ----------------",
            "  # Each of these comes from actually running the experiment.",
            "  # Until all four are present the entry is invalid, and the",
            "  # validator will name whichever one is still missing.",
            "  #",
            "  # measured_speedup_range: [1.15, 1.30]   # measured [low, high]",
            '  # source_citation: "in-house experiment <id>, or a published source"',
            "  # preconditions:                          # when this held",
            '  #   - {metric: "<metric>", op: ">", threshold: 0}',
            "  # fixture_pair:",
            f'  #   baseline_db:    "tests/fixtures/proven_optimizations/{entry_id}.baseline.db"',
            f'  #   optimized_db:   "tests/fixtures/proven_optimizations/{entry_id}.optimized.db"',
            f'  #   description_md: "tests/fixtures/proven_optimizations/{entry_id}.md"',
            "",
        ]
    )
    return "\n".join(lines)


def _suggest_id(proposal: Dict[str, Any]) -> str:
    """Derive a snake_case catalog id matching the schema's id pattern."""
    title = str(proposal.get("title", "")).lower()
    kept = [char if char.isalnum() else "_" for char in title]
    slug = "".join(kept).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug[:48].strip("_")
    if not slug or not slug[0].isalpha():
        slug = f"proposal_{slug}".strip("_")
    return slug or "promoted_proposal"


# -- Entry point -----------------------------------------------------------


def run_proposals(args: argparse.Namespace) -> int:
    """Execute ``perfxpert proposals``."""
    try:
        payload = _read(args.result_json)
    except FileNotFoundError:
        print(f"error: file not found: {args.result_json}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"error: {args.result_json} is not valid JSON: {exc}", file=sys.stderr)
        return 2

    proposals = load_proposals(payload)
    action = getattr(args, "proposals_action", None)

    if action == "list":
        if getattr(args, "json", False):
            print(json.dumps(proposals, indent=2))
        else:
            sys.stdout.write(_render_list(proposals))
        return 0

    proposal = _find(proposals, args.proposal_id)
    if proposal is None:
        print(f"error: no proposal {args.proposal_id!r} in {args.result_json}", file=sys.stderr)
        if proposals:
            print("known ids:", file=sys.stderr)
            for known in proposals:
                print(f"  {known.get('proposal_id')}", file=sys.stderr)
        return 2

    if action == "show":
        sys.stdout.write(_render_show(proposal))
        return 0

    if action == "promote":
        rendered = promotion_skeleton(
            proposal, promoted_by=getattr(args, "promoted_by", None)
        )
        output = getattr(args, "output", None)
        if output:
            Path(output).write_text(rendered, encoding="utf-8")
            print(f"Wrote promotion skeleton to {output}")
        else:
            sys.stdout.write(rendered)
        print(
            "\nThis entry is incomplete by design: "
            f"{', '.join(UNMEASURED_FIELDS)} still need real measurements.",
            file=sys.stderr,
        )
        return 0

    print(f"error: unknown action {action!r}", file=sys.stderr)
    return 2
