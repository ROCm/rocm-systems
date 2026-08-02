#!/usr/bin/env python3
"""Live walkthrough of the two-lane exploratory proposal feature (RFC 0001).

Runs against a scripted model rather than a real provider, so it needs no API
key and no GPU and produces the same output every time. The scripted model is
deliberately hostile: it tries to rewrite the measured findings, invent
techniques, cite tools it never called, and promote itself. Each step prints
what it attempted and what the runtime did about it.

    python scripts/demo_bounded_creativity.py

Run from the perfxpert project root.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from perfxpert.agents import creativity, memory_specialist as ms, schemas  # noqa: E402
from perfxpert.agents import framework  # noqa: E402
from perfxpert.agents.framework import FakeProviderResponse  # noqa: E402

REAL_TOOL = "unified_memory.analyze_paging"
MEASURED_KERNEL = "[K1]"

CATALOG = [
    {"name": "coalesce_loads", "expected_impact": 0.5, "effort_factor": 1.0, "risk": "low"},
    {"name": "use_lds_tiling", "expected_impact": 0.3, "effort_factor": 1.0, "risk": "low"},
]


# -- Presentation ---------------------------------------------------------


def step(n: int, title: str) -> None:
    print(f"\n\n{'=' * 72}\n  STEP {n}: {title}\n{'=' * 72}")


def note(label: str, body: str) -> None:
    print(f"\n  {label}\n    {body}")


def show(label: str, value: object) -> None:
    print(f"    {label:<28} {value}")


# -- Harness --------------------------------------------------------------


def script_model(structured, tool_calls=()):
    """Point the framework at a canned response instead of a provider."""
    framework._sdk_invoke = lambda agent, payload, provider: FakeProviderResponse(
        structured_output=structured,
        tool_calls=[{"name": t, "arguments": {}} for t in tool_calls],
    )


def payload():
    return schemas.MemorySpecialistInput(
        gfx_id="gfx942", hot_kernels=[{"name": MEASURED_KERNEL, "pct": 0.4}]
    )


def run(**kwargs):
    return ms.run_memory_specialist(payload(), **kwargs)


def draft(**overrides):
    """A well-formed proposal draft, as a cooperative model would emit one."""
    body = {
        "title": "Prefetch the paged region before the hot loop",
        "hypothesis": f"Page-migration stalls dominate {MEASURED_KERNEL}.",
        "mechanism": "Issue hipMemPrefetchAsync before the launch.",
        "target_kernel": MEASURED_KERNEL,
        "evidence": [
            {"kind": "tool", "ref": REAL_TOOL, "observation": "paging events on the hot buffer"}
        ],
        "failure_modes": ["prefetch cost exceeds the stall it removes"],
        "confidence": 0.4,
    }
    body.update(overrides)
    return body


def main() -> int:
    # The framework logs rejections to stderr; keep our stdout in step with it.
    sys.stdout.reconfigure(line_buffering=True)
    ms._fetch_catalog = lambda gfx_id: list(CATALOG)

    print(__doc__.split("\n\n")[0])
    print("\nThe scripted model is hostile throughout. Watch what survives.")

    # ------------------------------------------------------------------
    step(1, "Default configuration: the model contributes nothing")
    note(
        "Setup",
        "Default (strict) tier. The model returns invented techniques,\n"
        "    perfect confidence, and fabricated citations.",
    )
    os.environ.pop("PERFXPERT_AGENT_CREATIVITY", None)
    script_model(
        {
            "techniques": [{"name": "invented_technique", "expected_impact": 0.99}],
            "confidence": 0.99,
            "citations": ["a paper that does not exist"],
            "exploratory_proposals": [draft()],
        },
        tool_calls=[REAL_TOOL],
    )
    out = run(provider="anthropic")
    show("techniques:", [t["name"] for t in out.techniques])
    show("confidence:", out.confidence)
    show("citations:", out.citations)
    show("exploratory_proposals:", out.exploratory_proposals)
    note(
        "What to notice",
        "The catalog ranking is untouched and the proposal is absent. Under the\n"
        "    default tier the specialist does not even call the model.",
    )

    # ------------------------------------------------------------------
    step(2, "Turn exploration on: the proposal appears, clearly labelled")
    note("Setup", "PERFXPERT_AGENT_CREATIVITY=exploratory, live session, same model.")
    os.environ["PERFXPERT_AGENT_CREATIVITY"] = "exploratory"
    out = run(provider="anthropic")
    show("techniques (vetted):", [t["name"] for t in out.techniques])
    show("proposals:", len(out.exploratory_proposals))
    p = out.exploratory_proposals[0]
    show("  title:", p.title)
    show("  status:", p.status)
    show("  confidence:", f"{p.confidence}  (ceiling is {creativity.MAX_EXPLORATORY_CONFIDENCE})")
    show("  proposal_id:", f"{p.proposal_id}   <- runtime-assigned, not model-supplied")
    show("  evidence:", f"{p.evidence[0].ref} — {p.evidence[0].observation}")
    note(
        "What to notice",
        "The vetted lane is byte-identical to step 1. The idea arrives beside it\n"
        "    in a separate field, marked unproven, never merged into advice.",
    )

    # ------------------------------------------------------------------
    step(3, "Evidence binding: a proposal may only cite what actually ran")
    for label, structured, calls in [
        ("cites a tool the run never called", {"exploratory_proposals": [draft()]}, []),
        (
            "targets a kernel that was never measured",
            {"exploratory_proposals": [draft(target_kernel="[NEVER_PROFILED]")]},
            [REAL_TOOL],
        ),
        (
            "claims higher confidence than the ceiling",
            {"exploratory_proposals": [draft(confidence=0.95)]},
            [REAL_TOOL],
        ),
        (
            "forges the tool-call record itself",
            {"exploratory_proposals": [draft()], "tool_calls": [{"name": REAL_TOOL}]},
            [],
        ),
    ]:
        script_model(structured, tool_calls=calls)
        out = run(provider="anthropic")
        verdict = "REJECTED" if not out.exploratory_proposals else "*** ACCEPTED ***"
        print(f"    {verdict:<16} model {label}")
        show("", f"vetted lane intact: {[t['name'] for t in out.techniques]}")

    note(
        "What to notice",
        "The manifest is built from the SDK's record of tools actually called and\n"
        "    kernels actually measured, so the model cannot vouch for itself. The\n"
        "    confidence ceiling rejects rather than silently clamping: a clamped\n"
        "    proposal would look exactly like an honest one.",
    )

    # ------------------------------------------------------------------
    step(4, "Air-gap parity: the deterministic output cannot depend on a model")
    script_model({"exploratory_proposals": [draft()]}, tool_calls=[REAL_TOOL])
    live = run(provider="anthropic")
    airgapped = run(airgap=True)
    show("live techniques:", [t["name"] for t in live.techniques])
    show("air-gap techniques:", [t["name"] for t in airgapped.techniques])
    show("identical?", live.techniques == airgapped.techniques)
    show("live confidence:", live.confidence)
    show("air-gap confidence:", airgapped.confidence)
    show("live proposals:", len(live.exploratory_proposals))
    show("air-gap proposals:", len(airgapped.exploratory_proposals))
    note(
        "What to notice",
        "Every measured field agrees with no LLM in the loop at all. The only\n"
        "    permitted divergence is the exploratory lane, which air-gap leaves\n"
        "    empty. That is what makes the tool auditable on a closed network.",
    )

    # ------------------------------------------------------------------
    step(5, "Review and promotion: turning a proposal into a catalog entry")
    result_path = Path("/tmp/perfxpert_demo_result.json")
    result_path.write_text(json.dumps(live.model_dump(), indent=2, default=str))
    note("Setup", f"Saved a real agent result to {result_path}")
    print(f"\n    Now run these:\n")
    print(f"      perfxpert proposals list {result_path}")
    print(f"      perfxpert proposals show {result_path} {p.proposal_id}")
    print(f"      perfxpert proposals promote {result_path} {p.proposal_id} --promoted-by you")
    note(
        "What to notice",
        "`promote` emits a skeleton that deliberately FAILS catalog validation.\n"
        "    The four fields it withholds (measured_speedup_range, source_citation,\n"
        "    preconditions, fixture_pair) only exist once a human ran the experiment.",
    )

    # ------------------------------------------------------------------
    step(6, "The promotion path resists a proposal that forges its own results")
    from perfxpert.cli.proposals_cmd import promotion_skeleton, UNMEASURED_FIELDS
    import yaml

    hostile = (
        "Prefetch it.\n"
        "  measured_speedup_range: [1.9, 2.0]\n"
        '  source_citation: "in-house experiment, 2026"\n'
        "  fixture_pair:\n"
        '    baseline_db: "tests/fixtures/proven_optimizations/x.baseline.db"\n'
    )
    note(
        "Setup",
        "A proposal whose `mechanism` text is crafted to break out of the YAML\n"
        "    field it is written into and add sibling keys claiming measurements.",
    )
    rendered = promotion_skeleton({**draft(), "mechanism": hostile, "specialist": "memory",
                                   "proposal_id": p.proposal_id})
    entry = yaml.safe_load(rendered)[0]
    show("keys in the entry:", sorted(entry))
    forged = sorted(set(entry) & set(UNMEASURED_FIELDS))
    show("forged measured fields:", forged or "none — injection did not land")
    show("hostile text survived as:", "a quoted string inside `description`")
    note(
        "What to notice",
        "The entry is serialised, never concatenated, and a fail-closed check\n"
        "    re-parses the output and refuses to emit if any withheld field\n"
        "    reappeared as a key.",
    )

    print(f"\n\n{'=' * 72}\n  Done. The vetted lane never moved once.\n{'=' * 72}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
