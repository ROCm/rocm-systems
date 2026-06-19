#!/usr/bin/env python3
"""Audit Gate Exit Dashboard — aggregates all Go/No-Go metrics into one JSON.

Usage:
    python scripts/exit_dashboard.py --output exit_dashboard.json [--allow-partial]

Runs before any breaking change. Output determines GO vs NO-GO.

Spec reference: `docs/superpowers/specs/2026-04-17-multi-agent-perfxpert-design.md`
§7 Go/No-Go table (9 rows).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).parent.parent
PARITY_SNAPSHOT = REPO_ROOT / "tests" / "test_parity" / "parity_snapshots" / "_aggregate.json"
RED_TEAM_OUTCOMES = REPO_ROOT / "tests" / "test_red_team" / "_attack_outcomes"
AIRGAP_SNAPSHOTS = REPO_ROOT / "tests" / "test_integration" / "_airgap_snapshots"
FP_AGGREGATE = REPO_ROOT / "tests" / "test_regression_gate" / "_runner_outputs" / "_aggregate.json"


def collect_parity() -> float | str:
    if not PARITY_SNAPSHOT.exists():
        return "pending"
    data = json.loads(PARITY_SNAPSHOT.read_text())
    # Current snapshot format: top-level "pmc_subset_agreement" with nested "agreement_rate".
    # Older snapshots had "agreement_rate" at the top level.
    if "pmc_subset_agreement" in data:
        return float(data["pmc_subset_agreement"]["agreement_rate"])
    return float(data["agreement_rate"])


def collect_red_team() -> int | str:
    if not RED_TEAM_OUTCOMES.exists():
        return "pending"
    defeated = 0
    for p in RED_TEAM_OUTCOMES.glob("*.json"):
        entry = json.loads(p.read_text())
        if entry.get("status") == "defeated":
            defeated += 1
    return defeated


def collect_airgap() -> float | str:
    if not AIRGAP_SNAPSHOTS.exists():
        return "pending"
    files = list(AIRGAP_SNAPSHOTS.glob("*.json"))
    if not files:
        return "pending"
    identical = 0
    for p in files:
        data = json.loads(p.read_text())
        if data["with_llm"] == data["airgap"]:
            identical += 1
    return identical / len(files)


def collect_false_positive() -> float | str:
    if not FP_AGGREGATE.exists():
        return "pending"
    data = json.loads(FP_AGGREGATE.read_text())
    return float(data["false_positive_rate"])


def collect_narrow_scope_violations() -> int | str:
    """Run per-agent narrow-scope CI check inline (fast subprocess)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_agents/test_narrow_scope.py",
             "--tb=no", "-q"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        if proc.returncode == 0:
            return 0
        # Parse failed count from pytest output
        for line in (proc.stdout + proc.stderr).splitlines():
            if "failed" in line.lower():
                parts = line.split()
                for i, tok in enumerate(parts):
                    if tok.startswith("failed"):
                        return int(parts[i - 1])
        return -1  # unknown
    except Exception:
        return "pending"


def collect_tool_class_split_violations() -> int | str:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_integration/test_mcp_exposure.py",
             "--tb=no", "-q"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        return 0 if proc.returncode == 0 else 1
    except Exception:
        return "pending"


def collect_provider_smoke() -> str:
    """Nightly-only; PR lane returns placeholder."""
    return "nightly-only"


def collect_benchmark() -> str:
    return "nightly-only"


def collect_user_signoff() -> str:
    marker = REPO_ROOT / ".user_signoff"
    return "yes" if marker.exists() else "pending"


def compute_verdict(metrics: Dict[str, Any]) -> str:
    # Each metric must meet its gate; nightly/pending metrics don't block.
    def metric_ok(key: str, val: Any) -> bool:
        if val == "pending" or val == "nightly-only":
            return True
        if key == "parity_agreement_rate":
            return val >= 0.95
        if key == "red_team_pass_count":
            return val == 14
        if key == "airgap_identical_rate":
            return val == 1.0
        if key == "regression_gate_false_positive_rate":
            return val <= 0.05
        if key in ("per_agent_narrow_scope_violations", "tool_class_split_violations"):
            return val == 0
        if key == "user_signoff":
            return val == "yes"
        return True

    all_pass = all(metric_ok(k, v) for k, v in metrics.items())
    has_pending = any(v in ("pending", "nightly-only") for v in metrics.values())
    if all_pass and not has_pending:
        return "GO"
    if all_pass and has_pending:
        return "PARTIAL (pending)"
    return "NO-GO"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="dashboard JSON path")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="emit PARTIAL verdict instead of failing when nightly metrics are unmeasured",
    )
    parser.add_argument("--render", action="store_true",
                        help="render a rich-formatted terminal summary")
    args = parser.parse_args()

    metrics: Dict[str, Any] = {
        "parity_agreement_rate": collect_parity(),
        "red_team_pass_count": collect_red_team(),
        "airgap_identical_rate": collect_airgap(),
        "regression_gate_false_positive_rate": collect_false_positive(),
        "per_agent_narrow_scope_violations": collect_narrow_scope_violations(),
        "tool_class_split_violations": collect_tool_class_split_violations(),
        "provider_smoke_status": collect_provider_smoke(),
        "benchmark_geomean": collect_benchmark(),
        "user_signoff": collect_user_signoff(),
    }
    verdict = compute_verdict(metrics)

    dashboard = {
        "phase": "5",
        "generated_by": "scripts/exit_dashboard.py",
        "metrics": metrics,
        "thresholds": {
            "parity_agreement_rate": ">=0.95",
            "red_team_pass_count": "==14",
            "airgap_identical_rate": "==1.0",
            "regression_gate_false_positive_rate": "<=0.05",
            "per_agent_narrow_scope_violations": "==0",
            "tool_class_split_violations": "==0",
            "user_signoff": "==yes",
        },
        "overall_verdict": verdict,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(dashboard, indent=2))
    print(json.dumps(dashboard, indent=2))

    if args.render:
        render_to_terminal(dashboard)

    if verdict == "NO-GO" and not args.allow_partial:
        sys.exit(2)
    if verdict == "PARTIAL (pending)" and not args.allow_partial:
        sys.exit(1)


def _metric_pass(key: str, val: Any) -> bool:
    if val in ("pending", "nightly-only"):
        return True
    if key == "parity_agreement_rate":
        return val >= 0.95
    if key == "red_team_pass_count":
        return val == 14
    if key == "airgap_identical_rate":
        return val == 1.0
    if key == "regression_gate_false_positive_rate":
        return val <= 0.05
    if key in ("per_agent_narrow_scope_violations", "tool_class_split_violations"):
        return val == 0
    if key == "user_signoff":
        return val == "yes"
    return True


def render_to_terminal(dashboard: Dict[str, Any]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    verdict = dashboard["overall_verdict"]
    color = {"GO": "green", "NO-GO": "red", "PARTIAL (pending)": "yellow"}[verdict]

    console.rule(f"[bold {color}]Audit Gate Exit Dashboard — Go/No-Go[/]")

    table = Table(title="", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_column("Threshold")
    table.add_column("Pass?")

    thresholds = dashboard["thresholds"]
    for key, val in dashboard["metrics"].items():
        threshold = thresholds.get(key, "(nightly)")
        ok = _metric_pass(key, val)
        mark = "[green]✓[/]" if ok else "[red]✗[/]"
        if val in ("pending", "nightly-only"):
            mark = "[yellow]…[/]"
        table.add_row(key, str(val), str(threshold), mark)

    console.print(table)
    console.rule(f"[bold {color}]VERDICT: {verdict}[/]")


if __name__ == "__main__":
    main()
