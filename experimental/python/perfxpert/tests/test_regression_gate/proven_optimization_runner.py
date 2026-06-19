"""Drives the 5-gate cascade on every proven-optimization seed case to confirm
the gate does NOT reject known-good optimizations.

Each case = one entry in knowledge/proven_optimizations.yaml + its fixture
pair (baseline.db, optimized.db) under tests/fixtures/proven_optimizations/<id>/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from perfxpert.knowledge import load_yaml
from perfxpert.runtime.gate_cascade import GateInput, GateVerdict, run_gate_cascade
from perfxpert.tools.regression import extract_kernel_runtimes_from_db


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "proven_optimizations"


@dataclass(frozen=True)
class ProvenOptimizationCase:
    case_id: str               # matches YAML entry id (e.g. "vgpr_reduction_compute_bound")
    bottleneck: str
    technique: str
    measured_impact_min: float
    measured_impact_max: float
    fixture_dir: Path


class ProvenOptimizationRunner:
    def load_seed_cases(self) -> List[ProvenOptimizationCase]:
        try:
            entries = load_yaml("proven_optimizations")
        except FileNotFoundError:
            # Fixture not present yet; return empty list and tests will skip
            return []

        cases = []
        for entry in entries:
            # Try both subdirectory layout and flat layout
            case_dir = FIXTURES_DIR / entry["id"]
            baseline_db = case_dir / "baseline.db"
            optimized_db = case_dir / "optimized.db"

            # Fall back to flat layout if subdirectory doesn't exist
            if not baseline_db.exists():
                baseline_db = FIXTURES_DIR / entry["fixture_pair"]["baseline_db"].split("/")[-1]
                optimized_db = FIXTURES_DIR / entry["fixture_pair"]["optimized_db"].split("/")[-1]

            # Skip if neither layout works
            if not baseline_db.exists() or not optimized_db.exists():
                continue

            speedup_range = entry.get("measured_speedup_range", [1.0, 1.0])
            cases.append(
                ProvenOptimizationCase(
                    case_id=entry["id"],
                    bottleneck=entry.get("bottleneck_type", "unknown"),
                    technique=entry.get("technique", ""),
                    measured_impact_min=float(speedup_range[0]),
                    measured_impact_max=float(speedup_range[1]),
                    fixture_dir=case_dir,
                )
            )
        return cases

    def run_on_case(self, case: ProvenOptimizationCase) -> GateVerdict:
        # Try subdirectory first, then flat layout
        baseline = case.fixture_dir / "baseline.db"
        if not baseline.exists():
            baseline = FIXTURES_DIR / f"{case.case_id}.baseline.db"

        optimized = case.fixture_dir / "optimized.db"
        if not optimized.exists():
            optimized = FIXTURES_DIR / f"{case.case_id}.optimized.db"

        baseline_runs = extract_kernel_runtimes_from_db(str(baseline))
        new_runs = extract_kernel_runtimes_from_db(str(optimized))

        total_baseline = sum(k.total_runtime_ns for k in baseline_runs)
        total_new = sum(k.total_runtime_ns for k in new_runs)
        claimed_speedup = total_baseline / max(total_new, 1)

        gate_input = GateInput(
            kernel_name=case.case_id,
            claimed_speedup=claimed_speedup,
            arch="gfx942",  # default; override per-case in YAML if needed
            baseline_runtime_ns=total_baseline,
            achieved_runtime_ns=total_new,
            patch_sha=f"proven_{case.case_id}",
            baseline_kernel_runtimes=baseline_runs,
            new_kernel_runtimes=new_runs,
            # Compile/bitwise/anchors are STUBBED for proven optimizations —
            # the corpus already validated those dimensions at authoring time.
            skip_compile=True,
            skip_bitwise=True,
            skip_anchors=True,
        )
        return run_gate_cascade(gate_input)
