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
            case_dir = FIXTURES_DIR / entry["id"]
            if not case_dir.exists():
                continue  # allow partial corpus in early development
            cases.append(
                ProvenOptimizationCase(
                    case_id=entry["id"],
                    bottleneck=entry["bottleneck"],
                    technique=entry["technique"],
                    measured_impact_min=float(entry["measured_impact"]["min"]),
                    measured_impact_max=float(entry["measured_impact"]["max"]),
                    fixture_dir=case_dir,
                )
            )
        return cases

    def run_on_case(self, case: ProvenOptimizationCase) -> GateVerdict:
        baseline = case.fixture_dir / "baseline.db"
        optimized = case.fixture_dir / "optimized.db"

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
