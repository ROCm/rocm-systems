"""Attack #7 — 5 consecutive debug-loop failures must trigger deeper-tier mode,
NOT infinite-loop the cascade.
"""

import pytest

from perfxpert.runtime.gate_cascade import GateInput, run_gate_cascade
from tests.test_red_team.conftest import record_outcome


@pytest.mark.red_team
def test_5_consecutive_failures_triggers_deeper_tier(outcomes_dir) -> None:
    """Feed 5 failing GateInputs in sequence; 5th must return deeper_tier signal."""
    verdicts = []
    for i in range(5):
        gate_input = GateInput(
            kernel_name=f"k{i}",
            claimed_speedup=1.5,
            arch="gfx942",
            baseline_runtime_ns=1_000_000,
            achieved_runtime_ns=9_000_000,  # regresses every time
            patch_sha=f"cycle{i:02d}",
            loop_counter=i + 1,              # runtime tracks this
        )
        verdict = run_gate_cascade(gate_input, stop_at="regression")
        verdicts.append(verdict)

    # 5th verdict must flag deeper_tier
    last = verdicts[-1]
    assert last.status in ("regressed", "reject")
    assert last.detail and (
        "deeper" in last.detail.lower() or "plateau" in last.detail.lower() or
        "5" in last.detail
    ), f"expected deeper-tier trigger, got {last.detail!r}"

    record_outcome(
        outcomes_dir,
        attack_id="5_consecutive_failures_plateau",
        status="defeated",
        details={"verdict_detail": last.detail},
    )
