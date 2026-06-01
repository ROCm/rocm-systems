"""Attack #8 — LLM unavailable MUST yield identical gate decisions to LLM mode.

This is the air-gap invariant re-asserted under the red-team umbrella
(spec §5 air-gap invariant, §7 exit criterion 3).
"""

import os
from unittest.mock import patch

import pytest

from perfxpert.runtime.gate_cascade import GateInput, run_gate_cascade
from tests.test_red_team.conftest import record_outcome


@pytest.mark.red_team
def test_gate_decisions_identical_with_and_without_llm(outcomes_dir) -> None:
    gate_input = GateInput(
        kernel_name="parity_kernel",
        claimed_speedup=1.15,
        arch="gfx942",
        baseline_runtime_ns=1_000_000,
        achieved_runtime_ns=870_000,
        patch_sha="airgap01",
    )

    # Run with LLM 'available' (mode default)
    verdict_llm_on = run_gate_cascade(gate_input)

    # Run with LLM forced off
    with patch.dict(os.environ, {"PERFXPERT_AIRGAP": "1"}):
        verdict_llm_off = run_gate_cascade(gate_input)

    assert verdict_llm_on.status == verdict_llm_off.status
    assert verdict_llm_on.failing_gate == verdict_llm_off.failing_gate
    # Narrative may differ; status + failing_gate MUST match bit-for-bit.

    record_outcome(
        outcomes_dir,
        attack_id="llm_unavailable_airgap_parity",
        status="defeated",
        details={
            "llm_on": {"status": verdict_llm_on.status, "gate": verdict_llm_on.failing_gate},
            "llm_off": {"status": verdict_llm_off.status, "gate": verdict_llm_off.failing_gate},
        },
    )
