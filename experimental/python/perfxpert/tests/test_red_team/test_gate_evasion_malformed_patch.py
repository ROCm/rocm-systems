"""Attack #2 — malformed patch: diff applies cleanly but fails to compile.

Exercises runtime/gate_cascade.py GATE 1 (compile.build).
"""

import textwrap
from pathlib import Path

import pytest

from perfxpert.runtime.gate_cascade import GateInput, run_gate_cascade
from tests.test_red_team.conftest import record_outcome

MALFORMED_DIFF = textwrap.dedent(
    """\
    --- a/main.cpp
    +++ b/main.cpp
    @@ -1,3 +1,5 @@
     int main() {
    +    this_is_not_valid_cpp_syntax @@@;
    +    float x = ;
     return 0;
     }
    """
)


@pytest.mark.red_team
def test_compile_rejects_malformed_patch(outcomes_dir, tmp_path: Path) -> None:
    source = tmp_path / "main.cpp"
    source.write_text("int main() { return 0; }\n")

    gate_input = GateInput(
        kernel_name="main",
        claimed_speedup=1.1,
        arch="gfx942",
        baseline_runtime_ns=1_000_000,
        achieved_runtime_ns=900_000,
        patch_sha="malformed01",
        source_file=source,
        diff_payload=MALFORMED_DIFF,
        project_root=tmp_path,
    )

    verdict = run_gate_cascade(gate_input, stop_at="compile")
    assert verdict.status == "reject"
    assert verdict.failing_gate == "compile"

    record_outcome(
        outcomes_dir,
        attack_id="malformed_patch_compile_fail",
        status="defeated",
        details={"verdict_status": verdict.status, "failing_gate": verdict.failing_gate},
    )
