from contextlib import contextmanager
from pathlib import Path
import tempfile

import consan_validation as validation

# The top-k exception was retired after the runtime identity bug was fixed.
# This is the sole producer-shaped fixture for the generic coverage-output
# parser and bound validation; it is not a declaration for the current top-k
# workload and the archived diagnostic logs are parser-format fixtures, not
# reproducers for a still-expected runtime result.
RETIRED_TOPK_CODE_OBJECT_FINGERPRINT = "fnv1a64:3833562345afa454"
RETIRED_TOPK_RECORD_REPLAY_INSTRUCTION_GROUPS = (
    (0xFE964, 0xFE96C, 0xFE974, 0xFE97C),
    (0xFE9C4, 0xFE9CC, 0xFE9D4, 0xFE9DC),
    (0xFEA68, 0xFEA70, 0xFEA78, 0xFEA80),
    (0xFEB40, 0xFEB48, 0xFEB50, 0xFEB58),
)
RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT = validation.CoverageOutputContract(
    profile="record-replay",
    diagnostics=("exact-lds-write-write",),
    max_diagnostics=4,
    instruction_groups=RETIRED_TOPK_RECORD_REPLAY_INSTRUCTION_GROUPS,
    code_object_fingerprint=RETIRED_TOPK_CODE_OBJECT_FINGERPRINT,
    tracking_issue="bd-test",
    withhold_fault_qualification=True,
    fault_qualification_withheld_reason="test-only exception",
)


@contextmanager
def temporary_root():
    with tempfile.TemporaryDirectory() as temporary:
        yield Path(temporary)
