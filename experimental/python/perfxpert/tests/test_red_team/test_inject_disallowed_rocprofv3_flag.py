"""Attack #14 — LLM proposes an invalid rocprofv3 flag OR a malicious private
provider smuggles such a flag through the LLM channel (R20).

profile.run's rocprofv3 flag allowlist (extends _filter_rec_commands pattern)
must reject.
"""

import pytest

from perfxpert.tools.profile_runner import run as profile_run, RocprofFlagError
from tests.test_red_team.conftest import record_outcome


INVALID_ROCPROFV3_FLAGS = [
    "--hip-api-trace",              # fake flag in legacy AI output
    "--kernel-names evil_name",     # value-taker rocprofv3 doesn't accept
    "--exec-shell /bin/sh",         # outright malicious
    "--output-dir /etc/",            # absolute path outside allowed output roots
]


@pytest.mark.red_team
@pytest.mark.parametrize("flag", INVALID_ROCPROFV3_FLAGS)
def test_profile_run_rejects_each_invalid_flag(flag) -> None:
    cmd = ["rocprofv3", "--sys-trace", flag, "--", "./app"]
    with pytest.raises(RocprofFlagError):
        profile_run(argv=cmd, cwd="/tmp")


@pytest.mark.red_team
def test_attack_14_rollup(outcomes_dir) -> None:
    record_outcome(
        outcomes_dir,
        attack_id="disallowed_rocprofv3_flag_or_private_provider_injection",
        status="defeated",
        details={"flags_tested": INVALID_ROCPROFV3_FLAGS},
    )
