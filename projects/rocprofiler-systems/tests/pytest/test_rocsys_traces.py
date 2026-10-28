# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``rocsys --traces`` resolution, conflicts, and legacy-flag coverage."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
from conftest import RocprofsysTest
from rocprofsys.config import RocprofsysConfig

pytestmark = [pytest.mark.rocsys, pytest.mark.rocprof_binary]

_MIN_ZERO = re.compile(r"(--[A-Za-z][\w-]*)\s+\(min:\s*0")
_TRACE_RUNTIMES = {"mpi", "openmp", "rccl", "ucx", "shmem"}
_VALUE_FLAGS = (
    ["--rocm=hip"],
    ["--rocm-domains", "hip_runtime_api"],
    ["--parallel=mpi"],
    ["-I", "mpip"],
    ["-E", "ompt"],
    ["-I", "all"],
    ["--preset=trace-hpc"],
    ["--trace-hpc"],
)


def _load_export(text: str) -> dict | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        return None


def _trace_slice(config: dict | None) -> tuple:
    """ROCm API domains and Phase 2 host runtimes. Queue grouping is not a --traces setting."""
    if not config:
        return (None, None)
    domains = config.get("domains") or {}
    api_domains = (domains.get("rocm") or {}).get("api_domains")
    runtimes = (domains.get("parallel") or {}).get("runtimes") or {}
    kept = {name: runtimes.get(name) for name in _TRACE_RUNTIMES if name in runtimes}
    return (api_domains, kept or None)


def _mpi_is_on() -> bool:
    value = os.environ.get("ROCPROFSYS_USE_MPIP")
    if value is None or not value.strip():
        return True
    return value.strip().lower() not in {"0", "false", "off", "no", "n", "f"}


@pytest.mark.class_name("rocsys-traces")
class TestRocsysTraces(RocprofsysTest):
    """Dry-run and error paths for ``--traces``. No GPU."""

    target = "rocsys"

    @pytest.mark.timeout(60)
    def test_dry_run_replaces_defaults(self) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--dry-run", "--traces", "hip,kernel", "--", "/bin/true"],
            fail_on_not_found=True,
        )
        notes = [r"MPI \(MPIP\)"] if _mpi_is_on() else []
        self.assert_regex(
            result,
            pass_regex=[
                r"verb: profile",
                r"ROCPROFSYS_ROCM_DOMAINS=hip_api,kernel_dispatch origin=--traces embed=false",
                r"ROCPROFSYS_USE_MPIP=false origin=--traces embed=false",
                r"ROCPROFSYS_USE_OMPT=false origin=--traces embed=false",
                *notes,
            ],
            fail_regex=[r"ROCPROFSYS_USE_KOKKOSP", r"\bkokkos\b"],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(60)
    def test_runtime_only_clears_rocm_domains(self) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--dry-run", "--traces", "mpi", "--", "/bin/true"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"ROCPROFSYS_ROCM_DOMAINS= origin=--traces embed=false",
                r"ROCPROFSYS_USE_MPIP=true origin=--traces embed=false",
            ],
            fail_regex=[r"MPI \(MPIP\)"],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "args, argv_regex, embed",
        [
            (
                ["instrument", "--dry-run", "--traces", "kernel", "--", "/bin/true"],
                r"rocprof-sys-instrument -- /bin/true",
                "false",
            ),
            (
                ["python", "--dry-run", "--traces", "hip", "--", "script.py"],
                r"rocprof-sys-python -- script.py",
                "false",
            ),
            (
                ["attach", "--dry-run", "--traces", "kernel", "-p", "1"],
                r"rocprof-sys-attach -p 1",
                "false",
            ),
            (
                ["rewrite", "--dry-run", "--traces", "kernel", "--", "/bin/true"],
                r"--env ROCPROFSYS_ROCM_DOMAINS=kernel_dispatch",
                "true",
            ),
        ],
    )
    def test_every_tracing_verb_resolves(
        self, args: list[str], argv_regex: str, embed: str
    ) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=args,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[argv_regex, rf"embed={embed}"],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(30)
    def test_rewrite_labels_embedded_defaults(self) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["rewrite", "--dry-run", "--traces", "mpi", "--", "/bin/true"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"default in binary",
                r"--env ROCPROFSYS_ROCM_DOMAINS=",
                r"ROCPROFSYS_USE_MPIP=true origin=--traces embed=true",
            ],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(30)
    @pytest.mark.parametrize(
        "args, pattern",
        [
            (["--traces", "hpp", "--", "/bin/true"], r"did you mean 'hip'"),
            (["--traces", "hip", "/bin/true"], r"missing '--'"),
            (
                ["--traces", "hip", "--preset=balanced", "--", "/bin/true"],
                r"use one or the other",
            ),
            (
                ["--traces", "hip", "--balanced", "--", "/bin/true"],
                r"--balanced conflicts with --traces",
            ),
            (
                ["--traces", "hip", "--export-config", "--", "/bin/true"],
                r"use --dry-run",
            ),
            (
                ["causal", "--traces", "hip", "--", "/bin/true"],
                r"not available for 'causal'",
            ),
            (["avail", "--traces", "hip"], r"does not take a value"),
            (["--traces", "--", "/bin/true"], r"missing value for --traces"),
        ],
    )
    def test_rejects_bad_traces(self, args: list[str], pattern: str) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=args,
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=[pattern], use_abort_fail_regex=False)

    @pytest.mark.timeout(30)
    def test_kokkos_flag_and_unrelated_env_are_not_conflicts(self) -> None:
        kept = self.run_test(
            "baseline",
            target=self.target,
            run_args=[
                "--dry-run",
                "--traces",
                "hip",
                "--use-kokkosp",
                "--",
                "/bin/true",
            ],
            fail_on_not_found=True,
        )
        self.assert_regex(
            kept,
            pass_regex=[r"ROCPROFSYS_ROCM_DOMAINS=hip_api"],
            use_abort_fail_regex=False,
        )
        other = self.run_test(
            "baseline",
            target=self.target,
            run_args=[
                "instrument",
                "--dry-run",
                "--traces",
                "hip",
                "--env",
                "ROCPROFSYS_VERBOSE=1",
                "--",
                "/bin/true",
            ],
            fail_on_not_found=True,
        )
        self.assert_regex(
            other,
            pass_regex=[r"ROCPROFSYS_VERBOSE=1"],
            use_abort_fail_regex=False,
        )

    @pytest.mark.timeout(30)
    def test_instrument_env_that_writes_a_trace_key_conflicts(self) -> None:
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=[
                "instrument",
                "--traces",
                "hip",
                "--env",
                "ROCPROFSYS_ROCM_DOMAINS=hip_api",
                "--",
                "/bin/true",
            ],
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"--env conflicts with --traces"],
            use_abort_fail_regex=False,
        )


def _run(binary: Path, args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(binary), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )


@pytest.mark.timeout(180)
@pytest.mark.rocsys
def test_flags_that_write_trace_settings_have_a_conflict(
    rocprof_config: RocprofsysConfig,
) -> None:
    """Every legacy run flag that changes a --traces setting must be rejected."""
    run = rocprof_config.get_target_executable("rocprof-sys-run")
    rocsys = rocprof_config.get_target_executable("rocsys")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = rocprof_config.get_library_path()

    help_text = _run(run, ["--help=all"], env)
    assert help_text.returncode == 0, help_text.stderr
    flags = [[match] for match in _MIN_ZERO.findall(help_text.stdout)]
    flags.extend(_VALUE_FLAGS)

    baseline = _load_export(_run(run, ["--export-config", "--", "/bin/true"], env).stdout)
    assert baseline is not None
    base_slice = _trace_slice(baseline)
    offenders: list[str] = []
    for args in flags:
        if args == ["--export-config"]:
            continue
        try:
            exported = _run(run, [*args, "--export-config", "--", "/bin/true"], env)
        except subprocess.TimeoutExpired:
            continue
        got = _load_export(exported.stdout)
        if exported.returncode != 0 or got is None:
            continue
        if _trace_slice(got) == base_slice:
            continue
        rejected = _run(
            rocsys, ["--dry-run", "--traces", "hip", *args, "--", "/bin/true"], env
        )
        if rejected.returncode == 0 or "conflicts with --traces" not in (
            rejected.stdout + rejected.stderr
        ):
            offenders.append(" ".join(args))
    assert not offenders, f"flags write a --traces setting and are not conflicts: {offenders}"
