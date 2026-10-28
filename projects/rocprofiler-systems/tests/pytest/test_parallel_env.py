# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lock the legacy parallel-runtime environment exported by run and sample."""

from __future__ import annotations

import json

import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.rocsys, pytest.mark.rocprof_binary]


def _exported_runtimes(output: str) -> dict[str, dict[str, bool]] | None:
    start = output.find("{")
    assert start >= 0, output
    config = json.loads(output[start:])
    runtimes = config.get("domains", {}).get("parallel", {}).get("runtimes")
    if runtimes is None:
        return None
    return runtimes


# Captured from rocprof-sys-run/sample --export-config before the shared table.
# Only the parallel section is compared; the rest of the export is machine-specific.
_PARALLEL_CASES = [
    pytest.param(
        "rocprof-sys-run",
        ["--export-config", "--", "/bin/true"],
        None,
        id="run-default",
    ),
    pytest.param(
        "rocprof-sys-run",
        ["--parallel", "--export-config", "--", "/bin/true"],
        {
            "kokkos": {"enabled": True},
            "mpi": {"enabled": True},
            "openmp": {"enabled": True},
            "rccl": {"enabled": True},
        },
        id="run-parallel",
    ),
    pytest.param(
        "rocprof-sys-run",
        ["--parallel=mpi,openmp", "--export-config", "--", "/bin/true"],
        {"mpi": {"enabled": True}, "openmp": {"enabled": True}},
        id="run-parallel-list",
    ),
    pytest.param(
        "rocprof-sys-run",
        ["--parallel=all", "--export-config", "--", "/bin/true"],
        {
            "kokkos": {"enabled": True},
            "mpi": {"enabled": True},
            "openmp": {"enabled": True},
            "rccl": {"enabled": True},
        },
        id="run-parallel-all",
    ),
    pytest.param(
        "rocprof-sys-run",
        ["--parallel=ucx,shmem", "--export-config", "--", "/bin/true"],
        None,
        id="run-parallel-not-in-legacy",
    ),
    pytest.param(
        "rocprof-sys-run",
        ["-I", "mpip", "--export-config", "--", "/bin/true"],
        {"mpi": {"enabled": True}},
        id="run-include-mpip",
    ),
    pytest.param(
        "rocprof-sys-run",
        ["-E", "mpip", "--export-config", "--", "/bin/true"],
        {"mpi": {"enabled": False}},
        id="run-exclude-mpip",
    ),
    pytest.param(
        "rocprof-sys-sample",
        ["--parallel=mpi,rccl", "--export-config", "--", "/bin/true"],
        {"mpi": {"enabled": True}, "rccl": {"enabled": True}},
        id="sample-parallel-list",
    ),
]


@pytest.mark.class_name("parallel-env")
class TestParallelEnv(RocprofsysTest):
    """`--export-config` results for the host runtimes must stay unchanged."""

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("binary, args, expected", _PARALLEL_CASES)
    def test_export_config_parallel_runtimes(
        self, binary: str, args: list[str], expected: dict[str, dict[str, bool]] | None
    ) -> None:
        result = self.run_test(
            "baseline",
            target=binary,
            run_args=args,
            no_base_env=True,
            fail_on_not_found=True,
        )
        assert _exported_runtimes(result.test_output) == expected
