#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""GPU-free unit tests for the rocprofv3 MPI rank/size environment handling."""

import os
import sys

import pytest

TOOL_LIBRARY = "lib/rocprofiler-sdk/librocprofiler-sdk-tool.so"

# resolve_library_path() only checks that these exist, so empty files are enough.
STUB_LIBRARIES = (
    TOOL_LIBRARY,
    "lib/librocprofiler-sdk.so",
    "lib/librocprofiler-sdk-roctx.so",
    "lib/rocprofiler-sdk/librocprofiler-sdk-tool-kokkosp.so",
    "lib/rocprofiler-sdk/librocprofv3-list-avail.so",
)

# Every rank/size pair get_mpi_rank_and_size() consults, in precedence order.
AUTODETECTED_VARIABLES = (
    "PBS_NODENUM",
    "PBS_O_TASKNUM",
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "PMI_RANK",
    "PMI_SIZE",
    "MV2_COMM_WORLD_RANK",
    "MV2_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "MPI_RANKID",
    "MPI_NRANKS",
    "MPI_LOCALRANKID",
    "MPI_LOCALNRANKS",
    "MPI_RANK",
    "MPI_SIZE",
)

PAIR_REQUIRED_ERROR = (
    "[rocprofv3] Fatal error: When using custom MPI environment variables, "
    "both --mpi-world-rank-variable and --mpi-world-size-variable must be specified\n"
)


class _Launched(Exception):
    """Raised in place of the exec that would replace the test process."""


@pytest.fixture(autouse=True)
def clean_mpi_env(monkeypatch):
    # A launcher or scheduler in the ambient environment would otherwise be
    # indistinguishable from the variables under test.
    for name in AUTODETECTED_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ROCPROF_MPI_RANKS", raising=False)
    monkeypatch.delenv("ROCPROF_MPI_RANK_VAR", raising=False)
    monkeypatch.delenv("ROCPROF_MPI_SIZE_VAR", raising=False)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.delenv("ROCPROF_PRELOAD", raising=False)


@pytest.fixture
def rocm_root(tmp_path):
    root = tmp_path / "rocm"
    for relpath in STUB_LIBRARIES:
        library = root / relpath
        library.parent.mkdir(parents=True, exist_ok=True)
        library.touch()
    return root


@pytest.fixture
def launch(rocprofv3, rocm_root, monkeypatch):
    """Run the launcher and return the environment it would hand to the application."""

    def _launch(*argv):
        captured = {}

        def execvpe(file, args, env):
            captured.update(env)
            raise _Launched

        monkeypatch.setattr(rocprofv3.os, "execvpe", execvpe)

        with pytest.raises(_Launched):
            rocprofv3.main(
                [
                    "--rocm-root",
                    str(rocm_root),
                    *argv,
                    "--kernel-trace",
                    "--",
                    "/bin/true",
                ]
            )

        return captured

    return _launch


@pytest.fixture
def expect_fatal(rocprofv3, tmp_path, capsys):
    """Run the launcher expecting a fatal error, and return nothing written."""

    def _expect_fatal(argv, message):
        output_path = tmp_path / "output"

        with pytest.raises(SystemExit) as exc_info:
            rocprofv3.main([*argv, "-d", str(output_path), "--", "/bin/true"])

        assert exc_info.value.code == 1
        assert capsys.readouterr().err == message
        assert not output_path.exists()

    return _expect_fatal


def test_openmpi_variables_are_auto_detected(launch, monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")

    env = launch()

    assert env["ROCPROF_MPI_RANK_VAR"] == "OMPI_COMM_WORLD_RANK"
    assert env["ROCPROF_MPI_SIZE_VAR"] == "OMPI_COMM_WORLD_SIZE"


def test_slurm_takes_precedence_over_openmpi(launch, monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "1")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "3")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "8")

    env = launch()

    assert env["ROCPROF_MPI_RANK_VAR"] == "SLURM_PROCID"
    assert env["ROCPROF_MPI_SIZE_VAR"] == "SLURM_NTASKS"


def test_custom_rank_is_selected(launch, rocm_root, monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "3")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "8")
    monkeypatch.setenv("MY_RANK", "2")
    monkeypatch.setenv("MY_SIZE", "4")

    # Selecting only the custom rank proves the override drives rank selection, not
    # merely which variable names are forwarded.
    env = launch(
        "--mpi-world-rank-variable",
        "MY_RANK",
        "--mpi-world-size-variable",
        "MY_SIZE",
        "--profile-mpi-ranks",
        "2",
    )

    assert env["ROCPROF_MPI_RANK_VAR"] == "MY_RANK"
    assert env["ROCPROF_MPI_SIZE_VAR"] == "MY_SIZE"
    assert str(rocm_root / TOOL_LIBRARY) in env["LD_PRELOAD"]


def test_auto_detected_rank_is_ignored(launch, monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "3")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "8")
    monkeypatch.setenv("MY_RANK", "2")
    monkeypatch.setenv("MY_SIZE", "4")

    env = launch(
        "--mpi-world-rank-variable",
        "MY_RANK",
        "--mpi-world-size-variable",
        "MY_SIZE",
        "--profile-mpi-ranks",
        "3",
    )

    assert "LD_PRELOAD" not in env


def test_no_mpi_variables_without_mpi_env(launch):
    env = launch()

    assert "ROCPROF_MPI_RANK_VAR" not in env
    assert "ROCPROF_MPI_SIZE_VAR" not in env


def test_absent_custom_variables_still_profile(launch, rocm_root):
    env = launch(
        "--mpi-world-rank-variable",
        "MISSING_RANK",
        "--mpi-world-size-variable",
        "MISSING_SIZE",
        "--profile-mpi-ranks",
        "0",
    )

    # Rank is undetectable, so the launcher assumes this rank provides output.
    assert env["ROCPROF_MPI_RANK_VAR"] == "MISSING_RANK"
    assert env["ROCPROF_MPI_SIZE_VAR"] == "MISSING_SIZE"
    assert str(rocm_root / TOOL_LIBRARY) in env["LD_PRELOAD"]


def test_selected_rank_is_profiled(launch, rocm_root, monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")

    env = launch("--profile-mpi-ranks", "0-1")

    assert str(rocm_root / TOOL_LIBRARY) in env["LD_PRELOAD"]


def test_unselected_rank_is_not_profiled(launch, monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")

    env = launch("--profile-mpi-ranks", "0-1")

    # The application is executed before any instrumentation is added, so asserting the
    # variables are absent rather than merely tool-free keeps this failing closed.
    assert "LD_PRELOAD" not in env
    assert "ROCP_TOOL_LIBRARIES" not in env
    assert env["ROCPROF_MPI_RANK_VAR"] == "OMPI_COMM_WORLD_RANK"


CUSTOM_VARIABLES = {"MY_RANK": "0", "MY_SIZE": "4"}
OPENMPI_VARIABLES = {"OMPI_COMM_WORLD_RANK": "0", "OMPI_COMM_WORLD_SIZE": "4"}

CUSTOM_VARIABLE_ARGS = [
    "--mpi-world-rank-variable",
    "MY_RANK",
    "--mpi-world-size-variable",
    "MY_SIZE",
]


@pytest.mark.parametrize(
    "env,argv,expected",
    [
        ({}, ["--mpi-world-rank-variable", "MY_RANK"], PAIR_REQUIRED_ERROR),
        ({}, ["--mpi-world-size-variable", "MY_SIZE"], PAIR_REQUIRED_ERROR),
        (
            CUSTOM_VARIABLES,
            [*CUSTOM_VARIABLE_ARGS, "--profile-mpi-ranks", "7"],
            "[rocprofv3] Fatal error: Invalid rank specification: rank 7 is out of "
            "range. MPI world size is 4 (valid ranks: 0-3)\n",
        ),
        (
            OPENMPI_VARIABLES,
            ["--profile-mpi-ranks", "3-1"],
            "[rocprofv3] Fatal error: Invalid range: 3-1 (start > end)\n",
        ),
        (
            OPENMPI_VARIABLES,
            ["--profile-mpi-ranks", "0,abc"],
            "[rocprofv3] Fatal error: Invalid rank specification 'abc': not a valid "
            "integer or range\n",
        ),
    ],
)
def test_invalid_mpi_options_fail_before_launch(
    expect_fatal, monkeypatch, env, argv, expected
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    expect_fatal(argv, expected)


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
