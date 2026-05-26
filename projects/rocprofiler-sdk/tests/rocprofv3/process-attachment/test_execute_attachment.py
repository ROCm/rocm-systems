#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Execute live process-attachment scenarios (CTest execute fixtures)."""

from __future__ import annotations

import sys

import pytest

from attachment_utils import (
    MPI_ATTACH_STARTUP_SEC,
    build_mpi_simple_attach_cmd,
    build_openmp_attach_cmd,
    OPENMP_ATTACH_STARTUP_SEC,
    PMC_MULTITHREAD_APP_ARGS,
    PMC_MULTITHREAD_ATTACH_MSEC,
    PMC_ROCPD_SMOKE_APP_ARGS,
    PMC_ROCPD_SMOKE_ATTACH_MSEC,
    TRANSPOSE_LONG_PMC_APP_ARGS,
    TRANSPOSE_LONG_PMC_WAIT_AFTER_ATTACH_SEC,
    TRANSPOSE_PMC_APP_STARTUP_SEC,
    TRANSPOSE_PC_SAMPLING_APP_ARGS,
    TRANSPOSE_PC_SAMPLING_APP_STARTUP_SEC,
    assert_pc_sampling_attach_output,
    hip_rocpd_attach_extra,
    mpi_attach_env,
    mpi_attach_rocprofv3_extra,
    openmp_attach_rocprofv3_extra,
    openmp_attach_env,
    pc_sampling_selected_regions_attach_extra,
    pmc_rocpd_attach_extra,
    require_options,
    require_pc_sampling_available,
    rocprofv3_supports_attach_sync_output,
    run_process_attachment,
    skip_if_mpi_unavailable,
)


@pytest.fixture
def execution_config(request):
    opts = require_options(
        request.config,
        ("test_app", "rocprofv3", "output_dir", "rocprof_log_level", "output_name"),
    )
    return opts


def test_attach_hip_rocpd(execution_config):
    """Timed attach with HIP + kernel tracing and rocpd output."""
    run_process_attachment(
        app_cmd=[execution_config["test_app"], "2", "1"],
        rocprofv3=execution_config["rocprofv3"],
        output_dir=execution_config["output_dir"],
        log_level=execution_config["rocprof_log_level"],
        output_name=execution_config["output_name"],
        rocprofv3_extra=hip_rocpd_attach_extra(execution_config["rocprofv3"]),
    )


def test_attach_rocpd_sync(execution_config):
    """Timed attach with rocpd output and synchronous detach.

    HIP trace is required so at least one profiling service is active and rocpd
    output is generated (-f rocpd alone produces no output files).
    """
    rocprofv3 = execution_config["rocprofv3"]
    extra = ["--attach-duration-msec", "5000", "--hip-trace", "-f", "rocpd"]
    if rocprofv3_supports_attach_sync_output(rocprofv3):
        extra.append("--attach-sync-output")
    run_process_attachment(
        app_cmd=[execution_config["test_app"], "2", "1"],
        rocprofv3=rocprofv3,
        output_dir=execution_config["output_dir"],
        log_level=execution_config["rocprof_log_level"],
        output_name=execution_config["output_name"],
        rocprofv3_extra=extra,
    )


def test_attach_pmc_rocpd_smoke(execution_config):
    """Light PMC + rocpd attach smoke test (attachment-test, 5s attach).

    Intended to pass on fixed ROCm (e.g. install_18May) in normal CI.
    """
    rocprofv3 = execution_config["rocprofv3"]
    run_process_attachment(
        app_cmd=[execution_config["test_app"], *PMC_ROCPD_SMOKE_APP_ARGS],
        rocprofv3=rocprofv3,
        output_dir=execution_config["output_dir"],
        log_level=execution_config["rocprof_log_level"],
        output_name=execution_config["output_name"],
        rocprofv3_extra=pmc_rocpd_attach_extra(
            rocprofv3, duration_msec=PMC_ROCPD_SMOKE_ATTACH_MSEC
        ),
    )


def test_attach_pmc_rocpd_transpose_long(execution_config):
    """Live attach with PMC + rocpd on transpose under sustained GPU load.

    Uses the transpose benchmark with a large iteration count and a 60s attach
    window so detach must flush a large rocpd database (kernel trace + PMC).
    """
    rocprofv3 = execution_config["rocprofv3"]
    run_process_attachment(
        app_cmd=[execution_config["test_app"], *TRANSPOSE_LONG_PMC_APP_ARGS],
        rocprofv3=rocprofv3,
        output_dir=execution_config["output_dir"],
        log_level=execution_config["rocprof_log_level"],
        output_name=execution_config["output_name"],
        app_startup_sec=TRANSPOSE_PMC_APP_STARTUP_SEC,
        app_wait_after_attach_sec=TRANSPOSE_LONG_PMC_WAIT_AFTER_ATTACH_SEC,
        rocprofv3_extra=pmc_rocpd_attach_extra(rocprofv3),
    )


def test_attach_pc_sampling_transpose_selected_regions(execution_config):
    """Live attach with host-trap PC sampling and roctx selected-regions on transpose.

    transpose wraps its compute loop in roctxProfilerResume/Pause so --selected-regions
    collects kernel trace and PC samples only during that window.
    """
    rocprofv3 = execution_config["rocprofv3"]
    output_dir = execution_config["output_dir"]
    output_name = execution_config["output_name"]
    require_pc_sampling_available(output_dir, rocprofv3)
    run_process_attachment(
        app_cmd=[execution_config["test_app"], *TRANSPOSE_PC_SAMPLING_APP_ARGS],
        rocprofv3=rocprofv3,
        output_dir=output_dir,
        log_level=execution_config["rocprof_log_level"],
        output_name=output_name,
        app_startup_sec=TRANSPOSE_PC_SAMPLING_APP_STARTUP_SEC,
        rocprofv3_extra=pc_sampling_selected_regions_attach_extra(rocprofv3),
    )
    assert_pc_sampling_attach_output(output_dir, output_name)


def test_attach_pmc_rocpd_multithread(execution_config):
    """Live attach with PMC + rocpd while attachment-test runs many GPU threads.

    Stresses agent selection and PMC collection under concurrent kernel traffic
    (8 device threads) with a shorter attach duration than the transpose-long case.
    """
    rocprofv3 = execution_config["rocprofv3"]
    run_process_attachment(
        app_cmd=[execution_config["test_app"], *PMC_MULTITHREAD_APP_ARGS],
        rocprofv3=rocprofv3,
        output_dir=execution_config["output_dir"],
        log_level=execution_config["rocprof_log_level"],
        output_name=execution_config["output_name"],
        rocprofv3_extra=pmc_rocpd_attach_extra(
            rocprofv3, duration_msec=PMC_MULTITHREAD_ATTACH_MSEC
        ),
    )


def test_attach_mpi_simple_transpose(execution_config, request):
    """Live attach to mpiexec-launched mpi-simple-attach (2 ranks, long-running matrixTranspose).

    Attaches to the mpiexec process tree after ranks have started (see process-attachment docs).
    """
    mpiexec = request.config.getoption("--mpiexec")
    mpi_numproc_flag = request.config.getoption("--mpi-numproc-flag")
    skip_if_mpi_unavailable(execution_config["output_dir"], mpiexec)
    app_cmd = build_mpi_simple_attach_cmd(
        mpiexec, mpi_numproc_flag, execution_config["test_app"]
    )
    run_process_attachment(
        app_cmd=app_cmd,
        rocprofv3=execution_config["rocprofv3"],
        output_dir=execution_config["output_dir"],
        log_level=execution_config["rocprof_log_level"],
        output_name=execution_config["output_name"],
        app_startup_sec=MPI_ATTACH_STARTUP_SEC,
        app_env=mpi_attach_env(),
        rocprofv3_extra=mpi_attach_rocprofv3_extra(execution_config["rocprofv3"]),
    )


def test_attach_openmp_offload(execution_config):
    """Live attach to openmp-attach (long-running OpenMP offload, single stable PID)."""
    run_process_attachment(
        app_cmd=build_openmp_attach_cmd(execution_config["test_app"]),
        rocprofv3=execution_config["rocprofv3"],
        output_dir=execution_config["output_dir"],
        log_level=execution_config["rocprof_log_level"],
        output_name=execution_config["output_name"],
        app_startup_sec=OPENMP_ATTACH_STARTUP_SEC,
        app_env=openmp_attach_env(),
        rocprofv3_extra=openmp_attach_rocprofv3_extra(execution_config["rocprofv3"]),
    )


def test_attach_sys_trace_csv(execution_config):
    """Timed attach with sys-trace and CSV output (documentation example)."""
    run_process_attachment(
        app_cmd=[execution_config["test_app"], "2", "1"],
        rocprofv3=execution_config["rocprofv3"],
        output_dir=execution_config["output_dir"],
        log_level=execution_config["rocprof_log_level"],
        output_name=execution_config["output_name"],
        rocprofv3_extra=[
            "--attach-duration-msec",
            "5000",
            "--sys-trace",
            "--output-format",
            "csv",
        ],
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-vv", "-s"] + sys.argv[1:]))
