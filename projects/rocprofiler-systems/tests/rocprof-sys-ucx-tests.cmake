# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

# -------------------------------------------------------------------------------------- #
#
# UCX tests - MPI examples with UCX transport
#
# -------------------------------------------------------------------------------------- #

# UCX tests require MPI examples since UCX is MPI's transport layer
if(NOT ROCPROFSYS_USE_MPI AND NOT ROCPROFSYS_USE_MPI_HEADERS)
    return()
endif()

# Force OpenMPI to use UCX transport via environment variables
set(_ucxp_mpi_environment
    "OMPI_MCA_pml=ucx" # Use UCX point-to-point messaging layer
    "OMPI_MCA_osc=ucx" # Use UCX one-sided communications
    "OMPI_MCA_pml_ucx_tls=any" # Accept any UCX transport (tcp, self, sysv, posix, cma)
    "OMPI_MCA_pml_ucx_devices=any" # Accept any device (not just InfiniBand/Mellanox)
)

# Base environment for UCX tests
set(_ucx_base_environment
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_USE_SAMPLING=OFF"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=OFF"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_FILE_OUTPUT=ON"
    "ROCPROFSYS_USE_UCX=ON"
    "ROCPROFSYS_DEBUG=OFF"
    "ROCPROFSYS_VERBOSE=2"
    "ROCPROFSYS_DL_VERBOSE=2"
    "${_ucxp_mpi_environment}"
    "${_test_openmp_env}"
    "${_test_library_path}"
)

# First test: UCX availability check using mpi-example (basic test)
# This test checks if UCX is available. If not, subsequent UCX tests will be marked as skipped.
# Note: SKIP_BASELINE is required because baseline tests don't use the MCA parameter overrides
#       and will fail on systems without InfiniBand hardware (requires OMPI_MCA_pml_ucx_tls=any)
rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME
    NAME "ucx-availability-check"
    TARGET mpi-example
    MPI ON
    NUM_PROCS 2
    LABELS "ucx;availability"
    REWRITE_ARGS
        -e
        -v
        2
        --label
        file
        line
        return
        args
        --min-instructions
        0
    ENVIRONMENT "${_ucx_base_environment};ROCPROFSYS_VERBOSE=3"
    REWRITE_RUN_PASS_REGEX
        "UCX.*configured|ucp_|uct_|UCX transport|pml.*ucx"
    REWRITE_RUN_FAIL_REGEX
        "PML ucx cannot be selected|UCX is not available|No UCX support found|Failed to select|ROCPROFSYS_ABORT_FAIL_REGEX"
    REWRITE_RUN_SKIP_REGEX
        "PML ucx cannot be selected|UCX is not available|No UCX support found|Failed to select"
)

# Enhanced UCX environment with more detailed logging
set(_ucx_environment
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_USE_SAMPLING=OFF"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=OFF"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_FILE_OUTPUT=ON"
    "ROCPROFSYS_USE_UCX=ON"
    "ROCPROFSYS_DEBUG=ON"
    "ROCPROFSYS_VERBOSE=3"
    "ROCPROFSYS_DL_VERBOSE=3"
    "${_ucxp_mpi_environment}"
    "${_test_openmp_env}"
    "${_test_library_path}"
)

# UCX perfetto trace test
rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME
    NAME "ucx-perfetto"
    TARGET mpi-example
    MPI ON
    NUM_PROCS 2
    LABELS "ucx;perfetto"
    REWRITE_ARGS
        -e
        -v
        2
        --label
        file
        line
        --min-instructions
        0
    ENVIRONMENT "${_ucx_base_environment};ROCPROFSYS_VERBOSE=1"
    REWRITE_RUN_PASS_REGEX
        "(/[A-Za-z-]+/perfetto-trace-0.proto).*(/[A-Za-z-]+/wall_clock-0.txt')"
    REWRITE_RUN_FAIL_REGEX
        "(perfetto-trace|trip_count|sampling_percent|sampling_cpu_clock|sampling_wall_clock|wall_clock)-[0-9][0-9]+.(json|txt|proto)|ROCPROFSYS_ABORT_FAIL_REGEX"
)

# Validation test for UCX perfetto trace to ensure communication tracks are present
rocprofiler_systems_add_validation_test(
    NAME ucx-perfetto-sys-run
    PERFETTO_METRIC "ucx_comm_api"
    PERFETTO_FILE "merged.proto"
    LABELS "ucx;perfetto"
    ARGS --counter-names "UCX Comm Recv" "UCX Comm Send" -p
)

# Test all MPI example binaries with UCX transport
foreach(
    _UCX_EXAMPLE
    all2all
    allgather
    allreduce
    bcast
    reduce
    scatter-gather
    send-recv
)
    rocprofiler_systems_add_test(
        SKIP_BASELINE SKIP_RUNTIME SKIP_SAMPLING
        NAME "ucx-${_UCX_EXAMPLE}"
        TARGET mpi-${_UCX_EXAMPLE}
        MPI ON
        NUM_PROCS 2
        LABELS "ucx"
        REWRITE_ARGS -e -v 2 --label file line --min-instructions 0
        RUN_ARGS 30
        ENVIRONMENT "${_ucx_environment}"
        REWRITE_RUN_PASS_REGEX
            "UCX.*trace|ucp_.*trace|Category.*ucx|UCX function.*called"
    )

    # Add validation test to check for UCX communication tracks and bytes
    rocprofiler_systems_add_validation_test(
        NAME ucx-${_UCX_EXAMPLE}-sys-run
        PERFETTO_METRIC "ucx_comm_api"
        PERFETTO_FILE "merged.proto"
        LABELS "ucx"
        ARGS --counter-names "UCX Comm Recv" "UCX Comm Send" -p
    )
endforeach()

# UCX with MPIP integration test
rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME
    NAME "ucx-mpip-integration"
    TARGET mpi-all2all
    MPI ON
    NUM_PROCS 2
    LABELS "ucx;mpip"
    REWRITE_ARGS
        -e
        -v
        2
        --label
        file
        line
        args
        --min-instructions
        0
    ENVIRONMENT
        "${_ucx_environment};ROCPROFSYS_USE_MPIP=ON"
    RUN_ARGS 30
    REWRITE_RUN_PASS_REGEX
        "UCX.*trace.*MPI.*trace|ucp_.*MPI_|Category.*ucx.*Category.*mpi"
)

# UCX detailed function tracing test
rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME
    NAME "ucx-detailed-tracing"
    TARGET mpi-send-recv
    MPI ON
    NUM_PROCS 2
    LABELS "ucx;detailed"
    REWRITE_ARGS
        -e
        -v
        3
        --label
        file
        line
        args
        --min-instructions
        0
    ENVIRONMENT
        "${_ucx_environment};ROCPROFSYS_VERBOSE=3;ROCPROFSYS_DL_VERBOSE=3"
    RUN_ARGS 100
    REWRITE_RUN_PASS_REGEX
        "ucp_tag_send_nbx|ucp_tag_recv_nbx|ucp_worker_progress|ucp_request.*|UCX Gotcha.*audit"
)

# Validation test for detailed UCX tracing to ensure communication tracking
rocprofiler_systems_add_validation_test(
    NAME ucx-detailed-tracing-sys-run
    PERFETTO_METRIC "ucx_comm_api"
    PERFETTO_FILE "merged.proto"
    LABELS "ucx;detailed"
    ARGS --counter-names "UCX Comm Recv" "UCX Comm Send" -p
)

# UCX memory operations test
rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME
    NAME "ucx-memory-ops"
    TARGET mpi-allgather
    MPI ON
    NUM_PROCS 2
    LABELS "ucx;memory"
    REWRITE_ARGS
        -e
        -v
        2
        --label
        file
        line
        --min-instructions
        0
    ENVIRONMENT "${_ucx_environment}"
    RUN_ARGS 50
    REWRITE_RUN_PASS_REGEX
        "ucp_put|ucp_get|ucp_mem_|ucp_rkey_|ucp_atomic_"
)

# Validation test for UCX memory operations to ensure bytes are recorded
rocprofiler_systems_add_validation_test(
    NAME ucx-memory-ops-sys-run
    PERFETTO_METRIC "ucx_comm_api"
    PERFETTO_FILE "merged.proto"
    LABELS "ucx;memory"
    ARGS --counter-names "UCX Comm Recv" "UCX Comm Send" -p
)

# UCX with different message sizes
foreach(_MSG_SIZE 1024 4096 16384)
    rocprofiler_systems_add_test(
        SKIP_BASELINE SKIP_RUNTIME
        NAME "ucx-bcast-${_MSG_SIZE}"
        TARGET mpi-bcast
        MPI ON
        NUM_PROCS 2
        LABELS "ucx;bcast"
        REWRITE_ARGS
            -e
            -v
            2
            --label
            file
            line
            --min-instructions
            0
        ENVIRONMENT "${_ucx_environment}"
        RUN_ARGS ${_MSG_SIZE}
        REWRITE_RUN_PASS_REGEX
            "UCX.*trace|ucp_.*send|ucp_.*recv|Category.*ucx"
    )
endforeach()

# Test UCX active message functionality
rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME
    NAME "ucx-active-messages"
    TARGET mpi-allreduce
    MPI ON
    NUM_PROCS 2
    LABELS "ucx;am"
    REWRITE_ARGS
        -e
        -v
        2
        --label
        file
        line
        --min-instructions
        0
    ENVIRONMENT "${_ucx_environment};OMPI_MCA_btl=^vader,tcp,openib,uct"
    RUN_ARGS 64
    REWRITE_RUN_PASS_REGEX
        "ucp_am_send|ucp_am_recv|uct_ep_am|Active.*Message"
)
