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
# MPI tests
#
# -------------------------------------------------------------------------------------- #

if(NOT ROCPROFSYS_USE_MPI AND NOT ROCPROFSYS_USE_MPI_HEADERS)
    return()
endif()

rocprofiler_systems_add_test(
    SKIP_RUNTIME
    NAME "mpi"
    TARGET mpi-example
    MPI ON
    NUM_PROCS 2
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
    ENVIRONMENT "${_base_environment};ROCPROFSYS_VERBOSE=1"
    REWRITE_RUN_PASS_REGEX
        "(/[A-Za-z-]+/perfetto-trace-0.proto).*(/[A-Za-z-]+/wall_clock-0.txt')"
    REWRITE_RUN_FAIL_REGEX
        "Outputting.*(perfetto-trace|trip_count|sampling_percent|sampling_cpu_clock|sampling_wall_clock|wall_clock)-[0-9][0-9]+.(json|txt|proto)|ROCPROFSYS_ABORT_FAIL_REGEX"
)

# mpi-perfetto-merge requires legacy trace mode because MPI trace combining
# uses MPI communication (mpi_get) which is only implemented in the legacy path
rocprofiler_systems_add_test(
    SKIP_RUNTIME
    NAME "mpi-perfetto-merge"
    TARGET mpi-example
    MPI ON
    NUM_PROCS 2
    LABELS "perfetto-merge"
    REWRITE_ARGS
        -e
        -v
        2
        --label
        file
        line
        --min-instructions
        0
    ENVIRONMENT
        "${_base_environment};ROCPROFSYS_VERBOSE=1;ROCPROFSYS_TRACE_LEGACY=ON;ROCPROFSYS_PERFETTO_COMBINE_TRACES=ON"
    REWRITE_RUN_PASS_REGEX
        "Successfully executed: .+rocprof-sys-merge-output.sh.*"
    REWRITE_RUN_FAIL_REGEX
        "Script not found|Failed to execute|ROCPROFSYS_ABORT_FAIL_REGEX"
)

rocprofiler_systems_add_test(
    SKIP_RUNTIME
    NAME "mpi-flat-mpip"
    TARGET mpi-example
    MPI ON
    NUM_PROCS 2
    LABELS "mpip"
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
        "${_flat_environment};ROCPROFSYS_USE_SAMPLING=OFF;ROCPROFSYS_STRICT_CONFIG=OFF;ROCPROFSYS_USE_MPIP=ON"
    REWRITE_RUN_PASS_REGEX
        ">>> mpi-flat-mpip.inst(.*\n.*)>>> MPI_Init_thread(.*\n.*)>>> pthread_create(.*\n.*)>>> MPI_Comm_size(.*\n.*)>>> MPI_Comm_rank(.*\n.*)>>> MPI_Barrier(.*\n.*)>>> MPI_Alltoall"
)

rocprofiler_systems_add_test(
    SKIP_RUNTIME
    NAME "mpi-flat"
    TARGET mpi-example
    MPI ON
    NUM_PROCS 2
    LABELS "mpip"
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
    ENVIRONMENT "${_flat_environment};ROCPROFSYS_USE_SAMPLING=OFF"
    REWRITE_RUN_PASS_REGEX
        ">>> mpi-flat.inst(.*\n.*)>>> MPI_Init_thread(.*\n.*)>>> pthread_create(.*\n.*)>>> MPI_Comm_size(.*\n.*)>>> MPI_Comm_rank(.*\n.*)>>> MPI_Barrier(.*\n.*)>>> MPI_Alltoall"
)

set(_mpip_environment
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_USE_SAMPLING=OFF"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=OFF"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_FILE_OUTPUT=ON"
    "ROCPROFSYS_USE_MPIP=ON"
    "ROCPROFSYS_DEBUG=OFF"
    "ROCPROFSYS_VERBOSE=2"
    "ROCPROFSYS_DL_VERBOSE=2"
    "${_test_openmp_env}"
    "${_test_library_path}"
)

set(_mpip_all2all_environment
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_USE_SAMPLING=OFF"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=OFF"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_FILE_OUTPUT=ON"
    "ROCPROFSYS_USE_MPIP=ON"
    "ROCPROFSYS_DEBUG=ON"
    "ROCPROFSYS_VERBOSE=3"
    "ROCPROFSYS_DL_VERBOSE=3"
    "${_test_openmp_env}"
    "${_test_library_path}"
)

foreach(
    _EXAMPLE
    all2all
    allgather
    allreduce
    bcast
    reduce
    scatter-gather
    send-recv
)
    if("${_mpip_${_EXAMPLE}_environment}" STREQUAL "")
        set(_mpip_${_EXAMPLE}_environment "${_mpip_environment}")
    endif()
    rocprofiler_systems_add_test(
        SKIP_RUNTIME SKIP_SAMPLING
        NAME "mpi-${_EXAMPLE}"
        TARGET mpi-${_EXAMPLE}
        MPI ON
        NUM_PROCS 2
        LABELS "mpip"
        REWRITE_ARGS -e -v 2 --label file line --min-instructions 0
        RUN_ARGS 30
        ENVIRONMENT "${_mpip_${_EXAMPLE}_environment}"
    )
endforeach()

if(ENABLE_FORTRAN_MPI_CTESTS)
    set(_fortran_mpip_flat_environment
        "ROCPROFSYS_FLAT_PROFILE=ON"
        "ROCPROFSYS_COUT_OUTPUT=ON"
        "ROCPROFSYS_TIMELINE_PROFILE=OFF"
        "ROCPROFSYS_COLLAPSE_PROCESSES=ON"
        "ROCPROFSYS_COLLAPSE_THREADS=ON"
        "ROCPROFSYS_SAMPLING_FREQ=50"
        "ROCPROFSYS_TIMEMORY_COMPONENTS=wall_clock,trip_count"
        "${_mpip_environment}"
    )

    if(ROCPROFSYS_USE_MPI)
        set(MPI_FORTRAN_REWRITE_RUN_REGEX
            ">>> MPI_Init(.*\n.*)>>> MPI_Send(.*\n.*)>>> MPI_Recv(.*\n.*)>>> MPI_Comm_size(.*\n.*)>>> MPI_Comm_rank(.*\n.*)"
        )
    else()
        set(MPI_FORTRAN_REWRITE_RUN_REGEX
            ">>> PMPI_Init(.*\n.*)>>> PMPI_Send(.*\n.*)>>> PMPI_Recv(.*\n.*)>>> PMPI_Comm_size(.*\n.*)>>> PMPI_Comm_rank(.*\n.*)"
        )
    endif()

    foreach(_FORTRAN_EXAMPLE intervals)
        rocprofiler_systems_add_test(
            SKIP_RUNTIME
            NAME "mpi-fortran-${_FORTRAN_EXAMPLE}"
            TARGET mpi-fortran-${_FORTRAN_EXAMPLE}
            MPI ON
            NUM_PROCS 2
            LABELS "mpip;fortran"
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
            ENVIRONMENT "${_fortran_mpip_flat_environment}"
            REWRITE_RUN_PASS_REGEX
                ">>> mpi-fortran-${_FORTRAN_EXAMPLE}.inst(.*\n.*)${MPI_FORTRAN_REWRITE_RUN_REGEX}"
        )
    endforeach()
endif()
