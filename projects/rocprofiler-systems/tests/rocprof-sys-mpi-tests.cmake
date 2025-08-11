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
        "(perfetto-trace|trip_count|sampling_percent|sampling_cpu_clock|sampling_wall_clock|wall_clock)-[0-9][0-9]+.(json|txt|proto)|ROCPROFSYS_ABORT_FAIL_REGEX"
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
