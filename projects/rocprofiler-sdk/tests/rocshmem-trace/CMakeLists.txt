# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocshmem-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# Use the launcher (mpiexec) from the same MPI package the demo links against, rather than
# an independent find_program search, so the launch MPI cannot diverge from the MPI
# rocSHMEM and the demo were built against. find_package(MPI) provides MPIEXEC_EXECUTABLE
# for this.
find_package(MPI COMPONENTS CXX)

# The integration test is only enabled when (a) the in-tree rocshmem-demo target is built
# (which itself depends on rocshmem >= 3.3.0 + rocprofiler-register being installed), and
# (b) an MPI launcher was located. The MPI env vars below assume Open MPI (matches the
# mpi-ranks test pattern; allows root execution in container CI).
set(IS_DISABLED OFF)
if(NOT TARGET rocshmem-demo)
    set(IS_DISABLED ON)
endif()
if(NOT MPIEXEC_EXECUTABLE)
    set(IS_DISABLED ON)
endif()

set(MPI_ENV_VARS OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
                 OMPI_MCA_btl=self,vader OMPI_MCA_pml=ob1 PSM3_DEVICES=self,shm)

# Build the LD_PRELOAD value: the rocprofiler-sdk-json-tool library captures every tracing
# domain it knows about (including ROCSHMEM_API on this branch) and writes a single JSON
# file at tool finalization. Memory checker preload (if set) is concatenated in front so
# memcheck still works.
if(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}:$<TARGET_FILE:rocprofiler-sdk-json-tool>"
        )
else()
    set(PRELOAD_ENV "$<TARGET_FILE:rocprofiler-sdk-json-tool>")
endif()

# Each MPI rank writes to its own JSON file so the two rocshmem-demo processes do not race
# on the same path. json-tool reads ROCPROFILER_TOOL_OUTPUT_FILE verbatim (no built-in
# placeholder expansion), so use a `bash -c` shim to expand $OMPI_COMM_WORLD_RANK at the
# per-rank level. Validation only inspects rank 0's trace; both ranks exercise the same
# set of APIs.
#
# Only register the tests when the rocshmem-demo target exists. When it is absent (e.g.
# the code-coverage build, which has no MPI/rocSHMEM), skipping registration avoids
# emitting an unlabeled disabled placeholder that would fail the test-label verification.
if(TARGET rocshmem-demo)
    rocprofiler_add_integration_execute_test(
        rocshmem-tracing
        COMMAND
            ${MPIEXEC_EXECUTABLE} -np 2 bash -c
            "ROCPROFILER_TOOL_OUTPUT_FILE='${CMAKE_CURRENT_BINARY_DIR}/rocshmem-tracing-test.rank-'\${OMPI_COMM_WORLD_RANK}.json exec timeout -k 5 30 '$<IF:$<TARGET_EXISTS:rocshmem-demo>,$<TARGET_FILE:rocshmem-demo>,rocshmem-demo>'"
        DEPENDS rocshmem-demo
        TIMEOUT 45
        LABELS "integration-tests"
        ENVIRONMENT ${MPI_ENV_VARS}
        PRELOAD "${PRELOAD_ENV}"
        PASS_REGULAR_EXPRESSION "Finalization complete"
        FIXTURES_SETUP rocshmem-tracing
        DISABLED "${IS_DISABLED}")

    rocprofiler_add_integration_validate_test(
        rocshmem-tracing
        TEST_PATHS validate.py
        COPY conftest.py
        CONFIG pytest.ini
        LABELS "integration-tests"
        TIMEOUT 45
        ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/rocshmem-tracing-test.rank-0.json
        FIXTURES_REQUIRED rocshmem-tracing
        DISABLED "${IS_DISABLED}")
endif()
