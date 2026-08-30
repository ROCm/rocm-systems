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
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-rocshmem-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
# rocSHMEM's installed config references both MPI::MPI_CXX and
# rocprofiler-register::rocprofiler-register in the roc::rocshmem link interface, so both
# must be located first or find_package(rocshmem) will fail to define the imported target.
# Package name is lowercase to match /opt/rocm/lib/cmake/rocshmem/.
find_package(MPI COMPONENTS CXX)
find_package(rocprofiler-register CONFIG PATHS ${ROCM_PATH} /opt/rocm)
if(MPI_CXX_FOUND AND rocprofiler-register_FOUND)
    find_package(rocshmem CONFIG PATHS ${ROCM_PATH} /opt/rocm)
endif()

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(rocshmem-tracing-env "${PRELOAD_ENV}")

# rocSHMEM needs an Open MPI launcher (mpirun / mpiexec) to bootstrap the symmetric heap,
# so this integration test is only enabled when (a) the in-tree rocshmem-demo target is
# built, and (b) an mpirun / mpiexec executable is on PATH. Allow root execution for
# container CI (mirrors the mpi-ranks test pattern).
find_program(MPIRUN_EXECUTABLE NAMES mpirun mpiexec)

set(IS_DISABLED OFF)
if(NOT TARGET rocshmem-demo)
    set(IS_DISABLED ON)
endif()
if(NOT MPIRUN_EXECUTABLE)
    set(IS_DISABLED ON)
endif()

set(MPI_ENV_VARS OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1)

# rocSHMEM cannot bootstrap with a single PE (MPI_Win_create fails) so the demo is
# launched with -np 2. Each rank gets its own per-rank output directory keyed by
# %env{OMPI_COMM_WORLD_RANK}% so the two rocprofv3 instances don't race on the same files.
# Validation only inspects rank 0's trace; the API surface exercised by both ranks is
# identical. rocSHMEM is emitted directly only to JSON (used for validation) and rocpd
# (the default rocprofv3 output format); CSV/Perfetto/OTF2 are produced from the rocpd
# database via `rocpd convert`, so only `json rocpd` is requested here.
#
# Only register the tests when the rocshmem-demo target exists. When it is absent (e.g.
# the code-coverage build, which has no MPI/rocSHMEM), skipping registration avoids
# emitting an unlabeled disabled placeholder that would fail the test-label verification.
if(TARGET rocshmem-demo)
    rocprofiler_add_integration_execute_test(
        rocprofv3-test-rocshmem-tracing
        COMMAND
            ${CMAKE_COMMAND} -E env ${MPI_ENV_VARS} ${MPIRUN_EXECUTABLE} -np 2
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --rocshmem-trace -d
            ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace/rank.%env{OMPI_COMM_WORLD_RANK}% -o
            out --output-format json rocpd --log-level env --
            $<IF:$<TARGET_EXISTS:rocshmem-demo>,$<TARGET_FILE:rocshmem-demo>,rocshmem-demo>
        DEPENDS rocshmem-demo
        TIMEOUT 60
        LABELS "integration-tests"
        PRELOAD "${PRELOAD_ENV}"
        FIXTURES_SETUP rocprofv3-test-rocshmem-tracing
        DISABLED "${IS_DISABLED}")

    rocprofiler_add_integration_validate_test(
        rocprofv3-test-rocshmem-tracing
        TEST_PATHS validate.py
        COPY conftest.py
        CONFIG pytest.ini
        ARGS --json-input
             ${CMAKE_CURRENT_BINARY_DIR}/rocshmem-demo-trace/rank.0/out_results.json
             --rocpd-input
             ${CMAKE_CURRENT_BINARY_DIR}/rocshmem-demo-trace/rank.0/out_results.db
        TIMEOUT 45
        LABELS "integration-tests"
        FIXTURES_REQUIRED rocprofv3-test-rocshmem-tracing
        DISABLED "${IS_DISABLED}")
endif()
