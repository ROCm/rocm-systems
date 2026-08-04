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
# rocprofv3 tool test: OMPT-init deadlock regression
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-ompt-deadlock
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(ROCPROFILER_MEMCHECK_TYPES "ThreadSanitizer" "AddressSanitizer"
                               "UndefinedBehaviorSanitizer")

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(ompt-deadlock-env
    "OMP_NUM_THREADS=2"
    "OMP_DISPLAY_ENV=1"
    "ROCR_VISIBLE_DEVICES=0"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

# Regression test for the OMPT-init deadlock, exercised through rocprofv3. rocprofv3 loads
# the tool library (which now provides the ompt_start_tool symbol), so libomp engages OMPT
# and rocprofiler-sdk performs client discovery. The openmp-init-deadlock workload links a
# library whose DT_INIT constructor calls into libomp, reproducing the conditions under
# which discovery previously self-deadlocked. Reaching main means discovery completed
# without re-entering libomp's init lock.
#
# The assertion is twofold: TIMEOUT guards against the deadlock (a hang fails the test),
# and PASS_REGULAR_EXPRESSION ensures the workload actually reached main (so an unrelated
# early exit cannot pass silently). This is a host-only OpenMP workload, so it is gated on
# OMP_TARGET_COMPILER and the workload target rather than on target-offload support.
set(IS_INIT_DEADLOCK_DISABLED OFF)
if(NOT TARGET openmp-init-deadlock
   OR NOT OMP_TARGET_COMPILER
   OR (ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES))
    set(IS_INIT_DEADLOCK_DISABLED ON)
endif()

set(ompt-init-deadlock-output-dir ${CMAKE_CURRENT_BINARY_DIR}/openmp-init-deadlock-trace)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-ompt-init-deadlock
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --sys-trace -d
        ${ompt-init-deadlock-output-dir} -o out --output-format rocpd --log-level env --
        $<IF:$<TARGET_EXISTS:openmp-init-deadlock>,$<TARGET_FILE:openmp-init-deadlock>,openmp-init-deadlock>
    DEPENDS openmp-init-deadlock
    TIMEOUT 20
    LABELS "integration-tests;openmp-host"
    ENVIRONMENT "${ompt-deadlock-env}"
    PRELOAD "${PRELOAD_ENV}"
    PASS_REGULAR_EXPRESSION "reached main: no OMPT-init deadlock"
    DISABLED "${IS_INIT_DEADLOCK_DISABLED}")
