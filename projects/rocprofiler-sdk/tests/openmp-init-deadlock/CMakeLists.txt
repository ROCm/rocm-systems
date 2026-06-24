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
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-openmp-init-deadlock
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# the application is only built when LLVM libomp is found (see
# tests/bin/openmp-init-deadlock); skip registering the test if the target was not created
if(NOT TARGET openmp-init-deadlock)
    message(
        STATUS
            "[openmp-init-deadlock] application target not available; skipping regression test"
        )
    return()
endif()

# Regression test for the OMPT-init deadlock fixed in rocprofiler-sdk client discovery.
#
# rocprofiler-sdk is loaded as the OMPT tool (OMP_TOOL_LIBRARIES). When the linked
# blas-stub library's DT_INIT enters an omp parallel region, libomp performs OMPT tool
# discovery while holding its init lock. Previously, client discovery dlopen'd the
# configure-exporting client library (whose DT_INIT was still pending), re-entering libomp
# and self-deadlocking. With the fix, discovery resolves symbols via the ELF symbol table
# without dlopen, so the application runs to completion.
#
# The assertion is twofold: the TIMEOUT guards against the deadlock (a hang fails the
# test), and PASS_REGULAR_EXPRESSION ensures the application actually reached main (so an
# unrelated early exit cannot pass silently).
rocprofiler_add_integration_execute_test(
    openmp-init-deadlock
    TARGET openmp-init-deadlock
    TIMEOUT 60
    LABELS "integration-tests;openmp-init-deadlock"
    ENVIRONMENT
        "OMP_TOOL=enabled"
        "OMP_NUM_THREADS=2"
        "OMP_TOOL_LIBRARIES=$<TARGET_FILE:rocprofiler-sdk::rocprofiler-sdk-shared-library>"
        "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    PASS_REGULAR_EXPRESSION "reached main: no OMPT-init deadlock")
