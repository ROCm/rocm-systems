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

#
# rocprofiler-systems tests
#
include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-testing.cmake)

# test groups
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-config-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-instrument-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-pthread-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-rocm-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-user-api-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-mpi-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-kokkos-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-openmp-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-code-coverage-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-fork-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-time-window-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-attach-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-rccl-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-overflow-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-annotate-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-causal-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-python-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-decode-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-gpu-connect-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-nic-perf.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-roctx-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-rocm-hip-stream.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-binary-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-thread-limit-tests.cmake)

# -------------------------------------------------------------------------------------- #
#
# Global cleanup test for temporary files
# This runs once after ALL tests complete to clean up trace cache temporary files
# Uses FIXTURES_CLEANUP to ensure it runs after all tests requiring the fixture
#
# -------------------------------------------------------------------------------------- #

add_test(
    NAME rocprofsys-cleanup-tmp-files
    COMMAND sh -c "rm -f /tmp/buffered_storage*.bin /tmp/metadata*.json"
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

set_tests_properties(
    rocprofsys-cleanup-tmp-files
    PROPERTIES
        FIXTURES_CLEANUP rocprofsys-global-tmp-files
        LABELS "cleanup;global"
        TIMEOUT 30
)
