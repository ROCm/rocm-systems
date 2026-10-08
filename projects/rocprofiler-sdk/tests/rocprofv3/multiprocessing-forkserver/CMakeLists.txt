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
# rocprofv3 multiprocessing forkserver test
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-multiprocessing-forkserver
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(NOT Python3_EXECUTABLE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(FORKSERVER_APP ${CMAKE_CURRENT_SOURCE_DIR}/forkserver_app.py)
set(OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/multiprocessing-forkserver)

# The app has no device work, so a profiler that leaves the forkserver alone produces an
# empty trace and exits cleanly. A profiler that installs signal handlers before the
# interpreter starts leaves the forkserver unable to restore its saved dispositions: the
# pool then dies, and the app's internal timeout bounds the failure so a regression
# reports rather than hanging to the ctest TIMEOUT.
rocprofiler_add_integration_cleanup_test(
    multiprocessing-forkserver
    DIRECTORIES ${OUT_DIR}
    LABELS "integration-tests;multiprocessing-forkserver"
    FIXTURES_SETUP multiprocessing-forkserver-clean)

rocprofiler_add_integration_execute_test(
    multiprocessing-forkserver
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --kernel-trace --output-format json
            -d ${OUT_DIR} -o out_%pid% -- ${Python3_EXECUTABLE} ${FORKSERVER_APP}
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;multiprocessing-forkserver"
    PRELOAD "${PRELOAD_ENV}"
    PASS_REGULAR_EXPRESSION "forkserver pool completed"
    FIXTURES_REQUIRED multiprocessing-forkserver-clean)
