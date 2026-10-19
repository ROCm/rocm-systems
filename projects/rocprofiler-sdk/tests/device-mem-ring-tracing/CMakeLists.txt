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
# Kernel tracing against an application ring buffer resident in device memory. The inline
# queue-interposition path writes AQL packets into the application's own ring, so where
# that ring lives changes how those writes have to be ordered. The execute step keys on
# the placement the profiler resolves at queue creation, because the runtime falls back to
# a system-memory ring silently when device memory is unavailable and the test would
# otherwise pass while covering nothing. The validate step confirms every dispatch still
# produced a record.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-device-mem-ring-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# Device-memory ring buffers are selectable only on the architectures the runtime offers
# them on; everywhere else the request is accepted and ignored. Registering the test
# disabled rather than skipping the directory keeps it visible in `ctest -N`.
list(GET rocprofiler-sdk-tests-gfx-info 0 device-mem-ring-gpu-0-gfx-info)
if("${device-mem-ring-gpu-0-gfx-info}" MATCHES "^gfx(94[0-2]|95[0-2]|125[0-9])$")
    set(IS_DISABLED OFF)
else()
    set(IS_DISABLED ON)
endif()

# Workload geometry. Both steps read it from here, so the dispatch count the validation
# expects cannot drift from the arguments the workload ran with. Launches are batched
# between synchronizations to keep packets entering the ring without a drain between each
# one. The totals stay small: the subject is ring placement, not throughput, and the trace
# runs at a log level that reports every queue the profiler sees.
set(DEVICE_MEM_RING_ITERATIONS 50)
set(DEVICE_MEM_RING_THREADS 2)
set(DEVICE_MEM_RING_SPIN 1000)
set(DEVICE_MEM_RING_SYNC_INTERVAL 8)

rocprofiler_add_integration_execute_test(
    device-mem-ring-tracing
    TARGET reproducible-dispatch-count
    ARGS ${DEVICE_MEM_RING_ITERATIONS} ${DEVICE_MEM_RING_THREADS} ${DEVICE_MEM_RING_SPIN}
         ${DEVICE_MEM_RING_SYNC_INTERVAL}
    TIMEOUT 100
    LABELS "integration-tests"
    DISABLED "${IS_DISABLED}"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    ENVIRONMENT
        "ROCPROFILER_TOOL_OUTPUT_FILE=device-mem-ring-tracing-test.json"
        # The tool configures every service regardless; this names the ones it may start,
        # and starting any of the others disqualifies the inline path under test. The
        # configured-but-unstarted contexts still show up in the profiler's own count.
        "ROCPROFILER_TOOL_CONTEXTS=KERNEL_DISPATCH_CALLBACK,KERNEL_DISPATCH_BUFFERED,CODE_OBJECT"
        # Pin interposition on. It would default on for these contexts, but set it
        # explicitly so a future default-logic change cannot silently no-op the test.
        "ROCPROFILER_QUEUE_INTERPOSITION=1"
        # Place the application's ring buffer in device memory.
        "DEBUG_CLR_AQL_DEV_QUEUE=1"
        # The resolved ring placement is reported at this level, once per queue.
        "ROCPROFILER_LOG_LEVEL=info"
    # Matching the resolved placement is what turns a silent fallback to system memory
    # into a failure. It replaces the exit-code criterion: a run that dies partway is
    # caught by the validation finding too few records, and one that hangs by TIMEOUT.
    PASS_REGULAR_EXPRESSION "\\[queue-interposition\\] ring buffer.*placement device"
    FIXTURES_SETUP device-mem-ring-tracing)

rocprofiler_add_integration_validate_test(
    device-mem-ring-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    DISABLED "${IS_DISABLED}"
    FIXTURES_REQUIRED device-mem-ring-tracing
    # Collecting the tests runs pytest without the run arguments, and the geometry options
    # have no defaults, so collection has to be given them too. Both lists read the same
    # variables, so they cannot disagree.
    DISCOVERY_ARGS --iterations ${DEVICE_MEM_RING_ITERATIONS} --threads
                   ${DEVICE_MEM_RING_THREADS}
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/device-mem-ring-tracing-test.json
         --iterations ${DEVICE_MEM_RING_ITERATIONS} --threads ${DEVICE_MEM_RING_THREADS})
