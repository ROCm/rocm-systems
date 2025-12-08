# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -------------------------------------------------------------------------------------- #
#
# NIC performance tests
#
# -------------------------------------------------------------------------------------- #

# Get the name of the default NIC and write it to _network_interface.
execute_process(
    COMMAND "${CMAKE_SOURCE_DIR}/tests/get_default_nic.sh"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE _network_interface
)

message(STATUS "The list of default network interfaces is ${_network_interface}")

# Generate the list of all events that we want PAPI to record.
execute_process(
    COMMAND "${CMAKE_SOURCE_DIR}/tests/generate_papi_nic_events.sh"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE _event_list
)

message(STATUS "The list of all PAPI network events is ${_event_list}")

set(_nic_perf_environment
    "${_base_environment}"
    "ROCPROFSYS_OUTPUT_PATH=${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/nic-performance"
    "ROCPROFSYS_USE_PID=OFF"
    "ROCPROFSYS_VERBOSE=1"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=OFF"
    "ROCPROFSYS_SAMPLING_FREQ=50"
    "ROCPROFSYS_SAMPLING_CPUS=none"
    "ROCPROFSYS_USE_ROCM=OFF"
    "ROCPROFSYS_TIMEMORY_COMPONENTS=wall_clock,papi_array,network_stats"
    "ROCPROFSYS_NETWORK_INTERFACE=${_network_interface}"
    "ROCPROFSYS_PAPI_EVENTS=${_event_list}"
    "ROCPROFSYS_SAMPLING_DELAY=0.05"
)

# Set _download_url to a large file that will give rocprof-sys-sample time to collect NIC
# performance data (but not too large, because each test will time out after 2 min).
set(_download_url
    "https://github.com/ROCm/rocprofiler-systems/releases/download/rocm-6.4.1/rocprofiler-systems-1.0.1-ubuntu-22.04-ROCm-60400-PAPI-OMPT-Python3.sh"
)

# The second file to download. We are downloading two files (each about 90MB), because
# we want wget to run for at least 2s even on a fast network. This will give PAPI enough
# time to collect network metrics.
set(_download2_url
    "https://github.com/ROCm/rocprofiler-systems/releases/download/rocm-6.4.3/rocprofiler-systems-1.0.2-rhel-9.4-PAPI-OMPT-Python3.sh"
)

# Run the NIC performance test
add_test(
    NAME nic-performance
    COMMAND
        $<TARGET_FILE:rocprofiler-systems-sample> -- wget --no-check-certificate
        ${_download_url} ${_download2_url} -O
        ${PROJECT_BINARY_DIR}/rocprofiler-systems.test.bin
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

set_tests_properties(
    nic-performance
    PROPERTIES ENVIRONMENT "${_nic_perf_environment}" TIMEOUT 120 LABELS "network"
)

# Validate the perfetto file generated from NIC performance test output
add_test(
    NAME validate-nic-performance-perfetto
    COMMAND
        ${ROCPROFSYS_VALIDATION_PYTHON}
        ${CMAKE_CURRENT_LIST_DIR}/validate-perfetto-proto.py -i
        ${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/nic-performance/perfetto-trace.proto
        --counter-names rx:byte rx:packet tx:byte tx:packet -t
        /opt/trace_processor/bin/trace_processor_shell -p
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

set(_test_pass_regex
    "rocprof-sys-tests-output/nic-performance/perfetto-trace.proto validated"
)
set(_test_fail_regex
    "Failure validating rocprof-sys-tests-output/nic-performance/perfetto-trace.proto"
)

set_tests_properties(
    validate-nic-performance-perfetto
    PROPERTIES
        TIMEOUT 30
        LABELS "network"
        DEPENDS nic-performance
        PASS_REGULAR_EXPRESSION ${_test_pass_regex}
        FAIL_REGULAR_EXPRESSION ${_test_fail_regex}
)
