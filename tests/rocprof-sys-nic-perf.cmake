# -------------------------------------------------------------------------------------- #
#
# NIC performance tests
#
# -------------------------------------------------------------------------------------- #

set(_network_interface "lo")
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
    "ROCPROFSYS_PAPI_EVENTS=net:::${_network_interface}:tx:byte,net:::${_network_interface}:rx:byte,net:::${_network_interface}:rx:packet,net:::${_network_interface}:tx:packet"
    )

set(_download_url
    "https://github.com/ROCm/rocprofiler-systems/releases/download/rocm-6.3.1/rocprofiler-systems-0.1.0-ubuntu-20.04-ROCm-60200-PAPI-OMPT-Python3.sh"
    )

# Run the NIC performance test
add_test(
    NAME nic-performance
    COMMAND $<TARGET_FILE:rocprofiler-systems-sample> -- wget --no-check-certificate
            --quiet ${_download_url} -O /tmp/rocprofiler-systems-install.sh
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set_tests_properties(nic-performance PROPERTIES ENVIRONMENT "${_nic_perf_environment}"
                                                TIMEOUT 120 LABELS "network")

# Validate the perfetto file generated from NIC performance test output
add_test(
    NAME validate-nic-performance-perfetto
    COMMAND
        ${ROCPROFSYS_VALIDATION_PYTHON}
        ${CMAKE_CURRENT_LIST_DIR}/validate-perfetto-proto.py -i
        ${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/nic-performance/perfetto-trace.proto
        --counter-names rx:byte rx:packet tx:byte tx:packet -t
        /opt/trace_processor/bin/trace_processor_shell -p
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set(_test_pass_regex
    "rocprof-sys-tests-output/nic-performance/perfetto-trace.proto validated")
set(_test_fail_regex
    "Failure validating rocprof-sys-tests-output/nic-performance/perfetto-trace.proto")

set_tests_properties(
    validate-nic-performance-perfetto
    PROPERTIES TIMEOUT
               30
               LABELS
               "network"
               DEPENDS
               nic-performance
               PASS_REGULAR_EXPRESSION
               ${_test_pass_regex}
               FAIL_REGULAR_EXPRESSION
               ${_test_fail_regex})
