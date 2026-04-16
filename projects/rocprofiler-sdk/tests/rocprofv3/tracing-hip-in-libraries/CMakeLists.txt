#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-tracing-hip-in-libraries
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "" PRELOAD_ENV "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

if(PRELOAD_ENV)
    set(PRELOAD_ARGS --suppress-marker-preload --preload ${PRELOAD_ENV})
else()
    set(PRELOAD_ARGS)
endif()

foreach(_OUTPUT_FORMAT csv json pftrace otf2)
    rocprofiler_add_integration_execute_test(
        rocprofv3-test-trace-hip-in-libraries-${_OUTPUT_FORMAT}
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-runtime-trace --hsa-core-trace
            --hsa-amd-trace --marker-trace --kernel-trace --memory-copy-trace --stats
            --output-format ${_OUTPUT_FORMAT} -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace
            -o out ${PRELOAD_ARGS} -- $<TARGET_FILE:hip-in-libraries>
        DEPENDS hip-in-libraries
        TIMEOUT 120
        LABELS "integration-tests"
        PRELOAD "${PRELOAD_ENV}"
        FIXTURES_SETUP rocprofv3-test-trace-hip-in-libraries)
endforeach()

rocprofiler_add_integration_validate_test(
    rocprofv3-test-trace-hip-in-libraries
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --hsa-input
         ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_*_hsa_api_trace.csv
         --hip-input
         ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_*_hip_api_trace.csv
         --kernel-input
         ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_*_kernel_trace.csv
         --memory-copy-input
         ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_*_memory_copy_trace.csv
         --marker-input
         ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_*_marker_api_trace.csv
         --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_*_agent_info.csv
         --kernel-stats
         ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_*_kernel_stats.csv
         --hip-stats
         ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_*_hip_api_stats.csv
         --hsa-stats
         ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_*_hsa_api_stats.csv
         --memory-copy-stats
         ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_*_memory_copy_stats.csv
         --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_*_results.json
         --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_*_results.pftrace
         --otf2-input
         ${CMAKE_CURRENT_BINARY_DIR}/hip-in-libraries-trace/out_*_results.otf2
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-trace-hip-in-libraries)
