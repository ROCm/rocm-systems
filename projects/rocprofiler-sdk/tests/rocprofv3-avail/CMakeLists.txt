#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-avail
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)
rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY conftest.py validate.py)

set(PRELOAD_ENV "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

add_test(NAME rocprofv3-avail-test-hw-counters
         COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3-avail>
                 info --pmc)

add_test(NAME rocprofv3-avail-test-pc-sample-config
         COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3-avail>
                 info --pc-sampling)

add_test(NAME rocprofv3-avail-test-check-hw-counters
         COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3-avail> -d
                 0 pmc-check SQ_WAVES GRBM_COUNT TCC_HIT)

add_test(NAME rocprofv3-avail-test-check-hw-counters_2
         COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3-avail>
                 pmc-check SQ_WAVES GRBM_COUNT TCC_HIT:device=0)

# disable when GPU-0 is navi2, navi3, and navi4
list(GET rocprofiler-sdk-tests-gfx-info 0 pc-sampling-gpu-0-gfx-info)
if("${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx(10|11|12)[0-9][0-9]$"
   OR "${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx906$"
   OR ROCPROFILER_MEMCHECK STREQUAL "AddressSanitizer")
    set(IS_DISABLED ON)
endif()

set(test-rocprofv3-avail-env "${PRELOAD_ENV}")
set(enable_pc_sampling "ROCPROFILER_PC_SAMPLING_BETA_ENABLED=on")
set_tests_properties(
    rocprofv3-avail-test-hw-counters
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "integration-tests"
        ENVIRONMENT
        "${test-rocprofv3-avail-env}"
        PASS_REGULAR_EXPRESSION
        "GPU:[0-9]*\\n*;Name:\\t[a-zA-Z_]*\\n;counter_name:\\t[a-zA-Z_]*\\n;description:\\t(.*)\\n*;expression:\\t(.)*\\n*;block:\\t[a-zA-Z]*\\n*;dimension_name:\\t([A-Z_]*)\\n*; dimension_instances:\\t([[0-9]*:[0-9]*\\])*\\n*"
        DISABLED
        "${IS_DISABLED}")

set_tests_properties(
    rocprofv3-avail-test-pc-sample-config
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "integration-tests;pc-sampling"
        ENVIRONMENT
        "${test-rocprofv3-avail-env};${enable_pc_sampling}"
        PASS_REGULAR_EXPRESSION
        "GPU:[0-9]*\\n*;Name:\\t[a-zA-Z_]*\\n;method:(.*)\\n*;unit:(.*)\\n*;min_interval:[0-9]*\\n*;max_interval:[0-9]*\\n*;flags:(.*)\\n*"
        DISABLED
        "${IS_DISABLED}")

set_tests_properties(
    rocprofv3-avail-test-check-hw-counters
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT
               "${test-rocprofv3-avail-env}" DISABLED "${IS_DISABLED}")

set_tests_properties(
    rocprofv3-avail-test-check-hw-counters_2
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT
               "${test-rocprofv3-avail-env}" DISABLED "${IS_DISABLED}")

if(TARGET rocprofv3-list-avail)

    add_test(NAME rocprofv3-avail-test-validate
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py
                     --rocm-path ${CMAKE_BINARY_DIR})

    # for validate, explicitly set ROCPROF_LIST_AVAIL_TOOL_LIBRARY since we copy
    # rocprofv3-avail to directory
    set(test-rocprofv3-avail-validate-env
        "${PRELOAD_ENV}" "PYTHONPATH=$ENV{PYTHONPATH}:${CMAKE_BINARY_DIR}/bin/")

    set_tests_properties(
        rocprofv3-avail-test-validate
        PROPERTIES TIMEOUT
                   45
                   LABELS
                   "integration-tests"
                   ENVIRONMENT
                   "${test-rocprofv3-avail-validate-env}"
                   FAIL_REGULAR_EXPRESSION
                   "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
                   DEPENDS
                   rocprofv3-avail-test-copy-module
                   DISABLED
                   "${IS_DISABLED}")
endif()
