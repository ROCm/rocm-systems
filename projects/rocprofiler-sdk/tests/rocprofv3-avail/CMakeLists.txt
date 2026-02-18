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

set(PRELOAD_ENV "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}")

# disable when GPU-0 is navi2, navi3, and navi4
list(GET rocprofiler-sdk-tests-gfx-info 0 pc-sampling-gpu-0-gfx-info)
# Disable for all sanitizers - Python tests cannot load sanitizer libs dynamically due to
# static TLS block allocation limitations
if("${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx(10|11|12)[0-9][0-9]$"
   OR "${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx906$"
   OR ROCPROFILER_MEMCHECK)
    set(IS_DISABLED ON)
endif()

rocprofiler_add_integration_execute_test(
    rocprofv3-avail-test-hw-counters
    COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3-avail> info
            --pmc
    DEPENDS rocprofiler-sdk::rocprofv3-avail
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    PASS_REGULAR_EXPRESSION
        "GPU:[0-9]*\\n*;Name:\\t[a-zA-Z_]*\\n;Counter_Name:\\t[a-zA-Z_]*\\n;Description:\\t(.*)\\n*;Expression:\\t(.)*\\n*;Block:\\t[a-zA-Z]*\\n*;Dimensions:\\t([A-Z_]*)\\t([[0-9]*:[0-9]*\\])*\\n*"
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_execute_test(
    rocprofv3-avail-test-pc-sample-config
    COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3-avail> info
            --pc-sampling
    DEPENDS rocprofiler-sdk::rocprofv3-avail
    TIMEOUT 45
    LABELS "integration-tests;pc-sampling"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "ROCPROFILER_PC_SAMPLING_BETA_ENABLED=on"
    PASS_REGULAR_EXPRESSION
        "GPU[:0-9 ]+\\n*;Name[:a-zA-Z_ ]+\\n;configs[: ]+;Method:(.*)\\n*;Unit:(.*)\\n*;Min_Interval:[0-9]*\\n*;Max_Interval:[0-9]*\\n*;Flags:(.*)\\n*"
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_execute_test(
    rocprofv3-avail-test-check-hw-counters
    COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3-avail> -d 0
            pmc-check SQ_WAVES GRBM_COUNT TCC_HIT
    DEPENDS rocprofiler-sdk::rocprofv3-avail
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_execute_test(
    rocprofv3-avail-test-check-hw-counters_2
    COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3-avail>
            pmc-check SQ_WAVES GRBM_COUNT TCC_HIT:device=0
    DEPENDS rocprofiler-sdk::rocprofv3-avail
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-avail-test
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --rocm-path ${rocprofiler-sdk_ROOT_DIR}
    TIMEOUT 45 PRELOAD "${PRELOAD_ENV}"
    LABELS "integration-tests"
    DISABLED "${IS_DISABLED}")
