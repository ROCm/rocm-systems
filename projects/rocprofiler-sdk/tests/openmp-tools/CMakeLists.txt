#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-openmp-tools
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(PYTEST_ARGS)
if(ROCPROFILER_MEMCHECK MATCHES "(Address|Thread)Sanitizer" OR ROCPROFILER_BUILD_CODECOV)
    set(PYTEST_ARGS -k "not test_total_runtime")
endif()

set(ROCPROFILER_MEMCHECK_TYPES "ThreadSanitizer" "AddressSanitizer"
                               "UndefinedBehaviorSanitizer")

if(ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

# disable when GPU-0 is navi2, navi3, and navi4
list(GET rocprofiler-sdk-tests-gfx-info 0 openmp-tools-gpu-0-gfx-info)
if("${openmp-tools-gpu-0-gfx-info}" MATCHES "^gfx(10|11|12)[0-9][0-9]$")
    set(IS_DISABLED ON)
endif()

set(openmp-tools-env
    "OMP_NUM_THREADS=2"
    "OMP_DISPLAY_ENV=1"
    "OMP_TARGET_OFFLOAD=mandatory"
    "ROCR_VISIBLE_DEVICES=0"
    "ROCPROFILER_TOOL_OUTPUT_FILE=openmp-tools-test.json"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

rocprofiler_add_integration_execute_test(
    openmp-tools
    TARGET openmp-target
    TIMEOUT 100
    LABELS "integration-tests;openmp-target"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>" "${PRELOAD_ENV}"
    ENVIRONMENT "${openmp-tools-env}"
    DISABLED "${IS_DISABLED}"
    FIXTURES_SETUP openmp-tools)

rocprofiler_add_integration_validate_test(
    openmp-tools
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests;openmp-target"
    TIMEOUT 45
    FIXTURES_REQUIRED openmp-tools
    DISABLED "${IS_DISABLED}"
    ARGS ${PYTEST_ARGS} --input ${CMAKE_CURRENT_BINARY_DIR}/openmp-tools-test.json
    DISCOVERY_ARGS ${PYTEST_ARGS})
