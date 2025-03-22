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

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}:$<TARGET_FILE:rocprofiler-sdk-json-tool>")
else()
    set(PRELOAD_ENV "LD_PRELOAD=$<TARGET_FILE:rocprofiler-sdk-json-tool>")
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

add_test(NAME test-openmp-tools-execute COMMAND $<TARGET_FILE:openmp-target>)

set(openmp-tools-env
    "${PRELOAD_ENV}"
    "OMP_NUM_THREADS=2"
    "OMP_DISPLAY_ENV=1"
    "OMP_TARGET_OFFLOAD=mandatory"
    "ROCR_VISIBLE_DEVICES=0"
    "ROCPROFILER_TOOL_OUTPUT_FILE=openmp-tools-test.json"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

set_tests_properties(
    test-openmp-tools-execute
    PROPERTIES TIMEOUT
               100
               LABELS
               "integration-tests;openmp-target"
               ENVIRONMENT
               "${openmp-tools-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${IS_DISABLED}")

# copy to binary directory
rocprofiler_configure_pytest_files(COPY validate.py conftest.py CONFIG pytest.ini)

add_test(
    NAME test-openmp-tools-validate
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py ${PYTEST_ARGS}
            --input ${CMAKE_CURRENT_BINARY_DIR}/openmp-tools-test.json)

set_tests_properties(
    test-openmp-tools-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests;openmp-target"
               DEPENDS
               test-openmp-tools-execute
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${IS_DISABLED}")
