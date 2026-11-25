#
# rocprofv3 tool tests for kernel renaming
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-kernel-rename
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning") # info produces memory leak
else()
    set(LOG_LEVEL "info")
endif()

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input-kernel-rename.yml.in
               ${CMAKE_CURRENT_BINARY_DIR}/input-kernel-rename.yml @ONLY)

##########################################################################################
#
#   Command line input
#
##########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-kernel-rename-cmd-line
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -M --sys-trace no --hsa-trace=0
        --hsa-core-trace=1 --hip-compiler-trace False --hip-runtime-trace --kernel-trace
        --memory-copy-trace -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace-cmd-line -o out
        --output-format pftrace json otf2 rocpd --log-level env --kernel-rename
        --perfetto-shmem-size-hint 128 --perfetto-buffer-size 2048000
        --perfetto-buffer-fill-policy ring_buffer --perfetto-backend inprocess --
        $<TARGET_FILE:transpose>
    DEPENDS transpose
    TIMEOUT 45
    WSL_TIMEOUT_SCALE
    LABELS "integration-tests"
    ENVIRONMENT "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-kernel-rename-cmd-line)

# Convert the rocpd database to pftrace and otf2 via the rocpd python module
find_package(Python3 REQUIRED)

set(kernel-rename-rocpd-env
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )

rocprofiler_add_integration_execute_test(
    rocprofv3-test-kernel-rename-cmd-line-rocpd-convert
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f pftrace otf2 --kernel-rename -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-converted-cmd-line -o out -i
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-cmd-line/out_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${kernel-rename-rocpd-env}"
    FIXTURES_REQUIRED rocprofv3-test-kernel-rename-cmd-line
    FIXTURES_SETUP rocprofv3-test-kernel-rename-cmd-line-rocpd-convert)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-kernel-rename-cmd-line
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-cmd-line/out_results.json
         --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocpd-converted-cmd-line/out_results.pftrace
         --otf2-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocpd-converted-cmd-line/out_results.otf2
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-kernel-rename-cmd-line
                      rocprofv3-test-kernel-rename-cmd-line-rocpd-convert)

##########################################################################################
#
#   YAML input
#
##########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-kernel-rename-inp-yaml
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input-kernel-rename.yml -- $<TARGET_FILE:transpose>
    DEPENDS transpose
    TIMEOUT 45
    WSL_TIMEOUT_SCALE
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-kernel-rename-inp-yaml)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-kernel-rename-inp-yaml-rocpd-convert
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f pftrace otf2 --kernel-rename -d
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml -o out -i
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml/out_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${kernel-rename-rocpd-env}"
    FIXTURES_REQUIRED rocprofv3-test-kernel-rename-inp-yaml
    FIXTURES_SETUP rocprofv3-test-kernel-rename-inp-yaml-rocpd-convert)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-kernel-rename-inp-yaml
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml/out_results.json
         --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml/out_results.pftrace
         --otf2-input
         ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml/out_results.otf2
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-kernel-rename-inp-yaml
                      rocprofv3-test-kernel-rename-inp-yaml-rocpd-convert)
