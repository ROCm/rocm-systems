#
# rocprofv3 tool tests for kernel renaming
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

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

set(kernel-rename-env "${PRELOAD_ENV}")

rocprofiler_configure_pytest_files(CONFIG pytest.ini input-kernel-rename.yml
                                   COPY validate.py conftest.py)

##########################################################################################
#
#   Command line input
#
##########################################################################################

add_test(
    NAME rocprofv3-test-kernel-rename-cmd-line-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -M --sys-trace no --hsa-trace=0
        --hsa-core-trace=1 --hip-compiler-trace False --hip-runtime-trace --kernel-trace
        --memory-copy-trace -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace-cmd-line -o out
        --output-format pftrace json otf2 --log-level env --kernel-rename
        --perfetto-shmem-size-hint 128 --perfetto-buffer-size 2048000
        --perfetto-buffer-fill-policy ring_buffer --perfetto-backend inprocess --
        $<TARGET_FILE:transpose>)

set_tests_properties(
    rocprofv3-test-kernel-rename-cmd-line-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${kernel-rename-env}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME rocprofv3-test-kernel-rename-cmd-line-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-cmd-line/out_results.json
        --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-cmd-line/out_results.pftrace
        --otf2-input
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-cmd-line/out_results.otf2)

set_tests_properties(
    rocprofv3-test-kernel-rename-cmd-line-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               "rocprofv3-test-kernel-rename-execute" FAIL_REGULAR_EXPRESSION
               "AssertionError")

##########################################################################################
#
#   YAML input
#
##########################################################################################

add_test(
    NAME rocprofv3-test-kernel-rename-inp-yaml-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input-kernel-rename.yml -- $<TARGET_FILE:transpose>)

set_tests_properties(
    rocprofv3-test-kernel-rename-inp-yaml-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${kernel-rename-env}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME rocprofv3-test-kernel-rename-inp-yaml-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml/out_results.json
        --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml/out_results.pftrace
        --otf2-input
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-trace-inp-yaml/out_results.otf2)

set_tests_properties(
    rocprofv3-test-kernel-rename-inp-yaml-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               "rocprofv3-test-kernel-rename-inp-yaml-execute" FAIL_REGULAR_EXPRESSION
               "AssertionError")
