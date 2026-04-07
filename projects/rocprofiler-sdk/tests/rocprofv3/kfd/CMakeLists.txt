#
# rocprofv3 tool tests for kfd
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-kfd
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

set(kfd-xnack-on-env "${PRELOAD_ENV};HSA_XNACK=1")
set(kfd-xnack-off-env "${PRELOAD_ENV};HSA_XNACK=0")

foreach(_XNACK on off)
    rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)
    set(_kfd_labels "integration-tests")
    if(_XNACK STREQUAL "off")
        set(_kfd_labels "integration-tests;exclusive-gpu")
    endif()
    add_test(
        NAME rocprofv3-test-kfd-xnack-${_XNACK}-execute
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --kfd-trace -d
            ${CMAKE_CURRENT_BINARY_DIR}/kfd-xnack-${_XNACK}-trace -o out --output-format
            json --log-level env -- $<TARGET_FILE:transpose>)

    set_tests_properties(
        rocprofv3-test-kfd-xnack-${_XNACK}-execute
        PROPERTIES TIMEOUT 45 LABELS "${_kfd_labels}" ENVIRONMENT
                   "${kfd-xnack-${_XNACK}-env}" FAIL_REGULAR_EXPRESSION
                   "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

    add_test(
        NAME rocprofv3-test-kfd-xnack-${_XNACK}-validate
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
            ${CMAKE_CURRENT_BINARY_DIR}/kfd-xnack-${_XNACK}-trace/out_results.json)

    set_tests_properties(
        rocprofv3-test-kfd-xnack-${_XNACK}-validate
        PROPERTIES TIMEOUT 45 LABELS "${_kfd_labels}" DEPENDS
                   "rocprofv3-test-kfd-xnack-${_XNACK}-execute" FAIL_REGULAR_EXPRESSION
                   "AssertionError")
endforeach()
