#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-conversion-script
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
# copy to binary directory
rocprofiler_configure_pytest_files(COPY validate.py input.txt conftest.py
                                   CONFIG pytest.ini)

configure_file(${CMAKE_SOURCE_DIR}/source/scripts/convert-counters-collection-format.py
               ${CMAKE_CURRENT_BINARY_DIR}/convert-counters-collection-format.py COPYONLY)
add_test(
    NAME test-conversion-script-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.txt -T -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_conversion_script -o pmc1 --output-format csv --
        $<TARGET_FILE:vector-ops>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(cc-env-pmc "${PRELOAD_ENV}")

set_tests_properties(
    test-conversion-script-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${cc-env-pmc}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME test-conversion-script-convert
    COMMAND
        ${Python3_EXECUTABLE} convert-counters-collection-format.py --input
        ${CMAKE_CURRENT_BINARY_DIR}/out_conversion_script --output
        ${CMAKE_CURRENT_BINARY_DIR}/out_conversion_script/converted.csv)

set_tests_properties(
    test-conversion-script-convert
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               test-conversion-script-execute FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(NAME test-conversion-script-validate
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
                 ${CMAKE_CURRENT_BINARY_DIR}/out_conversion_script/converted.csv)

set_tests_properties(
    test-conversion-script-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               test-conversion-script-convert FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
