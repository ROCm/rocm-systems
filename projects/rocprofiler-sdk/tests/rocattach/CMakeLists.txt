#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocattach
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(rocattach-parallel-attach-test)
target_sources(rocattach-parallel-attach-test PRIVATE main.cpp)
target_link_libraries(
    rocattach-parallel-attach-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            rocprofiler-sdk::tests-common-library
            rocprofiler-sdk::rocprofiler-sdk-rocattach-shared-library)

add_test(NAME rocattach-parallel-attach-test
         COMMAND $<TARGET_FILE:rocattach-parallel-attach-test>
                 $<TARGET_FILE:attachment-test> $<TARGET_FILE:rocprofiler-sdk-c-tool>)

set_tests_properties(
    rocattach-parallel-attach-test
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}"
               # This regex search doesn't handle multiline searches, so main has to
               # verify instead.
               PASS_REGULAR_EXPRESSION
               "C tool was loaded in both processes."
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${IS_DISABLED}")
