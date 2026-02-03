#
# Unit tests for Python counter profiler C++ components
#

cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

# Source files for unit tests
set(ROCPROFILER_PYTHON_TEST_SOURCES types_test.cpp record_collector_test.cpp)

# Create the test executable
add_executable(rocprofiler-sdk-python-lib-tests)

target_sources(
    rocprofiler-sdk-python-lib-tests
    PRIVATE ${ROCPROFILER_PYTHON_TEST_SOURCES} ../source/record_collector.cpp
            ../source/counter_config_manager.cpp)

target_include_directories(rocprofiler-sdk-python-lib-tests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(
    rocprofiler-sdk-python-lib-tests
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-shared-library
            rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-common-library
            GTest::gtest
            GTest::gtest_main)

set_target_properties(
    rocprofiler-sdk-python-lib-tests
    PROPERTIES CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF)

# Add the unit tests using the rocprofiler utility function
rocprofiler_add_unit_test(
    TARGET rocprofiler-sdk-python-lib-tests
    SOURCES ${ROCPROFILER_PYTHON_TEST_SOURCES}
    TEST_LIST python_lib_TESTS
    TIMEOUT 30
    LABELS "unittests;python-bindings"
    ENVIRONMENT
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:$ENV{LD_LIBRARY_PATH}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
