#
# Late-start tracing integration test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

project(rocprofiler-sdk-tests-late-start-tracing LANGUAGES CXX HIP)

find_package(rocprofiler-sdk REQUIRED)
find_package(rocprofiler-sdk-roctx REQUIRED)

# Build the tool library (statically linked into the executable)
add_library(late-start-tracing-client SHARED)
target_sources(late-start-tracing-client PRIVATE late_start_tool.cpp)
target_link_libraries(late-start-tracing-client PRIVATE rocprofiler-sdk::rocprofiler-sdk)

# Build the test executable (links to tool library AND rocprofiler-sdk for init())
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)

add_executable(late-start-tracing-test)
target_sources(late-start-tracing-test PRIVATE main.cpp)
target_link_libraries(
    late-start-tracing-test PRIVATE late-start-tracing-client
                                    rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)

# Set up environment for test execution
set(late-start-tracing-env
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$<TARGET_FILE_DIR:late-start-tracing-client>:$ENV{LD_LIBRARY_PATH}"
    )

# Execute late-start test - generates JSON output
rocprofiler_add_integration_execute_test(
    late-start-tracing
    COMMAND $<TARGET_FILE:late-start-tracing-test>
    DEPENDS late-start-tracing-client
    TIMEOUT 120
    LABELS "integration-tests;late-start"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${late-start-tracing-env}"
                "ROCPROFILER_TOOL_OUTPUT_FILE=late-start-tracing.json"
    SKIP_REGULAR_EXPRESSION "test skipped"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_SETUP late-start-tracing)

# Validate JSON output using pytest
rocprofiler_add_integration_validate_test(
    late-start-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests;late-start"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_REQUIRED late-start-tracing
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/late-start-tracing.json)
