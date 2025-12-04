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

# Build the tool configuration library
add_library(late-start-dynamic-client SHARED)
target_sources(late-start-dynamic-client PRIVATE late_start_tool.cpp)
target_link_libraries(late-start-dynamic-client PRIVATE rocprofiler-sdk::rocprofiler-sdk)

# Build the dynamic test executable (does NOT link to rocprofiler-sdk)
set_source_files_properties(main_dynamic.cpp PROPERTIES LANGUAGE HIP)

add_executable(late-start-dynamic-test)
target_sources(late-start-dynamic-test PRIVATE main_dynamic.cpp)
target_link_libraries(late-start-dynamic-test PRIVATE ${CMAKE_DL_LIBS})

# Export symbols so the dynamically loaded tool library can access them
target_link_options(late-start-dynamic-test PRIVATE "-Wl,--export-dynamic")

# Set up environment for test execution
set(late-start-dynamic-env
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$<TARGET_FILE_DIR:late-start-dynamic-client>:$ENV{LD_LIBRARY_PATH}"
)

# Execute dynamic late-start test - validates via PASS/FAIL in output
rocprofiler_add_integration_execute_test(
    late-start-dynamic
    COMMAND $<TARGET_FILE:late-start-dynamic-test>
    DEPENDS late-start-dynamic-client
    TIMEOUT 120
    LABELS "integration-tests;late-start"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${late-start-dynamic-env}"
    SKIP_REGULAR_EXPRESSION "Test SKIPPED"
    PASS_REGULAR_EXPRESSION "Test PASSED"
    FAIL_REGULAR_EXPRESSION "Test FAILED|${ROCPROFILER_DEFAULT_FAIL_REGEX}")
