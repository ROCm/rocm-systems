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

project(
    rocprofiler-sdk-tests-late-start-tracing
    LANGUAGES CXX HIP
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# Mark source file as HIP language
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)

# Create the test application executable
add_executable(late-start-test-app main.cpp)

target_link_libraries(late-start-test-app
    PRIVATE
        ${CMAKE_DL_LIBS}  # For dlopen/dlsym
)

# The test application initializes HIP first, then loads rocprofiler-sdk dynamically
# and calls rocprofiler_start_late()

set(late-start-env
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
)

# Execute test - runs the late-start application
rocprofiler_add_integration_execute_test(
    late-start-tracing
    COMMAND $<TARGET_FILE:late-start-test-app>
    DEPENDS late-start-test-app rocprofiler-sdk::rocprofiler-sdk-shared-library
    TIMEOUT 120
    LABELS "integration-tests;late-start"
    ENVIRONMENT "${late-start-env}"
    SKIP_REGULAR_EXPRESSION "This test is skipped."
    FIXTURES_SETUP late-start-tracing
    FAIL_REGULAR_EXPRESSION "ERROR|FATAL|${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Validate test - validates the output
rocprofiler_add_integration_validate_test(
    late-start-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests;late-start"
    TIMEOUT 45
    SKIP_REGULAR_EXPRESSION "SKIPPED"
    FIXTURES_REQUIRED late-start-tracing
    ARGS --skip-if ${CMAKE_CURRENT_BINARY_DIR}/skipped)
