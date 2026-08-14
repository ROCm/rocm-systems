#
# Integration tests
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

if(COMMAND rocprofiler_deactivate_clang_tidy)
    rocprofiler_deactivate_clang_tidy()
endif()

project(rocprofiler-sdk-tests LANGUAGES C CXX)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "tests")
set(ROCPROFILER_SDK_TESTS_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
set(ROCPROFILER_SDK_TESTS_BINARY_DIR "${PROJECT_BINARY_DIR}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        "Release"
        CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# this should be defaulted to OFF by ROCm 7.0.1 or 7.1 this should only used to disable
# tests in extreme circumstances
option(ROCPROFILER_DISABLE_UNSTABLE_CTESTS "Disable unstable tests" ON)

enable_testing()
include(CTest)

include(GNUInstallDirs)
# always use lib instead of lib64
set(CMAKE_INSTALL_LIBDIR "lib")

# define the library output directory
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
else()
    set(CMAKE_MESSAGE_INDENT "[${PROJECT_NAME}] ")
endif()

# needed for validation
find_package(Python3 REQUIRED)

# generally needed
find_package(rocprofiler-sdk REQUIRED)
find_package(rocprofiler_sdk_pytest REQUIRED)

# [config-timing] capture the wall-clock start of test configuration so we can report
# the total time and attribute it to individual subdirectories below
rocprofiler_sdk_wallclock_ms(ROCPROFILER_SDK_TESTS_CONFIG_START_MS)

# get the gfx architectures that are present on the system
rocprofiler_sdk_get_gfx_architectures(rocprofiler-sdk-tests-gfx-info ECHO)

# configure python module <BINARY_DIR>/rocprofiler_sdk/pytest_utils
rocprofiler_sdk_timed_add_subdirectory(pytest-packages)

# common utilities
rocprofiler_sdk_timed_add_subdirectory(common)

# unit-test subdirectory only exists in rocprofiler-sdk-tests package
rocprofiler_sdk_timed_add_subdirectory(unit-tests)

# tool libraries used for data collection during integration tests
rocprofiler_sdk_timed_add_subdirectory(tools)

# libraries used by integration test applications
rocprofiler_sdk_timed_add_subdirectory(lib)

# applications used by integration tests
rocprofiler_sdk_timed_add_subdirectory(bin)

# validation tests
rocprofiler_sdk_timed_add_subdirectory(kernel-tracing)
rocprofiler_sdk_timed_add_subdirectory(async-copy-tracing)
rocprofiler_sdk_timed_add_subdirectory(rocshmem-trace)
rocprofiler_sdk_timed_add_subdirectory(hsa-memory-allocation)
rocprofiler_sdk_timed_add_subdirectory(scratch-memory-tracing)
rocprofiler_sdk_timed_add_subdirectory(c-tool)
rocprofiler_sdk_timed_add_subdirectory(thread-trace)
rocprofiler_sdk_timed_add_subdirectory(pc_sampling)
rocprofiler_sdk_timed_add_subdirectory(hip-graph-tracing)
rocprofiler_sdk_timed_add_subdirectory(counter-collection)
rocprofiler_sdk_timed_add_subdirectory(openmp-tools)
rocprofiler_sdk_timed_add_subdirectory(rocdecode)
rocprofiler_sdk_timed_add_subdirectory(rocjpeg)
rocprofiler_sdk_timed_add_subdirectory(hipfile)
rocprofiler_sdk_timed_add_subdirectory(hip-host-tracing)
rocprofiler_sdk_timed_add_subdirectory(code-object-multi-threaded)
rocprofiler_sdk_timed_add_subdirectory(spm)

# environment variable tests
rocprofiler_sdk_timed_add_subdirectory(environment)

# rocpd validation tests
rocprofiler_sdk_timed_add_subdirectory(rocpd)
rocprofiler_sdk_timed_add_subdirectory(rocpd-api)

# rocattach validation tests
rocprofiler_sdk_timed_add_subdirectory(rocattach)

# rocprofv3 validation tests
rocprofiler_sdk_timed_add_subdirectory(rocprofv3)

# late-start tracing validation tests
rocprofiler_sdk_timed_add_subdirectory(late-start-tracing)

# python bindings
rocprofiler_sdk_timed_add_subdirectory(python-bindings)
rocprofiler_sdk_timed_add_subdirectory(rocprofv3-avail)

# [config-timing] report the total configure wall-clock time and the aggregate cost of
# the repeated rocminfo invocations so the dispatched CI run pinpoints the bottleneck
rocprofiler_sdk_wallclock_ms(ROCPROFILER_SDK_TESTS_CONFIG_END_MS)
math(EXPR ROCPROFILER_SDK_TESTS_CONFIG_ELAPSED_MS
     "${ROCPROFILER_SDK_TESTS_CONFIG_END_MS} - ${ROCPROFILER_SDK_TESTS_CONFIG_START_MS}"
     )
get_property(ROCPROFILER_SDK_ROCMINFO_TOTAL_MS GLOBAL
             PROPERTY ROCPROFILER_SDK_ROCMINFO_TOTAL_MS)
get_property(ROCPROFILER_SDK_ROCMINFO_CALL_COUNT GLOBAL
             PROPERTY ROCPROFILER_SDK_ROCMINFO_CALL_COUNT)
message(
    STATUS
        "[config-timing] rocprofiler-sdk-tests configure took ${ROCPROFILER_SDK_TESTS_CONFIG_ELAPSED_MS} ms total"
    )
message(
    STATUS
        "[config-timing] rocminfo was invoked ${ROCPROFILER_SDK_ROCMINFO_CALL_COUNT} time(s) for a cumulative ${ROCPROFILER_SDK_ROCMINFO_TOTAL_MS} ms"
    )
