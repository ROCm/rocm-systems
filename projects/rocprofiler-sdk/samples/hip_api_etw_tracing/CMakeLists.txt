#
# HIP API tracing through the Windows ETW consumer
#
# Plain CXX, unlike the other samples: MSVC is not a HIP compiler, so the application calls the
# HIP runtime through its import library instead of launching kernels.
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(rocprofiler-sdk-samples-hip-api-etw-tracing LANGUAGES CXX)

find_package(rocprofiler-sdk REQUIRED)
find_package(hip REQUIRED)

add_library(hip-api-etw-tracing-client SHARED)
target_sources(hip-api-etw-tracing-client PRIVATE client.cpp client.hpp)
target_compile_features(hip-api-etw-tracing-client PRIVATE cxx_std_17)
target_link_libraries(hip-api-etw-tracing-client PRIVATE rocprofiler-sdk::rocprofiler-sdk)

add_executable(hip-api-etw-tracing)
target_sources(hip-api-etw-tracing PRIVATE main.cpp)
target_compile_features(hip-api-etw-tracing PRIVATE cxx_std_17)
target_link_libraries(hip-api-etw-tracing PRIVATE hip-api-etw-tracing-client hip::host)

# The test needs both the client and rocprofiler-sdk itself on the loader search path.
add_test(NAME hip-api-etw-tracing COMMAND $<TARGET_FILE:hip-api-etw-tracing>)

# Creating an ETW session requires membership in 'Performance Log Users', which not every
# machine grants, so an under-privileged run is reported as skipped rather than failed.
set_tests_properties(
    hip-api-etw-tracing
    PROPERTIES TIMEOUT
               45
               LABELS
               "samples"
               SKIP_RETURN_CODE
               77
               ENVIRONMENT_MODIFICATION
               "PATH=path_list_prepend:$<TARGET_FILE_DIR:hip-api-etw-tracing-client>"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
