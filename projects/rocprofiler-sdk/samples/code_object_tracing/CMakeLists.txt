#
#
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

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

project(rocprofiler-sdk-samples-code-object-tracing LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(rocprofiler-sdk REQUIRED)

add_library(code-object-tracing-client SHARED)
target_sources(code-object-tracing-client PRIVATE client.cpp)
target_link_libraries(
    code-object-tracing-client
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-build-flags
            rocprofiler-sdk::samples-common-library)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
find_package(Threads REQUIRED)

add_executable(code-object-tracing)
target_sources(code-object-tracing PRIVATE main.cpp)
target_link_libraries(
    code-object-tracing PRIVATE code-object-tracing-client Threads::Threads
                                rocprofiler-sdk::samples-build-flags)

# WSL2/DXG has no /dev/kfd and schedules GPU work through the dxg path. Under WSL2/DXG
# these workloads run slower in practice and need more timeout headroom (root cause TBD),
# so scale the timeout only there. Non-WSL keeps the develop baseline (45); WSL scales it
# up.
rocprofiler_sdk_wsl_timeout_scale(_CODE_OBJECT_TRACING_TIMEOUT BASE 45)

add_test(NAME code-object-tracing COMMAND $<TARGET_FILE:code-object-tracing>)

set_tests_properties(
    code-object-tracing
    PROPERTIES TIMEOUT ${_CODE_OBJECT_TRACING_TIMEOUT} LABELS "samples" ENVIRONMENT
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
