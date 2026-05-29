#
#
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

project(rocprofiler-sdk-samples-thread-trace LANGUAGES CXX HIP)

find_package(rocprofiler-sdk REQUIRED)
find_package(rocprofiler-sdk-roctx REQUIRED)
find_package(rocprof-trace-decoder REQUIRED)
find_package(Threads REQUIRED)
find_package(
    amd_comgr
    REQUIRED
    CONFIG
    HINTS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATHS
    ${rocm_version_DIR}
    ${ROCM_PATH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

set_source_files_properties(
    main.cpp PROPERTIES LANGUAGE HIP COMPILE_FLAGS "${CMAKE_HIP_FLAGS_RELWITHDEBINFO}")

# Single executable shared by both flavours; trace duration is controlled by the
# RUN_SECONDS env var (simple test runs short, buffered runs longer).
add_executable(thread-trace-sample)
target_sources(thread-trace-sample PRIVATE main.cpp)
target_link_libraries(
    thread-trace-sample
    PUBLIC rocprofiler-sdk::samples-build-flags
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-common-library
            rocprofiler-sdk-roctx::rocprofiler-sdk-roctx Threads::Threads)

# Simple agent: synchronous decode of each chunk in shader_data_callback.
add_library(thread-trace-simple-client SHARED)
target_sources(thread-trace-simple-client PRIVATE agent_simple.cpp)
target_link_libraries(
    thread-trace-simple-client
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-build-flags
            rocprofiler-sdk::samples-common-library amd_comgr
            rocprof-trace-decoder::rocprof-trace-decoder)

# Buffered agent: triple-buffered SQTT capture with inline quick-scan; emits the same
# disassembled hotspot report as the simple agent (needs amd_comgr via the shared
# thread_trace_hotspots.hpp).
add_library(thread-trace-buffered-client SHARED)
target_sources(thread-trace-buffered-client PRIVATE agent_buffered.cpp)
target_link_libraries(
    thread-trace-buffered-client
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-build-flags
            rocprofiler-sdk::samples-common-library amd_comgr
            rocprof-trace-decoder::rocprof-trace-decoder)

# --- simple test ---
rocprofiler_samples_get_preload_env(THREAD_TRACE_SIMPLE_PRELOAD_ENV
                                    thread-trace-simple-client)

add_test(NAME thread-trace-simple-sample COMMAND $<TARGET_FILE:thread-trace-sample>)

set_tests_properties(
    thread-trace-simple-sample
    PROPERTIES TIMEOUT 60 ENVIRONMENT "${THREAD_TRACE_SIMPLE_PRELOAD_ENV}" LABELS
               "samples;thread-trace")

# --- buffered test (requires triple-buffer support) ---
rocprofiler_samples_get_preload_env(THREAD_TRACE_BUFFERED_PRELOAD_ENV
                                    thread-trace-buffered-client)

set(IS_DISABLED ON)
rocprofiler_sdk_sqtt_triple_buffer_disabled(_TB_DISABLED)
if(NOT _TB_DISABLED)
    set(IS_DISABLED OFF)
endif()

add_test(NAME thread-trace-buffered-sample COMMAND $<TARGET_FILE:thread-trace-sample>)

set_tests_properties(
    thread-trace-buffered-sample
    PROPERTIES TIMEOUT 60 ENVIRONMENT
               "${THREAD_TRACE_BUFFERED_PRELOAD_ENV};RUN_SECONDS=3" LABELS
               "samples;thread-trace;triple-buffer" DISABLED ${IS_DISABLED})
