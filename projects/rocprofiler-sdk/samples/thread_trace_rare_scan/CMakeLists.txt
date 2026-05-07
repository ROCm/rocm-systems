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

project(rocprofiler-sdk-samples-thread-trace-rare-scan LANGUAGES CXX HIP)

find_package(rocprofiler-sdk REQUIRED)
find_package(rocprofiler-sdk-roctx REQUIRED)
find_package(rocprof-trace-decoder REQUIRED)

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

add_executable(thread-trace-rare-scan-sample)
target_sources(thread-trace-rare-scan-sample PRIVATE main.cpp agent.cpp)

target_link_libraries(
    thread-trace-rare-scan-sample
    PUBLIC rocprofiler-sdk::samples-build-flags
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-common-library
            rocprofiler-sdk-roctx::rocprofiler-sdk-roctx
            rocprof-trace-decoder::rocprof-trace-decoder)

add_test(NAME thread-trace-rare-scan-sample
         COMMAND $<TARGET_FILE:thread-trace-rare-scan-sample>)

set(IS_DISABLED ON)
rocprofiler_sdk_sqtt_triple_buffer_disabled(_TB_DISABLED)
if(NOT _TB_DISABLED)
    set(IS_DISABLED OFF)
endif()

rocprofiler_samples_get_preload_env(PRELOAD_ENV)
# Keep the test fast by default; the sample itself supports RUN_SECONDS=N.
list(APPEND PRELOAD_ENV "RUN_SECONDS=3")

set_tests_properties(
    thread-trace-rare-scan-sample
    PROPERTIES TIMEOUT 60 ENVIRONMENT "${PRELOAD_ENV}" LABELS
               "samples;thread-trace;triple-buffer" DISABLED ${IS_DISABLED})
