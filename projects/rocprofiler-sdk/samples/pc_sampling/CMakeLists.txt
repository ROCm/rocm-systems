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

project(rocprofiler-sdk-samples-pc-sampling LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(rocprofiler-sdk REQUIRED)

add_library(pc-sampling-client SHARED)
target_sources(pc-sampling-client PRIVATE client.cpp pcs.hpp pcs.cpp utils.hpp utils.cpp)
target_link_libraries(
    pc-sampling-client
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-build-flags
            rocprofiler-sdk::samples-common-library)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
find_package(Threads REQUIRED)

add_executable(pc-sampling)
target_sources(pc-sampling PRIVATE main.cpp)
target_link_libraries(
    pc-sampling
    PRIVATE pc-sampling-client Threads::Threads rocprofiler-sdk::samples-build-flags
            rocprofiler-sdk::samples-common-library)

rocprofiler_samples_get_preload_env(PRELOAD_ENV pc-sampling-client)
rocprofiler_samples_get_ld_library_path_env(LIBRARY_PATH_ENV)

# Check if PC sampling is disabled and whether we should disable the test
rocprofiler_sdk_pc_sampling_disabled(IS_PC_SAMPLING_DISABLED)

set(pc-sampling-env ${PRELOAD_ENV} ${LIBRARY_PATH_ENV})

add_test(NAME pc-sampling COMMAND $<TARGET_FILE:pc-sampling>)

set_tests_properties(
    pc-sampling
    PROPERTIES TIMEOUT
               45
               LABELS
               "samples;pc-sampling"
               ENVIRONMENT
               "${pc-sampling-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               SKIP_REGULAR_EXPRESSION
               "PC sampling unavailable"
               DISABLED
               "${IS_PC_SAMPLING_DISABLED}")
