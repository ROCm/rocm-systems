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

project(rocprofiler-sdk-samples-buffered-api-tracing LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(rocprofiler-sdk REQUIRED)

add_library(buffered-api-tracing-client SHARED)
target_sources(buffered-api-tracing-client PRIVATE client.cpp client.hpp)
target_link_libraries(
    buffered-api-tracing-client
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-build-flags
            rocprofiler-sdk::samples-common-library)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
find_package(Threads REQUIRED)

add_executable(buffered-api-tracing)
target_sources(buffered-api-tracing PRIVATE main.cpp)
target_link_libraries(
    buffered-api-tracing
    PRIVATE buffered-api-tracing-client Threads::Threads
            rocprofiler-sdk::samples-build-flags rocprofiler-sdk::samples-common-library)

rocprofiler_samples_get_preload_env(PRELOAD_ENV buffered-api-tracing-client)
rocprofiler_samples_get_ld_library_path_env(LIBRARY_PATH_ENV)

set(buffered-api-tracing-env ${PRELOAD_ENV} ${LIBRARY_PATH_ENV})

add_test(NAME buffered-api-tracing COMMAND $<TARGET_FILE:buffered-api-tracing>)

set_tests_properties(
    buffered-api-tracing
    PROPERTIES TIMEOUT 45 LABELS "samples" ENVIRONMENT "${buffered-api-tracing-env}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
