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

project(rocprofiler-sdk-samples-hip-event-tracing LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(rocprofiler-sdk REQUIRED)

add_library(hip-event-tracing-client SHARED)
target_sources(hip-event-tracing-client PRIVATE client.cpp client.hpp)
target_link_libraries(
    hip-event-tracing-client
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-build-flags
            rocprofiler-sdk::samples-common-library)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
find_package(Threads REQUIRED)

add_executable(hip-event-tracing)
target_sources(hip-event-tracing PRIVATE main.cpp)
target_link_libraries(
    hip-event-tracing
    PRIVATE hip-event-tracing-client Threads::Threads
            rocprofiler-sdk::samples-build-flags rocprofiler-sdk::samples-common-library)

rocprofiler_samples_get_preload_env(PRELOAD_ENV hip-event-tracing-client)
rocprofiler_samples_get_ld_library_path_env(LIBRARY_PATH_ENV)

set(hip-event-tracing-env ${PRELOAD_ENV} ${LIBRARY_PATH_ENV})

add_test(NAME hip-event-tracing COMMAND $<TARGET_FILE:hip-event-tracing>)

set_tests_properties(
    hip-event-tracing
    PROPERTIES TIMEOUT 60 LABELS "samples" ENVIRONMENT "${hip-event-tracing-env}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
