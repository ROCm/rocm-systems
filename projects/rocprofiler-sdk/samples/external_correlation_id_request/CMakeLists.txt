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

project(rocprofiler-sdk-samples-external-correlation-id-request LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

if(ROCPROFILER_DISABLE_UNSTABLE_CTESTS AND ROCPROFILER_MEMCHECK MATCHES "LeakSanitizer")
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

find_package(rocprofiler-sdk REQUIRED)

add_library(external-correlation-id-request-client SHARED)
target_sources(external-correlation-id-request-client PRIVATE client.cpp client.hpp)
target_link_libraries(
    external-correlation-id-request-client
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-build-flags
            rocprofiler-sdk::samples-common-library)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
find_package(Threads REQUIRED)

add_executable(external-correlation-id-request)
target_sources(external-correlation-id-request PRIVATE main.cpp)
target_link_libraries(
    external-correlation-id-request
    PRIVATE external-correlation-id-request-client Threads::Threads
            rocprofiler-sdk::samples-build-flags rocprofiler-sdk::samples-common-library)

rocprofiler_samples_get_preload_env(PRELOAD_ENV external-correlation-id-request-client)
rocprofiler_samples_get_ld_library_path_env(LIBRARY_PATH_ENV)

set(external-correlation-id-request-env ${PRELOAD_ENV} ${LIBRARY_PATH_ENV})

add_test(NAME external-correlation-id-request
         COMMAND $<TARGET_FILE:external-correlation-id-request>)

set_tests_properties(
    external-correlation-id-request
    PROPERTIES TIMEOUT
               45
               LABELS
               "samples"
               ENVIRONMENT
               "${external-correlation-id-request-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${IS_DISABLED}")
