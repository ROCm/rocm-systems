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

project(rocprofiler-sdk-samples-pc-sampling-integration-test LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(rocprofiler-sdk REQUIRED)

add_library(pc-sampling-integration-test-client SHARED)
target_sources(
    pc-sampling-integration-test-client
    PRIVATE address_translation.cpp
            address_translation.hpp
            client.cpp
            client.hpp
            cid_retirement.cpp
            cid_retirement.hpp
            codeobj.cpp
            codeobj.hpp
            external_cid.cpp
            external_cid.hpp
            kernel_tracing.cpp
            kernel_tracing.hpp
            pcs.hpp
            pcs.cpp
            utils.hpp
            utils.cpp)
target_link_libraries(
    pc-sampling-integration-test-client
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            rocprofiler-sdk::tests-common-library)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
find_package(Threads REQUIRED)

add_executable(pc-sampling-integration-test)
target_sources(pc-sampling-integration-test PRIVATE main.cpp)
target_link_libraries(
    pc-sampling-integration-test
    PRIVATE pc-sampling-integration-test-client Threads::Threads
            rocprofiler-sdk::tests-build-flags)

add_test(NAME pc-sampling-integration-test
         COMMAND $<TARGET_FILE:pc-sampling-integration-test>)

set(pc-sampling-integration-test-env "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set_tests_properties(
    pc-sampling-integration-test
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests;pc-sampling"
               SKIP_REGULAR_EXPRESSION
               "PC sampling unavailable"
               ENVIRONMENT
               "${pc-sampling-integration-test-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
