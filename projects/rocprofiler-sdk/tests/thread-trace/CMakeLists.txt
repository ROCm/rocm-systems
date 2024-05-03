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

project(
    rocprofiler-tests-thread-trace
    LANGUAGES CXX HIP
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(amd_comgr REQUIRED)

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

add_library(thread-trace-api-test-lib SHARED)
target_sources(thread-trace-api-test-lib PRIVATE verify_data.cpp)
target_link_libraries(
    thread-trace-api-test-lib PRIVATE rocprofiler::rocprofiler rocprofiler-sdk-codeobj
                                      amd_comgr dw)

set_source_files_properties(kernel_run.cpp PROPERTIES COMPILE_FLAGS "-g -O2")
set_source_files_properties(kernel_run.cpp PROPERTIES LANGUAGE HIP)

add_executable(thread-trace-api-test-binary)
target_sources(thread-trace-api-test-binary PRIVATE kernel_run.cpp)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}:$<TARGET_FILE:thread-trace-api-test-binary>")
else()
    set(PRELOAD_ENV "LD_PRELOAD=$<TARGET_FILE:thread-trace-api-test-binary>")
endif()

target_link_libraries(thread-trace-api-test-binary PRIVATE thread-trace-api-test-lib)

add_test(NAME thread-trace-api-tests COMMAND $<TARGET_FILE:thread-trace-api-test-binary>)

set_tests_properties(
    thread-trace-api-tests
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${PRELOAD_ENV}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
