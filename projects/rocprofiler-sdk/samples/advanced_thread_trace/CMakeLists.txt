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

project(rocprofiler-sdk-samples-advanced-thread-trace LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(rocprofiler-sdk REQUIRED)

add_library(advanced-thread-trace-client SHARED)
target_sources(advanced-thread-trace-client PRIVATE client.cpp)
target_link_libraries(
    advanced-thread-trace-client
    PRIVATE rocprofiler::rocprofiler rocprofiler::samples-build-flags
            rocprofiler-sdk-codeobj rocprofiler::samples-common-library)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(main.cpp PROPERTIES COMPILE_FLAGS "-g")
find_package(Threads REQUIRED)

add_executable(advanced-thread-trace)
target_sources(advanced-thread-trace PRIVATE main.cpp)
target_link_libraries(
    advanced-thread-trace PRIVATE advanced-thread-trace-client Threads::Threads
                                  rocprofiler::samples-build-flags)

add_test(NAME advanced-thread-trace COMMAND $<TARGET_FILE:advanced-thread-trace>)

set_tests_properties(
    advanced-thread-trace
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "samples"
        ENVIRONMENT
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV};HSA_TOOLS_LIB=$<TARGET_FILE:rocprofiler::rocprofiler-shared-library>"
        FAIL_REGULAR_EXPRESSION
        "threw an exception")
