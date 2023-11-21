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

project(rocprofiler-samples-counter-collection LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

if(NOT TARGET rocprofiler::rocprofiler)
    find_package(rocprofiler REQUIRED)
endif()

add_library(counter-collection-buffer-client SHARED)
target_sources(counter-collection-buffer-client PRIVATE client.cpp client.hpp)
target_link_libraries(counter-collection-buffer-client PRIVATE rocprofiler::rocprofiler)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(counter-collection-buffer)
target_sources(counter-collection-buffer PRIVATE main.cpp)
target_link_libraries(counter-collection-buffer PRIVATE counter-collection-buffer-client
                                                        Threads::Threads)

add_test(NAME counter-collection-buffer COMMAND $<TARGET_FILE:counter-collection-buffer>)

set_tests_properties(
    counter-collection-buffer
    PROPERTIES
        TIMEOUT
        300
        LABELS
        "samples"
        ENVIRONMENT
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV};HSA_TOOLS_LIB=$<TARGET_FILE:rocprofiler::rocprofiler-shared-library>"
        FAIL_REGULAR_EXPRESSION
        "threw an exception")

add_library(counter-collection-functional-counter-client SHARED)
target_sources(counter-collection-functional-counter-client
               PRIVATE print_functional_counters.cpp client.hpp)
target_link_libraries(counter-collection-functional-counter-client
                      PRIVATE rocprofiler::rocprofiler)
add_executable(counter-collection-print-functional-counters)
target_sources(counter-collection-print-functional-counters PRIVATE main.cpp)
target_link_libraries(
    counter-collection-print-functional-counters
    PRIVATE counter-collection-functional-counter-client Threads::Threads)

add_test(NAME counter-collection-print-functional-counters
         COMMAND $<TARGET_FILE:counter-collection-print-functional-counters>)

set_tests_properties(
    counter-collection-print-functional-counters
    PROPERTIES
        TIMEOUT
        300
        LABELS
        "samples"
        ENVIRONMENT
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV};HSA_TOOLS_LIB=$<TARGET_FILE:rocprofiler::rocprofiler-shared-library>"
        FAIL_REGULAR_EXPRESSION
        "threw an exception")
