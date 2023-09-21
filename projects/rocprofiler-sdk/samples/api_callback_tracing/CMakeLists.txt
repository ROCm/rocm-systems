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

project(rocprofiler-samples-callback-api-tracing LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

add_library(callback-api-tracing-client SHARED)
target_sources(callback-api-tracing-client PRIVATE client.cpp client.hpp)
target_link_libraries(callback-api-tracing-client
                      PRIVATE rocprofiler::rocprofiler-library)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
find_package(Threads REQUIRED)

add_executable(callback-api-tracing)
target_sources(callback-api-tracing PRIVATE main.cpp)
target_link_libraries(callback-api-tracing PRIVATE callback-api-tracing-client
                                                   Threads::Threads)

add_test(NAME callback-api-tracing COMMAND $<TARGET_FILE:callback-api-tracing>)

set_tests_properties(
    callback-api-tracing
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "samples"
        ENVIRONMENT
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV};HSA_TOOLS_LIB=$<TARGET_FILE:rocprofiler::rocprofiler-library>"
    )
