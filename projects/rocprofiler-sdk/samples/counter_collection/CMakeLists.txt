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

add_library(counter-collection-client SHARED)
target_sources(counter-collection-client PRIVATE client.cpp client.hpp)
target_link_libraries(counter-collection-client PRIVATE rocprofiler::rocprofiler)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
find_package(Threads REQUIRED)

add_executable(counter-collection)
target_sources(counter-collection PRIVATE main.cpp)
target_link_libraries(counter-collection PRIVATE counter-collection-client
                                                 Threads::Threads)

add_test(NAME counter-collection COMMAND $<TARGET_FILE:counter-collection>)

set_tests_properties(
    counter-collection
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "samples"
        ENVIRONMENT
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV};HSA_TOOLS_LIB=$<TARGET_FILE:rocprofiler::rocprofiler-library>"
    )
