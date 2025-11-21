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

project(rocprofiler-sdk-tests-lib-vector-operations LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

set_source_files_properties(vector-ops.cpp PROPERTIES LANGUAGE HIP)
add_library(vector-ops-shared-library SHARED)
target_sources(vector-ops-shared-library PRIVATE vector-ops.cpp)
target_compile_options(vector-ops-shared-library PRIVATE -W -Wall -Wextra -Wpedantic
                                                         -Wshadow -Werror)
target_include_directories(vector-ops-shared-library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(vector-ops-shared-library PROPERTIES OUTPUT_NAME vector-ops)

find_package(Threads REQUIRED)
target_link_libraries(vector-ops-shared-library
                      PRIVATE Threads::Threads rocprofiler-sdk::tests-common-library)
