# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(rocprofiler-systems-openmp LANGUAGES CXX)

if(ROCPROFSYS_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(
        ${PROJECT_NAME} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
        OR ${_DIR} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
    )
        return()
    endif()
endif()

file(GLOB common_source ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp)
add_library(openmp-common OBJECT ${common_source})
target_include_directories(openmp-common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_executable(
    openmp-cg
    ${CMAKE_CURRENT_SOURCE_DIR}/CG/cg.cpp
    $<TARGET_OBJECTS:openmp-common>
)
add_executable(
    openmp-lu
    ${CMAKE_CURRENT_SOURCE_DIR}/LU/lu.cpp
    $<TARGET_OBJECTS:openmp-common>
)

option(USE_CLANG_OMP "Use the clang OpenMP if available" ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    find_package(OpenMP REQUIRED)
    target_link_libraries(openmp-common PUBLIC OpenMP::OpenMP_CXX)
    set(ROCPROFSYS_OPENMP_USING_LIBOMP_LIBRARY
        ON
        CACHE INTERNAL
        "Used by rocprofiler-systems testing"
        FORCE
    )
else()
    find_program(CLANGXX_EXECUTABLE NAMES clang++)
    find_library(
        LIBOMP_LIBRARY
        NAMES omp omp5 ${CMAKE_SHARED_LIBRARY_PREFIX}omp${CMAKE_SHARED_LIBRARY_SUFFIX}.5
    )
    if(
        CLANGXX_EXECUTABLE
        AND LIBOMP_LIBRARY
        AND COMMAND rocprofiler_systems_custom_compilation
        AND USE_CLANG_OMP
    )
        target_compile_options(openmp-common PUBLIC -W -Wall -fopenmp=libomp)
        target_link_libraries(openmp-common PUBLIC ${LIBOMP_LIBRARY})
        rocprofiler_systems_custom_compilation(COMPILER ${CLANGXX_EXECUTABLE}
                                               TARGET openmp-common
        )
        rocprofiler_systems_custom_compilation(COMPILER ${CLANGXX_EXECUTABLE}
                                               TARGET openmp-cg
        )
        rocprofiler_systems_custom_compilation(COMPILER ${CLANGXX_EXECUTABLE}
                                               TARGET openmp-lu
        )
        set(ROCPROFSYS_OPENMP_USING_LIBOMP_LIBRARY
            ON
            CACHE INTERNAL
            "Used by rocprofiler-systems testing"
            FORCE
        )
    else()
        find_package(OpenMP REQUIRED)
        target_link_libraries(openmp-common PUBLIC OpenMP::OpenMP_CXX)
        set(ROCPROFSYS_OPENMP_USING_LIBOMP_LIBRARY
            OFF
            CACHE INTERNAL
            "Used by rocprofiler-systems testing"
            FORCE
        )
    endif()
endif()

target_link_libraries(openmp-cg PRIVATE openmp-common)
target_link_libraries(openmp-lu PRIVATE openmp-common)

if(ROCPROFSYS_INSTALL_EXAMPLES)
    install(
        TARGETS openmp-cg openmp-lu
        DESTINATION bin
        COMPONENT rocprofiler-systems-examples
    )
endif()

set(DEFAULT_GPU_TARGETS
    "gfx900"
    "gfx906"
    "gfx908"
    "gfx90a"
    "gfx942"
    "gfx950"
    "gfx1030"
    "gfx1010"
    "gfx1100"
    "gfx1101"
    "gfx1102"
)

set(GPU_TARGETS "${DEFAULT_GPU_TARGETS}" CACHE STRING "GPU targets to compile for")

if(ROCPROFSYS_USE_ROCM)
    add_subdirectory(external)
else()
    rocprofiler_systems_message(
      WARNING
      "ROCPROFSYS OMPVV CTests requires ROCm to be installed. Disabling OMPVV CTests..."
    )
endif()

if(ROCPROFSYS_DISABLE_EXAMPLES)
    if(NOT "openmp-target" IN_LIST ROCPROFSYS_DISABLE_EXAMPLES)
        add_subdirectory(target)
    endif()
else()
    add_subdirectory(target)
endif()
