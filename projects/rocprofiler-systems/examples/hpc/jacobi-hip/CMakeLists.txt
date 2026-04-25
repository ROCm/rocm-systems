# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

check_rocprofsys_disable_examples("jacobi-hip" _JACOBI_HIP_DISABLED)
check_rocprofsys_disable_examples("mpi" _MPI_DISABLED)
if(_JACOBI_HIP_DISABLED OR _MPI_DISABLED)
    return()
endif()

if(NOT HIPCC_EXECUTABLE)
    rocprofiler_systems_message(AUTHOR_WARNING "hipcc not found. Cannot build jacobi-hip target.")
    return()
endif()

if(NOT CMAKE_CXX_COMPILER_IS_HIPCC)
    if(
        CMAKE_CXX_COMPILER STREQUAL HIPCC_EXECUTABLE
        OR "${CMAKE_CXX_COMPILER}" MATCHES "hipcc"
    )
        set(CMAKE_CXX_COMPILER_IS_HIPCC 1 CACHE BOOL "HIP compiler")
    endif()
endif()

find_package(MPI QUIET COMPONENTS CXX)
if(NOT MPI_CXX_FOUND)
    rocprofiler_systems_message(AUTHOR_WARNING "MPI not found, this is required for jacobi-hip example")
    return()
endif()

find_library(ROCTX64_LIBRARY NAMES roctx64 PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64)

find_library(
    ROCTRACER64_LIBRARY
    NAMES roctracer64
    PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64
)

if(NOT ROCTX64_LIBRARY OR NOT ROCTRACER64_LIBRARY)
    rocprofiler_systems_message(AUTHOR_WARNING "roctracer not found, this is required for jacobi-hip example")
    return()
endif()

set(_JACOBI_HIP_SOURCES
    HaloExchange.hip
    Input.hip
    JacobiIteration.hip
    JacobiMain.hip
    JacobiRun.hip
    JacobiSetup.hip
    Laplacian.hip
    Norm.hip
)
set_source_files_properties(${_JACOBI_HIP_SOURCES} PROPERTIES LANGUAGE CXX)

set(HPC_EXAMPLE_NAME "jacobi-hip")
add_executable(${HPC_EXAMPLE_NAME} ${_JACOBI_HIP_SOURCES})

target_compile_options(${HPC_EXAMPLE_NAME} PRIVATE -xhip)
target_compile_definitions(${HPC_EXAMPLE_NAME} PRIVATE __HIP_PLATFORM_HCC__)
target_include_directories(
    ${HPC_EXAMPLE_NAME}
    PRIVATE ${ROCM_PATH}/roctracer/include ${ROCM_PATH}/include/roctracer
)
target_compile_options(
    ${HPC_EXAMPLE_NAME}
    PRIVATE -O3 -g -ggdb -fPIC -march=native -Wall -Wno-unused-result
)

foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
    target_compile_options(${HPC_EXAMPLE_NAME} PRIVATE --offload-arch=${arch})
    target_link_options(${HPC_EXAMPLE_NAME} PRIVATE --offload-arch=${arch})
endforeach()

target_link_libraries(
    ${HPC_EXAMPLE_NAME}
    PRIVATE MPI::MPI_CXX ${ROCTX64_LIBRARY} ${ROCTRACER64_LIBRARY}
)

if(NOT CMAKE_CXX_COMPILER_IS_HIPCC)
    rocprofiler_systems_custom_compilation(COMPILER ${HIPCC_EXECUTABLE} TARGET ${HPC_EXAMPLE_NAME})
endif()

rocprofiler_systems_install_hpc_example(${HPC_EXAMPLE_NAME})
