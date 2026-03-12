# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

check_rocprofsys_disable_examples("matrix-exponential" _MATRIX_EXPONENTIAL_DISABLED)
check_rocprofsys_disable_examples("openmp" _OPENMP_DISABLED)
if(_MATRIX_EXPONENTIAL_DISABLED OR _OPENMP_DISABLED)
    return()
endif()

if(NOT HIPCC_EXECUTABLE)
    rocprofiler_systems_message(AUTHOR_WARNING "hipcc not found. Cannot build matrix-exponential-streams-sync-hip target.")
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

find_package(OpenMP QUIET COMPONENTS CXX)
if(NOT OpenMP_CXX_FOUND)
    rocprofiler_systems_message(AUTHOR_WARNING "OpenMP not found, this is required for matrix-exponential example")
    return()
endif()

find_library(ROCBLAS_LIBRARY NAMES rocblas PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64)
if(NOT ROCBLAS_LIBRARY)
    rocprofiler_systems_message(AUTHOR_WARNING "rocblas not found, this is required for matrix-exponential example")
    return()
endif()

find_library(
    ROCTX_SDK_LIBRARY
    NAMES rocprofiler-sdk-roctx
    PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64
)
if(NOT ROCTX_SDK_LIBRARY)
    rocprofiler_systems_message(AUTHOR_WARNING "rocprofiler-sdk-roctx not found, this is required for matrix-exponential example")
    return()
endif()

set(HPC_EXAMPLE_NAME "matrix-exponential-streams-sync-hip")
add_executable(${HPC_EXAMPLE_NAME} mat_exp.cpp)

target_compile_options(
    ${HPC_EXAMPLE_NAME}
    PRIVATE ${OpenMP_CXX_FLAGS} -Wno-vla-cxx-extension
)
target_link_options(${HPC_EXAMPLE_NAME} PRIVATE -fopenmp)
target_link_libraries(${HPC_EXAMPLE_NAME} PRIVATE ${ROCBLAS_LIBRARY} ${ROCTX_SDK_LIBRARY})

foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
    target_compile_options(${HPC_EXAMPLE_NAME} PRIVATE --offload-arch=${arch})
    target_link_options(${HPC_EXAMPLE_NAME} PRIVATE --offload-arch=${arch})
endforeach()

if(NOT CMAKE_CXX_COMPILER_IS_HIPCC)
    rocprofiler_systems_custom_compilation(COMPILER ${HIPCC_EXECUTABLE} TARGET ${HPC_EXAMPLE_NAME})
endif()

rocprofiler_systems_install_hpc_example(${HPC_EXAMPLE_NAME})
