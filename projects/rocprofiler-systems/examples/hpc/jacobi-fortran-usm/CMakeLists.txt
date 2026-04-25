# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

check_rocprofsys_disable_examples("jacobi-fortran-usm" _JACOBI_FORTAN_USM_DISABLED)
check_rocprofsys_disable_examples("openmp" _OPENMP_DISABLED)
if(_JACOBI_FORTAN_USM_DISABLED OR _OPENMP_DISABLED)
    return()
endif()

if(NOT amdflang_EXECUTABLE)
    rocprofiler_systems_message(AUTHOR_WARNING "amdflang not found, this is required for jacobi-fortran-usm example")
    return()
endif()

check_compiler_version(amdflang_EXECUTABLE _AMDFLANG_VERSION)
if(_AMDFLANG_VERSION LESS 20) # amdflang openmp requirement
    rocprofiler_systems_message(AUTHOR_WARNING "amdflang version ${_AMDFLANG_VERSION} is less than the minimum required version 20 for jacobi-fortran-usm example")
    return()
endif()

set(offload_arch_flags "-fopenmp")
set(offload_link_flags "-fopenmp")
foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
    rocprofiler_systems_lookup_gfx(${arch} GPU_CATEGORIES)
    if(NOT "instinct" IN_LIST GPU_CATEGORIES)
        continue()
    endif()
    list(APPEND offload_arch_flags --offload-arch=${arch})
    list(APPEND offload_link_flags --offload-arch=${arch})
endforeach()

list(LENGTH offload_arch_flags _offload_flag_count)
if(_offload_flag_count LESS_EQUAL 1)
    rocprofiler_systems_message(AUTHOR_WARNING
        "Skipping jacobi-fortran-usm example: an instinct GPU is required"
    )
    return()
endif()

set(CMAKE_Fortran_COMPILER ${amdflang_EXECUTABLE})
enable_language(Fortran)

set(HPC_EXAMPLE_NAME "jacobi-fortran-usm")
add_executable(
    ${HPC_EXAMPLE_NAME}
    boundary.f90
    input.f90
    jacobi.f90
    kind.f90
    laplacian.f90
    main.f90
    mesh.f90
    norm.f90
    update.f90
)

target_compile_options(
    ${HPC_EXAMPLE_NAME}
    PRIVATE -g -O3 -ffast-math -ffree-form ${offload_arch_flags}
)

target_link_options(${HPC_EXAMPLE_NAME} PRIVATE ${offload_link_flags})

rocprofiler_systems_install_hpc_example(${HPC_EXAMPLE_NAME})
