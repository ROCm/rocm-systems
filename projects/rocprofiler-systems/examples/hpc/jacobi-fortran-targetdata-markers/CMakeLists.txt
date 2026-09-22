# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

check_rocprofsys_disable_examples("jacobi-fortran-targetdata-markers" _JACOBI_FORTRAN_TARGETDATA_MARKERS_DISABLED)
check_rocprofsys_disable_examples("roctx" _ROCTX_DISABLED)
check_rocprofsys_disable_examples("openmp" _OPENMP_DISABLED)
if(_JACOBI_FORTRAN_TARGETDATA_MARKERS_DISABLED OR _ROCTX_DISABLED OR _OPENMP_DISABLED)
    return()
endif()

if(NOT amdflang_EXECUTABLE)
    rocprofiler_systems_message(AUTHOR_WARNING "amdflang not found, this is required for jacobi-fortran-targetdata-markers example")
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
    list(APPEND offload_arch_flags --offload-arch=${arch})
    list(APPEND offload_link_flags --offload-arch=${arch})
endforeach()

set(CMAKE_Fortran_COMPILER ${amdflang_EXECUTABLE})
enable_language(Fortran)

find_library(
    ROCTX_LIBRARY
    NAMES rocprofiler-sdk-roctx
    PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64
)
if(NOT ROCTX_LIBRARY)
    rocprofiler_systems_message(AUTHOR_WARNING "rocprofiler-sdk-roctx not found, this is required for jacobi-fortran-targetdata-markers example")
    return()
endif()

# Search for AMD specific HIPFORT module
set(HIPFORT_MODULE_PATH_HINTS
    "${ROCM_PATH}/include/hipfort/amdgcn"
    "${ROCmVersion_DIR}/include/hipfort/amdgcn"
    "${ROCM_ROOT_DIR}/include/hipfort/amdgcn"
    "$ENV{ROCM_PATH}/include/hipfort/amdgcn"
    "/opt/rocm/include/hipfort/amdgcn"
)
find_path(HIPFORT_MODULE_PATH NAMES hipfort.mod HINTS ${HIPFORT_MODULE_PATH_HINTS})
mark_as_advanced(HIPFORT_MODULE_PATH)
if(NOT HIPFORT_MODULE_PATH)
    rocprofiler_systems_message(AUTHOR_WARNING "hipfort.mod not found, this is required for jacobi-fortran-targetdata-markers example")
    return()
endif()

set(HPC_EXAMPLE_NAME "jacobi-fortran-targetdata-markers")
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
target_include_directories(${HPC_EXAMPLE_NAME} PRIVATE ${HIPFORT_MODULE_PATH})
target_link_options(${HPC_EXAMPLE_NAME} PRIVATE ${offload_link_flags})
target_link_libraries(${HPC_EXAMPLE_NAME} PRIVATE ${ROCTX_LIBRARY})

rocprofiler_systems_install_hpc_example(${HPC_EXAMPLE_NAME})
