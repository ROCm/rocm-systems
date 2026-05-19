# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

set(RJ_TOOL_OBJECT_LIBS
    rocjitsu_analysis
    rocjitsu_code
    rocjitsu_isa
    rocjitsu_isa_cdna1
    rocjitsu_isa_cdna2
    rocjitsu_isa_cdna3
    rocjitsu_isa_cdna4
    rocjitsu_isa_rdna1
    rocjitsu_isa_rdna2
    rocjitsu_isa_rdna3
    rocjitsu_isa_rdna3_5
    rocjitsu_isa_rdna4
    # The DBT tool only needs generated AMDGPU decoders/disassemblers, but
    # those object files currently also contain instruction execute methods.
    # Until generated ISA decode/disassembly is split from VM execution, the
    # linker needs the AMDGPU VM support referenced by those execute methods.
    simdojo
    rocjitsu_vm_amdgpu
)

find_library(ROCM_HSA_RUNTIME64 hsa-runtime64
    PATHS /opt/rocm/lib ENV ROCM_PATH
    PATH_SUFFIXES lib)
find_path(ROCM_HSA_INCLUDE_DIR hsa/hsa_ext_amd.h
    PATHS /opt/rocm/include ENV ROCM_PATH
    PATH_SUFFIXES include)

add_library(rocjitsu_tooling STATIC
    dbt_translate.cpp
)
target_include_directories(rocjitsu_tooling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rocjitsu_tooling PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(rocjitsu_tooling PUBLIC
    ${RJ_TOOL_OBJECT_LIBS}
    util
    flatbuffers
    Threads::Threads
)
rj_configure_target(rocjitsu_tooling TOOL)

add_library(rocjitsu_hsa_tooling STATIC
    hsa_run_kernel.cpp
)
target_include_directories(rocjitsu_hsa_tooling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rocjitsu_hsa_tooling PUBLIC ${CMAKE_SOURCE_DIR})
if(ROCM_HSA_RUNTIME64 AND ROCM_HSA_INCLUDE_DIR)
    get_filename_component(ROCM_HSA_LIBRARY_DIR "${ROCM_HSA_RUNTIME64}" DIRECTORY)
    target_compile_definitions(rocjitsu_hsa_tooling PRIVATE RJ_HAS_HSA_RUNTIME=1)
    target_include_directories(rocjitsu_hsa_tooling PRIVATE ${ROCM_HSA_INCLUDE_DIR})
    target_link_libraries(rocjitsu_hsa_tooling PUBLIC ${ROCM_HSA_RUNTIME64})
endif()
rj_configure_target(rocjitsu_hsa_tooling TOOL)

add_executable(rj_dbt_translate dbt_translate_main.cpp)
target_link_libraries(rj_dbt_translate PRIVATE rocjitsu_tooling)
rj_configure_target(rj_dbt_translate TOOL)

if(ROCM_HSA_RUNTIME64 AND ROCM_HSA_INCLUDE_DIR)
    add_executable(rj_hsa_run hsa_run_kernel_main.cpp)
    target_link_libraries(rj_hsa_run PRIVATE rocjitsu_hsa_tooling)
    target_include_directories(rj_hsa_run PRIVATE ${ROCM_HSA_INCLUDE_DIR})
    target_link_options(rj_hsa_run PRIVATE -Wl,-rpath,${ROCM_HSA_LIBRARY_DIR})
    rj_configure_target(rj_hsa_run TOOL)
    message(STATUS "rj_hsa_run enabled (found ROCm HSA runtime)")
else()
    message(STATUS "rj_hsa_run disabled (ROCm HSA runtime or headers not found)")
endif()
