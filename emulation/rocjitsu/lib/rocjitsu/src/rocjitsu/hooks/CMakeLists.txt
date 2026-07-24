# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    rocjitsu_hooks
    SHARED
    rj_hsa_dbt_hooks.cpp
    sidecar_registry.cpp
    virtual_lds.cpp
)

set_target_properties(rocjitsu_hooks PROPERTIES OUTPUT_NAME rocjitsu_hooks)

target_include_directories(
    rocjitsu_hooks
    PRIVATE
        ${ROCJITSU_INCLUDE_DIR}
        ${ROCJITSU_SRC_DIR}
        ${GENERATED_DIR}
        ${RJ_VERSION_INCLUDE_DIR}
)

target_include_directories(rocjitsu_hooks SYSTEM PRIVATE ${HSA_INCLUDE_DIR})

target_link_libraries(
    rocjitsu_hooks
    PRIVATE
        rocjitsu_analysis
        rocjitsu_code
        rocjitsu_dbt_config
        rocjitsu_isa
        rocjitsu_amdgpu_isa_registry
        rocjitsu_vm_amdgpu
        util
        flatbuffers
        simdojo
        Threads::Threads
        ${CMAKE_DL_LIBS}
)

if(MSVC)
    target_compile_options(rocjitsu_hooks PRIVATE /W4 /WX)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(
        rocjitsu_hooks
        PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
endif()
