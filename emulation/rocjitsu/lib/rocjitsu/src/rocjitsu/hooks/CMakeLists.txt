# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    rocjitsu_hooks
    SHARED
    rj_hsa_dbt_hooks.cpp
    sidecar_registry.cpp
    virtual_lds.cpp
)

add_library(rocjitsu_comgr SHARED rj_comgr_hotswap.cpp)
set_target_properties(
    rocjitsu_comgr
    PROPERTIES OUTPUT_NAME rocjitsu_comgr VERSION 0.1.0 SOVERSION 0
)
target_include_directories(
    rocjitsu_comgr
    PRIVATE ${ROCJITSU_INCLUDE_DIR} ${ROCJITSU_SRC_DIR} ${GENERATED_DIR}
)
target_link_libraries(rocjitsu_comgr PRIVATE rocjitsu_dbt_internal)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(rocjitsu_comgr PRIVATE "LINKER:-z,defs")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_library(rocjitsu_hotswap_select SHARED rj_hotswap_dlopen_select.cpp)
    set_target_properties(
        rocjitsu_hotswap_select
        PROPERTIES OUTPUT_NAME rocjitsu_hotswap_select
    )
    target_include_directories(
        rocjitsu_hotswap_select
        PRIVATE ${ROCJITSU_INCLUDE_DIR}
    )
    target_link_libraries(rocjitsu_hotswap_select PRIVATE ${CMAKE_DL_LIBS})
endif()

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
        rocjitsu_isa_cdna1
        rocjitsu_isa_cdna2
        rocjitsu_isa_cdna3
        rocjitsu_isa_cdna4
        rocjitsu_isa_rdna1
        rocjitsu_isa_rdna2
        rocjitsu_isa_rdna3
        rocjitsu_isa_rdna3_5
        rocjitsu_isa_rdna4
        rocjitsu_isa_gfx1250
        # \NPI new ISA family: link its rocjitsu_isa_<isa> object library here.
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

if(MSVC)
    target_compile_options(rocjitsu_comgr PRIVATE /W4 /WX)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(
        rocjitsu_comgr
        PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
    if(TARGET rocjitsu_hotswap_select)
        target_compile_options(
            rocjitsu_hotswap_select
            PRIVATE -Wall -Wextra -Wpedantic -Werror
        )
    endif()
endif()
