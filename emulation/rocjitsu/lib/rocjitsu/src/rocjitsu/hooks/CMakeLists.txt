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

# Dedicated eager-only HSA hotswap hook. Keep this separate from the general
# DBT hook: its only translation dependency is the fixed-profile gfx1250
# B0-to-A0 library.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_library(hsa_hotswap_rocjitsu SHARED gfx1250_b0_a0_hotswap.cpp)
    set_target_properties(
        hsa_hotswap_rocjitsu
        PROPERTIES OUTPUT_NAME hsa_hotswap_rocjitsu
    )
    target_link_libraries(
        hsa_hotswap_rocjitsu
        PRIVATE rocjitsu_gfx1250_b0_to_a0 Threads::Threads
    )
    target_include_directories(
        hsa_hotswap_rocjitsu
        PRIVATE ${ROCJITSU_INCLUDE_DIR} ${ROCJITSU_SRC_DIR}
    )
    target_include_directories(
        hsa_hotswap_rocjitsu
        SYSTEM
        PRIVATE ${HSA_INCLUDE_DIR}
    )
    rj_configure_target(
        hsa_hotswap_rocjitsu
        WARNINGS
        HIDDEN
    )
    target_compile_options(
        hsa_hotswap_rocjitsu
        PRIVATE -ffunction-sections -fdata-sections
    )
    # Shipped DSO exports only OnLoad/OnUnload. A testing build additionally
    # exports rj_test_* (via the testable version script) and compiles the gated
    # rj_test_* entry points, so unit tests can observe internal lifecycle state
    # without widening the shipped ABI. This DSO is never installed.
    if(BUILD_TESTING)
        set(_hsa_hotswap_exports
            ${CMAKE_CURRENT_SOURCE_DIR}/hsa_hotswap_rocjitsu_testable.map
        )
        target_compile_definitions(
            hsa_hotswap_rocjitsu
            PRIVATE RJ_HOTSWAP_TEST_HOOKS
        )
    else()
        set(_hsa_hotswap_exports
            ${CMAKE_CURRENT_SOURCE_DIR}/hsa_hotswap_rocjitsu.map
        )
    endif()
    set_property(
        TARGET hsa_hotswap_rocjitsu
        APPEND
        PROPERTY LINK_DEPENDS ${_hsa_hotswap_exports}
    )
    target_link_options(
        hsa_hotswap_rocjitsu
        PRIVATE
            "LINKER:--gc-sections"
            "LINKER:--version-script=${_hsa_hotswap_exports}"
    )
endif()
