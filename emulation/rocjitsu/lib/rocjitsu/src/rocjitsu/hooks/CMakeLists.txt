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
    # Configure a hotswap-hook shared library. `_export_map` selects the version
    # script and `_test_hooks` decides whether the gated rj_test_* entry points are
    # compiled in. Factored into a function so the SHIPPED target and the TEST-ONLY
    # target stay identical except for those two axes -- the production target's ABI
    # and compile definitions must be invariant across BUILD_TESTING.
    function(rj_add_hotswap_hook _target _export_map _test_hooks)
        add_library(${_target} SHARED gfx1250_b0_a0_hotswap.cpp)
        set_target_properties(${_target} PROPERTIES OUTPUT_NAME ${_target})
        target_link_libraries(
            ${_target}
            PRIVATE rocjitsu_gfx1250_b0_to_a0 Threads::Threads
        )
        target_include_directories(
            ${_target}
            PRIVATE ${ROCJITSU_INCLUDE_DIR} ${ROCJITSU_SRC_DIR}
        )
        target_include_directories(${_target} SYSTEM PRIVATE ${HSA_INCLUDE_DIR})
        rj_configure_target(${_target} WARNINGS HIDDEN)
        target_compile_options(
            ${_target}
            PRIVATE -ffunction-sections -fdata-sections
        )
        if(_test_hooks)
            target_compile_definitions(${_target} PRIVATE RJ_HOTSWAP_TEST_HOOKS)
        endif()
        set(_map ${CMAKE_CURRENT_SOURCE_DIR}/${_export_map})
        set_property(TARGET ${_target} APPEND PROPERTY LINK_DEPENDS ${_map})
        target_link_options(
            ${_target}
            PRIVATE "LINKER:--gc-sections" "LINKER:--version-script=${_map}"
        )
    endfunction()

    # Production library: exports ONLY OnLoad/OnUnload, never compiles the rj_test_*
    # hooks, and is the one installed. Its ABI is identical regardless of
    # BUILD_TESTING.
    rj_add_hotswap_hook(hsa_hotswap_rocjitsu hsa_hotswap_rocjitsu.map OFF)

    # Test-only library: same code plus the gated rj_test_* entry points and a
    # version script that additionally exports rj_test_*. Built only when testing and
    # NEVER installed, so it cannot widen the shipped ABI. Unit tests link this.
    if(BUILD_TESTING)
        rj_add_hotswap_hook(
            hsa_hotswap_rocjitsu_testable
            hsa_hotswap_rocjitsu_testable.map
            ON
        )
    endif()
endif()
