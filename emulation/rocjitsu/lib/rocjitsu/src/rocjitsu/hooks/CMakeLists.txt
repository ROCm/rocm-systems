# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The hook sources are compiled once into an object library and linked into two
# shared libraries: the production rocjitsu_hooks, which exports only the HSA
# tool entry points, and the test-only rocjitsu_hooks_testing, which adds the
# exported C test seams (see rj_hsa_dbt_test_seams.cpp).
#
# Both shared libraries must name these dependencies themselves: CMake does not
# propagate an object library's object files transitively through another object
# library, so linking them only to rocjitsu_hooks_objects would leave the ISA and
# VM objects out of the shared libraries.
set(RJ_HOOKS_LINK_LIBS
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

add_library(
    rocjitsu_hooks_objects
    OBJECT
    rj_hsa_dbt_hooks.cpp
    sidecar_registry.cpp
    virtual_lds.cpp
)

set_target_properties(
    rocjitsu_hooks_objects
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)

target_include_directories(
    rocjitsu_hooks_objects
    PRIVATE
        ${ROCJITSU_INCLUDE_DIR}
        ${ROCJITSU_SRC_DIR}
        ${GENERATED_DIR}
        ${RJ_VERSION_INCLUDE_DIR}
)

target_include_directories(
    rocjitsu_hooks_objects
    SYSTEM
    PRIVATE ${HSA_INCLUDE_DIR}
)

target_link_libraries(rocjitsu_hooks_objects PRIVATE ${RJ_HOOKS_LINK_LIBS})

if(MSVC)
    target_compile_options(rocjitsu_hooks_objects PRIVATE /W4 /WX)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(
        rocjitsu_hooks_objects
        PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
endif()

# Production hook: exports the HSA tool entry points only.
add_library(rocjitsu_hooks SHARED)
set_target_properties(rocjitsu_hooks PROPERTIES OUTPUT_NAME rocjitsu_hooks)
target_link_libraries(
    rocjitsu_hooks
    PRIVATE rocjitsu_hooks_objects ${RJ_HOOKS_LINK_LIBS}
)

if(BUILD_TESTING)
    # Test-only hook: the same objects plus the exported test seams. Never
    # installed; only hook unit tests link it.
    add_library(rocjitsu_hooks_testing SHARED rj_hsa_dbt_test_seams.cpp)
    set_target_properties(
        rocjitsu_hooks_testing
        PROPERTIES OUTPUT_NAME rocjitsu_hooks_testing
    )
    target_include_directories(
        rocjitsu_hooks_testing
        PRIVATE ${ROCJITSU_SRC_DIR}
    )
    target_link_libraries(
        rocjitsu_hooks_testing
        PRIVATE rocjitsu_hooks_objects ${RJ_HOOKS_LINK_LIBS}
    )

    if(MSVC)
        target_compile_options(rocjitsu_hooks_testing PRIVATE /W4 /WX)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(
            rocjitsu_hooks_testing
            PRIVATE -Wall -Wextra -Wpedantic -Werror
        )
    endif()
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
        # The translation store lives in the DBT layer and is compiled in rather
        # than linked, because this DSO deliberately depends on nothing but the
        # fixed-profile translator. Another consumer that wants reuse adds the
        # same source to its own target; nothing about it is specific to this
        # hook, and the two would keep separate trees.
        add_library(
            ${_target}
            SHARED
            gfx1250_b0_a0_hotswap.cpp
            ${ROCJITSU_SRC_DIR}/rocjitsu/code/dbt/translation_store.cpp
        )
        set_target_properties(${_target} PROPERTIES OUTPUT_NAME ${_target})
        target_link_libraries(
            ${_target}
            PRIVATE rocjitsu_gfx1250_b0_to_a0 Threads::Threads ${CMAKE_DL_LIBS}
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
            target_compile_definitions(
                ${_target}
                PRIVATE RJ_HOTSWAP_TEST_HOOKS RJ_TRANSLATION_STORE_TEST_HOOKS
            )
        endif()
        set(_map ${CMAKE_CURRENT_SOURCE_DIR}/${_export_map})
        set_property(TARGET ${_target} APPEND PROPERTY LINK_DEPENDS ${_map})
        # --build-id is not decoration here: the translation store keys entries on
        # the TRANSLATOR's note, and the same flag is what gives that shared
        # object one. Without it the store finds no identity and disables itself.
        target_link_options(
            ${_target}
            PRIVATE
                "LINKER:--gc-sections"
                "LINKER:--version-script=${_map}"
                "LINKER:--build-id=sha1"
        )
        # The translator is a shared object installed beside this one. Resolving
        # it relative to our own location keeps an install relocatable and, more
        # importantly, keeps every consumer bound to the same copy: the cache
        # identity IS that object's build id, so two consumers resolving to
        # different copies would silently stop sharing entries.
        set_target_properties(
            ${_target}
            PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN"
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
