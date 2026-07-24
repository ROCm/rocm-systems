# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Apply the common sources, includes, link libraries, and warning flags shared by
# every rocjitsu_hooks variant. Keeps the shipped and test-instrumented builds in
# lock-step so they differ only by the test-hook compile definition.
function(rj_configure_hooks_target target)
    target_sources(
        ${target}
        PRIVATE rj_hsa_dbt_hooks.cpp sidecar_registry.cpp virtual_lds.cpp
    )

    target_include_directories(
        ${target}
        PRIVATE
            ${ROCJITSU_INCLUDE_DIR}
            ${ROCJITSU_SRC_DIR}
            ${GENERATED_DIR}
            ${RJ_VERSION_INCLUDE_DIR}
    )
    target_include_directories(${target} SYSTEM PRIVATE ${HSA_INCLUDE_DIR})

    target_link_libraries(
        ${target}
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
        target_compile_options(${target} PRIVATE /W4 /WX)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(
            ${target}
            PRIVATE -Wall -Wextra -Wpedantic -Werror
        )
    endif()
endfunction()

# Shipped library: NEVER carries the rj_test_* entry points, so a release build's
# public ABI is clean regardless of BUILD_TESTING. This is the only variant the
# top-level install rule installs.
add_library(rocjitsu_hooks SHARED)
set_target_properties(rocjitsu_hooks PROPERTIES OUTPUT_NAME rocjitsu_hooks)
rj_configure_hooks_target(rocjitsu_hooks)

# Test-only instrumented copy: identical build plus RJ_HOOK_TEST_HOOKS, which
# exports the rj_test_* symbols the unit tests link against. It is NOT installed,
# so the test ABI never ships. The tests link this target instead of the shipped
# one (see tests/CMakeLists.txt).
if(BUILD_TESTING)
    add_library(rocjitsu_hooks_testable SHARED)
    set_target_properties(
        rocjitsu_hooks_testable
        PROPERTIES OUTPUT_NAME rocjitsu_hooks_testable
    )
    rj_configure_hooks_target(rocjitsu_hooks_testable)
    target_compile_definitions(
        rocjitsu_hooks_testable
        PRIVATE RJ_HOOK_TEST_HOOKS=1
    )
endif()
