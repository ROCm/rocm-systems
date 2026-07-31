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
    rocjitsu_isa_registry
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
