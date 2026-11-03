# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

# Keep this opt-in: linker support alone does not imply that the deployment
# machine's dynamic loader understands DT_RELR.
option(
    ROCJITSU_PACK_RELATIVE_RELOCS
    "Pack relative relocations; requires glibc 2.36+ on deployment machines"
    OFF
)

if(ROCJITSU_PACK_RELATIVE_RELOCS)
    if(
        NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"
        OR NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"
    )
        message(
            FATAL_ERROR
            "ROCJITSU_PACK_RELATIVE_RELOCS requires Linux and GCC/Clang"
        )
    endif()

    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles(
        "#include <features.h>
         #if !defined(__GLIBC__) || !defined(__GLIBC_PREREQ)
         #error glibc required
         #elif !__GLIBC_PREREQ(2, 36)
         #error glibc 2.36 required
         #endif
         int main() { return 0; }"
        RJ_HAVE_RELR_GLIBC
    )
    if(NOT RJ_HAVE_RELR_GLIBC)
        message(
            FATAL_ERROR
            "ROCJITSU_PACK_RELATIVE_RELOCS requires glibc 2.36+ target headers"
        )
    endif()

    include(CheckLinkerFlag)
    check_linker_flag(
        CXX
        "LINKER:-z,pack-relative-relocs"
        RJ_HAVE_PACK_RELATIVE_RELOCS
    )
    if(NOT RJ_HAVE_PACK_RELATIVE_RELOCS)
        message(
            FATAL_ERROR
            "The linker does not support -z pack-relative-relocs"
        )
    endif()
    message(
        STATUS
        "Rocjitsu relative relocations: packed (deployment requires glibc 2.36+)"
    )
endif()

function(rj_pack_relative_relocations target)
    get_target_property(target_type ${target} TYPE)
    if(
        ROCJITSU_PACK_RELATIVE_RELOCS
        AND target_type MATCHES "^(EXECUTABLE|SHARED_LIBRARY|MODULE_LIBRARY)$"
    )
        target_link_options(${target} PRIVATE "LINKER:-z,pack-relative-relocs")
    endif()
endfunction()
