# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Keep ConSan's DBI hook isolated from the generic rocjitsu hook target. The
# output directory intentionally remains the parent hooks directory for
# compatibility with installed layouts and existing validation tooling.
add_library(
    rocjitsu_dbi_hooks
    SHARED
    rj_hsa_dbi_hook_config.cpp
    rj_hsa_dbi_hook_moi_report.cpp
    rj_hsa_dbi_hooks.cpp
)
set_target_properties(
    rocjitsu_dbi_hooks
    PROPERTIES
        OUTPUT_NAME rocjitsu_dbi_hooks
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.."
)

target_include_directories(
    rocjitsu_dbi_hooks
    PRIVATE ${ROCJITSU_INCLUDE_DIR} ${ROCJITSU_SRC_DIR}
)
target_include_directories(rocjitsu_dbi_hooks SYSTEM PRIVATE ${HSA_INCLUDE_DIR})

target_link_libraries(
    rocjitsu_dbi_hooks
    PRIVATE
        rocjitsu_waitcheck_objects
        rocjitsu_dbt_internal
        util
        Threads::Threads
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_rj_consan_linker_options "--gc-sections,--exclude-libs,ALL")
    if(
        NOT
        (
            RJ_ENABLE_ASAN
            OR RJ_ENABLE_UBSAN
            OR RJ_ENABLE_TSAN
            OR RJ_ENABLE_MSAN
        )
    )
        string(APPEND _rj_consan_linker_options ",--no-undefined")
    endif()
    target_link_options(
        rocjitsu_dbi_hooks
        PRIVATE "LINKER:${_rj_consan_linker_options}"
    )
    unset(_rj_consan_linker_options)
endif()

if(MSVC)
    target_compile_options(rocjitsu_dbi_hooks PRIVATE /W4 /WX)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(
        rocjitsu_dbi_hooks
        PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
endif()
