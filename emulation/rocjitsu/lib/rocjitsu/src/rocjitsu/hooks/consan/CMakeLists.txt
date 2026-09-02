# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Keep ConSan's DBI hook isolated from the generic rocjitsu hook target. The
# output directory intentionally remains the parent hooks directory for
# compatibility with installed layouts and existing validation tooling.
add_library(rocjitsu_consan_report_trust STATIC rj_hsa_dbi_moi_report_trust.cpp)
set_target_properties(
    rocjitsu_consan_report_trust
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
    rocjitsu_consan_report_trust
    PUBLIC ${ROCJITSU_SRC_DIR}
)

add_library(rocjitsu_consan_report_snapshot STATIC rj_hsa_dbi_moi_report_snapshot.cpp)
set_target_properties(
    rocjitsu_consan_report_snapshot
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
    rocjitsu_consan_report_snapshot
    PUBLIC ${ROCJITSU_INCLUDE_DIR} ${ROCJITSU_SRC_DIR}
)
target_link_libraries(rocjitsu_consan_report_snapshot PRIVATE util)

add_library(rocjitsu_consan_report_decoder STATIC rj_hsa_dbi_moi_report_decoder.cpp)
set_target_properties(
    rocjitsu_consan_report_decoder
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
    rocjitsu_consan_report_decoder
    PUBLIC ${ROCJITSU_INCLUDE_DIR} ${ROCJITSU_SRC_DIR}
)
target_link_libraries(rocjitsu_consan_report_decoder PRIVATE util)

add_library(
    rocjitsu_consan_report_analyzer
    STATIC
        rj_hsa_dbi_moi_record_replay_report_analyzer.cpp
        rj_hsa_dbi_moi_report_analyzer.cpp
        rj_hsa_dbi_moi_sampled_report_analyzer.cpp
)
set_target_properties(
    rocjitsu_consan_report_analyzer
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
    rocjitsu_consan_report_analyzer
    PUBLIC ${ROCJITSU_INCLUDE_DIR} ${ROCJITSU_SRC_DIR}
)
target_link_libraries(rocjitsu_consan_report_analyzer PRIVATE util)

add_library(rocjitsu_consan_report_renderer STATIC rj_hsa_dbi_moi_report_renderer.cpp)
set_target_properties(
    rocjitsu_consan_report_renderer
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
    rocjitsu_consan_report_renderer
    PUBLIC ${ROCJITSU_INCLUDE_DIR} ${ROCJITSU_SRC_DIR}
)
target_link_libraries(
    rocjitsu_consan_report_renderer
    PRIVATE rocjitsu_consan_report_analyzer util
)

add_library(
    rocjitsu_dbi_hooks
    SHARED
    rj_hsa_dbi_hook_config.cpp
    rj_hsa_dbi_hook_moi_report.cpp
    rj_hsa_dbi_moi_report_pipeline.cpp
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
        rocjitsu_hsa_tool_lifetime
        rocjitsu_consan_report_analyzer
        rocjitsu_consan_report_decoder
        rocjitsu_consan_report_renderer
        rocjitsu_consan_report_snapshot
        rocjitsu_consan_report_trust
        rocjitsu_waitcheck_objects
        rocjitsu_dbt_internal
        util
        Threads::Threads
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_rj_consan_linker_options "--gc-sections,--exclude-libs,ALL")
    if(
        NOT (RJ_ENABLE_ASAN OR RJ_ENABLE_UBSAN OR RJ_ENABLE_TSAN OR RJ_ENABLE_MSAN)
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
    target_compile_options(rocjitsu_consan_report_analyzer PRIVATE /W4 /WX)
    target_compile_options(rocjitsu_consan_report_decoder PRIVATE /W4 /WX)
    target_compile_options(rocjitsu_consan_report_renderer PRIVATE /W4 /WX)
    target_compile_options(rocjitsu_consan_report_snapshot PRIVATE /W4 /WX)
    target_compile_options(rocjitsu_consan_report_trust PRIVATE /W4 /WX)
    target_compile_options(rocjitsu_dbi_hooks PRIVATE /W4 /WX)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(
        rocjitsu_consan_report_analyzer
        PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
    target_compile_options(
        rocjitsu_consan_report_decoder
        PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
    target_compile_options(
        rocjitsu_consan_report_renderer
        PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
    target_compile_options(
        rocjitsu_consan_report_snapshot
        PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
    target_compile_options(
        rocjitsu_consan_report_trust
        PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
    target_compile_options(
        rocjitsu_dbi_hooks
        PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
endif()
