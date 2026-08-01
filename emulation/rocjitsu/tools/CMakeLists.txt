# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

set(RJ_TOOL_OBJECT_LIBS
    rocjitsu_analysis
    rocjitsu_code
    rocjitsu_isa
    rocjitsu_amdgpu_isa_registry
    simdojo
    rocjitsu_vm_amdgpu
)

add_library(rocjitsu_tooling STATIC dbt_translate.cpp)
target_include_directories(rocjitsu_tooling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rocjitsu_tooling PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(
    rocjitsu_tooling
    PUBLIC ${RJ_TOOL_OBJECT_LIBS} util flatbuffers Threads::Threads
)
rj_configure_target(rocjitsu_tooling TOOL)

add_library(rocjitsu_translate_cli STATIC dbt_translate_cli.cpp)
target_include_directories(
    rocjitsu_translate_cli
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(rocjitsu_translate_cli PUBLIC rocjitsu_tooling)
rj_configure_target(rocjitsu_translate_cli TOOL)

add_executable(rj_dbt_translate rj_dbt_translate_main.cpp)
target_link_libraries(rj_dbt_translate PRIVATE rocjitsu_translate_cli)
rj_configure_target(rj_dbt_translate TOOL)

add_executable(rj_waitcheck waitcheck_main.cpp)
target_sources(
    rj_waitcheck
    PRIVATE ${ROCJITSU_SRC_DIR}/rocjitsu/code/executable.cpp
)
target_link_libraries(
    rj_waitcheck
    PRIVATE
        rocjitsu_waitcheck_objects
        util
        Threads::Threads
        ZLIB::ZLIB
        zstd::libzstd
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(
        rj_waitcheck
        PRIVATE "LINKER:--gc-sections,--exclude-libs,ALL"
    )
endif()
rj_configure_target(rj_waitcheck TOOL)

add_executable(rj_co code_object_tool_main.cpp)
target_link_libraries(rj_co PRIVATE rocjitsu_tooling)
rj_configure_target(rj_co TOOL)

add_custom_target(
    rj_waitcheck_gfx950_tensile_e2e
    COMMAND
        ${CMAKE_SOURCE_DIR}/scripts/run_waitcheck_gfx950_tensile_e2e.sh
        --waitcheck $<TARGET_FILE:rj_waitcheck> --work-dir
        ${CMAKE_BINARY_DIR}/waitcheck-e2e/tensile-gfx950
    DEPENDS rj_waitcheck
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    USES_TERMINAL
    VERBATIM
    COMMENT "Build and waitcheck representative gfx950 Tensile kernels"
)
