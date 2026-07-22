# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

set(RJ_TOOL_OBJECT_LIBS
    rocjitsu_analysis
    rocjitsu_code
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
    # The DBT tool only needs generated AMDGPU decoders/disassemblers, but
    # those object files currently also contain instruction execute methods.
    # Until generated ISA decode/disassembly is split from VM execution, the
    # linker needs the AMDGPU VM support referenced by those execute methods.
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

add_executable(rj_dbt_translate dbt_translate_main.cpp)
target_link_libraries(rj_dbt_translate PRIVATE rocjitsu_tooling)
rj_configure_target(rj_dbt_translate TOOL)

add_executable(rj_waitcheck waitcheck_main.cpp)
target_link_libraries(rj_waitcheck PRIVATE rocjitsu_tooling)
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
