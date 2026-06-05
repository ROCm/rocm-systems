# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

add_subdirectory(gfx1250_semantics)

set(RJ_TRANSLATE_OBJECT_LIBS
    rocjitsu_analysis
    rocjitsu_code
    rocjitsu_isa
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
    rocjitsu_isa_risc_v
    simdojo
    rocjitsu_vm
    rocjitsu_vm_amdgpu
    rocjitsu_vm_risc_v
    rocjitsu_config
)

add_library(rocjitsu_tooling STATIC dbt_translate.cpp)
target_include_directories(rocjitsu_tooling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rocjitsu_tooling PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(
    rocjitsu_tooling
    PUBLIC ${RJ_TRANSLATE_OBJECT_LIBS} util flatbuffers Threads::Threads
)
rj_configure_target(rocjitsu_tooling TOOL)

add_executable(rj_dbt_translate dbt_translate_main.cpp)
target_link_libraries(rj_dbt_translate PRIVATE rocjitsu_tooling)
rj_configure_target(rj_dbt_translate TOOL)

add_executable(
    rocjitsu-translate-code-object
    rocjitsu_translate_code_object.cpp
)
target_link_libraries(
    rocjitsu-translate-code-object
    PRIVATE
        ${RJ_TRANSLATE_OBJECT_LIBS}
        rocjitsu_kmd
        util
        flatbuffers
        Threads::Threads
)
rj_configure_target(rocjitsu-translate-code-object TOOL)

add_executable(rocjitsu-translate-vmfb rocjitsu_translate_vmfb.cpp)
target_link_libraries(
    rocjitsu-translate-vmfb
    PRIVATE
        ${RJ_TRANSLATE_OBJECT_LIBS}
        rocjitsu_kmd
        util
        flatbuffers
        Threads::Threads
)
rj_configure_target(rocjitsu-translate-vmfb TOOL)
