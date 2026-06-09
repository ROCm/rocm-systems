# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_kmd
    events.cpp
    remote_driver.cpp
    simulated_driver.cpp
    sysfs.cpp
    transport.cpp
)
target_include_directories(rocjitsu_kmd PRIVATE ${GENERATED_DIR})

add_library(rocjitsu_kmd_shim SHARED interposer.cpp)
target_include_directories(
    rocjitsu_kmd_shim
    PRIVATE
        ${ROCJITSU_INCLUDE_DIR}
        ${ROCJITSU_SRC_DIR}
        ${HSA_INCLUDE_DIR}
        ${GENERATED_DIR}
)
target_link_libraries(
    rocjitsu_kmd_shim
    PRIVATE
        rocjitsu_kmd
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
        rocjitsu_plugin_logging
        rocjitsu_plugin_race_detector
        rocjitsu_vm_risc_v
        rocjitsu_config
        util
        flatbuffers
        Threads::Threads
        dl
)
target_compile_definitions(rocjitsu_kmd_shim PRIVATE _GNU_SOURCE)
target_compile_options(
    rocjitsu_kmd_shim
    PRIVATE -Wall -Wextra -Wpedantic -Werror
)
set_target_properties(rocjitsu_kmd_shim PROPERTIES OUTPUT_NAME "rocjitsu_kmd")
