# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

add_executable(dbt_translate
    main.cpp
)

target_link_libraries(dbt_translate PRIVATE
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
    util
    rocjitsu_config
    rocjitsu_vm
    rocjitsu_vm_amdgpu
    simdojo
    flatbuffers
    Threads::Threads
)

rj_configure_target(dbt_translate TOOL)
