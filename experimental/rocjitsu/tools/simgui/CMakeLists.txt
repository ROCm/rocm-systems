# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

add_executable(simgui
    main.cpp
    app.cpp
    topology_editor.cpp
    simulation_panel.cpp
)

target_include_directories(simgui PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(simgui PRIVATE
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
    rocjitsu_isa_risc_v
    simdojo
    rocjitsu_vm
    rocjitsu_vm_amdgpu
    rocjitsu_vm_risc_v
    rocjitsu_config
    util
    flatbuffers
    Threads::Threads
    imgui_lib
    imnodes_lib
    implot_lib
)

rj_configure_target(simgui TOOL)
