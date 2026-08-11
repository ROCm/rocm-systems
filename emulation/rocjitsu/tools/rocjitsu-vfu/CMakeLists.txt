# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

add_executable(rocjitsu_vfu_bin main.cpp)

target_include_directories(
    rocjitsu_vfu_bin
    PRIVATE
        ${ROCJITSU_INCLUDE_DIR}
        ${ROCJITSU_SRC_DIR}
        ${GENERATED_DIR}
)

target_link_libraries(
    rocjitsu_vfu_bin
    PRIVATE
        rocjitsu_vfio_host  # vfio-user transport (libvfio-user is PRIVATE inside)
        rocjitsu_gpu_pci_device
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
        rocjitsu_isa_registry
        rocjitsu_plugin_loader
        rocjitsu_vm_amdgpu
        rocjitsu_vm_risc_v
        rocjitsu_config
        rocjitsu_dbt_config
        util
        flatbuffers
        Threads::Threads
        dl
)

set_target_properties(rocjitsu_vfu_bin PROPERTIES OUTPUT_NAME rocjitsu-vfu)

rj_configure_target(rocjitsu_vfu_bin TOOL)

include(GNUInstallDirs)
install(TARGETS rocjitsu_vfu_bin RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
