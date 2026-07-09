# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

add_executable(rocjitsu_bin main.cpp)

target_include_directories(
    rocjitsu_bin
    PRIVATE ${GENERATED_DIR} ${RJ_VERSION_INCLUDE_DIR}
)

target_link_libraries(
    rocjitsu_bin
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
        # \NPI new ISA family: link its rocjitsu_isa_<isa> object library here.
        rocjitsu_isa_risc_v
        simdojo
        rocjitsu_vm
        rocjitsu_vm_amdgpu
        rocjitsu_vm_risc_v
        rocjitsu_config
        util
        flatbuffers
        Threads::Threads
        dl
)

set_target_properties(rocjitsu_bin PROPERTIES OUTPUT_NAME rocjitsu)

rj_configure_target(rocjitsu_bin TOOL)

# Install the rocjitsu CLI so it is usable from a packaged dist (the default
# entrypoint for driving emulation when mirage is not present).
include(GNUInstallDirs)
install(TARGETS rocjitsu_bin RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
