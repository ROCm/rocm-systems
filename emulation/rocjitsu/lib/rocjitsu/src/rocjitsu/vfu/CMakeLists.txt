# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# rocjitsu_vfu: libvfio-user server library for the rocjitsu GPU emulator.
# Built only when RJ_ENABLE_VFIO_USER=ON (requires libvfio-user installed).

add_library(
    rocjitsu_vfu STATIC
    pci_config.cpp
    bar0_vram.cpp
    bar2_doorbell.cpp
    bar5_mmio.cpp
    dma_mapper.cpp
    ip_discovery_blob.cpp
    vfu_server.cpp
)

target_include_directories(
    rocjitsu_vfu
    PUBLIC
        ${ROCJITSU_INCLUDE_DIR}
        ${ROCJITSU_SRC_DIR}
        ${GENERATED_DIR}
        ${PROJECT_SOURCE_DIR}/lib/simdojo/include
)

target_include_directories(rocjitsu_vfu SYSTEM PUBLIC ${HSA_INCLUDE_DIR})

target_link_libraries(
    rocjitsu_vfu
    PUBLIC
        PkgConfig::VFIO_USER
    PRIVATE
        rocjitsu_kmd
        rocjitsu_vm
        rocjitsu_vm_amdgpu
        util
        Threads::Threads
)

target_compile_features(rocjitsu_vfu PUBLIC cxx_std_20)
