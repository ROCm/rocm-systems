# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# vfio-user front-end for rocjitsu — split into two libraries:
#
#   rocjitsu_gpu_pci_device  — transport-neutral GPU device model.
#                              No libvfio-user dependency; buildable without
#                              RJ_ENABLE_VFIO_USER.
#
#   rocjitsu_vfio_host       — VfioDeviceHost and legacy VfuServer.
#                              Links libvfio-user PRIVATE so the dependency
#                              does not propagate to consumers.
#                              Only built when RJ_ENABLE_VFIO_USER=ON.
#
# rocjitsu_vfu is kept as an alias for rocjitsu_vfio_host for backward
# compatibility with the rocjitsu-vfu tool CMakeLists.

# ---------------------------------------------------------------------------
# rocjitsu_gpu_pci_device — no libvfio-user
# ---------------------------------------------------------------------------
add_library(
    rocjitsu_gpu_pci_device STATIC
    bar0_vram.cpp
    bar2_doorbell.cpp
    bar5_mmio.cpp
    ip_discovery_blob.cpp
    gpu_pci_device.cpp
    gpu_pci_device_provider.cpp
)

target_include_directories(
    rocjitsu_gpu_pci_device
    PUBLIC
        ${ROCJITSU_INCLUDE_DIR}
        ${ROCJITSU_SRC_DIR}
        ${GENERATED_DIR}
        ${PROJECT_SOURCE_DIR}/lib/simdojo/include
)

target_include_directories(rocjitsu_gpu_pci_device SYSTEM PUBLIC ${HSA_INCLUDE_DIR})

target_link_libraries(
    rocjitsu_gpu_pci_device
    PRIVATE
        rocjitsu_kmd
        util
        Threads::Threads
)

target_compile_features(rocjitsu_gpu_pci_device PUBLIC cxx_std_20)

# ---------------------------------------------------------------------------
# rocjitsu_vfio_host — libvfio-user linked PRIVATE
# (only built when the vfio-user transport is enabled)
# ---------------------------------------------------------------------------
add_library(
    rocjitsu_vfio_host STATIC
    pci_config.cpp
    dma_mapper.cpp
    vfu_server.cpp
)

target_include_directories(
    rocjitsu_vfio_host
    PUBLIC
        ${ROCJITSU_INCLUDE_DIR}
        ${ROCJITSU_SRC_DIR}
        ${GENERATED_DIR}
        ${PROJECT_SOURCE_DIR}/lib/simdojo/include
)

target_include_directories(rocjitsu_vfio_host SYSTEM PUBLIC ${HSA_INCLUDE_DIR})

target_link_libraries(
    rocjitsu_vfio_host
    PRIVATE
        PkgConfig::VFIO_USER      # libvfio-user is PRIVATE — not propagated
        rocjitsu_gpu_pci_device
        rocjitsu_kmd
        rocjitsu_vm
        rocjitsu_vm_amdgpu
        util
        Threads::Threads
)

target_compile_features(rocjitsu_vfio_host PUBLIC cxx_std_20)

# VfioDeviceHost lives in kmd/linux/ but its .cpp links the same libs.
# Add it to rocjitsu_vfio_host rather than creating a third target.
target_sources(
    rocjitsu_vfio_host
    PRIVATE
        ${ROCJITSU_SRC_DIR}/rocjitsu/kmd/linux/vfio_device_host.cpp
)

# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------
add_library(rocjitsu_vfu ALIAS rocjitsu_vfio_host)
