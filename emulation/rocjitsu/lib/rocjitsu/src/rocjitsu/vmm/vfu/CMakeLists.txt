# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The libvfio-user backend: serves a PCI function over the vfio-user protocol so
# a VMM can present it to its guest. Linux-only, and the only unit that links
# libvfio-user.
rj_add_object_library(
    rocjitsu_vmm_vfu
    bus_plan.cpp
    guest_region_map.cpp
    vfio_device_host.cpp
    vfio_server.cpp
)
target_link_libraries(
    rocjitsu_vmm_vfu
    PRIVATE libvfio-user::libvfio-user flatbuffers
)
# The server reads the simulation config, whose parser is generated.
target_include_directories(rocjitsu_vmm_vfu PRIVATE ${GENERATED_DIR})
add_dependencies(rocjitsu_vmm_vfu flatbuffers_schemas)
