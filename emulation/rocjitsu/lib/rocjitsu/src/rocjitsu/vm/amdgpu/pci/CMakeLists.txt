# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Device-side model of a PCI-attached GPU: the bus surface a guest driver sees,
# and the diagnostics for the registers it touches. Free of any VMM dependency,
# so every transport under vmm/ shares it.
set(RJ_PCI_SOURCES
    register_symbols.cpp
    bar_access_trace.cpp
    scratch_pci_device.cpp
    gpu_pci_device_spec.cpp
    ip_discovery.cpp
    ip_discovery_profile.cpp
)

# The GPU model shares its video memory with a guest through a file descriptor,
# which is a POSIX shared-memory facility with no portable equivalent, so it
# builds only where a transport could use it. Everything else here is portable.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND RJ_PCI_SOURCES gpu_pci_device.cpp)
endif()

rj_add_object_library(rocjitsu_pci ${RJ_PCI_SOURCES})
