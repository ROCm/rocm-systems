# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Device-side model of a PCI-attached GPU: the bus surface a guest driver sees,
# and the diagnostics for the registers it touches. Portable and free of any VMM
# dependency, so every transport under vmm/ shares it and it is built and tested
# in every configuration.
rj_add_object_library(
    rocjitsu_pci
    register_symbols.cpp
    bar_access_trace.cpp
    scratch_pci_device.cpp
)
