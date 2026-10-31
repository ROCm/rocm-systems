# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The hazard record format and its writer. The race plugin bundles these too.
rj_add_object_library(rocjitsu_hazard_record
    hazard_record.cpp
)

# Per-dispatch hazard sites and the core memory-wait report.
rj_add_object_library(rocjitsu_vm_diagnostics
    hazard_sites.cpp
    memory_wait_report.cpp
)
