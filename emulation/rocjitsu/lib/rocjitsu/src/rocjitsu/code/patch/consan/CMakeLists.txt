# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ConSan is built into rocjitsu_code, but its rapidly evolving implementation
# is registered here rather than growing the shared code manifest.
target_sources(
    rocjitsu_code
    PRIVATE
        consan.cpp
        consan_branch_only_relay_router.cpp
        consan_moi.cpp
        consan_moi_model.cpp
        consan_moi_report_plan.cpp
        consan_resource.cpp
        consan_types.cpp
)
