# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_code
    amdgpu_code_object.cpp
    basic_block.cpp
    executable.cpp
    kernel_symbol.cpp
    rj_code.cpp
    dbt/binary_translator.cpp
    dbt/rj_code_translate.cpp
    dbt/hazard_tracker.cpp
    dbt/kernel_descriptor_translator.cpp
    dbt/lane_permutation.cpp
    dbt/semantic/cdna4_to_cdna3.cpp
    dbt/semantic/cdna4_to_rdna_common.cpp
    dbt/semantic/cdna4_to_rdna3.cpp
    dbt/semantic/cdna4_to_rdna4.cpp
    dbt/semantic_translator.cpp
    dbt/waitcnt_translator.cpp
    patch/code_object_patcher.cpp
    patch/spill_manager.cpp
)
