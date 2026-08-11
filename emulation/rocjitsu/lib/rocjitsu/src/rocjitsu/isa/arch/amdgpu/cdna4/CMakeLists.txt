# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_isa_cdna4
    addr_calc.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/decoder.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/encodings.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/ds.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/flat.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/mtbuf.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/mubuf.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/operand.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/smem.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/sop1.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/sop2.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/sopc.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/sopk.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/sopp.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/vop1.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/vop2.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/vop3.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/vop3p.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna4/vopc.cpp
    target_provider.cpp
)

rj_add_isa_target_provider(
    rocjitsu_isa_cdna4
    HEADER rocjitsu/isa/arch/amdgpu/cdna4/target_provider.h
    BUILTIN
)
