# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_isa_rdna1
    addr_calc.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/decoder.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/ds.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/encodings.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/exp.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/flat.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/mimg.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/mtbuf.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/mubuf.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/operand.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/smem.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/sop1.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/sop2.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/sopc.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/sopk.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/sopp.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/vintrp.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/vop1.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/vop2.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/vop3.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/vop3p.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna1/vopc.cpp
    target_provider.cpp
)

rj_add_isa_target_provider(
    rocjitsu_isa_rdna1
    HEADER rocjitsu/isa/arch/amdgpu/rdna1/target_provider.h
    BUILTIN
)
