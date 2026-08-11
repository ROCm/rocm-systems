# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_isa_rdna2
    addr_calc.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/decoder.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/ds.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/encodings.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/exp.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/flat.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/mimg.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/mtbuf.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/mubuf.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/operand.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/smem.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/sop1.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/sop2.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/sopc.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/sopk.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/sopp.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/vintrp.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/vop1.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/vop2.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/vop3.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/vop3p.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna2/vopc.cpp
    target_provider.cpp
)

rj_add_isa_target_provider(
    rocjitsu_isa_rdna2
    HEADER rocjitsu/isa/arch/amdgpu/rdna2/target_provider.h
    BUILTIN
)
