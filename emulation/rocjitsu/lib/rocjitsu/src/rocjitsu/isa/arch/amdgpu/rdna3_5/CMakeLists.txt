# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_isa_rdna3_5
    addr_calc.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/decoder.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/ds.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/encodings.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/exp.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/flat.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/ldsdir.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/mimg.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/mtbuf.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/mubuf.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/operand.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/smem.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/sop1.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/sop2.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/sopc.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/sopk.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/sopp.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/vinterp.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/vop1.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/vop2.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/vop3.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/vop3p.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/vopd.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3_5/vopc.cpp
    target_provider.cpp
)

rj_add_isa_target_provider(
    rocjitsu_isa_rdna3_5
    HEADER rocjitsu/isa/arch/amdgpu/rdna3_5/target_provider.h
    BUILTIN
)
