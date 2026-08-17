# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_isa_rdna3
    addr_calc.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/decoder.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/ds.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/encodings.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/exp.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/flat.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/ldsdir.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/mimg.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/mtbuf.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/mubuf.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/operand.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/smem.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/sop1.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/sop2.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/sopc.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/sopk.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/sopp.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/vinterp.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/vop1.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/vop2.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/vop3.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/vop3p.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/vopd.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna3/vopc.cpp
    target_provider.cpp
)

rj_add_isa_target_provider(
    rocjitsu_isa_rdna3
    HEADER rocjitsu/isa/arch/amdgpu/rdna3/target_provider.h
    BUILTIN
)
