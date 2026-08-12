# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_isa_rdna4
    addr_calc.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/decoder.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/encodings.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/operand.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/smem.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/sop1.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/sop2.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/sopc.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/sopk.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/sopp.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/vbuffer.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/vds.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/vdsdir.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/vexport.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/vflat.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/vglobal.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/vimage.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/vinterp.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/vop1.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/vop2.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/vop3.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/vop3p.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/vopd.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/vopc.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/vsample.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/rdna4/vscratch.cpp
    target_provider.cpp
)

rj_add_isa_target_provider(
    rocjitsu_isa_rdna4
    HEADER rocjitsu/isa/arch/amdgpu/rdna4/target_provider.h
    BUILTIN
)
