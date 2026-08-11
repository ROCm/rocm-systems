# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_isa_cdna3
    addr_calc.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/decoder.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/encodings.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/ds.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/flat.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/mtbuf.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/mubuf.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/operand.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/smem.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/sop1.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/sop2.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/sopc.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/sopk.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/sopp.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/vop1.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/vop2.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/vop3.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/vop3p.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna3/vopc.cpp
    target_provider.cpp
)

rj_add_isa_target_provider(
    rocjitsu_isa_cdna3
    HEADER rocjitsu/isa/arch/amdgpu/cdna3/target_provider.h
    BUILTIN
)
