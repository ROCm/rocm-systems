# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_isa_cdna2
    addr_calc.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/decoder.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/ds.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/encodings.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/flat.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/mimg.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/mtbuf.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/mubuf.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/operand.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/smem.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/sop1.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/sop2.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/sopc.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/sopk.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/sopp.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/vop1.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/vop2.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/vop3.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/vop3p.cpp
    ${RJ_AMDGPU_GENERATED_DIR}/cdna2/vopc.cpp
    target_provider.cpp
)

rj_add_isa_target_provider(
    rocjitsu_isa_cdna2
    HEADER rocjitsu/isa/arch/amdgpu/cdna2/target_provider.h
    BUILTIN
)
