# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_isa_rdna3_5
    addr_calc.cpp
    decoder.cpp
    ds.cpp
    encodings.cpp
    exp.cpp
    flat.cpp
    ldsdir.cpp
    mimg.cpp
    mtbuf.cpp
    mubuf.cpp
    operand.cpp
    smem.cpp
    sop1.cpp
    sop2.cpp
    sopc.cpp
    sopk.cpp
    sopp.cpp
    vinterp.cpp
    vop1.cpp
    vop2.cpp
    vop3.cpp
    vop3p.cpp
    vopd.cpp
    vopc.cpp
    target_provider.cpp
)

rj_add_isa_target_provider(
    rocjitsu_isa_rdna3_5
    HEADER rocjitsu/isa/arch/amdgpu/rdna3_5/target_provider.h
    BUILTIN
)
