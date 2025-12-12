# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_isa_rdna2
    addr_calc.cpp
    decoder.cpp
    ds.cpp
    encodings.cpp
    exp.cpp
    flat.cpp
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
    vintrp.cpp
    vop1.cpp
    vop2.cpp
    vop3.cpp
    vop3p.cpp
    vopc.cpp
)
