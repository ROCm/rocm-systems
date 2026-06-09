# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

file(
    GLOB gfx1250_vop3_sources
    CONFIGURE_DEPENDS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "vop3_part*.cpp"
)
list(SORT gfx1250_vop3_sources)

file(
    GLOB gfx1250_vopc_sources
    CONFIGURE_DEPENDS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "vopc_part*.cpp"
)
list(SORT gfx1250_vopc_sources)

rj_add_object_library(rocjitsu_isa_gfx1250
    addr_calc.cpp
    decoder.cpp
    encodings.cpp
    operand.cpp
    smem.cpp
    sop1.cpp
    sop2.cpp
    sopc.cpp
    sopk.cpp
    sopp.cpp
    vbuffer.cpp
    vds.cpp
    vflat.cpp
    vglobal.cpp
    vimage.cpp
    vop1.cpp
    vop2.cpp
    ${gfx1250_vop3_sources}
    vop3p.cpp
    vopd.cpp
    ${gfx1250_vopc_sources}
    vscratch.cpp
)
