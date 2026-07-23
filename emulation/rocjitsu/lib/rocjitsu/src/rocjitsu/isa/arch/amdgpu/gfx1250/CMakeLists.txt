# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

file(
    GLOB gfx1250_vop3_sources
    CONFIGURE_DEPENDS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "vop3.cpp"
    "vop3_*.cpp"
)
list(FILTER gfx1250_vop3_sources EXCLUDE REGEX "_exec")
list(SORT gfx1250_vop3_sources)

file(
    GLOB gfx1250_vopc_sources
    CONFIGURE_DEPENDS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "vopc.cpp"
    "vopc_*.cpp"
)
list(FILTER gfx1250_vopc_sources EXCLUDE REGEX "_exec")
list(SORT gfx1250_vopc_sources)

file(
    GLOB gfx1250_execution_sources
    CONFIGURE_DEPENDS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*_exec*.cpp"
)
list(SORT gfx1250_execution_sources)

# Model objects are compiled once and shared by decoder-only frontends and the
# full simulator. Execution bodies live in distinct translation units so the
# boundary does not depend on preprocessor-selected type definitions.
rj_add_object_library(rocjitsu_isa_gfx1250_model
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

rj_add_object_library(rocjitsu_isa_gfx1250_exec
    addr_calc.cpp
    ${gfx1250_execution_sources}
)

# Preserve the canonical full-ISA target name while making the two object
# sets explicit. Consumers of this interface receive one copy of the model
# objects and one copy of the execution objects. Keep the execution set as
# TARGET_OBJECTS: its file-scope registrars are link side effects that an
# ordinary archive link could discard.
add_library(rocjitsu_isa_gfx1250 INTERFACE)
target_sources(
    rocjitsu_isa_gfx1250
    INTERFACE
        $<TARGET_OBJECTS:rocjitsu_isa_gfx1250_model>
        $<TARGET_OBJECTS:rocjitsu_isa_gfx1250_exec>
)
