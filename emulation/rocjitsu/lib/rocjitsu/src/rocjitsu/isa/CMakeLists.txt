# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_isa
    decoder.cpp
    operand.cpp
    register_set.cpp
    rj_decode.cpp
)

add_subdirectory(arch/amdgpu/cdna1)
add_subdirectory(arch/amdgpu/cdna2)
add_subdirectory(arch/amdgpu/cdna3)
add_subdirectory(arch/amdgpu/cdna4)
add_subdirectory(arch/amdgpu/rdna1)
add_subdirectory(arch/amdgpu/rdna2)
add_subdirectory(arch/amdgpu/rdna3)
add_subdirectory(arch/amdgpu/rdna3_5)
add_subdirectory(arch/amdgpu/rdna4)
add_subdirectory(arch/amdgpu/gfx1250)
# \NPI new ISA family: create arch/amdgpu/<isa>/CMakeLists.txt \
# (rj_add_object_library(rocjitsu_isa_<isa> ...)) and add_subdirectory() it here.
add_subdirectory(arch/risc_v)
