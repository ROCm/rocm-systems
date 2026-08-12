# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(gfx1250_generated_dir "${RJ_AMDGPU_GENERATED_DIR}/gfx1250")

file(
    GLOB gfx1250_vop3_sources
    CONFIGURE_DEPENDS
    "${gfx1250_generated_dir}/vop3.cpp"
    "${gfx1250_generated_dir}/vop3_*.cpp"
)
list(FILTER gfx1250_vop3_sources EXCLUDE REGEX "_exec")
list(SORT gfx1250_vop3_sources)

file(
    GLOB gfx1250_vopc_sources
    CONFIGURE_DEPENDS
    "${gfx1250_generated_dir}/vopc.cpp"
    "${gfx1250_generated_dir}/vopc_*.cpp"
)
list(FILTER gfx1250_vopc_sources EXCLUDE REGEX "_exec")
list(SORT gfx1250_vopc_sources)

file(
    GLOB gfx1250_execution_sources
    CONFIGURE_DEPENDS
    "${gfx1250_generated_dir}/*_exec*.cpp"
)
list(SORT gfx1250_execution_sources)

# Model objects are compiled once and shared by decoder-only frontends and the
# full simulator. Execution bodies live in distinct translation units so the
# boundary does not depend on preprocessor-selected type definitions.
rj_add_object_library(rocjitsu_isa_gfx1250_model
    ${gfx1250_generated_dir}/decoder.cpp
    ${gfx1250_generated_dir}/encodings.cpp
    ${gfx1250_generated_dir}/operand.cpp
    ${gfx1250_generated_dir}/smem.cpp
    ${gfx1250_generated_dir}/sop1.cpp
    ${gfx1250_generated_dir}/sop2.cpp
    ${gfx1250_generated_dir}/sopc.cpp
    ${gfx1250_generated_dir}/sopk.cpp
    ${gfx1250_generated_dir}/sopp.cpp
    ${gfx1250_generated_dir}/vbuffer.cpp
    ${gfx1250_generated_dir}/vds.cpp
    ${gfx1250_generated_dir}/vflat.cpp
    ${gfx1250_generated_dir}/vglobal.cpp
    ${gfx1250_generated_dir}/vimage.cpp
    ${gfx1250_generated_dir}/vop1.cpp
    ${gfx1250_generated_dir}/vop2.cpp
    ${gfx1250_vop3_sources}
    ${gfx1250_generated_dir}/vop3p.cpp
    ${gfx1250_generated_dir}/vopd.cpp
    ${gfx1250_vopc_sources}
    ${gfx1250_generated_dir}/vscratch.cpp
    model_target_provider.cpp
)

rj_add_object_library(rocjitsu_isa_gfx1250_exec
    addr_calc.cpp
    ${gfx1250_execution_sources}
    target_provider.cpp
)

# Preserve the canonical full-ISA target name while making the two object
# sets explicit. Consumers of this interface receive one copy of the model
# objects and one copy of the execution objects. Keep the execution set as
# TARGET_OBJECTS so consumers receive both generated object sets without
# recompiling either source set.
add_library(rocjitsu_isa_gfx1250 INTERFACE)
target_sources(
    rocjitsu_isa_gfx1250
    INTERFACE
        $<TARGET_OBJECTS:rocjitsu_isa_gfx1250_model>
        $<TARGET_OBJECTS:rocjitsu_isa_gfx1250_exec>
)
target_link_libraries(
    rocjitsu_isa_gfx1250
    INTERFACE rocjitsu_isa_gfx1250_model rocjitsu_isa_gfx1250_exec
)

rj_add_isa_target_provider(
    rocjitsu_isa_gfx1250
    HEADER rocjitsu/isa/arch/amdgpu/gfx1250/target_provider.h
    BUILTIN
)

# Decoder/patching consumers can select the model provider independently of
# the simulator's full provider. It contributes the same target identity but
# has no execution capability or dependency on execution objects.
rj_add_isa_target_provider(
    rocjitsu_isa_gfx1250_model
    HEADER rocjitsu/isa/arch/amdgpu/gfx1250/model_target_provider.h
)
