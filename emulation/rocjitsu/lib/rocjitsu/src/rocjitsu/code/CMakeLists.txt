# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Leaf helpers are shared by the transformation stack and native ConSan stages.
# Neither helper requires an ISA registry, compiler implementation, or VM.
add_library(rocjitsu_kernel_descriptor_scan STATIC kernel_descriptor_scan.cpp)
add_library(
    rocjitsu_kernarg_extension
    STATIC
    patch/kernarg_extension.cpp
    patch/kernarg_extension_metadata.cpp
)
foreach(_helper rocjitsu_kernel_descriptor_scan rocjitsu_kernarg_extension)
    target_include_directories(
        ${_helper}
        PUBLIC ${ROCJITSU_INCLUDE_DIR} ${ROCJITSU_SRC_DIR}
    )
    target_link_libraries(${_helper} PUBLIC util)
    set_target_properties(${_helper} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    rj_configure_target(${_helper} PRIVATE_INCLUDES WARNINGS HIDDEN)
endforeach()

# Code-object storage and inspection are shared by offline analyses and
# transformation frontends. Keep DBT, DBI, and executable loading out of this
# source group so offline consumers can link it independently.
rj_add_object_library(rocjitsu_code_object
    amdgpu_code_object.cpp
    file_io.cpp
    kernel_symbol.cpp
)
target_link_libraries(
    rocjitsu_code_object
    PUBLIC rocjitsu_kernel_descriptor_scan
)

# Executable loading is reusable by fuzz and runtime consumers, but remains
# outside the offline-analysis archive's symbol boundary.
rj_add_object_library(rocjitsu_executable
    executable.cpp
)

# CFG construction and dataflow share indirect-target discovery. Keep them in
# one object group; neither depends on the DBT/DBI transformation layer.
rj_add_object_library(rocjitsu_analysis
    basic_block.cpp
    kernel_scope.cpp
    relocation_function_table.cpp
    scoped_cfg_edges.cpp
    analysis/def_use_chain.cpp
    analysis/exec_state.cpp
    analysis/free_registers.cpp
    analysis/gfx1250_vgpr_msb.cpp
    analysis/indirect_branch_discovery.cpp
    analysis/liveness.cpp
)

rj_add_object_library(rocjitsu_code
    rj_code.cpp
    builders/instruction_builder.cpp
    dbt/binary_translator.cpp
    dbt/legalization/gfx1250_b0_to_a0.cpp
    dbt/rj_code_translate.cpp
    dbt/hazard_tracker.cpp
    dbt/kernel_descriptor_translator.cpp
    dbt/lane_permutation.cpp
    dbt/lds_virtualization.cpp
    dbt/semantic/cdna4_to_cdna3_virtual_lds.cpp
    dbt/semantic/cdna3_lds.cpp
    dbt/semantic/cdna3_scratch.cpp
    dbt/semantic/cdna4_to_cdna3.cpp
    dbt/semantic/cdna4_to_rdna_common.cpp
    dbt/semantic/cdna4_to_rdna3.cpp
    dbt/semantic/cdna4_to_rdna4.cpp
    dbt/semantic/gfx1250_b0_to_a0.cpp
    dbt/semantic/gfx1250_flat_scratch_base.cpp
    dbt/semantic_translator.cpp
    dbt/semantic_scratch.cpp
    dbt/virtual_lds.cpp
    patch/sidecar_metadata.cpp
    dbt/waitcnt_translator.cpp
    patch/code_object_patcher.cpp
    patch/entry_prologue.cpp
    patch/kernel_text_layout.cpp
    patch/log_buffer.cpp
    patch/instrumentor.cpp
    patch/probe_callable.cpp
    patch/probe_clobber.cpp
    patch/probe_live_in.cpp
    patch/probe_symbol.cpp
    patch/spill_manager.cpp
    patch/trampoline_builder.cpp
)
target_link_libraries(rocjitsu_code PUBLIC rocjitsu_kernarg_extension)
