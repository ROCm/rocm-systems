# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The production target graph follows the semantic direction of the lowering
# pipeline. Each component may use the components named before it; architecture
# boundary checks forbid the important reverse dependencies.
rj_add_object_library(
    rocjitsu_consan_contracts
        consan_access_classifier.cpp
        consan_atomic_classifier.cpp
        consan_access_policy.cpp
        consan_atomic_fence_policy.cpp
        consan_barrier_policy.cpp
        consan_input_layout.cpp
        consan_instruction_semantics.cpp
        consan_inventory_diagnostics.cpp
        consan_observation_policy.cpp
        consan_perturbation_policy.cpp
        consan_semantic_classifiers.cpp
        consan_sync_event_index.cpp
        consan_sync_metadata.cpp
        consan_types.cpp
)

rj_add_object_library(
    rocjitsu_consan_targets
        consan_fault_gfx12_target_ops.cpp
        consan_fault_gfx9_target_ops.cpp
        consan_fault_target_ops.cpp
        consan_gfx1250_lds_target_ops.cpp
        consan_gfx1250_vgpr_bank_state.cpp
        consan_moi_gfx9_target_ops.cpp
        consan_program_analysis_gfx9_cdna_target_ops.cpp
        consan_program_analysis_gfx1100_target_ops.cpp
        consan_program_analysis_gfx1201_target_ops.cpp
        consan_program_analysis_gfx1250_target_ops.cpp
        consan_program_analysis_target_ops.cpp
        consan_relay_target_ops.cpp
        consan_supercollider_gfx11_gfx12_target_ops.cpp
        consan_supercollider_gfx1250_target_ops.cpp
        consan_supercollider_gfx9_target_ops.cpp
        consan_supercollider_rdna3_target_ops.cpp
        consan_supercollider_rdna4_target_ops.cpp
        consan_supercollider_target_ops.cpp
        consan_validation_gfx12_target_ops.cpp
        consan_validation_gfx9_cdna_target_ops.cpp
        consan_validation_target_ops.cpp
)
target_link_libraries(rocjitsu_consan_targets PRIVATE rocjitsu_consan_contracts)

rj_add_object_library(
    rocjitsu_consan_analysis
        consan_fault_selection.cpp
        consan_program_analysis.cpp
        consan_sync_analysis.cpp
)
target_link_libraries(
    rocjitsu_consan_analysis
    PRIVATE rocjitsu_consan_targets
)

rj_add_object_library(
    rocjitsu_consan_transform
        consan_barrier_move_proof.cpp
        consan_branch_only_relay_router.cpp
        consan_descriptor_growth.cpp
        consan_fault_injection.cpp
        consan_moi.cpp
        consan_moi_access_apply.cpp
        consan_moi_access_target.cpp
        consan_moi_barrier.cpp
        consan_moi_candidate_projection.cpp
        consan_moi_common_emission.cpp
        consan_moi_dynamic_record_emission.cpp
        consan_moi_engine_contracts.cpp
        consan_moi_evidence_planning.cpp
        consan_moi_exact_shadow_emission.cpp
        consan_moi_inline_shadow_emission.cpp
        consan_moi_inline_shadow.cpp
        consan_moi_inline_shadow_report_plan.cpp
        consan_moi_memory_emission.cpp
        consan_moi_mode_planning.cpp
        consan_moi_model.cpp
        consan_moi_native_abi.cpp
        consan_moi_pipeline.cpp
        consan_moi_placement.cpp
        consan_moi_probe_contracts.cpp
        consan_moi_probe_planning.cpp
        consan_moi_prologue.cpp
        consan_moi_report_plan.cpp
        consan_moi_relocation.cpp
        consan_moi_report_emission.cpp
        consan_moi_record_event_emission.cpp
        consan_moi_record_planning.cpp
        consan_moi_record_replay.cpp
        consan_moi_record_replay_model.cpp
        consan_moi_record_replay_report_plan.cpp
        consan_moi_runtime_workgroup_gate.cpp
        consan_moi_sampled_access_emission.cpp
        consan_moi_sampled_contracts.cpp
        consan_moi_sampled.cpp
        consan_moi_sampled_model.cpp
        consan_moi_sampled_report_plan.cpp
        consan_moi_sampled_atomic_emission.cpp
        consan_moi_sampled_window_emission.cpp
        consan_moi_shared_lowering.cpp
        consan_moi_support.cpp
        consan_moi_sync_emission.cpp
        consan_moi_target_address.cpp
        consan_perturbation.cpp
        consan_placement.cpp
        consan_resource.cpp
        consan_supercollider.cpp
        consan_supercollider_report_plan.cpp
        consan_supercollider_support.cpp
)
target_link_libraries(
    rocjitsu_consan_transform
    PRIVATE rocjitsu_consan_analysis
)

rj_add_object_library(
    rocjitsu_consan_validation
        consan_final_validation.cpp
        consan_validation_inventory.cpp
)
target_link_libraries(
    rocjitsu_consan_validation
    PRIVATE rocjitsu_consan_analysis rocjitsu_consan_transform
)

rj_add_object_library(
    rocjitsu_consan_orchestration
        consan.cpp
        consan_composition.cpp
        consan_pipeline.cpp
)
target_link_libraries(
    rocjitsu_consan_orchestration
    PRIVATE rocjitsu_consan_transform rocjitsu_consan_validation
)

set(
    ROCJITSU_CONSAN_OBJECT_LIBS
    rocjitsu_consan_contracts
    rocjitsu_consan_targets
    rocjitsu_consan_analysis
    rocjitsu_consan_transform
    rocjitsu_consan_validation
    rocjitsu_consan_orchestration
)
set(ROCJITSU_CONSAN_OBJECT_LIBS ${ROCJITSU_CONSAN_OBJECT_LIBS} PARENT_SCOPE)

# Make source ownership exhaustive and exclusive. The retained lowerer file is
# an intentionally inert tombstone and is the only production .cpp outside the
# component graph.
set(_consan_component_sources)
foreach(_component IN LISTS ROCJITSU_CONSAN_OBJECT_LIBS)
    get_target_property(_component_sources ${_component} SOURCES)
    list(APPEND _consan_component_sources ${_component_sources})
endforeach()
set(_consan_unique_component_sources ${_consan_component_sources})
list(REMOVE_DUPLICATES _consan_unique_component_sources)
list(LENGTH _consan_component_sources _consan_component_source_count)
list(LENGTH _consan_unique_component_sources _consan_unique_component_source_count)
if(NOT _consan_component_source_count EQUAL _consan_unique_component_source_count)
    message(FATAL_ERROR "a ConSan source belongs to more than one component")
endif()
file(GLOB _consan_production_sources RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")
list(REMOVE_ITEM _consan_production_sources consan_lowerer.cpp)
list(SORT _consan_production_sources)
list(SORT _consan_unique_component_sources)
if(NOT _consan_production_sources STREQUAL _consan_unique_component_sources)
    message(FATAL_ERROR "the ConSan component graph does not own every production source")
endif()
