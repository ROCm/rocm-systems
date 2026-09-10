# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The production target graph follows the semantic direction of the lowering
# pipeline. Each component may use the components named before it.
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
        targets/consan_fault_target_ops.cpp
        targets/consan_program_analysis_target_ops.cpp
        targets/consan_supercollider_target_ops.cpp
        targets/consan_target_profiles.cpp
        targets/consan_validation_target_ops.cpp
        targets/rdna3/consan_program_analysis_rdna3_target_ops.cpp
        targets/rdna3/consan_supercollider_rdna3_target_ops.cpp
        targets/rdna4/consan_program_analysis_rdna4_target_ops.cpp
        targets/rdna4/consan_supercollider_rdna4_target_ops.cpp
        targets/cdna5/consan_cdna5_lds_target_ops.cpp
        targets/cdna5/consan_cdna5_vgpr_bank_state.cpp
        targets/cdna5/consan_program_analysis_cdna5_target_ops.cpp
        targets/cdna5/consan_supercollider_cdna5_target_ops.cpp
        targets/shared/cdna3_cdna4/consan_fault_cdna3_cdna4_target_ops.cpp
        targets/shared/cdna3_cdna4/consan_moi_cdna3_cdna4_target_ops.cpp
        targets/shared/cdna3_cdna4/consan_program_analysis_cdna3_cdna4_target_ops.cpp
        targets/shared/cdna3_cdna4/consan_supercollider_cdna3_cdna4_target_ops.cpp
        targets/shared/cdna3_cdna4/consan_validation_cdna3_cdna4_target_ops.cpp
        targets/shared/consan_fault_rdna4_cdna5_target_ops.cpp
        targets/shared/consan_supercollider_common_target_ops.cpp
        targets/shared/consan_validation_rdna4_cdna5_target_ops.cpp
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
        consan_descriptor_growth.cpp
        consan_fault_injection.cpp
        consan_moi.cpp
        consan_moi_access_apply.cpp
        consan_moi_access_target.cpp
        consan_moi_barrier.cpp
        consan_moi_candidate_projection.cpp
        consan_moi_dynamic_record_emission.cpp
        consan_moi_engine_contracts.cpp
        consan_moi_evidence_planning.cpp
        consan_moi_exact_shadow_emission.cpp
        modes/inline_shadow/consan_moi_inline_shadow_emission.cpp
        modes/inline_shadow/consan_moi_inline_atomic_emission.cpp
        modes/inline_shadow/consan_moi_inline_shadow.cpp
        modes/inline_shadow/consan_moi_inline_shadow_report_plan.cpp
        consan_moi_memory_emission.cpp
        consan_moi_mode_planning.cpp
        consan_moi_native_abi.cpp
        consan_moi_pipeline.cpp
        consan_moi_placement.cpp
        consan_moi_probe_contracts.cpp
        consan_moi_probe_planning.cpp
        consan_moi_prologue.cpp
        consan_moi_report_plan.cpp
        consan_moi_relocation.cpp
        consan_moi_report_emission.cpp
        modes/record_replay/consan_moi_record_event_emission.cpp
        modes/record_replay/consan_moi_record_planning.cpp
        modes/record_replay/consan_moi_record_replay.cpp
        modes/record_replay/consan_moi_record_replay_access_emission.cpp
        modes/record_replay/consan_moi_record_replay_model.cpp
        modes/record_replay/consan_moi_record_replay_report_plan.cpp
        consan_moi_runtime_workgroup_gate.cpp
        modes/sampled/consan_moi_sampled_access_emission.cpp
        modes/sampled/consan_moi_sampled_contracts.cpp
        modes/sampled/consan_moi_sampled.cpp
        modes/sampled/consan_moi_sampled_model.cpp
        modes/sampled/consan_moi_sampled_report_plan.cpp
        modes/sampled/consan_moi_sampled_atomic_emission.cpp
        modes/sampled/consan_moi_sampled_window_emission.cpp
        consan_moi_shared_lowering.cpp
        consan_moi_support.cpp
        consan_moi_sync_emission.cpp
        consan_moi_target_address.cpp
        consan_perturbation.cpp
        consan_placement.cpp
        consan_resource.cpp
        consan_text_relocation.cpp
        modes/supercollider/consan_supercollider.cpp
        modes/supercollider/consan_supercollider_report_plan.cpp
        modes/supercollider/consan_supercollider_support.cpp
        consan_unmatched_barrier_abort.cpp
)
target_link_libraries(
    rocjitsu_consan_transform
    PRIVATE rocjitsu_consan_analysis
)

rj_add_object_library(
    rocjitsu_consan_validation
        consan_final_validation.cpp
        consan_validation_inventory.cpp
        modes/inline_shadow/consan_moi_inline_shadow_final_validation.cpp
        modes/supercollider/consan_supercollider_final_validation.cpp
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
