# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ConSan is built into rocjitsu_code, but its rapidly evolving implementation
# is registered here rather than growing the shared code manifest.
target_sources(
    rocjitsu_code
    PRIVATE
        consan.cpp
        consan_access_classifier.cpp
        consan_atomic_classifier.cpp
        consan_access_policy.cpp
        consan_atomic_fence_policy.cpp
        consan_barrier_policy.cpp
        consan_barrier_move_proof.cpp
        consan_branch_only_relay_router.cpp
        consan_descriptor_growth.cpp
        consan_fault_selection.cpp
        consan_final_validation.cpp
        consan_input_layout.cpp
        consan_instruction_semantics.cpp
        consan_lowerer.cpp
        consan_moi.cpp
        consan_moi_candidate_projection.cpp
        consan_moi_engine_contracts.cpp
        consan_moi_memory_emission.cpp
        consan_moi_model.cpp
        consan_moi_report_plan.cpp
        consan_moi_relocation.cpp
        consan_moi_support.cpp
        consan_moi_target_address.cpp
        consan_observation_policy.cpp
        consan_perturbation_policy.cpp
        consan_pipeline.cpp
        consan_program_analysis.cpp
        consan_resource.cpp
        consan_relay_target_ops.cpp
        consan_semantic_classifiers.cpp
        consan_sync_event_index.cpp
        consan_sync_metadata.cpp
        consan_types.cpp
)
