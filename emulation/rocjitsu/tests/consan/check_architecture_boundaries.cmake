# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED ROCJITSU_SOURCE_DIR)
    message(FATAL_ERROR "ROCJITSU_SOURCE_DIR is required")
endif()

set(
    _consan_dir
    "${ROCJITSU_SOURCE_DIR}/lib/rocjitsu/src/rocjitsu/code/patch/consan"
)
set(
    _hook_dir
    "${ROCJITSU_SOURCE_DIR}/lib/rocjitsu/src/rocjitsu/hooks/consan"
)

function(_consan_assert_no_match file regex rule)
    if(NOT EXISTS "${file}")
        message(FATAL_ERROR "ConSan boundary check is missing ${file}")
    endif()
    file(STRINGS "${file}" _matches REGEX "${regex}")
    if(_matches)
        list(GET _matches 0 _first)
        string(STRIP "${_first}" _first)
        message(
            FATAL_ERROR
            "ConSan boundary violation (${rule}) in ${file}: ${_first}"
        )
    endif()
endfunction()

function(_consan_assert_match_count_at_most file regex maximum rule)
    if(NOT EXISTS "${file}")
        message(FATAL_ERROR "ConSan boundary check is missing ${file}")
    endif()
    file(READ "${file}" _contents)
    string(REGEX MATCHALL "${regex}" _matches "${_contents}")
    list(LENGTH _matches _count)
    if(_count GREATER maximum)
        message(
            FATAL_ERROR
            "ConSan boundary violation (${rule}) in ${file}: "
            "${_count} matches exceeds reviewed maximum ${maximum}"
        )
    endif()
endfunction()

# Semantic policy owns meaning, never an ISA recipe or product identity.
set(
    _semantic_policy_sources
    consan_access_policy.cpp
    consan_atomic_fence_policy.cpp
    consan_barrier_policy.cpp
    consan_observation_policy.cpp
    consan_perturbation_policy.cpp
)
foreach(_source IN LISTS _semantic_policy_sources)
    set(_file "${_consan_dir}/${_source}")
    _consan_assert_no_match(
        "${_file}"
        "isa/arch/amdgpu/generated/|ROCJITSU_CODE_ARCH_|(cdna[0-9_]*|rdna[0-9_]*)::"
        "semantic policy must be target-neutral"
    )
endforeach()

# Mode implementations may select typed target capabilities, but may not see
# product constants, generated ISA declarations, or member namespaces.
file(
    GLOB _mode_sources
    "${_consan_dir}/consan_moi_record*.cpp"
    "${_consan_dir}/consan_moi_record*.inc"
    "${_consan_dir}/consan_moi_sampled*.cpp"
    "${_consan_dir}/consan_moi_sampled*.inc"
    "${_consan_dir}/consan_moi_inline*.cpp"
    "${_consan_dir}/consan_moi_inline*.inc"
    "${_consan_dir}/consan_supercollider.cpp"
    "${_consan_dir}/consan_supercollider.inc"
    "${_consan_dir}/consan_supercollider_common.inc"
    "${_consan_dir}/consan_supercollider_flat.inc"
    "${_consan_dir}/consan_supercollider_lds.inc"
)
foreach(_file IN LISTS _mode_sources)
    if(_file MATCHES "target_ops" OR _file MATCHES "[.]h[.]inc$")
        continue()
    endif()
    _consan_assert_no_match(
        "${_file}"
        "isa/arch/amdgpu/generated/|ROCJITSU_CODE_ARCH_|(cdna[0-9_]*|rdna[0-9_]*)::"
        "mode implementation must consume typed target capabilities"
    )
endforeach()

# Generated ISA headers occur only in reviewed target normalization,
# family/member lowering, program analysis, or independent validation owners.
set(
    _generated_header_owners
    consan_access_classifier.cpp
    consan_fault_gfx12_target_ops.cpp
    consan_fault_gfx9_target_ops.cpp
    consan_final_validation.cpp
    consan_gfx1250_lds_target_ops.cpp
    consan_moi_gfx9_target_ops.cpp
    consan_program_analysis_gfx9_cdna_target_ops.cpp
    consan_program_analysis_gfx1100_target_ops.cpp
    consan_program_analysis_gfx1201_target_ops.cpp
    consan_program_analysis_gfx1250_target_ops.cpp
    consan_supercollider_gfx1250_target_ops.cpp
    consan_supercollider_gfx9_target_ops.cpp
    consan_supercollider_rdna3_target_ops.cpp
    consan_supercollider_rdna4_target_ops.cpp
)
file(
    GLOB _consan_sources
    "${_consan_dir}/*.cpp"
    "${_consan_dir}/*.h"
    "${_consan_dir}/*.inc"
)
foreach(_file IN LISTS _consan_sources)
    file(STRINGS "${_file}" _generated_includes REGEX "isa/arch/amdgpu/generated/")
    if(NOT _generated_includes)
        continue()
    endif()
    get_filename_component(_name "${_file}" NAME)
    list(FIND _generated_header_owners "${_name}" _owner_index)
    if(_owner_index EQUAL -1)
        message(
            FATAL_ERROR
            "ConSan boundary violation (unreviewed generated ISA dependency) in ${_file}"
        )
    endif()
endforeach()

# Program-analysis target operations normalize raw operands without acquiring
# mode policy. Concrete generated types remain behind the family/member
# implementations and their one narrow registry.
set(_program_analysis_target_contract
    "${_consan_dir}/consan_program_analysis_target_ops.h")
_consan_assert_no_match(
    "${_program_analysis_target_contract}"
    "ConSanMoiEngine|isa/arch/amdgpu/generated/|ROCJITSU_CODE_ARCH_|(cdna[0-9_]*|rdna[0-9_]*)::"
    "program-analysis target contract must be mode- and member-neutral"
)
file(READ "${_consan_dir}/consan_program_analysis.cpp" _program_analysis_source)
if(NOT _program_analysis_source MATCHES "consan_program_analysis_target_ops[.]h")
    message(FATAL_ERROR
        "ConSan program analysis must consume normalized target operations"
    )
endif()

# Raw full instruction words are narrower than generated-header dependencies;
# keep these exact recipes in their named family/member files too.
set(
    _raw_word_owners
    consan_gfx1250_vgpr_bank_state.cpp
    consan_supercollider_gfx11_gfx12_target_ops.cpp
    consan_supercollider_gfx9_target_ops.cpp
)
set(
    _raw_word_regex
    "0[xX][dD][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][uU]|0[xX][bB][fF]860000[uU]"
)
foreach(_file IN LISTS _consan_sources)
    file(STRINGS "${_file}" _raw_words REGEX "${_raw_word_regex}")
    if(NOT _raw_words)
        continue()
    endif()
    get_filename_component(_name "${_file}" NAME)
    list(FIND _raw_word_owners "${_name}" _owner_index)
    if(_owner_index EQUAL -1)
        list(GET _raw_words 0 _first)
        message(
            FATAL_ERROR
            "ConSan boundary violation (raw instruction outside family/member owner) "
            "in ${_file}: ${_first}"
        )
    endif()
endforeach()

# Concrete target-profile data belongs to gfx-named owners. The common
# capability contract may aggregate those immutable products, but it must not
# grow another mixed table of raw product identities.
set(
    _target_profile_fragments
    consan_gfx9_cdna_target_profile.h.inc
    consan_rdna_target_profile.h.inc
    consan_gfx942_target_profile.h.inc
    consan_gfx950_target_profile.h.inc
    consan_gfx1100_target_profile.h.inc
    consan_gfx1201_target_profile.h.inc
    consan_gfx1250_target_profile.h.inc
)
set(
    _concrete_target_profile_fragments
    consan_gfx942_target_profile.h.inc
    consan_gfx950_target_profile.h.inc
    consan_gfx1100_target_profile.h.inc
    consan_gfx1201_target_profile.h.inc
    consan_gfx1250_target_profile.h.inc
)
set(
    _concrete_target_profile_ids
    ROCJITSU_CODE_TARGET_GFX942
    ROCJITSU_CODE_TARGET_GFX950
    ROCJITSU_CODE_TARGET_GFX1100
    ROCJITSU_CODE_TARGET_GFX1201
    ROCJITSU_CODE_TARGET_GFX1250
)
set(
    _concrete_target_profile_arches
    ROCJITSU_CODE_ARCH_CDNA3
    ROCJITSU_CODE_ARCH_CDNA4
    ROCJITSU_CODE_ARCH_RDNA3
    ROCJITSU_CODE_ARCH_RDNA4
    ROCJITSU_CODE_ARCH_CDNA5
)
set(_capability_contract "${_consan_dir}/consan_capability_contract.h")
_consan_assert_no_match(
    "${_capability_contract}"
    "ROCJITSU_CODE_TARGET_GFX|ROCJITSU_CODE_ARCH_(CDNA|RDNA)[0-9]"
    "common capability contract must aggregate gfx-named target profiles"
)
file(READ "${_capability_contract}" _capability_contract_contents)
foreach(_fragment IN LISTS _target_profile_fragments)
    if(NOT EXISTS "${_consan_dir}/${_fragment}")
        message(FATAL_ERROR "ConSan boundary check is missing ${_fragment}")
    endif()
    string(REGEX MATCHALL "#include[^\n]*${_fragment}" _profile_includes
                          "${_capability_contract_contents}")
    list(LENGTH _profile_includes _profile_include_count)
    if(NOT _profile_include_count EQUAL 1)
        message(
            FATAL_ERROR
            "ConSan target-profile owner ${_fragment} has "
            "${_profile_include_count} imports in the common registry"
        )
    endif()
endforeach()
foreach(
    _fragment _target _arch
    IN ZIP_LISTS
       _concrete_target_profile_fragments
       _concrete_target_profile_ids
       _concrete_target_profile_arches
)
    file(READ "${_consan_dir}/${_fragment}" _profile_contents)
    if(NOT _profile_contents MATCHES "${_target}" OR NOT _profile_contents MATCHES "${_arch}")
        message(
            FATAL_ERROR
            "ConSan target-profile owner ${_fragment} does not own ${_target}/${_arch}"
        )
    endif()
endforeach()

# The HSA hook consumes public pipeline/report/diagnostic projections only.
file(GLOB _hook_sources "${_hook_dir}/*.cpp" "${_hook_dir}/*.h")
foreach(_file IN LISTS _hook_sources)
    _consan_assert_no_match(
        "${_file}"
        "consan_moi[.]h|consan_moi_internal[.]h|consan_resource[.]h|consan_lowering[.]h|consan_transform_debug[.]h|consan_.*target_ops[.]h"
        "hook must not include lowerer-private patch/resource/target headers"
    )
    _consan_assert_no_match(
        "${_file}"
        "ConSanPatchInfo|ConSanPatchKind|ConSanCandidateResourcePlan|ConSanCommittedLowering|ConSanTransformArtifacts"
        "hook must consume owned public projections"
    )
endforeach()

# Coverage publication consumes intents and typed lowering commits, never
# patch kinds or mutation geometry.
foreach(
    _source
    IN ITEMS consan_access_policy.cpp consan_observation_policy.cpp
)
    _consan_assert_no_match(
        "${_consan_dir}/${_source}"
        "ConSanPatchInfo|ConSanPatchKind|anchor_offset|trampoline_offset|trampoline_size"
        "coverage must not reconstruct meaning from patch geometry"
    )
endforeach()

# Immutable semantic selection and exact synchronization proof consume narrow
# inventory views, never the mutable transformation transaction.
foreach(
    _source
    IN ITEMS
       consan_fault_selection.h
       consan_fault_selection.cpp
       consan_fault_planning.h
       consan_sync_event_index.h
       consan_sync_event_index.cpp
)
    _consan_assert_no_match(
        "${_consan_dir}/${_source}"
        "ConSanTransformArtifacts"
        "immutable selection/proof must consume narrow inventory products"
    )
endforeach()
file(READ "${_consan_dir}/consan_fault_selection.h" _fault_selection_contract)
if(NOT _fault_selection_contract MATCHES "ConSanFaultSelectionView")
    message(FATAL_ERROR "ConSan fault selection lost its narrow immutable inventory contract")
endif()
file(READ "${_consan_dir}/consan_fault_planning.h" _fault_planning_contract)
if(NOT _fault_planning_contract MATCHES "ConSanFaultPlanningInput" OR
   NOT _fault_planning_contract MATCHES "ConSanFaultPlanningResult")
    message(FATAL_ERROR "ConSan fault planning lost its explicit input/product contract")
endif()
file(READ "${_consan_dir}/consan_fault_injection.h" _fault_application_contract)
foreach(
    _typed_fault_application
    IN ITEMS
       try_apply_barrier_drop_fault_patch
       try_apply_barrier_move_fault_patch
       try_apply_barrier_id_scope_fault_patch
       try_apply_barrier_participant_fault_patch
       try_apply_atomic_fault_patch
       try_apply_lds_fault_patch
       try_apply_ordinary_fault_patch
)
    if(NOT _fault_application_contract MATCHES
       "${_typed_fault_application}[^;]*ConSanFaultMutationPlan")
        message(FATAL_ERROR
            "ConSan fault application must consume its retained typed plan: ${_typed_fault_application}"
        )
    endif()
endforeach()

file(READ "${_consan_dir}/consan_growth_policy.h" _growth_policy_contract)
if(_growth_policy_contract MATCHES "ConSanOptions" OR
   NOT _growth_policy_contract MATCHES "ConSanPatchedImageGrowthLimit")
    message(FATAL_ERROR
        "ConSan image-growth policy must consume its narrow policy value, not the broad options bus"
    )
endif()

# Native emission and target-operation components accept narrow operation
# contracts, not the broad mutable MoiOptions bus.
file(
    GLOB _native_emitter_sources
    "${_consan_dir}/*emission*.cpp"
    "${_consan_dir}/*emission*.h"
    "${_consan_dir}/*target_ops*.cpp"
    "${_consan_dir}/*target_ops*.h"
    "${_consan_dir}/consan_moi_access_target.cpp"
    "${_consan_dir}/consan_moi_access_target.h"
    "${_consan_dir}/consan_moi_target_address.cpp"
    "${_consan_dir}/consan_moi_target_address.h"
)
foreach(_file IN LISTS _native_emitter_sources)
    _consan_assert_no_match(
        "${_file}"
        "MoiOptions"
        "native emitters must accept narrow typed plans"
    )
endforeach()

# Host decoding/analysis accepts typed static mappings rather than private
# patch telemetry. Keep this checked separately from the lifecycle hook.
set(
    _runtime_analysis_sources
    rj_hsa_dbi_moi_report_analyzer.cpp
    rj_hsa_dbi_moi_report_analyzer.h
    rj_hsa_dbi_moi_report_decoder.cpp
    rj_hsa_dbi_moi_report_decoder.h
    rj_hsa_dbi_moi_report_pipeline.cpp
    rj_hsa_dbi_moi_report_pipeline.h
)
foreach(_source IN LISTS _runtime_analysis_sources)
    _consan_assert_no_match(
        "${_hook_dir}/${_source}"
        "ConSanPatchInfo|ConSanPatchKind|ConSanTransformArtifacts|anchor_offset|trampoline_offset|trampoline_size"
        "runtime analysis must not consume lowerer telemetry"
    )
endforeach()
_consan_assert_match_count_at_most(
    "${_hook_dir}/rj_hsa_dbi_moi_report_pipeline.h"
    "AutoMoiRecordReplayStaticMapping"
    3
    "runtime mapping contract remains narrow and typed"
)
file(READ "${_hook_dir}/rj_hsa_dbi_moi_report_pipeline.h" _runtime_contract)
if(NOT _runtime_contract MATCHES "AutoMoiRecordReplayStaticMapping")
    message(FATAL_ERROR "ConSan runtime analysis lost its typed static mapping contract")
endif()

# Every active implementation fragment has one reviewed textual owner. No
# implementation fragment may include another except the three coherent
# SuperCollider regions assembled by its single component wrapper.
set(
    _active_implementation_fragments
    consan_analysis.inc
    consan_composition.inc
    consan_fault_injection.inc
    consan_moi_barrier.inc
    consan_moi_common_emission.inc
    consan_moi_inline_atomic.inc
    consan_moi_inline_shadow.inc
    consan_moi_pipeline.inc
    consan_moi_placement.inc
    consan_moi_record_atomic.inc
    consan_moi_record_fence.inc
    consan_moi_record_replay.inc
    consan_moi_sampled_access.inc
    consan_moi_sampled_sync.inc
    consan_placement.inc
    consan_supercollider.inc
    consan_supercollider_common.inc
    consan_supercollider_flat.inc
    consan_supercollider_lds.inc
    consan_sync_analysis.inc
    consan_validation.inc
)
foreach(_fragment IN LISTS _active_implementation_fragments)
    set(_include_count 0)
    foreach(_source IN LISTS _consan_sources)
        file(READ "${_source}" _contents)
        string(REGEX MATCHALL "#include[^\n]*${_fragment}" _includes "${_contents}")
        list(LENGTH _includes _source_count)
        math(EXPR _include_count "${_include_count} + ${_source_count}")
    endforeach()
    if(NOT _include_count EQUAL 1)
        message(
            FATAL_ERROR
            "ConSan implementation fragment ${_fragment} has ${_include_count} textual owners"
        )
    endif()
endforeach()

file(GLOB _implementation_fragments "${_consan_dir}/*.inc")
foreach(_file IN LISTS _implementation_fragments)
    if(_file MATCHES "[.]h[.]inc$")
        continue()
    endif()
    get_filename_component(_name "${_file}" NAME)
    if(_name STREQUAL "consan_supercollider.inc")
        # The exact three-region wrapper is checked below.
        continue()
    endif()
    _consan_assert_no_match(
        "${_file}"
        "#include.*[.]inc"
        "implementation components must meet through declared interfaces"
    )
endforeach()
file(READ "${_consan_dir}/consan_supercollider.inc" _supercollider_wrapper)
set(
    _expected_supercollider_wrapper
    "#include \"rocjitsu/code/patch/consan/consan_supercollider_common.inc\"\n\n#include \"rocjitsu/code/patch/consan/consan_supercollider_lds.inc\"\n\n#include \"rocjitsu/code/patch/consan/consan_supercollider_flat.inc\"\n"
)
if(NOT _supercollider_wrapper STREQUAL _expected_supercollider_wrapper)
    message(FATAL_ERROR "ConSan SuperCollider implementation wrapper changed without review")
endif()

# Retained filename tombstones are required by the operational no-delete rule;
# they must never regain definitions, declarations, or include closure.
set(
    _retired_fragments
    consan_moi_access_apply.inc
    consan_moi_candidates.inc
    consan_moi_emission.inc
    consan_moi_inline_shadow_emission.inc
    consan_moi_probe_planning.inc
    consan_moi_prologue.inc
    consan_moi_record_event_emission.inc
    consan_moi_record_planning.inc
    consan_moi_runtime_workgroup_gate_emission.inc
    consan_moi_sampled_access_emission.inc
    consan_moi_sampled_atomic_emission.inc
    consan_moi_sync_common.inc
)
foreach(_fragment IN LISTS _retired_fragments)
    _consan_assert_no_match(
        "${_consan_dir}/${_fragment}"
        "#include|[{};]"
        "retired fragment must remain an inert comment-only tombstone"
    )
endforeach()

# Common mode-aware authorities have been deep-reviewed: top-level dispatch,
# ABI/report planning, the shared resource solver, and named multi-mode
# barrier/prologue mechanics. New switches require an explicit review and an
# updated bound; silently growing deep mode behavior fails this test.
function(_consan_assert_reviewed_mode_switch_budget source maximum reason)
    _consan_assert_match_count_at_most(
        "${_consan_dir}/${source}"
        "ConSanMoiEngine::"
        "${maximum}"
        "reviewed mode switch region: ${reason}"
    )
endfunction()
_consan_assert_reviewed_mode_switch_budget(
    consan_moi_placement.inc 5 "shared immutable resource constraint solver"
)
_consan_assert_reviewed_mode_switch_budget(
    consan_moi.cpp 20 "top-level engine dispatch and result publication"
)
_consan_assert_reviewed_mode_switch_budget(
    consan_moi_report_plan.cpp 15 "mode-specific report ABI selection"
)
_consan_assert_reviewed_mode_switch_budget(
    consan_moi_barrier.inc 17 "named RecordReplay/Sampled/Inline barrier component"
)
_consan_assert_reviewed_mode_switch_budget(
    consan_moi_pipeline.inc 11 "resource-plan orchestration"
)
_consan_assert_reviewed_mode_switch_budget(
    consan_moi_support.cpp 10 "shared mode contract validation"
)
_consan_assert_reviewed_mode_switch_budget(
    consan_types.cpp 6 "mode parsing naming and request validation"
)
_consan_assert_reviewed_mode_switch_budget(
    consan_moi_prologue.cpp 4 "shared MOI entry-state initialization"
)
_consan_assert_reviewed_mode_switch_budget(
    consan_moi_model.cpp 4 "host model dispatch"
)

message(STATUS "ConSan architectural boundary checks passed")
