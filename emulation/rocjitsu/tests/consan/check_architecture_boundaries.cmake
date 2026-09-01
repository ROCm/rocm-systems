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

# The production build graph is intentionally small and forward-only:
# contracts -> target normalization -> analysis -> transformation ->
# independent validation -> orchestration. Validation also reads analysis
# products directly, but neither validation nor orchestration is visible below
# its own layer.
set(_consan_build_manifest "${_consan_dir}/CMakeLists.txt")
file(READ "${_consan_build_manifest}" _consan_build_graph)
foreach(
    _component
    IN ITEMS
        contracts
        targets
        analysis
        transform
        validation
        orchestration
)
    if(NOT _consan_build_graph MATCHES "rocjitsu_consan_${_component}")
        message(
            FATAL_ERROR
            "ConSan production build graph is missing ${_component}"
        )
    endif()
endforeach()
if(_consan_build_graph MATCHES "target_sources[(][ \n]*rocjitsu_code")
    message(
        FATAL_ERROR
        "ConSan production sources must not collapse back into rocjitsu_code"
    )
endif()

# Completed analysis products expose one immutable presentation facet. Runtime
# diagnostics copy that facet and may not maintain parallel schemas or
# field-by-field projection maps that can drift from the producing contract.
file(READ "${_consan_dir}/consan_fault_sync_types.h.inc" _fault_presentation_contract)
file(READ "${_consan_dir}/consan_transform_diagnostics.h" _transform_diagnostics_contract)
file(READ "${_consan_dir}/consan_pipeline.cpp" _consan_pipeline)
foreach(_product IN ITEMS FaultSite FaultMutationPlan BarrierMoveDestination)
    string(REPLACE "Plan" "" _presentation "${_product}")
    if(NOT _fault_presentation_contract MATCHES
       "struct ConSan${_product} : ConSan${_presentation}Presentation")
        message(FATAL_ERROR
            "ConSan ${_product} lost its immutable presentation facet"
        )
    endif()
endforeach()
foreach(_diagnostic IN ITEMS FaultSite FaultMutation BarrierMoveDestination)
    if(NOT _transform_diagnostics_contract MATCHES
       "using ConSan${_diagnostic}Diagnostic = ConSan${_diagnostic}Presentation")
        message(FATAL_ERROR
            "ConSan ${_diagnostic} diagnostics regained a parallel schema"
        )
    endif()
    if(NOT _consan_pipeline MATCHES
       "static_cast<const ConSan${_diagnostic}Presentation &>")
        message(FATAL_ERROR
            "ConSan ${_diagnostic} diagnostics lost direct facet publication"
        )
    endif()
endforeach()

set(
    _contract_sources
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
foreach(_source IN LISTS _contract_sources)
    _consan_assert_no_match(
        "${_consan_dir}/${_source}"
        "#include.*consan_(program_analysis|sync_analysis|fault_injection|moi|supercollider|final_validation|validation_inventory|composition|pipeline)[.]h"
        "contracts may not depend on analysis, transformation, validation, or orchestration"
    )
endforeach()

file(GLOB _target_component_sources "${_consan_dir}/*target_ops.cpp")
list(
    APPEND
    _target_component_sources
    "${_consan_dir}/consan_gfx1250_lds_target_ops.cpp"
    "${_consan_dir}/consan_gfx1250_vgpr_bank_state.cpp"
)
foreach(_file IN LISTS _target_component_sources)
    _consan_assert_no_match(
        "${_file}"
        "#include.*consan_(sync_analysis|fault_injection|moi_pipeline|supercollider|final_validation|validation_inventory|composition|pipeline)[.]h"
        "target normalization may not depend on analysis, transformation, validation, or orchestration"
    )
    _consan_assert_no_match(
        "${_file}"
        "ConSanMoiEngine|ConSanFlavor::(Moi|SuperCollider)|RecordReplay|InlineShadow|Sampled"
        "target providers may not decide mode policy"
    )
endforeach()

foreach(_source IN ITEMS consan_fault_selection.cpp consan_program_analysis.cpp consan_sync_analysis.cpp)
    _consan_assert_no_match(
        "${_consan_dir}/${_source}"
        "#include.*consan_(fault_injection|moi_pipeline|supercollider|final_validation|validation_inventory|composition|pipeline)[.]h"
        "analysis may not depend on transformation, validation, or orchestration"
    )
endforeach()

file(
    GLOB _transform_component_sources
    "${_consan_dir}/consan_moi*.cpp"
    "${_consan_dir}/consan_supercollider.cpp"
    "${_consan_dir}/consan_supercollider_report_plan.cpp"
    "${_consan_dir}/consan_supercollider_support.cpp"
    "${_consan_dir}/consan_fault_injection.cpp"
    "${_consan_dir}/consan_perturbation.cpp"
    "${_consan_dir}/consan_barrier_move_proof.cpp"
    "${_consan_dir}/consan_branch_only_relay_router.cpp"
    "${_consan_dir}/consan_descriptor_growth.cpp"
    "${_consan_dir}/consan_placement.cpp"
    "${_consan_dir}/consan_resource.cpp"
)
foreach(_file IN LISTS _transform_component_sources)
    _consan_assert_no_match(
        "${_file}"
        "#include.*consan_(final_validation|validation_inventory|composition|pipeline)[.]h"
        "transformation may not depend on validation or orchestration"
    )
endforeach()

# Descriptor resource mutation has one mechanical owner. MOI contributes its
# narrow policy adapter; SuperCollider's two access regions submit batches
# directly. Retired per-field and mode-local mutation APIs must not reappear.
set(
    _descriptor_mutation_clients
    "${_consan_dir}/consan_descriptor_growth.cpp"
    "${_consan_dir}/consan_moi_shared_lowering.cpp"
    "${_consan_dir}/consan_supercollider_flat.inc"
    "${_consan_dir}/consan_supercollider_lds.inc"
)
foreach(_file IN LISTS _consan_sources)
    if(_file IN_LIST _descriptor_mutation_clients)
        continue()
    endif()
    _consan_assert_no_match(
        "${_file}"
        "apply_consan_descriptor_mutations_to_(patcher|bytes)"
        "descriptor resource mutation must cross its declared owner boundary"
    )
endforeach()
set(
    _descriptor_primitive_owners
    "${_consan_dir}/consan_descriptor.h"
    "${_consan_dir}/consan_descriptor_growth.cpp"
    "${_consan_dir}/consan_validation.inc"
)
foreach(_file IN LISTS _consan_sources)
    if(_file IN_LIST _descriptor_primitive_owners)
        continue()
    endif()
    _consan_assert_no_match(
        "${_file}"
        "grow_descriptor_(vgpr|sgpr)_allocation|update_kernel_descriptor_for_spills"
        "descriptor resource primitives are private to the mutation owner and validation probe"
    )
endforeach()
foreach(_file IN LISTS _consan_sources)
    _consan_assert_no_match(
        "${_file}"
        "apply_consan_descriptor_(sgpr|vgpr)_growths|merge_consan_descriptor_register_growths|ConSanDescriptorRegisterGrowth|MoiActiveKernelResolver|grow_moi_kernel_descriptor_vgprs"
        "retired parallel descriptor-growth authority must not return"
    )
endforeach()
foreach(_source IN ITEMS consan_final_validation.cpp consan_validation_inventory.cpp)
    _consan_assert_no_match(
        "${_consan_dir}/${_source}"
        "#include.*consan_(composition|pipeline)[.]h"
        "validation may not depend on orchestration"
    )
endforeach()

# Relay proofs consume one independently derived patch-inventory product. The
# continuation, donor, and reservoir facets stay distinct so sharing the scan
# cannot broaden any validator's accepted graph.
file(READ "${_consan_dir}/consan_validation.inc" _consan_validation)
if(NOT _consan_validation MATCHES "struct BranchRelayValidationInventory" OR
   NOT _consan_validation MATCHES "make_branch_relay_validation_inventory")
    message(FATAL_ERROR "ConSan relay validation lost its shared typed inventory")
endif()
_consan_assert_no_match(
    "${_consan_dir}/consan_validation.inc"
    "(relay|island)_vertices|branch_only_relay_targets"
    "relay validators must not rebuild parallel patch-inventory views"
)

file(
    GLOB _extension_mode_sources
    "${_consan_dir}/consan_moi_record*.cpp"
    "${_consan_dir}/consan_moi_record*.inc"
    "${_consan_dir}/consan_moi_sampled*.cpp"
    "${_consan_dir}/consan_moi_sampled*.inc"
    "${_consan_dir}/consan_moi_inline*.cpp"
    "${_consan_dir}/consan_moi_inline*.inc"
    "${_consan_dir}/consan_supercollider.cpp"
    "${_consan_dir}/consan_supercollider.inc"
)
foreach(_file IN LISTS _extension_mode_sources)
    _consan_assert_no_match(
        "${_file}"
        "ROCJITSU_CODE_ARCH_|isa/arch/amdgpu/generated/"
        "mode providers may not name concrete target architectures"
    )
endforeach()
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_inline_model.h.inc"
    "ConSanMoiInlineReleaseClaim|ConSanMoiInlineVersionedReleaseState|ConSanMoiInlineReleaseClaimResult|ConSanMoiInlineReleaseTransactionEvent|consan_moi_inline_plan_release_claim|consan_moi_inline_release_transaction_is_sound|ConSanMoiInlineCausalTokenView|consan_moi_inline_capture_causal_snapshot|ConSanMoiInlineStableReleaseEvidence|ConSanMoiInlineQualificationExpectation|ConSanMoiInlineQualificationResult|consan_moi_inline_qualify_token_evidence|ConSanMoiInlineCausalImportPlan|consan_moi_inline_plan_causal_import"
    "host-only InlineShadow reference oracles must not return to the production report contract"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_shadow_models.h.inc"
    "ConSanMoiInlineAcquiredEpochTokenPublishResult|consan_moi_inline_publish_acquired_epoch_token|consan_moi_inline_acquired_epoch_orders|consan_moi_inline_acquired_epoch_orders_pair|consan_moi_inline_stable_token_orders\\("
    "host-only InlineShadow token reference oracles must not return to production shadow models"
)
file(READ "${ROCJITSU_SOURCE_DIR}/tests/patch/consan/analysis_test.cpp" _target_extension_test)
file(
    READ
    "${ROCJITSU_SOURCE_DIR}/tests/patch/consan/moi_mode_planning_test.cpp"
    _mode_extension_test
)
if(NOT _target_extension_test MATCHES
       "HypotheticalTargetRegistersNormalizedAnalysisWithoutModeChanges" OR
   NOT _mode_extension_test MATCHES
       "HypotheticalModeRegistersWithoutConcreteTargetChanges")
    message(FATAL_ERROR "ConSan extension-axis exercises are missing")
endif()

# Dispatch placement composes one mode-owned demand with one target-owned
# capability. The shared register search must not rediscover either axis from
# product/encoding predicates or a second mode-registry lookup.
set(_moi_placement_file "${_consan_dir}/consan_moi_placement.inc")
file(READ "${_moi_placement_file}" _moi_placement)
string(FIND "${_moi_placement}" "configure_automatic_moi_dispatch_id_sgprs" _dispatch_begin)
string(FIND "${_moi_placement}" "plan_moi_dispatch_id_fallback" _dispatch_end)
if(_dispatch_begin LESS 0 OR _dispatch_end LESS_EQUAL _dispatch_begin)
    message(FATAL_ERROR "ConSan dispatch-placement boundary could not be located")
endif()
math(EXPR _dispatch_length "${_dispatch_end} - ${_dispatch_begin}")
string(SUBSTRING "${_moi_placement}" ${_dispatch_begin} ${_dispatch_length} _dispatch_placement)
if(_dispatch_placement MATCHES
   "consan_(uses_gfx|arch_is_(cdna|rdna))|moi_mode_operations")
    message(
        FATAL_ERROR
        "ConSan dispatch placement must compose normalized target and mode capabilities"
    )
endif()

# Dispatch-key and call-return registers belong to one shared scalar-router
# allocation. Do not restore mode-prefixed copies or independently optional
# mechanism-owned fields in the broad operating point.
file(GLOB _consan_production_files "${_consan_dir}/*.cpp" "${_consan_dir}/*.h" "${_consan_dir}/*.inc")
foreach(_file IN LISTS _consan_production_files)
    _consan_assert_no_match(
        "${_file}"
        "kRdna4(Exec|Vcc|WorkitemIdX|ScopeDevice)"
        "universal AMDGPU ABI operands must use the target-neutral capability contract"
    )
    _consan_assert_no_match(
        "${_file}"
        "MoiOptions"
        "production must carry immutable input and MOI operating-point state separately"
    )
    _consan_assert_no_match(
        "${_file}"
        "moi_initializes_owner_epoch|moi_initialize_owner_epoch[.]value_or"
        "owner/epoch initialization must remain one resolved operating-point decision"
    )
    _consan_assert_no_match(
        "${_file}"
        "moi_record_replay_workgroup_private_offsets"
        "site-local private workgroup capture must not return to the code-object-wide operating point"
    )
    _consan_assert_no_match(
        "${_file}"
        "build_sampled_private_epoch_layout"
        "shared private layout construction must consume mode-supplied typed demand"
    )
    _consan_assert_no_match(
        "${_file}"
        "append_moi_(atomic|fence|barrier)_lowering_commit"
        "synchronization evidence plans must publish through one shared commit boundary"
    )
    _consan_assert_no_match(
        "${_file}"
        "moi_(inline|record_replay)_(indirect_(pc|scc)|dispatch_key|call_return)_sgpr|moi_router_(indirect_(pc|scc)|dispatch_key|call_return)_sgpr|moi_inline_(branch_only_scalar_spill|dynamic_stack_borrowed_sgpr)"
        "scalar-router and branch-only preservation state must remain shared typed allocations"
    )
    _consan_assert_no_match(
        "${_file}"
        "(^|[^A-Za-z0-9_])moi_(owner|epoch)_vgpr[(]"
        "accepted owner and epoch projections must stay on the typed pair"
    )
    _consan_assert_no_match(
        "${_file}"
        "moi_(record_replay_dense_barrier_router|inline_access_present)"
        "mode semantics must not return to the mutable operating point"
    )
endforeach()
file(READ "${_consan_dir}/consan_options.h.inc" _consan_options_contract)
if(NOT _consan_options_contract MATCHES
       "point[.]moi_initialize_owner_epoch[ \t]*=[ \t]*options[.]moi_init_owner_epoch")
    message(FATAL_ERROR
        "ConSan initial operating-point construction must resolve owner/epoch initialization"
    )
endif()
file(READ "${_consan_dir}/consan_capability_contract.h" _target_neutral_abi_contract)
foreach(_operand IN ITEMS ExecLo ExecHi VccLo VccHi WorkitemIdX ScopeDevice)
    if(NOT _target_neutral_abi_contract MATCHES "kAmdGpu${_operand}")
        message(FATAL_ERROR
            "ConSan target-neutral AMDGPU ABI contract lost ${_operand}"
        )
    endif()
endforeach()
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_native_abi.h"
    "kTtmp"
    "concrete command-processor TTMP indices belong to gfx-named target profiles"
)
foreach(_target_profile IN ITEMS gfx1201 gfx1250)
    file(
        READ
        "${_consan_dir}/consan_${_target_profile}_target_profile.h.inc"
        _workgroup_identity_profile
    )
    if(NOT _workgroup_identity_profile MATCHES
       "command_processor_workgroup_identity")
        message(FATAL_ERROR
            "ConSan ${_target_profile} profile lost its target-owned workgroup ABI"
        )
    endif()
endforeach()
string(FIND "${_moi_placement}" "moi_descriptor_workgroup_sources" _workgroup_source_begin)
string(FIND "${_moi_placement}" "record_replay_persistent_workgroup_sources"
       _workgroup_source_end)
if(_workgroup_source_begin LESS 0 OR _workgroup_source_end LESS_EQUAL _workgroup_source_begin)
    message(FATAL_ERROR "ConSan workgroup-source boundary could not be located")
endif()
math(EXPR _workgroup_source_length "${_workgroup_source_end} - ${_workgroup_source_begin}")
string(SUBSTRING "${_moi_placement}" ${_workgroup_source_begin}
       ${_workgroup_source_length} _workgroup_source_placement)
if(_workgroup_source_placement MATCHES "kTtmp|ConSanWorkgroupIdentitySource")
    message(FATAL_ERROR
        "ConSan common workgroup placement must consume the target-owned exact ABI"
    )
endif()
file(READ "${_consan_dir}/consan_moi_shared_lowering.h" _moi_private_layout_contract)
if(NOT _moi_private_layout_contract MATCHES "struct MoiPrivateStateDemand" OR
   NOT _moi_private_layout_contract MATCHES "class MoiPrivateEpochLayoutCache")
    message(FATAL_ERROR
        "ConSan private-state lowering lost its typed demand or shared descriptor cache"
    )
endif()

# Access scratch sizing and spill policy compose one target-normalized,
# operating-point-projected fact product. Mode providers may not regain the
# broad point or raw architecture identity through either policy context.
file(READ "${_consan_dir}/consan_moi_access_target.h" _moi_access_resource_contract)
file(READ "${_consan_dir}/consan_moi_mode_planning.h" _moi_mode_planning_contract)
if(NOT _moi_access_resource_contract MATCHES "struct MoiAccessResourceFacts" OR
   NOT _moi_access_resource_contract MATCHES "resolve_moi_access_resource_facts" OR
   NOT _moi_mode_planning_contract MATCHES
       "access_scratch_vgpr_count[^;]*MoiAccessResourceFacts")
    message(FATAL_ERROR
        "ConSan access resource planning lost its normalized fact product"
    )
endif()
foreach(_context IN ITEMS MoiOperandOverlapSpillContext MoiAccessSpillFallbackContext)
    if(_moi_mode_planning_contract MATCHES
       "struct ${_context}[^}]*ConSanMoiOperatingPoint" OR
       _moi_mode_planning_contract MATCHES "struct ${_context}[^}]*rj_code_arch_t")
        message(FATAL_ERROR
            "ConSan ${_context} must not expose the broad point or raw architecture"
        )
    endif()
endforeach()
foreach(_file IN LISTS _consan_production_files)
    _consan_assert_no_match(
        "${_file}"
        "sampled_(spill_backed_scratch_count|access_supports_spill_backed_operand_recovery)"
        "retired Sampled access-resource adapters must not return"
    )
endforeach()

# Scalar ABI and dense-router selection share one projection of the accepted
# operating point. Mode callbacks consume that projection, and dense routing
# consumes the scalar ABI already selected by the same mode rather than
# reconstructing it inside each router planner.
file(READ "${_consan_dir}/consan_moi_placement_contracts.h" _moi_placement_contract)
file(READ "${_consan_dir}/consan_moi_mode_planning.cpp" _moi_mode_planning_implementation)
file(READ "${_consan_dir}/consan_moi_sampled_contracts.h" _moi_sampled_contract)
if(NOT _moi_placement_contract MATCHES "struct MoiScalarRoutingState" OR
   NOT _moi_placement_contract MATCHES "project_moi_scalar_routing_state")
    message(FATAL_ERROR
        "ConSan scalar routing lost its narrow operating-point projection"
    )
endif()
foreach(_callback IN ITEMS scalar_abi dense_router)
    if(_moi_mode_planning_contract MATCHES
       "\\(\\*${_callback}\\)\\([^;]*ConSanMoiOperatingPoint")
        message(FATAL_ERROR
            "ConSan ${_callback} mode callback must not receive the broad operating point"
        )
    endif()
endforeach()
if(NOT _moi_mode_planning_contract MATCHES
       "dense_router[^;]*MoiScalarAbiPlan[^;]*MoiScalarRoutingState[^;]*MoiScalarTargetFacts" OR
   NOT _moi_mode_planning_implementation MATCHES
       "dense_router\\(operations\\.scalar_abi\\(routing_state\\),[ \t\n]*routing_state")
    message(FATAL_ERROR
        "ConSan dense routing must consume the mode-selected scalar ABI exactly once"
    )
endif()
if(_moi_mode_planning_contract MATCHES
   "\\(\\*dense_router\\)\\([^;]*ConSanTargetProfile")
    message(FATAL_ERROR
        "ConSan dense-router mode callbacks must not receive the complete target profile"
    )
endif()
if(_moi_sampled_contract MATCHES
   "moi_sampled_publication_state_sgprs[^;]*(ConSanRequest|ConSanMoiOperatingPoint)")
    message(FATAL_ERROR
        "ConSan Sampled publication layout must not regain request or operating-point buses"
    )
endif()
_consan_assert_no_match(
    "${_consan_dir}/consan_resource_types.h.inc"
    "bool[ \t]+changed"
    "placement products must not duplicate an operating-point transition bit"
)
file(READ "${_consan_dir}/consan_moi.cpp" _moi_coordinator)
string(REGEX MATCHALL
    "attempted_operating_point[ \t\n]*!=[ \t\n]*effective_point"
    _derived_operating_point_transitions
    "${_moi_coordinator}"
)
list(LENGTH _derived_operating_point_transitions _derived_operating_point_transition_count)
if(NOT _derived_operating_point_transition_count EQUAL 2)
    message(FATAL_ERROR
        "ConSan coordinator must derive both accepted placement transitions"
    )
endif()
file(READ "${_consan_dir}/consan_moi_sync_emission.h" _moi_sync_commit_contract)
if(NOT _moi_sync_commit_contract MATCHES "append_moi_sync_lowering_commit" OR
   NOT _moi_sync_commit_contract MATCHES "plan[.]intent_ids[(][)]")
    message(FATAL_ERROR
        "ConSan synchronization commit boundary lost its plan-owned intent set"
    )
endif()
foreach(_patch_commit_client IN ITEMS
    consan_moi_access_apply.cpp
    consan_moi_sync_emission.cpp
    consan_supercollider_common.inc
)
    _consan_assert_no_match(
        "${_consan_dir}/${_patch_commit_client}"
        "vector<ConSanCommittedLoweringLocation>[ \\t]+locations"
        "patch geometry must become semantic locations in one contract-level owner"
    )
    _consan_assert_no_match(
        "${_consan_dir}/${_patch_commit_client}"
        "make_moi_sync_lowering_commit"
        "MOI synchronization must use the shared patch-geometry commit owner"
    )
endforeach()
file(READ "${_consan_dir}/consan_placement.inc" _committed_patch_location_owner)
if(NOT _committed_patch_location_owner MATCHES
       "make_consan_instrumented_patch_lowering" OR
   NOT _committed_patch_location_owner MATCHES
       "patch[.]trampoline_size == 0u")
    message(FATAL_ERROR
        "ConSan shared committed-patch location construction is missing"
    )
endif()
foreach(_semantic_commit_client IN LISTS _consan_production_files)
    if(_semantic_commit_client MATCHES
       "consan_(access_policy[.]cpp|observation_plan[.]h[.]inc|placement[.]inc)$")
        continue()
    endif()
    _consan_assert_no_match(
        "${_semantic_commit_client}"
        "make_consan_committed_lowering"
        "lowering clients must use the typed instrumented or rejection publication boundary"
    )
endforeach()
file(READ "${_consan_dir}/consan_observation_plan.h.inc" _rejection_publication_contract)
if(NOT _rejection_publication_contract MATCHES "publish_lowering_rejection")
    message(FATAL_ERROR
        "ConSan coverage ledger lost its non-instrumented publication authority"
    )
endif()

# Object-wide mode semantics are selected once by their mode owners and then
# consumed as one immutable product. Common resource and emission components
# must not become a second authority for either decision.
foreach(_file IN LISTS _consan_production_files)
    get_filename_component(_name "${_file}" NAME)
    if(NOT _name STREQUAL "consan_moi_internal.h" AND
       NOT _name STREQUAL "consan_moi_record_replay.cpp")
        _consan_assert_no_match(
            "${_file}"
            "semantics[.]dense_barrier_router[ \t]*="
            "dense barrier routing must be selected only by RecordReplay planning"
        )
    endif()
    if(NOT _name STREQUAL "consan_moi_internal.h" AND
       NOT _name STREQUAL "consan_moi_inline_shadow.cpp")
        _consan_assert_no_match(
            "${_file}"
            "semantics[.]inline_access_present[ \t]*="
            "InlineShadow access presence must be selected only by InlineShadow planning"
        )
    endif()
    if(NOT _name STREQUAL "consan_moi_record_replay.cpp" AND
       NOT _name STREQUAL "consan_moi_sampled.cpp" AND
       NOT _name STREQUAL "consan_moi_inline_shadow.cpp")
        _consan_assert_no_match(
            "${_file}"
            "semantics[.]report_layout[ \t]*="
            "report layout must be selected only by mode-owned planning"
        )
    endif()
    if(NOT _name STREQUAL "consan_moi_sampled.cpp")
        _consan_assert_no_match(
            "${_file}"
            "semantics[.]reserved_(barrier|atomic)_island_count[ \t]*="
            "Sampled island reservations must be selected only by Sampled planning"
        )
    endif()
    _consan_assert_no_match(
        "${_file}"
        "sampled_reserved_(barrier|atomic)_island_count"
        "Sampled lowering must consume its once-selected island reservations"
    )
    if(NOT _name STREQUAL "consan_moi_engine_contracts.cpp" AND
       NOT _name STREQUAL "consan_moi_engine_contracts.h" AND
       NOT _name STREQUAL "consan_moi_record_replay.cpp" AND
       NOT _name STREQUAL "consan_moi_sampled.cpp" AND
       NOT _name STREQUAL "consan_moi_inline_shadow.cpp")
        _consan_assert_no_match(
            "${_file}"
            "resolve_moi_report_layout"
            "lowering must consume the once-selected object-mode report layout"
        )
    endif()
endforeach()

# Dispatch identity is one typed allocation. Its scalar/vector choice and
# scalar-placement provenance must not return as independently mutable fields.
_consan_assert_no_match(
    "${_consan_dir}/consan_options.h.inc"
    "(bool|std::optional<uint16_t>)[ \t]+(automatic_moi_dispatch_id_sgprs|automatic_moi_private_dispatch_id|moi_dispatch_id_sgpr|moi_dispatch_id_vgpr)[ \t]*;"
    "dispatch identity must remain one typed operating-point allocation"
)

# Scalar allocations carry their selection provenance. The owner scalar must
# not return to a loose optional plus a separately mutable automatic marker.
_consan_assert_no_match(
    "${_consan_dir}/consan_options.h.inc"
    "(bool[ \t]+automatic_moi_owner_sgpr|std::optional<uint16_t>[ \t]+moi_owner_sgpr)[ \t]*;"
    "owner scalar and its provenance must remain one typed allocation"
)

# Exact workgroup identity is an optional tuple, not four independently
# optional coordinates. Register and private-offset storage share that one
# all-or-none mechanism.
_consan_assert_no_match(
    "${_consan_dir}/consan_options.h.inc"
    "(struct|class)[ \t]+ConSanMoiPersistentWorkgroup(Registers|PrivateOffsets)"
    "persistent workgroup storage must remain one shared all-or-none tuple"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_options.h.inc"
    "std::optional<(uint16_t|uint32_t)>[ \t]+(x|y|z|cluster_workgroup_id)[ \t]*;"
    "persistent workgroup coordinates must not regain independent optional storage"
)
# Accepted persistent state must not regain independently optional fields.
# `ConSanMoiOwnerEpochVgprSources` deliberately permits partial, site-local
# materialization under the short names `owner` and `epoch`, so the guard is
# scoped to the accepted-state field spellings rather than every optional
# owner/epoch projection in this contracts file.
foreach(_persistent_state_file IN ITEMS
    consan_options.h.inc
    consan_moi_record_event_emission.h
)
  _consan_assert_no_match(
      "${_consan_dir}/${_persistent_state_file}"
      "std::optional<uint16_t>[ \t]+moi_(owner|epoch)_vgpr[ \t]*;"
      "accepted persistent owner and epoch VGPRs must remain one typed pair"
  )
endforeach()
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_placement_contracts.h"
    "std::optional<uint16_t>[ \t]+(owner|epoch)[ \t]*;"
    "relay-visible persistent owner and epoch VGPRs must remain one typed pair"
)
_consan_assert_match_count_at_most(
    "${_consan_dir}/consan_options.h.inc"
    "std::optional<ConSanMoiOwnerEpochRegisters>"
    1
    "only the typed owner/epoch state may store an optional raw pair"
)
foreach(_file IN LISTS _consan_production_files)
    if(NOT _file STREQUAL "${_consan_dir}/consan_options.h.inc")
        _consan_assert_no_match(
            "${_file}"
            "std::optional<ConSanMoiOwnerEpochRegisters>"
            "accepted owner/epoch carriers must reuse the typed state"
        )
    endif()
endforeach()
foreach(_file IN LISTS _consan_production_files)
    _consan_assert_no_match(
        "${_file}"
        "find_moi_borrowed_entry_backup_vgpr_for_(point|state)"
        "borrowed-entry planning must consume the one persistent-VGPR state view"
    )
endforeach()
file(READ "${_consan_dir}/consan_moi_placement_contracts.h" _persistent_vgpr_view_contract)
if(NOT _persistent_vgpr_view_contract MATCHES
       "moi_persistent_vgpr_state_view[^}]*ConSanMoiOperatingPoint" OR
   NOT _persistent_vgpr_view_contract MATCHES
       "moi_persistent_vgpr_state_view[^}]*ConSanMoiPersistentVgprAssignment" OR
   NOT _persistent_vgpr_view_contract MATCHES "for_each_range")
    message(FATAL_ERROR
        "ConSan placement lost its shared persistent-VGPR state projection"
    )
endif()
foreach(_persistent_sgpr_consumer IN ITEMS
    consan_moi_placement.inc
    consan_moi_prologue.cpp
    consan_moi_sampled_sync.inc
)
    _consan_assert_no_match(
        "${_consan_dir}/${_persistent_sgpr_consumer}"
        "moi_persistent_sgprs[.]record_replay_workgroup[.]values[(][)]"
        "persistent-SGPR consumers must use the state-owned width-aware traversal"
    )
endforeach()
file(READ "${_consan_dir}/consan_options.h.inc" _persistent_sgpr_state_contract)
if(NOT _persistent_sgpr_state_contract MATCHES
       "class ConSanMoiPersistentSgprState[^}]*for_each_range")
    message(FATAL_ERROR
        "ConSan persistent-SGPR state lost its complete range traversal"
    )
endif()

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

_consan_assert_no_match(
    "${_consan_dir}/consan_atomic_classifier.cpp"
    "ROCJITSU_CODE_ARCH_"
    "atomic classification must consume the typed target profile"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_sync_analysis.inc"
    "ROCJITSU_CODE_ARCH_"
    "synchronization analysis must consume the typed target profile"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_sync_analysis.inc"
    "site->raw_(scope|th)|consan_uses_gfx9_cdna_encoding|consan_uses_gfx12_(cdna|rdna)_execution"
    "synchronization analysis must consume target-normalized workgroup-acquire ordering"
)
foreach(_cache_semantic_consumer IN ITEMS
    consan_sync_analysis.inc
    consan_sync_metadata.cpp
    consan_fault_selection.cpp
)
    _consan_assert_no_match(
        "${_consan_dir}/${_cache_semantic_consumer}"
        "global_(wb|inv)|buffer_(wb|wbl2|inv|gl0_inv|gl1_inv)|s_dcache_inv|consan_uses_gfx11_encoding"
        "cache semantic consumers must use target-normalized cache operations"
    )
endforeach()
foreach(_wait_semantic_consumer IN ITEMS
    consan_sync_analysis.inc
    consan_fault_injection.inc
)
    _consan_assert_no_match(
        "${_consan_dir}/${_wait_semantic_consumer}"
        "build_(rdna3_)?s_wait_(global|flat|lds|storecnt|loadcnt|vscnt)"
        "wait semantic consumers must use target-normalized wait effects"
    )
    _consan_assert_no_match(
        "${_consan_dir}/${_wait_semantic_consumer}"
        "mnemonic[ \t]*==[ \t]*\"s_wait"
        "wait semantic consumers must not recognize target-native wait mnemonics"
    )
endforeach()
file(READ "${_consan_dir}/consan_program_analysis_target_ops.h"
     _program_analysis_normalized_contract)
if(NOT _program_analysis_normalized_contract MATCHES "workgroup_acquire_ordering")
    message(FATAL_ERROR
        "ConSan program-analysis target operations lost normalized workgroup-acquire ordering"
    )
endif()
if(NOT _program_analysis_normalized_contract MATCHES "classify_cache_operation")
    message(FATAL_ERROR
        "ConSan program-analysis target operations lost normalized cache-operation classification"
    )
endif()
if(NOT _program_analysis_normalized_contract MATCHES "classify_wait_instruction")
    message(FATAL_ERROR
        "ConSan program-analysis target operations lost normalized wait-effect classification"
    )
endif()
if(NOT _program_analysis_normalized_contract MATCHES "enum class ConSanMemoryScope")
    message(FATAL_ERROR
        "ConSan program-analysis target operations lost normalized memory scope"
    )
endif()
_consan_assert_no_match(
    "${_consan_dir}/consan_atomic_classifier.h"
    "(ConSanMemoryScope|uint32_t)[ \t]+scope"
    "atomic address lowering must not duplicate the semantic scope authority"
)
foreach(_scope_semantic_consumer IN ITEMS
    consan_analysis.inc
    consan_atomic_classifier.cpp
    consan_atomic_fence_policy.cpp
    consan_fault_selection.cpp
    consan_moi_record_event_emission.cpp
    consan_moi_sampled_atomic_emission.cpp
    consan_moi_sampled_sync.inc
    consan_moi_sync_emission.cpp
    consan_perturbation_policy.cpp
    consan_sync_analysis.inc
    consan_sync_metadata.cpp
)
    _consan_assert_no_match(
        "${_consan_dir}/${_scope_semantic_consumer}"
        "static_cast<ConSanMemoryScope>|scope[ \t]*([!=<>]=?|value_or)[ \t]*[({]?[0-3]u?"
        "semantic consumers must not reinterpret numeric memory-scope codes"
    )
endforeach()
foreach(_scope_semantic_only_consumer IN ITEMS
    consan_atomic_classifier.cpp
    consan_fault_selection.cpp
    consan_moi_record_event_emission.cpp
    consan_moi_sampled_atomic_emission.cpp
    consan_moi_sampled_sync.inc
    consan_perturbation_policy.cpp
    consan_sync_metadata.cpp
)
    _consan_assert_no_match(
        "${_consan_dir}/${_scope_semantic_only_consumer}"
        "raw_scope"
        "semantic scope consumers must not depend on retained raw target scope"
    )
endforeach()
file(READ "${_consan_dir}/consan_program_analysis_gfx12_target_ops.h"
     _gfx12_program_analysis_contract)
if(NOT _gfx12_program_analysis_contract MATCHES "normalize_gfx12_memory_scope")
    message(FATAL_ERROR
        "gfx12 program analysis lost its explicit raw-to-semantic scope mapping"
    )
endif()
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_sampled_atomic_emission.cpp"
    "sampled_atomic_scope"
    "Sampled emitters must share the mode-owned normalized scope-to-report mapping"
)
file(READ "${_consan_dir}/consan_moi_record_replay_types.h.inc"
     _record_replay_scope_contract)
if(NOT _record_replay_scope_contract MATCHES "consan_moi_record_replay_scope")
    message(FATAL_ERROR
        "Record/Replay lost its explicit normalized scope-to-report mapping"
    )
endif()
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_record_event_emission.cpp"
    "static_cast<uint32_t>\\([^\\)]*scope"
    "Record/Replay emission must use its mode-owned scope-to-report mapping"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_capability_contract.h"
    "ConSanWaitCounterFamily|wait_counter_family"
    "mode code must consume an exact wait operation rather than a broad encoding-family fact"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_supercollider_flat.inc"
    "ConSanWaitCounterFamily|wait_counter_family|build_s_wait_flat_load0"
    "SuperCollider flat lowering must consume its target-owned completion-wait operation"
)
file(READ "${_consan_dir}/consan_supercollider_target_ops.h" _sc_target_operations)
if(NOT _sc_target_operations MATCHES "consan_sc_build_guest_flat_completion_wait")
    message(FATAL_ERROR
        "ConSan SuperCollider target operations lost guest-flat completion-wait ownership"
    )
endif()
_consan_assert_no_match(
    "${_consan_dir}/../instrumentation_builder.h"
    "build_s_wait_flat_(load|store)_lds0[(]|build_s_wait_alu_va_sdst0[(]rj_code_arch_t"
    "instrumentation must not restore superseded duplicate wait-builder aliases"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_access_classifier.cpp"
    "isa/arch/amdgpu/generated/|ROCJITSU_CODE_ARCH_"
    "access classification must consume normalized inventory and the typed target profile"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_analysis.inc"
    "ROCJITSU_CODE_ARCH_"
    "program analysis must consume target-normalized decodes and profile facts"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_support.cpp"
    "ROCJITSU_CODE_ARCH_"
    "shared MOI support must consume target-neutral mechanism operations"
)
foreach(_relocation_client IN ITEMS
    consan_moi_sampled_access_emission.cpp
    consan_moi_shared_lowering.cpp
    consan_moi_inline_shadow_emission.cpp
)
    _consan_assert_no_match(
        "${_consan_dir}/${_relocation_client}"
        "build_moi_relocated_guest_access_words"
        "mode emitters must publish relocated guest words through one shared append boundary"
    )
endforeach()
file(READ "${_consan_dir}/consan_moi_relocation.cpp" _moi_relocation_owner)
string(REGEX MATCHALL "append_pc_delta_builder" _moi_indirect_target_recipes
       "${_moi_relocation_owner}")
list(LENGTH _moi_indirect_target_recipes _moi_indirect_target_recipe_count)
if(NOT _moi_indirect_target_recipe_count EQUAL 1)
    message(FATAL_ERROR
        "ConSan SCC-preserving indirect jumps must share one target-preparation recipe"
    )
endif()
# Barrier and atomic synchronization consume the same Sampled causal-window
# identity policy. Keep that policy in its mode-local emission owner instead of
# allowing the two lowering routes to drift apart again.
foreach(_sampled_window_client IN ITEMS
    consan_moi_sampled_sync.inc
    consan_moi_sampled_atomic_emission.cpp
)
    file(READ "${_consan_dir}/${_sampled_window_client}" _sampled_window_client_source)
    string(REGEX MATCHALL "append_sampled_causal_window_validation"
           _sampled_window_validation_calls "${_sampled_window_client_source}")
    list(LENGTH _sampled_window_validation_calls _sampled_window_validation_call_count)
    if(NOT _sampled_window_validation_call_count EQUAL 1)
        message(FATAL_ERROR
            "ConSan Sampled ${_sampled_window_client} must consume one shared causal-window validator"
        )
    endif()
    _consan_assert_no_match(
        "${_consan_dir}/${_sampled_window_client}"
        "narrow_equal_(literal|vgpr|dispatch_id|workgroup)"
        "Sampled synchronization routes must not redeclare causal-window validation policy"
    )
endforeach()
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_sync_emission.cpp"
    "consan_uses_gfx9_cdna_encoding|ConSanMoiLiteralDispatchIdPolicy|moi_report_dispatch_id_source_permitted"
    "shared synchronization emission must consume normalized target facts and authorized dispatch sources"
)
# InlineShadow atomic tables share one private address-hash implementation;
# their ABI entry types select only the distinct 32- and 40-byte strides.
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_sync_emission.h"
    "append_inline_(atomic|causal).*address"
    "InlineShadow atomic-table address emission must remain private to its implementation owner"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_shared_lowering.cpp"
    "append_inline_(workgroup_key|acquired_token_slot_address|atomic.*address|causal.*address)"
    "shared lowering must not redeclare unrelated emission-owner helpers"
)
_consan_assert_match_count_at_most(
    "${_consan_dir}/consan_moi_sync_emission.cpp"
    "0x85ebca6bu"
    1
    "InlineShadow atomic release and causal-snapshot tables must share one address hash"
)
file(READ "${_consan_dir}/consan_moi_sync_emission.cpp" _moi_sync_emission_owner)
if(NOT _moi_sync_emission_owner MATCHES "class InlineExecMaskEmission")
    message(FATAL_ERROR
        "ConSan InlineShadow synchronization lost its shared EXEC-mask emitter"
    )
endif()
string(REGEX MATCHALL "InlineExecMaskEmission[ \t]+exec_masks"
       _inline_exec_mask_consumers "${_moi_sync_emission_owner}")
list(LENGTH _inline_exec_mask_consumers _inline_exec_mask_consumer_count)
if(NOT _inline_exec_mask_consumer_count EQUAL 3)
    message(FATAL_ERROR
        "ConSan InlineShadow synchronization must share EXEC-mask emission across three transactions"
    )
endif()
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_sync_emission.cpp"
    "const auto (restore_exec|save_exec|narrow_vcc)[^;]*instrumentation::build_"
    "InlineShadow transactions must not redeclare EXEC-mask instruction recipes"
)
if(NOT _moi_sync_emission_owner MATCHES
       "append_inline_atomic_table_address<ConSanMoiInlineAtomicReleaseSlot>" OR
   NOT _moi_sync_emission_owner MATCHES
       "append_inline_atomic_table_address<ConSanMoiInlineCausalSnapshot>")
    message(FATAL_ERROR
        "ConSan InlineShadow atomic tables lost their one ABI-typed address mechanism"
    )
endif()
# RecordReplay records and dispatch slots, Sampled watchpoint/window tables,
# dynamic report records, and all three InlineShadow metadata-table ABIs use
# the same target-normalized base + index * stride operation. Hash/key and bank
# construction remain with each semantic owner; stride and 64-bit address
# formation must not regain mode-local copies.
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_internal.h"
    "MoiDynamicRecordAddressRequest|consan_detail::append_dynamic_record_address"
    "the shared indexed-address contract must not narrow back to dynamic records"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_sync_emission.cpp"
    "build_v_add_u64_vgpr_offset"
    "synchronization metadata tables must use the shared indexed-address operation"
)
foreach(
    _source IN ITEMS
    consan_moi_sampled_access_emission.cpp
    consan_moi_sampled_atomic_emission.cpp
    consan_moi_sampled_sync.inc
    consan_moi_shared_lowering.cpp
)
    _consan_assert_no_match(
        "${_consan_dir}/${_source}"
        "build_v_add_u64_vgpr_offset"
        "mode-owned report tables must use the shared indexed-address operation"
    )
endforeach()
foreach(
    _source IN ITEMS
    consan_moi_sampled_access_emission.h
    consan_moi_sampled_access_emission.cpp
    consan_moi_sampled_atomic_emission.cpp
    consan_moi_sampled_sync.inc
)
    _consan_assert_no_match(
        "${_consan_dir}/${_source}"
        "append_sampled_indexed_address"
        "Sampled components must consume the common indexed-address contract directly"
    )
endforeach()
string(
    REGEX MATCHALL
    "consan_detail::append_moi_indexed_address"
    _moi_sync_indexed_address_calls
    "${_moi_sync_emission_owner}"
)
list(LENGTH _moi_sync_indexed_address_calls _moi_sync_indexed_address_call_count)
if(NOT _moi_sync_indexed_address_call_count EQUAL 2)
    message(FATAL_ERROR
        "ConSan release/snapshot and acquired-token tables must share indexed addressing"
    )
endif()
file(READ "${_consan_dir}/consan_moi_dynamic_record_emission.cpp" _moi_dynamic_record_owner)
string(
    REGEX MATCHALL
    "consan_detail::append_moi_indexed_address"
    _moi_dynamic_indexed_address_calls
    "${_moi_dynamic_record_owner}"
)
list(LENGTH _moi_dynamic_indexed_address_calls _moi_dynamic_indexed_address_call_count)
if(NOT _moi_dynamic_indexed_address_call_count EQUAL 1)
    message(FATAL_ERROR
        "ConSan dynamic records must share the target-normalized indexed-address operation"
    )
endif()
file(READ "${_consan_dir}/consan_moi_sampled_access_emission.cpp" _moi_sampled_access_owner)
string(
    REGEX MATCHALL
    "consan_detail::append_moi_indexed_address"
    _moi_sampled_indexed_address_calls
    "${_moi_sampled_access_owner}"
)
list(LENGTH _moi_sampled_indexed_address_calls _moi_sampled_indexed_address_call_count)
if(NOT _moi_sampled_indexed_address_call_count EQUAL 2)
    message(FATAL_ERROR
        "ConSan Sampled indexed and banked tables must share indexed addressing"
    )
endif()
file(READ "${_consan_dir}/consan_moi_sampled_atomic_emission.cpp" _moi_sampled_atomic_owner)
string(
    REGEX MATCHALL
    "consan_detail::append_moi_indexed_address"
    _moi_sampled_atomic_indexed_address_calls
    "${_moi_sampled_atomic_owner}"
)
list(LENGTH _moi_sampled_atomic_indexed_address_calls _moi_sampled_atomic_indexed_address_count)
if(NOT _moi_sampled_atomic_indexed_address_count EQUAL 6)
    message(FATAL_ERROR
        "ConSan Sampled atomic tables must share indexed addressing"
    )
endif()
file(READ "${_consan_dir}/consan_moi_sampled_sync.inc" _moi_sampled_sync_owner)
foreach(_sampled_owner IN ITEMS consan_moi_sampled.h consan_moi_sampled_access.inc consan_moi_sampled_sync.inc)
    _consan_assert_no_match(
        "${_consan_dir}/${_sampled_owner}"
        "MoiOptions|static_cast<const ConSan(Request|MoiOperatingPoint|BoundRuntimeResources)"
        "Sampled components must separate immutable input from operating-point state"
    )
endforeach()
string(
    REGEX MATCHALL
    "consan_detail::append_moi_indexed_address"
    _moi_sampled_sync_indexed_address_calls
    "${_moi_sampled_sync_owner}"
)
list(LENGTH _moi_sampled_sync_indexed_address_calls _moi_sampled_sync_indexed_address_count)
if(NOT _moi_sampled_sync_indexed_address_count EQUAL 5)
    message(FATAL_ERROR
        "ConSan Sampled barrier tables must share indexed addressing"
    )
endif()
string(
    REGEX MATCHALL
    "build_sampled_dense_sync_dispatcher"
    _moi_sampled_dense_sync_dispatcher_refs
    "${_moi_sampled_sync_owner}"
)
list(LENGTH _moi_sampled_dense_sync_dispatcher_refs _moi_sampled_dense_sync_dispatcher_ref_count)
if(NOT _moi_sampled_dense_sync_dispatcher_ref_count EQUAL 3)
    message(FATAL_ERROR
        "ConSan Sampled barrier and atomic relays must share one dense dispatcher"
    )
endif()
foreach(_recipe IN ITEMS
    "dispatcher_offset - island_offset"
    "append_restore_moi_scc_from_route_key"
)
    string(REGEX MATCHALL "${_recipe}" _moi_sampled_dense_recipe_refs "${_moi_sampled_sync_owner}")
    list(LENGTH _moi_sampled_dense_recipe_refs _moi_sampled_dense_recipe_ref_count)
    if(NOT _moi_sampled_dense_recipe_ref_count EQUAL 1)
        message(FATAL_ERROR
            "ConSan Sampled dense dispatcher recipe '${_recipe}' must have one owner"
        )
    endif()
endforeach()
file(READ "${_consan_dir}/consan_moi_shared_lowering.cpp" _moi_shared_lowering_owner)
string(
    REGEX MATCHALL
    "consan_detail::append_moi_indexed_address"
    _moi_record_replay_indexed_address_calls
    "${_moi_shared_lowering_owner}"
)
list(LENGTH _moi_record_replay_indexed_address_calls _moi_record_replay_indexed_address_call_count)
if(NOT _moi_record_replay_indexed_address_call_count EQUAL 2)
    message(FATAL_ERROR
        "ConSan RecordReplay access and dispatch tables must share indexed addressing"
    )
endif()
foreach(
    _record_replay_owner
    IN ITEMS
        consan_moi_record_replay.h
        consan_moi_record_replay.inc
        consan_moi_record_atomic.inc
        consan_moi_record_fence.inc
)
    _consan_assert_no_match(
        "${_consan_dir}/${_record_replay_owner}"
        "MoiOptions|static_cast<const ConSan(Request|MoiOperatingPoint|BoundRuntimeResources)"
        "Record/Replay components must separate immutable input from operating-point state"
    )
endforeach()
foreach(_sampling_mode_owner IN ITEMS consan_moi_record_replay.inc consan_moi_sampled_access.inc)
    _consan_assert_no_match(
        "${_consan_dir}/${_sampling_mode_owner}"
        "moi_(runtime_)?sample_stride[ \t]*==[ \t]*0|moi_(runtime_)?sample_offset[ \t]*>="
        "sampling validity must remain owned by the typed configuration-stage contract"
    )
endforeach()
foreach(
    _inline_shadow_owner
    IN ITEMS
        consan_moi_inline_shadow.h
        consan_moi_inline_shadow.inc
        consan_moi_inline_atomic.inc
)
    _consan_assert_no_match(
        "${_consan_dir}/${_inline_shadow_owner}"
        "MoiOptions|static_cast<const ConSan(Request|MoiOperatingPoint|BoundRuntimeResources)"
        "InlineShadow components must separate immutable input from operating-point state"
    )
endforeach()
foreach(_barrier_owner IN ITEMS consan_moi_barrier.h consan_moi_barrier.inc)
    _consan_assert_no_match(
        "${_consan_dir}/${_barrier_owner}"
        "MoiOptions|static_cast<const ConSan(Request|MoiOperatingPoint|BoundRuntimeResources)"
        "shared MOI barrier construction must separate immutable input from operating-point state"
    )
endforeach()
foreach(_prologue_owner IN ITEMS consan_moi_prologue.h consan_moi_prologue.cpp)
    _consan_assert_no_match(
        "${_consan_dir}/${_prologue_owner}"
        "MoiOptions|static_cast<const ConSan(Request|MoiOperatingPoint|BoundRuntimeResources)"
        "shared MOI prologue construction must separate immutable input from operating-point state"
    )
endforeach()
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_placement.inc"
    "GFX90A_ACCUM_OFFSET"
    "common placement must consume the normalized descriptor VGPR-allocation product"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_target_address.cpp"
    "ROCJITSU_CODE_ARCH_"
    "atomic-address materialization must consume typed target capabilities"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_fault_injection.inc"
    "ROCJITSU_CODE_ARCH_"
    "common fault mutation must derive encoding from its code-object target"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_fault_injection.inc"
    "sign_extend_24|words\\[0\\].*0xff|words\\[1\\].*18u|words\\[2\\]"
    "common atomic fault mutation must not edit target instruction bitfields"
)
file(READ "${_consan_dir}/consan_fault_injection.inc" _fault_target_operation_consumer)
if(NOT _fault_target_operation_consumer MATCHES "rewrite_consan_atomic_fault_address" OR
   NOT _fault_target_operation_consumer MATCHES "rewrite_consan_atomic_fault_scope_to_wave")
    message(FATAL_ERROR
        "ConSan atomic fault mutation lost its target-owned rewrite operations"
    )
endif()
file(READ "${_consan_dir}/consan_fault_gfx12_target_ops.cpp" _gfx12_fault_target_owner)
if(NOT _gfx12_fault_target_owner MATCHES "rewrite_gfx12_atomic_fault_address" OR
   NOT _gfx12_fault_target_owner MATCHES "rewrite_gfx12_atomic_fault_scope_to_wave" OR
   NOT _gfx12_fault_target_owner MATCHES "rdna4::VdsMachineInst" OR
   NOT _gfx12_fault_target_owner MATCHES "rdna4::VflatMachineInst" OR
   NOT _gfx12_fault_target_owner MATCHES "rdna4::VbufferMachineInst")
    message(FATAL_ERROR
        "ConSan gfx12 fault target owner lost a concrete atomic rewrite recipe"
    )
endif()
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_access_apply.h"
    "ROCJITSU_CODE_ARCH_"
    "common access application must consume typed target facts"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_placement.inc"
    "ROCJITSU_CODE_ARCH_"
    "common placement must consume typed target facts"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_evidence_planning.h"
    "ConSanTransformArtifacts"
    "evidence planning must consume immutable forward products, not the mutable transaction bus"
)
file(READ "${_consan_dir}/consan_moi.h" _moi_retry_contract)
if(NOT _moi_retry_contract MATCHES
       "struct ConSanMoiRetryInventory" OR
   NOT _moi_retry_contract MATCHES "ProgramInventory" OR
   NOT _moi_retry_contract MATCHES "ConSanCoverageLedger" OR
   _moi_retry_contract MATCHES
       "retry_patch_consan_moi_from_inventory[^;]*ConSanTransformArtifacts[ \t]+inventory")
    message(FATAL_ERROR
        "ConSan MOI resume must consume immutable inventory products, not a reconstructed transaction"
    )
endif()
file(READ "${_consan_dir}/consan_pipeline.h" _pipeline_result_contract)
if(_pipeline_result_contract MATCHES "take_lowering_artifacts" OR
   _pipeline_result_contract MATCHES
       "PrivateLoweringArtifacts[^}]*ConSanMoiOperatingPoint")
    message(FATAL_ERROR
        "ConSan pipeline result regained a reverse lowerer transaction or dead operating point"
    )
endif()
foreach(_resource_planning_owner IN ITEMS consan_moi_pipeline.h consan_moi_pipeline.inc)
    _consan_assert_no_match(
        "${_consan_dir}/${_resource_planning_owner}"
        "const[ \\t]+ConSanTransformArtifacts"
        "resource planning must consume its immutable problem, not the mutable transaction bus"
    )
    _consan_assert_no_match(
        "${_consan_dir}/${_resource_planning_owner}"
        "rebuild_moi_resource_plans"
        "resource planning must return a typed result for coordinator publication"
    )
endforeach()
foreach(_read_only_transaction_client IN ITEMS
    consan_sync_analysis.inc
    consan_placement.h
    consan_placement.inc
    consan_moi_shared_lowering.h
    consan_moi_shared_lowering.cpp
    consan_moi_barrier.inc
    consan_supercollider_common.inc
)
    _consan_assert_no_match(
        "${_consan_dir}/${_read_only_transaction_client}"
        "const[ \\t]+ConSanTransformArtifacts[ \\t]*&"
        "read-only helpers must consume immutable inventory, coverage, or patch products"
    )
endforeach()
foreach(_file IN LISTS _consan_production_files)
    _consan_assert_no_match(
        "${_file}"
        "active_moi_bytes"
        "the transformation transaction must own current candidate-image selection"
    )
endforeach()
file(READ "${_consan_dir}/consan_moi_shared_lowering.h" _moi_descriptor_mutation_contract)
if(NOT _moi_descriptor_mutation_contract MATCHES
       "apply_moi_descriptor_requirements[^;]*ProgramInventory" OR
   _moi_descriptor_mutation_contract MATCHES
       "apply_moi_descriptor_requirements[^;]*ConSanTransformArtifacts")
    message(FATAL_ERROR
        "ConSan descriptor mutation must consume immutable inventory, not the broad transaction"
    )
endif()
file(READ "${_consan_dir}/consan_placement.h" _common_placement_contract)
if(NOT _common_placement_contract MATCHES
       "reserved_ranges_for_existing_patches[^;]*span<const ConSanPatchInfo>")
    message(FATAL_ERROR
        "ConSan existing-patch reservation must consume the patch-proof span"
    )
endif()
# Normalized access operands define one scratch-allocation contract. The
# common placement owner supplies it to both LDS and SuperCollider FLAT
# lowering; a mode-local copy would allow tuple and operand exclusions to
# drift again.
file(READ "${_consan_dir}/consan_placement.inc" _access_scratch_owner)
file(READ "${_consan_dir}/consan_supercollider_support.h" _sc_support_contract)
file(READ "${_consan_dir}/consan_supercollider_support.cpp" _sc_support_body)
file(READ "${_consan_dir}/consan_supercollider_flat.inc" _sc_flat_body)
foreach(_operation IN ITEMS
    access_dword_count
    access_scratch_tuple_base_is_valid
    access_scratch_search_start
    choose_scratch_vgpr
    choose_spill_scratch_vgpr
)
    if(NOT _common_placement_contract MATCHES "${_operation}" OR
       NOT _access_scratch_owner MATCHES "${_operation}" OR
       NOT _sc_flat_body MATCHES "${_operation}")
        message(FATAL_ERROR
            "ConSan normalized access scratch operation lost its shared placement owner: ${_operation}"
        )
    endif()
endforeach()
if(_sc_support_contract MATCHES "flat_(dword_count|scratch)" OR
   _sc_support_body MATCHES "flat_(dword_count|scratch)|choose_flat_(scratch|spill)")
    message(FATAL_ERROR
        "ConSan SuperCollider support regained a mode-local access scratch contract"
    )
endif()
_consan_assert_no_match(
    "${_consan_dir}/consan_pipeline.cpp"
    "plan_consan_(record_replay|sampled|inline_shadow)_evidence"
    "the common pipeline must compose mode-owned evidence planning through the mode registry"
)
foreach(_file IN LISTS _consan_production_files)
    _consan_assert_no_match(
        "${_file}"
        "plan_consan_(record_replay|sampled|inline_shadow)_evidence"
        "the mode registry is the only production MOI evidence-planning entry point"
    )
endforeach()
foreach(_source IN ITEMS consan_moi_barrier.inc consan_moi_sync_emission.cpp)
    _consan_assert_no_match(
        "${_consan_dir}/${_source}"
        "find_kernel_by_(name|descriptor)|find_function_by_name|is_rocclr_runtime_kernel_name"
        "evidence kinds must consume the shared decoded-container projection"
    )
endforeach()
_consan_assert_no_match(
    "${_consan_dir}/consan_program_analysis_target_ops.cpp"
    "consan_program_analysis_target_detail::"
    "common program-analysis dispatch must consume the target-owned operations product"
)
_consan_assert_match_count_at_most(
    "${_consan_dir}/consan_program_analysis_target_ops.cpp"
    "ROCJITSU_CODE_ARCH_"
    5
    "program-analysis target selection must remain one narrow registry"
)

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

# Dense access routing resolves mode policy once at the registry boundary.
# Shared placement and emission consume the resulting typed plan and may not
# regain an engine enum or an InlineShadow-only scalar-ABI peephole.
foreach(_file IN LISTS _consan_production_files)
    _consan_assert_no_match(
        "${_file}"
        "MoiDenseAccessRouteAbi|MoiInlineDenseRouterScalarAbi|moi_inline_dense_router_scalar_abi|inline_shadow_route"
        "dense routing must consume the mode-published router plan"
    )
endforeach()
foreach(_mode IN ITEMS record_replay sampled inline_shadow)
    set(_mode_owner "${_consan_dir}/consan_moi_${_mode}.cpp")
    file(READ "${_mode_owner}" _mode_owner_contents)
    if(NOT _mode_owner_contents MATCHES "[.]dense_router[ 	]*=")
        message(FATAL_ERROR
            "ConSan ${_mode} lost its dense-router registry operation"
        )
    endif()
endforeach()
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_mode_planning.cpp"
    "ConSanEncodingFamily::|ROCJITSU_CODE_ARCH_"
    "common mode planning must consume normalized target capabilities"
)

# Generated ISA headers occur only in reviewed target normalization,
# family/member lowering, program analysis, or independent validation owners.
set(
    _generated_header_owners
    consan_fault_gfx12_target_ops.cpp
    consan_fault_gfx9_target_ops.cpp
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
    consan_validation_gfx12_target_ops.cpp
)
_consan_assert_no_match(
    "${_consan_dir}/consan_final_validation.cpp"
    "isa/arch/amdgpu/generated/|(cdna[0-9_]*|rdna[0-9_]*)::"
    "common final validation must consume independent target validation operations"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_validation.inc"
    "isa/arch/amdgpu/generated/|(cdna[0-9_]*|rdna[0-9_]*)::"
    "common validation implementation must consume independent target validation operations"
)
file(READ "${_consan_dir}/consan_validation.inc" _final_validation_owner)
if(NOT _final_validation_owner MATCHES "struct FinalValidationEnvironment" OR
   NOT _final_validation_owner MATCHES "struct ValidationText")
    message(FATAL_ERROR "ConSan final proof passes lost their immutable validation environment")
endif()
string(REGEX MATCHALL "Decoder::create" _final_validation_decoders "${_final_validation_owner}")
list(LENGTH _final_validation_decoders _final_validation_decoder_count)
string(
    REGEX MATCHALL
    "FinalValidationEnvironment environment"
    _final_validation_environments
    "${_final_validation_owner}"
)
list(LENGTH _final_validation_environments _final_validation_environment_count)
if(NOT _final_validation_decoder_count EQUAL 1 OR
   NOT _final_validation_environment_count EQUAL 1)
    message(
        FATAL_ERROR
        "ConSan final proof passes must share exactly one parse/target/decoder environment"
    )
endif()
_consan_assert_no_match(
    "${_consan_dir}/consan_validation_target_ops.h"
    "validate_consan_(ordinary_global|atomic)"
    "encoded mutation validation must remain one operation over a typed semantic kind"
)
_consan_assert_no_match(
    "${_consan_dir}/consan_validation.inc"
    "COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET"
    "common validation must consume target-owned descriptor resource proof"
)
file(
    GLOB _consan_sources
    "${_consan_dir}/*.cpp"
    "${_consan_dir}/*.h"
    "${_consan_dir}/*.inc"
)
foreach(_file IN LISTS _consan_sources)
    _consan_assert_no_match(
        "${_file}"
        "#include.*consan_lowerer[.]h|run_consan_lowering_core|install_consan_lowering_observation|apply_consan_fault_mutation_plans"
        "core callers must use the actual composition boundary without a forwarding facade"
    )
endforeach()
_consan_assert_no_match(
    "${_consan_dir}/consan_validation_inventory.h"
    "ConSanTransformArtifacts|ConSanPerturbationPlanningState"
    "independent validation inventory must expose only immutable proof facts"
)
file(READ "${_consan_dir}/consan_validation_inventory.h" _validation_inventory_contract)
if(NOT _validation_inventory_contract MATCHES "ConSanMutationValidationInventory" OR
   NOT _validation_inventory_contract MATCHES "ConSanPerturbationValidationInventory" OR
   NOT _validation_inventory_contract MATCHES "ConSanFaultSelectionView" OR
   NOT _validation_inventory_contract MATCHES "candidates")
    message(FATAL_ERROR "ConSan pristine validation lost its narrow proof inventories")
endif()
file(READ "${_consan_dir}/consan_validation_inventory.cpp" _validation_inventory_owner)
if(_validation_inventory_owner MATCHES "consan_composition[.]h|compose_consan_lowering" OR
   NOT _validation_inventory_owner MATCHES "analyze_consan_program_inventory" OR
   NOT _validation_inventory_owner MATCHES "fault_drop_barrier = true")
    message(
        FATAL_ERROR
        "ConSan validation must rederive its two semantic inventories below composition"
    )
endif()
foreach(_file IN LISTS _consan_sources)
    _consan_assert_no_match(
        "${_file}"
        "ConSanMoiLiteralDispatchIdPolicy|ConSanMoiReportDispatchIdWordSource|ConSanMoiReportDispatchIdSources|moi_permits_literal_dispatch_identity|moi_report_dispatch_id_sources|moi_report_dispatch_id_source_permitted"
        "dispatch identity must be authorized once as one inseparable source-planning product"
    )
endforeach()
_consan_assert_no_match(
    "${_consan_dir}/consan_moi_runtime_workgroup_gate.cpp"
    "ConSanMoiEngine"
    "the shared Record/Replay-Sampled runtime gate must consume its exact-subset flavor"
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
if(_capability_contract_contents MATCHES
       "ConSanKernelTargetProfile|consan_kernel_target_profile|consan_profile_supports_wave_size|consan_profile_vgpr_allocation_granularity|consan_is_capability_target|consan_uses_gfx12_rdna_execution|consan_arch_has_s_call_b64|consan_arch_has_descriptor_partitioned_accumulators")
    message(
        FATAL_ERROR
        "ConSan target contract regained an unused projection or test-only profile predicate"
    )
endif()
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

# Temporary physical-GPU qualification telemetry must not return as a hidden
# report-ABI mode. The accepted InlineShadow path and ordinary report decoder
# now cover the behavior that the side channel was created to investigate.
foreach(_file IN LISTS _consan_production_files _hook_sources)
    _consan_assert_no_match(
        "${_file}"
        "moi_partition_mask_debug|RJ_CONSAN_MOI_PARTITION_MASK_DEBUG"
        "retired partition-mask qualification telemetry must remain deleted"
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

# Accepted lowering transactions have one owner. The transformation bus must
# not regain a synchronization staging inventory, parallel committed-lowering
# inventory, or accumulated runtime projection.
foreach(_file IN LISTS _consan_production_files)
    _consan_assert_no_match(
        "${_file}"
        "committed_lowerings"
        "accepted lowering transactions must remain owned by the coverage ledger"
    )
    _consan_assert_no_match(
        "${_file}"
        "staged_moi_sync_lowerings"
        "synchronization lowering must coalesce at the coverage-ledger owner"
    )
    _consan_assert_no_match(
        "${_file}"
        "publish_moi_sync_lowering_commits"
        "synchronization bytes and semantic commits must publish as one transaction"
    )
    _consan_assert_no_match(
        "${_file}"
        "publish_moi_access_patch"
        "all engines must publish completed access lowering through the transform transaction owner"
    )
    if(NOT _file MATCHES
       "(consan_access_policy.cpp|consan_observation_plan.h.inc|consan_result.h.inc|consan_pipeline.cpp)$"
    )
        _consan_assert_no_match(
            "${_file}"
            "publish_lowering_commits"
            "ordinary lowering commits may publish only through their coverage owner or the access transaction"
        )
    endif()
    if(NOT _file MATCHES
       "(consan_access_policy.cpp|consan_observation_plan.h.inc|consan_moi_sync_emission.cpp)$"
    )
        _consan_assert_no_match(
            "${_file}"
            "publish_coalescing_instrumented_commits"
            "only the shared synchronization transaction may publish coalescing commits"
        )
    endif()
endforeach()
_consan_assert_no_match(
    "${_consan_dir}/consan_supercollider_lds.inc"
    "result[.](patches|replacement|mark_modified)|discard_candidate_modification|publish_lowering_commits"
    "SuperCollider LDS access construction must publish local bytes and proof through the shared access transaction"
)
foreach(
    _access_mode_source
    IN ITEMS
        consan_moi_record_replay.inc
        consan_moi_sampled_access.inc
        consan_moi_inline_shadow.inc
)
    _consan_assert_no_match(
        "${_consan_dir}/${_access_mode_source}"
        "publish_lowering_commits|result[.]replacement|result[.]mark_modified|discard_candidate_modification"
        "mode-local access emission must publish bytes, proof, and lowering through the shared access transaction"
    )
endforeach()
foreach(_result_bus IN ITEMS consan_result.h.inc consan_pipeline.h)
    _consan_assert_no_match(
        "${_consan_dir}/${_result_bus}"
        "ConSanRuntimeStaticMapping[ \t]+runtime_static_mapping[ \t]*;"
        "result buses must derive runtime attribution from committed lowering"
    )
endforeach()
_consan_assert_no_match(
    "${_consan_dir}/consan_observation_plan.h.inc"
    "record_replay_accesses|sampled_accesses|inline_compact_accesses"
    "runtime static attribution must not regain parallel mode containers"
)
file(READ "${_consan_dir}/consan_observation_plan.h.inc" _runtime_mapping_contract)
if(NOT _runtime_mapping_contract MATCHES
   "std::variant<std::monostate, RecordReplay, Sampled, InlineCompact>"
)
    message(FATAL_ERROR "ConSan runtime static attribution lost its discriminated mode product")
endif()
set(_report_pipeline_contract
    "${_hook_dir}/rj_hsa_dbi_moi_report_pipeline.h"
)
_consan_assert_no_match(
    "${_report_pipeline_contract}"
    "record_replay_static_mappings|sampled_static_mappings|compact_token_mapping_count|compact_token_mapping_malformed|record_replay_static_mapping_malformed|sampled_static_mapping_malformed"
    "runtime report input must not regain parallel mode metadata fields"
)
file(READ "${_report_pipeline_contract}" _report_pipeline_contract_contents)
if(NOT _report_pipeline_contract_contents MATCHES
   "const AutoMoiRuntimeStaticMetadata \\*static_metadata"
)
    message(FATAL_ERROR "ConSan runtime report input lost its discriminated mode metadata")
endif()
file(READ "${_hook_dir}/rj_hsa_dbi_hook_moi_report.cpp" _report_registry_contract_contents)
if(NOT _report_registry_contract_contents MATCHES
   "AutoMoiRuntimeStaticMetadata static_metadata;"
)
    message(FATAL_ERROR "ConSan runtime report registry lost its shared mode metadata product")
endif()
_consan_assert_no_match(
    "${_hook_dir}/rj_hsa_dbi_hook_moi_report.cpp"
    "entry->(record_replay_static_mappings|sampled_static_mappings|compact_token_mapping_count|compact_token_mapping_malformed|record_replay_static_mapping_malformed|sampled_static_mapping_malformed)"
    "runtime report registry must not regain parallel mode metadata fields"
)
file(READ "${_consan_dir}/consan_observation_plan.h.inc" _coverage_ledger_contract)
foreach(
    _owned_lowering_operation
    IN ITEMS
        observation_plan_
        lowering_commits_
        runtime_static_mapping
        publish_coalescing_instrumented_commits
        discard_instrumented_lowerings
)
    if(NOT _coverage_ledger_contract MATCHES "${_owned_lowering_operation}")
        message(
            FATAL_ERROR
            "ConSan coverage ledger lost lowering ownership: ${_owned_lowering_operation}"
        )
    endif()
endforeach()
foreach(_parallel_plan_holder IN ITEMS consan_result.h.inc consan_pipeline.h)
    _consan_assert_no_match(
        "${_consan_dir}/${_parallel_plan_holder}"
        "ConSanObservationPlan[ \t]+observation_plan[ \t]*;"
        "immutable observation policy must remain owned once by the coverage ledger"
    )
endforeach()
_consan_assert_no_match(
    "${_consan_dir}/consan_observation_plan.h.inc"
    "matches_plan"
    "the coverage ledger must not regain a second plan representation to compare"
)
foreach(
    _parallel_decision_inventory
    IN ITEMS site_decisions_ barrier_site_decisions_ atomic_site_decisions_ fence_site_decisions_
)
    _consan_assert_no_match(
        "${_consan_dir}/consan_observation_plan.h.inc"
        "${_parallel_decision_inventory}"
        "semantic decisions must remain in the ledger-owned observation plan"
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
if(NOT _fault_application_contract MATCHES
   "apply_consan_fault_mutations[^;]*ConSanPatchedImageGrowthLimit[^;]*ConSanFaultMutationPlan" OR
   _fault_application_contract MATCHES "ConSanOptions")
    message(FATAL_ERROR
        "ConSan fault application lost its one complete typed-plan transaction"
    )
endif()
file(READ "${_consan_dir}/consan_placement.h" _placement_contract)
if(NOT _placement_contract MATCHES
   "find_uncovered_nop_caves[^;]*ProgramInventory" OR
   _placement_contract MATCHES
   "find_uncovered_nop_caves[^;]*ConSanTransformArtifacts")
    message(FATAL_ERROR
        "ConSan local-cave discovery must consume immutable program inventory"
    )
endif()
set(
    _private_fault_mechanisms
    try_apply_barrier_drop_fault_patch
    try_apply_barrier_move_fault_patch
    try_apply_barrier_id_scope_fault_patch
    try_apply_barrier_participant_fault_patch
    try_apply_atomic_fault_patch
    try_apply_lds_fault_patch
    try_apply_ordinary_fault_patch
)
foreach(_private_fault_mechanism IN LISTS _private_fault_mechanisms)
    if(_fault_application_contract MATCHES "${_private_fault_mechanism}")
        message(FATAL_ERROR
            "ConSan fault mechanism leaked through the component contract: ${_private_fault_mechanism}"
        )
    endif()
endforeach()
file(READ "${_consan_dir}/consan_composition.inc" _fault_composition_body)
if(_fault_composition_body MATCHES "find_fault_plan" OR
   _fault_composition_body MATCHES "try_apply_.*fault_patch")
    message(FATAL_ERROR
        "ConSan composition must not rediscover or dispatch private fault mechanisms"
    )
endif()

file(READ "${_consan_dir}/consan_growth_policy.h" _growth_policy_contract)
if(_growth_policy_contract MATCHES "ConSanOptions" OR
   _growth_policy_contract MATCHES "ConSanTransformArtifacts" OR
   NOT _growth_policy_contract MATCHES "ConSanPatchedImageGrowthLimit")
    message(FATAL_ERROR
        "ConSan image-growth policy must consume narrow identity, policy, and diagnostic products"
    )
endif()
file(READ "${_consan_dir}/consan_fault_injection.inc" _fault_application_body)
if(NOT _fault_application_body MATCHES "struct FaultApplicationState" OR
   NOT _fault_application_body MATCHES "FaultApplicationState transaction")
    message(FATAL_ERROR
        "ConSan fault mechanisms lost their private complete-plan candidate transaction"
    )
endif()
foreach(_private_fault_mechanism IN LISTS _private_fault_mechanisms)
    if(NOT _fault_application_body MATCHES "static void ${_private_fault_mechanism}" OR
       NOT _fault_application_body MATCHES
       "${_private_fault_mechanism}\\([^\\{]*FaultApplicationState &result\\)")
        message(FATAL_ERROR
            "ConSan fault mechanism lost internal linkage or bypasses the candidate transaction: ${_private_fault_mechanism}"
        )
    endif()
    if(_fault_application_body MATCHES
       "${_private_fault_mechanism}\\([^\\{]*ConSanTransformArtifacts")
        message(FATAL_ERROR
            "ConSan private fault mechanism regained the broad transform bus: ${_private_fault_mechanism}"
        )
    endif()
endforeach()

_consan_assert_no_match(
    "${_consan_dir}/consan_moi_dynamic_record_emission.h"
    "ConSanMoiOperatingPoint|BoundRuntimeResources"
    "dynamic record emission must consume planned dispatch-identity sources"
)
_consan_assert_match_count_at_most(
    "${_consan_dir}/consan_moi_report_emission.h"
    "ConSanMoiOperatingPoint"
    1
    "report emission may inspect the operating point only at its source-planning boundary"
)

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
_consan_assert_no_match(
    "${_hook_dir}/rj_hsa_dbi_moi_report_pipeline.h"
    "fence_record_capacity|direct_sampled|inline_shadow"
    "runtime report input must carry one complete typed layout"
)
_consan_assert_no_match(
    "${_hook_dir}/rj_hsa_dbi_moi_report_decoder.cpp"
    "input[.](fence_record_capacity|direct_sampled|inline_shadow)"
    "runtime report decoding must derive mode and capacities from the layout"
)
_consan_assert_no_match(
    "${_hook_dir}/rj_hsa_dbi_hook_moi_report.cpp"
    "entry[.](access_record_capacity|barrier_record_capacity|atomic_record_capacity|fence_record_capacity|diagnostic_capacity|exact_shadow_entry_capacity|inline_atomic_release_capacity|inline_acquired_epoch_token_capacity|inline_causal_snapshot_capacity|sampled_watchpoint_capacity|direct_sampled|inline_shadow)"
    "runtime report registry entries must retain one layout authority"
)

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
    consan_moi_placement.inc 0 "shared immutable resource constraint solver"
)
_consan_assert_reviewed_mode_switch_budget(
    consan_moi.cpp 0 "mode-neutral forward-only MOI coordinator"
)
_consan_assert_reviewed_mode_switch_budget(
    consan_moi_report_plan.cpp 0 "mode-neutral report and evidence composition"
)
_consan_assert_reviewed_mode_switch_budget(
    consan_moi_barrier.inc 0 "evidence-operation-driven shared barrier component"
)
_consan_assert_reviewed_mode_switch_budget(
    consan_moi_pipeline.inc 0 "semantic-operation resource-plan orchestration"
)
_consan_assert_reviewed_mode_switch_budget(
    consan_moi_support.cpp 0 "mode-neutral scalar ABI dispatch"
)
_consan_assert_reviewed_mode_switch_budget(
    consan_types.cpp 6 "mode parsing naming and request validation"
)
_consan_assert_reviewed_mode_switch_budget(
    consan_moi_prologue.cpp 0 "mode-neutral MOI entry-state construction"
)
_consan_assert_reviewed_mode_switch_budget(
    consan_moi_model.cpp 4 "mode-owned replay diagnostic provenance constants"
)

message(STATUS "ConSan architectural boundary checks passed")
