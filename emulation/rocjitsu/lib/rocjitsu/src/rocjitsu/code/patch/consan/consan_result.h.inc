struct ConSanPerturbationCandidate {
  ConSanPerturbationKind kind = ConSanPerturbationKind::None;
  ConSanPerturbationEdge edge = ConSanPerturbationEdge::Release;
  std::string identity;
  std::string sequence_identity;
  std::string container_name;
  bool in_kernel = true;
  uint32_t basic_block_index = 0;
  std::string anchor_event_identity;
  uint64_t anchor_text_offset = 0;
  uint32_t anchor_size = 0;
  std::vector<std::string> ordered_member_identities;
  bool eligible = false;
  std::string rejection_reason;
};

struct ConSanPerturbationPlan {
  std::string candidate_identity;
  std::string sequence_identity;
  ConSanPerturbationKind kind = ConSanPerturbationKind::None;
  ConSanPerturbationEdge edge = ConSanPerturbationEdge::Release;
  std::string anchor_event_identity;
  uint64_t anchor_text_offset = 0;
  uint32_t anchor_size = 0;
  uint32_t sleep_imm = 0;
  std::string source_candidate_identity;
  std::string source_sequence_identity;
  std::string source_container_name;
  std::string source_anchor_event_identity;
  bool source_in_kernel = true;
  uint64_t source_anchor_text_offset = 0;
  uint32_t source_anchor_size = 0;
  std::optional<uint64_t> source_owner_descriptor_file_offset;
  bool overlaps_atomic_mutation = false;
  bool removed_cache_boundary = false;
};

/// A contiguous static barrier lifecycle. On GFX12.5 `s_barrier_leave` has no
/// operand and drops the last named barrier joined by the thread, so a group
/// is admissible only when a matching static join proves that association.
struct ConSanBarrierLifecycleGroup {
  std::string identity;
  std::string container_name;
  bool in_kernel = true;
  std::optional<uint32_t> basic_block_index;
  uint64_t begin_text_offset = 0;
  uint64_t end_text_offset = 0;
  std::vector<std::string> member_event_identities;
  std::optional<int32_t> barrier_id;
  ConSanBarrierSite::Scope barrier_scope = ConSanBarrierSite::Scope::Unknown;
  bool admissible = false;
  ConSanSemanticConfidence confidence = ConSanSemanticConfidence::Unsupported;
  std::string rejection_reason;
};

/// One canonical physical native-LDS access in SuperCollider's resolved
/// planning scope. The hook consumes this ledger instead of reconstructing
/// applicability from aliased kernel inventories that may lack a complete
/// all-owner CFG/liveness context.
struct ConSanScLdsCoverageSite {
  uint64_t file_offset = 0;
  bool supported = false;
};

struct ConSanResult {
  bool visited_code_object = false;
  bool modified = false;
  bool final_validation_passed = false;
  /// Both the pristine-to-mutation and mutation-to-instrumentation images were
  /// independently structurally and semantically validated before their patch
  /// inventories were composed.
  bool staged_composition_validated = false;
  /// Internal second-stage proof over a mutation image using a pristine plan.
  bool carried_composite_instrumentation_stage = false;
  ConSanTransformOutcome outcome = ConSanTransformOutcome::Unchanged;
  ConSanFlavor flavor = ConSanFlavor::None;
  ConSanMoiEngine moi_engine = ConSanMoiEngine::RecordReplay;
  /// Original barrier instructions in Sampled sequences that can order at
  /// least one selected LDS causal window. Raw inventory may contain earlier
  /// barriers that are not applicable to the sampled LDS analysis.
  uint32_t sampled_barrier_applicable_event_count = 0;
  std::optional<uint16_t> resolved_moi_owner_vgpr;
  std::optional<uint16_t> resolved_moi_epoch_vgpr;
  std::optional<uint16_t> resolved_moi_workgroup_key_vgpr;
  ConSanMoiPersistentWorkgroupRegisters resolved_moi_record_replay_workgroup_vgprs;
  std::vector<ConSanMoiPersistentVgprAssignment> resolved_moi_persistent_vgpr_assignments;
  std::optional<uint16_t> resolved_moi_persistent_owner_sgpr;
  std::optional<uint16_t> resolved_moi_persistent_epoch_sgpr;
  std::optional<uint16_t> resolved_moi_persistent_workgroup_key_sgpr;
  ConSanMoiPersistentWorkgroupRegisters resolved_moi_record_replay_workgroup_sgprs;
  std::vector<ConSanMoiPrologueScratchVgprAssignment>
      resolved_moi_prologue_scratch_vgpr_assignments;
  std::optional<uint16_t> resolved_moi_exec_save_sgpr;
  std::vector<ConSanMoiTransientSgprAssignment> resolved_moi_transient_sgpr_assignments;
  std::optional<uint16_t> resolved_moi_owner_sgpr;
  std::optional<uint16_t> resolved_moi_dispatch_id_sgpr;
  /// Original owner-descriptor offsets whose active replacement descriptors
  /// were explicitly expanded to carry the complete RDNA workgroup-ID launch
  /// payload. Final validation resolves these identities after text growth.
  std::vector<uint64_t> moi_full_workgroup_id_descriptor_offsets;
  uint64_t moi_report_dispatch_id = 0;
  bool moi_persistent_vgprs_automatic = false;
  bool moi_persistent_sgprs_automatic = false;
  bool moi_private_epoch_automatic = false;
  bool moi_exec_save_sgprs_automatic = false;
  bool moi_owner_sgpr_automatic = false;
  bool moi_dispatch_id_sgprs_automatic = false;
  size_t input_size = 0;
  std::string input_fingerprint;
  /// Immutable semantic options captured before the first MOI planning pass.
  /// A report-buffer retry derives its options from this snapshot and can
  /// change only ConSanMoiReportRetryConfig fields.
  std::optional<ConSanOptions> moi_inventory_options;
  /// Resolved target architecture.  Production transforms set this before
  /// candidate classification.  The hook rejects candidate-bearing results
  /// that retain the invalid default before coverage or require-patch
  /// classification.
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  /// Set immediately before architecture-dependent decoding and semantic
  /// classification begin. Parse-only results, including unsupported targets,
  /// leave this false even though they retain section and symbol inventory.
  /// Every later result stage is copied from that classified result, so new
  /// semantic fields added after the transition inherit this invariant without
  /// extending a hand-maintained list of members.
  bool semantic_arch_required = false;
  std::string target_name;
  /// Display/logging projection only.  Semantic code must branch on arch.
  std::string arch_display_name;
  std::vector<ConSanTextSection> text_sections;
  std::vector<ConSanKernelInfo> kernels;
  std::vector<ConSanFunctionInfo> functions;
  std::vector<ConSanFaultSite> fault_sites;
  std::vector<ConSanBarrierMoveDestination> barrier_move_destinations;
  std::vector<ConSanFaultMutationPlan> fault_plans;
  std::vector<ConSanSyncEvent> sync_events;
  std::vector<ConSanSyncSequence> sync_sequences;
  std::vector<ConSanCommunicationAddressRecipe> communication_address_recipes;
  std::vector<ConSanMoiFenceCandidate> moi_fence_candidates;
  std::vector<ConSanPerturbationCandidate> perturbation_candidates;
  std::vector<ConSanPerturbationPlan> perturbation_plans;
  size_t requested_perturbations = 0;
  size_t planned_perturbations = 0;
  size_t applied_perturbations = 0;
  std::vector<ConSanBarrierLifecycleGroup> barrier_lifecycle_groups;
  size_t requested_fault_mutations = 0;
  size_t planned_fault_mutations = 0;
  size_t applied_fault_mutations = 0;
  /// Stable pristine logical identity retained across staged mutation plus
  /// instrumentation composition. Empty when no mutation is committed.
  std::optional<std::string> applied_fault_logical_identity;
  std::vector<ConSanMoiCandidate> moi_candidates;
  std::vector<ConSanSiteDispositionRecord> site_dispositions;
  /// True when sc_lds_coverage_sites is the authoritative physical native-LDS
  /// denominator for this transform. Independently planned FLAT sites remain
  /// outside this ledger.
  bool sc_lds_coverage_resolved = false;
  std::vector<ConSanScLdsCoverageSite> sc_lds_coverage_sites;
  /// Warning boundary immediately before MOI planning. A validated retry with
  /// a runtime-allocated report buffer trims only the prior MOI-stage
  /// diagnostics while retaining parser and semantic-analysis warnings.
  size_t moi_stage_warning_begin = 0;
  std::vector<ConSanCandidateResourcePlan> resource_plans;
  ConSanResourcePlanSummary resource_plan_summary;
  std::vector<ConSanAccessPlan> access_plans;
  std::optional<ConSanCompositeProof> composite_proof;
  std::vector<ConSanPatchInfo> patches;
  std::vector<uint8_t> elf_bytes;
  std::vector<std::string> errors;
  std::vector<std::string> warnings;
};

/// Whether every architecture-dependent result stage has a typed architecture.
/// Parse-only results are safe to pass through without one.
[[nodiscard]] inline bool consan_result_has_resolved_semantic_arch(const ConSanResult &result) {
  return !result.semantic_arch_required || result.arch != ROCJITSU_CODE_ARCH_INVALID;
}

/// Whether a semantic claim is at least as strong as the weakest confidence a
/// consumer explicitly accepts.
[[nodiscard]] constexpr bool
consan_sync_confidence_meets(ConSanSemanticConfidence actual,
                             ConSanSemanticConfidence weakest_accepted) {
  return static_cast<uint8_t>(actual) <= static_cast<uint8_t>(weakest_accepted);
}

/// Finds the unique logical sequence containing a stable decoded event.
/// Returns null when the event is absent or appears in more than one sequence.
[[nodiscard]] const ConSanSyncSequence *
find_consan_sync_sequence_for_event(const ConSanResult &result, std::string_view event_identity);

[[nodiscard]] const char *consan_flavor_name(ConSanFlavor flavor);
[[nodiscard]] const char *consan_moi_engine_name(ConSanMoiEngine engine);
[[nodiscard]] const char *consan_transform_outcome_name(ConSanTransformOutcome outcome);
[[nodiscard]] const char *consan_resource_site_kind_name(ConSanResourceSiteKind kind);
[[nodiscard]] const char *
consan_register_allocation_source_name(ConSanRegisterAllocationSource source);
[[nodiscard]] const char *consan_register_plan_reason_name(ConSanRegisterPlanReason reason);
[[nodiscard]] const char *consan_delay_mode_name(ConSanDelayMode mode);
[[nodiscard]] const char *
consan_barrier_operand_source_name(ConSanBarrierSite::OperandSource source);
[[nodiscard]] const char *consan_barrier_scope_name(ConSanBarrierSite::Scope scope);
[[nodiscard]] const char *consan_moi_candidate_source_name(ConSanMoiCandidateSource source);
[[nodiscard]] const char *consan_lds_access_kind_name(ConSanLdsAccessKind kind);
[[nodiscard]] const char *consan_flat_address_space_hint_name(ConSanFlatAddressSpaceHint hint);
[[nodiscard]] const char *consan_site_disposition_name(ConSanSiteDisposition disposition);
[[nodiscard]] const char *consan_site_disposition_reason_name(ConSanSiteDispositionReason reason);
[[nodiscard]] const char *consan_site_lowering_outcome_name(ConSanSiteLoweringOutcome outcome);
[[nodiscard]] const char *consan_site_lowering_reason_name(ConSanSiteLoweringReason reason);
[[nodiscard]] ConSanInstallAction consan_install_action(const ConSanResult &result,
                                                        bool fail_closed);

[[nodiscard]] std::optional<ConSanFlavor> parse_consan_flavor(std::string_view value);
[[nodiscard]] std::optional<ConSanMoiEngine> parse_consan_moi_engine(std::string_view value);

/// @brief Try to apply ConSan instrumentation to @p code_object_bytes.
///
/// @details The early DBI slices parse and inventory the code object, return
/// `modified=false`, and leave `elf_bytes` empty so callers continue loading the
/// original code object.
[[nodiscard]] ConSanResult try_patch_consan(std::span<const uint8_t> code_object_bytes,
                                            const ConSanOptions &options);

/// Re-run the structural final-ELF validation used before a modified result is
/// made installable. Exposed for bounded mutation/property tests.
[[nodiscard]] std::vector<std::string>
validate_consan_modified_elf(std::span<const uint8_t> original_bytes,
                             const ConSanResult &modified_result);
