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

enum class ConSanScAccessCoverageKind : uint8_t {
  NativeLds,
  FlatGroup,
};

/// One canonical physical access in SuperCollider's resolved planning scope.
/// The hook consumes this ledger instead of reconstructing applicability from
/// aliased kernel inventories that may lack a complete all-owner CFG/liveness
/// context.
struct ConSanScAccessCoverageSite {
  ConSanScAccessCoverageKind kind = ConSanScAccessCoverageKind::NativeLds;
  uint64_t file_offset = 0;
  /// True after the matching planner has evaluated this physical site.
  /// Unevaluated sites remain fail-closed for require-patch policy.
  bool evaluated = false;
  bool supported = false;
};

/// Machine-readable resource-policy rejection retained alongside the human
/// diagnostic. Required bytes are total growth from the original input;
/// transaction bytes are the exact alignment-inclusive cost of this stage.
struct ConSanPatchedImageGrowthRejection {
  std::string operation;
  ConSanPatchedImageGrowthLimit policy;
  size_t input_image_bytes = 0;
  size_t existing_growth_bytes = 0;
  size_t transaction_growth_bytes = 0;
  size_t required_total_growth_bytes = 0;
  size_t limit_bytes = 0;
};

/// Aggregate branch-only routing activity. Pair attempts and route-failure
/// counters count submitted and rejected request pairs, respectively. Plan
/// calls count router invocations, including LDS convergence iterations.
/// Fallback counters count pairs that reached the corresponding tier, whether
/// or not that tier produced a route. Greedy fallback attempts are a subset of
/// exact-pair fallback attempts; the two counters must not be summed.
/// Work-limit exhaustion counts plan calls whose qualification, routing,
/// setup, or greedy tier reached its configured limit, including calls
/// recovered by fallback. Qualification and routing exhaustion are also
/// counted separately; one call may increment both phase counters. The work
/// counters may remain zero when an initial setup precharge exceeds the whole
/// allowance. Pristine occupancy rejections count otherwise valid offers that
/// the placement planner filtered before routing. Greedy fallback is owner-
/// oblivious and therefore identifies pairs whose retained route has no exact
/// minimization guarantee, even when an earlier exact-pair minimization ran
/// before an independent removal limit forced fallback. Reservation failures
/// count failed reservations and commits. Unavailable pristine relay storage
/// is excluded during qualification, so it contributes to the appropriate
/// unreachable route counter instead of a reservation failure. LDS convergence
/// iterations remain in this aggregate; a losing FLAT direct-reservoir retry
/// is moved to `discarded_branch_only_routing`.
///
/// Route-optimization exhaustion separately counts plan calls in which at
/// least one bounded lexicographic minimization pass reached its own limit and
/// retained its best known feasible route, which may be the feasibility
/// baseline or a partially proven improvement. The objective minimizes newly
/// activated owner groups first and relay count for capped exact-pair
/// fallback, including inventories with no owner tags. Its search/scan work
/// has separate counters so routing work remains comparable with routing
/// limits. Search counters measure charged work units, not physical relay
/// inspections: a mixed owner inventory may perform one uncharged second cost
/// tier, so physical inspections are at most twice the charged traversal
/// units for both feasibility and optimization.
/// Routing invariant failures count exact tiers abandoned after defensive
/// validation rejected inconsistent solver inputs or output. A later
/// exact-pair or greedy tier may recover the plan.
/// Route-optimization invariant failures count defensive fallbacks that retain
/// the feasibility route after detecting inconsistent minimizer inputs. They
/// are distinct from work-budget exhaustion because raising a limit cannot
/// repair them. Excess relay claims count only relays retained above the first
/// exact feasibility route while improving the primary owner-group objective.
///
/// The compatibility search/scan totals remain stable. All search work is
/// feasibility work. The scan phase counters expose the composition of scan
/// work, while route-optimization work remains outside both totals.
struct ConSanBranchOnlyRoutingTelemetry {
  size_t pair_attempt_count = 0;
  size_t plan_call_count = 0;
  size_t entry_route_failure_count = 0;
  size_t return_route_failure_count = 0;
  size_t relay_contention_failure_count = 0;
  size_t work_budget_failure_count = 0;
  size_t work_budget_exhaustion_count = 0;
  size_t relay_qualification_exhaustion_count = 0;
  size_t routing_work_exhaustion_count = 0;
  size_t routing_invariant_failure_count = 0;
  size_t route_optimization_exhaustion_count = 0;
  size_t route_optimization_invariant_failure_count = 0;
  size_t pristine_relay_occupancy_rejection_count = 0;
  size_t route_optimization_excess_relay_claim_count = 0;
  size_t reservation_failure_count = 0;
  size_t exact_pair_fallback_attempt_count = 0;
  size_t greedy_pair_fallback_attempt_count = 0;
  size_t search_work_count = 0;
  size_t scan_work_count = 0;
  size_t route_optimization_search_work_count = 0;
  size_t route_optimization_scan_work_count = 0;
  size_t relay_qualification_work_count = 0;
  size_t fallback_setup_work_count = 0;
  size_t feasibility_scan_work_count = 0;

  bool operator==(const ConSanBranchOnlyRoutingTelemetry &) const = default;
};

struct ConSanBranchOnlyRoutingTelemetryField {
  std::string_view log_name;
  size_t ConSanBranchOnlyRoutingTelemetry::*member = nullptr;
};

inline constexpr std::array kConSanBranchOnlyRoutingTelemetryFields = {
    ConSanBranchOnlyRoutingTelemetryField{"pair_attempts",
                                          &ConSanBranchOnlyRoutingTelemetry::pair_attempt_count},
    ConSanBranchOnlyRoutingTelemetryField{"plan_calls",
                                          &ConSanBranchOnlyRoutingTelemetry::plan_call_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "entry_route_failed", &ConSanBranchOnlyRoutingTelemetry::entry_route_failure_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "return_route_failed", &ConSanBranchOnlyRoutingTelemetry::return_route_failure_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "relay_contention_failed",
        &ConSanBranchOnlyRoutingTelemetry::relay_contention_failure_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "work_budget_failed", &ConSanBranchOnlyRoutingTelemetry::work_budget_failure_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "work_budget_exhaustions", &ConSanBranchOnlyRoutingTelemetry::work_budget_exhaustion_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "relay_qualification_exhaustions",
        &ConSanBranchOnlyRoutingTelemetry::relay_qualification_exhaustion_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "routing_work_exhaustions",
        &ConSanBranchOnlyRoutingTelemetry::routing_work_exhaustion_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "routing_invariant_failures",
        &ConSanBranchOnlyRoutingTelemetry::routing_invariant_failure_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "route_optimization_exhaustions",
        &ConSanBranchOnlyRoutingTelemetry::route_optimization_exhaustion_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "route_optimization_invariant_failures",
        &ConSanBranchOnlyRoutingTelemetry::route_optimization_invariant_failure_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "pristine_relay_occupancy_rejections",
        &ConSanBranchOnlyRoutingTelemetry::pristine_relay_occupancy_rejection_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "route_optimization_excess_relay_claims",
        &ConSanBranchOnlyRoutingTelemetry::route_optimization_excess_relay_claim_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "reservation_failed", &ConSanBranchOnlyRoutingTelemetry::reservation_failure_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "exact_pair_fallback_attempts",
        &ConSanBranchOnlyRoutingTelemetry::exact_pair_fallback_attempt_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "greedy_pair_fallback_attempts",
        &ConSanBranchOnlyRoutingTelemetry::greedy_pair_fallback_attempt_count},
    ConSanBranchOnlyRoutingTelemetryField{"search_work",
                                          &ConSanBranchOnlyRoutingTelemetry::search_work_count},
    ConSanBranchOnlyRoutingTelemetryField{"scan_work",
                                          &ConSanBranchOnlyRoutingTelemetry::scan_work_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "route_optimization_search_work",
        &ConSanBranchOnlyRoutingTelemetry::route_optimization_search_work_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "route_optimization_scan_work",
        &ConSanBranchOnlyRoutingTelemetry::route_optimization_scan_work_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "relay_qualification_work",
        &ConSanBranchOnlyRoutingTelemetry::relay_qualification_work_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "fallback_setup_work", &ConSanBranchOnlyRoutingTelemetry::fallback_setup_work_count},
    ConSanBranchOnlyRoutingTelemetryField{
        "feasibility_scan_work", &ConSanBranchOnlyRoutingTelemetry::feasibility_scan_work_count},
};

static_assert(kConSanBranchOnlyRoutingTelemetryFields.size() == 24u);
static_assert(sizeof(ConSanBranchOnlyRoutingTelemetry) ==
                  kConSanBranchOnlyRoutingTelemetryFields.size() * sizeof(size_t),
              "add every routing counter to the typed telemetry field table");
static_assert([] {
  for (size_t lhs = 0u; lhs < kConSanBranchOnlyRoutingTelemetryFields.size(); ++lhs) {
    for (size_t rhs = lhs + 1u; rhs < kConSanBranchOnlyRoutingTelemetryFields.size(); ++rhs) {
      if (kConSanBranchOnlyRoutingTelemetryFields[lhs].member ==
              kConSanBranchOnlyRoutingTelemetryFields[rhs].member ||
          kConSanBranchOnlyRoutingTelemetryFields[lhs].log_name ==
              kConSanBranchOnlyRoutingTelemetryFields[rhs].log_name)
        return false;
    }
  }
  return true;
}(), "routing telemetry field table must name every counter exactly once");

[[nodiscard]] inline std::string
format_consan_branch_only_routing_telemetry(const ConSanBranchOnlyRoutingTelemetry &telemetry) {
  std::string result;
  for (const auto &[name, member] : kConSanBranchOnlyRoutingTelemetryFields) {
    if (!result.empty())
      result.push_back(' ');
    result.append(name);
    result.push_back('=');
    result.append(std::to_string(telemetry.*member));
  }
  return result;
}

/// Instruction reservoirs retained for shared relay routing, including both
/// branch-only and legacy indirect relay consumers. Appended byte counts
/// include each complete reserved appended body. Unused reservations remain
/// explicit NOP padding so later planned offsets stay stable, and this
/// inventory makes that retained footprint observable. It is an observational
/// inventory, not a score for the unweighted owner-group minimizer.
struct ConSanBranchOnlyReservoirTelemetry {
  size_t planned_reservoir_count = 0;
  size_t used_reservoir_count = 0;
  size_t unused_reservoir_count = 0;
  size_t planned_appended_bytes = 0;
  size_t used_appended_bytes = 0;
  size_t unused_appended_bytes = 0;
  /// Release-visible defensive signal owned by the LDS reservoir replay loop.
  /// Correct convergence leaves this zero; other reservoir producers cannot
  /// set it and do not log it.
  size_t lds_replay_limit_reached_count = 0;

  bool operator==(const ConSanBranchOnlyReservoirTelemetry &) const = default;
};

static_assert(sizeof(ConSanBranchOnlyReservoirTelemetry) == 7u * sizeof(size_t),
              "update the reservoir telemetry schema tests");

/// Records one complete reserved appended footprint. Keeping the payload and
/// final return-control bytes separate makes the shared normalization
/// structural instead of relying on placement-specific body-size conventions.
inline void
accumulate_branch_only_reservoir_telemetry(ConSanBranchOnlyReservoirTelemetry &telemetry,
                                           uint64_t retained_payload_bytes,
                                           uint64_t retained_return_bytes, bool used) {
  const auto add = [](size_t &total, size_t value) {
    total = value > std::numeric_limits<size_t>::max() - total ? std::numeric_limits<size_t>::max()
                                                               : total + value;
  };
  const uint64_t retained_appended_bytes =
      retained_return_bytes > std::numeric_limits<uint64_t>::max() - retained_payload_bytes
          ? std::numeric_limits<uint64_t>::max()
          : retained_payload_bytes + retained_return_bytes;
  const size_t bytes = [&] {
    if constexpr (sizeof(uint64_t) > sizeof(size_t)) {
      if (retained_appended_bytes > std::numeric_limits<size_t>::max())
        return std::numeric_limits<size_t>::max();
    }
    return static_cast<size_t>(retained_appended_bytes);
  }();

  add(telemetry.planned_reservoir_count, 1u);
  add(telemetry.planned_appended_bytes, bytes);
  if (used) {
    add(telemetry.used_reservoir_count, 1u);
    add(telemetry.used_appended_bytes, bytes);
  } else {
    add(telemetry.unused_reservoir_count, 1u);
    add(telemetry.unused_appended_bytes, bytes);
  }
}

[[nodiscard]] inline bool
branch_only_relay_telemetry_is_empty(const ConSanBranchOnlyRoutingTelemetry &telemetry) {
  return telemetry == ConSanBranchOnlyRoutingTelemetry{};
}

[[nodiscard]] inline bool
branch_only_reservoir_telemetry_is_empty(const ConSanBranchOnlyReservoirTelemetry &telemetry) {
  return telemetry == ConSanBranchOnlyReservoirTelemetry{};
}

/// Work consumed outside the fixed-pair branch router. Counts are measured
/// units charged by the shared bounded-planning meter; exhaustion counts are
/// call- or convergence-scope events, not elapsed-time samples.
struct ConSanPlanningWorkTelemetry {
  size_t sopp_relay_work_count = 0;
  size_t sopp_relay_exhaustion_count = 0;
  size_t direct_reservoir_work_count = 0;
  size_t direct_reservoir_exhaustion_count = 0;
  size_t lds_relay_layout_work_count = 0;
  size_t lds_relay_layout_exhaustion_count = 0;
  size_t lds_convergence_work_count = 0;
  size_t lds_convergence_exhaustion_count = 0;

  bool operator==(const ConSanPlanningWorkTelemetry &) const = default;
};

static_assert(sizeof(ConSanPlanningWorkTelemetry) == 8u * sizeof(size_t),
              "update the planning-work telemetry schema tests");

[[nodiscard]] inline bool
consan_planning_work_telemetry_is_empty(const ConSanPlanningWorkTelemetry &telemetry) {
  return telemetry == ConSanPlanningWorkTelemetry{};
}

/// Machine-readable outcome of SuperCollider group-FLAT candidate selection.
/// This remains available for successful partial transforms without forcing
/// callers to parse a long warning string.
struct ConSanFlatSelectionTelemetry {
  size_t supported_candidate_count = 0;
  size_t selection_target = 0;
  size_t selected_candidate_count = 0;
  size_t branch_only_candidate_count = 0;
  size_t fixed_stack_candidate_count = 0;
  size_t dynamic_stack_candidate_count = 0;
  size_t mixed_stack_candidate_count = 0;
  size_t missing_vcc_save_candidate_count = 0;
  size_t missing_scratch_candidate_count = 0;
  size_t spill_backed_candidate_count = 0;
  size_t mixed_stack_spill_rejection_count = 0;
  size_t dynamic_stack_spill_encoding_failure_count = 0;
  size_t private_spill_encoding_failure_count = 0;
  size_t branch_only_selected_count = 0;
  size_t branch_only_placement_failure_count = 0;
  ConSanBranchOnlyRoutingTelemetry branch_only_routing;
  /// Direct-reservoir footprint retained in the emitted FLAT patch. This is
  /// published only after successful final emission, so an earlier rejection
  /// leaves it zero. A losing speculative retry is rolled back and therefore
  /// contributes no retained footprint.
  ConSanBranchOnlyReservoirTelemetry branch_only_reservoir_telemetry;
  /// Placement failures and routing activity from a losing direct-reservoir
  /// retry that was transactionally discarded. Kept separate so primary
  /// attempt/failure rates describe the selection path retained in the
  /// emitted patch. The reservoir inventory records the speculative appended
  /// footprint before rollback.
  size_t discarded_branch_only_placement_failure_count = 0;
  ConSanBranchOnlyRoutingTelemetry discarded_branch_only_routing;
  ConSanBranchOnlyReservoirTelemetry discarded_branch_only_reservoir_telemetry;
  size_t original_nop_relay_slot_count = 0;
  size_t selected_anchor_relay_slot_count = 0;
};

struct ConSanResult {
  bool visited_code_object = false;
  /// True after the input has been parsed as a structurally valid AMDGPU code
  /// object. This remains true when its machine target is not enumerated.
  bool parsed_code_object = false;
  bool modified = false;
  bool final_validation_passed = false;
  /// Both the pristine-to-mutation and mutation-to-instrumentation images were
  /// independently structurally and semantically validated before their patch
  /// inventories were composed.
  bool staged_composition_validated = false;
  /// Internal second-stage proof over a mutation image using a pristine plan.
  bool carried_composite_instrumentation_stage = false;
  /// MOI branch-only candidates skipped because their appended body could not
  /// be reserved without overlapping an accepted placement.
  size_t moi_branch_only_placement_failure_count = 0;
  ConSanBranchOnlyRoutingTelemetry moi_branch_only_routing_telemetry;
  /// Reservoir footprint retained in a successfully emitted MOI patch. A
  /// rejected transform retains no output image, so this remains zero; any
  /// attempted-plan inventory must use separate telemetry.
  ConSanBranchOnlyReservoirTelemetry moi_branch_only_reservoir_telemetry;
  ConSanBranchOnlyRoutingTelemetry lds_branch_only_routing_telemetry;
  /// Reservoir footprint retained by all successfully emitted LDS relay
  /// consumers, including branch-only and legacy indirect routing. A rejected
  /// transform retains no output image, so this remains zero; any attempted-
  /// plan inventory must use separate telemetry.
  ConSanBranchOnlyReservoirTelemetry lds_relay_reservoir_telemetry;
  ConSanPlanningWorkTelemetry planning_work_telemetry;
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
  std::optional<uint16_t> resolved_moi_dispatch_id_vgpr;
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
  bool moi_dispatch_id_vgprs_automatic = false;
  size_t input_size = 0;
  std::string input_fingerprint;
  /// Machine-readable input-metadata trust state retained with the diagnostic.
  bool kernel_metadata_trustworthy = false;
  size_t malformed_kernel_metadata_note_count = 0;
  /// Immutable semantic options captured before the first MOI planning pass.
  /// A report-buffer retry derives its options from this snapshot and can
  /// change only ConSanMoiReportRetryConfig fields plus the narrow live-fault
  /// selection carried by ConSanFaultMutationRetryConfig.
  std::optional<ConSanOptions> moi_inventory_options;
  /// Resolved target architecture.  Production transforms set this before
  /// candidate classification.  The hook rejects candidate-bearing results
  /// that retain the invalid default before coverage or require-patch
  /// classification.
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  /// Exact code-object target identity. Its display name is derived from this
  /// value, which must not participate in semantic decisions.
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID;
  /// Set immediately before architecture-dependent decoding and semantic
  /// classification begin. Parse-only results, including unsupported targets,
  /// leave this false even though they retain section and symbol inventory.
  /// Every later result stage is copied from that classified result, so new
  /// semantic fields added after the transition inherit this invariant without
  /// extending a hand-maintained list of members.
  bool semantic_arch_required = false;
  std::vector<ConSanTextSection> text_sections;
  std::vector<ConSanKernelInfo> kernels;
  std::vector<ConSanFunctionInfo> functions;
  std::vector<ConSanFaultSite> fault_sites;
  std::vector<ConSanPatchedImageGrowthRejection> patched_image_growth_rejections;
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
  /// True when sc_access_coverage_sites is the authoritative physical access
  /// denominator for this SuperCollider transform. Planning can promote a
  /// site to supported only after every alias and execution owner has a
  /// complete CFG/liveness context.
  bool sc_access_coverage_resolved = false;
  std::vector<ConSanScAccessCoverageSite> sc_access_coverage_sites;
  /// Warning boundary immediately before MOI planning. A validated retry with
  /// a runtime-allocated report buffer trims only the prior MOI-stage
  /// diagnostics while retaining parser and semantic-analysis warnings.
  size_t moi_stage_warning_begin = 0;
  std::vector<ConSanCandidateResourcePlan> resource_plans;
  ConSanResourcePlanSummary resource_plan_summary;
  std::optional<ConSanFlatSelectionTelemetry> flat_selection_telemetry;
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
[[nodiscard]] ConSanResourcePlanAlternativeOutcome
consan_resource_plan_alternative_outcome(const ConSanCandidateResourcePlan &plan,
                                         const ConSanResourcePlanAlternative &alternative);
[[nodiscard]] const char *
consan_resource_plan_alternative_kind_name(ConSanResourcePlanAlternativeKind kind);
[[nodiscard]] const char *
consan_resource_plan_alternative_outcome_name(ConSanResourcePlanAlternativeOutcome outcome);
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
