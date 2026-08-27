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

/// Retained branch-only routing activity used to verify integration behavior.
/// Pair attempts count submitted requests and plan calls count router
/// invocations, including LDS convergence iterations; together they detect an
/// accidental return to per-site inventory construction. Entry-route, relay-
/// contention, and work-budget failures retain the rejection classes exercised
/// by integration tests. Detailed return, reservation, invariant, and
/// optimizer outcomes remain on the router's direct result, where focused unit
/// tests inspect them without copying them into every completed transform.
///
/// Fallback counters count pairs that reached the corresponding tier, whether
/// or not it produced a route. Greedy fallback attempts are a subset of exact-
/// pair fallback attempts and the counters must not be summed. Work-budget
/// exhaustion counts plan calls whose qualification, routing, setup, or greedy
/// tier reached its configured limit, including calls recovered by fallback.
/// The individual work counters identify qualification, setup, feasibility
/// scanning, and route-optimization phases. Search counters measure charged
/// work units rather than physical relay inspections: a mixed owner inventory
/// may perform one uncharged second cost tier, so physical inspections are at
/// most twice the charged traversal units for optimization. A losing FLAT
/// direct-reservoir retry is transactionally discarded and does not contribute.
struct ConSanBranchOnlyRoutingTelemetry {
  size_t pair_attempt_count = 0;
  size_t plan_call_count = 0;
  size_t entry_route_failure_count = 0;
  size_t relay_contention_failure_count = 0;
  size_t work_budget_failure_count = 0;
  size_t work_budget_exhaustion_count = 0;
  size_t exact_pair_fallback_attempt_count = 0;
  size_t greedy_pair_fallback_attempt_count = 0;
  size_t route_optimization_search_work_count = 0;
  size_t route_optimization_scan_work_count = 0;
  size_t relay_qualification_work_count = 0;
  size_t fallback_setup_work_count = 0;
  size_t feasibility_scan_work_count = 0;

  bool operator==(const ConSanBranchOnlyRoutingTelemetry &) const = default;
};

/// Instruction reservoirs retained for shared relay routing, including both
/// branch-only and legacy indirect relay consumers. Appended byte counts
/// include each complete reserved appended body. Unused reservations remain
/// explicit NOP padding so later planned offsets stay stable, and this
/// inventory makes that retained footprint observable. It is an observational
/// inventory, not a score for the unweighted owner-group minimizer.
struct ConSanBranchOnlyReservoirTelemetry {
  size_t planned_reservoir_count = 0;
  size_t used_reservoir_count = 0;
  size_t planned_appended_bytes = 0;
  size_t used_appended_bytes = 0;

  bool operator==(const ConSanBranchOnlyReservoirTelemetry &) const = default;
};

static_assert(sizeof(ConSanBranchOnlyReservoirTelemetry) == 4u * sizeof(size_t),
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
  }
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

/// Counts one validation-only mutation domain as it moves from a user request
/// through semantic selection to emitted replacement bytes.
///
/// A count greater than one is not inherently malformed: the planner retains
/// ambiguous matches so the runtime policy can reject them with a precise
/// reason, and fault-injection tests deliberately model multiple applied
/// mutations. This tally reports transform facts only. It does not claim that
/// a replacement code object was installed by a runtime.
struct ConSanMutationTally {
  /// Number of mutations requested by the validated mutation policy.
  size_t requested = 0;
  /// Number of concrete semantic mutations selected by planning.
  size_t planned = 0;
  /// Number of selected mutations whose bytes were emitted successfully.
  size_t applied = 0;

  /// Return whether planning found at least one matching mutation.
  [[nodiscard]] bool has_plan() const { return planned != 0u; }
  /// Return whether selection is ambiguous rather than exactly one match.
  [[nodiscard]] bool has_ambiguous_plan() const { return planned > 1u; }
  /// Return whether the replacement image contains a mutation from this
  /// domain, independently of whether a runtime later installed that image.
  [[nodiscard]] bool has_application() const { return applied != 0u; }

  bool operator==(const ConSanMutationTally &) const = default;
};

/// Typed transform output for validation-only fault mutation and perturbation.
///
/// Fault mutation intentionally corrupts a synchronization or memory idiom;
/// perturbation changes scheduling without changing the sanitizer's ordinary
/// observation contract. Keeping both tallies beside the stable identity of
/// an applied fault lets runtime selection, strict exactly-one enforcement,
/// and post-load installation accounting consume one immutable result instead
/// of inspecting lowering patches. Runtime installation state and process-wide
/// reservation state are deliberately owned by the runtime coordinator and do
/// not appear here.
struct ConSanMutationOutcome {
  /// Destructive fault-mutation request, plan, and emission counts.
  ConSanMutationTally fault;
  /// Non-destructive scheduling-perturbation request, plan, and emission
  /// counts.
  ConSanMutationTally perturbation;
  /// Stable semantic identity of the applied fault, when the selected fault
  /// form defines one and its bytes were emitted.
  std::optional<std::string> applied_fault_logical_identity;

  bool operator==(const ConSanMutationOutcome &) const = default;
};

struct ConSanResult {
  bool modified = false;
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
  std::optional<uint16_t> resolved_moi_owner_vgpr;
  std::optional<uint16_t> resolved_moi_epoch_vgpr;
  std::optional<uint16_t> resolved_moi_workgroup_key_vgpr;
  ConSanMoiPersistentWorkgroupRegisters resolved_moi_record_replay_workgroup_vgprs;
  std::vector<ConSanMoiPersistentVgprAssignment> resolved_moi_persistent_vgpr_assignments;
  std::optional<uint16_t> resolved_moi_persistent_owner_sgpr;
  std::optional<uint16_t> resolved_moi_persistent_epoch_sgpr;
  std::optional<uint16_t> resolved_moi_persistent_workgroup_key_sgpr;
  ConSanMoiPersistentWorkgroupRegisters resolved_moi_record_replay_workgroup_sgprs;
  std::optional<uint16_t> resolved_moi_exec_save_sgpr;
  std::vector<ConSanMoiTransientSgprAssignment> resolved_moi_transient_sgpr_assignments;
  std::optional<uint16_t> resolved_moi_dispatch_id_sgpr;
  std::optional<uint16_t> resolved_moi_dispatch_id_vgpr;
  uint64_t moi_report_dispatch_id = 0;
  /// Immutable ownership of original code-object/container/access facts.
  ProgramInventory program_inventory;
  /// Target-neutral policy output retained independently of current lowering
  /// candidates and patch telemetry. Slice 4A initially populates access
  /// decisions and intents; later slices extend the same plan with sync forms.
  ConSanObservationPlan observation_plan;
  /// Joined static policy and lowering state. This is initialized from
  /// `observation_plan` before resource planning and never infers policy from
  /// an emitted patch kind.
  ConSanCoverageLedger coverage_ledger;
  std::vector<ConSanFaultSite> fault_sites;
  std::vector<ConSanBarrierMoveDestination> barrier_move_destinations;
  std::vector<ConSanFaultMutationPlan> fault_plans;
  std::vector<ConSanPerturbationCandidate> perturbation_candidates;
  std::vector<ConSanPerturbationPlan> perturbation_plans;
  /// Validation-only mutation facts. The transform pipeline moves this value
  /// into its public immutable result before exposing legacy mechanism detail.
  ConSanMutationOutcome mutation;
  std::vector<ConSanMoiCandidate> moi_candidates;
  /// Warning boundary immediately before MOI planning. A validated retry with
  /// a runtime-allocated report buffer trims only the prior MOI-stage
  /// diagnostics while retaining parser and semantic-analysis warnings.
  size_t moi_stage_warning_begin = 0;
  std::vector<ConSanCandidateResourcePlan> resource_plans;
  std::vector<ConSanPatchInfo> patches;
  std::vector<uint8_t> elf_bytes;
  std::vector<std::string> errors;
  std::vector<std::string> warnings;
};

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

[[nodiscard]] std::optional<ConSanFlavor> parse_consan_flavor(std::string_view value);
[[nodiscard]] std::optional<ConSanMoiEngine> parse_consan_moi_engine(std::string_view value);

/// Re-run the structural final-ELF validation used before a modified result is
/// made installable. Exposed for bounded mutation/property tests.
[[nodiscard]] std::vector<std::string>
validate_consan_modified_elf(std::span<const uint8_t> original_bytes,
                             const ConSanResult &modified_result);
