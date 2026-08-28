/// Temporary lowering state for one access admitted by semantic policy.
///
/// Its `ConSanAccessInventorySite` base is the exact normalized record selected
/// by the observation plan. Lowering reads that snapshot without re-decoding
/// or reclassifying it. The remaining members are facts created only after
/// admission: target execution state, the selected execution owner, and the
/// compact nonnegative ranges consumed by current emitters. Extending the
/// inventory value keeps its established field vocabulary without maintaining
/// an independent access identity, operand, or decode schema.
struct ConSanMoiCandidate : ConSanAccessInventorySite {
  /// Every observation intent coalesced onto this physical access.
  std::vector<ConSanProbeIntentId> intent_ids;
  /// Return the original-text instruction offset used as this lowering
  /// operation's physical insertion anchor.
  [[nodiscard]] uint64_t anchor() const { return physical_id.original_text_offset; }
  /// Return the pristine guest instruction size in bytes. Lowering may reserve
  /// a larger replacement body, but that does not change this displaced span.
  [[nodiscard]] uint32_t size() const { return instruction_size; }
  /// Return the normalized width in bits of each semantic access range.
  [[nodiscard]] uint32_t width_bits() const { return decoded_width_bits; }
  /// Return the nonnegative displacement consumed by native emission. FLAT
  /// lowering first materializes its complete effective address, so its
  /// inventory displacement must not be applied a second time. Admitted
  /// native sites always carry a representable decoded displacement.
  [[nodiscard]] uint32_t lowering_offset(const ConSanAccessRange &range) const {
    return origin == ConSanAccessOrigin::Flat
               ? 0u
               : static_cast<uint32_t>(range.static_byte_offset.value());
  }

  /// Target-native selectable-VGPR-bank mode active at this instruction, when
  /// analysis could establish it for lowering. Targets without selectable
  /// banks leave this absent; a present zero value records the ordinary low
  /// bank explicitly.
  std::optional<uint16_t> incoming_vgpr_bank_mode;
  /// Descriptor selected as the concrete execution owner for this lowering
  /// operation. Shared helper sites may be copied once per reachable owner.
  std::optional<uint64_t> kernel_descriptor_file_offset;
};

/// One synchronization-sequence edge considered for scheduling perturbation.
///
/// Analysis derives these records from the immutable synchronization
/// inventory. They are temporary evidence for selecting a perturbation, not a
/// transform result: emitted patches retain the stable semantic identity and
/// source coordinates required by final validation.
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

/// One admitted scheduling perturbation ready for byte-level lowering.
///
/// The plan binds a candidate's semantic identity to its selected sleep
/// policy. Composite fault/perturbation lowering also records the pristine
/// identity and its translated anchor. This is temporary planner-to-emitter
/// input; mutation tallies and patch metadata are the durable output.
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

/// Temporary workspace shared only by perturbation analysis, selection, and
/// emission during one lowering attempt.
///
/// Keeping this state outside `ConSanTransformArtifacts` prevents callers and
/// later pipeline stages from treating candidates or pre-emission plans as a
/// lasting behavioral result. The optional inspection path used by unit tests
/// observes this workspace without extending its production lifetime.
struct ConSanPerturbationPlanningState {
  /// All release and acquire edges derived from synchronization sequences,
  /// including rejected edges whose reason is useful to planner unit tests.
  std::vector<ConSanPerturbationCandidate> candidates;
  /// The bounded subset selected for emission under the current options.
  std::vector<ConSanPerturbationPlan> plans;
};

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

/// Semantic and lowering artifacts owned by one static ConSan transformation.
///
/// The internal native lowerer builds this value incrementally, but ownership
/// crosses into `TransformResult` as one subobject after lowering finishes.
/// Keeping these artifacts together prevents the production pipeline from
/// redeclaring and individually projecting inventory, policy, coverage,
/// mutation, resource, patch, outcome, and diagnostic state. Runtime evidence,
/// installation state and private working candidates deliberately live
/// elsewhere.
struct ConSanTransformArtifacts {
  /// Immutable ownership of original code-object/container/access facts.
  ProgramInventory program_inventory;
  /// Target-neutral policy output retained independently of current lowering
  /// candidates and patch telemetry.
  ConSanObservationPlan observation_plan;
  /// Joined static policy and lowering state initialized from the observation
  /// plan rather than inferred from emitted patch kinds.
  ConSanCoverageLedger coverage_ledger;
  /// Stable semantic sites eligible for validation-only fault selection.
  std::vector<ConSanFaultSite> fault_sites;
  /// Candidate destinations for a barrier-move validation mutation.
  std::vector<ConSanBarrierMoveDestination> barrier_move_destinations;
  /// Concrete validation-only fault plans selected from semantic sites.
  std::vector<ConSanFaultMutationPlan> fault_plans;
  /// Validation-only mutation facts produced by this static transform.
  ConSanMutationOutcome mutation;
  /// Resource alternatives considered by lowering.
  std::vector<ConSanCandidateResourcePlan> resource_plans;
  /// Accepted MOI operating point consumed by emitted mechanisms and final
  /// semantic validation. Non-MOI transforms leave this value empty.
  ConSanMoiOperatingPoint moi_operating_point;
  /// Authoritative intent-bound lowering results published by placement.
  std::vector<ConSanCommittedLowering> committed_lowerings;
  /// Validated runtime-facing projection accumulated from committed lowering
  /// transactions. Hooks consume this product instead of patch telemetry.
  ConSanRuntimeStaticMapping runtime_static_mapping;
  /// Intent-bound synchronization products staged across the MOI atomic,
  /// fence, and barrier emitters. A shared address-capture intent can serve
  /// more than one emitted record, so the pipeline merges all of its machine
  /// locations before publishing one authoritative result at finalization.
  std::vector<ConSanCommittedLowering> staged_moi_sync_lowerings;
  /// Validated byte-level changes emitted by lowering.
  std::vector<ConSanPatchInfo> patches;
  /// Complete validated code-object replacement, or empty when none is installable.
  std::vector<uint8_t> replacement;
  /// Final static classification of the transformation attempt.
  ConSanTransformOutcome outcome = ConSanTransformOutcome::Unchanged;
  /// Non-fatal diagnostics retained from analysis and lowering.
  std::vector<std::string> warnings;
  /// Fatal static-transform diagnostics emitted by lowering or final
  /// validation. Typed request, capability, and binding failures live in their
  /// pipeline stage record instead of being duplicated here.
  std::vector<std::string> errors;

  /// Whether lowering has staged changes that finalization must validate or
  /// discard, derived from its outcome and emitted patch telemetry.
  [[nodiscard]] bool modified() const {
    return outcome == ConSanTransformOutcome::ModifiedValid || !patches.empty();
  }
  /// Mark a candidate image without overwriting a prior failure outcome.
  void mark_modified() {
    if (outcome == ConSanTransformOutcome::Unchanged)
      outcome = ConSanTransformOutcome::ModifiedValid;
  }
  /// Publish and retain one authoritative intent-bound lowering result.
  [[nodiscard]] bool publish_lowering_commit(ConSanCommittedLowering commit) {
    std::vector<ConSanCommittedLowering> commits;
    commits.push_back(std::move(commit));
    return publish_lowering_commits(std::move(commits));
  }
  /// Atomically publish and retain a complete group of lowering results.
  [[nodiscard]] bool publish_lowering_commits(std::vector<ConSanCommittedLowering> commits) {
    ConSanCoverageLedger next_ledger = coverage_ledger;
    ConSanRuntimeStaticMapping next_mapping = runtime_static_mapping;
    for (const ConSanCommittedLowering &commit : commits) {
      if (!consan_runtime_static_mapping_matches_commit(observation_plan, commit) ||
          !next_ledger.publish_lowering_commit(commit))
        return false;
      next_mapping.append(commit.runtime_mapping);
    }
    coverage_ledger = std::move(next_ledger);
    runtime_static_mapping = std::move(next_mapping);
    committed_lowerings.reserve(committed_lowerings.size() + commits.size());
    committed_lowerings.insert(committed_lowerings.end(), std::make_move_iterator(commits.begin()),
                               std::make_move_iterator(commits.end()));
    return true;
  }
  /// Stage and transitively merge MOI synchronization commits that share an
  /// intent. Publication is deferred until every synchronization emitter has
  /// contributed, allowing one address-capture intent to name both an atomic
  /// record and an associated fence record without duplicate ownership.
  [[nodiscard]] bool stage_moi_sync_lowering_commits(std::vector<ConSanCommittedLowering> commits) {
    std::vector<ConSanCommittedLowering> next = staged_moi_sync_lowerings;
    for (ConSanCommittedLowering &incoming : commits) {
      if (incoming.outcome != ConSanLoweringOutcomeKind::Instrumented)
        return false;
      std::vector<ConSanProbeIntentId> ids = incoming.intent_ids;
      std::vector<ConSanCommittedLoweringLocation> locations = incoming.locations;
      ConSanRuntimeStaticMapping runtime_mapping = std::move(incoming.runtime_mapping);
      for (size_t index = 0; index < next.size();) {
        const bool overlaps = std::ranges::any_of(next[index].intent_ids, [&](const auto &id) {
          return std::ranges::find(ids, id) != ids.end();
        });
        if (!overlaps) {
          ++index;
          continue;
        }
        for (ConSanProbeIntentId id : next[index].intent_ids) {
          if (std::ranges::find(ids, id) == ids.end())
            ids.push_back(id);
        }
        for (const ConSanCommittedLoweringLocation &location : next[index].locations) {
          if (std::ranges::find(locations, location) == locations.end())
            locations.push_back(location);
        }
        runtime_mapping.append(std::move(next[index].runtime_mapping));
        next.erase(next.begin() + static_cast<std::ptrdiff_t>(index));
      }
      auto merged = make_consan_committed_lowering(observation_plan, ids, locations,
                                                   ConSanLoweringOutcomeKind::Instrumented, {},
                                                   std::move(runtime_mapping));
      if (!merged)
        return false;
      next.push_back(std::move(*merged));
    }
    staged_moi_sync_lowerings = std::move(next);
    return true;
  }
  /// Publish the complete synchronization transaction after every emitter has
  /// contributed its locations.
  [[nodiscard]] bool publish_staged_moi_sync_lowerings() {
    if (!publish_lowering_commits(staged_moi_sync_lowerings))
      return false;
    staged_moi_sync_lowerings.clear();
    return true;
  }
  /// Discard the candidate image and its patch telemetry without changing the
  /// caller-owned outcome, diagnostics, semantic inventory, or mutation facts.
  void discard_candidate_modification() {
    replacement.clear();
    patches.clear();
    runtime_static_mapping = {};
    staged_moi_sync_lowerings.clear();
    std::erase_if(committed_lowerings, [](const ConSanCommittedLowering &commit) {
      return commit.outcome == ConSanLoweringOutcomeKind::Instrumented;
    });
    coverage_ledger = ConSanCoverageLedger(observation_plan);
    for (const ConSanCommittedLowering &commit : committed_lowerings)
      (void)coverage_ledger.publish_lowering_commit(commit);
  }
};

/// Whether a semantic claim is at least as strong as the weakest confidence a
/// consumer explicitly accepts.
[[nodiscard]] constexpr bool
consan_sync_confidence_meets(ConSanSemanticConfidence actual,
                             ConSanSemanticConfidence weakest_accepted) {
  return static_cast<uint8_t>(actual) <= static_cast<uint8_t>(weakest_accepted);
}

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

[[nodiscard]] std::optional<ConSanFlavor> parse_consan_flavor(std::string_view value);
[[nodiscard]] std::optional<ConSanMoiEngine> parse_consan_moi_engine(std::string_view value);

/// Re-run the structural final-ELF validation used before a modified result is
/// made installable. The expected report dispatch identity is a runtime input,
/// not transform output; tests exercising a nonzero literal identity must pass
/// the same value used for lowering. Exposed for bounded mutation/property
/// tests.
[[nodiscard]] std::vector<std::string>
validate_consan_modified_elf(std::span<const uint8_t> original_bytes,
                             const ConSanTransformArtifacts &modified_result,
                             uint64_t expected_moi_report_dispatch_id = 0);
