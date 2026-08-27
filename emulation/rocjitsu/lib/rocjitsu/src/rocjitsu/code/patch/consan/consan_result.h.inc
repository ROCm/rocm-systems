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

  /// gfx1250 VGPR-bank mode active at this instruction, when analysis could
  /// establish it for lowering.
  std::optional<uint16_t> gfx1250_vgpr_msb_mode;
  /// Descriptor selected as the concrete execution owner for this lowering
  /// operation. Shared helper sites may be copied once per reachable owner.
  std::optional<uint64_t> kernel_descriptor_file_offset;
};

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

[[nodiscard]] std::optional<ConSanFlavor> parse_consan_flavor(std::string_view value);
[[nodiscard]] std::optional<ConSanMoiEngine> parse_consan_moi_engine(std::string_view value);

/// Re-run the structural final-ELF validation used before a modified result is
/// made installable. Exposed for bounded mutation/property tests.
[[nodiscard]] std::vector<std::string>
validate_consan_modified_elf(std::span<const uint8_t> original_bytes,
                             const ConSanResult &modified_result);
