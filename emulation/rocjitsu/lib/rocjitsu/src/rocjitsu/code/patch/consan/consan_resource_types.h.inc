enum class ConSanResourceSiteKind : uint8_t {
  Access,
  Barrier,
  Atomic,
  Fence,
};

/// Number of directly encoded ordinary VGPRs in the ConSan instrumentation
/// ABIs. REGISTER_SET_MAX_VGPRS is intentionally larger so analysis can also
/// represent architecture-specific banks; it is not an emission bound.
inline constexpr uint16_t kConSanOrdinaryVgprLimit = 256u;

/// A structurally distinct fallback considered after the primary resource plan.
///
/// Keeping every attempt explicit lets diagnostics distinguish rejected,
/// superseded, contributing, and vetoed plans from sites that never had a
/// correctness-preserving alternative. Producers stamp the outcome while the
/// planning branch that made the decision is still available.
enum class ConSanResourcePlanAlternativeKind : uint8_t {
  GuestOperandOverlapSpill,
  SpillBackedOperandRecovery,
  EmptyAccumulatorDescriptorGrowth,
};

enum class ConSanResourcePlanAlternativeOutcome : uint8_t {
  Selected,
  Rejected,
  Superseded,
  Contributed,
  Vetoed,
};

struct ConSanResourcePlanAlternative {
  ConSanResourcePlanAlternativeKind kind =
      ConSanResourcePlanAlternativeKind::GuestOperandOverlapSpill;
  ConSanRegisterAllocationSource source = ConSanRegisterAllocationSource::Unsupported;
  ConSanRegisterPlanReason reason = ConSanRegisterPlanReason::None;
  uint16_t scratch_vgpr_count = 0;
  ConSanResourcePlanAlternativeOutcome outcome = ConSanResourcePlanAlternativeOutcome::Rejected;
};

struct ConSanCandidateResourcePlan {
  ConSanResourceSiteKind site_kind = ConSanResourceSiteKind::Access;
  size_t candidate_index = 0;
  /// Physical instruction where liveness and allocation were evaluated.
  uint64_t text_offset = 0;
  /// Logical synchronization site represented by a patch whose physical
  /// anchor must precede that site. For example, an ordinary acquire records
  /// its cache fence but branches before the address-bearing load so it can
  /// preserve an address that the load overwrites. Absent when both sites are
  /// identical.
  std::optional<uint64_t> semantic_text_offset;
  std::vector<uint64_t> owner_descriptor_file_offsets;
  ConSanRegisterAllocationSource source = ConSanRegisterAllocationSource::Unsupported;
  ConSanRegisterPlanReason reason = ConSanRegisterPlanReason::None;
  std::optional<uint16_t> scratch_vgpr;
  uint16_t scratch_vgpr_count = 0;
  uint16_t current_vgpr_count = 0;
  uint16_t max_referenced_vgpr_count = 0;
  uint16_t ordinary_vgpr_limit = kConSanOrdinaryVgprLimit;
  bool has_indirect_vgpr_access = false;
  uint16_t required_vgpr_count = 0;
  uint16_t current_sgpr_count = 0;
  uint16_t max_referenced_sgpr_count = 0;
  /// Per-owner dominating bound used by scalar persistent-state placement.
  uint16_t scalar_tail_floor = 0;
  /// Diagnostic aggregation; placement computes scalar_tail_floor per owner.
  bool has_indirect_sgpr_access = false;
  bool sgpr_reference_coverage_complete = false;
  uint32_t original_private_segment_size = 0;
  /// Every structurally distinct fallback evaluated while producing this plan,
  /// in chronological planning order. At most one attempt is Selected; nested
  /// attempts that enable it are Contributed.
  std::vector<ConSanResourcePlanAlternative> alternatives;
};

/// Complete, unpublished output of one MOI resource-planning attempt.
///
/// `plans` contains every access and synchronization-site allocation derived
/// for one operating point. `errors` contains structural failures discovered
/// while canonicalizing those admitted sites. The planner builds both fields
/// privately and returns them together, allowing its caller to replace the
/// prior plan set and append diagnostics in one explicit publication step.
/// This prevents a retry from exposing a partial vector or mutating transform
/// diagnostics while it is still evaluating a candidate operating point.
struct ConSanMoiResourcePlanningResult {
  /// Resource alternatives and selected allocation for every planned site.
  std::vector<ConSanCandidateResourcePlan> plans;
  /// Fatal structural diagnostics produced while deriving this plan set.
  std::vector<std::string> errors;

  /// Whether planning completed without a structural diagnostic. Individual
  /// sites may still carry a typed Unsupported allocation in a successful set.
  [[nodiscard]] bool success() const { return errors.empty(); }
};

/// Aggregate telemetry derived from completed resource plans and emitted
/// patches. Its counters describe planner outcomes and the spill storage used
/// by emitted mechanics. This type owns no transform state; construct it with
/// `summarize_consan_resource_plans` whenever a diagnostic or test needs the
/// reporting view.
struct ConSanResourcePlanSummary {
  size_t explicit_plans = 0;
  size_t dead_plans = 0;
  size_t descriptor_growth_plans = 0;
  size_t spill_plans = 0;
  size_t unsupported_plans = 0;
  size_t planned_spill_slot_bytes = 0;
  size_t emitted_spill_patches = 0;
  size_t emitted_spill_slot_bytes = 0;
  size_t alternative_attempts = 0;
  size_t alternative_selected = 0;
  size_t alternative_rejected = 0;
  size_t alternative_superseded = 0;
  size_t alternative_contributed = 0;
  size_t alternative_vetoed = 0;
};
