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

/// Stable disposition of one decoded final-code site before engine-specific
/// candidate filtering. Coverage is derived from this ledger rather than
/// reconstructing discovery from the surviving candidate vector.
enum class ConSanSiteDisposition : uint8_t {
  NotApplicable,
  Supported,
  Unsupported,
};

/// Stable, machine-auditable reason attached to a site disposition.
enum class ConSanSiteDispositionReason : uint8_t {
  None,
  NonAccessInstruction,
  UnsupportedMnemonic,
  InvalidInstructionSize,
  InvalidAccessWidth,
  MissingAddressOperand,
  InstructionOutOfBounds,
  UnsupportedFlatEncoding,
  NonzeroFlatOffset,
  ReservedFlatAddressRegister,
  NonGroupAddressSpace,
  FlatProvenancePolicyExcluded,
  RuntimeKernelExcluded,
  NonFlatAtomicAddress,
  MissingAtomicOperands,
  MissingOrderingMetadata,
  UnqualifiedSyncSequence,
  NoAtomicAcquireConsumer,
  NoPrecedingSampledWindow,
  UnsupportedMemoryRole,
  MissingScope,
  UnsupportedScope,
  CompareExchangeOutcomeUnavailable,
  UnsupportedAtomicOutcome,
  UnsupportedAtomicEncoding,
  UnsupportedMetadataEncoding,
  IneligibleFence,
  MissingCommunicationEvent,
  InvalidBarrierEncoding,
};

/// Durable final outcome for a semantically inventoried site. Unlike
/// ConSanSiteDisposition, this records what happened after candidate
/// selection, register planning, placement, and lowering.
enum class ConSanSiteLoweringOutcome : uint8_t {
  NotApplicable,
  Unsupported,
  Pending,
  Patched,
  ResourceFailed,
  PlacementOrLoweringFailed,
};

/// Stable category for the final lowering outcome. Detailed register-plan
/// failures remain available in resource_reason without parsing diagnostics.
enum class ConSanSiteLoweringReason : uint8_t {
  None,
  SemanticNotApplicable,
  SemanticUnsupported,
  UnsupportedResourcePlan,
  InstrumentationPatchMissing,
};

struct ConSanSiteDispositionRecord {
  ConSanResourceSiteKind site_kind = ConSanResourceSiteKind::Access;
  ConSanSiteDisposition disposition = ConSanSiteDisposition::NotApplicable;
  ConSanSiteDispositionReason reason = ConSanSiteDispositionReason::None;
  std::string container_name;
  bool in_kernel = true;
  uint64_t text_offset = 0;
  std::string mnemonic;
  ConSanSiteLoweringOutcome lowering_outcome = ConSanSiteLoweringOutcome::NotApplicable;
  ConSanSiteLoweringReason lowering_reason = ConSanSiteLoweringReason::SemanticNotApplicable;
  ConSanRegisterPlanReason resource_reason = ConSanRegisterPlanReason::None;
};

/// A structurally distinct fallback considered after the primary resource plan.
///
/// Keeping every attempt explicit lets diagnostics distinguish rejected,
/// superseded, contributing, and vetoed plans from sites that never had a
/// correctness-preserving alternative. Producers stamp the outcome while the
/// planning branch that made the decision is still available.
enum class ConSanResourcePlanAlternativeKind : uint8_t {
  GuestOperandOverlapSpill,
  SpillBackedOperandRecovery,
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
  ConSanResourcePlanAlternativeOutcome outcome =
      ConSanResourcePlanAlternativeOutcome::Rejected;
};

struct ConSanCandidateResourcePlan {
  ConSanResourceSiteKind site_kind = ConSanResourceSiteKind::Access;
  size_t candidate_index = 0;
  uint64_t text_offset = 0;
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

/// Exact redundant-access selection proven by an internal validated composite
/// transform while the externally visible invocation remains a dry run.
struct ConSanAccessPlan {
  std::string identity;
  std::string container_name;
  bool in_kernel = true;
  std::string kind;
  uint64_t text_offset = 0;
};

/// Source-to-translated perturbation evidence retained from one independently
/// validated staged composite transform.
struct ConSanCompositeProof {
  std::string pristine_identity;
  std::string pristine_sequence;
  std::string pristine_container;
  bool pristine_in_kernel = true;
  uint64_t pristine_owner_descriptor = 0;
  ConSanPerturbationEdge pristine_edge = ConSanPerturbationEdge::Release;
  std::string pristine_anchor;
  uint64_t pristine_anchor_text_offset = 0;
  uint32_t pristine_anchor_size = 0;
  std::string translated_identity;
  ConSanPerturbationEdge translated_edge = ConSanPerturbationEdge::Release;
  uint64_t translated_anchor_text_offset = 0;
  uint32_t translated_anchor_size = 0;
  std::string anchor_relation;
  std::string cache_companion_identity = "-";
  bool atomic_overlap = false;
  bool removed_cache_boundary = false;
  bool removed_cache_non_resurrection_validated = false;
  bool removed_cache_non_resurrection_applicable = false;
  std::optional<uint64_t> atomic_mutation_anchor_text_offset;
};
