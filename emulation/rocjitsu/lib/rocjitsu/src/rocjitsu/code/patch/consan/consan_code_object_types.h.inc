struct ConSanTextSection {
  std::string name;
  uint64_t file_offset = 0;
  uint64_t virtual_address = 0;
  uint64_t size = 0;
};

struct ConSanKernelStats {
  uint64_t instruction_count = 0;
  uint64_t decode_error_count = 0;
  uint64_t lds_read_count = 0;
  uint64_t lds_write_count = 0;
  uint64_t lds_atomic_count = 0;
  uint64_t ds_other_count = 0;
  uint64_t flat_read_count = 0;
  uint64_t flat_write_count = 0;
  uint64_t flat_atomic_count = 0;
  uint64_t flat_group_hint_count = 0;
  uint64_t flat_private_hint_count = 0;
  uint64_t flat_maybe_group_hint_count = 0;
  uint64_t flat_maybe_private_hint_count = 0;
  uint64_t flat_global_hint_count = 0;
  uint64_t flat_unknown_hint_count = 0;
  uint64_t global_memory_count = 0;
  uint64_t scratch_memory_count = 0;
  uint64_t barrier_count = 0;
  uint64_t wait_count = 0;
  uint64_t fence_like_count = 0;
};

enum class ConSanFlatAddressSpaceHint : uint8_t {
  Unknown,
  Group,
  Private,
  MaybeGroup,
  MaybePrivate,
  Global,
};

enum class ConSanAtomicAddressSpaceHint : uint8_t {
  Unknown,
  Lds,
  FlatGroup,
  FlatPrivate,
  FlatMaybeGroup,
  FlatMaybePrivate,
  FlatGlobal,
  FlatUnknown,
  Global,
  Scratch,
  Buffer,
  Scalar,
};

/// Static disposition of one kernel after ConSan has inventoried its code.
///
/// This is a transformation fact, not loader policy. In particular, `Blocked`
/// does not decide whether the runtime loads the pristine image or rejects the
/// code-object load; `RuntimePolicy` makes that decision from the completed
/// transform outcome.
enum class ConSanPreflightAction : uint8_t {
  /// Inventory has not yet assigned a disposition.
  NotRun,
  /// The kernel contains at least one supported native-LDS candidate.
  Candidate,
  /// The kernel has no native-LDS candidate, but another access family may
  /// still be instrumentable and the kernel does not invalidate the image.
  Skip,
  /// Static facts make the complete code object unsafe to transform. The
  /// resulting transform is `Unsupported`, independent of loader policy.
  Blocked,
};

enum class ConSanTransformOutcome : uint8_t {
  Unchanged,
  ModifiedValid,
  Unsupported,
  Invalid,
};

enum class ConSanInstallAction : uint8_t {
  LoadOriginal,
  LoadReplacement,
  Reject,
};

enum class ConSanPatchKind : uint8_t {
  InlineNopRewrite,
  InlineEndpgmRewrite,
  InlineLdsEndpgmRewrite,
  LdsLoadCheckTrap,
  LdsStoreCheckTrap,
  // Compatibility spellings for callers that have not yet stopped describing
  // semantic LDS instrumentation by its former physical placement.
  InlineLdsLoadCheckTrap = LdsLoadCheckTrap,
  LocalCaveLdsLoadCheckTrap = LdsLoadCheckTrap,
  InlineLdsStoreCheckTrap = LdsStoreCheckTrap,
  LocalCaveLdsStoreCheckTrap = LdsStoreCheckTrap,
  FlatLoadCheckTrap,
  FlatStoreCheckTrap,
  // Compatibility spellings for callers that have not yet stopped describing
  // semantic FLAT instrumentation by its former physical placement.
  InlineFlatLoadCheckTrap = FlatLoadCheckTrap,
  LocalCaveFlatLoadCheckTrap = FlatLoadCheckTrap,
  InlineFlatStoreCheckTrap = FlatStoreCheckTrap,
  LocalCaveFlatStoreCheckTrap = FlatStoreCheckTrap,
  InlineFlatTrapRewrite,
  InlineBarrierNopRewrite,
  InlineBarrierIdScopeRewrite,
  InlineBarrierParticipantCountRewrite,
  InlineBarrierMoveSourceRewrite,
  InlineBarrierMoveTargetRewrite,
  InlineAtomicAddressRewrite,
  InlineAtomicOrderRewrite,
  InlineAtomicScopeRewrite,
  InlineLdsAddressRewrite,
  InlineOrdinaryAddressRewrite,
  InlineOrdinaryOrderRewrite,
  InlineOrdinaryScopeRewrite,
  InlineMoiAccessRecordStore,
  TrampolineMoiAccessRecordStore,
  InlineMoiExactShadowStore,
  TrampolineMoiExactShadowStore,
  InlineMoiSampledWatchpointStore,
  TrampolineMoiSampledWatchpointStore,
  KernelEntryMoiOwnerEpochPrologue,
  KernelEntryMoiPrivateEpochPrologue,
  TrampolineMoiBarrierRecord,
  TrampolineMoiInlineEpochBarrier,
  TrampolineMoiInlineAtomicOrdering,
  TrampolineMoiAtomicRecord,
  TrampolineMoiSampledSyncMetadata,
  TrampolineMoiFenceRecord,
  InlineMalformedBarrierAbort,
  TrampolineScPerturbation,
  TrampolineScIndirectBranchIsland,
  TrampolineScDenseCallDispatcher,
  TrampolineScDenseEntryHost,
  TrampolineBranchRelayReservoir,
  TrampolineMoiIndirectBranchIsland,
  TrampolineNop,
  /// Appended after the original patch-kind sequence so existing numeric
  /// diagnostics remain stable across this addition.
  TrampolineNopBranchRelay,
  /// An s_clause scheduling hint neutralized because one of its covered
  /// memory instructions is replaced by an instrumentation trampoline.
  InlineScalarClauseNopRewrite,
};

enum class ConSanPatchPhase : uint8_t {
  Mutation,
  Instrumentation,
};

enum class ConSanOrdinaryMemoryOperation : uint8_t {
  Load,
  Store,
};

/// A typed result for the bounded ordinary-memory decode used by fault
/// sequence discovery. Unsupported records remain in inventory; consumers
/// must never infer support merely from a mnemonic match.
enum class ConSanOrdinaryMemorySupportReason : uint8_t {
  NotApplicable,
  Supported,
  SupportedSynchronizationOnly,
  UnsupportedArchitecture,
  UnsupportedEncodingSize,
  MalformedEncoding,
  MissingAddressVgpr,
  MissingDestinationVgpr,
  MissingValueVgpr,
};

struct ConSanOrdinaryMemorySite {
  ConSanOrdinaryMemoryOperation operation = ConSanOrdinaryMemoryOperation::Load;
  ConSanOrdinaryMemorySupportReason support_reason =
      ConSanOrdinaryMemorySupportReason::UnsupportedArchitecture;
  // FLAT ordinary instructions can be the load/store half of a
  // language-level atomic sequence. Preserve their segment provenance so
  // synchronization analysis can distinguish a bounded group-FLAT sequence
  // from an otherwise identical global/private instruction.
  ConSanFlatAddressSpaceHint flat_address_space_hint = ConSanFlatAddressSpaceHint::Unknown;
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
  uint32_t width_bits = 0;
  std::optional<uint16_t> destination_vgpr;
  std::optional<uint16_t> address_vgpr;
  std::optional<uint16_t> address_sgpr;
  std::optional<uint16_t> value_vgpr;
  std::optional<uint32_t> raw_saddr;
  std::optional<uint32_t> raw_nv;
  std::optional<bool> raw_scale_offset;
  std::optional<uint32_t> raw_sve;
  std::optional<uint32_t> raw_rsrc;
  std::optional<uint32_t> raw_soffset;
  std::optional<bool> raw_offen;
  std::optional<bool> raw_idxen;
  std::optional<uint32_t> raw_vaddr;
  std::optional<uint32_t> raw_vsrc;
  std::optional<uint32_t> raw_vdst;
  std::optional<int32_t> raw_ioffset;
  /// Target-facing diagnostic scope code. This is the encoded operand when
  /// one exists and retains the legacy numeric spelling for implicit forms.
  /// Semantic consumers must use `scope`.
  std::optional<uint32_t> raw_scope;
  std::optional<ConSanMemoryScope> scope;
  std::optional<uint32_t> raw_th;
  /// Target-normalized exact ordering qualifier for a compiler-generated
  /// workgroup group-FLAT acquire. The raw fields remain available for
  /// diagnostics and mutation proof, but common synchronization analysis does
  /// not reinterpret them by target family.
  bool workgroup_acquire_ordering = false;
  std::string mnemonic;
};

struct ConSanBarrierSite {
  enum class Operation : uint8_t {
    Unknown,
    Signal,
    Wait,
    Full,
    Init,
    Join,
    Leave,
    Wakeup,
    StateQuery,
  } operation = Operation::Unknown;
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
  std::optional<int32_t> barrier_id;
  enum class OperandSource : uint8_t {
    Unknown,
    Immediate,
    DynamicM0,
    StaticM0Literal32,
    Literal32,
    Literal64,
  } operand_source = OperandSource::Unknown;
  std::optional<uint32_t> raw_operand_selector;
  std::optional<uint32_t> literal_width_bits;
  std::optional<uint64_t> literal_value;
  std::optional<uint16_t> raw_simm16;
  enum class Scope : uint8_t {
    Unknown,
    Workgroup,
    Cluster,
  } scope = Scope::Unknown;
  enum class ParticipantEncoding : uint8_t {
    Unavailable,
    M0Literal32,
  } participant_encoding = ParticipantEncoding::Unavailable;
  std::optional<uint32_t> participant_count;
  std::optional<uint64_t> participant_mask;
  std::optional<uint64_t> participant_setup_text_offset;
  std::optional<uint64_t> participant_setup_file_offset;
  std::optional<uint32_t> participant_setup_literal;
  std::string mnemonic;
};

struct ConSanFenceSite {
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
  ConSanCacheOperation cache_operation = ConSanCacheOperation::Unsupported;
  bool ordinary_acquire_mutation_supported = false;
  std::string mnemonic;
};

struct ConSanAtomicSite {
  ConSanAtomicAddressSpaceHint address_space_hint = ConSanAtomicAddressSpaceHint::Unknown;
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
  uint32_t width_bits = 0;
  std::optional<uint16_t> dst_vgpr;
  std::optional<uint16_t> addr_vgpr;
  std::optional<uint16_t> data_vgpr;
  std::optional<uint16_t> saddr_sgpr;
  std::optional<uint32_t> raw_op;
  std::optional<uint32_t> raw_addr;
  std::optional<uint32_t> raw_data0;
  std::optional<uint32_t> raw_data1;
  std::optional<uint32_t> raw_vdst;
  std::optional<uint32_t> raw_saddr;
  std::optional<bool> raw_scale_offset;
  std::optional<uint32_t> raw_vaddr;
  std::optional<uint32_t> raw_vsrc;
  std::optional<uint32_t> raw_vdata;
  std::optional<uint32_t> raw_rsrc;
  std::optional<uint32_t> raw_soffset;
  std::optional<bool> raw_offen;
  std::optional<bool> raw_idxen;
  std::optional<int32_t> raw_ioffset;
  /// Target-facing diagnostic scope code. This is the encoded operand when
  /// one exists and retains the legacy numeric spelling for implicit forms.
  /// Semantic consumers must use `scope`; diagnostics and mutation proof may
  /// retain this target-owned presentation value.
  std::optional<uint32_t> raw_scope;
  /// Target-normalized scope. Target forms with an implicit scope publish the
  /// same semantic value as forms carrying an encoded scope field.
  std::optional<ConSanMemoryScope> scope;
  std::optional<uint32_t> raw_th;
  std::optional<bool> returns_old_value;
  std::string mnemonic;
};

[[nodiscard]] constexpr bool consan_atomic_is_compare_exchange(const ConSanAtomicSite &site) {
  return site.mnemonic.find("cmpswap") != std::string::npos ||
         site.mnemonic.find("cmpxchg") != std::string::npos;
}

/// Kind of original program container through which an instruction was
/// decoded.
///
/// Multiple symbols can alias the same physical instruction. The container is
/// therefore attribution and ownership evidence, not part of
/// `PhysicalSiteId` and not permission to instrument the instruction twice.
enum class ConSanProgramContainerKind : uint8_t {
  /// A dispatchable kernel descriptor and its executable text range.
  Kernel,
  /// A non-kernel function symbol, commonly a helper reached by one or more
  /// kernels.
  Function,
  /// One past the last kind. This is an iteration bound and never a valid
  /// container reference.
  Count,
};

/// Attribution reference from a semantic site to the symbol range through
/// which it was decoded.
///
/// Kernel descriptor identity is optional because function aliases do not own
/// a descriptor. Execution owners are kept separately: a shared function can
/// be decoded through one symbol yet execute for several kernel descriptors.
struct ConSanProgramContainerRef {
  /// Whether the symbol is a dispatchable kernel or a helper function.
  ConSanProgramContainerKind kind = ConSanProgramContainerKind::Count;
  /// Symbol name retained for diagnostics and symbol-attributed inventory lookup.
  std::string name;
  /// Original executable-text entry offset of the symbol range.
  uint64_t entry_text_offset = 0;
  /// File offset corresponding to `entry_text_offset` in the pristine image.
  uint64_t text_file_offset = 0;
  /// Whether a zero-sized symbol forced the parser to infer this range's end.
  bool code_size_inferred_from_zero = false;
  /// Descriptor file offset for a kernel; absent for a function.
  std::optional<uint64_t> kernel_descriptor_file_offset;
  /// Whether this container's code observes the target-provided cluster
  /// workgroup coordinate. This is decoded program behavior, not a product
  /// test; targets without cluster facilities always leave it false.
  bool uses_cluster_workgroup_id = false;

  /// Return whether this attribution record names a dispatchable kernel. The
  /// result is derived from `kind`, so no parallel boolean can disagree.
  [[nodiscard]] bool is_kernel() const { return kind == ConSanProgramContainerKind::Kernel; }

  bool operator==(const ConSanProgramContainerRef &) const = default;
};

/// Instruction family from which one normalized access fact originated.
///
/// This is decoder provenance, not engine applicability. `DirectToLds` names
/// a global-memory instruction whose architectural destination is LDS; it is
/// kept distinct because its address and payload operands differ from an
/// ordinary DS instruction even though both access group memory.
enum class ConSanAccessOrigin : uint8_t {
  /// Native DS/LDS instruction.
  NativeLds,
  /// FLAT instruction with separately classified address-space provenance.
  Flat,
  /// Global-memory operation carrying a direct-to-LDS modifier.
  DirectToLds,
  /// One past the last origin. This is an iteration bound and never a decoded
  /// access origin.
  Count,
};

/// Decoder operands retained by normalized access inventory and native
/// lowering.
///
/// Optional fields preserve “not encoded/not decoded” rather than inventing a
/// register or immediate. Raw FLAT fields remain target-neutral facts here;
/// interpreting whether a particular combination is instrumentable belongs to
/// engine policy and target lowering. The MOI lowering candidate embeds this
/// same record rather than copying selected operands into a parallel schema.
struct ConSanAccessOperandFacts {
  std::optional<uint16_t> destination_vgpr;
  std::optional<uint16_t> destination_accvgpr;
  std::optional<uint16_t> address_vgpr;
  /// Target-normalized scalar half of a scalar-vector FLAT address. The raw
  /// SADDR encoding remains below for exact lowering validation; absence here
  /// means that the target decoder identified its vector-only sentinel.
  std::optional<uint16_t> scalar_address_sgpr;
  std::optional<uint16_t> data_vgpr;
  std::optional<uint16_t> second_data_vgpr;
  std::optional<uint32_t> raw_op;
  std::optional<uint32_t> raw_saddr;
  std::optional<bool> raw_scale_offset;
  std::optional<uint32_t> raw_vaddr;
  std::optional<uint32_t> raw_vsrc;
  std::optional<uint32_t> raw_vdst;
  std::optional<int32_t> raw_ioffset;
  std::optional<uint32_t> raw_segment;
  std::optional<uint32_t> raw_scope;
  std::optional<uint32_t> raw_th;

  bool operator==(const ConSanAccessOperandFacts &) const = default;
};

/// Shared access-stage record, completed by `ProgramInventoryBuilder` after
/// instruction decode and cross-function provenance relay.
///
/// The forward declaration lets decoder helpers append directly to the
/// builder-owned inventory collection without reviving container-owned LDS or
/// FLAT representations. The complete type and its construction-state
/// contract are declared by the program-inventory layer later in `consan.h`.
struct ConSanAccessInventorySite;

/// Immutable original-program facts and decoded semantic inventory for one
/// dispatchable kernel.
///
/// The record belongs to `ProgramInventory`: descriptor and metadata fields
/// describe the pristine code object, while the site vectors describe decoded
/// guest instructions before any ConSan policy or lowering decision. It owns
/// no report address, register assignment, patch, or transformed bytes. Later
/// stages may therefore use it to derive policy and capacity without reopening
/// the ELF or depending on mutable lowering telemetry.
struct ConSanKernelInfo {
  std::string name;
  uint64_t descriptor_file_offset = 0;
  /// Fixed group-segment bytes declared by the pristine kernel descriptor.
  /// Absence means the descriptor bytes were not safely readable; zero is a
  /// valid declaration and remains distinct from missing evidence.
  std::optional<uint32_t> declared_group_segment_bytes;
  uint64_t entry_text_offset = 0;
  uint64_t text_file_offset = 0;
  uint64_t code_size = 0;
  bool code_size_inferred_from_zero = false;
  bool has_text_range = false;
  bool has_dynamic_lds = false;
  std::optional<bool> uses_dynamic_stack;
  std::optional<uint16_t> vgpr_count;
  std::optional<uint16_t> agpr_count;
  std::optional<uint16_t> sgpr_count;
  std::optional<std::array<uint32_t, 3>> required_workgroup_size;
  bool decoded = false;
  bool uses_cluster_workgroup_id = false;
  ConSanKernelStats stats;
  std::vector<ConSanOrdinaryMemorySite> ordinary_memory_sites;
  std::vector<ConSanBarrierSite> barrier_sites;
  std::vector<ConSanFenceSite> fence_sites;
  std::vector<ConSanAtomicSite> atomic_sites;
  ConSanPreflightAction preflight_action = ConSanPreflightAction::NotRun;
  std::vector<std::string> preflight_reasons;
};

struct ConSanFunctionInfo {
  std::string name;
  uint64_t entry_text_offset = 0;
  uint64_t text_file_offset = 0;
  uint64_t code_size = 0;
  bool code_size_inferred_from_zero = false;
  bool decoded = false;
  bool uses_cluster_workgroup_id = false;
  ConSanKernelStats stats;
  std::vector<ConSanOrdinaryMemorySite> ordinary_memory_sites;
  std::vector<ConSanBarrierSite> barrier_sites;
  std::vector<ConSanFenceSite> fence_sites;
  std::vector<ConSanAtomicSite> atomic_sites;
};

/// Build the stable symbol attribution shared by every access decoded through
/// a dispatchable kernel. Code-object identity and execution ownership are
/// deliberately absent: the inventory publication stage owns those facts.
[[nodiscard]] inline ConSanProgramContainerRef
consan_program_container_ref(const ConSanKernelInfo &kernel) {
  return {
      .kind = ConSanProgramContainerKind::Kernel,
      .name = kernel.name,
      .entry_text_offset = kernel.entry_text_offset,
      .text_file_offset = kernel.text_file_offset,
      .code_size_inferred_from_zero = kernel.code_size_inferred_from_zero,
      .kernel_descriptor_file_offset = kernel.descriptor_file_offset,
      .uses_cluster_workgroup_id = kernel.uses_cluster_workgroup_id,
  };
}

/// Build the stable symbol attribution shared by every access decoded through
/// a non-dispatchable function. A function never invents a kernel descriptor;
/// execution-owner analysis associates its completed accesses separately.
[[nodiscard]] inline ConSanProgramContainerRef
consan_program_container_ref(const ConSanFunctionInfo &function) {
  return {
      .kind = ConSanProgramContainerKind::Function,
      .name = function.name,
      .entry_text_offset = function.entry_text_offset,
      .text_file_offset = function.text_file_offset,
      .code_size_inferred_from_zero = function.code_size_inferred_from_zero,
      .kernel_descriptor_file_offset = std::nullopt,
      .uses_cluster_workgroup_id = function.uses_cluster_workgroup_id,
  };
}

/// Byte ranges and stable identity of one committed patch. This is the only
/// slice needed to publish an intent-bound lowering location.
struct ConSanCommittedPatchGeometry {
  ConSanPatchPhase phase = ConSanPatchPhase::Instrumentation;
  ConSanPatchKind kind = ConSanPatchKind::InlineNopRewrite;
  uint64_t anchor_offset = 0;
  uint64_t trampoline_offset = 0;
  uint32_t original_size = 0;
  uint32_t trampoline_size = 0;
  /// Exact replacement-text location of a guest instruction relocated into
  /// this instrumentation body. Mutation composition uses this provenance
  /// instead of assuming a fixed position within a variable-sized probe.
  std::optional<uint64_t> relocated_guest_instruction_offset;
};

/// Exact control-flow routing proof retained for final validation. Emitters
/// populate this slice, but semantic policy and runtime reporting never see it.
struct ConSanPatchRoutingProof {
  /// Branch-relay words owned by this patch, either in a dead tail of its
  /// replaced anchor or after its primary trampoline body. Relay-reservoir
  /// validation uses this exact provenance when proving routes.
  std::vector<uint64_t> branch_relay_offsets;
  /// Exact forward/return route for a trampoline body whose continuation uses
  /// branch-only control flow. The entry variant prevents ordinary relays, a
  /// borrowed scalar tuple, and a composed kernel prologue from coexisting.
  std::optional<ConSanBranchOnlyContinuation> branch_only_route;
  /// Trampoline reached after this entry prologue completes. This is recorded
  /// on the prologue patch so final validation can prove composed entry
  /// ownership without treating the overwritten anchor branch as live.
  std::optional<uint64_t> entry_prologue_chained_trampoline_offset;
  /// Full dispatch-ID initialization bodies reached through paired
  /// kernarg-preload entry relays. Absent when initialization is inline in
  /// the two 256-byte hardware entry slots.
  std::optional<uint64_t> dispatch_id_primary_prologue_offset;
  std::optional<uint64_t> dispatch_id_secondary_prologue_offset;
};

/// Validation-only description of a deliberately mutated guest operation.
struct ConSanPatchMutationProof {
  /// Whole-pair markerless barrier moves record whether the exact barrier
  /// bytes precede or follow the displaced destination instruction.
  std::optional<ConSanBarrierMoveDirection> barrier_move_direction;
  ConSanBarrierMoveCfgContract barrier_move_cfg_contract = ConSanBarrierMoveCfgContract::SameBlock;
  std::optional<uint32_t> structured_guard_block_index;
  std::optional<uint32_t> structured_destination_block_index;
  std::optional<uint32_t> structured_source_block_index;
  std::optional<uint64_t> structured_guard_offset;
  std::optional<uint64_t> structured_destination_offset;
  std::optional<uint64_t> structured_source_offset;
  std::optional<ConSanPerturbationEdge> perturbation_edge;
  std::string perturbation_sequence_identity;
  std::optional<uint32_t> original_participant_count;
  std::optional<uint32_t> target_participant_count;
  /// Pristine identity and ownership retained only for a staged barrier
  /// mutation plus perturbation proof. The ordinary single-stage path leaves
  /// these fields empty.
  std::string perturbation_source_candidate_identity;
  std::string perturbation_source_sequence_identity;
  std::string perturbation_source_container_name;
  std::string perturbation_source_anchor_identity;
  bool perturbation_source_in_kernel = true;
  uint64_t perturbation_source_anchor_offset = 0;
  uint32_t perturbation_source_anchor_size = 0;
  std::optional<uint64_t> perturbation_source_owner_descriptor_file_offset;
  bool perturbation_composite_atomic_overlap = false;
  bool perturbation_composite_removed_boundary = false;
};

/// Register, private-memory, group-memory, and descriptor effects committed by
/// instrumentation. Descriptor mutation consumes this slice independently of
/// byte geometry and validation identities.
struct ConSanPatchAbiEffects {
  std::optional<uint16_t> scratch_vgpr;
  /// SuperCollider-only preservation of a borrowed live scalar VCC-save
  /// range. One optional owns the scalar source, vector reservoir, and exact
  /// preservation strategy; other engines leave it absent.
  std::optional<ConSanSuperColliderScalarVccSpill> sc_scalar_vcc_spill;
  /// Complete vector-register state initialized or consumed by this patch.
  /// The typed lifetime distinguishes persistent state from transient
  /// owner/epoch scratch and prevents an incomplete pair or contradictory
  /// owner-local/ABI flags.
  std::optional<ConSanMoiVgprStateEffect> moi_vgpr_state;
  /// Scalar-persistent state initialized by this entry prologue. Access and
  /// synchronization patches consume this exact ABI; other patch kinds leave
  /// the aggregate empty.
  ConSanMoiPersistentSgprState persistent_sgpr_state;
  /// Entry-local carrier and contiguous guest-SGPR window preserved while
  /// persistent vector state is initialized. The carrier is dead again before
  /// any displaced guest instruction executes.
  std::optional<ConSanMoiEntryScalarBackup> entry_scalar_backup;
  /// Minimum ordinary SGPR allocation required by this patch, independent of
  /// whether its placement uses an indirect control-flow relay.
  uint16_t required_sgpr_count = 0;
  /// Exact route of an ordinary, non-dense SuperCollider indirect body.
  std::optional<ConSanSuperColliderIndirectBodyRoute> sc_indirect_body_route;
  /// Mutually exclusive wave32-VCC or ordinary-scalar route used by an
  /// indirect SuperCollider relay reservoir.
  /// Entry island paired with this MOI dense dispatcher. This explicit
  /// relation lets later synchronization composition reuse an access router
  /// without depending on incidental result.patches ordering.
  std::optional<uint64_t> moi_dense_entry_island_offset;
  /// Exact SuperCollider-only dispatcher route used by this coordinator or
  /// body. Other modes and non-dense SuperCollider patches leave it absent.
  std::optional<ConSanSuperColliderDenseRouteEffect> sc_dense_route;
  /// Complete owner-local persistent identity layout and the distinct,
  /// target-normalized base available to ephemeral spill state.
  std::optional<ConSanMoiPrivateStateLayout> private_state_layout;
  /// Exact preload transformation and unique persistent register
  /// representation proved by an emitted dispatch-ID entry prologue.
  std::optional<ConSanMoiDispatchIdPrologueEffect> dispatch_id_prologue;
  uint16_t spilled_vgpr_count = 0;
  /// Absolute per-workitem private-segment minimum encoded into the owning
  /// kernel descriptor.
  uint32_t required_private_segment_size = 0;
  /// Site-local frame depth added above a launch's runtime-selected dynamic
  /// stack extent. Multiple alternative patches contribute their maximum
  /// addend rather than a sum because their frames are never simultaneously
  /// active.
  uint32_t dynamic_private_segment_addend = 0;
  /// Exact workgroup-local shadow ABI emitted by this patch. Keeping the typed
  /// layout intact prevents prologue composition and final validation from
  /// reconstructing it from a flattened mode-specific field replica.
  std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow;
  [[nodiscard]] uint32_t required_group_segment_size() const {
    return workgroup_shadow ? workgroup_shadow->required_group_segment_size : 0u;
  }
  /// Kernel descriptors whose dispatches can execute this patch. Shared
  /// helper patches name every reachable owner.
  std::vector<uint64_t> owner_descriptor_file_offsets;
};

/// Placement's complete mechanical output: control-flow proof plus the ABI
/// effects that make the emitted route executable by every owner.
struct ConSanPatchPlacementEffects : ConSanPatchRoutingProof, ConSanPatchAbiEffects {};

/// Complete product visible to ordinary lowering composition. Validation-only
/// mutation and fault identities are deliberately absent from this contract.
struct ConSanPatchLoweringProduct : ConSanCommittedPatchGeometry, ConSanPatchPlacementEffects {};

/// Pristine identities and decoded operands used only to validate fault
/// mutations against the original code object.
struct ConSanPatchFaultProof {
  /// Pristine identities retained for exact ordinary-load mutation proof.
  std::string fault_primary_identity;
  std::string fault_companion_identity;
  std::string fault_sequence_identity;
  /// Decoder-derived source and explicitly requested replacement operands for
  /// an LDS address mutation. Final validation re-derives the source from the
  /// pristine object and proves that no other instruction field changed.
  std::optional<uint16_t> fault_original_address_vgpr;
  std::optional<uint16_t> fault_target_address_vgpr;
};

/// Product visible to staged mutation composition. It can translate and
/// reserve mutation geometry without seeing pristine fault identities.
struct ConSanPatchMutationProduct : ConSanPatchLoweringProduct, ConSanPatchMutationProof {};

/// Private construction record joining the independently owned patch products.
/// Consumers should accept the narrow base contract they need; only final
/// validation is expected to inspect the complete joined proof.
struct ConSanPatchInfo : ConSanPatchMutationProduct, ConSanPatchFaultProof {};

/// Image-level proof for one whole-text relocation transaction. This is not a
/// semantic site patch: it accounts for the active relocated bodies while the
/// ordinary patch inventory continues to describe only instrumented sites.
struct ConSanTextRelocationProof {
  uint64_t source_text_size = 0;
  uint64_t relocated_text_size = 0;
};
