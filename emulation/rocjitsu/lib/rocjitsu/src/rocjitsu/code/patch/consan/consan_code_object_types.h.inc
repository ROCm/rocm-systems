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

enum class ConSanLdsAccessKind : uint8_t {
  Read,
  Write,
  Atomic,
  Other,
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
  InlineLdsLoadCheckTrap,
  InlineLdsStoreCheckTrap,
  LocalCaveLdsLoadCheckTrap,
  LocalCaveLdsStoreCheckTrap,
  InlineFlatLoadCheckTrap,
  InlineFlatStoreCheckTrap,
  LocalCaveFlatLoadCheckTrap,
  LocalCaveFlatStoreCheckTrap,
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
  TrampolineScBranchRelayDonor,
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

struct ConSanLdsSite {
  ConSanLdsAccessKind kind = ConSanLdsAccessKind::Other;
  bool supported_mvp = false;
  /// A MUBUF load whose LDS modifier redirects the global payload to a
  /// physical-lane slot at M0 rather than defining an ordinary VGPR result.
  bool direct_to_lds = false;
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
  uint32_t width_bits = 0;
  std::optional<uint16_t> dst_vgpr;
  std::optional<uint16_t> dst_accvgpr;
  std::optional<uint16_t> addr_vgpr;
  std::optional<uint16_t> data_vgpr;
  std::optional<uint16_t> second_data_vgpr;
  std::string mnemonic;
  /// Kernel descriptors whose dispatches can execute this site. Shared
  /// helpers name every reachable owner after execution-owner analysis.
  std::vector<uint64_t> owner_descriptor_file_offsets;
};

struct ConSanFlatSite {
  ConSanLdsAccessKind kind = ConSanLdsAccessKind::Other;
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
  uint32_t width_bits = 0;
  std::optional<uint16_t> dst_vgpr;
  std::optional<uint16_t> addr_vgpr;
  std::optional<uint16_t> data_vgpr;
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
  ConSanFlatAddressSpaceHint address_space_hint = ConSanFlatAddressSpaceHint::Unknown;
  std::vector<uint64_t> owner_descriptor_file_offsets;
  std::string mnemonic;
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
  std::optional<uint32_t> raw_scope;
  std::optional<uint32_t> raw_th;
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
  /// Common scope code. This preserves an encoded field when one exists; for
  /// target forms whose scope is implicit, analysis records the qualified
  /// semantic scope so downstream engines do not invent target-specific rules.
  std::optional<uint32_t> raw_scope;
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
  /// Symbol name retained for diagnostics and temporary legacy lookup.
  std::string name;
  /// Original executable-text entry offset of the symbol range.
  uint64_t entry_text_offset = 0;
  /// Descriptor file offset for a kernel; absent for a function.
  std::optional<uint64_t> kernel_descriptor_file_offset;
  /// Whether this container's code observes the gfx1250 cluster workgroup ID.
  bool uses_gfx1250_cluster_workgroup_id = false;

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

/// Complete iterable set of decoded access origins.
inline constexpr std::array<ConSanAccessOrigin, 3> kConSanAccessOrigins = {
    ConSanAccessOrigin::NativeLds,
    ConSanAccessOrigin::Flat,
    ConSanAccessOrigin::DirectToLds,
};

/// Return the stable diagnostic spelling of an access-origin fact.
[[nodiscard]] constexpr std::string_view consan_access_origin_name(ConSanAccessOrigin origin) {
  switch (origin) {
  case ConSanAccessOrigin::NativeLds:
    return "native-lds";
  case ConSanAccessOrigin::Flat:
    return "flat";
  case ConSanAccessOrigin::DirectToLds:
    return "direct-to-lds";
  case ConSanAccessOrigin::Count:
    break;
  }
  return "invalid-access-origin";
}

static_assert(kConSanAccessOrigins.size() == static_cast<size_t>(ConSanAccessOrigin::Count));

/// Decoder operands retained by normalized access inventory and temporary
/// compatibility lowering.
///
/// Optional fields preserve “not encoded/not decoded” rather than inventing a
/// register or immediate. Raw FLAT fields remain target-neutral facts here;
/// interpreting whether a particular combination is instrumentable belongs to
/// engine policy and target lowering. The temporary MOI candidate embeds this
/// same record rather than copying selected operands into a parallel schema.
struct ConSanAccessOperandFacts {
  std::optional<uint16_t> destination_vgpr;
  std::optional<uint16_t> destination_accvgpr;
  std::optional<uint16_t> address_vgpr;
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
  bool uses_gfx1250_cluster_workgroup_id = false;
  ConSanKernelStats stats;
  std::vector<ConSanLdsSite> lds_sites;
  std::vector<ConSanFlatSite> flat_sites;
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
  bool uses_gfx1250_cluster_workgroup_id = false;
  ConSanKernelStats stats;
  std::vector<ConSanLdsSite> lds_sites;
  std::vector<ConSanFlatSite> flat_sites;
  std::vector<ConSanOrdinaryMemorySite> ordinary_memory_sites;
  std::vector<ConSanBarrierSite> barrier_sites;
  std::vector<ConSanFenceSite> fence_sites;
  std::vector<ConSanAtomicSite> atomic_sites;
};

struct ConSanPatchInfo {
  ConSanPatchPhase phase = ConSanPatchPhase::Instrumentation;
  ConSanPatchKind kind = ConSanPatchKind::InlineNopRewrite;
  uint64_t anchor_offset = 0;
  uint64_t trampoline_offset = 0;
  uint32_t original_size = 0;
  uint32_t trampoline_size = 0;
  /// Branch-relay words owned by this patch, either in a dead tail of its
  /// replaced anchor or after its primary trampoline body. Relay-reservoir
  /// validation uses this exact provenance when proving routes.
  std::vector<uint64_t> branch_relay_offsets;
  /// Exact forward and return relay chains for a trampoline body whose
  /// continuation uses branch-only control flow. Ordinarily the forward chain
  /// also terminates at the trampoline. A borrowed indirect entry instead
  /// preserves its owner-local PC/SCC tuple in a liveness-proven dead VGPR and leaves
  /// the entry chain empty; the return chain still terminates after the
  /// displaced anchor.
  bool branch_only_continuation = false;
  bool branch_only_borrowed_indirect_entry = false;
  /// A liveness-proven dead VGPR used only to carry the borrowed scalar tuple
  /// from the expanded anchor into its appended body. This is intentionally
  /// distinct from `scratch_vgpr`: spill-backed probe scratch may overlap live
  /// guest values until the body saves them.
  std::optional<uint16_t> branch_only_borrowed_backup_vgpr;
  /// Target of the optional empty-EXEC bypass after a borrowed entry restores
  /// its scalar tuple. Vector-memory probes use it to skip instrumentation and
  /// resume at the relocated scalar continuation. Scalar barrier probes do not
  /// need an EXEC bypass and leave it absent.
  std::optional<uint64_t> branch_only_borrowed_continuation_offset;
  std::vector<uint64_t> branch_only_entry_relay_offsets;
  std::vector<uint64_t> branch_only_return_relay_offsets;
  /// Entry prologue that supersedes this patch's original anchor-to-body
  /// route. The prologue owns the hardware entry and chains directly into
  /// this trampoline after initializing persistent state.
  std::optional<uint64_t> branch_only_entry_prologue_offset;
  /// Trampoline reached after this entry prologue completes. This is recorded
  /// on the prologue patch so final validation can prove composed entry
  /// ownership without treating the overwritten anchor branch as live.
  std::optional<uint64_t> entry_prologue_chained_trampoline_offset;
  /// Full dispatch-ID initialization bodies reached through paired
  /// kernarg-preload entry relays. Absent when initialization is inline in
  /// the two 256-byte hardware entry slots.
  std::optional<uint64_t> dispatch_id_primary_prologue_offset;
  std::optional<uint64_t> dispatch_id_secondary_prologue_offset;
  /// Original synchronization events represented by one typed sequence body.
  /// Zero means the ordinary one-anchor/one-site accounting rule.
  uint32_t covered_sync_event_count = 0;
  /// Sampled access patches reserve a contiguous immutable bank beginning at
  /// this slot. Synchronization patches use the same dispatch/workgroup hash
  /// to select the matching dynamic window.
  uint32_t sampled_first_slot = 0;
  uint32_t sampled_window_bank_count = 0;
  ConSanLdsAccessKind sampled_access_kind = ConSanLdsAccessKind::Other;
  uint32_t sampled_access_range_count = 0;
  /// Exact replacement-text location of a guest instruction relocated into
  /// this instrumentation body. Mutation composition uses this provenance
  /// instead of assuming a fixed position within a variable-sized probe.
  std::optional<uint64_t> relocated_guest_instruction_offset;
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
  std::optional<uint16_t> scratch_vgpr;
  /// Ordinary scalar borrowed by a spill-backed VCC save. The scalar is
  /// preserved through scalar_vcc_spill_vgpr and is never a fresh allocation.
  std::optional<uint16_t> scalar_vcc_spill_sgpr;
  /// First VGPR used as a reservoir for a borrowed live scalar VCC-save pair.
  /// A non-spill patch uses fixed lanes in one VGPR. A private-spill patch uses
  /// two consecutive active-lane reservoirs so inactive lanes remain untouched.
  /// A dynamic-stack bootstrap uses two additional reservoirs for the incoming
  /// frame base and SCC.
  std::optional<uint16_t> scalar_vcc_spill_vgpr;
  /// Number of consecutive scalar-reservoir VGPRs beginning at
  /// scalar_vcc_spill_vgpr: one fixed-lane reservoir, two private-spill
  /// reservoirs, or four dynamic-stack bootstrap reservoirs. Zero means the
  /// scalar-pressure fallback was unused.
  uint16_t scalar_vcc_spill_vgpr_count = 0;
  std::optional<uint16_t> persistent_owner_vgpr;
  std::optional<uint16_t> persistent_epoch_vgpr;
  std::optional<uint16_t> persistent_workgroup_key_vgpr;
  ConSanMoiPersistentWorkgroupRegisters persistent_record_replay_workgroup_vgprs;
  /// Entry-local VGPR whose lanes preserve a runtime-selected transient SGPR
  /// window while persistent vector state is initialized. The register is
  /// dead again before any displaced guest instruction executes.
  std::optional<uint16_t> entry_scalar_backup_vgpr;
  /// First transient SGPR preserved in entry_scalar_backup_vgpr.
  std::optional<uint16_t> entry_scalar_backup_sgpr_base;
  /// Number of consecutive transient SGPRs preserved in fixed VGPR lanes.
  uint16_t entry_scalar_backup_sgpr_count = 0;
  /// Minimum ordinary SGPR allocation required by this patch, independent of
  /// whether its placement uses an indirect control-flow relay.
  uint16_t required_sgpr_count = 0;
  std::optional<uint16_t> indirect_pc_sgpr;
  std::optional<uint16_t> indirect_saved_scc_sgpr;
  std::optional<uint16_t> indirect_saved_vcc_sgpr;
  std::optional<uint64_t> indirect_return_offset;
  std::optional<uint16_t> indirect_return_pc_sgpr;
  /// Return-end SCC save used by a wave64 relay reservoir whose absolute
  /// branch scratch is an endpoint-dead ordinary SGPR pair. Wave32 reservoirs
  /// instead preserve SCC in inactive EXEC_HI and populate the VCC-save field.
  std::optional<uint16_t> indirect_return_saved_scc_sgpr;
  std::optional<uint16_t> indirect_return_saved_vcc_sgpr;
  uint16_t indirect_required_sgpr_count = 0;
  /// Entry island paired with this MOI dense dispatcher. This explicit
  /// relation lets later synchronization composition reuse an access router
  /// without depending on incidental result.patches ordering.
  std::optional<uint64_t> moi_dense_entry_island_offset;
  /// Shared SuperCollider dense-call dispatcher selected by this access body.
  /// A call-return pair is distinct from the absolute-jump scratch pair. A
  /// bounded explicit-key fallback instead reuses the body's dead VCC-save
  /// SGPR and keeps each two-word anchor spill-free.
  std::optional<uint64_t> sc_dense_dispatcher_offset;
  std::optional<uint16_t> sc_dense_call_return_sgpr;
  std::optional<uint16_t> sc_dense_explicit_key_sgpr;
  std::optional<uint32_t> persistent_epoch_private_offset;
  /// Entry-captured raw workitem identity used to derive a stable logical
  /// wave owner after the guest has reused v0.
  std::optional<uint32_t> persistent_owner_private_offset;
  /// Entry-captured exact workgroup identity for kernels whose ordinary VGPR
  /// bank has no lifetime-safe persistent slot below ACCUM_OFFSET.
  std::optional<uint32_t> persistent_workgroup_key_private_offset;
  /// Entry-captured hardware dispatch identity for an owner whose guest SGPR
  /// pressure excludes the object-wide persistent pair. The high dword is in
  /// the immediately following private slot.
  std::optional<uint32_t> persistent_dispatch_id_private_offset;
  /// Full Record/Replay coordinates captured at entry. Unlike the compact
  /// Inline key above, these offsets preserve x/y/z without packing limits.
  ConSanMoiPersistentWorkgroupPrivateOffsets persistent_record_replay_workgroup_private_offsets;
  /// First private byte available after all persistent MOI state.
  std::optional<uint32_t> persistent_private_state_end;
  /// Persistent instrumentation-owned register pair receiving the full
  /// hardware dispatch ID at kernel entry. Exactly one representation is
  /// present, and either proves that the prologue checked the preload map.
  std::optional<uint16_t> dispatch_id_capture_sgpr;
  std::optional<uint16_t> dispatch_id_capture_vgpr;
  uint16_t dispatch_id_source_sgpr = 0;
  uint16_t dispatch_id_original_user_sgpr_count = 0;
  uint16_t dispatch_id_expanded_user_sgpr_count = 0;
  uint16_t dispatch_id_system_sgpr_count = 0;
  uint16_t dispatch_id_kernarg_reload_sgpr = 0;
  uint16_t dispatch_id_kernarg_reload_base_sgpr = 0;
  uint16_t dispatch_id_kernarg_reload_offset_dwords = 0;
  uint16_t dispatch_id_kernarg_reload_count = 0;
  uint16_t dispatch_id_shifted_guest_sgpr_count = 0;
  uint16_t dispatch_id_shifted_system_sgpr_count = 0;
  uint16_t dispatch_id_system_sgpr_shift = 0;
  uint16_t dispatch_id_original_kernarg_preload_length = 0;
  uint16_t dispatch_id_replacement_kernarg_preload_length = 0;
  uint16_t dispatch_id_required_sgpr_count = 0;
  bool dispatch_id_preload_inserted = false;
  uint16_t spilled_vgpr_count = 0;
  /// Absolute per-workitem private-segment minimum encoded into the owning
  /// kernel descriptor.
  uint32_t required_private_segment_size = 0;
  /// Site-local frame depth added above a launch's runtime-selected dynamic
  /// stack extent. Multiple alternative patches contribute their maximum
  /// addend rather than a sum because their frames are never simultaneously
  /// active.
  uint32_t dynamic_private_segment_addend = 0;
  uint32_t workgroup_shadow_base = 0;
  uint32_t workgroup_shadow_size = 0;
  uint32_t workgroup_shadow_validity_base = 0;
  uint32_t workgroup_shadow_validity_size = 0;
  bool workgroup_shadow_lazy_initialization = false;
  bool workgroup_shadow_compact = false;
  uint16_t workgroup_shadow_compact_token = 0;
  uint32_t required_group_segment_size = 0;
  /// Kernel descriptors whose dispatches can execute this patch. Shared
  /// helper patches name every reachable owner.
  std::vector<uint64_t> owner_descriptor_file_offsets;
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
