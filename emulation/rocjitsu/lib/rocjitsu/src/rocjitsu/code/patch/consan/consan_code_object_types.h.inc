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

enum class ConSanPreflightAction : uint8_t {
  NotRun,
  Candidate,
  Skip,
  Reject,
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
  TrampolineScRelayReservoir,
  TrampolineMoiIndirectBranchIsland,
  TrampolineNop,
};

enum class ConSanPatchPhase : uint8_t {
  Mutation,
  Instrumentation,
};

struct ConSanLdsSite {
  ConSanLdsAccessKind kind = ConSanLdsAccessKind::Other;
  bool supported_mvp = false;
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
  std::optional<uint32_t> raw_vaddr;
  std::optional<uint32_t> raw_vsrc;
  std::optional<uint32_t> raw_vdata;
  std::optional<uint32_t> raw_rsrc;
  std::optional<uint32_t> raw_soffset;
  std::optional<bool> raw_offen;
  std::optional<bool> raw_idxen;
  std::optional<int32_t> raw_ioffset;
  std::optional<uint32_t> raw_scope;
  std::optional<uint32_t> raw_th;
  std::optional<bool> returns_old_value;
  std::string mnemonic;
};

[[nodiscard]] constexpr bool consan_atomic_is_compare_exchange(const ConSanAtomicSite &site) {
  return site.mnemonic.find("cmpswap") != std::string::npos ||
         site.mnemonic.find("cmpxchg") != std::string::npos;
}

enum class ConSanMoiCandidateSource : uint8_t {
  NativeLds,
  FlatGroup,
  FlatMaybeGroup,
};

struct ConSanMoiCandidate {
  std::string container_name;
  bool in_kernel = true;
  uint64_t container_entry_text_offset = 0;
  std::optional<uint16_t> gfx1250_vgpr_msb_mode;
  ConSanMoiCandidateSource source = ConSanMoiCandidateSource::NativeLds;
  ConSanLdsAccessKind kind = ConSanLdsAccessKind::Other;
  ConSanFlatAddressSpaceHint flat_address_space_hint = ConSanFlatAddressSpaceHint::Unknown;
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
  uint32_t width_bits = 0;
  std::optional<uint16_t> dst_vgpr;
  std::optional<uint16_t> addr_vgpr;
  std::optional<uint16_t> data_vgpr;
  std::optional<uint16_t> second_data_vgpr;
  std::optional<uint64_t> kernel_descriptor_file_offset;
  bool uses_gfx1250_cluster_workgroup_id = false;
  std::optional<uint32_t> raw_op;
  std::optional<uint32_t> raw_saddr;
  std::optional<uint32_t> raw_vaddr;
  std::optional<uint32_t> raw_vsrc;
  std::optional<uint32_t> raw_vdst;
  std::optional<int32_t> raw_ioffset;
  std::optional<uint32_t> raw_segment;
  std::optional<uint32_t> raw_scope;
  std::optional<uint32_t> raw_th;
  std::string mnemonic;
};

struct ConSanKernelInfo {
  std::string name;
  uint64_t descriptor_file_offset = 0;
  uint64_t entry_text_offset = 0;
  uint64_t text_file_offset = 0;
  uint64_t code_size = 0;
  bool code_size_inferred_from_zero = false;
  bool has_text_range = false;
  bool has_dynamic_lds = false;
  std::optional<bool> uses_dynamic_stack;
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
  /// Branch-relay words owned by this patch but placed after its primary
  /// trampoline body. Relay-reservoir validation uses this exact provenance
  /// when proving routes through appended text.
  std::vector<uint64_t> branch_relay_offsets;
  /// Exact forward and return relay chains for a SuperCollider body whose
  /// anchor and continuation use branch-only control flow.  These routes do
  /// not borrow an indirect PC pair: the forward chain terminates at the
  /// trampoline and the return chain terminates after the displaced anchor.
  bool sc_branch_only_continuation = false;
  std::vector<uint64_t> sc_branch_only_entry_relay_offsets;
  std::vector<uint64_t> sc_branch_only_return_relay_offsets;
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
  /// VGPR used as a fixed-lane reservoir for a borrowed live scalar VCC-save
  /// pair. Presence proves the patch used the scalar-pressure fallback rather
  /// than a liveness-dead SGPR pair.
  std::optional<uint16_t> scalar_vcc_spill_vgpr;
  std::optional<uint16_t> persistent_owner_vgpr;
  std::optional<uint16_t> persistent_epoch_vgpr;
  std::optional<uint16_t> persistent_workgroup_key_vgpr;
  /// Minimum ordinary SGPR allocation required by this patch, independent of
  /// whether its placement uses an indirect control-flow relay.
  uint16_t required_sgpr_count = 0;
  std::optional<uint16_t> indirect_pc_sgpr;
  std::optional<uint16_t> indirect_saved_scc_sgpr;
  std::optional<uint16_t> indirect_saved_vcc_sgpr;
  std::optional<uint64_t> indirect_return_offset;
  std::optional<uint16_t> indirect_return_pc_sgpr;
  std::optional<uint16_t> indirect_return_saved_vcc_sgpr;
  uint16_t indirect_required_sgpr_count = 0;
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
  /// First private byte available after all persistent MOI state.
  std::optional<uint32_t> persistent_private_state_end;
  /// Persistent instrumentation-owned SGPR pair receiving the full hardware
  /// dispatch ID at kernel entry. Presence also proves that this prologue
  /// checked the descriptor's preload map.
  std::optional<uint16_t> dispatch_id_capture_sgpr;
  uint16_t dispatch_id_source_sgpr = 0;
  uint16_t dispatch_id_original_user_sgpr_count = 0;
  uint16_t dispatch_id_expanded_user_sgpr_count = 0;
  uint16_t dispatch_id_system_sgpr_count = 0;
  uint16_t dispatch_id_kernarg_reload_sgpr = 0;
  uint16_t dispatch_id_kernarg_reload_base_sgpr = 0;
  uint16_t dispatch_id_kernarg_reload_offset_dwords = 0;
  uint16_t dispatch_id_kernarg_reload_count = 0;
  uint16_t dispatch_id_shifted_system_sgpr_count = 0;
  uint16_t dispatch_id_system_sgpr_shift = 0;
  uint16_t dispatch_id_original_kernarg_preload_length = 0;
  uint16_t dispatch_id_replacement_kernarg_preload_length = 0;
  uint16_t dispatch_id_required_sgpr_count = 0;
  bool dispatch_id_preload_inserted = false;
  uint16_t spilled_vgpr_count = 0;
  uint32_t required_private_segment_size = 0;
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
};
