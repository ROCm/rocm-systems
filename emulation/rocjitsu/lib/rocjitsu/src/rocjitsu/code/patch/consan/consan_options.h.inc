namespace rocjitsu {

/// Default maximum additional ELF storage one ConSan transformation may
/// materialize, including alignment padding. Dense production code objects can
/// require one out-of-line body per admitted access, so retain the established
/// 192 MiB default until workload data supports changing it.
inline constexpr uint64_t kConSanDefaultMaxPatchedImageGrowthBytes = uint64_t{192} * 1024 * 1024;

enum class ConSanPatchedImageGrowthLimitKind : uint8_t {
  AbsoluteBytes,
  InputPercent,
};

/// Resource policy for one complete ConSan code-object transformation.
///
/// Both forms bound total additional ELF bytes across one complete code-object
/// transformation. The relative form resolves against the original input image
/// rather than an intermediate image produced by an earlier ConSan stage. It is
/// an alternative to the absolute form rather than a second silently combined
/// cap.
struct ConSanPatchedImageGrowthLimit {
  ConSanPatchedImageGrowthLimitKind kind = ConSanPatchedImageGrowthLimitKind::AbsoluteBytes;
  uint64_t absolute_bytes = kConSanDefaultMaxPatchedImageGrowthBytes;
  uint32_t input_percent = 0;

  auto operator<=>(const ConSanPatchedImageGrowthLimit &) const = default;
};

/// Resolve a configured policy without overflowing size_t. An invalid enum
/// value is rejected instead of silently selecting a fallback.
[[nodiscard]] inline std::optional<size_t>
consan_patched_image_growth_limit_bytes(const ConSanPatchedImageGrowthLimit &policy,
                                        size_t input_image_bytes) {
  static_assert(sizeof(size_t) >= sizeof(uint64_t),
                "ConSan patched-image accounting requires a 64-bit size_t");
  switch (policy.kind) {
  case ConSanPatchedImageGrowthLimitKind::AbsoluteBytes:
    return static_cast<size_t>(policy.absolute_bytes);
  case ConSanPatchedImageGrowthLimitKind::InputPercent: {
    const size_t whole = input_image_bytes / 100u;
    const size_t remainder = input_image_bytes % 100u;
    if (whole != 0u && policy.input_percent > std::numeric_limits<size_t>::max() / whole) {
      return std::numeric_limits<size_t>::max();
    }
    const size_t whole_bytes = whole * policy.input_percent;
    const size_t remainder_bytes =
        static_cast<size_t>((static_cast<uint64_t>(remainder) * policy.input_percent) / 100u);
    if (remainder_bytes > std::numeric_limits<size_t>::max() - whole_bytes)
      return std::numeric_limits<size_t>::max();
    return whole_bytes + remainder_bytes;
  }
  }
  return std::nullopt;
}

struct ConSanPatchedImageGrowthBudget {
  size_t input_image_bytes = 0;
  size_t current_image_bytes = 0;
  size_t total_limit_bytes = 0;
  size_t existing_growth_bytes = 0;
  size_t remaining_growth_bytes = 0;
  bool already_exceeded = false;
};

/// Resolve the remaining budget for a transformation stage against the
/// original code-object image. This prevents successive stages from each
/// receiving a fresh full allowance.
[[nodiscard]] inline std::optional<ConSanPatchedImageGrowthBudget>
consan_patched_image_growth_budget(const ConSanPatchedImageGrowthLimit &policy,
                                   size_t input_image_bytes, size_t current_image_bytes) {
  const auto total_limit = consan_patched_image_growth_limit_bytes(policy, input_image_bytes);
  if (!total_limit)
    return std::nullopt;
  const size_t existing_growth =
      current_image_bytes > input_image_bytes ? current_image_bytes - input_image_bytes : 0u;
  return ConSanPatchedImageGrowthBudget{
      .input_image_bytes = input_image_bytes,
      .current_image_bytes = current_image_bytes,
      .total_limit_bytes = *total_limit,
      .existing_growth_bytes = existing_growth,
      .remaining_growth_bytes =
          existing_growth > *total_limit ? 0u : *total_limit - existing_growth,
      .already_exceeded = existing_growth > *total_limit,
  };
}

enum class ConSanFlavor : uint8_t {
  None,
  SuperCollider,
  Moi,
};

enum class ConSanMoiEngine : uint8_t {
  RecordReplay,
  InlineShadow,
  Sampled,
};

enum class ConSanDelayMode : uint8_t {
  Nop,
  Sleep,
  SleepVar,
};

enum class ConSanBarrierMoveDirection : uint8_t {
  LegacyMarker,
  Earlier,
  Later,
};

/// CFG/convergence proof attached to a cross-block whole-barrier move.
enum class ConSanBarrierMoveCfgContract : uint8_t {
  SameBlock,
  /// Both successors of an acyclic conditional guard reach the reconverged
  /// source exactly once; moving the pair into the guard remains completing.
  CompletingStructuredDiamond,
  /// The destination is one optional EXEC-narrowed arm. This intentionally
  /// creates divergent participation and requires destructive X0 containment.
  DestructiveStructuredExecDiamond,
};

enum class ConSanPerturbationKind : uint8_t {
  None,
  Barrier,
  Atomic,
};

enum class ConSanPerturbationEdge : uint8_t {
  Release,
  Acquire,
};

/// Which cache boundary an atomic weaken-order fault removes. `Any` preserves
/// the historical first-qualified-boundary behavior; explicit edges are used
/// by fence-pair qualification and fail closed when that edge is absent.
enum class ConSanAtomicOrderEdge : uint8_t {
  Any,
  Release,
  Acquire,
};

/// Barrier-mutation forms whose evidence must not be conflated across host
/// code-object proof, live hardware, or multi-device/cluster topology.
enum class ConSanBarrierMutationForm : uint8_t {
  PairScopeCrossing,
  PairCompletingId,
  StaticNamedLifecycle,
};

enum class ConSanQualificationEvidence : uint8_t {
  None,
  Proven,
  DeferredA1,
};

struct ConSanBarrierMutationQualification {
  ConSanQualificationEvidence host = ConSanQualificationEvidence::None;
  ConSanQualificationEvidence live_gpu = ConSanQualificationEvidence::None;
  ConSanQualificationEvidence cluster_or_multi_device = ConSanQualificationEvidence::None;
};

/// Durable qualification boundary for the currently implemented barrier
/// mutation slices. This is deliberately narrower than encoder availability:
/// successful synthetic patching cannot silently become a live-GPU or
/// topology claim.
[[nodiscard]] constexpr ConSanBarrierMutationQualification
consan_barrier_mutation_qualification(std::string_view target, ConSanBarrierMutationForm form) {
  if (target == "gfx1201" && form == ConSanBarrierMutationForm::PairScopeCrossing)
    return {.host = ConSanQualificationEvidence::Proven};
  if (target == "gfx1250" && form == ConSanBarrierMutationForm::StaticNamedLifecycle) {
    return {.host = ConSanQualificationEvidence::Proven,
            .live_gpu = ConSanQualificationEvidence::DeferredA1};
  }
  return {};
}
enum class ConSanMoiOwnerSource : uint8_t {
  WorkitemId,
  HwId,
};

/// Controls whether flat/VFLAT sites whose group provenance is heuristic are
/// eligible for instrumentation.
enum class ConSanFlatProvenanceMode : uint8_t {
  Likely,
  Strict,
};

enum class ConSanRegisterAllocationSource : uint8_t {
  Unsupported,
  Explicit,
  LivenessDead,
  DescriptorGrowth,
  SpillRequired,
};

enum class ConSanRegisterPlanReason : uint8_t {
  None,
  InvalidRequest,
  ExplicitMisaligned,
  ExplicitOutOfRange,
  ExplicitLive,
  ForbiddenOverlap,
  MissingInstruction,
  MissingOwner,
  AmbiguousOwners,
  InvalidDescriptor,
  NoLegalWindow,
  DynamicStack,
};

/// Internal ownership retained while composing instrumentation over a staged
/// fault trampoline in a local or appended cave outside its source kernel's
/// FUNC symbol.
struct ConSanPreappliedCodeRange {
  uint64_t text_offset = 0;
  uint32_t size = 0;
  std::string kernel_name;
  std::optional<uint64_t> continuation_text_offset;
};

/// Bytes unavailable to the current placement pass. Mutation composition uses
/// this for already committed fault bytes; multi-stage instrumentation can
/// also reserve an anchor for a deferred patch in the same transaction.
/// Placement must never consume these ranges as newly available padding or a
/// cave.
struct ConSanPreappliedReservedRange {
  uint64_t text_offset = 0;
  uint32_t size = 0;
};

/// Caller-supplied exact MOI report layout. Auto allocation inventories the
/// final code, plans one heterogeneous ABI layout, and passes this same value
/// to the transformer, allocator, and reader. Explicit user buffers leave it
/// unset and retain the legacy size-derived layout contract.
struct ConSanMoiReportLayoutOverride {
  ConSanMoiEngine engine = ConSanMoiEngine::RecordReplay;
  uint32_t access_record_capacity = 0;
  uint32_t record_replay_logical_access_range_count = 0;
  uint32_t record_replay_dispatch_token_capacity = 0;
  uint32_t record_replay_access_dispatch_bank_count = 1;
  uint32_t record_replay_access_owner_bank_count = 1;
  uint32_t barrier_record_capacity = 0;
  uint32_t atomic_record_capacity = 0;
  uint32_t fence_record_capacity = 0;
  uint32_t diagnostic_capacity = 0;
  uint32_t exact_shadow_entry_capacity = 0;
  uint32_t inline_exact_dispatch_bank_count = 0;
  uint32_t inline_atomic_release_capacity = 0;
  uint32_t inline_acquired_epoch_token_capacity = 0;
  uint32_t inline_causal_snapshot_capacity = 0;
  uint32_t inline_compact_token_mapping_capacity = 0;
  uint32_t sampled_watchpoint_capacity = 0;
  uint32_t sampled_causal_window_capacity = 0;
  uint32_t sampled_sync_metadata_capacity = 0;
  uint32_t sampled_pending_acquire_capacity = 0;
  uint64_t record_replay_dispatch_tokens_offset = 0;
  uint64_t access_records_offset = 0;
  uint64_t barrier_records_offset = 0;
  uint64_t atomic_records_offset = 0;
  uint64_t fence_records_offset = 0;
  uint64_t diagnostic_records_offset = 0;
  uint64_t exact_shadow_entries_offset = 0;
  uint64_t inline_atomic_release_slots_offset = 0;
  uint64_t inline_acquired_epoch_token_slots_offset = 0;
  uint64_t inline_causal_snapshots_offset = 0;
  uint64_t inline_compact_token_mappings_offset = 0;
  uint64_t sampled_watchpoints_offset = 0;
  uint64_t sampled_causal_windows_offset = 0;
  uint64_t sampled_sync_metadata_offset = 0;
  uint64_t sampled_pending_acquires_offset = 0;
  uint64_t required_bytes = 0;
};

/// Exact persistent Record/Replay workgroup tuple. Record/Replay publishes
/// full coordinates, unlike Inline Shadow's separately packed cell key.
struct ConSanMoiPersistentWorkgroupRegisters {
  std::optional<uint16_t> x;
  std::optional<uint16_t> y;
  std::optional<uint16_t> z;

  [[nodiscard]] bool empty() const { return !x && !y && !z; }
  [[nodiscard]] bool complete() const { return x && y && z; }

  auto operator<=>(const ConSanMoiPersistentWorkgroupRegisters &) const = default;
};

/// Per-workitem private offsets holding the exact Record/Replay workgroup
/// tuple when no lifetime-safe persistent register window exists.
struct ConSanMoiPersistentWorkgroupPrivateOffsets {
  std::optional<uint32_t> x;
  std::optional<uint32_t> y;
  std::optional<uint32_t> z;

  [[nodiscard]] bool empty() const { return !x && !y && !z; }
  [[nodiscard]] bool complete() const { return x && y && z; }

  auto operator<=>(const ConSanMoiPersistentWorkgroupPrivateOffsets &) const = default;
};

/// Persistent vector state selected for one connected set of owning kernels.
/// A shared helper connects all of its owners because its instruction stream
/// must use one tuple, while unrelated kernels may use different safe tails.
struct ConSanMoiPersistentVgprAssignment {
  uint64_t descriptor_file_offset = 0;
  uint16_t owner_vgpr = 0;
  uint16_t epoch_vgpr = 0;
  std::optional<uint16_t> workgroup_key_vgpr;
  ConSanMoiPersistentWorkgroupRegisters record_replay_workgroup_vgprs;

  auto operator<=>(const ConSanMoiPersistentVgprAssignment &) const = default;
};

/// Entry-local vector temporaries for a kernel whose persistent Inline state
/// is scalar. They are proven dead before the displaced first instruction and
/// never cross guest code.
struct ConSanMoiPrologueScratchVgprAssignment {
  uint64_t descriptor_file_offset = 0;
  uint16_t scratch_vgpr = 0;

  auto operator<=>(const ConSanMoiPrologueScratchVgprAssignment &) const = default;
};

/// Probe-local scalar state selected for one connected set of owning kernels.
/// A shared helper connects its callers because its single instruction stream
/// cannot use a different transient window for each caller.
struct ConSanMoiTransientSgprAssignment {
  uint64_t descriptor_file_offset = 0;
  uint16_t exec_save_sgpr = 0;
  /// A code-object-wide dispatch pair can be unavailable to an otherwise
  /// patchable full-pressure owner. Such an owner keeps its instrumentation
  /// and records zero generation rather than clobbering guest scalar state.
  std::optional<uint16_t> dispatch_id_sgpr;
  bool spill_backed = false;
  std::optional<uint16_t> indirect_pc_sgpr;
  std::optional<uint16_t> indirect_scc_sgpr;
  std::optional<uint16_t> dispatch_key_sgpr;
  std::optional<uint16_t> call_return_sgpr;
  std::optional<uint16_t> visible_evidence_sgpr;

  auto operator<=>(const ConSanMoiTransientSgprAssignment &) const = default;
};

struct ConSanOptions {
  ConSanFlavor flavor = ConSanFlavor::None;
  ConSanMoiEngine moi_engine = ConSanMoiEngine::RecordReplay;
  ConSanMoiOwnerSource moi_owner_source = ConSanMoiOwnerSource::WorkitemId;
  ConSanFlatProvenanceMode flat_provenance_mode = ConSanFlatProvenanceMode::Likely;
  bool fail_closed = false;
  bool probe_nop = false;
  bool probe_trampoline_nop = false;
  bool probe_endpgm = false;
  bool probe_lds_endpgm = false;
  bool probe_lds_check_trap = false;
  bool probe_flat_check_trap = false;
  bool probe_flat_trap = false;
  /// Replace a statically unique immediate barrier wait that belongs to no
  /// decoded synchronization sequence with a terminal instruction. This is a
  /// bounded fail-fast diagnostic for definitely ill-formed programs, not a
  /// guess about dynamic or ambiguous barrier state.
  bool abort_unmatched_barrier_wait = false;
  bool fault_drop_barrier = false;
  /// Explicit destructive-test opt-in for dropping only one instruction from
  /// an otherwise qualified logical barrier pair.
  bool fault_allow_destructive_incomplete_barrier_drop = false;
  bool fault_move_barrier = false;
  /// Build the potentially expensive all-instruction destination inventory
  /// used only by markerless barrier movement. Direct library callers retain
  /// the complete analysis by default; production hooks may disable it when
  /// no move fault is requested.
  bool collect_barrier_move_destinations = true;
  /// Preserve the extended same-block barrier-pair inventory required by a
  /// late-bound fault after the mutation flags themselves are cleared for an
  /// immutable MOI sizing pass. Retry checks this inventory-shaping predicate
  /// before reusing the result.
  bool qualify_extended_barrier_pairs = false;
  /// Explicit non-destructive-test opt-in for moving a reconverged barrier
  /// earlier into the guard of a narrowly proven completing CFG diamond.
  bool fault_allow_completing_conditional_barrier_move = false;
  /// Explicit destructive-test opt-in for moving a uniform barrier pair into
  /// one arm of a narrowly proven structured EXEC diamond.
  bool fault_allow_destructive_divergent_barrier_move = false;
  bool fault_mutate_barrier_id_scope = false;
  bool fault_mutate_barrier_participants = false;
  bool fault_atomic_wrong_address = false;
  bool fault_atomic_weaken_order = false;
  ConSanAtomicOrderEdge fault_atomic_order_edge = ConSanAtomicOrderEdge::Any;
  bool fault_atomic_weaken_scope = false;
  bool fault_ordinary_wrong_address = false;
  bool fault_ordinary_weaken_order = false;
  bool fault_ordinary_weaken_scope = false;
  /// Caller-validated, aligned byte delta to a distinct valid atomic target.
  uint32_t fault_atomic_address_delta = 4;
  /// Caller-validated, aligned byte delta to a distinct valid ordinary-load target.
  uint32_t fault_ordinary_address_delta = 4;
  bool fault_dry_run = false;
  bool fault_require_exactly_one = false;
  /// SuperCollider-only bounded synchronization-edge perturbation request.
  /// SC2A1 inventories and plans the request but does not emit code.
  ConSanPerturbationKind sc_perturb_kind = ConSanPerturbationKind::None;
  ConSanPerturbationEdge sc_perturb_edge = ConSanPerturbationEdge::Release;
  std::string sc_perturb_identity;
  uint32_t sc_perturb_index = 0;
  uint32_t sc_perturb_max = 1;
  uint32_t sc_perturb_sleep = 1;
  uint32_t sc_perturb_required_count = 0;
  bool moi_init_owner_epoch = false;
  bool moi_track_barriers = false;
  bool moi_track_atomics = false;
  bool moi_dynamic_access_records = false;
  /// Internal rollout marker for the collision-free per-workgroup Inline
  /// shadow. The hook enables this for InlineShadow; direct patcher tests may
  /// leave it disabled while exercising the legacy global layout.
  bool moi_inline_workgroup_shadow = false;
  /// Check the immediately preceding direct-sampled slot in-kernel and
  /// increment the shared diagnostic counter on a sampled conflict.
  bool moi_sampled_check = false;
  /// Debug-only retained capture of the first address-partition pending/group
  /// masks. Never enable this for qualification or acceptance runs.
  bool moi_partition_mask_debug = false;
  /// Test-only control that exercises the spill tier even when a dead or
  /// descriptor-growth VGPR window is available.
  bool force_vgpr_spill = false;
  /// Test-only control that exercises derived-owner/private-epoch state even
  /// when a dedicated persistent VGPR pair is available.
  bool force_private_epoch = false;
  /// Test-only substring filter for selecting MOI candidates from one kernel.
  std::string test_kernel_name_filter;
  /// Internal marker set after ConSan assigns persistent owner/epoch VGPRs.
  bool automatic_moi_persistent_vgprs = false;
  /// Internal per-owner assignments used when one CDNA4 code-object-wide
  /// tuple cannot remain below every compiler-defined AccVGPR boundary.
  std::vector<ConSanMoiPersistentVgprAssignment> moi_persistent_vgpr_assignments;
  /// Internal marker set when a full CDNA ordinary-VGPR bank forces
  /// persistent MOI wave identity into uniform scalar state instead.
  bool automatic_moi_persistent_sgprs = false;
  std::vector<ConSanMoiPrologueScratchVgprAssignment> moi_prologue_scratch_vgpr_assignments;
  /// Internal marker set when inline shadow derives owner at each probe and
  /// keeps epoch in per-lane private memory.
  bool automatic_moi_private_epoch = false;
  /// Internal marker set after ConSan assigns the scalar EXEC-save window.
  bool automatic_moi_exec_save_sgprs = false;
  /// Per-owner overrides used when an automatic code-object-wide transient
  /// window is unsafe for only a subset of independent owner components.
  std::vector<ConSanMoiTransientSgprAssignment> moi_transient_sgpr_assignments;
  /// Internal gfx1250 fallback: one high code-object-wide EXEC-save window is
  /// retained, while resource planning excludes only owner groups whose guest
  /// scalar allocation reaches that window.
  bool automatic_moi_partial_exec_save_sgprs = false;
  /// Internal gfx1250 Inline fallback: the transient scalar window is saved
  /// to per-lane private memory around every access probe because no complete
  /// liveness-dead window exists at all access sites.
  bool automatic_moi_inline_sgpr_spill = false;
  /// Internal Record/Replay fallback: EXEC/VCC destinations are preserved in
  /// private memory while dense routing uses separately proven-dead scalars.
  bool automatic_moi_record_replay_sgpr_spill = false;
  /// Internal operating-point marker for RDNA-family and CDNA4 Record/Replay objects
  /// whose barrier inventory can require the relocated dense router. Such
  /// objects reserve a dispatch key and call-return pair in addition to the
  /// ordinary five-register EXEC/VCC/SCC window.
  bool moi_record_replay_dense_barrier_router = false;
  /// Internal proof marker set only when the automatic EXEC-save window lies
  /// above every guest-referenced SGPR and can retain cross-site wave state.
  bool moi_exec_save_sgprs_persistent = false;
  /// Internal marker set when a probe needs the final scalar save slot for a
  /// site-local dynamic-stack spill frame.
  bool moi_dynamic_stack_spill = false;
  /// Internal marker set after inventory proves that Inline will emit at
  /// least one ordinary access probe. Atomic-only objects need fewer
  /// transient EXEC pairs than access probes with causal-token lookup.
  bool moi_inline_access_present = false;
  /// Internal proof marker: an appended Inline access body saves and restores
  /// its complete scratch window before executing the displaced guest access.
  bool moi_spill_overlaps_guest_operands = false;
  /// Internal marker set after ConSan assigns the hw_id owner temporary SGPR.
  bool automatic_moi_owner_sgpr = false;
  /// Internal marker set after ConSan assigns the persistent hardware
  /// dispatch-ID SGPR pair used by dispatch-qualified engine metadata.
  bool automatic_moi_dispatch_id_sgprs = false;
  /// Internal metadata used only by the mutation-then-instrument composition
  /// path. It prevents appended fault code from being attributed to an
  /// unrelated tail function merely because that FUNC covers the cave bytes.
  std::vector<ConSanPreappliedCodeRange> preapplied_code_ranges;
  std::vector<ConSanPreappliedReservedRange> preapplied_reserved_ranges;
  uint32_t fault_barrier_index = 0;
  uint32_t fault_atomic_index = 0;
  uint32_t fault_ordinary_index = 0;
  /// Exact stable identity from ConSanResult::fault_sites. When set, this
  /// supersedes the legacy per-kind global index.
  std::string fault_site_identity;
  /// Exact stable real-instruction destination for markerless barrier movement.
  std::string fault_barrier_destination_identity;
  /// Exact logical barrier sequence selected for a whole-pair ID/scope dry run.
  std::string fault_barrier_sequence_identity;
  /// Optional independently resolved second complete barrier pair. When both
  /// fields are set, a drop-barrier request is one bounded all-or-nothing
  /// logical mutation over the two exact physical pairs.
  std::string fault_barrier_companion_site_identity;
  std::string fault_barrier_companion_sequence_identity;
  std::optional<int32_t> fault_barrier_target_id;
  std::optional<uint32_t> fault_barrier_target_participant_count;
  std::optional<uint64_t> fault_barrier_target_participant_mask;
  ConSanBarrierMoveDirection fault_barrier_move_direction =
      ConSanBarrierMoveDirection::LegacyMarker;
  /// Test-only s_nop immediate used as the destination of a moved barrier.
  uint16_t fault_barrier_marker_imm = 42;
  /// Internal recursion guard: faults have already been applied to this ELF.
  bool faults_preapplied = false;
  ConSanDelayMode delay_mode = ConSanDelayMode::Nop;
  uint16_t delay_var_ssrc = 106;
  std::optional<uint16_t> scratch_vgpr;
  std::optional<uint16_t> moi_exec_save_sgpr;
  /// Scalar state kept outside a spill-backed Inline transient window.
  std::optional<uint16_t> moi_inline_indirect_pc_sgpr;
  std::optional<uint16_t> moi_inline_call_return_sgpr;
  std::optional<uint16_t> moi_inline_dispatch_key_sgpr;
  std::optional<uint16_t> moi_inline_indirect_scc_sgpr;
  std::optional<uint16_t> moi_inline_visible_evidence_sgpr;
  std::optional<uint16_t> moi_record_replay_dispatch_key_sgpr;
  std::optional<uint16_t> moi_record_replay_call_return_sgpr;
  std::optional<uint16_t> moi_owner_sgpr;
  std::optional<uint16_t> moi_dispatch_id_sgpr;
  std::optional<uint16_t> moi_owner_vgpr;
  std::optional<uint16_t> moi_epoch_vgpr;
  /// Uniform persistent state used when a dynamic-stack MOI owner has no
  /// ordinary-VGPR tuple available below its compiler-defined accumulator
  /// boundary. The owner is the wave ID captured from the entry workitem ID;
  /// epoch and workgroup key are uniform for every lane.
  std::optional<uint16_t> moi_persistent_owner_sgpr;
  std::optional<uint16_t> moi_persistent_epoch_sgpr;
  std::optional<uint16_t> moi_persistent_workgroup_key_sgpr;
  /// Exact persistent workgroup coordinates used by every Record/Replay
  /// layout. All three registers are assigned together; absent launch
  /// dimensions are initialized to zero at kernel entry.
  ConSanMoiPersistentWorkgroupRegisters moi_record_replay_workgroup_sgprs;
  ConSanMoiPersistentWorkgroupRegisters moi_record_replay_workgroup_vgprs;
  ConSanMoiPersistentWorkgroupPrivateOffsets moi_record_replay_workgroup_private_offsets;
  /// Persistent packed workgroup identity used only by Inline Shadow. Launch
  /// SGPRs or RDNA TTMPs are copied into this VGPR at entry because they are
  /// not stable after arbitrary guest code.
  std::optional<uint16_t> moi_workgroup_key_vgpr;
  std::optional<uint64_t> report_buffer_address;
  std::optional<uint64_t> moi_report_buffer_address;
  uint64_t moi_report_buffer_size = 0;
  /// Effective per-workgroup LDS aperture supplied by the execution runtime.
  /// Offline callers leave this unset and use the architecture's configured
  /// default (gfx1250.json for gfx1250).
  std::optional<uint32_t> moi_max_workgroup_lds_bytes;
  std::optional<ConSanMoiReportLayoutOverride> moi_report_layout;
  /// Generation stored in direct sampled entries. Auto report buffers set
  /// this from their versioned header; explicit buffers default to zero.
  uint64_t moi_report_generation = 0;
  /// Dispatch identity stored in direct sampled causal-window descriptors.
  /// Auto report buffers use the stable code-object reader identity.
  uint64_t moi_report_dispatch_id = 0;
  uint32_t delay_nops = 0;
  ConSanPatchedImageGrowthLimit patched_image_growth_limit;
  /// Internal immutable baseline set by the public transform entry point and
  /// retained by every staged composition pass.
  std::optional<size_t> patched_image_growth_input_bytes;
  uint32_t max_patches = 1;
  // Programmatic callers set max_patches as an explicit cap by default.  The
  // runtime hook clears this for its all-supported policy so independent
  // instrumentation phases do not consume one another's nominal budget.
  bool max_patches_is_expert_limit = true;
  uint32_t moi_sample_stride = 1;
  uint32_t moi_sample_offset = 0;
  /// Runtime deterministic wave selector for direct sampled probes. The
  /// Stride is a power of two and selects a dispatch/workgroup/wave/access hash.
  uint32_t moi_runtime_sample_stride = 1;
  uint32_t moi_runtime_sample_offset = 0;
  uint32_t report_marker = 1;
};

[[nodiscard]] inline bool consan_qualifies_extended_barrier_pairs(const ConSanOptions &options) {
  return options.qualify_extended_barrier_pairs || options.fault_drop_barrier ||
         options.fault_move_barrier || options.fault_mutate_barrier_id_scope ||
         options.fault_mutate_barrier_participants || options.abort_unmatched_barrier_wait ||
         (options.moi_engine == ConSanMoiEngine::Sampled && options.moi_track_barriers);
}

using ConSanFaultMutationMember = bool ConSanOptions::*;
inline constexpr std::array<ConSanFaultMutationMember, 10> kConSanFaultMutationMembers = {
    &ConSanOptions::fault_drop_barrier,
    &ConSanOptions::fault_move_barrier,
    &ConSanOptions::fault_mutate_barrier_id_scope,
    &ConSanOptions::fault_mutate_barrier_participants,
    &ConSanOptions::fault_atomic_wrong_address,
    &ConSanOptions::fault_atomic_weaken_order,
    &ConSanOptions::fault_atomic_weaken_scope,
    &ConSanOptions::fault_ordinary_wrong_address,
    &ConSanOptions::fault_ordinary_weaken_order,
    &ConSanOptions::fault_ordinary_weaken_scope,
};

[[nodiscard]] inline bool consan_fault_mutations_enabled(const ConSanOptions &options) {
  for (ConSanFaultMutationMember member : kConSanFaultMutationMembers) {
    if (options.*member)
      return true;
  }
  return false;
}

inline void disable_consan_fault_mutations(ConSanOptions &options) {
  for (ConSanFaultMutationMember member : kConSanFaultMutationMembers)
    options.*member = false;
  options.fault_require_exactly_one = false;
}

/// Runtime report-buffer state that may change after an immutable MOI semantic
/// inventory has been built. Retry does not accept another full ConSanOptions
/// object, so parser, CFG, ownership, and admission semantics cannot drift
/// between the inventory and lowering phases.
struct ConSanMoiReportRetryConfig {
  std::optional<uint64_t> buffer_address;
  uint64_t buffer_size = 0;
  std::optional<ConSanMoiReportLayoutOverride> layout;
  uint64_t generation = 0;
  uint64_t dispatch_id = 0;
};

/// Fault-selection state that may be bound after an immutable MOI semantic
/// inventory has been built. The report-buffer retry accepts this narrow
/// projection instead of another full ConSanOptions object, so unrelated
/// parser, CFG, ownership, and admission semantics cannot drift.
struct ConSanFaultMutationRetryConfig {
  bool dry_run = false;
  bool drop_barrier = false;
  bool allow_destructive_incomplete_barrier_drop = false;
  bool move_barrier = false;
  bool allow_completing_conditional_barrier_move = false;
  bool allow_destructive_divergent_barrier_move = false;
  bool mutate_barrier_id_scope = false;
  bool mutate_barrier_participants = false;
  bool atomic_wrong_address = false;
  bool atomic_weaken_order = false;
  ConSanAtomicOrderEdge atomic_order_edge = ConSanAtomicOrderEdge::Any;
  bool atomic_weaken_scope = false;
  bool ordinary_wrong_address = false;
  bool ordinary_weaken_order = false;
  bool ordinary_weaken_scope = false;
  uint32_t atomic_address_delta = 4;
  uint32_t ordinary_address_delta = 4;
  bool require_exactly_one = false;
  uint32_t barrier_index = 0;
  uint32_t atomic_index = 0;
  uint32_t ordinary_index = 0;
  std::string site_identity;
  std::string barrier_destination_identity;
  std::string barrier_sequence_identity;
  std::string barrier_companion_site_identity;
  std::string barrier_companion_sequence_identity;
  std::optional<int32_t> barrier_target_id;
  std::optional<uint32_t> barrier_target_participant_count;
  std::optional<uint64_t> barrier_target_participant_mask;
  ConSanBarrierMoveDirection barrier_move_direction = ConSanBarrierMoveDirection::LegacyMarker;
  uint16_t barrier_marker_imm = 42;

  auto operator<=>(const ConSanFaultMutationRetryConfig &) const = default;

  [[nodiscard]] bool enabled() const {
    return drop_barrier || move_barrier || mutate_barrier_id_scope || mutate_barrier_participants ||
           atomic_wrong_address || atomic_weaken_order || atomic_weaken_scope ||
           ordinary_wrong_address || ordinary_weaken_order || ordinary_weaken_scope;
  }

  [[nodiscard]] static ConSanFaultMutationRetryConfig from_options(const ConSanOptions &options) {
    return {
        .dry_run = options.fault_dry_run,
        .drop_barrier = options.fault_drop_barrier,
        .allow_destructive_incomplete_barrier_drop =
            options.fault_allow_destructive_incomplete_barrier_drop,
        .move_barrier = options.fault_move_barrier,
        .allow_completing_conditional_barrier_move =
            options.fault_allow_completing_conditional_barrier_move,
        .allow_destructive_divergent_barrier_move =
            options.fault_allow_destructive_divergent_barrier_move,
        .mutate_barrier_id_scope = options.fault_mutate_barrier_id_scope,
        .mutate_barrier_participants = options.fault_mutate_barrier_participants,
        .atomic_wrong_address = options.fault_atomic_wrong_address,
        .atomic_weaken_order = options.fault_atomic_weaken_order,
        .atomic_order_edge = options.fault_atomic_order_edge,
        .atomic_weaken_scope = options.fault_atomic_weaken_scope,
        .ordinary_wrong_address = options.fault_ordinary_wrong_address,
        .ordinary_weaken_order = options.fault_ordinary_weaken_order,
        .ordinary_weaken_scope = options.fault_ordinary_weaken_scope,
        .atomic_address_delta = options.fault_atomic_address_delta,
        .ordinary_address_delta = options.fault_ordinary_address_delta,
        .require_exactly_one = options.fault_require_exactly_one,
        .barrier_index = options.fault_barrier_index,
        .atomic_index = options.fault_atomic_index,
        .ordinary_index = options.fault_ordinary_index,
        .site_identity = options.fault_site_identity,
        .barrier_destination_identity = options.fault_barrier_destination_identity,
        .barrier_sequence_identity = options.fault_barrier_sequence_identity,
        .barrier_companion_site_identity = options.fault_barrier_companion_site_identity,
        .barrier_companion_sequence_identity = options.fault_barrier_companion_sequence_identity,
        .barrier_target_id = options.fault_barrier_target_id,
        .barrier_target_participant_count = options.fault_barrier_target_participant_count,
        .barrier_target_participant_mask = options.fault_barrier_target_participant_mask,
        .barrier_move_direction = options.fault_barrier_move_direction,
        .barrier_marker_imm = options.fault_barrier_marker_imm,
    };
  }

  void apply_to(ConSanOptions &options) const {
    options.fault_dry_run = dry_run;
    options.fault_drop_barrier = drop_barrier;
    options.fault_allow_destructive_incomplete_barrier_drop =
        allow_destructive_incomplete_barrier_drop;
    options.fault_move_barrier = move_barrier;
    options.fault_allow_completing_conditional_barrier_move =
        allow_completing_conditional_barrier_move;
    options.fault_allow_destructive_divergent_barrier_move =
        allow_destructive_divergent_barrier_move;
    options.fault_mutate_barrier_id_scope = mutate_barrier_id_scope;
    options.fault_mutate_barrier_participants = mutate_barrier_participants;
    options.fault_atomic_wrong_address = atomic_wrong_address;
    options.fault_atomic_weaken_order = atomic_weaken_order;
    options.fault_atomic_order_edge = atomic_order_edge;
    options.fault_atomic_weaken_scope = atomic_weaken_scope;
    options.fault_ordinary_wrong_address = ordinary_wrong_address;
    options.fault_ordinary_weaken_order = ordinary_weaken_order;
    options.fault_ordinary_weaken_scope = ordinary_weaken_scope;
    options.fault_atomic_address_delta = atomic_address_delta;
    options.fault_ordinary_address_delta = ordinary_address_delta;
    options.fault_require_exactly_one = require_exactly_one;
    options.fault_barrier_index = barrier_index;
    options.fault_atomic_index = atomic_index;
    options.fault_ordinary_index = ordinary_index;
    options.fault_site_identity = site_identity;
    options.fault_barrier_destination_identity = barrier_destination_identity;
    options.fault_barrier_sequence_identity = barrier_sequence_identity;
    options.fault_barrier_companion_site_identity = barrier_companion_site_identity;
    options.fault_barrier_companion_sequence_identity = barrier_companion_sequence_identity;
    options.fault_barrier_target_id = barrier_target_id;
    options.fault_barrier_target_participant_count = barrier_target_participant_count;
    options.fault_barrier_target_participant_mask = barrier_target_participant_mask;
    options.fault_barrier_move_direction = barrier_move_direction;
    options.fault_barrier_marker_imm = barrier_marker_imm;
    options.faults_preapplied = false;
  }
};

struct ConSanMoiInventoryRetryConfig {
  ConSanMoiReportRetryConfig report;
  std::optional<ConSanFaultMutationRetryConfig> fault;
};

[[nodiscard]] inline bool moi_uses_scalar_persistent_state(const ConSanOptions &options) {
  return options.moi_persistent_owner_sgpr && options.moi_persistent_epoch_sgpr;
}

/// Process-local exact-one budget for repeated dynamic loads of one stable
/// fault-site identity. The caller serializes observe() with code-object load
/// selection. Occurrences are one-based and overflow is permanently fatal.
struct ConSanFaultLoadSelection {
  uint64_t occurrence = 0;
  bool selected = false;
  bool overflow = false;
};

class ConSanFaultLoadSelector {
public:
  explicit ConSanFaultLoadSelector(uint64_t requested_occurrence = 0)
      : requested_occurrence_(requested_occurrence) {}

  [[nodiscard]] ConSanFaultLoadSelection observe() {
    if (observed_ == std::numeric_limits<uint64_t>::max()) {
      overflow_ = true;
      return {observed_, false, true};
    }
    ++observed_;
    const bool selected = requested_occurrence_ != 0 && observed_ == requested_occurrence_;
    selected_ += selected ? 1u : 0u;
    return {observed_, selected, false};
  }

  [[nodiscard]] uint64_t requested_occurrence() const { return requested_occurrence_; }
  [[nodiscard]] uint64_t observed() const { return observed_; }
  [[nodiscard]] uint64_t selected() const { return selected_; }
  [[nodiscard]] bool overflow() const { return overflow_; }
  [[nodiscard]] bool accepted() const {
    return requested_occurrence_ != 0 && !overflow_ && observed_ >= requested_occurrence_ &&
           selected_ == 1;
  }

private:
  uint64_t requested_occurrence_ = 0;
  uint64_t observed_ = 0;
  uint64_t selected_ = 0;
  bool overflow_ = false;
};
