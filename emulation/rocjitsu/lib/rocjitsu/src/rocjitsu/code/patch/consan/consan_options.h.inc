#include "rocjitsu/code/patch/consan/consan_moi_report_buffer_layout.h"

namespace rocjitsu {

/// Default maximum additional ELF storage one ConSan transformation may
/// materialize, including alignment padding. Dense production code objects can
/// require one out-of-line body per admitted access. Qualified generated
/// operator libraries require up to about 385 MiB for complete Inline or
/// Record/Replay coverage, so retain a bounded 400 MiB default while leaving
/// process-wide ceilings independent.
inline constexpr uint64_t kConSanDefaultMaxPatchedImageGrowthBytes = uint64_t{400} * 1024 * 1024;

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

enum class ConSanMoiOwnerSource : uint8_t {
  Automatic,
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

  bool operator==(const ConSanPreappliedCodeRange &) const = default;
};

/// Bytes unavailable to the current placement pass. Mutation composition uses
/// this for already committed fault bytes; multi-stage instrumentation can
/// also reserve an anchor for a deferred patch in the same transaction.
/// Placement must never consume these ranges as newly available padding or a
/// cave.
struct ConSanPreappliedReservedRange {
  uint64_t text_offset = 0;
  uint32_t size = 0;

  bool operator==(const ConSanPreappliedReservedRange &) const = default;
};

/// Geometry already committed by an earlier stage of one composed transform.
///
/// A fault mutation can add executable bytes outside the source kernel's
/// symbol range and can occupy bytes that a following instrumentation stage
/// would otherwise mistake for padding or a code cave. This value carries the
/// ownership and reservation facts into construction of the mutated image's
/// immutable `ProgramInventory`. It contains no sanitizer request, register
/// choice, or engine working state.
struct ConSanPreappliedMutationLayout {
  /// Executable ranges introduced by the earlier mutation, paired with the
  /// dispatchable owner and continuation needed to recover their CFG scope.
  std::vector<ConSanPreappliedCodeRange> code_ranges;
  /// Original anchors and generated ranges that later placement passes must
  /// treat as occupied even when their bytes resemble reusable padding.
  std::vector<ConSanPreappliedReservedRange> reserved_ranges;

  bool operator==(const ConSanPreappliedMutationLayout &) const = default;
};

/// Optional exact x/y/z and cluster workgroup identity in one storage domain.
///
/// Record/Replay and Sampled always consume x/y/z as one identity. Keeping the
/// optionality around the tuple, rather than around each coordinate, prevents
/// placement, ABI telemetry, and private-state planning from publishing a
/// partial identity. The optional cluster coordinate refines that complete
/// identity only on targets that expose cluster-local workgroup state.
template <typename T> class ConSanMoiPersistentWorkgroupTuple {
public:
  ConSanMoiPersistentWorkgroupTuple() = default;
  ConSanMoiPersistentWorkgroupTuple(T x, T y, T z,
                                    std::optional<T> cluster_workgroup_id = std::nullopt)
      : coordinates_{{x, y, z}}, cluster_workgroup_id_(cluster_workgroup_id) {}

  [[nodiscard]] bool empty() const { return !coordinates_; }
  [[nodiscard]] bool complete() const { return coordinates_.has_value(); }
  [[nodiscard]] std::optional<T> x() const {
    return coordinates_ ? std::optional{(*coordinates_)[0]} : std::nullopt;
  }
  [[nodiscard]] std::optional<T> y() const {
    return coordinates_ ? std::optional{(*coordinates_)[1]} : std::nullopt;
  }
  [[nodiscard]] std::optional<T> z() const {
    return coordinates_ ? std::optional{(*coordinates_)[2]} : std::nullopt;
  }
  [[nodiscard]] std::optional<T> cluster_workgroup_id() const { return cluster_workgroup_id_; }
  [[nodiscard]] std::array<std::optional<T>, 4> values() const {
    return {x(), y(), z(), cluster_workgroup_id()};
  }

  void reset() {
    coordinates_.reset();
    cluster_workgroup_id_.reset();
  }

  auto operator<=>(const ConSanMoiPersistentWorkgroupTuple &) const = default;

private:
  std::optional<std::array<T, 3>> coordinates_;
  std::optional<T> cluster_workgroup_id_;
};

/// Exact persistent Record/Replay workgroup tuple. Record/Replay publishes
/// full coordinates, unlike Inline Shadow's separately packed cell key.
using ConSanMoiPersistentWorkgroupRegisters = ConSanMoiPersistentWorkgroupTuple<uint16_t>;

/// Per-workitem private offsets holding the exact Record/Replay workgroup
/// tuple when no lifetime-safe persistent register window exists.
using ConSanMoiPersistentWorkgroupPrivateOffsets = ConSanMoiPersistentWorkgroupTuple<uint32_t>;

/// Register pair carrying one owner identity and its synchronization epoch.
/// Scalar and vector persistent representations use the same semantic pair;
/// their containing allocation determines the register file and lifetime.
struct ConSanMoiOwnerEpochRegisters {
  uint16_t owner = 0;
  uint16_t epoch = 0;

  auto operator<=>(const ConSanMoiOwnerEpochRegisters &) const = default;
};

/// Optional accepted owner/epoch register state. The pair can be absent but
/// cannot be represented or mutated one register at a time.
class ConSanMoiOwnerEpochRegisterState {
public:
  ConSanMoiOwnerEpochRegisterState() = default;
  ConSanMoiOwnerEpochRegisterState(ConSanMoiOwnerEpochRegisters registers)
      : registers_(registers) {}

  [[nodiscard]] std::optional<uint16_t> owner() const {
    return registers_ ? std::optional{registers_->owner} : std::nullopt;
  }
  [[nodiscard]] std::optional<uint16_t> epoch() const {
    return registers_ ? std::optional{registers_->epoch} : std::nullopt;
  }
  [[nodiscard]] bool complete() const { return registers_.has_value(); }
  [[nodiscard]] explicit operator bool() const { return complete(); }
  [[nodiscard]] const ConSanMoiOwnerEpochRegisters *operator->() const { return &*registers_; }
  void set_owner_epoch(uint16_t owner, uint16_t epoch) {
    registers_ = ConSanMoiOwnerEpochRegisters{owner, epoch};
  }
  void reset_owner_epoch() { registers_.reset(); }

  auto operator<=>(const ConSanMoiOwnerEpochRegisterState &) const = default;

private:
  std::optional<ConSanMoiOwnerEpochRegisters> registers_;
};

/// Effective per-site vector sources. Unlike accepted persistent allocation,
/// materialization may provide only one half when the other value is derived
/// directly at the probe.
struct ConSanMoiOwnerEpochVgprSources {
  std::optional<uint16_t> owner;
  std::optional<uint16_t> epoch;

  bool operator==(const ConSanMoiOwnerEpochVgprSources &) const = default;
};

[[nodiscard]] inline ConSanMoiOwnerEpochVgprSources
moi_owner_epoch_vgpr_sources(const ConSanMoiOwnerEpochRegisterState &state) {
  return {state.owner(), state.epoch()};
}

/// Exact ordinary-SGPR ABI that retains MOI state across arbitrary guest
/// instructions.
///
/// Register placement builds this tuple as one indivisible lowering choice.
/// Every entry prologue and access or synchronization probe owned by the same
/// kernel consumes that choice. Emitted prologue metadata records the same
/// value so the final mechanism remains self-describing after mutable lowering
/// state is gone; it is not an independently reconstructed representation.
class ConSanMoiPersistentSgprState : public ConSanMoiOwnerEpochRegisterState {
public:
  /// Compact workgroup identity used by InlineShadow when that engine does
  /// not require the exact coordinate tuple.
  std::optional<uint16_t> workgroup_key;
  /// Exact x/y/z and optional gfx1250 cluster coordinates retained by
  /// Record/Replay and Sampled instrumentation.
  ConSanMoiPersistentWorkgroupRegisters record_replay_workgroup;

  template <typename Visitor> void for_each_range(Visitor &&visit) const {
    visit(owner(), 1u);
    visit(epoch(), 1u);
    visit(workgroup_key, 1u);
    for (std::optional<uint16_t> reg : record_replay_workgroup.values())
      visit(reg, 1u);
  }

  auto operator<=>(const ConSanMoiPersistentSgprState &) const = default;
};

/// Persistent vector state selected for one connected set of owning kernels.
/// A shared helper connects all of its owners because its instruction stream
/// must use one tuple, while unrelated kernels may use different safe tails.
struct ConSanMoiPersistentVgprAssignment {
  uint64_t descriptor_file_offset = 0;
  ConSanMoiOwnerEpochRegisters owner_epoch_vgprs;
  std::optional<uint16_t> workgroup_key_vgpr;
  ConSanMoiPersistentWorkgroupRegisters record_replay_workgroup_vgprs;
  /// Entry-captured hardware dispatch identity. The pair is used when no
  /// lifetime-safe persistent SGPR pair exists for an automatic banked report.
  std::optional<uint16_t> dispatch_id_vgpr;

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

/// Scalar PC pair and saved-SCC register used by one preserving indirect jump.
/// The jump sequence cannot be emitted safely with only one of these pieces,
/// so placement publishes them as one allocation.
struct ConSanMoiIndirectJumpSgprs {
  uint16_t pc_sgpr = 0;
  uint16_t scc_save_sgpr = 0;

  auto operator<=>(const ConSanMoiIndirectJumpSgprs &) const = default;
};

/// Scalar key and return pair retained across one generated router call.
/// These registers are selected and consumed together; representing them as
/// one allocation prevents accepted operating points from carrying a partial
/// mode-specific routing ABI.
struct ConSanMoiRouterCallSgprs {
  uint16_t dispatch_key_sgpr = 0;
  uint16_t call_return_sgpr = 0;

  auto operator<=>(const ConSanMoiRouterCallSgprs &) const = default;
};

/// Inline spill state reached through branch-only relays. A dynamic-stack
/// bootstrap may additionally borrow one live ordinary pair, but that pair
/// cannot exist independently of the branch-only preservation strategy.
struct ConSanMoiBranchOnlyScalarSpill {
  std::optional<uint16_t> dynamic_stack_borrowed_sgpr;

  auto operator<=>(const ConSanMoiBranchOnlyScalarSpill &) const = default;
};

/// One optional SGPR allocation together with the authority that selected it.
/// Provenance cannot remain set after the register is cleared.
class ConSanMoiSgprAllocation {
public:
  [[nodiscard]] std::optional<uint16_t> base() const { return base_; }
  [[nodiscard]] bool automatic() const { return base_ && automatic_; }

  void set(std::optional<uint16_t> base, bool automatic = false) {
    base_ = base;
    automatic_ = base && automatic;
  }
  void reset() { set(std::nullopt); }

  auto operator<=>(const ConSanMoiSgprAllocation &) const = default;

private:
  std::optional<uint16_t> base_;
  bool automatic_ = false;
};

/// Persistent hardware dispatch identity selected for an MOI operating point.
///
/// The scalar and vector register forms are alternatives. Scalar provenance is
/// part of the selected alternative, so an automatic-placement marker cannot
/// drift away from the pair that it describes. A private-entry fallback is an
/// independent component policy: it may coexist with an object-wide scalar
/// pair when only some owner components cannot retain that pair.
class ConSanMoiDispatchIdentityAllocation {
public:
  [[nodiscard]] std::optional<uint16_t> sgpr() const { return sgpr_.base(); }
  [[nodiscard]] std::optional<uint16_t> vgpr() const { return vgpr_; }
  [[nodiscard]] bool automatic_sgpr() const { return sgpr_.automatic(); }
  [[nodiscard]] bool private_fallback() const { return private_fallback_; }

  void set_sgpr(std::optional<uint16_t> base, bool automatic = false) {
    sgpr_.set(base, automatic);
    if (base)
      vgpr_.reset();
  }
  void set_vgpr(std::optional<uint16_t> base) {
    vgpr_ = base;
    if (base)
      sgpr_.reset();
  }
  void reset_sgpr() { sgpr_.reset(); }
  void reset_vgpr() { vgpr_.reset(); }
  void set_private_fallback(bool enabled) { private_fallback_ = enabled; }

  auto operator<=>(const ConSanMoiDispatchIdentityAllocation &) const = default;

private:
  ConSanMoiSgprAllocation sgpr_;
  std::optional<uint16_t> vgpr_;
  bool private_fallback_ = false;
};

/// Probe-local scalar state selected for one connected set of owning kernels.
/// A shared helper connects its callers because its single instruction stream
/// cannot use a different transient window for each caller.
struct ConSanMoiTransientSgprAssignment {
  uint64_t descriptor_file_offset = 0;
  uint16_t exec_save_sgpr = 0;
  /// Probe-local normalized resident-wave identity. Automatic Inline
  /// assignments keep this in the final slot of their saved scalar window.
  std::optional<uint16_t> owner_sgpr;
  /// A code-object-wide dispatch pair can be unavailable to an otherwise
  /// patchable full-pressure owner. Such an owner keeps its instrumentation
  /// and records zero generation rather than clobbering guest scalar state.
  std::optional<uint16_t> dispatch_id_sgpr;
  bool spill_backed = false;
  std::optional<ConSanMoiIndirectJumpSgprs> router_jump;
  std::optional<ConSanMoiRouterCallSgprs> router_call;
  std::optional<uint16_t> visible_evidence_sgpr;
  /// The complete scalar window is preserved in private memory and control
  /// reaches the appended body through branch-only relays. This avoids any
  /// scalar value that would otherwise have to remain live from the guest
  /// anchor through the router.
  std::optional<ConSanMoiBranchOnlyScalarSpill> branch_only_spill;

  auto operator<=>(const ConSanMoiTransientSgprAssignment &) const = default;
};

/// Evidence retained when SuperCollider observes a mismatched replay.
///
/// Trap-only instrumentation reports synchronously and owns no runtime
/// allocation. Sticky-marker instrumentation also publishes a host-visible
/// word and therefore requires a concrete executable-lifetime binding before
/// lowering. Keeping this choice in the semantic request prevents a
/// hook-local presentation option from silently changing the pipeline's
/// evidence and resource contracts.
enum class ConSanSuperColliderEvidenceMode : uint8_t {
  TrapOnly,
  StickyMarker,
  Count,
};

/// User-visible sanitizer choices that determine which semantic observations
/// ConSan requests from a code object.
///
/// This value is mutable only while a configuration source constructs it and
/// is thereafter passed by const reference. It deliberately contains no
/// selected registers, patch geometry, device addresses, mutation controls, or
/// test-only switches. `flavor == std::nullopt` is the single construction-time
/// sentinel; validation rejects it before the request reaches the transformer.
struct ConSanRequest {
  /// Selected sanitizer family. The parser resolves load-only activation to
  /// the supported default before validating the request.
  std::optional<ConSanFlavor> flavor;
  /// MOI implementation selected when `flavor` is `ConSanFlavor::Moi`.
  ConSanMoiEngine moi_engine = ConSanMoiEngine::RecordReplay;
  /// Source of the logical owner identity recorded by MOI probes.
  ConSanMoiOwnerSource moi_owner_source = ConSanMoiOwnerSource::Automatic;
  /// Admission policy for FLAT sites whose group provenance is heuristic.
  ConSanFlatProvenanceMode flat_provenance_mode = ConSanFlatProvenanceMode::Likely;

  /// SuperCollider observes native LDS sites when this domain is enabled.
  bool probe_lds_check_trap = false;
  /// SuperCollider observes group-FLAT sites when this domain is enabled.
  bool probe_flat_check_trap = false;
  /// Capture persistent owner/epoch state at kernel entry for MOI engines that
  /// require it.
  bool moi_init_owner_epoch = false;
  /// Include supported barrier semantics in the MOI observation request.
  bool moi_track_barriers = false;
  /// Include supported atomic and fence semantics in the MOI observation
  /// request.
  bool moi_track_atomics = false;
  /// Request Record/Replay's dynamic access-record representation.
  bool moi_dynamic_access_records = false;
  /// Request Sampled's immediate in-kernel conflict check in addition to its
  /// retained evidence.
  bool moi_sampled_check = false;

  /// Static site selection stride used by the Sampled engine.
  uint32_t moi_sample_stride = 1;
  /// Selected residue in the static Sampled site stride.
  uint32_t moi_sample_offset = 0;
  /// Power-of-two dynamic selector stride used by Record/Replay and Sampled.
  uint32_t moi_runtime_sample_stride = 1;
  /// Selected residue in the dynamic runtime stride.
  uint32_t moi_runtime_sample_offset = 0;

  /// SuperCollider delay implementation used between the original and repeat
  /// observation.
  ConSanDelayMode delay_mode = ConSanDelayMode::Nop;
  /// Immediate delay amount interpreted by `delay_mode`.
  uint32_t delay_nops = 0;
  /// Scalar source used only by the variable-delay form.
  uint16_t delay_var_ssrc = 106;
  /// Sticky value written when SuperCollider observes instability.
  uint32_t report_marker = 1;
  /// Mismatch evidence protocol selected for SuperCollider instrumentation.
  ConSanSuperColliderEvidenceMode supercollider_evidence_mode =
      ConSanSuperColliderEvidenceMode::StickyMarker;

  /// Maximum automatic MOI evidence allocation requested for one code object.
  /// Zero disables automatic allocation without supplying an address.
  uint64_t moi_auto_report_buffer_size = 0;

  /// Production exact-name allowlist. Kernel-local sites are selected by exact
  /// entry name; a shared physical site is selected only when every known
  /// execution owner is present, so an unselected entry is never modified.
  std::vector<std::string> kernel_name_allowlist;

  bool operator==(const ConSanRequest &) const = default;
};

/// Focused-test, diagnostic, and early-development controls that are outside
/// ConSan's supported semantic request.
///
/// Keeping this type separate makes a call site visibly opt into prototype
/// probes, forced allocation paths, or explicit registers. None of these
/// fields may influence the meaning of a normal `ConSanRequest`.
struct ConSanDebugOverrides {
  /// Replace a selected instruction with a NOP for focused patch testing.
  bool probe_nop = false;
  /// Route a selected instruction through a NOP trampoline.
  bool probe_trampoline_nop = false;
  /// Replace a selected instruction with program termination.
  bool probe_endpgm = false;
  /// Apply the termination probe only to native LDS instructions.
  bool probe_lds_endpgm = false;
  /// Replace a selected group-FLAT site with a direct trap probe.
  bool probe_flat_trap = false;
  /// Turn a statically unique unmatched barrier wait into a fail-fast probe.
  bool abort_unmatched_barrier_wait = false;
  /// Retain address-partition masks for diagnostic inspection.
  bool moi_partition_mask_debug = false;
  /// Force the legacy VGPR spill path even when a cheaper plan exists.
  bool test_force_vgpr_spill = false;
  /// Force private epoch state even when persistent VGPR state is available.
  bool test_force_private_epoch = false;
  /// Seed an odd Inline exact-shadow publication for a teardown fixture.
  bool test_seed_inline_exact_odd = false;
  /// Restrict candidate discovery to matching kernel names in focused tests.
  std::string test_kernel_name_filter;

  /// Explicit scratch VGPR, bypassing normal resource selection.
  std::optional<uint16_t> scratch_vgpr;
  /// Caller-requested even SGPR pair used to preserve EXEC. This is an input
  /// constraint, not the allocator's eventual selection: an empty value asks
  /// placement to choose a suitable pair automatically.
  std::optional<uint16_t> requested_moi_exec_save_sgpr;
  /// Caller-requested scalar owner register. This optional override is copied
  /// into a fresh lowering attempt and never mutated as placement explores
  /// automatic alternatives.
  std::optional<uint16_t> requested_moi_owner_sgpr;
  /// Caller-requested vector owner register. An empty value leaves the
  /// persistent owner representation and register choice to placement.
  std::optional<uint16_t> requested_moi_owner_vgpr;
  /// Caller-requested vector epoch register. An empty value permits placement
  /// to select either a register or another supported epoch representation.
  std::optional<uint16_t> requested_moi_epoch_vgpr;

  /// Runtime assertion requiring at least one visible MOI record.
  bool moi_require_records = false;
  /// Runtime assertion requiring at least one emitted diagnostic.
  bool moi_require_diagnostics = false;
  /// Runtime assertion forbidding emitted diagnostics.
  bool moi_forbid_diagnostics = false;
  /// Runtime assertion requiring a Record/Replay conflict.
  bool moi_require_replay_conflict = false;
  /// Runtime assertion forbidding bounded-evidence overflow.
  bool moi_forbid_overflow = false;

  bool operator==(const ConSanDebugOverrides &) const = default;
};

/// Explicit request to corrupt or perturb a program for validation.
///
/// Mutation is transformational and is never implied by ordinary sanitizer
/// activation. Exact site identities and selection indices live here so they
/// cannot accidentally alter semantic inventory or report planning for a
/// normal request.
struct MutationRequest {
  bool fault_drop_barrier = false;
  bool fault_allow_destructive_incomplete_barrier_drop = false;
  bool fault_move_barrier = false;
  bool fault_allow_completing_conditional_barrier_move = false;
  bool fault_allow_destructive_divergent_barrier_move = false;
  bool fault_mutate_barrier_id_scope = false;
  bool fault_mutate_barrier_participants = false;
  bool fault_atomic_wrong_address = false;
  bool fault_atomic_weaken_order = false;
  ConSanAtomicOrderEdge fault_atomic_order_edge = ConSanAtomicOrderEdge::Any;
  bool fault_atomic_weaken_scope = false;
  bool fault_lds_wrong_address = false;
  bool fault_ordinary_wrong_address = false;
  bool fault_ordinary_weaken_order = false;
  bool fault_ordinary_weaken_scope = false;

  uint32_t fault_atomic_address_delta = 4;
  std::optional<uint16_t> fault_lds_address_vgpr;
  uint32_t fault_ordinary_address_delta = 4;
  bool fault_dry_run = false;
  bool fault_require_exactly_one = false;
  uint32_t fault_reservation_timeout_ms = 30000;
  std::optional<uint32_t> fault_load_occurrence;

  uint32_t fault_barrier_index = 0;
  uint32_t fault_atomic_index = 0;
  uint32_t fault_lds_index = 0;
  uint32_t fault_ordinary_index = 0;
  std::string fault_site_identity;
  std::string fault_barrier_destination_identity;
  std::string fault_barrier_sequence_identity;
  std::string fault_barrier_companion_site_identity;
  std::string fault_barrier_companion_sequence_identity;
  std::optional<int32_t> fault_barrier_target_id;
  std::optional<uint32_t> fault_barrier_target_participant_count;
  std::optional<uint64_t> fault_barrier_target_participant_mask;
  ConSanBarrierMoveDirection fault_barrier_move_direction =
      ConSanBarrierMoveDirection::LegacyMarker;

  ConSanPerturbationKind sc_perturb_kind = ConSanPerturbationKind::None;
  ConSanPerturbationEdge sc_perturb_edge = ConSanPerturbationEdge::Release;
  std::string sc_perturb_identity;
  uint32_t sc_perturb_index = 0;
  uint32_t sc_perturb_max = 1;
  uint32_t sc_perturb_sleep = 1;
  uint32_t sc_perturb_required_count = 0;

  /// Return whether a fault transform (as opposed to timing perturbation) was
  /// requested.
  [[nodiscard]] bool has_fault_mutation() const { return fault_mutation_count() != 0u; }

  /// Return the number of independently requested fault transformations.
  /// Selection controls and perturbations are not fault transformations.
  [[nodiscard]] size_t fault_mutation_count() const {
    return static_cast<size_t>(fault_drop_barrier) + static_cast<size_t>(fault_move_barrier) +
           static_cast<size_t>(fault_mutate_barrier_id_scope) +
           static_cast<size_t>(fault_mutate_barrier_participants) +
           static_cast<size_t>(fault_atomic_wrong_address) +
           static_cast<size_t>(fault_atomic_weaken_order) +
           static_cast<size_t>(fault_atomic_weaken_scope) +
           static_cast<size_t>(fault_lds_wrong_address) +
           static_cast<size_t>(fault_ordinary_wrong_address) +
           static_cast<size_t>(fault_ordinary_weaken_order) +
           static_cast<size_t>(fault_ordinary_weaken_scope);
  }

  /// Return whether either a fault transform or SuperCollider perturbation was
  /// requested.
  [[nodiscard]] bool has_mutation() const {
    return has_fault_mutation() || sc_perturb_kind != ConSanPerturbationKind::None;
  }

  bool operator==(const MutationRequest &) const = default;
};

/// Provenance of the runtime facts supplied to one ConSan transformation.
///
/// This identifies the query mechanism, not the GPU architecture and not the
/// sanitizer engine. Keeping provenance explicit lets fixtures supply
/// simulator facts without pretending that they came from HSA, while the
/// construction-time `Unknown` state prevents an unqueried default object from
/// being mistaken for a backend that lacks every optional facility.
enum class ConSanRuntimeBackend : uint8_t {
  /// No backend has supplied facts yet; validated production inputs reject
  /// this construction-time sentinel.
  Unknown,
  /// Capabilities were queried from a physical HSA agent.
  PhysicalHsa,
  /// Capabilities were supplied by a RocJitsu simulation configuration.
  RocJitsuSimulator,
  /// One past the last declared value. This is an iteration bound and an
  /// invalid runtime backend, not a backend that may supply capabilities.
  Count,
};

/// Every declared runtime-backend value, including the construction sentinel
/// but excluding the one-past-the-end iteration bound. Keeping this inventory
/// beside the enum makes exhaustive tests and diagnostic rendering fail
/// visibly when the backend vocabulary grows.
inline constexpr std::array<ConSanRuntimeBackend, 3> kConSanRuntimeBackends = {
    ConSanRuntimeBackend::Unknown,
    ConSanRuntimeBackend::PhysicalHsa,
    ConSanRuntimeBackend::RocJitsuSimulator,
};

/// Return the stable diagnostic spelling of a runtime-backend provenance.
/// Invalid values, including `Count`, deliberately share an invalid spelling
/// so they cannot masquerade as a queried backend in logs or test fixtures.
[[nodiscard]] constexpr std::string_view consan_runtime_backend_name(ConSanRuntimeBackend backend) {
  switch (backend) {
  case ConSanRuntimeBackend::Unknown:
    return "unknown";
  case ConSanRuntimeBackend::PhysicalHsa:
    return "physical-hsa";
  case ConSanRuntimeBackend::RocJitsuSimulator:
    return "rocjitsu-simulator";
  case ConSanRuntimeBackend::Count:
    break;
  }
  return "invalid-runtime-backend";
}

static_assert(kConSanRuntimeBackends.size() == static_cast<size_t>(ConSanRuntimeBackend::Count));

/// Facts discovered from the selected execution backend and agent.
///
/// These are runtime facts, not implications of a `gfx*` name. Optional
/// quantities distinguish an unavailable query from a genuine zero-sized
/// facility. Consumers state their requirements explicitly and reject missing
/// properties before allocating or lowering code that depends on them.
struct RuntimeCapabilities {
  ConSanRuntimeBackend backend = ConSanRuntimeBackend::Unknown;
  /// A global allocation can be made visible to both host and selected agent.
  bool host_device_visible_memory = false;
  /// Host observations are coherent with committed device publications.
  bool host_device_coherent_memory = false;
  /// The report memory supports the atomic publication protocol required by
  /// the selected evidence schema.
  bool device_atomic_publication = false;
  /// Maximum eligible report allocation exposed by the backend query.
  std::optional<uint64_t> max_report_allocation_bytes;
  /// Maximum effective group-segment aperture for a workgroup dispatch.
  std::optional<uint32_t> max_workgroup_lds_bytes;
  /// The backend can bind a replacement code object to an executable.
  bool executable_binding = false;
  /// The backend can apply planned private/group segment additions at dispatch.
  bool dispatch_segment_binding = false;

  bool operator==(const RuntimeCapabilities &) const = default;
};

/// Lifetime domain attached to addresses in `BoundRuntimeResources`.
///
/// The scope tells the runtime coordinator when an address must be rebound or
/// retired; it does not transfer ownership to the transformer. `Unbound` is a
/// construction-time state in which all addresses must be absent. Every other
/// value states the narrowest runtime identity that keeps the binding valid.
enum class ConSanRuntimeResourceScope : uint8_t {
  /// The resource has not been bound; addresses must be absent.
  Unbound,
  /// The resource belongs to one code-object reader load.
  CodeObject,
  /// The resource remains valid for an executable lifetime.
  Executable,
  /// The resource is rebound for each dispatch generation.
  Dispatch,
  /// One past the last declared value. This is an iteration bound and never a
  /// valid resource lifetime.
  Count,
};

/// Every valid resource-scope value, including the explicitly unbound state
/// but excluding the one-past-the-end iteration bound.
inline constexpr std::array<ConSanRuntimeResourceScope, 4> kConSanRuntimeResourceScopes = {
    ConSanRuntimeResourceScope::Unbound,
    ConSanRuntimeResourceScope::CodeObject,
    ConSanRuntimeResourceScope::Executable,
    ConSanRuntimeResourceScope::Dispatch,
};

/// Return the stable diagnostic spelling of a runtime-resource lifetime.
/// Invalid values never acquire a plausible lifetime name.
[[nodiscard]] constexpr std::string_view
consan_runtime_resource_scope_name(ConSanRuntimeResourceScope scope) {
  switch (scope) {
  case ConSanRuntimeResourceScope::Unbound:
    return "unbound";
  case ConSanRuntimeResourceScope::CodeObject:
    return "code-object";
  case ConSanRuntimeResourceScope::Executable:
    return "executable";
  case ConSanRuntimeResourceScope::Dispatch:
    return "dispatch";
  case ConSanRuntimeResourceScope::Count:
    break;
  }
  return "invalid-runtime-resource-scope";
}

static_assert(kConSanRuntimeResourceScopes.size() ==
              static_cast<size_t>(ConSanRuntimeResourceScope::Count));

/// Concrete runtime allocations and identities bound after address-free
/// planning.
///
/// This value may change when an inventory is retried with an automatic report
/// allocation, but each individual value is immutable once handed to the
/// transformer. It contains no sanitizer policy, mutation request, or test
/// override. Lifetime ownership remains with the runtime coordinator; this
/// projection carries only the identity needed by legacy lowering.
struct BoundRuntimeResources {
  ConSanRuntimeResourceScope scope = ConSanRuntimeResourceScope::Unbound;
  std::optional<uint64_t> report_buffer_address;
  std::optional<uint64_t> moi_report_buffer_address;
  uint64_t moi_report_buffer_size = 0;
  std::optional<ConSanMoiReportBufferLayout> moi_report_layout;
  uint64_t moi_report_generation = 0;
  uint64_t moi_report_dispatch_id = 0;

  [[nodiscard]] bool bound() const {
    return report_buffer_address.has_value() || moi_report_buffer_address.has_value();
  }

  bool operator==(const BoundRuntimeResources &) const = default;
};

/// Limits on the amount of transformation and placement work one code object
/// may consume.
///
/// These limits constrain implementation cost but do not alter which semantic
/// sites are relevant. Resolved image sizes, selected caves, registers, and
/// patch bytes are outputs and therefore cannot appear here.
struct TransformPolicy {
  /// Total growth permitted relative to the pristine input image.
  ConSanPatchedImageGrowthLimit patched_image_growth_limit;
  /// Maximum number of admitted legacy patches when an expert cap is active.
  uint32_t max_patches = 1;
  /// False means the legacy patch count is an implementation-sized all-sites
  /// allowance rather than a user-requested semantic omission.
  bool max_patches_is_expert_limit = true;
  /// Work allowance for SOPP relay search.
  PlanningWorkLimit sopp_relay_planning_work_limit = kDefaultSoppRelayPlanningWorkLimit;
  /// Work allowance for direct reservoir search.
  PlanningWorkLimit direct_reservoir_planning_work_limit = kDefaultDirectReservoirPlanningWorkLimit;
  /// Work allowance for LDS relay layout refinement.
  PlanningWorkLimit lds_relay_layout_planning_work_limit = kDefaultLdsRelayLayoutPlanningWorkLimit;
  /// Work allowance for LDS convergence planning.
  PlanningWorkLimit lds_convergence_planning_work_limit = kDefaultLdsConvergencePlanningWorkLimit;

  [[nodiscard]] bool operator==(const TransformPolicy &other) const {
    const auto same_limit = [](PlanningWorkLimit left, PlanningWorkLimit right) {
      return left.base == right.base && left.per_input == right.per_input;
    };
    return patched_image_growth_limit == other.patched_image_growth_limit &&
           max_patches == other.max_patches &&
           max_patches_is_expert_limit == other.max_patches_is_expert_limit &&
           same_limit(sopp_relay_planning_work_limit, other.sopp_relay_planning_work_limit) &&
           same_limit(direct_reservoir_planning_work_limit,
                      other.direct_reservoir_planning_work_limit) &&
           same_limit(lds_relay_layout_planning_work_limit,
                      other.lds_relay_layout_planning_work_limit) &&
           same_limit(lds_convergence_planning_work_limit,
                      other.lds_convergence_planning_work_limit);
  }
};

/// Complete caller-owned input to the internal native ConSan lowerer.
///
/// The inherited request, debug-override, mutation, runtime-fact, binding, and
/// transform-policy subobjects are the sole representations of input state.
/// Placement and emission must not add selected registers, storage, routes, or
/// retry markers to this value; both an in-progress MOI attempt and an accepted
/// result represent those decisions with `ConSanMoiOperatingPoint`.
struct ConSanOptions : ConSanRequest,
                       ConSanDebugOverrides,
                       MutationRequest,
                       RuntimeCapabilities,
                       BoundRuntimeResources,
                       TransformPolicy {
  /// Default construction is retained for focused mechanism tests. Production
  /// orchestration uses the typed constructor below.
  ConSanOptions() { flavor = ConSanFlavor::None; }

  /// Construct one immutable input aggregate from the complete production
  /// contracts. The six base subobjects preserve those inputs verbatim.
  ConSanOptions(const ConSanRequest &request, const TransformPolicy &transform_policy,
                const ConSanDebugOverrides &debug, const MutationRequest &mutation,
                const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources)
      : ConSanRequest(request), ConSanDebugOverrides(debug), MutationRequest(mutation),
        RuntimeCapabilities(capabilities), BoundRuntimeResources(resources),
        TransformPolicy(transform_policy) {}
};

[[nodiscard]] inline std::string_view consan_normalize_kernel_name(std::string_view name) {
  if (name.ends_with(".kd"))
    name.remove_suffix(3);
  return name;
}

[[nodiscard]] inline bool consan_container_selected(const ConSanOptions &options,
                                                    std::string_view container_name) {
  if (!options.test_kernel_name_filter.empty() &&
      container_name.find(options.test_kernel_name_filter) == std::string_view::npos)
    return false;
  if (options.kernel_name_allowlist.empty())
    return true;
  const std::string_view normalized = consan_normalize_kernel_name(container_name);
  return std::ranges::any_of(options.kernel_name_allowlist, [&](std::string_view allowed) {
    return consan_normalize_kernel_name(allowed) == normalized;
  });
}

/// Authoritative resource and persistent-state selection for one MOI attempt.
///
/// None of this value is caller configuration. Placement refines a complete
/// attempt while searching; retry checkpoints and the accepted transform copy
/// the same type. Code-object-wide choices live alongside owner-component
/// assignments so equality, rollback, emission, and validation cannot observe
/// independently assembled subsets of the accepted allocation. Site-local
/// lowering may copy the operating point and bind one owner assignment without
/// changing the accepted code-object-wide value.
enum class ConSanMoiScalarSpillLayout : uint8_t {
  None,
  Inline,
  Compact,
};

struct ConSanMoiOperatingPoint {
  /// Placement's effective owner/epoch-prologue decision. Initial point
  /// construction resolves the caller request into this value once; mode
  /// planning and placement may subsequently refine it without requiring
  /// downstream components to consult a parallel configuration authority.
  bool moi_initialize_owner_epoch = false;
  /// First SGPR in the code-object-wide EXEC/VCC/SCC save window accepted by
  /// placement. This is seeded from the corresponding caller override when
  /// present, but thereafter records the selected lowering state rather than
  /// caller input. Independent owner components may replace it with an entry
  /// in `ConSanMoiOperatingPoint::owner_transient_sgprs`.
  std::optional<uint16_t> moi_exec_save_sgpr;
  /// Code-object-wide scalar owner register accepted by placement. It carries
  /// the probe-local normalized resident-wave identity when the selected
  /// engine represents that identity in transient scalar state.
  ConSanMoiSgprAllocation moi_owner_sgpr;
  /// Code-object-wide persistent per-lane owner/epoch pair accepted by
  /// placement. Some operating points instead use per-owner assignments,
  /// scalar state, or private state, in which case this value is empty.
  ConSanMoiOwnerEpochRegisterState moi_owner_epoch_vgprs;
  /// Internal marker set after ConSan assigns persistent owner/epoch VGPRs.
  bool automatic_moi_persistent_vgprs = false;
  /// Internal marker set when inline shadow derives owner at each probe and
  /// keeps epoch in per-lane private memory.
  bool automatic_moi_private_epoch = false;
  /// Internal gfx1250 fallback: one high code-object-wide EXEC-save window is
  /// retained, while resource planning excludes only owner groups whose guest
  /// scalar allocation reaches that window.
  bool automatic_moi_partial_exec_save_sgprs = false;
  /// Spill-backed scalar layout selected after no complete liveness-dead
  /// transient window exists. Inline and compact Record/Replay-style layouts
  /// have different ABI fields and are mutually exclusive.
  ConSanMoiScalarSpillLayout automatic_moi_scalar_spill_layout = ConSanMoiScalarSpillLayout::None;
  /// Internal proof marker set only when the automatic EXEC-save window lies
  /// above every guest-referenced SGPR and can retain cross-site wave state.
  bool moi_exec_save_sgprs_persistent = false;
  /// Internal marker set when a probe needs the base scalar save slot for a
  /// site-local dynamic-stack spill frame.
  bool moi_dynamic_stack_spill = false;
  /// Explicit scalar-router ABI selected when a spill-backed transient window
  /// cannot use the mode's fixed register layout. The indirect jump, dispatch
  /// key, and call-return pair are one shared routing mechanism used by Inline,
  /// Record/Replay, and Sampled rather than separate mode-owned state.
  std::optional<ConSanMoiIndirectJumpSgprs> moi_router_jump;
  std::optional<ConSanMoiRouterCallSgprs> moi_router_call;
  /// Inline-only visible evidence temporary within that shared router ABI.
  std::optional<uint16_t> moi_inline_visible_evidence_sgpr;
  /// Inline component whose saved scalar window is reached through branch-only
  /// relays instead of live-through SGPR state, including its optional
  /// dynamic-stack bootstrap pair.
  std::optional<ConSanMoiBranchOnlyScalarSpill> moi_branch_only_spill;
  ConSanMoiDispatchIdentityAllocation moi_dispatch_identity;
  /// Uniform persistent scalar state selected when an MOI owner cannot retain
  /// the equivalent values in ordinary VGPRs. The owner is captured from the
  /// entry workitem ID; epoch, compact workgroup key, and exact Record/Replay
  /// coordinates are uniform for every lane. Keeping the complete ABI in one
  /// value prevents planning, emission, and prologue telemetry from assembling
  /// different partial tuples.
  ConSanMoiPersistentSgprState moi_persistent_sgprs;
  ConSanMoiPersistentWorkgroupRegisters moi_record_replay_workgroup_vgprs;
  /// Persistent packed workgroup identity used only by Inline Shadow. Launch
  /// SGPRs or RDNA TTMPs are copied into this VGPR at entry because they are
  /// not stable after arbitrary guest code.
  std::optional<uint16_t> moi_workgroup_key_vgpr;

  /// Per-owner persistent vector tuples selected when one code-object-wide
  /// tuple cannot satisfy every owner's ordinary-VGPR boundary.
  std::vector<ConSanMoiPersistentVgprAssignment> owner_persistent_vgprs;
  /// Per-owner-component transient scalar overrides selected by placement.
  std::vector<ConSanMoiTransientSgprAssignment> owner_transient_sgprs;

  [[nodiscard]] bool has_moi_scalar_spill() const {
    return automatic_moi_scalar_spill_layout != ConSanMoiScalarSpillLayout::None;
  }
  [[nodiscard]] bool has_inline_moi_scalar_spill() const {
    return automatic_moi_scalar_spill_layout == ConSanMoiScalarSpillLayout::Inline;
  }
  [[nodiscard]] bool has_compact_moi_scalar_spill() const {
    return automatic_moi_scalar_spill_layout == ConSanMoiScalarSpillLayout::Compact;
  }
  void set_moi_owner_epoch_vgprs(uint16_t owner, uint16_t epoch) {
    moi_owner_epoch_vgprs.set_owner_epoch(owner, epoch);
  }
  void reset_moi_owner_epoch_vgprs() { moi_owner_epoch_vgprs.reset_owner_epoch(); }

  /// Visit every fixed-width auxiliary scalar reservation represented by this
  /// operating point. EXEC, owner, dispatch, and persistent state have
  /// separate lifetime rules and are intentionally not part of this view.
  template <typename Visitor> void for_each_auxiliary_sgpr_range(Visitor &&visit) const {
    visit(moi_router_jump ? std::optional{moi_router_jump->pc_sgpr} : std::nullopt, 2u);
    visit(moi_router_jump ? std::optional{moi_router_jump->scc_save_sgpr} : std::nullopt, 1u);
    visit(moi_router_call ? std::optional{moi_router_call->dispatch_key_sgpr} : std::nullopt, 1u);
    visit(moi_router_call ? std::optional{moi_router_call->call_return_sgpr} : std::nullopt, 2u);
    visit(moi_inline_visible_evidence_sgpr, 1u);
    visit(moi_branch_only_spill ? moi_branch_only_spill->dynamic_stack_borrowed_sgpr : std::nullopt,
          2u);
  }

  bool operator==(const ConSanMoiOperatingPoint &) const = default;
};

/// Construct the initial unresolved point from explicit caller overrides.
/// Placement owns every subsequent mutation; keeping this conversion narrow
/// prevents immutable request state and solver output from becoming one
/// inherited attempt bus.
[[nodiscard]] inline ConSanMoiOperatingPoint
initial_consan_moi_operating_point(const ConSanOptions &options) {
  ConSanMoiOperatingPoint point;
  point.moi_initialize_owner_epoch = options.moi_init_owner_epoch;
  point.moi_exec_save_sgpr = options.requested_moi_exec_save_sgpr;
  point.moi_owner_sgpr.set(options.requested_moi_owner_sgpr);
  if (options.requested_moi_owner_vgpr && options.requested_moi_epoch_vgpr) {
    point.set_moi_owner_epoch_vgprs(*options.requested_moi_owner_vgpr,
                                    *options.requested_moi_epoch_vgpr);
  }
  return point;
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
