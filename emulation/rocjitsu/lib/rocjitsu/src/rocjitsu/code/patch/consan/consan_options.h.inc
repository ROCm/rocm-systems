#include "rocjitsu/code/patch/consan/consan_mode.h"
#include "rocjitsu/code/patch/consan/consan_report_buffer_layout.h"

namespace rocjitsu::consan {

/// Default maximum additional ELF storage one ConSan transformation may
/// materialize, including alignment padding. Dense production code objects can
/// require one out-of-line body per admitted access. The per-object growth
/// bound is independent of process-wide memory ceilings.
inline constexpr uint64_t kDefaultMaxPatchedImageGrowthBytes = uint64_t{896} * 1024 * 1024;

enum class PatchedImageGrowthLimitKind : uint8_t {
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
struct PatchedImageGrowthLimit {
  PatchedImageGrowthLimitKind kind = PatchedImageGrowthLimitKind::AbsoluteBytes;
  uint64_t absolute_bytes = kDefaultMaxPatchedImageGrowthBytes;
  uint32_t input_percent = 0;

  auto operator<=>(const PatchedImageGrowthLimit &) const = default;
};

/// Resolve a configured policy without overflowing size_t. An invalid enum
/// value is rejected instead of silently selecting a fallback.
[[nodiscard]] inline constexpr std::optional<size_t>
patched_image_growth_limit_bytes(const PatchedImageGrowthLimit &policy, size_t input_image_bytes) {
  static_assert(sizeof(size_t) >= sizeof(uint64_t),
                "ConSan patched-image accounting requires a 64-bit size_t");
  switch (policy.kind) {
  case PatchedImageGrowthLimitKind::AbsoluteBytes:
    return static_cast<size_t>(policy.absolute_bytes);
  case PatchedImageGrowthLimitKind::InputPercent: {
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

struct PatchedImageGrowthBudget {
  size_t total_limit_bytes = 0;
  size_t existing_growth_bytes = 0;
  size_t remaining_growth_bytes = 0;
  bool already_exceeded = false;
};

/// Resolve the remaining budget for a transformation stage against the
/// original code-object image. This prevents successive stages from each
/// receiving a fresh full allowance.
[[nodiscard]] inline std::optional<PatchedImageGrowthBudget>
patched_image_growth_budget(const PatchedImageGrowthLimit &policy, size_t input_image_bytes,
                            size_t current_image_bytes) {
  const auto total_limit = patched_image_growth_limit_bytes(policy, input_image_bytes);
  if (!total_limit)
    return std::nullopt;
  const size_t existing_growth =
      current_image_bytes > input_image_bytes ? current_image_bytes - input_image_bytes : 0u;
  return PatchedImageGrowthBudget{
      .total_limit_bytes = *total_limit,
      .existing_growth_bytes = existing_growth,
      .remaining_growth_bytes =
          existing_growth > *total_limit ? 0u : *total_limit - existing_growth,
      .already_exceeded = existing_growth > *total_limit,
  };
}

enum class SuperColliderDelayMode : uint8_t {
  Nop,
  Sleep,
  SleepVar,
  SleepWave,
};

enum class BarrierMoveDirection : uint8_t {
  LegacyMarker,
  Earlier,
  Later,
};

/// CFG/convergence proof attached to a cross-block whole-barrier move.
enum class BarrierMoveCfgContract : uint8_t {
  SameBlock,
  /// Both successors of an acyclic conditional guard reach the reconverged
  /// source exactly once; moving the pair into the guard remains completing.
  CompletingStructuredDiamond,
  /// The destination is one optional EXEC-narrowed arm. This intentionally
  /// creates divergent participation and requires destructive X0 containment.
  DestructiveStructuredExecDiamond,
};

enum class SuperColliderPerturbationKind : uint8_t {
  None,
  Barrier,
  Atomic,
};

enum class SuperColliderPerturbationEdge : uint8_t {
  Release,
  Acquire,
};

/// Which cache boundary an atomic weaken-order fault removes. `Any` preserves
/// the historical first-qualified-boundary behavior; explicit edges are used
/// by fence-pair qualification and fail closed when that edge is absent.
enum class AtomicOrderEdge : uint8_t {
  Any,
  Release,
  Acquire,
};

enum class OwnerSource : uint8_t {
  Automatic,
  WorkitemId,
  HwId,
};

/// Controls whether flat/VFLAT sites whose group provenance is heuristic are
/// eligible for instrumentation.
enum class FlatProvenanceMode : uint8_t {
  Likely,
  Strict,
};

enum class RegisterAllocationSource : uint8_t {
  Unsupported,
  Explicit,
  LivenessDead,
  DescriptorGrowth,
  SpillRequired,
};

enum class RegisterPlanReason : uint8_t {
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
struct PreappliedCodeRange {
  uint64_t text_offset = 0;
  uint32_t size = 0;
  std::string kernel_name;
  std::optional<uint64_t> continuation_text_offset;

  bool operator==(const PreappliedCodeRange &) const = default;
};

/// Bytes unavailable to the current placement pass. Mutation composition uses
/// this for already committed fault bytes; multi-stage instrumentation can
/// also reserve an anchor for a deferred patch in the same transaction.
/// Placement must never consume these ranges as newly available padding or a
/// cave.
struct PreappliedReservedRange {
  uint64_t text_offset = 0;
  uint32_t size = 0;

  bool operator==(const PreappliedReservedRange &) const = default;
};

/// Geometry already committed by an earlier stage of one composed transform.
///
/// A fault mutation can add executable bytes outside the source kernel's
/// symbol range and can occupy bytes that a following instrumentation stage
/// would otherwise mistake for padding or a code cave. This value carries the
/// ownership and reservation facts into construction of the mutated image's
/// immutable `ProgramInventory`. It contains no sanitizer request, register
/// choice, or mode working state.
struct PreappliedMutationLayout {
  /// Executable ranges introduced by the earlier mutation, paired with the
  /// dispatchable owner and continuation needed to recover their CFG scope.
  std::vector<PreappliedCodeRange> code_ranges;
  /// Original anchors and generated ranges that later placement passes must
  /// treat as occupied even when their bytes resemble reusable padding.
  std::vector<PreappliedReservedRange> reserved_ranges;

  bool operator==(const PreappliedMutationLayout &) const = default;
};

/// Optional exact x/y/z and cluster workgroup identity in one storage domain.
///
/// ConSan consumes x/y/z as one identity. Keeping the
/// optionality around the tuple, rather than around each coordinate, prevents
/// placement, ABI telemetry, and private-state planning from publishing a
/// partial identity. The optional cluster coordinate refines that complete
/// identity only on targets that expose cluster-local workgroup state.
template <typename T> class PersistentWorkgroupTuple {
public:
  PersistentWorkgroupTuple() = default;
  PersistentWorkgroupTuple(T x, T y, T z, std::optional<T> cluster_workgroup_id = std::nullopt)
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

  auto operator<=>(const PersistentWorkgroupTuple &) const = default;

private:
  std::optional<std::array<T, 3>> coordinates_;
  std::optional<T> cluster_workgroup_id_;
};

/// Exact persistent workgroup tuple published with ConSan evidence.
using PersistentWorkgroupRegisters = PersistentWorkgroupTuple<uint16_t>;

/// Per-workitem private offsets holding that exact workgroup tuple when no
/// lifetime-safe persistent register window exists.
using PersistentWorkgroupPrivateOffsets = PersistentWorkgroupTuple<uint32_t>;

/// Register pair carrying one owner identity and its synchronization epoch.
/// Scalar and vector persistent representations use the same semantic pair;
/// their containing allocation determines the register file and lifetime.
struct OwnerEpochRegisters {
  uint16_t owner = 0;
  uint16_t epoch = 0;

  auto operator<=>(const OwnerEpochRegisters &) const = default;
};

/// Optional accepted owner/epoch register state. The pair can be absent but
/// cannot be represented or mutated one register at a time.
class OwnerEpochRegisterState {
public:
  OwnerEpochRegisterState() = default;
  OwnerEpochRegisterState(OwnerEpochRegisters registers) : registers_(registers) {}

  [[nodiscard]] std::optional<uint16_t> owner() const {
    return registers_ ? std::optional{registers_->owner} : std::nullopt;
  }
  [[nodiscard]] std::optional<uint16_t> epoch() const {
    return registers_ ? std::optional{registers_->epoch} : std::nullopt;
  }
  [[nodiscard]] bool complete() const { return registers_.has_value(); }
  [[nodiscard]] explicit operator bool() const { return complete(); }
  [[nodiscard]] const OwnerEpochRegisters *operator->() const { return &*registers_; }
  void set_owner_epoch(uint16_t owner, uint16_t epoch) {
    registers_ = OwnerEpochRegisters{owner, epoch};
  }
  void reset_owner_epoch() { registers_.reset(); }

  auto operator<=>(const OwnerEpochRegisterState &) const = default;

private:
  std::optional<OwnerEpochRegisters> registers_;
};

/// Effective per-site vector sources. Unlike accepted persistent allocation,
/// materialization may provide only one half when the other value is derived
/// directly at the probe.
struct OwnerEpochVgprSources {
  std::optional<uint16_t> owner;
  std::optional<uint16_t> epoch;

  bool operator==(const OwnerEpochVgprSources &) const = default;
};

[[nodiscard]] inline OwnerEpochVgprSources
owner_epoch_vgpr_sources(const OwnerEpochRegisterState &state) {
  return {state.owner(), state.epoch()};
}

/// Exact ordinary-SGPR ABI that retains ConSan state across arbitrary guest
/// instructions.
///
/// Register placement builds this tuple as one indivisible lowering choice.
/// Every entry prologue and access or synchronization probe owned by the same
/// kernel consumes that choice. Emitted prologue metadata records the same
/// value so the final mechanism remains self-describing after mutable lowering
/// state is gone; it is not an independently reconstructed representation.
class PersistentSgprState : public OwnerEpochRegisterState {
public:
  /// Exact x/y/z and optional CDNA5 cluster coordinates retained by
  /// ConSan instrumentation.
  PersistentWorkgroupRegisters exact_workgroup;

  template <typename Visitor> void for_each_range(Visitor &&visit) const {
    visit(owner(), 1u);
    visit(epoch(), 1u);
    for (std::optional<uint16_t> reg : exact_workgroup.values())
      visit(reg, 1u);
  }

  auto operator<=>(const PersistentSgprState &) const = default;
};

/// Persistent vector state selected for one connected set of owning kernels.
/// A shared helper connects all of its owners because its instruction stream
/// must use one tuple, while unrelated kernels may use different safe tails.
struct PersistentVgprAssignment {
  uint64_t descriptor_file_offset = 0;
  OwnerEpochRegisters owner_epoch_vgprs;
  PersistentWorkgroupRegisters exact_workgroup_vgprs;
  /// Entry-captured hardware dispatch identity. The pair is used when no
  /// lifetime-safe persistent SGPR pair exists for an automatic banked report.

  auto operator<=>(const PersistentVgprAssignment &) const = default;
};

/// Entry-local vector temporaries for a kernel with persistent scalar state. They are proven dead
/// before the displaced first instruction and never cross guest code.
struct PrologueScratchVgprAssignment {
  uint64_t descriptor_file_offset = 0;
  uint16_t scratch_vgpr = 0;

  auto operator<=>(const PrologueScratchVgprAssignment &) const = default;
};

/// Scalar temporaries needed while a spill-backed probe establishes its
/// private frame and reloads owner-local state.
struct ScalarSpillSetup {
  ScalarSpillTemporaries temporaries;

  template <typename Visitor> void for_each_sgpr_range(Visitor &&visit) const {
    visit(temporaries.frame_base_sgpr, 2u);
    visit(temporaries.scc_save_sgpr, 1u);
  }

  auto operator<=>(const ScalarSpillSetup &) const = default;
};

/// Scalar spill state reached through branch-only relays. A dynamic-stack
/// bootstrap may additionally borrow one live ordinary pair, but that pair
/// cannot exist independently of the branch-only preservation strategy.
struct BranchOnlyScalarSpill {
  std::optional<uint16_t> dynamic_stack_borrowed_sgpr;

  auto operator<=>(const BranchOnlyScalarSpill &) const = default;
};

/// One optional SGPR allocation together with the authority that selected it.
/// Provenance cannot remain set after the register is cleared.
class SgprAllocation {
public:
  [[nodiscard]] std::optional<uint16_t> base() const { return base_; }
  [[nodiscard]] bool automatic() const { return base_ && automatic_; }

  void set(std::optional<uint16_t> base, bool automatic = false) {
    base_ = base;
    automatic_ = base && automatic;
  }
  void reset() { set(std::nullopt); }

  auto operator<=>(const SgprAllocation &) const = default;

private:
  std::optional<uint16_t> base_;
  bool automatic_ = false;
};

/// Probe-local scalar state selected for one connected set of owning kernels.
/// A shared helper connects its callers because its single instruction stream
/// cannot use a different transient window for each caller.
struct TransientSgprAssignment {
  uint64_t descriptor_file_offset = 0;
  uint16_t exec_save_sgpr = 0;
  /// Probe-local normalized resident-wave identity.
  std::optional<uint16_t> owner_sgpr;
  /// A code-object-wide dispatch pair can be unavailable to an otherwise
  /// patchable full-pressure owner. Such an owner keeps its instrumentation
  /// and records zero generation rather than clobbering guest scalar state.
  std::optional<uint16_t> dispatch_id_sgpr;
  bool spill_backed = false;
  std::optional<ScalarSpillSetup> scalar_spill_setup;
  /// The complete scalar window is preserved in private memory and control
  /// reaches the appended body through branch-only relays. This avoids any
  /// scalar value that would otherwise have to remain live from the guest
  /// anchor through probe setup.
  std::optional<BranchOnlyScalarSpill> branch_only_spill;

  auto operator<=>(const TransientSgprAssignment &) const = default;
};

/// Evidence retained when SuperCollider observes a mismatched replay.
///
/// Trap-only instrumentation reports synchronously and owns no runtime
/// allocation. Sticky-marker instrumentation also publishes a host-visible
/// word and therefore requires a concrete executable-lifetime binding before
/// lowering. Keeping this choice in the semantic request prevents a
/// hook-local presentation option from silently changing the pipeline's
/// evidence and resource contracts.
enum class SuperColliderEvidenceMode : uint8_t {
  TrapOnly,
  StickyMarker,
  Count,
};

/// Default workgroup and cell sampling stride for the runtime ConSan profile.
inline constexpr uint32_t kDefaultRuntimeSampleStride = 256u;

/// A deterministic residue selector over a power-of-two stride.
struct SampleSelector {
  uint32_t stride = 1;
  uint32_t offset = 0;
  friend bool operator==(const SampleSelector &, const SampleSelector &) = default;
};

/// User-visible sanitizer choices that determine which semantic observations
/// ConSan requests from a code object.
///
/// This value is mutable only while a configuration source constructs it and
/// is thereafter passed by const reference. It deliberately contains no
/// selected registers, patch geometry, device addresses, mutation controls, or
/// test-only switches. `mode == std::nullopt` is the single construction-time
/// sentinel; validation rejects it before the request reaches the transformer.
struct Request {
  /// Selected sanitizer family. The parser resolves load-only activation to
  /// the supported default before validating the request.
  std::optional<Mode> mode;
  /// Source of the logical owner identity recorded by ConSan probes.
  OwnerSource owner_source = OwnerSource::Automatic;
  /// Admission policy for FLAT sites whose group provenance is heuristic.
  FlatProvenanceMode flat_provenance_mode = FlatProvenanceMode::Likely;

  /// SuperCollider observes native LDS sites when this domain is enabled.
  bool probe_lds_check_trap = false;
  /// SuperCollider observes group-FLAT sites when this domain is enabled.
  bool probe_flat_check_trap = false;
  /// Capture persistent owner/epoch state at kernel entry for ConSan probes.
  bool init_owner_epoch = false;
  /// Include supported barrier semantics in the ConSan observation request.
  bool track_barriers = false;
  /// Include supported atomic and fence semantics in the ConSan observation
  /// request.
  bool track_atomics = false;
  /// Request ConSan's immediate in-kernel conflict check in addition to its
  /// retained evidence.
  bool device_conflict_check = false;

  /// Static site selection stride used by the ConSan mode.
  uint32_t sample_stride = 1;
  /// Selected residue in the static ConSan site stride.
  uint32_t sample_offset = 0;
  /// Power-of-two dynamic selector stride used by ConSan.
  uint32_t runtime_sample_stride = 1;
  /// Selected residue in the dynamic runtime stride.
  uint32_t runtime_sample_offset = 0;
  /// ConSan LDS-cell selector. When absent, use the legacy runtime selector
  /// for both workgroups and cells. Synchronization uses only the workgroup
  /// selector, irrespective of which cells are retained.
  std::optional<SampleSelector> cell_selection;
  /// Requested maximum banks per logical ConSan range: 0 (auto) or a power of two up to 1024.
  /// The report ceiling can reduce the achieved count.
  uint32_t watchpoint_banks = 0;

  [[nodiscard]] constexpr SampleSelector cell_selector() const {
    return cell_selection.value_or(SampleSelector{runtime_sample_stride, runtime_sample_offset});
  }

  /// SuperCollider delay implementation used between the original and repeat
  /// observation.
  SuperColliderDelayMode supercollider_delay_mode = SuperColliderDelayMode::Nop;
  /// Immediate delay amount interpreted by `supercollider_delay_mode`.
  uint32_t supercollider_delay_nops = 0;
  /// Scalar source used only by the variable-delay form.
  uint16_t supercollider_delay_var_ssrc = 106;
  /// Sticky value written when SuperCollider observes instability.
  uint32_t supercollider_report_marker = 1;
  /// Mismatch evidence protocol selected for SuperCollider instrumentation.
  SuperColliderEvidenceMode supercollider_evidence_mode = SuperColliderEvidenceMode::StickyMarker;

  /// Maximum automatic ConSan evidence allocation requested for one code object.
  /// Zero disables automatic allocation without supplying an address.
  uint64_t auto_report_buffer_size = 0;

  /// Production exact-name allowlist. Kernel-local sites are selected by exact
  /// entry name; a shared physical site is selected only when every known
  /// execution owner is present, so an unselected entry is never modified.
  std::vector<std::string> kernel_name_allowlist;

  bool operator==(const Request &) const = default;
};

/// Focused-test, diagnostic, and early-development controls that are outside
/// ConSan's supported semantic request.
///
/// Keeping this type separate makes a call site visibly opt into diagnostic
/// probes, forced allocation paths, or explicit registers. None of these
/// fields may influence the meaning of a normal `Request`.
struct DebugOverrides {
  /// Turn a statically unique unmatched barrier wait into a fail-fast probe.
  bool abort_unmatched_barrier_wait = false;
  /// Force the legacy VGPR spill path even when a cheaper plan exists.
  bool test_force_vgpr_spill = false;
  /// Force private epoch state even when persistent VGPR state is available.
  bool test_force_private_epoch = false;
  /// Restrict candidate discovery to matching kernel names in focused tests.
  std::string test_kernel_name_filter;

  /// Explicit scratch VGPR, bypassing normal resource selection.
  std::optional<uint16_t> scratch_vgpr;
  /// Caller-requested even SGPR pair used to preserve EXEC. This is an input
  /// constraint, not the allocator's eventual selection: an empty value asks
  /// placement to choose a suitable pair automatically.
  std::optional<uint16_t> requested_exec_save_sgpr;
  /// Caller-requested scalar owner register. This optional override is copied
  /// into a fresh lowering attempt and never mutated as placement explores
  /// automatic alternatives.
  std::optional<uint16_t> requested_owner_sgpr;
  /// Caller-requested vector owner register. An empty value leaves the
  /// persistent owner representation and register choice to placement.
  std::optional<uint16_t> requested_owner_vgpr;
  /// Caller-requested vector epoch register. An empty value permits placement
  /// to select either a register or another supported epoch representation.
  std::optional<uint16_t> requested_epoch_vgpr;

  /// Runtime assertion requiring at least one visible ConSan record.
  bool require_records = false;
  /// Runtime assertion requiring at least one emitted diagnostic.
  bool require_diagnostics = false;
  /// Runtime assertion forbidding emitted diagnostics.
  bool forbid_diagnostics = false;
  /// Runtime assertion forbidding bounded-evidence overflow.
  bool forbid_overflow = false;

  bool operator==(const DebugOverrides &) const = default;
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
  AtomicOrderEdge fault_atomic_order_edge = AtomicOrderEdge::Any;
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
  BarrierMoveDirection fault_barrier_move_direction = BarrierMoveDirection::LegacyMarker;

  SuperColliderPerturbationKind supercollider_perturb_kind = SuperColliderPerturbationKind::None;
  SuperColliderPerturbationEdge supercollider_perturb_edge = SuperColliderPerturbationEdge::Release;
  std::string supercollider_perturb_identity;
  uint32_t supercollider_perturb_index = 0;
  uint32_t supercollider_perturb_max = 1;
  uint32_t supercollider_perturb_sleep = 1;
  uint32_t supercollider_perturb_required_count = 0;

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
    return has_fault_mutation() ||
           supercollider_perturb_kind != SuperColliderPerturbationKind::None;
  }

  bool operator==(const MutationRequest &) const = default;
};

/// Provenance of the runtime facts supplied to one ConSan transformation.
///
/// This identifies the query mechanism, not the GPU architecture and not the
/// sanitizer mode. Keeping provenance explicit lets fixtures supply
/// simulator facts without pretending that they came from HSA, while the
/// construction-time `Unknown` state prevents an unqueried default object from
/// being mistaken for a backend that lacks every optional facility.
enum class RuntimeBackend : uint8_t {
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

/// Facts discovered from the selected execution backend and agent.
///
/// These are runtime facts, not implications of a `gfx*` name. Optional
/// quantities distinguish an unavailable query from a genuine zero-sized
/// facility. Consumers state their requirements explicitly and reject missing
/// properties before allocating or lowering code that depends on them.
struct RuntimeCapabilities {
  RuntimeBackend backend = RuntimeBackend::Unknown;
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
enum class RuntimeResourceScope : uint8_t {
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
inline constexpr auto kRuntimeResourceScopes = [] {
  using E = RuntimeResourceScope;
  return make_enum_vocabulary("invalid-runtime-resource-scope", enum_entry(E::Unbound, "unbound"),
                              enum_entry(E::CodeObject, "code-object"),
                              enum_entry(E::Executable, "executable"),
                              enum_entry(E::Dispatch, "dispatch"));
}();

/// Return the stable diagnostic spelling of a runtime-resource lifetime.
/// Invalid values never acquire a plausible lifetime name.
[[nodiscard]] constexpr std::string_view runtime_resource_scope_name(RuntimeResourceScope scope) {
  return kRuntimeResourceScopes.name(scope);
}

static_assert(kRuntimeResourceScopes.size() == static_cast<size_t>(RuntimeResourceScope::Count));

/// Concrete runtime allocations and identities bound after address-free
/// planning.
///
/// This value may change when an inventory is retried with an automatic report
/// allocation, but each individual value is immutable once handed to the
/// transformer. It contains no sanitizer policy, mutation request, or test
/// override. Lifetime ownership remains with the runtime coordinator; this
/// value supplies the allocation geometry and identities needed by lowering.
struct BoundRuntimeResources {
  RuntimeResourceScope scope = RuntimeResourceScope::Unbound;
  std::optional<uint64_t> supercollider_report_buffer_address;
  std::optional<uint64_t> report_buffer_address;
  uint64_t report_buffer_size = 0;
  std::optional<ReportBufferLayout> report_layout;
  uint64_t report_generation = 0;
  uint64_t report_dispatch_id = 0;

  [[nodiscard]] bool bound() const {
    return supercollider_report_buffer_address.has_value() || report_buffer_address.has_value();
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
  PatchedImageGrowthLimit patched_image_growth_limit;
  /// Maximum number of admitted legacy patches when an expert cap is active.
  uint32_t max_patches = 1;
  /// False means the legacy patch count is an implementation-sized all-sites
  /// allowance rather than a user-requested semantic omission.
  bool max_patches_is_expert_limit = true;

  [[nodiscard]] bool operator==(const TransformPolicy &other) const {
    return patched_image_growth_limit == other.patched_image_growth_limit &&
           max_patches == other.max_patches &&
           max_patches_is_expert_limit == other.max_patches_is_expert_limit;
  }
};

/// Complete caller-owned input to the internal native ConSan lowerer.
///
/// The inherited request, debug-override, mutation, runtime-fact, binding, and
/// transform-policy subobjects are the sole representations of input state.
/// Placement and emission must not add selected registers, storage, routes, or
/// retry markers to this value; both an in-progress ConSan attempt and an accepted
/// result represent those decisions with `OperatingPoint`.
struct Options : Request,
                 DebugOverrides,
                 MutationRequest,
                 RuntimeCapabilities,
                 BoundRuntimeResources,
                 TransformPolicy {
  /// Default construction is retained for focused mechanism tests. Production
  /// orchestration uses the typed constructor below.
  Options() { mode = Mode::None; }

  /// Construct one immutable input aggregate from the complete production
  /// contracts. The six base subobjects preserve those inputs verbatim.
  Options(const Request &request, const TransformPolicy &transform_policy,
          const DebugOverrides &debug, const MutationRequest &mutation,
          const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources)
      : Request(request), DebugOverrides(debug), MutationRequest(mutation),
        RuntimeCapabilities(capabilities), BoundRuntimeResources(resources),
        TransformPolicy(transform_policy) {}
};

[[nodiscard]] inline bool container_selected(const Request &request, const DebugOverrides &debug,
                                             std::string_view container_name) {
  if (!debug.test_kernel_name_filter.empty() &&
      container_name.find(debug.test_kernel_name_filter) == std::string_view::npos)
    return false;
  if (request.kernel_name_allowlist.empty())
    return true;
  return std::ranges::any_of(request.kernel_name_allowlist, [&](std::string_view allowed) {
    return kernel_symbol_names_match(allowed, container_name);
  });
}

/// Authoritative resource and persistent-state selection for one ConSan attempt.
///
/// None of this value is caller configuration. Placement refines a complete
/// attempt while searching; retry checkpoints and the accepted transform copy
/// the same type. Code-object-wide choices live alongside owner-component
/// assignments so equality, rollback, emission, and validation cannot observe
/// independently assembled subsets of the accepted allocation. Site-local
/// lowering may copy the operating point and bind one owner assignment without
/// changing the accepted code-object-wide value.
enum class ScalarSpillLayout : uint8_t {
  None,
  Compact,
};

struct OperatingPoint {
  /// Placement's effective owner/epoch-prologue decision. Initial point
  /// construction resolves the caller request into this value once; mode
  /// planning and placement may subsequently refine it without requiring
  /// downstream components to consult a parallel configuration authority.
  bool initialize_owner_epoch = false;
  /// First SGPR in the code-object-wide EXEC/VCC/SCC save window accepted by
  /// placement. This is seeded from the corresponding caller override when
  /// present, but thereafter records the selected lowering state rather than
  /// caller input. Independent owner components may replace it with an entry
  /// in `OperatingPoint::owner_transient_sgprs`.
  std::optional<uint16_t> exec_save_sgpr;
  /// Code-object-wide scalar owner register accepted by placement. It carries
  /// the probe-local normalized resident-wave identity when the selected
  /// mode represents that identity in transient scalar state.
  SgprAllocation owner_sgpr;
  /// Code-object-wide persistent per-lane owner/epoch pair accepted by
  /// placement. Some operating points instead use per-owner assignments,
  /// scalar state, or private state, in which case this value is empty.
  OwnerEpochRegisterState owner_epoch_vgprs;
  /// Internal marker set after ConSan assigns persistent owner/epoch VGPRs.
  bool automatic_persistent_vgprs = false;
  /// Internal marker set when ConSan derives owner at each probe and
  /// keeps epoch in per-lane private memory.
  bool automatic_private_epoch = false;
  /// Internal CDNA5 fallback: one high code-object-wide EXEC-save window is
  /// retained, while resource planning excludes only owner groups whose guest
  /// scalar allocation reaches that window.
  bool automatic_partial_exec_save_sgprs = false;
  /// Spill-backed scalar layout selected after no complete liveness-dead
  /// transient window exists.
  ScalarSpillLayout automatic_scalar_spill_layout = ScalarSpillLayout::None;
  /// Internal proof marker set only when the automatic EXEC-save window lies
  /// above every guest-referenced SGPR and can retain cross-site wave state.
  bool exec_save_sgprs_persistent = false;
  /// Internal marker set when a probe needs the base scalar save slot for a
  /// site-local dynamic-stack spill frame.
  bool dynamic_stack_spill = false;
  /// Scalar setup state used when a spill-backed transient window cannot use
  /// the fixed register layout: frame pointer pair and SCC-save temporary.
  std::optional<ScalarSpillSetup> scalar_spill_setup;
  /// Component whose saved scalar window is reached through branch-only
  /// relays instead of live-through SGPR state, including its optional
  /// dynamic-stack bootstrap pair.
  std::optional<BranchOnlyScalarSpill> branch_only_spill;
  SgprAllocation dispatch_sgpr;
  /// Uniform persistent scalar state selected when a ConSan owner cannot retain
  /// the equivalent values in ordinary VGPRs. The owner is captured from the
  /// entry workitem ID; epoch, compact workgroup key, and exact workgroup
  /// coordinates are uniform for every lane. Keeping the complete ABI in one
  /// value prevents planning, emission, and prologue telemetry from assembling
  /// different partial tuples.
  PersistentSgprState persistent_sgprs;
  PersistentWorkgroupRegisters exact_workgroup_vgprs;

  /// Per-owner persistent vector tuples selected when one code-object-wide
  /// tuple cannot satisfy every owner's ordinary-VGPR boundary.
  std::vector<PersistentVgprAssignment> owner_persistent_vgprs;
  /// Per-owner-component transient scalar overrides selected by placement.
  std::vector<TransientSgprAssignment> owner_transient_sgprs;

  [[nodiscard]] bool has_scalar_spill() const {
    return automatic_scalar_spill_layout != ScalarSpillLayout::None;
  }

  [[nodiscard]] bool has_compact_scalar_spill() const {
    return automatic_scalar_spill_layout == ScalarSpillLayout::Compact;
  }
  void set_owner_epoch_vgprs(uint16_t owner, uint16_t epoch) {
    owner_epoch_vgprs.set_owner_epoch(owner, epoch);
  }
  void reset_owner_epoch_vgprs() { owner_epoch_vgprs.reset_owner_epoch(); }

  /// Visit every fixed-width auxiliary scalar reservation represented by this
  /// operating point. EXEC, owner, dispatch, and persistent state have
  /// separate lifetime rules and are intentionally not part of this view.
  template <typename Visitor> void for_each_auxiliary_sgpr_range(Visitor &&visit) const {
    if (scalar_spill_setup)
      scalar_spill_setup->for_each_sgpr_range(visit);
    visit(branch_only_spill ? branch_only_spill->dynamic_stack_borrowed_sgpr : std::nullopt, 2u);
  }

  bool operator==(const OperatingPoint &) const = default;
};

static_assert(sizeof(OperatingPoint) <= 128);

/// Construct the initial unresolved point from explicit caller overrides.
/// Placement owns every subsequent mutation; keeping this conversion narrow
/// prevents immutable request state and solver output from becoming one
/// inherited attempt bus.
[[nodiscard]] inline OperatingPoint initial_operating_point(const Request &request,
                                                            const DebugOverrides &debug) {
  OperatingPoint point;
  point.initialize_owner_epoch = request.init_owner_epoch;
  point.exec_save_sgpr = debug.requested_exec_save_sgpr;
  point.owner_sgpr.set(debug.requested_owner_sgpr);
  if (debug.requested_owner_vgpr && debug.requested_epoch_vgpr) {
    point.set_owner_epoch_vgprs(*debug.requested_owner_vgpr, *debug.requested_epoch_vgpr);
  }
  return point;
}

/// Process-local exact-one budget for repeated dynamic loads of one stable
/// fault-site identity. The caller serializes observe() with code-object load
/// selection. Occurrences are one-based and overflow is permanently fatal.
struct FaultLoadSelection {
  uint64_t occurrence = 0;
  bool selected = false;
  bool overflow = false;
};

class FaultLoadSelector {
public:
  explicit FaultLoadSelector(uint64_t requested_occurrence = 0)
      : requested_occurrence_(requested_occurrence) {}

  [[nodiscard]] FaultLoadSelection observe() {
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
