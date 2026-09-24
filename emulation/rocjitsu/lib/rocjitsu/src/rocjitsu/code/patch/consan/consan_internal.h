// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_internal.h
/// @brief Private, directly testable invariants shared by ConSan lowering paths.
///
/// Keep pure release-active checks here when multiple lowering `.inc` files
/// depend on them and direct unit coverage is more precise than a test hook in
/// the full patching pipeline.

#pragma once

#include "rocjitsu/code/patch/consan/consan_dispatch_preload.h"
#include "rocjitsu/code/patch/consan/consan_instrumentation.h"
#include "rocjitsu/code/patch/consan/targets/consan_target_ops.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/spill_manager.h"

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu::consan::detail {

/// Adds ConSan's common first-failure diagnostic policy to an instruction
/// sequence. Helpers can append directly to the shared word stream; invoking
/// this requirement records only the first failing operation and makes the
/// whole sequence fail atomically.
class EmissionRequirement {
public:
  EmissionRequirement(InstructionSequence &sequence, std::vector<std::string> &errors)
      : sequence_(sequence), errors_(errors) {}

  void operator()(bool success, std::string_view message = {}) {
    if (sequence_ && !success && !message.empty())
      errors_.emplace_back(message);
    sequence_.require(success);
  }

  template <typename... Values> void append(std::string_view message, const Values &...values) {
    operator()(sequence_.emit_all(values...), message);
  }

private:
  InstructionSequence &sequence_;
  std::vector<std::string> &errors_;
};

/// Stage-aware first-failure diagnostics for one transactional ConSan program.
///
/// Large publication transactions advance through named semantic stages while
/// appending to one InstructionSequence. This owner records the stage at which
/// the sequence first failed, preserves an optional leaf diagnostic, and emits
/// the common transaction diagnostic unless finish() succeeds. Keeping that
/// lifecycle here prevents mode emitters from independently implementing
/// subtly different first-failure rules.
class StagedEmission {
public:
  StagedEmission(InstructionSequence &sequence, std::vector<std::string> &errors,
                 std::string_view diagnostic_prefix)
      : sequence_(sequence), errors_(errors), diagnostic_prefix_(diagnostic_prefix) {}

  ~StagedEmission() {
    if (!succeeded_)
      errors_.emplace_back(std::string(diagnostic_prefix_) + " failed during " +
                           std::string(failed_stage_.empty() ? stage_ : failed_stage_));
  }

  StagedEmission(const StagedEmission &) = delete;
  StagedEmission &operator=(const StagedEmission &) = delete;

  void require(bool success, std::string_view message = {}) {
    if (sequence_ && !success) {
      failed_stage_ = stage_;
      if (!message.empty())
        errors_.emplace_back(message);
    }
    sequence_.require(success);
  }

  void operator()(bool success, std::string_view message = {}) { require(success, message); }

  void stage(std::string_view stage) {
    if (!sequence_ && failed_stage_.empty())
      failed_stage_ = stage_;
    stage_ = stage;
  }

  [[nodiscard]] bool finish(rj_code_arch_t arch) {
    succeeded_ = sequence_.finish(arch);
    return succeeded_;
  }

private:
  InstructionSequence &sequence_;
  std::vector<std::string> &errors_;
  std::string_view diagnostic_prefix_;
  std::string_view stage_ = "preflight";
  std::string_view failed_stage_;
  bool succeeded_ = false;
};

} // namespace rocjitsu::consan::detail

namespace rocjitsu::consan {

/// Immutable input to one ConSan resource-solving run.
///
/// The problem binds the exact image and target to effective semantic request,
/// runtime binding, published inventory/policy, and normalized access sites.
/// It deliberately contains no selected register, site plan, diagnostic, CFG
/// cache, or mutable reservation. Those belong to an attempted operating point,
/// its result, or the solver's private workspace.
class ResourceProblem {
public:
  ResourceProblem(std::span<const uint8_t> image, rj_code_arch_t arch, const Request &request,
                  const BoundRuntimeResources &resources, const ProgramInventory &inventory,
                  const ObservationPlan &observation_plan, std::span<const Candidate> candidates)
      : image_(image), arch_(arch), request_(&request), resources_(&resources),
        inventory_(&inventory), observation_plan_(&observation_plan), candidates_(candidates) {}

  [[nodiscard]] std::span<const uint8_t> image() const { return image_; }
  [[nodiscard]] rj_code_arch_t arch() const { return arch_; }
  [[nodiscard]] const Request &request() const { return *request_; }
  [[nodiscard]] const BoundRuntimeResources &resources() const { return *resources_; }
  [[nodiscard]] const ProgramInventory &inventory() const { return *inventory_; }
  [[nodiscard]] const ObservationPlan &observation_plan() const { return *observation_plan_; }
  [[nodiscard]] std::span<const Candidate> candidates() const { return candidates_; }

private:
  std::span<const uint8_t> image_;
  rj_code_arch_t arch_ = ROCJITSU_CODE_ARCH_INVALID;
  const Request *request_ = nullptr;
  const BoundRuntimeResources *resources_ = nullptr;
  const ProgramInventory *inventory_ = nullptr;
  const ObservationPlan *observation_plan_ = nullptr;
  std::span<const Candidate> candidates_;
};

/// Immutable facts that determine the size of ConSan's transient scalar-save ABI.
///
/// The construction authority projects only request, runtime-binding, and
/// attempted-operating-point facts that affect the ABI. Placement and emission
/// consume this value instead of independently consulting the complete ConSan
/// option bus, so changing an unrelated option cannot change scalar sizing.
struct ExecSaveRequirement {
  bool has_report_buffer = false;
  bool track_atomics = false;
  bool scalar_spill = false;
  bool dynamic_stack_spill = false;

  bool operator==(const ExecSaveRequirement &) const = default;
};

static_assert(sizeof(ExecSaveRequirement) <= 4);

[[nodiscard]] ExecSaveRequirement
resolve_exec_save_requirement(const Request &request, const BoundRuntimeResources &resources,
                              const OperatingPoint &operating_point);

[[nodiscard]] uint16_t exec_save_sgpr_count(const ExecSaveRequirement &requirement,
                                            rj_code_arch_t arch);

/// One scalar, vector, or entry-captured private-state source for a
/// workgroup-coordinate component.
///
/// The none-set state represents an absent dimension. More than one set source
/// is malformed. Keeping this release-active invariant beside the
/// representation prevents individual emitters from interpreting ambiguous
/// sources differently.
struct WorkgroupSource {
  std::optional<uint16_t> scalar_src;
  std::optional<uint16_t> vector_src;
  std::optional<uint32_t> private_offset;
  uint8_t right_shift = 0;
  /// Preserve this many low bits after shifting. Zero means no mask.
  uint8_t low_bit_count = 0;

  [[nodiscard]] uint8_t source_count() const {
    return static_cast<uint8_t>(scalar_src.has_value()) +
           static_cast<uint8_t>(vector_src.has_value()) +
           static_cast<uint8_t>(private_offset.has_value());
  }
  [[nodiscard]] bool has_value() const { return source_count() == 1u; }
  [[nodiscard]] bool is_well_formed() const {
    return source_count() <= 1u && right_shift < 32u && low_bit_count <= 32u - right_shift &&
           (source_count() != 0u || (right_shift == 0u && low_bit_count == 0u));
  }
  [[nodiscard]] std::optional<uint16_t> operand() const;

  [[nodiscard]] static WorkgroupSource scalar(uint16_t source, uint8_t right_shift = 0u,
                                              uint8_t low_bit_count = 0u) {
    WorkgroupSource result;
    result.scalar_src = source;
    result.right_shift = right_shift;
    result.low_bit_count = low_bit_count;
    return result;
  }
  [[nodiscard]] static WorkgroupSource vector(uint16_t source, uint8_t right_shift = 0u,
                                              uint8_t low_bit_count = 0u) {
    WorkgroupSource result;
    result.vector_src = source;
    result.right_shift = right_shift;
    result.low_bit_count = low_bit_count;
    return result;
  }
  [[nodiscard]] static WorkgroupSource private_state(uint32_t offset) {
    WorkgroupSource result;
    result.private_offset = offset;
    return result;
  }

  auto operator<=>(const WorkgroupSource &) const = default;
};

/// Complete descriptor-resolved launch-coordinate view for one kernel entry.
///
/// The four coordinate members use the same typed scalar, vector, private, or
/// absent representation so prologue and event emitters cannot disagree about
/// operand interpretation. The CDNA payload members describe a temporary
/// expanded system-SGPR suffix: an entry prologue may capture coordinates from
/// that suffix and then restore the packed suffix expected by guest code.
struct WorkgroupSources {
  /// Source of the mandatory x workgroup coordinate.
  WorkgroupSource x;

  /// Source of y, or an absent source for a one-dimensional launch.
  WorkgroupSource y;

  /// Source of z, or an absent source for a one- or two-dimensional launch.
  WorkgroupSource z;

  /// Source of the CDNA5 cluster-local workgroup coordinate, or absent when
  /// the launch ABI does not expose one.
  WorkgroupSource cluster_workgroup_id;

  /// First SGPR of an expanded CDNA x/y/z system payload, when planning had to
  /// enable the complete tuple in the descriptor.
  std::optional<uint16_t> cdna_full_payload_base;

  /// First SGPR of the guest-visible packed CDNA payload restored after entry
  /// instrumentation.
  std::optional<uint16_t> cdna_guest_payload_base;

  /// Descriptor bit mask naming the packed CDNA dimensions guest code expects.
  uint8_t cdna_guest_payload_mask = 0;

  /// Return whether no coordinate names multiple simultaneous representations.
  [[nodiscard]] bool is_well_formed() const {
    return x.is_well_formed() && y.is_well_formed() && z.is_well_formed() &&
           cluster_workgroup_id.is_well_formed();
  }

  auto operator<=>(const WorkgroupSources &) const = default;
};

namespace detail {

/// Return whether two half-open register ranges overlap.
[[nodiscard]] bool range_overlaps(uint16_t lhs_base, uint16_t lhs_count, uint16_t rhs_base,
                                  uint16_t rhs_count);

/// Reject an optional persistent VGPR that aliases a transient scratch window.
[[nodiscard]] bool reject_optional_scratch_range_overlap(std::optional<uint16_t> value,
                                                         uint16_t scratch_vgpr,
                                                         uint16_t scratch_count,
                                                         std::string_view value_name,
                                                         std::vector<std::string> &errors);

/// Complete semantic input to one private-state entry-initialization body.
///
/// Placement constructs this plan after it has resolved the private layout,
/// temporary register window, guest-state preservation, launch ABI sources,
/// and optional dispatch capture for one kernel. The
/// native emitter receives this plan plus only the body and return addresses,
/// which may differ between paired kernarg-preload entries. It therefore
/// cannot independently reinterpret full request/operating-point state, patch metadata, or the
/// kernel descriptor while emitting those bodies.
struct PrivateEpochPrologueEmissionPlan {
  /// First VGPR in the entry-local temporary window.
  uint16_t scratch_vgpr = 0;

  /// Complete persistent identity and ephemeral-allocation layout selected by
  /// private-state planning.
  PrivateStateLayout private_state_layout;

  /// Owned save/restore program for the borrowed temporary VGPR window.
  VgprSpillSequence spill;

  /// Owned optional save/restore program for entry ABI SGPRs borrowed by the
  /// prologue.
  std::optional<SgprSpillSequence> entry_scalar_spill;

  /// Resolved launch-coordinate sources needed by workgroup identity.
  std::optional<WorkgroupSources> workgroup_sources;

  /// Descriptor-derived dispatch preload transformation fixed by planning.
  std::optional<DispatchIdPreloadPlan> dispatch_plan;

  /// Unique register representation that receives the dispatch identity.
  DispatchIdCapture dispatch_capture;

  /// Number of consecutive temporary VGPRs required by the selected semantic
  /// operations, independent of their target instruction encodings.
  [[nodiscard]] uint16_t required_scratch_vgpr_count() const {
    if (private_state_layout.dispatch_id_offset)
      return 2u;
    return 1u;
  }

  /// Return whether the address-free structural contract is safe to lower.
  /// Target-specific instruction availability remains the emitter's concern.
  [[nodiscard]] bool is_well_formed() const {
    const uint32_t scratch_end =
        static_cast<uint32_t>(scratch_vgpr) + required_scratch_vgpr_count();
    if (spill.vgpr_base != scratch_vgpr || spill.vgpr_count < required_scratch_vgpr_count() ||
        scratch_end > 256u) {
      return false;
    }
    if (!private_state_layout.is_well_formed() ||
        (private_state_layout.dispatch_id_offset &&
         (!dispatch_plan || dispatch_capture.sgpr() ||
          dispatch_capture.vgpr() != std::optional<uint16_t>{scratch_vgpr}))) {
      return false;
    }
    return true;
  }
};

/// Complete semantic input to one register-backed owner/epoch entry body.
///
/// Placement resolves this record once per kernel after assigning persistent
/// registers and decoding the kernel-entry ABI. Body and return addresses and
/// displaced guest words remain call-specific because paired kernarg-preload
/// entries may share this semantic plan while occupying different locations.
/// The native emitter therefore cannot consult full request/operating-point state, patch metadata,
/// or a kernel descriptor to rediscover any initialization decision.
struct OwnerEpochPrologueEmissionPlan {
  /// Complete vector destinations and the lifetime of the owner/epoch pair.
  VgprStateEffect vgpr_state;

  /// Logical shift converting entry workitem-x into a wave owner.
  uint16_t owner_shift_bits = 0;

  /// Resolved architectural source of owner identity.
  OwnerSource owner_source = OwnerSource::Automatic;

  /// Scalar temporary or persistent destination required by HW_ID ownership.
  std::optional<uint16_t> owner_sgpr;

  /// Persistent scalar destinations for owner, epoch, and exact workgroup identity.
  PersistentSgprState persistent_sgprs;

  /// Descriptor-derived dispatch preload transformation fixed by planning.
  std::optional<DispatchIdPreloadPlan> dispatch_plan;

  /// Unique register representation receiving the dispatch identity.
  DispatchIdCapture dispatch_capture;

  /// Optional entry-local carrier preserving a borrowed guest SGPR window.
  std::optional<EntryScalarBackup> entry_scalar_backup;

  /// Resolved launch-coordinate sources needed by workgroup identity.
  std::optional<WorkgroupSources> workgroup_sources;

  /// Return whether all target-independent invariants are safe to lower.
  [[nodiscard]] bool is_well_formed() const {
    if (!vgpr_state.is_well_formed() || owner_shift_bits >= 32u ||
        owner_source == OwnerSource::Automatic ||
        (owner_source == OwnerSource::HwId && !owner_sgpr) ||
        (workgroup_sources && !workgroup_sources->is_well_formed())) {
      return false;
    }
    return true;
  }
};

/// Semantic request to materialize the current resident wave's owner identity
/// in one scalar register.
///
/// `destination_sgpr` names the register selected by resource planning.
/// The request deliberately contains no architecture or HWREG numbers:
/// those belong to `TargetProfile` and are lowered by the target
/// operation below.
struct ResidentWaveOwnerRequest {
  uint16_t destination_sgpr = 0;

  bool operator==(const ResidentWaveOwnerRequest &) const = default;
};

/// Append the target instruction sequence implementing one resident-wave
/// owner request.
///
/// The operation is transactional: an invalid destination or target encoding
/// returns false without changing `words`. Successful output ends with the
/// target's required scalar-to-vector dependency wait, so later vector
/// consumers observe the completed identity. The target profile supplies all
/// architectural selection; callers supply only semantic intent.
[[nodiscard]] bool append_resident_wave_owner(std::vector<uint32_t> &words,
                                              const ResidentWaveOwnerRequest &request,
                                              const TargetProfile &target);

/// Complete semantic request to replay one displaced guest LDS access.
///
/// `image` is the pristine code-object image from which an unchanged guest
/// instruction can be copied. `candidate` supplies the normalized access
/// semantics and original file range. `target` binds the request to the
/// selected architectural contract. `replay_address_vgpr` is the address base
/// that is valid at the relocation site; `adjusted_address_vgpr` is optional
/// scratch reserved by planning for a split address whose static offset does
/// not fit the target instruction. No mode policy or report state belongs in
/// this request.
struct GuestAccessRelocationRequest {
  std::span<const uint8_t> image;
  const Candidate *candidate = nullptr;
  const TargetProfile *target = nullptr;
  uint16_t replay_address_vgpr = 0;
  std::optional<uint16_t> adjusted_address_vgpr;
};

/// Return whether relocation of this candidate requires an extra address
/// VGPR. The answer is shared by resource planning and target emission so they
/// cannot disagree about the split-instruction scratch contract.
[[nodiscard]] bool guest_access_relocation_requires_adjusted_address(const Candidate &candidate,
                                                                     const TargetProfile &target);

/// Build the target instruction words that replay one displaced guest access.
///
/// Ordinary targets copy the pristine instruction exactly. A target whose
/// profile requires a two-address split receives semantically equivalent
/// single-address instructions using the request's replay address. Malformed
/// input returns no words and appends a diagnostic; the pristine image is
/// never modified.
[[nodiscard]] std::optional<std::vector<uint32_t>>
build_relocated_guest_access_words(const GuestAccessRelocationRequest &request,
                                   std::vector<std::string> &errors);

/// Build and append one relocated guest access, publishing its exact encoded
/// width when requested. This is the single mutation boundary shared by mode
/// emitters; address selection and surrounding ordering remain mode policy.
[[nodiscard]] bool append_relocated_guest_access(
    std::vector<uint32_t> &words, std::span<const uint8_t> image, const Candidate &candidate,
    const TargetProfile *target, uint16_t replay_address_vgpr,
    std::optional<uint16_t> adjusted_address_vgpr, std::vector<std::string> &errors,
    uint32_t *guest_instruction_word_count = nullptr);

/// Selected scalar-register plan for preserving the guest's VCC and SCC
/// across one injected operation sequence.
///
/// `vcc_save_sgpr` is the first register of an aligned scalar pair that holds
/// the complete guest VCC value. `scc_save_sgpr` holds zero or one, produced by
/// materializing the incoming SCC predicate. The two roles are intentionally
/// named rather than represented as offsets from a broad EXEC-save window:
/// placement owns the layout, while the target operation preserves the values.
struct SpecialStateSgprs {
  uint16_t vcc_save_sgpr = 0;
  uint16_t scc_save_sgpr = 0;

  bool operator==(const SpecialStateSgprs &) const = default;
};

/// Append a transactional snapshot of guest SCC followed by guest VCC.
///
/// SCC is captured first because every later scalar instruction is permitted
/// to overwrite it. Invalid register assignments or unsupported target
/// encodings return false without appending either instruction.
[[nodiscard]] bool append_save_special_state(std::vector<uint32_t> &words,
                                             const SpecialStateSgprs &registers,
                                             const TargetProfile &target);

/// Append a transactional restoration of guest VCC followed by guest SCC.
///
/// SCC restoration is deliberately last so no injected scalar comparison can
/// overwrite the value that the displaced guest instruction observes.
/// Invalid assignments return false without changing `words`.
[[nodiscard]] bool append_restore_special_state(std::vector<uint32_t> &words,
                                                const SpecialStateSgprs &registers,
                                                const TargetProfile &target);

/// Append the target's device-scope cache refresh before retrying a contended
/// global publication.
///
/// CDNA3/CDNA4 targets require an explicit buffer invalidate sequence;
/// targets whose coherent atomic-load path needs no extra instruction succeed
/// without appending words. Encoding failure returns false without partial
/// output.
[[nodiscard]] bool append_device_cache_refresh(std::vector<uint32_t> &words,
                                               const TargetProfile &target);

/// Append the waits required after a returning device-scope global atomic.
///
/// Every target waits for the returned load value. Targets with separately
/// tracked global-store completion also wait for the memory-side effect; CDNA3/CDNA4
/// CDNA's unified VM counter needs only the first wait. The target profile owns
/// that distinction. Encoding failure is transactional and leaves `words`
/// unchanged.
[[nodiscard]] bool append_global_atomic_completion(std::vector<uint32_t> &words,
                                                   const TargetProfile &target);

/// Semantic request to reserve the next slot from a device-visible report
/// counter and return its previous value.
///
/// `counter_address` identifies the 32-bit counter. `address_vgpr` names a
/// consecutive pair used to materialize that address, while `result_vgpr`
/// carries the increment operand and receives the prior counter value. The
/// result must not overlap the address pair. This request is shared by access,
/// synchronization, diagnostic, and visibility publications; record meaning
/// remains outside the target operation.
struct AtomicCounterIncrementRequest {
  uint64_t counter_address = 0;
  uint16_t result_vgpr = 0;
  uint16_t address_vgpr = 0;

  bool operator==(const AtomicCounterIncrementRequest &) const = default;
};

/// Append one returning device-scope atomic increment and its completion
/// waits.
///
/// Invalid or overlapping register assignments and unsupported target
/// encodings return false without partial output. On success, the result VGPR
/// contains the pre-increment counter value and can be used as a record slot.
[[nodiscard]] bool append_atomic_counter_increment(std::vector<uint32_t> &words,
                                                   const AtomicCounterIncrementRequest &request,
                                                   const TargetProfile &target);

/// Resolved source and wave geometry used to derive a workitem-based owner.
///
/// ConSan identifies an owner by shifting the workitem-x identity
/// by `wave_size_shift`, yielding the wave's index within its workgroup. When
/// `entry_workitem_x_private_offset` is absent, lowering reads the live ABI
/// workitem-x VGPR. When present, lowering reloads the value captured by the
/// entry prologue; this is required when guest code may have overwritten the
/// live ABI VGPR before the synchronization event.
///
/// This type is the planning-to-emission contract. In particular, it contains
/// the already-resolved shift rather than a kernel-descriptor address, so
/// emitters cannot reinterpret descriptor ABI state or choose a different
/// wave geometry.
struct WorkitemOwnerDerivationPlan {
  /// Byte offset of an entry-captured workitem-x value in private memory. An
  /// absent offset selects the live ABI workitem-x VGPR instead.
  std::optional<uint32_t> entry_workitem_x_private_offset;

  /// Logical right shift converting workitem-x into a zero-based wave owner.
  uint16_t wave_size_shift = 0;

  /// A b32 logical shift admits values 0..31. Production wave sizes currently
  /// resolve to shifts 5 or 6, but keeping the invariant instruction-shaped
  /// avoids baking the present target set into the semantic contract.
  [[nodiscard]] bool is_well_formed() const { return wave_size_shift < 32u; }

  bool operator==(const WorkitemOwnerDerivationPlan &) const = default;
};

/// Semantic request to materialize a planned workitem owner in one VGPR.
///
/// `plan` fixes both the identity source and wave geometry. `result_vgpr` is a
/// temporary selected by the event-specific resource plan. The request does
/// not expose instruction encodings, wait counters, or target-family choices.
struct WorkitemOwnerDerivationRequest {
  /// Complete source and wave-geometry decision made during planning.
  WorkitemOwnerDerivationPlan plan;

  /// VGPR that receives the derived owner value.
  uint16_t result_vgpr = 0;

  [[nodiscard]] bool is_well_formed() const { return plan.is_well_formed() && result_vgpr < 256u; }

  bool operator==(const WorkitemOwnerDerivationRequest &) const = default;
};

/// Immutable handoff from synchronization evidence policy to ConSan atomic
/// resource planning and emission.
///
/// The observation plan owns why this operation must be observed. This value
/// joins that decision to the sole normalized synchronization sequence and to
/// the stable decoded source needed by native lowering. A target emitter
/// resolves operands from `source_site` at its boundary, but it must not
/// rediscover release/acquire meaning from instruction bits or choose a
/// different evidence intent. Language-level atomic load/store sequences use
/// the same handle contract; their ordered suffix remains owned by `sequence`.
struct AtomicEvidenceSitePlan {
  /// Address-only modification intent; no synchronization sequence is required.
  bool publication_modification = false;
  /// Authoritative synchronization event selected by evidence policy.
  SyncEventId event;

  /// Stable inventory-local identity of the sequence that establishes ordering.
  SyncSequenceId sequence;

  /// Before-guest intent that preserves the effective communication address.
  ProbeIntentId address_capture_intent;

  /// After-guest evidence intent implemented by this plan.
  ProbeIntentId evidence_intent;

  /// Classifier-owned target form shared by address planning and the selected
  /// ordering mechanism. Native lowerers consume it without re-admission.
  AtomicLoweringForm lowering_form;

  /// Stable decoded source whose operands and execution owners scope lowering.
  ProgramSiteId source_site;

  /// Verify the cross-stage identities and intent invariants.
  [[nodiscard]] std::array<ProbeIntentId, 2> intent_ids() const {
    return {address_capture_intent, evidence_intent};
  }
  [[nodiscard]] bool is_well_formed() const {
    return (publication_modification || (event.valid() && sequence.valid())) &&
           source_site.valid() && address_capture_intent.valid() && evidence_intent.valid() &&
           address_capture_intent != evidence_intent &&
           lowering_form.kind != AtomicLoweringFormKind::Count;
  }
};

/// Immutable handoff from barrier evidence policy to common ConSan resource
/// planning and a mode's barrier emitter.
///
/// Barrier policy may coalesce a signal/wait pair or a longer lifecycle into
/// one intent placed at its completing instruction. This plan names exactly
/// that admitted placement event and the unique normalized graph sequence it
/// completes, while retaining only the decoded instruction and container
/// coordinates required by ConSan lowering.
struct BarrierEvidenceSitePlan {
  /// Stable inventory-local identity of the completing barrier event.
  SyncEventId event;

  /// Stable inventory-local identity of the normalized barrier sequence.
  SyncSequenceId sequence;

  /// Barrier evidence intent implemented by this plan.
  ProbeIntentId evidence_intent;

  /// Stable decoded source whose operands and execution owners scope lowering.
  ProgramSiteId source_site;

  /// Verify that the policy intent, graph event, and decoded insertion site
  /// form one complete barrier lowering operation.
  [[nodiscard]] std::array<ProbeIntentId, 1> intent_ids() const { return {evidence_intent}; }
  [[nodiscard]] bool is_well_formed() const {
    return event.valid() && sequence.valid() && source_site.valid() && evidence_intent.valid();
  }
};

/// Append the target sequence that derives one planned workitem owner.
///
/// A private-state source is reloaded and waited for before the shift; a live
/// source reads the ABI workitem-x value directly. Invalid requests or target
/// encodings fail transactionally and leave `words` unchanged.
[[nodiscard]] bool append_workitem_owner_derivation(std::vector<uint32_t> &words,
                                                    const WorkitemOwnerDerivationRequest &request,
                                                    const TargetProfile &target);

/// Return whether one emitted ConSan patch makes its owning kernel consume all
/// three launch workgroup coordinates. This shared descriptor-mutation and
/// validation predicate distinguishes ConSan patches from independently composed
/// mutation and malformed-barrier patches. CDNA consumes the tuple only
/// through an entry capture with complete persistent storage; RDNA and CDNA5
/// observation bodies consume the firmware payload directly.
[[nodiscard]] inline bool
patch_requires_full_workgroup_id_payload(Mode mode, rj_code_arch_t arch,
                                         const PatchLoweringProduct &patch) {
  const TargetProfile *target = target_profile(arch);
  if (target == nullptr || patch.owner_descriptor_file_offsets.empty() ||
      mode == Mode::SuperCollider || mode == Mode::None) {
    return false;
  }
  if (target->placement.full_workgroup_payload_consumption ==
      WorkgroupPayloadConsumption::EntryCapture) {
    const bool entry_capture = patch.kind == PatchKind::KernelEntryOwnerEpochPrologue ||
                               patch.kind == PatchKind::KernelEntryPrivateEpochPrologue;
    return entry_capture && (patch.persistent_sgpr_state.exact_workgroup.complete() ||
                             (patch.vgpr_state && patch.vgpr_state->exact_workgroup.complete()) ||
                             (patch.private_state_layout &&
                              patch.private_state_layout->exact_workgroup_offsets.complete()));
  }
  return (patch.kind >= PatchKind::InlineWatchpointStore &&
          patch.kind <= PatchKind::TrampolineSyncMetadata);
}

/// Release-active outcome from recovering one guest VGPR out of an
/// instrumentation spill window.
enum class SpilledVgprReloadResult : uint8_t {
  Appended,
  SourceOutsideWindow,
  IncompleteSlotMetadata,
  UnsupportedEncoding,
};

[[nodiscard]] const char *spilled_vgpr_reload_result_name(SpilledVgprReloadResult result);

/// Append one private-memory reload without leaving partial output on failure.
///
/// Dynamic-stack spills select the target-native scalar-addressed encoding.
/// Fixed-frame spills use the address-free private-load encoding.
[[nodiscard]] SpilledVgprReloadResult
append_reload_spilled_vgpr(std::vector<uint32_t> &words, const VgprSpillSequence &spill,
                           uint16_t destination, uint16_t source, rj_code_arch_t arch);

struct ScalarOwnerContextSummary {
  std::optional<uint64_t> descriptor_file_offset;
  uint16_t current_sgpr_count = 0;
  uint16_t max_referenced_sgpr_count = 0;
  /// Scalar-relative access can reach registers absent from explicit def/use
  /// sets, so a static maximum alone does not bound this owner.
  bool has_indirect_sgpr_access = false;
  /// True only when every executable control-flow destination is represented
  /// by the owner reference scan.
  bool sgpr_reference_coverage_complete = false;
  bool descriptor_valid = false;
};

struct ScalarOwnerContextResolution {
  std::vector<size_t> context_indices;
  /// Hard bound that dominates every SGPR any resolved owner may reach.
  uint32_t tail_floor = 0;
};

[[nodiscard]] uint16_t scalar_owner_tail_floor(const ScalarOwnerContextSummary &context);

struct ScalarOwnerSgprRange {
  uint16_t base = 0;
  uint16_t width = 0;
};

/// Return whether any owner's original or grown CDNA physical VCC aliases one
/// of the requested ordinary-SGPR ranges. Empty or invalid owner sets fail
/// closed.
[[nodiscard]] bool scalar_owner_contexts_conflict_with_physical_vcc(
    std::span<const ScalarOwnerContextSummary> contexts,
    std::span<const ScalarOwnerSgprRange> ranges, uint32_t required_sgpr_count_floor = 0u);

/// Return whether every owner admits a persistent ordinary-SGPR window above
/// its complete scalar tail. CDNA callers additionally request physical-VCC
/// qualification.
[[nodiscard]] bool
scalar_owner_contexts_admit_reserved_window(std::span<const ScalarOwnerContextSummary> contexts,
                                            uint16_t base, uint16_t width,
                                            bool protect_physical_vcc);

/// Resolve every requested owner to one valid context and compute the scalar
/// tail beyond all original allocations and statically referenced registers.
/// Empty owner sets and every inconsistent planning state fail closed.
[[nodiscard]] std::optional<ScalarOwnerContextResolution>
resolve_scalar_owner_contexts(bool planning_state_valid,
                              std::span<const ScalarOwnerContextSummary> contexts,
                              std::span<const uint64_t> owners);

/// Validate the site-local VGPR half of scalar-persistent ConSan state before
/// emission. This remains release-active because ConSan rewrites untrusted
/// code objects and must fail cleanly if placement and emission ever diverge.
[[nodiscard]] bool validate_scalar_state_temporaries(const PersistentSgprState &persistent_sgprs,
                                                     const OwnerEpochVgprSources &owner_epoch_vgprs,
                                                     std::string_view consumer,
                                                     std::vector<std::string> &errors);

/// Materialize one persistent workgroup-coordinate source, including any
/// ABI-specific extraction applied after a scalar, vector, or private load.
[[nodiscard]] bool append_workgroup_source_value(std::vector<uint32_t> &words,
                                                 const WorkgroupSource &source, uint16_t value_vgpr,
                                                 rj_code_arch_t arch);

/// Semantic request to materialize one indexed table entry.
///
/// `table_address` is the absolute address of entry zero, `stride_bytes` is
/// the distance between adjacent entries, and `index_vgpr` holds the runtime
/// index. `address_vgpr` names the consecutive output pair. Keeping these
/// roles in one request prevents consumers from reordering a positional
/// address/index pair or supplying an architecture as semantic input. Callers
/// guarantee that `index * stride_bytes` fits in 32 bits.
struct IndexedAddressRequest {
  uint64_t table_address = 0;
  uint32_t stride_bytes = 0;
  uint16_t address_vgpr = 0;
  uint16_t index_vgpr = 0;

  bool operator==(const IndexedAddressRequest &) const = default;
};

/// Append the target sequence for `table_address + index * stride_bytes`.
///
/// The address occupies `address_vgpr:address_vgpr+1`; the index must be
/// distinct from that pair. The operation uses only the address pair as
/// temporary storage, preserves the index, and clobbers VCC. Invalid requests
/// or target encodings fail transactionally without partial output.
[[nodiscard]] bool append_indexed_address(std::vector<uint32_t> &words,
                                          const IndexedAddressRequest &request,
                                          const TargetProfile &target);

} // namespace detail
} // namespace rocjitsu::consan
