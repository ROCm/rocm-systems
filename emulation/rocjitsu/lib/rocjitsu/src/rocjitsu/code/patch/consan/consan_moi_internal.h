// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_internal.h
/// @brief Private, directly testable invariants shared by MOI lowering paths.
///
/// Keep pure release-active checks here when multiple lowering `.inc` files
/// depend on them and direct unit coverage is more precise than a test hook in
/// the full patching pipeline.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/code/patch/consan/consan_moi_dispatch_preload.h"
#include "rocjitsu/code/patch/consan/targets/consan_moi_target_ops.h"
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

namespace rocjitsu::consan_moi_impl {

/// Adds MOI's common first-failure diagnostic policy to an instruction
/// sequence. Helpers can append directly to the shared word stream; invoking
/// this requirement records only the first failing operation and makes the
/// whole sequence fail atomically.
class MoiEmissionRequirement {
public:
  MoiEmissionRequirement(InstructionSequence &sequence, std::vector<std::string> &errors)
      : sequence_(sequence), errors_(errors) {}

  void operator()(bool success, std::string_view message = {}) {
    if (sequence_ && !success && !message.empty())
      errors_.emplace_back(message);
    sequence_.require(success);
  }

private:
  InstructionSequence &sequence_;
  std::vector<std::string> &errors_;
};

/// Stage-aware first-failure diagnostics for one transactional MOI program.
///
/// Large publication transactions advance through named semantic stages while
/// appending to one InstructionSequence. This owner records the stage at which
/// the sequence first failed, preserves an optional leaf diagnostic, and emits
/// the common transaction diagnostic unless finish() succeeds. Keeping that
/// lifecycle here prevents mode emitters from independently implementing
/// subtly different first-failure rules.
class MoiStagedEmission {
public:
  MoiStagedEmission(InstructionSequence &sequence, std::vector<std::string> &errors,
                    std::string_view diagnostic_prefix)
      : sequence_(sequence), errors_(errors), diagnostic_prefix_(diagnostic_prefix) {}

  ~MoiStagedEmission() {
    if (!succeeded_)
      errors_.emplace_back(std::string(diagnostic_prefix_) + " failed during " +
                           std::string(failed_stage_.empty() ? stage_ : failed_stage_));
  }

  MoiStagedEmission(const MoiStagedEmission &) = delete;
  MoiStagedEmission &operator=(const MoiStagedEmission &) = delete;

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

/// Immutable mode semantics and ABI facts that affect common resource solving
/// and emission. These decisions are selected once by the mode owner from
/// normalized object and runtime-binding facts. They are not solver choices
/// and therefore must not be copied into or rediscovered from the mutable
/// operating point.
struct MoiObjectModeSemantics {
  bool inline_access_present = false;
  bool automatic_banked_record_capture = false;
  ConSanMoiReportBufferLayout report_layout;
  uint32_t reserved_atomic_patch_count = 0;

  bool operator==(const MoiObjectModeSemantics &) const = default;
};

} // namespace rocjitsu::consan_moi_impl

namespace rocjitsu {

/// Immutable input to one MOI resource-solving run.
///
/// The problem binds the exact image and target to effective semantic request,
/// runtime binding, published inventory/policy, and normalized access sites.
/// It deliberately contains no selected register, site plan, diagnostic, CFG
/// cache, or mutable reservation. Those belong to an attempted operating point,
/// its result, or the solver's private workspace.
class MoiResourceProblem {
public:
  MoiResourceProblem(std::span<const uint8_t> image, rj_code_arch_t arch,
                     const ConSanRequest &request, const BoundRuntimeResources &resources,
                     const ProgramInventory &inventory,
                     const ConSanObservationPlan &observation_plan,
                     std::span<const ConSanMoiCandidate> candidates,
                     consan_moi_impl::MoiObjectModeSemantics mode_semantics)
      : image_(image), arch_(arch), request_(&request), resources_(&resources),
        inventory_(&inventory), observation_plan_(&observation_plan), candidates_(candidates),
        mode_semantics_(mode_semantics) {}

  [[nodiscard]] std::span<const uint8_t> image() const { return image_; }
  [[nodiscard]] rj_code_arch_t arch() const { return arch_; }
  [[nodiscard]] const ConSanRequest &request() const { return *request_; }
  [[nodiscard]] const BoundRuntimeResources &resources() const { return *resources_; }
  [[nodiscard]] const ProgramInventory &inventory() const { return *inventory_; }
  [[nodiscard]] const ConSanObservationPlan &observation_plan() const { return *observation_plan_; }
  [[nodiscard]] std::span<const ConSanMoiCandidate> candidates() const { return candidates_; }
  [[nodiscard]] const consan_moi_impl::MoiObjectModeSemantics &mode_semantics() const {
    return mode_semantics_;
  }

private:
  std::span<const uint8_t> image_;
  rj_code_arch_t arch_ = ROCJITSU_CODE_ARCH_INVALID;
  const ConSanRequest *request_ = nullptr;
  const BoundRuntimeResources *resources_ = nullptr;
  const ProgramInventory *inventory_ = nullptr;
  const ConSanObservationPlan *observation_plan_ = nullptr;
  std::span<const ConSanMoiCandidate> candidates_;
  consan_moi_impl::MoiObjectModeSemantics mode_semantics_;
};

/// Immutable facts that determine the size of MOI's transient scalar-save ABI.
///
/// The construction authority projects only request, runtime-binding, and
/// attempted-operating-point facts that affect the ABI. Placement and emission
/// consume this value instead of independently consulting the complete MOI
/// option bus, so changing an unrelated option cannot change scalar sizing.
struct MoiExecSaveRequirement {
  ConSanMoiEngine engine = ConSanMoiEngine::RecordReplay;
  bool has_report_buffer = false;
  bool track_atomics = false;
  bool automatic_banked_record_capture = false;
  uint32_t runtime_sample_stride = 1u;
  bool scalar_spill = false;
  bool dynamic_stack_spill = false;
  bool inline_access_present = false;

  bool operator==(const MoiExecSaveRequirement &) const = default;
};

[[nodiscard]] MoiExecSaveRequirement
resolve_moi_exec_save_requirement(const ConSanRequest &request,
                                  const BoundRuntimeResources &resources,
                                  const ConSanMoiOperatingPoint &operating_point,
                                  const consan_moi_impl::MoiObjectModeSemantics &mode_semantics);

[[nodiscard]] std::optional<uint16_t>
moi_dynamic_stack_frame_save_sgpr_offset(ConSanMoiEngine engine);

[[nodiscard]] uint16_t moi_exec_save_sgpr_count(const MoiExecSaveRequirement &requirement,
                                                rj_code_arch_t arch);

/// One scalar, vector, or entry-captured private-state source for a
/// workgroup-coordinate component.
///
/// The none-set state represents an absent dimension. More than one set source
/// is malformed. Keeping this release-active invariant beside the
/// representation prevents individual emitters from interpreting ambiguous
/// sources differently.
struct ConSanMoiWorkgroupSource {
  std::optional<uint16_t> scalar_src;
  std::optional<uint16_t> vector_src;
  std::optional<uint32_t> private_offset;
  bool shift_right_16 = false;
  bool mask_low_16 = false;

  [[nodiscard]] uint8_t source_count() const {
    return static_cast<uint8_t>(scalar_src.has_value()) +
           static_cast<uint8_t>(vector_src.has_value()) +
           static_cast<uint8_t>(private_offset.has_value());
  }
  [[nodiscard]] bool has_value() const { return source_count() == 1u; }
  [[nodiscard]] bool is_well_formed() const { return source_count() <= 1u; }
  [[nodiscard]] std::optional<uint16_t> operand() const;

  [[nodiscard]] static ConSanMoiWorkgroupSource scalar(uint16_t source, bool shift_right_16 = false,
                                                       bool mask_low_16 = false) {
    ConSanMoiWorkgroupSource result;
    result.scalar_src = source;
    result.shift_right_16 = shift_right_16;
    result.mask_low_16 = mask_low_16;
    return result;
  }
  [[nodiscard]] static ConSanMoiWorkgroupSource vector(uint16_t source, bool shift_right_16 = false,
                                                       bool mask_low_16 = false) {
    ConSanMoiWorkgroupSource result;
    result.vector_src = source;
    result.shift_right_16 = shift_right_16;
    result.mask_low_16 = mask_low_16;
    return result;
  }
  [[nodiscard]] static ConSanMoiWorkgroupSource private_state(uint32_t offset) {
    ConSanMoiWorkgroupSource result;
    result.private_offset = offset;
    return result;
  }

  auto operator<=>(const ConSanMoiWorkgroupSource &) const = default;
};

/// Complete descriptor-resolved launch-coordinate view for one kernel entry.
///
/// The four coordinate members use the same typed scalar, vector, private, or
/// absent representation so prologue and event emitters cannot disagree about
/// operand interpretation. The CDNA payload members describe a temporary
/// expanded system-SGPR suffix: an entry prologue may capture coordinates from
/// that suffix and then restore the packed suffix expected by guest code.
struct ConSanMoiWorkgroupSources {
  /// Source of the mandatory x workgroup coordinate.
  ConSanMoiWorkgroupSource x;

  /// Source of y, or an absent source for a one-dimensional launch.
  ConSanMoiWorkgroupSource y;

  /// Source of z, or an absent source for a one- or two-dimensional launch.
  ConSanMoiWorkgroupSource z;

  /// Source of the CDNA5 cluster-local workgroup coordinate, or absent when
  /// the launch ABI does not expose one.
  ConSanMoiWorkgroupSource cluster_workgroup_id;

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

  auto operator<=>(const ConSanMoiWorkgroupSources &) const = default;
};

namespace consan_detail {

/// Return whether two half-open register ranges overlap.
[[nodiscard]] bool range_overlaps(uint16_t lhs_base, uint16_t lhs_count, uint16_t rhs_base,
                                  uint16_t rhs_count);

/// Reject an optional persistent VGPR that aliases a transient scratch window.
[[nodiscard]] bool reject_optional_scratch_range_overlap(std::optional<uint16_t> value,
                                                         uint16_t scratch_vgpr,
                                                         uint16_t scratch_count,
                                                         std::string_view value_name,
                                                         std::vector<std::string> &errors);

[[nodiscard]] bool reject_atomic_candidate_scratch_overlap(const ConSanAtomicLoweringForm &form,
                                                           uint16_t scratch_vgpr,
                                                           uint16_t scratch_vgpr_count,
                                                           std::vector<std::string> &errors);

/// Return whether one of the three preceding dwords is a saveexec operation.
/// This conservative guard prevents insertion inside compiler-produced EXEC
/// narrowing sequences.
[[nodiscard]] bool has_recent_saveexec(std::span<const uint8_t> bytes,
                                       const ConSanMoiCandidate &candidate);

/// Complete semantic input to one private-state entry-initialization body.
///
/// Placement constructs this plan after it has resolved the private layout,
/// temporary register window, guest-state preservation, launch ABI sources,
/// and optional dispatch and runtime-selection behavior for one kernel. The
/// native emitter receives this plan plus only the body and return addresses,
/// which may differ between paired kernarg-preload entries. It therefore
/// cannot independently reinterpret full request/operating-point state, patch metadata, or the
/// kernel descriptor while emitting those bodies.
struct MoiPrivateEpochPrologueEmissionPlan {
  /// First VGPR in the entry-local temporary window.
  uint16_t scratch_vgpr = 0;

  /// Complete persistent identity and ephemeral-allocation layout selected by
  /// private-state planning.
  ConSanMoiPrivateStateLayout private_state_layout;

  /// Owned save/restore program for the borrowed temporary VGPR window.
  VgprSpillSequence spill;

  /// Owned optional save/restore program for entry ABI SGPRs borrowed by the
  /// prologue.
  std::optional<SgprSpillSequence> entry_scalar_spill;

  /// Optional LDS shadow region initialized cooperatively at kernel entry.
  std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow;

  /// Resolved launch-coordinate sources needed by workgroup identity and
  /// runtime-selection initialization.
  std::optional<ConSanMoiWorkgroupSources> workgroup_sources;

  /// Descriptor-derived dispatch preload transformation fixed by planning.
  std::optional<ConSanMoiDispatchIdPreloadPlan> dispatch_plan;

  /// Unique register representation that receives the dispatch identity.
  ConSanMoiDispatchIdCapture dispatch_capture;

  /// Optional launch-coordinate source controlling runtime workgroup
  /// selection for sampled and Record/Replay instrumentation.
  std::optional<ConSanMoiWorkgroupSource> runtime_workgroup_selection_source;

  /// First SGPR in the entry-local scalar scratch window, when required.
  std::optional<uint16_t> return_pc_sgpr;

  /// Power-of-two modulus used by runtime workgroup selection.
  uint32_t runtime_sample_stride = 1u;

  /// Selected residue in `[0, runtime_sample_stride)`.
  uint32_t runtime_sample_offset = 0u;

  /// Stable report identity mixed into runtime workgroup selection.
  uint64_t runtime_report_dispatch_id = 0u;

  /// Number of consecutive temporary VGPRs required by the selected semantic
  /// operations, independent of their target instruction encodings.
  [[nodiscard]] uint16_t required_scratch_vgpr_count() const {
    if (private_state_layout.workgroup_key_offset)
      return 3u;
    if (private_state_layout.dispatch_id_offset || workgroup_shadow)
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
    return runtime_sample_stride != 0u &&
           (runtime_sample_stride & (runtime_sample_stride - 1u)) == 0u &&
           runtime_sample_offset < runtime_sample_stride;
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
struct MoiOwnerEpochPrologueEmissionPlan {
  /// Complete vector destinations and the lifetime of the owner/epoch pair.
  ConSanMoiVgprStateEffect vgpr_state;

  /// Logical shift converting entry workitem-x into a wave owner.
  uint16_t owner_shift_bits = 0;

  /// Resolved architectural source of owner identity.
  ConSanMoiOwnerSource owner_source = ConSanMoiOwnerSource::Automatic;

  /// Scalar temporary or persistent destination required by HW_ID ownership.
  std::optional<uint16_t> owner_sgpr;

  /// Whether zero is reserved for an empty shadow cell and real owners are
  /// represented as their architectural identity plus one.
  bool one_based_owner_ids = false;

  /// Persistent scalar destinations selected for owner, epoch, workgroup key,
  /// and the exact workgroup identity shared by exact-coordinate modes.
  ConSanMoiPersistentSgprState persistent_sgprs;

  /// Descriptor-derived dispatch preload transformation fixed by planning.
  std::optional<ConSanMoiDispatchIdPreloadPlan> dispatch_plan;

  /// Unique register representation receiving the dispatch identity.
  ConSanMoiDispatchIdCapture dispatch_capture;

  /// Optional launch-coordinate source controlling runtime workgroup
  /// selection.
  std::optional<ConSanMoiWorkgroupSource> runtime_workgroup_selection_source;

  /// First SGPR in the entry-local scalar scratch window, when required.
  std::optional<uint16_t> return_pc_sgpr;

  /// Power-of-two modulus used by runtime workgroup selection.
  uint32_t runtime_sample_stride = 1u;

  /// Selected residue in `[0, runtime_sample_stride)`.
  uint32_t runtime_sample_offset = 0u;

  /// Stable report identity mixed into runtime workgroup selection.
  uint64_t runtime_report_dispatch_id = 0u;

  /// Optional entry-local carrier preserving a borrowed guest SGPR window.
  std::optional<ConSanMoiEntryScalarBackup> entry_scalar_backup;

  /// Optional LDS shadow region initialized cooperatively at kernel entry.
  std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow;

  /// Whether workgroup-shadow initialization can use the target's four-VGPR
  /// zero tuple instead of the portable two-VGPR form.
  bool has_quad_zero_tuple = false;

  /// Resolved launch-coordinate sources needed by workgroup identity and
  /// runtime-selection initialization.
  std::optional<ConSanMoiWorkgroupSources> workgroup_sources;

  /// Return whether all target-independent invariants are safe to lower.
  [[nodiscard]] bool is_well_formed() const {
    if (!vgpr_state.is_well_formed() || owner_shift_bits >= 32u ||
        owner_source == ConSanMoiOwnerSource::Automatic ||
        (owner_source == ConSanMoiOwnerSource::HwId && !owner_sgpr) ||
        (workgroup_sources && !workgroup_sources->is_well_formed())) {
      return false;
    }
    return runtime_sample_stride != 0u &&
           (runtime_sample_stride & (runtime_sample_stride - 1u)) == 0u &&
           runtime_sample_offset < runtime_sample_stride;
  }
};

/// Resolved persistent-register inputs consumed while deriving one compact
/// workgroup key.
///
/// `exec_save_sgpr` is the base of the scalar window used to narrow and later
/// restore participating lanes. A previously initialized key may reside in
/// either `cached_key_vgpr` or `cached_key_sgpr`; when neither is present, the
/// emitter derives the key from the supplied launch-coordinate sources.
/// Keeping this plan separate from the full operating point prevents the key emitter from
/// observing engine policy, report layout, or unrelated resource choices.
struct MoiWorkgroupKeyRegisterPlan {
  std::optional<uint16_t> exec_save_sgpr;
  std::optional<uint16_t> cached_key_vgpr;
  std::optional<uint16_t> cached_key_sgpr;

  /// A usable plan has EXEC preservation and at most one cached
  /// representation. Two cached sources would make selection ambiguous.
  [[nodiscard]] bool is_well_formed() const {
    return exec_save_sgpr.has_value() && !(cached_key_vgpr && cached_key_sgpr);
  }

  bool operator==(const MoiWorkgroupKeyRegisterPlan &) const = default;
};

/// Semantic request to materialize the current resident wave's owner identity
/// in one scalar register.
///
/// `destination_sgpr` names the register selected by resource planning.
/// `one_based` requests the representation used by packed ConSan shadows,
/// where zero denotes an empty cell and architectural identity N is stored as
/// N+1. The request deliberately contains no architecture or HWREG numbers:
/// those belong to `ConSanTargetProfile` and are lowered by the target
/// operation below.
struct MoiResidentWaveOwnerRequest {
  uint16_t destination_sgpr = 0;
  bool one_based = false;

  bool operator==(const MoiResidentWaveOwnerRequest &) const = default;
};

/// Append the target instruction sequence implementing one resident-wave
/// owner request.
///
/// The operation is transactional: an invalid destination or target encoding
/// returns false without changing `words`. Successful output ends with the
/// target's required scalar-to-vector dependency wait, so later vector
/// consumers observe the completed identity. The target profile supplies all
/// architectural selection; callers supply only semantic intent.
[[nodiscard]] bool append_moi_resident_wave_owner(std::vector<uint32_t> &words,
                                                  const MoiResidentWaveOwnerRequest &request,
                                                  const ConSanTargetProfile &target);

/// Complete semantic request to replay one displaced guest LDS access.
///
/// `image` is the pristine code-object image from which an unchanged guest
/// instruction can be copied. `candidate` supplies the normalized access
/// semantics and original file range. `target` binds the request to the
/// selected architectural contract. `replay_address_vgpr` is the address base
/// that is valid at the relocation site; `adjusted_address_vgpr` is optional
/// scratch reserved by planning for a split address whose static offset does
/// not fit the target instruction. No engine policy or report state belongs in
/// this request.
struct MoiGuestAccessRelocationRequest {
  std::span<const uint8_t> image;
  const ConSanMoiCandidate *candidate = nullptr;
  const ConSanTargetProfile *target = nullptr;
  uint16_t replay_address_vgpr = 0;
  std::optional<uint16_t> adjusted_address_vgpr;
};

/// Return whether relocation of this candidate requires an extra address
/// VGPR. The answer is shared by resource planning and target emission so they
/// cannot disagree about the split-instruction scratch contract.
[[nodiscard]] bool
moi_guest_access_relocation_requires_adjusted_address(const ConSanMoiCandidate &candidate,
                                                      const ConSanTargetProfile &target);

/// Build the target instruction words that replay one displaced guest access.
///
/// Ordinary targets copy the pristine instruction exactly. A target whose
/// profile requires a two-address split receives semantically equivalent
/// single-address instructions using the request's replay address. Malformed
/// input returns no words and appends a diagnostic; the pristine image is
/// never modified.
[[nodiscard]] std::optional<std::vector<uint32_t>>
build_moi_relocated_guest_access_words(const MoiGuestAccessRelocationRequest &request,
                                       std::vector<std::string> &errors);

/// Build and append one relocated guest access, publishing its exact encoded
/// width when requested. This is the single mutation boundary shared by mode
/// emitters; address selection and surrounding ordering remain mode policy.
[[nodiscard]] bool append_moi_relocated_guest_access(
    std::vector<uint32_t> &words, std::span<const uint8_t> image,
    const ConSanMoiCandidate &candidate, const ConSanTargetProfile *target,
    uint16_t replay_address_vgpr, std::optional<uint16_t> adjusted_address_vgpr,
    std::vector<std::string> &errors, uint32_t *guest_instruction_word_count = nullptr);

/// Selected scalar-register plan for preserving the guest's VCC and SCC
/// across one injected operation sequence.
///
/// `vcc_save_sgpr` is the first register of an aligned scalar pair that holds
/// the complete guest VCC value. `scc_save_sgpr` holds zero or one, produced by
/// materializing the incoming SCC predicate. The two roles are intentionally
/// named rather than represented as offsets from a broad EXEC-save window:
/// placement may choose different layouts for Record/Replay, Sampled, and
/// InlineShadow, while the target preservation operation is identical.
struct MoiSpecialStateSgprs {
  uint16_t vcc_save_sgpr = 0;
  uint16_t scc_save_sgpr = 0;

  bool operator==(const MoiSpecialStateSgprs &) const = default;
};

/// Append a transactional snapshot of guest SCC followed by guest VCC.
///
/// SCC is captured first because every later scalar instruction is permitted
/// to overwrite it. Invalid register assignments or unsupported target
/// encodings return false without appending either instruction.
[[nodiscard]] bool append_save_moi_special_state(std::vector<uint32_t> &words,
                                                 const MoiSpecialStateSgprs &registers,
                                                 const ConSanTargetProfile &target);

/// Append a transactional restoration of guest VCC followed by guest SCC.
///
/// SCC restoration is deliberately last so no injected scalar comparison can
/// overwrite the value that the displaced guest instruction observes.
/// Invalid assignments return false without changing `words`.
[[nodiscard]] bool append_restore_moi_special_state(std::vector<uint32_t> &words,
                                                    const MoiSpecialStateSgprs &registers,
                                                    const ConSanTargetProfile &target);

/// Append the target's device-scope cache refresh before retrying a contended
/// global publication.
///
/// CDNA3/CDNA4 targets require an explicit buffer invalidate sequence;
/// targets whose coherent atomic-load path needs no extra instruction succeed
/// without appending words. Encoding failure returns false without partial
/// output.
[[nodiscard]] bool append_moi_device_cache_refresh(std::vector<uint32_t> &words,
                                                   const ConSanTargetProfile &target);

/// Append the waits required after a returning device-scope global atomic.
///
/// Every target waits for the returned load value. Targets with separately
/// tracked global-store completion also wait for the memory-side effect; CDNA3/CDNA4
/// CDNA's unified VM counter needs only the first wait. The target profile owns
/// that distinction. Encoding failure is transactional and leaves `words`
/// unchanged.
[[nodiscard]] bool append_moi_global_atomic_completion(std::vector<uint32_t> &words,
                                                       const ConSanTargetProfile &target);

/// Semantic request to reserve the next slot from a device-visible report
/// counter and return its previous value.
///
/// `counter_address` identifies the 32-bit counter. `address_vgpr` names a
/// consecutive pair used to materialize that address, while `result_vgpr`
/// carries the increment operand and receives the prior counter value. The
/// result must not overlap the address pair. This request is shared by access,
/// synchronization, diagnostic, and visibility publications; record meaning
/// remains outside the target operation.
struct MoiAtomicCounterIncrementRequest {
  uint64_t counter_address = 0;
  uint16_t result_vgpr = 0;
  uint16_t address_vgpr = 0;

  bool operator==(const MoiAtomicCounterIncrementRequest &) const = default;
};

/// Append one returning device-scope atomic increment and its completion
/// waits.
///
/// Invalid or overlapping register assignments and unsupported target
/// encodings return false without partial output. On success, the result VGPR
/// contains the pre-increment counter value and can be used as a record slot.
[[nodiscard]] bool
append_moi_atomic_counter_increment(std::vector<uint32_t> &words,
                                    const MoiAtomicCounterIncrementRequest &request,
                                    const ConSanTargetProfile &target);

/// Resolved source and wave geometry used to derive a workitem-based owner.
///
/// Several MOI engines identify an owner by shifting the workitem-x identity
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
struct MoiWorkitemOwnerDerivationPlan {
  /// Byte offset of an entry-captured workitem-x value in private memory. An
  /// absent offset selects the live ABI workitem-x VGPR instead.
  std::optional<uint32_t> entry_workitem_x_private_offset;

  /// Logical right shift converting workitem-x into a zero-based wave owner.
  uint16_t wave_size_shift = 0;

  /// A b32 logical shift admits values 0..31. Production wave sizes currently
  /// resolve to shifts 5 or 6, but keeping the invariant instruction-shaped
  /// avoids baking the present target set into the semantic contract.
  [[nodiscard]] bool is_well_formed() const { return wave_size_shift < 32u; }

  bool operator==(const MoiWorkitemOwnerDerivationPlan &) const = default;
};

/// Semantic request to materialize a planned workitem owner in one VGPR.
///
/// `plan` fixes both the identity source and wave geometry. `result_vgpr` is a
/// temporary selected by the event-specific resource plan. The request does
/// not expose instruction encodings, wait counters, or target-family choices.
struct MoiWorkitemOwnerDerivationRequest {
  /// Complete source and wave-geometry decision made during planning.
  MoiWorkitemOwnerDerivationPlan plan;

  /// VGPR that receives the derived owner value.
  uint16_t result_vgpr = 0;

  [[nodiscard]] bool is_well_formed() const { return plan.is_well_formed() && result_vgpr < 256u; }

  bool operator==(const MoiWorkitemOwnerDerivationRequest &) const = default;
};

/// Immutable handoff from synchronization evidence policy to MOI atomic
/// resource planning and emission.
///
/// The observation plan owns why this operation must be observed. This value
/// joins that decision to the sole normalized synchronization sequence and to
/// the stable decoded source needed by native lowering. A target emitter
/// resolves operands from `source_site` at its boundary, but it must not
/// rediscover release/acquire meaning from instruction bits or choose a
/// different evidence intent. Language-level atomic load/store sequences use
/// the same handle contract; their ordered suffix remains owned by `sequence`.
struct MoiAtomicEvidenceSitePlan {
  /// Authoritative synchronization event selected by evidence policy.
  ConSanSyncEventId event;

  /// Stable inventory-local identity of the sequence that establishes ordering.
  ConSanSyncSequenceId sequence;

  /// Before-guest intent that preserves the effective communication address.
  ConSanProbeIntentId address_capture_intent;

  /// Engine-specific after-guest evidence intent implemented by this plan.
  ConSanProbeIntentId evidence_intent;

  /// Classifier-owned target form shared by address planning and the selected
  /// ordering mechanism. Native lowerers consume it without re-admission.
  ConSanAtomicLoweringForm lowering_form;

  /// Stable decoded source whose operands and execution owners scope lowering.
  ConSanProgramSiteId source_site;

  /// Verify the cross-stage identities and intent invariants.
  [[nodiscard]] std::array<ConSanProbeIntentId, 2> intent_ids() const {
    return {address_capture_intent, evidence_intent};
  }
  [[nodiscard]] bool is_well_formed() const {
    return event.valid() && sequence.valid() && source_site.valid() &&
           address_capture_intent.valid() && evidence_intent.valid() &&
           address_capture_intent != evidence_intent &&
           lowering_form.kind != ConSanAtomicLoweringFormKind::Count;
  }
};

/// Immutable handoff from Record/Replay fence policy to resource planning and
/// native fence emission.
///
/// A fence record represents one graph-qualified communication sequence, not
/// every cache instruction that happens to resemble a fence. This value joins
/// the admitted `FenceRecord` intent to its normalized fence association and
/// to the stable communication source whose effective address must be
/// reported. The patch range may cover only the fence or, for an acquire
/// sequence, the complete address-bearing load-through-fence interval. Native
/// lowering resolves decoded operands from `source_site` only at its boundary;
/// it must not rescan the synchronization graph or reinterpret cache ordering.
struct MoiFenceEvidenceSitePlan {
  /// Stable inventory-local identity of the admitted fence event.
  ConSanSyncEventId event;

  /// Stable inventory-local identity of the qualified communication sequence.
  ConSanSyncSequenceId sequence;

  /// After-guest `FenceRecord` intent implemented by this lowering plan.
  ConSanProbeIntentId evidence_intent;

  /// Before-guest intent that preserves the communication address reported
  /// by the fence record. This intent can name the address-bearing ordinary
  /// operation while `event` names the completing fence.
  ConSanProbeIntentId address_capture_intent;

  /// Classifier-owned address form for the communication operation.
  ConSanAtomicLoweringForm communication_lowering_form;

  /// Stable communication source whose operands and owners scope lowering.
  ConSanProgramSiteId source_site;

  /// Verify that policy, graph association, decode, and replacement range all
  /// name one complete lowering operation.
  [[nodiscard]] std::array<ConSanProbeIntentId, 2> intent_ids() const {
    return {address_capture_intent, evidence_intent};
  }
  [[nodiscard]] bool is_well_formed() const {
    return event.valid() && sequence.valid() && source_site.valid() && evidence_intent.valid() &&
           address_capture_intent.valid() && evidence_intent != address_capture_intent &&
           communication_lowering_form.kind != ConSanAtomicLoweringFormKind::Count;
  }
};

/// Immutable handoff from barrier evidence policy to common MOI resource
/// planning and an engine's barrier emitter.
///
/// Barrier policy may coalesce a signal/wait pair or a longer lifecycle into
/// one intent placed at its completing instruction. This plan names exactly
/// that admitted placement event and the unique normalized graph sequence it
/// completes, while retaining only the decoded instruction and container
/// coordinates required by lowering. Record/Replay, Sampled, and InlineShadow
/// share this selection contract even though their evidence bodies remain
/// intentionally different.
struct MoiBarrierEvidenceSitePlan {
  /// Stable inventory-local identity of the completing barrier event.
  ConSanSyncEventId event;

  /// Stable inventory-local identity of the normalized barrier sequence.
  ConSanSyncSequenceId sequence;

  /// Engine-specific barrier evidence intent implemented by this plan.
  ConSanProbeIntentId evidence_intent;

  /// Stable decoded source whose operands and execution owners scope lowering.
  ConSanProgramSiteId source_site;

  /// Verify that the policy intent, graph event, and decoded insertion site
  /// form one complete barrier lowering operation.
  [[nodiscard]] std::array<ConSanProbeIntentId, 1> intent_ids() const { return {evidence_intent}; }
  [[nodiscard]] bool is_well_formed() const {
    return event.valid() && sequence.valid() && source_site.valid() && evidence_intent.valid();
  }
};

/// Append the target sequence that derives one planned workitem owner.
///
/// A private-state source is reloaded and waited for before the shift; a live
/// source reads the ABI workitem-x value directly. Invalid requests or target
/// encodings fail transactionally and leave `words` unchanged.
[[nodiscard]] bool
append_moi_workitem_owner_derivation(std::vector<uint32_t> &words,
                                     const MoiWorkitemOwnerDerivationRequest &request,
                                     const ConSanTargetProfile &target);

/// Names the concrete store shape selected for one parallel workgroup-shadow
/// clear loop.
///
/// Unlike `ConSanWorkgroupShadowClearEncoding`, which records the widest form
/// a target admits, this enum is the resolved lowering decision for one clear.
/// A split pair writes the low and high 32-bit halves separately; the packed
/// forms write the complete eight- or sixteen-byte zero tuple atomically with
/// respect to instruction issue.
enum class MoiWorkgroupShadowClearStoreForm : uint8_t {
  SplitB32Pair,
  PackedB64,
  PackedB128,
};

/// Complete target-qualified lowering plan for one parallel LDS shadow clear.
///
/// `store_form` is the concrete instruction shape selected after considering
/// target capability, range alignment, and available consecutive zero VGPRs.
/// `initialization_lanes` is the already-planned number of entry lanes that
/// participate in each x-row. `zero_vgpr_count` is the exact consecutive tuple
/// consumed by the selected store. The emitter consumes this record without
/// inspecting an architecture ID or independently recomputing resource needs.
struct MoiWorkgroupShadowClearPlan {
  MoiWorkgroupShadowClearStoreForm store_form = MoiWorkgroupShadowClearStoreForm::PackedB64;
  uint16_t initialization_lanes = 32;
  uint16_t zero_vgpr_count = 2;

  bool operator==(const MoiWorkgroupShadowClearPlan &) const = default;
};

/// Select the number of entry lanes used by a target's shadow-clear loop.
///
/// A known x dimension wider than one wave may use up to the target's declared
/// maximum. Missing or smaller launch geometry keeps the conservative 32-lane
/// plan. The target profile is assumed to have passed the capability-contract
/// validator.
[[nodiscard]] constexpr uint16_t moi_workgroup_shadow_initialization_lanes(
    const ConSanTargetProfile &target,
    const std::optional<std::array<uint32_t, 3>> &required_workgroup_size) {
  constexpr uint16_t kConservativeLanes = 32u;
  if (!required_workgroup_size || (*required_workgroup_size)[0] <= kConservativeLanes)
    return kConservativeLanes;
  return static_cast<uint16_t>(std::min<uint32_t>((*required_workgroup_size)[0],
                                                  target.workgroup_shadow_clear.maximum_lanes));
}

/// Return the consecutive zero-VGPR tuple reserved by automatic planning.
///
/// A target capable of a packed 128-bit clear reserves four registers so
/// aligned layouts can use that form. Every other target reserves the two
/// registers needed to clear one eight-byte shadow slot, including targets
/// that encode those halves as separate 32-bit stores.
[[nodiscard]] constexpr uint16_t
moi_workgroup_shadow_preferred_zero_vgpr_count(const ConSanTargetProfile &target) {
  return target.workgroup_shadow_clear.encoding == ConSanWorkgroupShadowClearEncoding::PackedB128
             ? 4u
             : 2u;
}

/// Resolve a target capability into the concrete store plan for one layout.
///
/// The input size must describe a nonempty sequence of complete eight-byte
/// shadow slots, and the planned lane count must fit the target contract. A
/// packed 128-bit target falls back to 64-bit stores if the range is not
/// sixteen-byte aligned or resource planning did not supply a four-VGPR zero
/// tuple. Invalid inputs return no plan rather than leaving emission to infer a
/// partially valid fallback.
[[nodiscard]] constexpr std::optional<MoiWorkgroupShadowClearPlan>
plan_moi_workgroup_shadow_clear(const ConSanTargetProfile &target, uint32_t initialization_size,
                                uint16_t initialization_lanes, bool has_quad_zero_tuple) {
  if (initialization_size == 0u || initialization_size % 8u != 0u || initialization_lanes == 0u ||
      initialization_lanes > target.workgroup_shadow_clear.maximum_lanes) {
    return std::nullopt;
  }
  switch (target.workgroup_shadow_clear.encoding) {
  case ConSanWorkgroupShadowClearEncoding::SplitB32Pair:
    return MoiWorkgroupShadowClearPlan{
        .store_form = MoiWorkgroupShadowClearStoreForm::SplitB32Pair,
        .initialization_lanes = initialization_lanes,
        .zero_vgpr_count = 2u,
    };
  case ConSanWorkgroupShadowClearEncoding::PackedB64:
    return MoiWorkgroupShadowClearPlan{
        .store_form = MoiWorkgroupShadowClearStoreForm::PackedB64,
        .initialization_lanes = initialization_lanes,
        .zero_vgpr_count = 2u,
    };
  case ConSanWorkgroupShadowClearEncoding::PackedB128:
    if (has_quad_zero_tuple && initialization_size % 16u == 0u) {
      return MoiWorkgroupShadowClearPlan{
          .store_form = MoiWorkgroupShadowClearStoreForm::PackedB128,
          .initialization_lanes = initialization_lanes,
          .zero_vgpr_count = 4u,
      };
    }
    return MoiWorkgroupShadowClearPlan{
        .store_form = MoiWorkgroupShadowClearStoreForm::PackedB64,
        .initialization_lanes = initialization_lanes,
        .zero_vgpr_count = 2u,
    };
  }
  return std::nullopt;
}

/// Return whether one emitted MOI patch makes its owning kernel consume all
/// three launch workgroup coordinates. This shared descriptor-mutation and
/// validation predicate distinguishes MOI patches from independently composed
/// mutation and malformed-barrier patches. CDNA consumes the tuple only
/// through an entry capture with complete persistent storage; RDNA and CDNA5
/// observation bodies consume the firmware payload directly.
[[nodiscard]] inline bool
patch_requires_full_workgroup_id_payload(ConSanCapabilityEngine engine, rj_code_arch_t arch,
                                         const ConSanPatchLoweringProduct &patch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr || patch.owner_descriptor_file_offsets.empty() ||
      engine == ConSanCapabilityEngine::SuperCollider || engine == ConSanCapabilityEngine::Count) {
    return false;
  }
  if (target->moi_placement.full_workgroup_payload_consumption ==
      ConSanMoiWorkgroupPayloadConsumption::EntryCapture) {
    const bool entry_capture = patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue ||
                               patch.kind == ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
    return entry_capture &&
           (patch.persistent_sgpr_state.exact_workgroup.complete() ||
            (patch.moi_vgpr_state && patch.moi_vgpr_state->exact_workgroup.complete()) ||
            (patch.private_state_layout &&
             patch.private_state_layout->exact_workgroup_offsets.complete()));
  }
  return (patch.kind >= ConSanPatchKind::InlineMoiAccessRecordStore &&
          patch.kind <= ConSanPatchKind::TrampolineMoiFenceRecord);
}

/// Release-active outcome from recovering one guest VGPR out of an
/// instrumentation spill window.
enum class MoiSpilledVgprReloadResult : uint8_t {
  Appended,
  SourceOutsideWindow,
  IncompleteSlotMetadata,
  UnsupportedEncoding,
};

[[nodiscard]] const char *moi_spilled_vgpr_reload_result_name(MoiSpilledVgprReloadResult result);

/// Append one private-memory reload without leaving partial output on failure.
///
/// Dynamic-stack spills select the target-native scalar-addressed encoding.
/// Fixed-frame spills use the address-free private-load encoding.
[[nodiscard]] MoiSpilledVgprReloadResult
append_reload_moi_spilled_vgpr(std::vector<uint32_t> &words, const VgprSpillSequence &spill,
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
    std::span<const ScalarOwnerSgprRange> ranges);

/// Return whether every owner admits a persistent ordinary-SGPR window above
/// its complete scalar tail. CDNA callers additionally request physical-VCC
/// qualification.
[[nodiscard]] bool
scalar_owner_contexts_admit_reserved_window(std::span<const ScalarOwnerContextSummary> contexts,
                                            uint16_t base, uint16_t width,
                                            bool protect_physical_vcc);

/// Fully encoded semantic identity for one sampled atomic synchronization
/// candidate. Physical aliases may fold only when every field matches.
struct SampledAtomicSemantics {
  ConSanMoiSampledSyncRole role = ConSanMoiSampledSyncRole::None;
  ConSanMoiSampledSyncScope scope = ConSanMoiSampledSyncScope::None;
  ConSanMoiSampledSyncOutcome outcome = ConSanMoiSampledSyncOutcome::NotApplicable;
  uint32_t byte_count = 0;
  uint32_t descriptor = 0;
  std::optional<uint32_t> cas_failure_descriptor;

  bool operator==(const SampledAtomicSemantics &) const = default;
};

/// Return the inline-shadow transaction scratch size shared by placement and
/// emission. Atomic tracking retains additional publication state.
[[nodiscard]] constexpr uint16_t inline_shadow_transaction_scratch_count(bool has_exec_save,
                                                                         bool track_atomics) {
  return has_exec_save ? (track_atomics ? 24u : 16u) : 11u;
}

/// Return the first scratch VGPR reserved for a wide-access cell loop.
[[nodiscard]] constexpr uint16_t
inline_shadow_loop_counter_vgpr(uint16_t scratch_vgpr, bool has_exec_save, bool track_atomics) {
  return static_cast<uint16_t>(
      scratch_vgpr + inline_shadow_transaction_scratch_count(has_exec_save, track_atomics));
}

/// Return the transaction register that holds the expected version for CAS.
///
/// The address-formation phase completes before this register is initialized,
/// so a narrow spill-backed probe may use it temporarily to recover one guest
/// LDS address component.
[[nodiscard]] constexpr uint16_t inline_shadow_cas_expected_vgpr(uint16_t old_value_vgpr) {
  return static_cast<uint16_t>(old_value_vgpr + 10u);
}

/// Return the offset/counter scratch reserved when a wide access spans enough
/// exact-shadow cells to make a compact runtime loop preferable to unrolling.
[[nodiscard]] constexpr uint16_t inline_shadow_loop_scratch_count(uint32_t width_bits,
                                                                  uint32_t granule_bytes) {
  return width_bits > granule_bytes * 8u ? 2u : 0u;
}

/// Resolve every requested owner to one valid context and compute the scalar
/// tail beyond all original allocations and statically referenced registers.
/// Empty owner sets and every inconsistent planning state fail closed.
[[nodiscard]] std::optional<ScalarOwnerContextResolution>
resolve_scalar_owner_contexts(bool planning_state_valid,
                              std::span<const ScalarOwnerContextSummary> contexts,
                              std::span<const uint64_t> owners);

/// Validate the site-local VGPR half of scalar-persistent MOI state before
/// emission. This remains release-active because ConSan rewrites untrusted
/// code objects and must fail cleanly if placement and emission ever diverge.
[[nodiscard]] bool
validate_scalar_state_temporaries(const ConSanMoiPersistentSgprState &persistent_sgprs,
                                  const ConSanMoiOwnerEpochVgprSources &owner_epoch_vgprs,
                                  std::string_view consumer, std::vector<std::string> &errors);

/// Materialize one persistent workgroup-coordinate source, including any
/// ABI-specific extraction applied after a scalar, vector, or private load.
[[nodiscard]] bool append_workgroup_source_value(std::vector<uint32_t> &words,
                                                 const ConSanMoiWorkgroupSource &source,
                                                 uint16_t value_vgpr, rj_code_arch_t arch);

/// Semantic request to materialize one indexed table entry.
///
/// `table_address` is the absolute address of entry zero, `stride_bytes` is
/// the distance between adjacent entries, and `index_vgpr` holds the runtime
/// index. `address_vgpr` names the consecutive output pair. Keeping these
/// roles in one request prevents consumers from reordering a positional
/// address/index pair or supplying an architecture as semantic input. Callers
/// guarantee that `index * stride_bytes` fits in 32 bits.
struct MoiIndexedAddressRequest {
  uint64_t table_address = 0;
  uint32_t stride_bytes = 0;
  uint16_t address_vgpr = 0;
  uint16_t index_vgpr = 0;

  bool operator==(const MoiIndexedAddressRequest &) const = default;
};

/// Append the target sequence for `table_address + index * stride_bytes`.
///
/// The address occupies `address_vgpr:address_vgpr+1`; the index must be
/// distinct from that pair. The operation uses only the address pair as
/// temporary storage, preserves the index, and clobbers VCC. Invalid requests
/// or target encodings fail transactionally without partial output.
[[nodiscard]] bool append_moi_indexed_address(std::vector<uint32_t> &words,
                                              const MoiIndexedAddressRequest &request,
                                              const ConSanTargetProfile &target);

/// Return the nearest emitted trampoline body strictly after `offset` across
/// both already committed and current-pass patch inventories. Empty bodies do
/// not reserve bytes. Incremental lowering must use both inventories when it
/// grows a shared dispatcher, or it can overwrite a body emitted earlier in
/// the current pass.
template <typename CommittedPatches, typename CurrentPatches>
[[nodiscard]] std::optional<uint64_t>
next_moi_trampoline_boundary(uint64_t offset, const CommittedPatches &committed,
                             const CurrentPatches &current_pass) {
  uint64_t boundary = std::numeric_limits<uint64_t>::max();
  const auto inspect = [&](const auto &patches) {
    for (const ConSanCommittedPatchGeometry &patch : patches) {
      if (patch.trampoline_size != 0u && patch.trampoline_offset > offset)
        boundary = std::min(boundary, patch.trampoline_offset);
    }
  };
  inspect(committed);
  inspect(current_pass);
  return boundary == std::numeric_limits<uint64_t>::max() ? std::nullopt
                                                          : std::optional<uint64_t>(boundary);
}

} // namespace consan_detail
} // namespace rocjitsu
