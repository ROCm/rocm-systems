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

namespace rocjitsu {

struct MoiOptions;

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

  /// Source of the gfx1250 cluster-local workgroup coordinate, or absent when
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

/// Selects the one persistent representation that receives a dispatch ID at
/// kernel entry.
///
/// The scalar and vector forms are alternatives, never simultaneous. An
/// empty value means the current prologue does not capture dispatch identity.
/// Keeping this choice typed prevents entry initialization from silently
/// writing two independently configured representations of the same ID.
struct ConSanMoiDispatchIdCapture {
  std::optional<uint16_t> sgpr;
  std::optional<uint16_t> vgpr;

  [[nodiscard]] bool present() const { return sgpr || vgpr; }
  [[nodiscard]] bool unambiguous() const {
    return static_cast<bool>(sgpr) != static_cast<bool>(vgpr);
  }

  bool operator==(const ConSanMoiDispatchIdCapture &) const = default;
};

/// Complete semantic input to one private-state entry-initialization body.
///
/// Placement constructs this plan after it has resolved the private layout,
/// temporary register window, guest-state preservation, launch ABI sources,
/// and optional dispatch and runtime-selection behavior for one kernel. The
/// native emitter receives this plan plus only the body and return addresses,
/// which may differ between paired kernarg-preload entries. It therefore
/// cannot independently reinterpret `MoiOptions`, patch metadata, or the
/// kernel descriptor while emitting those bodies.
struct MoiPrivateEpochPrologueEmissionPlan {
  /// First VGPR in the entry-local temporary window.
  uint16_t scratch_vgpr = 0;

  /// Private-memory byte offset initialized to epoch zero.
  uint32_t epoch_offset = 0;

  /// Optional private-memory byte offset receiving entry workitem-x.
  std::optional<uint32_t> owner_offset;

  /// Optional private-memory byte offset receiving the compact workgroup key.
  std::optional<uint32_t> workgroup_key_offset;

  /// Optional first private-memory byte offset receiving the 64-bit dispatch
  /// identity. The high half occupies the next private slot.
  std::optional<uint32_t> dispatch_id_offset;

  /// Private slots receiving the exact x/y/z/cluster workgroup tuple used by
  /// Record/Replay evidence.
  ConSanMoiPersistentWorkgroupPrivateOffsets record_replay_workgroup_offsets;

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
    if (workgroup_key_offset)
      return 3u;
    if (dispatch_id_offset || workgroup_shadow)
      return 2u;
    return 1u;
  }

  /// Return whether the address-free structural contract is safe to lower.
  /// Target-specific instruction availability remains the emitter's concern.
  [[nodiscard]] bool is_well_formed() const {
    const uint32_t scratch_end =
        static_cast<uint32_t>(scratch_vgpr) + required_scratch_vgpr_count();
    if (spill.vgpr_base != scratch_vgpr || spill.vgpr_count < required_scratch_vgpr_count() ||
        scratch_end > 256u || (dispatch_capture.present() && !dispatch_capture.unambiguous())) {
      return false;
    }
    if (dispatch_id_offset && (!dispatch_plan || dispatch_capture.sgpr ||
                               dispatch_capture.vgpr != std::optional<uint16_t>{scratch_vgpr})) {
      return false;
    }
    return runtime_sample_stride != 0u &&
           (runtime_sample_stride & (runtime_sample_stride - 1u)) == 0u &&
           runtime_sample_offset < runtime_sample_stride;
  }
};

/// Register window copied into one entry-local VGPR while a prologue borrows
/// scalar ABI state.
///
/// `vgpr` is the wave-local carrier. `sgpr_base` and `sgpr_count` name the
/// contiguous scalar window transferred through its lanes. The same plan is
/// consumed for save and restore, preventing the two halves of preservation
/// from drifting. Limits are supplied explicitly because they are resolved
/// target facts rather than properties of this target-independent record.
struct MoiEntryScalarBackup {
  uint16_t vgpr = 0;
  uint16_t sgpr_base = 0;
  uint16_t sgpr_count = 0;

  [[nodiscard]] bool is_well_formed(uint16_t vgpr_limit, uint16_t sgpr_limit) const {
    return vgpr < vgpr_limit && sgpr_count != 0u && sgpr_count <= 64u &&
           static_cast<uint32_t>(sgpr_base) + sgpr_count <= sgpr_limit;
  }

  bool operator==(const MoiEntryScalarBackup &) const = default;
};

/// Complete semantic input to one register-backed owner/epoch entry body.
///
/// Placement resolves this record once per kernel after assigning persistent
/// registers and decoding the kernel-entry ABI. Body and return addresses and
/// displaced guest words remain call-specific because paired kernarg-preload
/// entries may share this semantic plan while occupying different locations.
/// The native emitter therefore cannot consult `MoiOptions`, patch metadata,
/// or a kernel descriptor to rediscover any initialization decision.
struct MoiOwnerEpochPrologueEmissionPlan {
  /// VGPR receiving the owner identity when no persistent owner SGPR exists.
  uint16_t owner_vgpr = 0;

  /// VGPR receiving epoch zero when no persistent epoch SGPR exists.
  uint16_t epoch_vgpr = 0;

  /// Optional persistent VGPR receiving the compact workgroup key.
  std::optional<uint16_t> workgroup_key_vgpr;

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
  /// and exact Record/Replay workgroup identity.
  ConSanMoiPersistentSgprState persistent_sgprs;

  /// Persistent vector destinations selected for exact Record/Replay
  /// workgroup identity.
  ConSanMoiPersistentWorkgroupRegisters record_replay_workgroup_vgprs;

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
  std::optional<MoiEntryScalarBackup> entry_scalar_backup;

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
    if (owner_vgpr >= 256u || epoch_vgpr >= 256u || owner_vgpr == epoch_vgpr ||
        owner_shift_bits >= 32u || owner_source == ConSanMoiOwnerSource::Automatic ||
        (owner_source == ConSanMoiOwnerSource::HwId && !owner_sgpr) ||
        (dispatch_capture.present() && !dispatch_capture.unambiguous()) ||
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
/// Keeping this plan separate from `MoiOptions` prevents the key emitter from
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

/// Semantic request to restore SCC from bit zero of a scalar route key.
///
/// Dense routers may pack the caller's boolean SCC value into the otherwise
/// aligned low bit of a route key. `encoded_sgpr` names that key; the target
/// operation owns the generation-specific bit-test opcode. Some routers need
/// to keep using the key after restoring SCC and therefore normalize the key
/// itself to zero or one; routers that have already consumed the key can omit
/// that write. The request states this semantic lifetime choice without
/// exposing either native instruction to routing policy.
struct MoiEncodedSccRestoreRequest {
  /// Scalar register whose bit zero contains the SCC value to restore.
  uint16_t encoded_sgpr = 0;

  /// Whether to replace the encoded key with its normalized boolean value
  /// after restoring SCC. False preserves the key and emits only the bit test.
  bool normalize_encoded_sgpr = true;

  bool operator==(const MoiEncodedSccRestoreRequest &) const = default;
};

/// Append the target sequence that restores SCC from an encoded route key.
///
/// Only target profiles with the qualified gfx9 CDNA bit-test forms are
/// admitted. When requested, normalization follows the bit test. Unsupported
/// targets or invalid register assignments return false without changing
/// `words`.
[[nodiscard]] bool append_restore_moi_scc_from_route_key(std::vector<uint32_t> &words,
                                                         const MoiEncodedSccRestoreRequest &request,
                                                         const ConSanTargetProfile &target);

/// Append the target's device-scope cache refresh before retrying a contended
/// global publication.
///
/// Qualified gfx9 CDNA targets require an explicit buffer invalidate sequence;
/// targets whose coherent atomic-load path needs no extra instruction succeed
/// without appending words. Encoding failure returns false without partial
/// output.
[[nodiscard]] bool append_moi_device_cache_refresh(std::vector<uint32_t> &words,
                                                   const ConSanTargetProfile &target);

/// Append the waits required after a returning device-scope global atomic.
///
/// Every target waits for the returned load value. Targets with separately
/// tracked global-store completion also wait for the memory-side effect; gfx9
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
/// the operand-rich guest instruction needed by native lowering. A target
/// emitter may consume the copied `site` operands, but it must not rediscover
/// release/acquire meaning from instruction bits or choose a different
/// evidence intent. Language-level atomic load/store sequences use the same
/// contract with `is_rmw == false`; their complete ordered suffix is retained
/// in `ordered_sequence_end_text_offset`.
struct MoiAtomicEvidenceSitePlan {
  /// Authoritative original-program event selected by evidence policy.
  SemanticSiteId semantic_site;

  /// Stable identity of the normalized sequence that establishes ordering.
  ConSanSynchronizationAssociationId association;

  /// Before-guest intent that preserves the effective communication address.
  ConSanProbeIntentId address_capture_intent;

  /// Engine-specific after-guest evidence intent implemented by this plan.
  ConSanProbeIntentId evidence_intent;

  /// Diagnostic container spelling, including `kernel:` or `function:`.
  std::string container_name;

  /// Decoded operands and encoding fields required to relocate the guest and
  /// materialize its address. Semantic ordering does not come from this copy.
  ConSanAtomicSite site;

  /// Unique dispatchable owner descriptor, when graph ownership proved one.
  std::optional<uint64_t> kernel_descriptor_file_offset;

  /// Normalized release/acquire role selected from the shared sequence.
  ConSanMoiAtomicEventKind event_kind = ConSanMoiAtomicEventKind::Release;

  /// Whether kernel identity includes the cluster workgroup coordinate.
  bool uses_cluster_workgroup_id = false;

  /// True for a native RMW/CAS and false for an ordered ordinary load/store.
  bool is_rmw = true;

  /// End of the complete ordered guest sequence in original text coordinates.
  uint64_t ordered_sequence_end_text_offset = 0;

  /// Scalar scheduling hint that must be neutralized before inserting control
  /// flow around the guest operation.
  std::optional<uint64_t> scalar_clause_text_offset;

  /// Verify the cross-stage identities and basic guest-range invariants.
  [[nodiscard]] bool is_well_formed() const {
    return semantic_site.valid() && association.valid() && address_capture_intent.valid() &&
           evidence_intent.valid() && address_capture_intent != evidence_intent &&
           !container_name.empty() && site.size != 0u && site.width_bits != 0u &&
           ordered_sequence_end_text_offset >= site.text_offset + site.size;
  }
};

/// Immutable handoff from Record/Replay fence policy to resource planning and
/// native fence emission.
///
/// A fence record represents one graph-qualified communication sequence, not
/// every cache instruction that happens to resemble a fence. This value joins
/// the admitted `FenceRecord` intent to its normalized fence association and
/// to the operand-rich communication instruction whose effective address must
/// be reported. The patch range may cover only the fence or, for an acquire
/// sequence, the complete address-bearing load-through-fence interval. Native
/// lowering consumes these already-selected facts and must not rescan the
/// synchronization graph or reinterpret cache ordering.
struct MoiFenceEvidenceSitePlan {
  /// Stable identity of the admitted fence event in the original code object.
  SemanticSiteId semantic_site;

  /// Stable graph identity of the complete qualified communication sequence.
  ConSanSynchronizationAssociationId association;

  /// After-guest `FenceRecord` intent implemented by this lowering plan.
  ConSanProbeIntentId evidence_intent;

  /// Diagnostic container spelling, including `kernel:` or `function:`.
  std::string container_name;

  /// Operand-rich address-bearing atomic or ordinary-memory instruction.
  /// Ordering meaning remains owned by `association`, not this decode copy.
  ConSanAtomicSite communication_site;

  /// Release or acquire role already established by the graph association.
  ConSanSyncMemoryRole memory_role = ConSanSyncMemoryRole::Unknown;

  /// Unique dispatchable owner descriptor when graph ownership proved one.
  std::optional<uint64_t> kernel_descriptor_file_offset;

  /// Original text entry of the containing kernel or local function.
  uint64_t container_entry_text_offset = 0;

  /// File offset corresponding to the container's original text section.
  uint64_t text_file_offset = 0;

  /// Beginning of the exact original byte range replaced by the probe.
  uint64_t patch_text_offset = 0;

  /// Code-object file offset corresponding to `patch_text_offset`.
  uint64_t patch_file_offset = 0;

  /// Number of original bytes displaced by fence lowering.
  uint32_t patch_size = 0;

  /// Whether the communication address must be copied before guest execution.
  bool capture_address_before_guest = false;

  /// Scalar scheduling hint neutralized before inserting control flow.
  std::optional<uint64_t> scalar_clause_text_offset;

  /// Verify that policy, graph association, decode, and replacement range all
  /// name one complete lowering operation.
  [[nodiscard]] bool is_well_formed() const {
    return semantic_site.valid() &&
           semantic_site.domain == ConSanSemanticSiteDomain::SynchronizationEvent &&
           association.valid() && evidence_intent.valid() && !container_name.empty() &&
           communication_site.size != 0u && communication_site.width_bits != 0u &&
           (memory_role == ConSanSyncMemoryRole::Release ||
            memory_role == ConSanSyncMemoryRole::Acquire) &&
           patch_size != 0u &&
           patch_text_offset <= std::numeric_limits<uint64_t>::max() - patch_size;
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
  /// Stable identity of the barrier instruction where evidence is inserted.
  SemanticSiteId semantic_site;

  /// Stable identity of the normalized barrier sequence completed here.
  ConSanSynchronizationAssociationId association;

  /// Engine-specific barrier evidence intent implemented by this plan.
  ConSanProbeIntentId evidence_intent;

  /// Diagnostic container spelling, including `kernel:` or `function:`.
  std::string container_name;

  /// Whether the containing symbol is a dispatchable kernel.
  bool in_kernel = true;

  /// Decoded completing barrier instruction used only for native lowering.
  ConSanBarrierSite site;

  /// Unique dispatchable owner descriptor when graph ownership proved one.
  std::optional<uint64_t> kernel_descriptor_file_offset;

  /// Original text entry of the containing kernel or local function.
  uint64_t container_entry_text_offset = 0;

  /// Original byte extent of the containing symbol when statically known.
  uint64_t container_code_size = 0;

  /// File offset corresponding to the container's original text section.
  uint64_t text_file_offset = 0;

  /// Verify that the policy intent, graph event, and decoded insertion site
  /// form one complete barrier lowering operation.
  [[nodiscard]] bool is_well_formed() const {
    return semantic_site.valid() &&
           semantic_site.domain == ConSanSemanticSiteDomain::SynchronizationEvent &&
           association.valid() && evidence_intent.valid() && !container_name.empty() &&
           site.size != 0u && site.text_offset == semantic_site.physical.original_text_offset;
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

/// Canonical set of occupied half-open ranges in pristine executable text.
///
/// Dense relay placement must reject original instructions already owned by
/// an access or synchronization site. Callers may supply overlapping,
/// adjacent, empty, and unsorted ranges; construction removes empty ranges and
/// coalesces the rest. Keeping normalization and overlap queries in this type
/// prevents each instrumentation engine from implementing subtly different
/// interval arithmetic around the same original code.
class MoiOccupiedTextRanges {
public:
  /// Build a canonical occupied-range set from half-open `[begin, end)` pairs.
  explicit MoiOccupiedTextRanges(std::vector<std::pair<uint64_t, uint64_t>> ranges) {
    std::erase_if(ranges, [](const auto &range) { return range.first >= range.second; });
    std::ranges::sort(ranges);
    for (const auto &range : ranges) {
      if (!ranges_.empty() && ranges_.back().second >= range.first) {
        ranges_.back().second = std::max(ranges_.back().second, range.second);
      } else {
        ranges_.push_back(range);
      }
    }
  }

  /// Return whether `[begin, end)` intersects any nonempty occupied range.
  /// Empty or reversed queries do not overlap.
  [[nodiscard]] bool overlaps(uint64_t begin, uint64_t end) const {
    if (begin >= end)
      return false;
    const auto after =
        std::ranges::lower_bound(ranges_, end, {}, [](const auto &range) { return range.first; });
    return after != ranges_.begin() && std::prev(after)->second > begin;
  }

  /// Return the sorted, disjoint ranges retained by this set. This observation
  /// API exists for precise unit tests and diagnostics; placement needs only
  /// `overlaps`.
  [[nodiscard]] std::span<const std::pair<uint64_t, uint64_t>> ranges() const { return ranges_; }

private:
  /// Sorted, nonempty, pairwise-disjoint half-open ranges.
  std::vector<std::pair<uint64_t, uint64_t>> ranges_;
};

/// Stable identity of one kernel or helper function during dense access-route
/// planning.
///
/// The kind is retained explicitly because a kernel and helper may share a
/// symbol name without sharing an execution owner or legal relocation range.
/// Descriptor identity is deliberately absent: one helper group may execute
/// for several descriptors while still needing one physical entry relay.
struct MoiDenseRouteOwner {
  ConSanProgramContainerKind kind = ConSanProgramContainerKind::Count;
  std::string name;

  auto operator<=>(const MoiDenseRouteOwner &) const = default;
};

/// Minimal target-independent site presented to the common dense-route
/// partitioner.
///
/// `source_index` remains in the caller's domain, allowing access, barrier,
/// and atomic planners to share grouping without sharing their semantic
/// candidate types.
struct MoiDenseRouteSite {
  MoiDenseRouteOwner owner;
  uint64_t anchor = 0;
  size_t source_index = 0;
};

/// One owner-local, branch-reachable set of caller indices that may share a
/// dense dispatcher.
///
/// Indices are ordered by their corresponding pristine-text anchor.
/// `first_anchor` makes the common branch-span invariant explicit without
/// requiring callers to reconstruct it from engine-private candidate values.
struct MoiDenseRouteIndexGroup {
  MoiDenseRouteOwner owner;
  uint64_t first_anchor = 0;
  std::vector<size_t> source_indices;
};

/// Partition heterogeneous lowering sites by typed owner, dispatcher
/// capacity, and the maximum span reachable from one SOPP relay.
///
/// A zero capacity means unlimited sites per group. Every returned group is
/// nonempty; input order does not affect the result except as a deterministic
/// tie-break for sites with the same owner and anchor.
[[nodiscard]] inline std::vector<MoiDenseRouteIndexGroup>
partition_moi_dense_route_sites(std::span<const MoiDenseRouteSite> sites,
                                size_t max_sites_per_group) {
  std::vector<MoiDenseRouteSite> ordered(sites.begin(), sites.end());
  std::ranges::sort(ordered, [](const MoiDenseRouteSite &lhs, const MoiDenseRouteSite &rhs) {
    return std::tie(lhs.owner, lhs.anchor, lhs.source_index) <
           std::tie(rhs.owner, rhs.anchor, rhs.source_index);
  });
  constexpr uint64_t kMaxCommonRelayAnchorSpan =
      static_cast<uint64_t>(static_cast<int64_t>(std::numeric_limits<int16_t>::max()) -
                            static_cast<int64_t>(std::numeric_limits<int16_t>::min())) *
      sizeof(uint32_t);
  std::vector<MoiDenseRouteIndexGroup> groups;
  for (const MoiDenseRouteSite &site : ordered) {
    const bool capacity_reached = max_sites_per_group != 0u && !groups.empty() &&
                                  groups.back().source_indices.size() >= max_sites_per_group;
    if (groups.empty() || groups.back().owner != site.owner || capacity_reached ||
        site.anchor - groups.back().first_anchor > kMaxCommonRelayAnchorSpan) {
      groups.push_back({.owner = site.owner, .first_anchor = site.anchor, .source_indices = {}});
    }
    groups.back().source_indices.push_back(site.source_index);
  }
  return groups;
}

/// One bounded, branch-reachable group of access candidates that may share a
/// dense entry relay and dispatcher.
///
/// All candidates belong to `owner`, are ordered by original-text anchor, and
/// fit within the common SOPP branch span. `first_candidate_index` refers to
/// the caller's unmodified candidate sequence and therefore identifies the
/// appended relay-bank slot owned by this group.
struct MoiDenseCandidateGroup {
  MoiDenseRouteOwner owner;
  std::vector<const ConSanMoiCandidate *> candidates;
  size_t first_candidate_index = 0;

  /// Return the original-text anchors in their canonical group order.
  [[nodiscard]] std::vector<uint64_t> anchors() const {
    std::vector<uint64_t> result;
    result.reserve(candidates.size());
    for (const ConSanMoiCandidate *candidate : candidates)
      result.push_back(candidate->anchor());
    return result;
  }

  /// Return whether adjacent sites are too close to own independent entry
  /// regions of `entry_bytes` each.
  [[nodiscard]] bool has_overlapping_entries(uint64_t entry_bytes) const {
    for (size_t index = 1; index < candidates.size(); ++index) {
      if (candidates[index]->anchor() < candidates[index - 1u]->anchor() + entry_bytes)
        return true;
    }
    return false;
  }

  /// Return whether every guest instruction provides at least `minimum_bytes`
  /// of explicit anchor storage for a keyed dense route.
  [[nodiscard]] bool every_site_has_size(uint32_t minimum_bytes) const {
    return std::ranges::all_of(candidates, [&](const ConSanMoiCandidate *candidate) {
      return candidate->size() >= minimum_bytes;
    });
  }
};

/// Shared, immutable partition of access candidates used by dense route
/// planners in Record/Replay, Sampled, and InlineShadow.
///
/// `candidates_by_owner` protects every access anchor in an owner even when
/// that owner is split across several relay groups. `index_by_candidate`
/// preserves the caller's relay-bank assignment. `groups` is the only place
/// where owner boundaries, anchor ordering, maximum group size, and common
/// branch reach are combined.
struct MoiDenseCandidatePartition {
  std::map<MoiDenseRouteOwner, std::vector<const ConSanMoiCandidate *>> candidates_by_owner;
  std::unordered_map<const ConSanMoiCandidate *, size_t> index_by_candidate;
  std::vector<MoiDenseCandidateGroup> groups;
};

/// Partition candidates for shared dense entry routing.
///
/// A zero `max_candidates_per_group` means that capacity does not split an
/// owner; SOPP reach still does. Nonzero limits encode engine/target dispatcher
/// capacity and are policy supplied by the caller rather than hidden in the
/// shared mechanism.
[[nodiscard]] inline MoiDenseCandidatePartition
partition_moi_dense_candidates(std::span<const ConSanMoiCandidate *const> candidates,
                               size_t max_candidates_per_group) {
  MoiDenseCandidatePartition partition;
  partition.index_by_candidate.reserve(candidates.size());
  std::vector<MoiDenseRouteSite> sites;
  sites.reserve(candidates.size());
  for (size_t index = 0; index < candidates.size(); ++index) {
    const ConSanMoiCandidate *candidate = candidates[index];
    partition.index_by_candidate.emplace(candidate, index);
    const MoiDenseRouteOwner owner{candidate->container.kind, candidate->container.name};
    partition.candidates_by_owner[owner].push_back(candidate);
    sites.push_back({.owner = owner, .anchor = candidate->anchor(), .source_index = index});
  }

  for (auto &[owner, owner_candidates] : partition.candidates_by_owner) {
    (void)owner;
    std::ranges::sort(owner_candidates, {},
                      [](const ConSanMoiCandidate *candidate) { return candidate->anchor(); });
  }
  for (const MoiDenseRouteIndexGroup &group :
       partition_moi_dense_route_sites(sites, max_candidates_per_group)) {
    MoiDenseCandidateGroup candidates_group{
        .owner = group.owner,
        .candidates = {},
        .first_candidate_index = group.source_indices.front(),
    };
    candidates_group.candidates.reserve(group.source_indices.size());
    for (size_t index : group.source_indices)
      candidates_group.candidates.push_back(candidates[index]);
    partition.groups.push_back(std::move(candidates_group));
  }
  return partition;
}

/// Return whether one emitted MOI patch makes its owning kernel consume all
/// three launch workgroup coordinates. This shared descriptor-mutation and
/// validation predicate distinguishes MOI patches from independently composed
/// mutation and malformed-barrier patches. CDNA consumes the tuple only
/// through an entry capture with complete persistent storage; RDNA and CDNA5
/// observation bodies consume the firmware payload directly.
[[nodiscard]] inline bool patch_requires_full_workgroup_id_payload(ConSanCapabilityEngine engine,
                                                                   rj_code_arch_t arch,
                                                                   const ConSanPatchInfo &patch) {
  if (!consan_is_capability_arch(arch) || patch.owner_descriptor_file_offsets.empty() ||
      engine == ConSanCapabilityEngine::SuperCollider || engine == ConSanCapabilityEngine::Count) {
    return false;
  }
  if (consan_uses_gfx9_cdna_encoding(arch)) {
    const bool entry_capture = patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue ||
                               patch.kind == ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
    return entry_capture && (patch.persistent_sgpr_state.record_replay_workgroup.complete() ||
                             patch.persistent_record_replay_workgroup_vgprs.complete() ||
                             patch.persistent_record_replay_workgroup_private_offsets.complete());
  }
  return (patch.kind >= ConSanPatchKind::InlineMoiAccessRecordStore &&
          patch.kind <= ConSanPatchKind::TrampolineMoiFenceRecord) ||
         patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
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
[[nodiscard]] bool validate_scalar_state_temporaries(const MoiOptions &options,
                                                     std::string_view consumer,
                                                     std::vector<std::string> &errors);

/// Materialize one persistent workgroup-coordinate source, including any
/// ABI-specific extraction applied after a scalar, vector, or private load.
[[nodiscard]] bool append_workgroup_source_value(std::vector<uint32_t> &words,
                                                 const ConSanMoiWorkgroupSource &source,
                                                 uint16_t value_vgpr, rj_code_arch_t arch);

/// Semantic request to materialize one dynamically indexed report field.
///
/// `field_address` is the absolute address of field zero, `stride_bytes` is
/// the report-layout distance between adjacent records, and `slot_vgpr` holds
/// the runtime record index. `address_vgpr` names the consecutive output pair.
/// Keeping these roles in one request prevents record kinds from reordering a
/// positional address/slot pair or supplying an architecture as semantic
/// input. The report-layout validator guarantees that `slot * stride_bytes`
/// fits in 32 bits.
struct MoiDynamicRecordAddressRequest {
  uint64_t field_address = 0;
  uint32_t stride_bytes = 0;
  uint16_t address_vgpr = 0;
  uint16_t slot_vgpr = 0;

  bool operator==(const MoiDynamicRecordAddressRequest &) const = default;
};

/// Append the target sequence for `field_address + slot * stride_bytes`.
///
/// The address occupies `address_vgpr:address_vgpr+1`; the slot must be
/// distinct from that pair. The operation uses only the address pair as
/// temporary storage, preserves the slot, and clobbers VCC. Invalid requests
/// or target encodings fail transactionally without partial output.
[[nodiscard]] bool append_dynamic_record_address(std::vector<uint32_t> &words,
                                                 const MoiDynamicRecordAddressRequest &request,
                                                 const ConSanTargetProfile &target);

/// Return the nearest emitted trampoline body strictly after `offset` across
/// both already committed and current-pass patch inventories. Empty bodies do
/// not reserve bytes. Incremental lowering must use both inventories when it
/// grows a shared dispatcher, or it can overwrite a body emitted earlier in
/// the current pass.
[[nodiscard]] std::optional<uint64_t>
next_moi_trampoline_boundary(uint64_t offset, std::span<const ConSanPatchInfo> committed,
                             std::span<const ConSanPatchInfo> current_pass);

} // namespace consan_detail
} // namespace rocjitsu
