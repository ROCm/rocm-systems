// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan.h
/// @brief Entry point for ConSan DBI code-object patching.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

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

/// Distinguishes decode/inventory knowledge from a native emitter that is
/// safe for live instrumentation on the selected architecture.
enum class ConSanNativeSupport : uint8_t {
  Unavailable,
  InventoryOnly,
  NativeEmission,
};

enum class ConSanNativeFeature : uint8_t {
  LdsAccess,
  GroupFlatAccess,
  Barrier,
  Atomic,
  WorkgroupIdentity,
  StableWaveOwner,
  HwIdOwner,
  ScratchSpill,
  NonScratchWait,
  ReportPublication,
  SuperCollider,
  RecordReplay,
  Sampled,
  InlineShadow,
};

struct ConSanTargetCapabilities {
  rj_code_arch_t architecture = ROCJITSU_CODE_ARCH_INVALID;
  ConSanNativeSupport lds_access = ConSanNativeSupport::Unavailable;
  ConSanNativeSupport group_flat_access = ConSanNativeSupport::Unavailable;
  ConSanNativeSupport barrier = ConSanNativeSupport::Unavailable;
  ConSanNativeSupport atomic = ConSanNativeSupport::Unavailable;
  ConSanNativeSupport workgroup_identity = ConSanNativeSupport::Unavailable;
  ConSanNativeSupport stable_wave_owner = ConSanNativeSupport::Unavailable;
  ConSanNativeSupport hw_id_owner = ConSanNativeSupport::Unavailable;
  ConSanNativeSupport scratch_spill = ConSanNativeSupport::Unavailable;
  ConSanNativeSupport non_scratch_wait = ConSanNativeSupport::Unavailable;
  ConSanNativeSupport report_publication = ConSanNativeSupport::Unavailable;
  ConSanNativeSupport supercollider = ConSanNativeSupport::Unavailable;
  ConSanNativeSupport record_replay = ConSanNativeSupport::Unavailable;
  ConSanNativeSupport sampled = ConSanNativeSupport::Unavailable;
  ConSanNativeSupport inline_shadow = ConSanNativeSupport::Unavailable;
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
  bool fault_drop_barrier = false;
  bool moi_init_owner_epoch = false;
  bool moi_track_barriers = false;
  bool moi_track_atomics = false;
  bool moi_dynamic_access_records = false;
  /// Check the immediately preceding direct-sampled slot in-kernel and
  /// increment the shared diagnostic counter on a sampled conflict.
  bool moi_sampled_check = false;
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
  /// Internal marker set when the persistent set also snapshots CDNA4
  /// workgroup identity at kernel entry.
  bool automatic_moi_identity_vgprs = false;
  /// Internal marker set when inline shadow derives owner at each probe and
  /// keeps epoch in per-lane private memory.
  bool automatic_moi_private_epoch = false;
  /// Internal marker set after ConSan assigns the scalar EXEC-save window.
  bool automatic_moi_exec_save_sgprs = false;
  /// Internal marker set after ConSan assigns the hw_id owner temporary SGPR.
  bool automatic_moi_owner_sgpr = false;
  uint32_t fault_barrier_index = 0;
  ConSanDelayMode delay_mode = ConSanDelayMode::Nop;
  uint16_t delay_var_ssrc = 106;
  std::optional<uint16_t> scratch_vgpr;
  std::optional<uint16_t> moi_exec_save_sgpr;
  std::optional<uint16_t> moi_owner_sgpr;
  std::optional<uint16_t> moi_owner_vgpr;
  std::optional<uint16_t> moi_epoch_vgpr;
  std::array<std::optional<uint16_t>, 3> moi_workgroup_vgprs;
  /// Internal two-SGPR window used to unpack AQL workgroup dimensions in the
  /// CDNA4 kernel-entry identity prologue.
  std::optional<uint16_t> moi_identity_sgpr;
  std::optional<uint64_t> report_buffer_address;
  std::optional<uint64_t> moi_report_buffer_address;
  uint64_t moi_report_buffer_size = 0;
  /// Generation stored in direct sampled entries. Auto report buffers set
  /// this from their versioned header; explicit buffers default to zero.
  uint32_t moi_report_generation = 0;
  uint32_t delay_nops = 0;
  uint32_t max_patches = 1;
  uint32_t moi_sample_stride = 1;
  uint32_t moi_sample_offset = 0;
  /// Runtime deterministic wave selector for direct sampled probes. The
  /// stride is a power of two and selects owner & (stride - 1) == offset.
  uint32_t moi_runtime_sample_stride = 1;
  uint32_t moi_runtime_sample_offset = 0;
  uint32_t report_marker = 1;
};

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
  TrampolineNop,
};

struct ConSanLdsSite {
  ConSanLdsAccessKind kind = ConSanLdsAccessKind::Other;
  bool supported_mvp = false;
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
  uint32_t width_bits = 0;
  std::optional<uint16_t> dst_vgpr;
  std::optional<uint16_t> addr_vgpr;
  std::optional<uint16_t> data_vgpr;
  std::optional<uint16_t> secondary_data_vgpr;
  std::optional<uint32_t> raw_offset0;
  std::optional<uint32_t> raw_offset1;
  std::optional<uint32_t> raw_op;
  std::optional<bool> raw_gds;
  std::string mnemonic;
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
  std::string mnemonic;
};

struct ConSanBarrierSite {
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
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
  std::optional<int32_t> raw_ioffset;
  std::optional<uint32_t> raw_segment;
  std::optional<uint32_t> raw_scope;
  std::optional<uint32_t> raw_th;
  std::optional<bool> returns_old_value;
  std::string mnemonic;
};

enum class ConSanMoiCandidateSource : uint8_t {
  NativeLds,
  FlatGroup,
  FlatMaybeGroup,
};

struct ConSanLdsStaticRange {
  uint32_t byte_offset = 0;
  uint32_t byte_count = 0;
};

enum class ConSanMoiLdsExclusionReason : uint8_t {
  GdsReserved,
  AtomicReserved,
  PermuteOrSwizzle,
  Transpose,
  UnsupportedAccessForm,
  OtherDs,
};

struct ConSanMoiLdsExclusion {
  std::string container_name;
  bool in_kernel = true;
  uint64_t text_offset = 0;
  ConSanMoiLdsExclusionReason reason = ConSanMoiLdsExclusionReason::OtherDs;
  std::string mnemonic;
};

struct ConSanMoiCandidate {
  std::string container_name;
  bool in_kernel = true;
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
  std::optional<uint16_t> secondary_data_vgpr;
  std::optional<uint32_t> raw_offset0;
  std::optional<uint32_t> raw_offset1;
  std::optional<uint32_t> raw_op;
  std::optional<bool> raw_gds;
  std::vector<ConSanLdsStaticRange> native_static_ranges;
  std::optional<uint64_t> kernel_descriptor_file_offset;
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
  bool has_text_range = false;
  std::optional<bool> uses_dynamic_stack;
  bool decoded = false;
  ConSanKernelStats stats;
  std::vector<ConSanLdsSite> lds_sites;
  std::vector<ConSanFlatSite> flat_sites;
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
  bool decoded = false;
  ConSanKernelStats stats;
  std::vector<ConSanLdsSite> lds_sites;
  std::vector<ConSanFlatSite> flat_sites;
  std::vector<ConSanBarrierSite> barrier_sites;
  std::vector<ConSanFenceSite> fence_sites;
  std::vector<ConSanAtomicSite> atomic_sites;
};

struct ConSanPatchInfo {
  ConSanPatchKind kind = ConSanPatchKind::InlineNopRewrite;
  uint64_t anchor_offset = 0;
  uint64_t trampoline_offset = 0;
  uint32_t original_size = 0;
  uint32_t trampoline_size = 0;
  std::optional<uint16_t> scratch_vgpr;
  std::optional<uint32_t> persistent_epoch_private_offset;
  uint16_t spilled_vgpr_count = 0;
  uint32_t required_private_segment_size = 0;
  /// Kernel descriptors whose dispatches can execute this patch. Shared
  /// helper patches name every reachable owner.
  std::vector<uint64_t> owner_descriptor_file_offsets;
};

enum class ConSanResourceSiteKind : uint8_t {
  Access,
  Barrier,
  Atomic,
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
  uint16_t required_vgpr_count = 0;
  uint16_t current_sgpr_count = 0;
  uint16_t max_referenced_sgpr_count = 0;
  uint32_t original_private_segment_size = 0;
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
};

struct ConSanResult {
  bool visited_code_object = false;
  bool modified = false;
  ConSanFlavor flavor = ConSanFlavor::None;
  ConSanMoiEngine moi_engine = ConSanMoiEngine::RecordReplay;
  std::optional<uint16_t> resolved_moi_owner_vgpr;
  std::optional<uint16_t> resolved_moi_epoch_vgpr;
  std::array<std::optional<uint16_t>, 3> resolved_moi_workgroup_vgprs;
  std::optional<uint16_t> resolved_moi_identity_sgpr;
  std::optional<uint16_t> resolved_moi_exec_save_sgpr;
  std::optional<uint16_t> resolved_moi_owner_sgpr;
  bool moi_persistent_vgprs_automatic = false;
  bool moi_private_epoch_automatic = false;
  bool moi_exec_save_sgprs_automatic = false;
  bool moi_owner_sgpr_automatic = false;
  size_t input_size = 0;
  std::string target_name;
  std::string arch_name;
  std::vector<ConSanTextSection> text_sections;
  std::vector<ConSanKernelInfo> kernels;
  std::vector<ConSanFunctionInfo> functions;
  std::vector<ConSanMoiCandidate> moi_candidates;
  std::vector<ConSanMoiLdsExclusion> moi_lds_exclusions;
  std::vector<ConSanCandidateResourcePlan> resource_plans;
  ConSanResourcePlanSummary resource_plan_summary;
  std::vector<ConSanPatchInfo> patches;
  std::vector<uint8_t> elf_bytes;
  std::vector<std::string> errors;
  std::vector<std::string> warnings;
};

[[nodiscard]] const char *consan_flavor_name(ConSanFlavor flavor);
[[nodiscard]] const char *consan_moi_engine_name(ConSanMoiEngine engine);
[[nodiscard]] const char *consan_native_feature_name(ConSanNativeFeature feature);
[[nodiscard]] const char *consan_native_support_name(ConSanNativeSupport support);
[[nodiscard]] ConSanTargetCapabilities consan_target_capabilities(rj_code_arch_t architecture);
[[nodiscard]] ConSanNativeSupport
consan_native_feature_support(const ConSanTargetCapabilities &capabilities,
                              ConSanNativeFeature feature);

[[nodiscard]] std::optional<ConSanFlavor> parse_consan_flavor(std::string_view value);
[[nodiscard]] std::optional<ConSanMoiEngine> parse_consan_moi_engine(std::string_view value);

/// @brief Try to apply ConSan instrumentation to @p code_object_bytes.
///
/// @details The early DBI slices parse and inventory the code object, return
/// `modified=false`, and leave `elf_bytes` empty so callers continue loading the
/// original code object.
[[nodiscard]] ConSanResult try_patch_consan(std::span<const uint8_t> code_object_bytes,
                                            const ConSanOptions &options);

} // namespace rocjitsu
