// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan.h
/// @brief Entry point for ConSan DBI code-object patching.

#pragma once

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
  uint32_t fault_barrier_index = 0;
  ConSanDelayMode delay_mode = ConSanDelayMode::Nop;
  uint16_t delay_var_ssrc = 106;
  std::optional<uint16_t> scratch_vgpr;
  std::optional<uint16_t> moi_exec_save_sgpr;
  std::optional<uint16_t> moi_owner_sgpr;
  std::optional<uint16_t> moi_owner_vgpr;
  std::optional<uint16_t> moi_epoch_vgpr;
  std::optional<uint64_t> report_buffer_address;
  std::optional<uint64_t> moi_report_buffer_address;
  uint64_t moi_report_buffer_size = 0;
  uint32_t delay_nops = 0;
  uint32_t max_patches = 1;
  uint32_t moi_sample_stride = 1;
  uint32_t moi_sample_offset = 0;
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
  std::optional<uint32_t> raw_saddr;
  std::optional<uint32_t> raw_vaddr;
  std::optional<uint32_t> raw_vsrc;
  std::optional<uint32_t> raw_vdst;
  std::optional<int32_t> raw_ioffset;
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
  std::optional<uint64_t> kernel_descriptor_file_offset;
  std::optional<uint32_t> raw_saddr;
  std::optional<uint32_t> raw_vaddr;
  std::optional<uint32_t> raw_vsrc;
  std::optional<uint32_t> raw_vdst;
  std::optional<int32_t> raw_ioffset;
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
};

struct ConSanCandidateResourcePlan {
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
  uint32_t original_private_segment_size = 0;
};

struct ConSanResult {
  bool visited_code_object = false;
  bool modified = false;
  ConSanFlavor flavor = ConSanFlavor::None;
  ConSanMoiEngine moi_engine = ConSanMoiEngine::RecordReplay;
  size_t input_size = 0;
  std::string target_name;
  std::string arch_name;
  std::vector<ConSanTextSection> text_sections;
  std::vector<ConSanKernelInfo> kernels;
  std::vector<ConSanFunctionInfo> functions;
  std::vector<ConSanMoiCandidate> moi_candidates;
  std::vector<ConSanCandidateResourcePlan> resource_plans;
  std::vector<ConSanPatchInfo> patches;
  std::vector<uint8_t> elf_bytes;
  std::vector<std::string> errors;
  std::vector<std::string> warnings;
};

[[nodiscard]] const char *consan_flavor_name(ConSanFlavor flavor);
[[nodiscard]] const char *consan_moi_engine_name(ConSanMoiEngine engine);

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
