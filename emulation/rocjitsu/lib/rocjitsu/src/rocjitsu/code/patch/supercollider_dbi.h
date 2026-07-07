// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file supercollider_dbi.h
/// @brief Entry point for SuperCollider-style DBI code-object patching.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

enum class SuperColliderDbiDelayMode : uint8_t {
  Nop,
  Sleep,
  SleepVar,
};

struct SuperColliderDbiOptions {
  bool enabled = false;
  bool fail_closed = false;
  bool probe_nop = false;
  bool probe_trampoline_nop = false;
  bool probe_endpgm = false;
  bool probe_lds_endpgm = false;
  bool probe_lds_check_trap = false;
  bool probe_flat_check_trap = false;
  bool probe_flat_trap = false;
  bool fault_drop_barrier = false;
  uint32_t fault_barrier_index = 0;
  SuperColliderDbiDelayMode delay_mode = SuperColliderDbiDelayMode::Nop;
  uint16_t delay_var_ssrc = 106;
  std::optional<uint16_t> scratch_vgpr;
  uint32_t delay_nops = 0;
  uint32_t max_patches = 1;
};

struct SuperColliderDbiTextSection {
  std::string name;
  uint64_t file_offset = 0;
  uint64_t virtual_address = 0;
  uint64_t size = 0;
};

struct SuperColliderDbiKernelStats {
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

enum class SuperColliderDbiLdsAccessKind : uint8_t {
  Read,
  Write,
  Atomic,
  Other,
};

enum class SuperColliderDbiFlatAddressSpaceHint : uint8_t {
  Unknown,
  Group,
  Private,
  MaybeGroup,
  MaybePrivate,
  Global,
};

enum class SuperColliderDbiPreflightAction : uint8_t {
  NotRun,
  Candidate,
  Skip,
  Reject,
};

enum class SuperColliderDbiPatchKind : uint8_t {
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
  TrampolineNop,
};

struct SuperColliderDbiLdsSite {
  SuperColliderDbiLdsAccessKind kind = SuperColliderDbiLdsAccessKind::Other;
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

struct SuperColliderDbiFlatSite {
  SuperColliderDbiLdsAccessKind kind = SuperColliderDbiLdsAccessKind::Other;
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
  SuperColliderDbiFlatAddressSpaceHint address_space_hint =
      SuperColliderDbiFlatAddressSpaceHint::Unknown;
  std::string mnemonic;
};

struct SuperColliderDbiKernelInfo {
  std::string name;
  uint64_t descriptor_file_offset = 0;
  uint64_t entry_text_offset = 0;
  uint64_t text_file_offset = 0;
  uint64_t code_size = 0;
  bool has_text_range = false;
  bool decoded = false;
  SuperColliderDbiKernelStats stats;
  std::vector<SuperColliderDbiLdsSite> lds_sites;
  std::vector<SuperColliderDbiFlatSite> flat_sites;
  SuperColliderDbiPreflightAction preflight_action = SuperColliderDbiPreflightAction::NotRun;
  std::vector<std::string> preflight_reasons;
};

struct SuperColliderDbiFunctionInfo {
  std::string name;
  uint64_t entry_text_offset = 0;
  uint64_t text_file_offset = 0;
  uint64_t code_size = 0;
  bool decoded = false;
  SuperColliderDbiKernelStats stats;
  std::vector<SuperColliderDbiLdsSite> lds_sites;
  std::vector<SuperColliderDbiFlatSite> flat_sites;
};

struct SuperColliderDbiPatchInfo {
  SuperColliderDbiPatchKind kind = SuperColliderDbiPatchKind::InlineNopRewrite;
  uint64_t anchor_offset = 0;
  uint64_t trampoline_offset = 0;
  uint32_t original_size = 0;
  std::optional<uint16_t> scratch_vgpr;
};

struct SuperColliderDbiResult {
  bool visited_code_object = false;
  bool modified = false;
  size_t input_size = 0;
  std::string target_name;
  std::string arch_name;
  std::vector<SuperColliderDbiTextSection> text_sections;
  std::vector<SuperColliderDbiKernelInfo> kernels;
  std::vector<SuperColliderDbiFunctionInfo> functions;
  std::vector<SuperColliderDbiPatchInfo> patches;
  std::vector<uint8_t> elf_bytes;
  std::vector<std::string> errors;
  std::vector<std::string> warnings;
};

/// @brief Try to apply DBI SuperCollider instrumentation to @p code_object_bytes.
///
/// @details The early DBI slices parse and inventory the code object, return
/// `modified=false`, and leave `elf_bytes` empty so callers continue loading the
/// original code object.
[[nodiscard]] SuperColliderDbiResult
try_patch_supercollider_dbi(std::span<const uint8_t> code_object_bytes,
                            const SuperColliderDbiOptions &options);

} // namespace rocjitsu
