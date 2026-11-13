// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file translated_local_cache_policy.h
/// @brief Select cacheable local lines from translated VM accesses.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "simdojo/components/cache.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu {

/// A complete local cache line selected by a live page-table walk. Keeping the
/// physical line and all PTE policy bits in the tag lets a later hit reject a
/// changed mapping even when the broad cache class has not changed.
struct TranslatedLocalLine {
  uint64_t backing_address = 0;
  uint64_t mapping_attributes = 0;
  Mtype effective_type = Mtype::UC;
};

inline uint64_t translated_mapping_attributes(const VmTranslation &translation) {
  const VmPermissions rights = translation.permissions;
  return static_cast<uint64_t>(translation.mtype) |
         (translation.gfx12_pte_mtype
              ? (static_cast<uint64_t>(*translation.gfx12_pte_mtype) + 1) << 8
              : 0) |
         (static_cast<uint64_t>(rights.readable) << 16) |
         (static_cast<uint64_t>(rights.writable) << 17) |
         (static_cast<uint64_t>(rights.executable) << 18);
}

inline std::optional<TranslatedLocalLine> query_translated_local_line(const GpuVmAccess &access,
                                                                      uint64_t line_address,
                                                                      std::size_t line_size,
                                                                      Mtype instruction_type) {
  const VmTranslationResult result =
      access.query_translation(line_address, line_size, VmAccessKind::Read);
  if (!result || result.translation.domain != VmMemoryDomain::Local ||
      result.translation.contiguous_bytes < line_size || !result.translation.permissions.readable)
    return std::nullopt;
  const Mtype effective = effective_mtype(instruction_type, result.translation.mtype);
  if (effective != Mtype::RW && effective != Mtype::WB && effective != Mtype::NT)
    return std::nullopt;
  return TranslatedLocalLine{.backing_address = result.translation.address,
                             .mapping_attributes =
                                 translated_mapping_attributes(result.translation),
                             .effective_type = effective};
}

inline bool translated_local_tag_matches(const simdojo::CacheTag &tag,
                                         const TranslatedLocalLine &line, uint64_t cache_identity,
                                         uint64_t backing_epoch, uint64_t line_epoch) {
  return tag.address_space_identity == cache_identity && tag.cache_identity == cache_identity &&
         tag.backing_address == line.backing_address &&
         tag.mapping_attributes == line.mapping_attributes &&
         tag.local_backing_epoch == backing_epoch && tag.local_line_epoch == line_epoch &&
         !tag.dirty;
}

/// Completion owns both numeric-VMID lines and translated-local lines keyed
/// by its stable address-space identity. The latter use VMID zero in the
/// generic cache so a PASID cannot be mistaken for a hardware VMID.
inline bool cache_tag_matches_completion_access(const simdojo::CacheTag &tag, uint32_t vmid,
                                                uint64_t cache_identity) {
  if (tag.cache_identity != cache_identity)
    return false;
  return (tag.vmid == vmid && tag.address_space_identity == 0) ||
         (tag.vmid == 0 && tag.address_space_identity == cache_identity);
}

} // namespace rocjitsu::amdgpu
