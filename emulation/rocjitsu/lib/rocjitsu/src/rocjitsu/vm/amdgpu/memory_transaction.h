// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// Maximum supported transaction buffer size, independent of cache geometry.
inline constexpr uint32_t kMaxMemoryTransactionBytes = 128;

/// @brief Observation boundary for contiguous, non-overlapping memory operands.
///
/// @details Cache lines, fabric beats, and indivisible protocol records are distinct.
/// RCCL selects 128-byte LL records for gfx1250 and 64-byte records otherwise:
/// https://github.com/ROCm/rocm-systems/blob/9c35798820670b1a2301730c9b4ee90200731160/projects/rccl/src/include/archinfo.h#L40-L56
/// This supports the gfx1250 protocol model, not arbitrary snapshots under
/// concurrent buffer reuse or a general hardware guarantee. CDNA3/4 fabric
/// documentation describes 32/64-byte requests, but does not fully specify
/// atomic observation. Retain the prior model on unqualified targets until
/// the relevant hardware access paths are tested. In particular, do not infer
/// gfx1251 or RDNA behavior from gfx1250's protocol.
///
/// The legacy mapped backend enforces this boundary through shared storage
/// locks. Reordered translated requests use the same boundary, while default
/// lane-ordered requests retain backend batching. External backends remain
/// responsible for their snapshot, publication, and translation-span contracts.
class MemoryTransactionPolicy {
public:
  constexpr MemoryTransactionPolicy() = default;

  /// @brief Exercise a narrower boundary without selecting it for a production target.
  /// Hardware qualification is required before enabling this policy by default.
  static constexpr MemoryTransactionPolicy unqualified_64() { return MemoryTransactionPolicy(64); }

  constexpr uint32_t bytes() const { return bytes_; }
  constexpr size_t chunk_size(uint64_t address, size_t size) const {
    return std::min(size, size_t(bytes() - (address & (bytes() - 1))));
  }

private:
  explicit constexpr MemoryTransactionPolicy(uint32_t bytes) : bytes_(bytes) {
    assert(bytes != 0 && (bytes & (bytes - 1)) == 0 && bytes <= kMaxMemoryTransactionBytes);
  }
  uint32_t bytes_ = kMaxMemoryTransactionBytes;
};

} // namespace rocjitsu::amdgpu
