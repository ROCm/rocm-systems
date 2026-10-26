// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/wavefront.h"

namespace rocjitsu::amdgpu {

/// @brief Bytes of one lane's private memory stored contiguously in scratch.
/// @details The ISA's SWIZZLE(offset, ThreadID) interleaves private memory by dword across the
/// wave. rocm-dbgapi reads private memory with the same 4-byte interleave (rocdbgapi memory.cpp
/// private_swizzled), so ROCgdb depends on execution and wait planning agreeing on it.
inline constexpr uint32_t kScratchInterleaveBytes = sizeof(uint32_t);

/// @brief Byte distance between consecutive interleave units of one lane.
constexpr uint32_t scratch_unit_stride(uint32_t wave_size) {
  return wave_size * kScratchInterleaveBytes;
}

/// @brief Map a lane's private byte offset to its interleaved scratch address.
/// @details Callers decode operands and apply each architecture's signedness first.
constexpr uint64_t scratch_lane_address(uint64_t scratch_base, uint64_t private_offset,
                                        uint32_t lane, uint32_t wave_size) {
  return scratch_base + private_offset / kScratchInterleaveBytes * scratch_unit_stride(wave_size) +
         uint64_t{lane} * kScratchInterleaveBytes + private_offset % kScratchInterleaveBytes;
}

enum class FlatPrivateLayout { None, Interleaved, Linear, EncodedLane };

struct FlatAddress {
  uint64_t value;
  bool private_address = false;
  bool swizzled = false;
};

/// Pure address translation shared by execution and pre-execution wait planning.
inline FlatAddress translate_flat_address(const Wavefront &wf, uint64_t address, unsigned lane,
                                          FlatPrivateLayout layout) {
  if (layout == FlatPrivateLayout::None)
    return {address};
  uint64_t offset;
  if (layout == FlatPrivateLayout::EncodedLane) {
    if (!wf.scratch_lane_size())
      return {address};
    const unsigned shift = wf.wf_size() == 64 ? 51 : 52;
    const uint64_t mask = uint64_t{wf.wf_size() - 1} << shift;
    const uint64_t base = wf.scratch_base() & ~mask;
    const uint64_t without_lane = address & ~mask;
    if (without_lane < base || without_lane - base > UINT32_MAX)
      return {address};
    offset = without_lane - base;
    lane = (address & mask) >> shift;
  } else {
    const uint32_t private_hi = wf.private_aperture_base() >> 32;
    if (!private_hi || static_cast<uint32_t>(address >> 32) != private_hi)
      return {address};
    offset = static_cast<uint32_t>(address);
  }
  if (layout == FlatPrivateLayout::Linear)
    return {wf.scratch_base() + uint64_t{lane} * wf.scratch_lane_size() + offset, true};
  return {scratch_lane_address(wf.scratch_base(), offset, lane, wf.wf_size()), true, true};
}

inline uint64_t flat_vector_offset(uint64_t value, bool signed_offset, unsigned scale = 1) {
  return signed_offset ? static_cast<uint64_t>(int64_t{static_cast<int32_t>(value)} * scale)
                       : value;
}

} // namespace rocjitsu::amdgpu
