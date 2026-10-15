// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/wavefront.h"

namespace rocjitsu::amdgpu {

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
  return {wf.scratch_base() + (offset / 4) * wf.wf_size() * 4 + uint64_t{lane} * 4 + offset % 4,
          true, true};
}

inline uint64_t flat_vector_offset(uint64_t value, bool signed_offset, unsigned scale = 1) {
  return signed_offset ? static_cast<uint64_t>(int64_t{static_cast<int32_t>(value)} * scale)
                       : value;
}

} // namespace rocjitsu::amdgpu
