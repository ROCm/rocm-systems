// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/wavefront.h"

/// \NPI Qualify workgroup addressing and bounds when adding an ISA.
namespace rocjitsu::amdgpu {

enum class FlatPrivateLayout { None, Interleaved, EncodedLane };

/// @brief Generic-private backing layout for each supported architecture.
inline constexpr FlatPrivateLayout flat_private_layout(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
    return FlatPrivateLayout::Interleaved;
  case ROCJITSU_CODE_ARCH_CDNA5:
    return FlatPrivateLayout::EncodedLane;
  default:
    return FlatPrivateLayout::None;
  }
}

/// Map private bytes to the dword-interleaved scratch backing shared by all lanes.
constexpr uint64_t scratch_lane_address(uint64_t base, uint64_t offset, unsigned lane,
                                        unsigned wave_size) {
  return base + (offset / 4) * wave_size * 4 + uint64_t{lane} * 4 + offset % 4;
}

/// The gfx1250 wave32 flat scratch pointer names the first DWORD of lane zero.
inline uint64_t gfx1250_flat_scratch_base(const Wavefront &wf) {
  constexpr uint64_t scratch_address_mask = (uint64_t{1} << 57) - 1;
  return wf.private_aperture_base() | ((wf.scratch_base() & scratch_address_mask) >> 7 << 2);
}

struct FlatAddress {
  uint64_t value;
  bool private_address = false;
  bool swizzled = false;
  uint32_t private_offset = 0;
};

/// GFX11 and GFX12 select the FLAT segment before adding the instruction immediate.
/// GFX1250 tests the completed 64-bit address instead.
inline constexpr bool flat_selects_segment_before_offset(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4;
}

inline constexpr bool flat_in_shared_aperture(uint64_t pointer, uint64_t base, uint64_t limit) {
  return base && pointer >= base && pointer <= limit;
}

/// Pure address translation shared by execution and pre-execution wait planning.
inline FlatAddress translate_flat_address(const Wavefront &wf, uint64_t address, unsigned lane,
                                          FlatPrivateLayout layout, int32_t segment_offset = 0) {
  if (layout == FlatPrivateLayout::None)
    return {address + segment_offset};
  if (layout == FlatPrivateLayout::EncodedLane) {
    const uint64_t pointer = address + segment_offset;
    const uint64_t aperture = wf.private_aperture_base();
    if (!aperture || pointer < aperture || pointer > wf.private_aperture_limit())
      return {pointer};
    // The pointer contains the backing DWORD index and the target lane,
    // including when an instruction addresses another wave's scratch.
    constexpr uint64_t dword_mask = ((uint64_t{1} << 52) - 1) & ~uint64_t{3};
    const uint64_t backing =
        ((pointer & dword_mask) << 5) | (((pointer >> 52) & 31) << 2) | (pointer & 3);
    return {backing, true, true};
  }
  uint64_t offset;
  if (wf.flat_pointer32()) {
    const uint64_t base = wf.private_aperture_base();
    if (!base || address < base || address > wf.private_aperture_limit())
      return {address + segment_offset};
    offset = address - base;
  } else {
    const uint32_t private_hi = wf.private_aperture_base() >> 32;
    if (!private_hi || static_cast<uint32_t>(address >> 32) != private_hi)
      return {address + segment_offset};
    offset = static_cast<uint32_t>(address);
  }
  // The private swizzle takes the low 32-bit lane offset.
  offset = static_cast<uint32_t>(offset + segment_offset);
  return {scratch_lane_address(wf.scratch_base(), offset, lane, wf.wf_size()), true, true,
          static_cast<uint32_t>(offset)};
}

inline uint64_t flat_vector_offset(uint64_t value, bool signed_offset, unsigned scale = 1) {
  return signed_offset ? static_cast<uint64_t>(int64_t{static_cast<int32_t>(value)} * scale)
                       : value;
}

} // namespace rocjitsu::amdgpu
