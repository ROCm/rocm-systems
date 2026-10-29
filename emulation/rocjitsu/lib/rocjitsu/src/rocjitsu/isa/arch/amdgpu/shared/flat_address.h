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

/// CDNA5 private-aperture prefix: address bits [63:58] (section 10.2).
inline constexpr unsigned kCdna5PrivatePrefixShift = 58;

/// @brief CDNA5 FLAT_SCRATCH_BASE for this wave (section 6.10).
/// @details wave32: {PRIVATE_BASE[63:58], 0, 5'b0, SCRATCH_BASE[56:7], 2'b0};
/// wave64: {PRIVATE_BASE[63:58], 1, 6'b0, SCRATCH_BASE[56:8], 2'b0}. A thread
/// forms its private pointer by adding its lane at bit 52 (51 for wave64) and
/// its byte offset, which is what LLVM emits for gfx1250.
inline uint64_t cdna5_flat_scratch_base(const Wavefront &wf) {
  const uint64_t prefix = wf.private_aperture_base() >> kCdna5PrivatePrefixShift
                                                            << kCdna5PrivatePrefixShift;
  if (wf.wf_size() == 64)
    return prefix | uint64_t{1} << 57 | (wf.scratch_base() >> 8 & ((uint64_t{1} << 49) - 1)) << 2;
  return prefix | (wf.scratch_base() >> 7 & ((uint64_t{1} << 50) - 1)) << 2;
}

/// Pure address translation shared by execution and pre-execution wait planning.
/// @details CDNA5 (EncodedLane) decodes the private aperture from the pointer
/// alone, per section 10.2: the prefix selects the aperture, bit 57 the
/// owning wave's width, the next bits its lane, and the rest the dword of
/// that wave's swizzled scratch. Any wave may dereference another's pointer
/// (section 10.2.1), so the issuing wave's own scratch state is not used.
inline FlatAddress translate_flat_address(const Wavefront &wf, uint64_t address, unsigned lane,
                                          FlatPrivateLayout layout) {
  if (layout == FlatPrivateLayout::None)
    return {address};
  uint64_t offset;
  if (layout == FlatPrivateLayout::EncodedLane) {
    const uint64_t prefix = wf.private_aperture_base() >> kCdna5PrivatePrefixShift;
    if (prefix == 0 || address >> kCdna5PrivatePrefixShift != prefix)
      return {address};
    const bool wave64 = (address >> 57) & 1;
    const unsigned lane_shift = wave64 ? 51 : 52;
    const uint64_t owner_lane = (address >> lane_shift) & (wave64 ? 63 : 31);
    const uint64_t dword = (address >> 2) & ((uint64_t{1} << (lane_shift - 2)) - 1);
    return {dword << (wave64 ? 8 : 7) | owner_lane << 2 | (address & 3), true, true};
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

/// Global virtual-address width of RDNA3, RDNA3.5 and RDNA4.
inline constexpr unsigned kRdnaFlatVaBits = 48;
/// Global virtual-address width of CDNA5.
inline constexpr unsigned kCdna5FlatVaBits = 57;

/// @brief Whether a FLAT address falls in the hole between the canonical
/// global ranges and is not in the shared or private aperture.
/// @details Global FLAT addresses are sign-extended from @p va_bits: 48 bits on
/// RDNA3-4 and 57 bits on CDNA5. Shared and private apertures are valid
/// exceptions. Callers choose which address to test: RDNA checks the base
/// before IOFFSET is added, CDNA5 checks the final address.
/// @param va_bits Architectural global virtual-address width.
inline bool flat_address_in_hole(const Wavefront &wf, uint64_t address, unsigned va_bits,
                                 FlatPrivateLayout layout) {
  const uint64_t high = address >> (va_bits - 1);
  if (high == 0 || high == (UINT64_MAX >> (va_bits - 1)))
    return false;
  if (wf.shared_aperture_base() != 0 && address >= wf.shared_aperture_base() &&
      address <= wf.shared_aperture_limit())
    return false;
  return !translate_flat_address(wf, address, 0, layout).private_address;
}

/// @brief Address of an RDNA FLAT lane after IOFFSET is added.
/// @details RDNA3/3.5/4 select the aperture from the base address, before
/// IOFFSET (section 11.3), and keep a shared or private offset to 32 bits, so
/// an IOFFSET that would cross the aperture boundary wraps inside it. Section
/// 11.1.1 leaves such crossings undefined; this follows gfx1100 and gfx1201:
/// private base + 0xfffffff8 with IOFFSET 16 reads scratch offset 8, private
/// base with IOFFSET -8 addresses scratch offset 0xfffffff8, and a 64-bit
/// atomic there still raises the private-atomic MEMVIOL. A global base adds
/// IOFFSET without wrapping, even into the hole.
/// @param base VADDR plus SADDR, before IOFFSET.
inline uint64_t rdna_flat_lane_address(const Wavefront &wf, uint64_t base, int64_t ioffset) {
  const auto wrap_within = [&](uint64_t aperture) {
    return aperture + static_cast<uint32_t>(base - aperture + static_cast<uint64_t>(ioffset));
  };
  const uint64_t shared_base = wf.shared_aperture_base();
  if (shared_base != 0 && base >= shared_base && base <= wf.shared_aperture_limit())
    return wrap_within(shared_base);
  const uint64_t private_window = wf.private_aperture_base() & ~uint64_t{UINT32_MAX};
  if (private_window != 0 && (base & ~uint64_t{UINT32_MAX}) == private_window)
    return wrap_within(private_window);
  return base + static_cast<uint64_t>(ioffset);
}

/// @brief Whether a FLAT atomic that resolved to the private aperture must
/// raise MEMVIOL.
/// @details RDNA3/3.5/4 section 11.1.1 and CDNA5 section 10.1.1: FLAT atomics
/// that map into scratch support 4-byte operands, and 8-byte atomics return
/// MEMVIOL. Each architecture decides which address is private (and whether
/// IOFFSET is applied first); this only states the operand-width rule.
/// @param private_address The lane's address translated to the private aperture.
/// @param atomic The instruction is an atomic read-modify-write.
/// @param elem_size Atomic operand size in bytes.
inline bool rejects_private_flat_atomic(bool private_address, bool atomic, uint32_t elem_size) {
  return private_address && atomic && elem_size == sizeof(uint64_t);
}

inline uint64_t flat_vector_offset(uint64_t value, bool signed_offset, unsigned scale = 1) {
  return signed_offset ? static_cast<uint64_t>(int64_t{static_cast<int32_t>(value)} * scale)
                       : value;
}

} // namespace rocjitsu::amdgpu
