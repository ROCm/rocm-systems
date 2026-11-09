// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vgpr_writeback.h
/// @brief Lane-major load data into its destination VGPRs.
///
/// Shared by global, scratch, buffer, FLAT and DS loads and returning atomics
/// (memory_pipeline.cpp vector_complete()). write_back_as<F>() is the inlined
/// path; write_back() picks the format at run time. What format_of() declines
/// stays in vector_complete()'s per-lane loop, and every path here writes the
/// register file that loop would.

#ifndef ROCJITSU_VM_AMDGPU_VGPR_WRITEBACK_H_
#define ROCJITSU_VM_AMDGPU_VGPR_WRITEBACK_H_

#include "rocjitsu/base/rj_compiler.h"
#include "util/host_target.h"

#include <bit>
#include <cstdint>
#include <cstring>
#include <optional>

namespace rocjitsu {
namespace amdgpu {
namespace vgpr_writeback {

/// @brief What a load's data becomes in its destination VGPRs.
struct DstFormat {
  uint8_t elem_size = 4;
  uint8_t num_elems = 1;
  bool sign_extend = false; ///< i8 / i16.
  bool d16_hi = false;      ///< Writes the upper 16 bits.
  bool d16_lo = false;      ///< Writes the lower 16 bits.

  constexpr bool operator==(const DstFormat &) const = default;
};

enum class Isa { scalar, avx2 };

/// The implementation write_back_as() uses unless told otherwise.
inline constexpr Isa kHostIsa = RJ_HOST_HAS_AVX2 ? Isa::avx2 : Isa::scalar;

template <DstFormat F> constexpr uint32_t kStride = F.elem_size * F.num_elems;
/// @brief Destination VGPRs a load of format @p F writes.
template <DstFormat F> constexpr uint32_t kRegs = (kStride<F> + 3) / 4;
template <DstFormat F> constexpr bool kD16 = F.d16_hi || F.d16_lo;

/// Largest per-lane footprint of a format (b128).
inline constexpr uint32_t kMaxStride = 16;

/// EXEC with lanes 0 .. lanes - 1 active, for 1 <= lanes <= 64: Wave32 is
/// 0xFFFF'FFFF, Wave64 all 64 bits.
constexpr uint64_t prefix(uint32_t lanes) { return 0xFFFF'FFFF'FFFF'FFFFull >> (64 - lanes); }
static_assert(prefix(32) == 0xFFFF'FFFFull && prefix(64) == 0xFFFF'FFFF'FFFF'FFFFull);

/// @brief The format a load of @p elem_size x @p num_elems bytes per lane
/// writes back as, or none when only the per-lane loop takes it.
/// @details Whole dwords are one format per register count: lane-major data
/// of 8-byte elements and of twice as many 4-byte ones is the same bytes, and
/// sign_extend and d16 only act on 8- and 16-bit elements. Narrow elements
/// take one register; several of them per lane are left to the per-lane loop.
constexpr std::optional<DstFormat> format_of(uint32_t elem_size, uint32_t num_elems,
                                             bool sign_extend, bool d16_hi, bool d16_lo) {
  const uint32_t stride = elem_size * num_elems;
  if (elem_size >= 4) {
    if (elem_size % 4 != 0 || stride == 0 || stride > kMaxStride)
      return std::nullopt;
    return DstFormat{.elem_size = 4, .num_elems = static_cast<uint8_t>(stride / 4)};
  }
  if ((elem_size != 1 && elem_size != 2) || num_elems != 1 || (d16_hi && d16_lo))
    return std::nullopt;
  return DstFormat{.elem_size = static_cast<uint8_t>(elem_size),
                   .num_elems = 1,
                   .sign_extend = sign_extend,
                   .d16_hi = d16_hi,
                   .d16_lo = d16_lo};
}

/// @brief Destination VGPRs @p f writes: kRegs<F> at run time.
constexpr uint32_t regs(const DstFormat &f) { return (f.elem_size * f.num_elems + 3) / 4; }

/// @brief Write the lanes in @p exec, lane-major, to their destination VGPRs.
/// @details @p f comes from format_of(). @p dst holds the raw lanes of
/// regs(@p f) registers, each at least @p wave lanes long; @p lanes holds
/// @p wave lanes, every lane readable; @p exec has no lanes past the wave.
/// A handful of lanes is written one at a time, without the per-format call.
void write_back(const DstFormat &f, uint8_t *const *dst, bool sram_ecc, uint32_t wave,
                uint64_t exec, const uint8_t *lanes);

/// One writeback. Inlined, so its fields stay in registers; the arrays are
/// copies because the stores below can alias the caller's. Groups may sit
/// apart: a response buffer's groups are contiguous, and a source read in place
/// hands over whatever it holds.
template <DstFormat F, uint32_t Wave> struct Target {
  uint8_t *dst[kRegs<F>];          // lane l of register r: dst[r] + 4 * l
  const uint8_t *group[Wave / 16]; // lane l: group[l / 16] + (l % 16) * kStride<F>
  bool ecc;                        // SRAM ECC: d16 loads zero the other half
};

template <DstFormat F, uint32_t Wave>
RJ_ALWAYS_INLINE inline Target<F, Wave> target(uint8_t *const *dst, bool sram_ecc,
                                               const uint8_t *const *groups) {
  Target<F, Wave> t;
  for (uint32_t r = 0; r < kRegs<F>; ++r)
    t.dst[r] = dst[r];
  for (uint32_t g = 0; g < Wave / 16; ++g)
    t.group[g] = groups[g];
  t.ecc = kD16<F> && sram_ecc;
  return t;
}

// Defined below and in vgpr_writeback_avx2.h.
namespace scalar {
template <DstFormat F, uint32_t Wave>
RJ_NOINLINE void for_wave(uint8_t *const *dst, bool sram_ecc, uint64_t exec,
                          const uint8_t *const *groups);
}
namespace avx2 {
template <DstFormat F, uint32_t Wave>
RJ_ALWAYS_INLINE inline void by_exec(const Target<F, Wave> &t, uint64_t exec);
}

/// @brief Lanes in @p exec, leaving every other lane unchanged.
/// @details The wave size is a template so each lane loop's trip count is
/// constant. avx2 is inlined here, where the group pointers are still in
/// registers.
template <DstFormat F, Isa I = kHostIsa>
RJ_ALWAYS_INLINE inline void write_back_as(uint8_t *const *dst, bool sram_ecc, uint32_t wave,
                                           uint64_t exec, const uint8_t *const *groups) {
  static_assert(I != Isa::avx2 || RJ_HOST_HAS_AVX2, "Isa::avx2 needs an AVX2 build");
  if constexpr (I == Isa::avx2) {
    if (wave == 32)
      return avx2::by_exec<F, 32>(target<F, 32>(dst, sram_ecc, groups), exec);
    avx2::by_exec<F, 64>(target<F, 64>(dst, sram_ecc, groups), exec);
  } else {
    if (wave == 32)
      return scalar::for_wave<F, 32>(dst, sram_ecc, exec, groups);
    scalar::for_wave<F, 64>(dst, sram_ecc, exec, groups);
  }
}

namespace scalar {

enum class Coverage { AllLanes, FewLanes, Chunks };

template <DstFormat F, uint32_t Wave>
RJ_ALWAYS_INLINE inline void by_exec(const Target<F, Wave> &t, uint64_t exec);
template <DstFormat F, uint32_t Wave>
RJ_ALWAYS_INLINE inline void store_wave(const Target<F, Wave> &t);
template <DstFormat F, uint32_t Wave, uint32_t N>
RJ_ALWAYS_INLINE inline void store_lanes(const Target<F, Wave> &t, uint32_t first);
template <DstFormat F, uint32_t Wave>
RJ_ALWAYS_INLINE inline void store_active(const Target<F, Wave> &t, uint64_t lanes);
template <DstFormat F, uint32_t Wave>
RJ_ALWAYS_INLINE inline void store_chunks(const Target<F, Wave> &t, uint64_t exec);
template <DstFormat F, uint32_t Wave>
RJ_ALWAYS_INLINE inline void lane(const Target<F, Wave> &t, uint32_t l);
template <DstFormat F, uint32_t Wave>
RJ_ALWAYS_INLINE inline const uint8_t *src(const Target<F, Wave> &t, uint32_t l);
template <DstFormat F>
RJ_ALWAYS_INLINE inline uint32_t narrow(const uint8_t *src, uint32_t old, bool ecc);
RJ_ALWAYS_INLINE inline uint64_t load64(const uint8_t *p);
RJ_ALWAYS_INLINE inline void store64(uint8_t *p, uint64_t v);

/// A function per wave size, so each gets its own register allocation: with
/// both waves inlined into one function, scalar u16 spilled its constants and
/// got much slower. It builds the Target itself: receiving one by value costs
/// two store-forwarding stalls per call, which more than doubled a b32 full
/// wave.
template <DstFormat F, uint32_t Wave>
RJ_NOINLINE void for_wave(uint8_t *const *dst, bool sram_ecc, uint64_t exec,
                          const uint8_t *const *groups) {
  by_exec<F, Wave>(target<F, Wave>(dst, sram_ecc, groups), exec);
}

template <uint32_t Wave> RJ_ALWAYS_INLINE inline Coverage choose(uint64_t exec) {
  if (exec == prefix(Wave))
    return Coverage::AllLanes;
  // Walking all 8-lane chunks for these instead is measurably slower.
  if (std::popcount(exec) <= 4)
    return Coverage::FewLanes;
  return Coverage::Chunks;
}

/// Each EXEC shape has a cheaper walk than the other two. A branch on the
/// shape mispredicts often; avx2 masks b32 instead of branching.
template <DstFormat F, uint32_t Wave> void by_exec(const Target<F, Wave> &t, uint64_t exec) {
  switch (choose<Wave>(exec)) {
  case Coverage::AllLanes:
    return store_wave<F, Wave>(t);
  case Coverage::FewLanes:
    return store_active<F, Wave>(t, exec);
  case Coverage::Chunks:
    return store_chunks<F, Wave>(t, exec);
  }
}

enum class Form { Block, Pairs, Extend, PerLane };

template <DstFormat F>
constexpr Form kForm = kStride<F> == 4               ? Form::Block
                       : kRegs<F> % 2 == 0           ? Form::Pairs
                       : F.elem_size < 4 && !kD16<F> ? Form::Extend
                                                     : Form::PerLane;

template <DstFormat F, uint32_t Wave, uint32_t N>
RJ_ALWAYS_INLINE inline void block(const Target<F, Wave> &t, uint32_t first);
template <DstFormat F, uint32_t Wave, uint32_t N>
RJ_ALWAYS_INLINE inline void pairs(const Target<F, Wave> &t, uint32_t first);
template <DstFormat F, uint32_t Wave, uint32_t N>
RJ_ALWAYS_INLINE inline void extend(const Target<F, Wave> &t, uint32_t first);

/// Every lane, a group at a time: groups may sit apart.
template <DstFormat F, uint32_t Wave> void store_wave(const Target<F, Wave> &t) {
  for (uint32_t g = 0; g < Wave / 16; ++g)
    store_lanes<F, Wave, 16>(t, 16 * g);
}

/// Lanes first..first+N-1, all active and in one group.
template <DstFormat F, uint32_t Wave, uint32_t N>
void store_lanes(const Target<F, Wave> &t, uint32_t first) {
  static_assert(N <= 16 && 16 % N == 0);
  if constexpr (kForm<F> == Form::Block) {
    block<F, Wave, N>(t, first);
  } else if constexpr (kForm<F> == Form::Pairs) {
    pairs<F, Wave, N>(t, first);
  } else if constexpr (kForm<F> == Form::Extend) {
    extend<F, Wave, N>(t, first);
  } else {
    // b96 and d16 are rare, so they stay one lane at a time.
    for (uint32_t l = 0; l < N; ++l)
      lane<F, Wave>(t, first + l);
  }
}

/// The lanes in `lanes`, one at a time. A branch-free form for up to four
/// lanes (lowest, second lowest, second highest, highest) is slower on the
/// FewLanes cell and no better on the mix.
template <DstFormat F, uint32_t Wave> void store_active(const Target<F, Wave> &t, uint64_t lanes) {
  for (; lanes; lanes &= lanes - 1)
    lane<F, Wave>(t, std::countr_zero(lanes));
}

/// Fully active 8-lane chunks as store_lanes<8>, the rest one lane at a time:
/// the 48-lane prefix is six chunks. store_active over all of EXEC instead is
/// much slower on the b32 Chunks cell.
template <DstFormat F, uint32_t Wave> void store_chunks(const Target<F, Wave> &t, uint64_t exec) {
  for (uint32_t c = 0; c < Wave / 8; ++c) {
    const uint64_t chunk = uint64_t{0xFF} << (8 * c);
    if ((exec & chunk) == chunk)
      store_lanes<F, Wave, 8>(t, 8 * c);
    else
      store_active<F, Wave>(t, exec & chunk);
  }
}

/// b32: lane-major data is, for one register, the register layout.
///   data  │L0│L1│L2│L3│...   =   v0  │L0│L1│L2│L3│...
template <DstFormat F, uint32_t Wave, uint32_t N>
void block(const Target<F, Wave> &t, uint32_t first) {
  std::memcpy(t.dst[0] + 4 * first, src(t, first), 4 * N);
}

/// b64, b128: a 64-bit load holds one lane's dwords for two registers, so two
/// lanes give one 64-bit store per register:
///   a = │L0.r0 L0.r1│  b = │L1.r0 L1.r1│   ->   v0 │L0.r0 L1.r0│   v1 │L0.r1 L1.r1│
/// A 32-bit copy per register and lane instead, or this loop not fully
/// unrolled, is much slower on full b64 and b128 waves.
template <DstFormat F, uint32_t Wave, uint32_t N>
void pairs(const Target<F, Wave> &t, uint32_t first) {
  const uint8_t *s = src(t, first);
#pragma GCC unroll 8
  for (uint32_t l = 0; l < N; l += 2)
    for (uint32_t r = 0; r < kRegs<F>; r += 2) {
      const uint64_t a = load64(s + l * kStride<F> + 4 * r);
      const uint64_t b = load64(s + (l + 1) * kStride<F> + 4 * r);
      store64(t.dst[r] + 4 * (first + l), (a & 0xFFFFFFFFu) | b << 32);
      store64(t.dst[r + 1] + 4 * (first + l), a >> 32 | (b & 0xFFFFFFFF00000000u));
    }
}

template <DstFormat F, uint32_t Wave, uint32_t N>
void extend(const Target<F, Wave> &t, uint32_t first) {
  const uint8_t *s = src(t, first);
  for (uint32_t l = 0; l < N; l += 2) {
    const uint64_t a = narrow<F>(s + l * F.elem_size, 0, t.ecc);
    const uint64_t b = narrow<F>(s + (l + 1) * F.elem_size, 0, t.ecc);
    store64(t.dst[0] + 4 * (first + l), a | b << 32);
  }
}

template <DstFormat F, uint32_t Wave> void lane(const Target<F, Wave> &t, uint32_t l) {
  const uint8_t *s = src(t, l);
  if constexpr (F.elem_size >= 4) {
    for (uint32_t r = 0; r < kRegs<F>; ++r)
      std::memcpy(t.dst[r] + 4 * l, s + 4 * r, 4);
  } else {
    uint32_t v;
    std::memcpy(&v, t.dst[0] + 4 * l, 4);
    v = narrow<F>(s, v, t.ecc);
    std::memcpy(t.dst[0] + 4 * l, &v, 4);
  }
}

template <DstFormat F, uint32_t Wave> const uint8_t *src(const Target<F, Wave> &t, uint32_t l) {
  return t.group[l / 16] + (l % 16) * kStride<F>;
}

/// The dword an 8- or 16-bit load leaves in a register that held old
/// (high half │ low half):
///   u8 u16 i8 i16   │   extended h   │
///   d16_hi          │ h      │ old   │    SRAM ECC: │ h │ 0 │
///   d16_lo          │ old    │ h     │    SRAM ECC: │ 0 │ h │
template <DstFormat F> uint32_t narrow(const uint8_t *src, uint32_t old, bool ecc) {
  static_assert(F.num_elems == 1 && (F.elem_size == 1 || F.elem_size == 2));
  uint32_t h;
  if constexpr (F.elem_size == 1) {
    h = F.sign_extend ? static_cast<uint32_t>(static_cast<int8_t>(*src)) : *src;
  } else {
    uint16_t x;
    std::memcpy(&x, src, 2);
    h = F.sign_extend ? static_cast<uint32_t>(static_cast<int16_t>(x)) : x;
  }
  if constexpr (F.d16_hi)
    return (ecc ? 0 : old & 0xFFFFu) | h << 16;
  else if constexpr (F.d16_lo)
    return (ecc ? 0 : old & 0xFFFF0000u) | (h & 0xFFFFu);
  return h;
}

uint64_t load64(const uint8_t *p) {
  uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}

void store64(uint8_t *p, uint64_t v) { std::memcpy(p, &v, 8); }

} // namespace scalar
} // namespace vgpr_writeback
} // namespace amdgpu
} // namespace rocjitsu

#if RJ_HOST_HAS_AVX2
#include "rocjitsu/vm/amdgpu/vgpr_writeback_avx2.h"
#endif

#endif // ROCJITSU_VM_AMDGPU_VGPR_WRITEBACK_H_
