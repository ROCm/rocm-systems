// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file lds_writeback.h
/// @brief A DS load's writeback: its lane-major data into its destination
/// VGPRs, specialized by opcode and by wave size.
///
/// @details Reads top-down; stop at the level that answers your question.
///
///   Level 1  dispatch: by opcode, by wave, by EXEC shape
///   Level 2  the stores each format allows
///   Level 3  per-lane values and memory helpers
///
/// Generated execute_impl() code describes each plain load with constants, so
/// the opcode's destination format is a template parameter.
///
/// Census writebacks by EXEC shape (wave64, one mark per two lanes):
///
///   Coverage  share
///   Wave      65.6%  ████████████████████████████████
///   FewLanes  29.4%  ·····█··························  almost always one lane
///   Chunks     5.0%  ████████████████████████········  mostly a 48-lane prefix
///
/// and by format: b32 83.9%, b128 7.9%, u16 5.6%, b64 2.5%, the rest 0.1%.
///
/// The data is read 16 lanes at a time: lanes 16g..16g+15 are lane-major at
/// group[g], in staging or in place in LDS (LdsStagedLoad::group).
///
/// Two implementations with the same levels:
///   Isa::scalar  below; plain C++, builds anywhere;
///   Isa::avx2    lds_writeback_avx2.h; the default in AVX2 builds.
/// Standalone comparison over the DS-pattern census
/// (~/rj-lds-patterns/microbench/writeback, Ryzen 9 9950X), ns per writeback,
/// census mix / runs16: per-lane baseline 29.2 / 26.6, scalar 9.1 / 5.1,
/// avx2 6.2 / 4.3.

#ifndef ROCJITSU_VM_AMDGPU_LDS_WRITEBACK_H_
#define ROCJITSU_VM_AMDGPU_LDS_WRITEBACK_H_

#include "rocjitsu/base/rj_compiler.h"
#include "util/host_target.h"

#include <bit>
#include <cstdint>
#include <cstring>

namespace rocjitsu {
namespace amdgpu {
namespace lds_writeback {

/// @brief What a DS load's data becomes in its destination VGPRs.
struct DstFormat {
  uint8_t elem_size = 4;
  uint8_t num_elems = 1;
  bool sign_extend = false; ///< i8 / i16.
  bool d16_hi = false;      ///< Writes the upper 16 bits.
  bool d16_lo = false;      ///< Writes the lower 16 bits.
};

enum class Isa { scalar, avx2 };

/// The implementation write_back_as() uses unless told otherwise.
inline constexpr Isa kHostIsa = RJ_HOST_HAS_AVX2 ? Isa::avx2 : Isa::scalar;

template <DstFormat F> constexpr uint32_t kStride = F.elem_size * F.num_elems;
/// @brief Destination VGPRs a load of format @p F writes.
template <DstFormat F> constexpr uint32_t kRegs = (kStride<F> + 3) / 4;
template <DstFormat F> constexpr bool kD16 = F.d16_hi || F.d16_lo;
template <uint32_t Wave>
constexpr uint64_t kAll = Wave == 64 ? ~uint64_t{0} : (uint64_t{1} << Wave) - 1;

/// One writeback, as every level sees it. It never crosses an out-of-line call:
/// everything that takes it is inlined, so its fields live in registers. Its
/// arrays are copies of the caller's, which the stores below could alias.
template <DstFormat F, uint32_t Wave> struct Target {
  uint8_t *dst[kRegs<F>];          // lane l of register r: dst[r] + 4 * l
  const uint8_t *group[Wave / 16]; // lane l: group[l / 16] + (l % 16) * kStride<F>
  bool ecc;                        // SRAM ECC: d16 loads zero the other half
};

template <DstFormat F, uint32_t Wave>
[[gnu::always_inline]] inline Target<F, Wave> target(uint8_t *const *dst, bool sram_ecc,
                                                     const uint8_t *const *groups) {
  Target<F, Wave> t;
  for (uint32_t r = 0; r < kRegs<F>; ++r)
    t.dst[r] = dst[r];
  for (uint32_t g = 0; g < Wave / 16; ++g)
    t.group[g] = groups[g];
  t.ecc = kD16<F> && sram_ecc;
  return t;
}

// The implementations' levels, defined below and in lds_writeback_avx2.h.
namespace scalar {
template <DstFormat F, uint32_t Wave>
RJ_NOINLINE void for_wave(uint8_t *const *dst, bool sram_ecc, uint64_t exec,
                          const uint8_t *const *groups);
}
namespace avx2 {
template <DstFormat F, uint32_t Wave>
[[gnu::always_inline]] inline void by_exec(const Target<F, Wave> &t, uint64_t exec);
}

// ---------------------------------------------------------------------------
// Level 1: dispatch by opcode, by wave, by EXEC shape.
// ---------------------------------------------------------------------------

/// @brief Write a load's data to its destination VGPRs for the lanes in
/// @p exec, leaving every other lane unchanged.
/// @details By opcode: F is the opcode's destination format; @p dst holds the
/// raw lanes of its kRegs<F> registers. By wave: Wavefront::exec() has no lanes
/// past the wave, so the wave size can be a template parameter and every lane
/// loop has a constant trip count. @p groups holds Wave / 16 group pointers.
template <DstFormat F, Isa I = kHostIsa>
[[gnu::always_inline]] inline void write_back_as(uint8_t *const *dst, bool sram_ecc, uint32_t wave,
                                                 uint64_t exec, const uint8_t *const *groups) {
  static_assert(I == Isa::scalar || RJ_HOST_HAS_AVX2, "Isa::avx2 needs an AVX2 build");
  if constexpr (I == Isa::avx2) {
    // Inline in the load's handler, where the group pointers are still in
    // registers (lds_inline_access_steps.h).
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

enum class Coverage { Wave, FewLanes, Chunks };

template <DstFormat F, uint32_t Wave>
[[gnu::always_inline]] inline void by_exec(const Target<F, Wave> &t, uint64_t exec);
template <DstFormat F, uint32_t Wave>
[[gnu::always_inline]] inline void store_wave(const Target<F, Wave> &t);
template <DstFormat F, uint32_t Wave, uint32_t N>
[[gnu::always_inline]] inline void store_lanes(const Target<F, Wave> &t, uint32_t first);
template <DstFormat F, uint32_t Wave>
[[gnu::always_inline]] inline void store_active(const Target<F, Wave> &t, uint64_t lanes);
template <DstFormat F, uint32_t Wave>
[[gnu::always_inline]] inline void store_chunks(const Target<F, Wave> &t, uint64_t exec);
template <DstFormat F, uint32_t Wave>
[[gnu::always_inline]] inline void lane(const Target<F, Wave> &t, uint32_t l);
template <DstFormat F, uint32_t Wave>
[[gnu::always_inline]] inline const uint8_t *src(const Target<F, Wave> &t, uint32_t l);
template <DstFormat F>
[[gnu::always_inline]] inline uint32_t narrow(const uint8_t *src, uint32_t old, bool ecc);
[[gnu::always_inline]] inline uint64_t load64(const uint8_t *p);
[[gnu::always_inline]] inline void store64(uint8_t *p, uint64_t v);

/// A function per wave size, so each gets its own register allocation: with
/// both waves inlined into one function, scalar u16 spilled its constants
/// (6.7 -> 10.8 ns). It builds the Target itself: receiving one by value
/// costs store-forwarding stalls (b32 full 3.5 -> 8.3 ns).
template <DstFormat F, uint32_t Wave>
RJ_NOINLINE void for_wave(uint8_t *const *dst, bool sram_ecc, uint64_t exec,
                          const uint8_t *const *groups) {
  by_exec<F, Wave>(target<F, Wave>(dst, sram_ecc, groups), exec);
}

template <uint32_t Wave> [[gnu::always_inline]] inline Coverage choose(uint64_t exec) {
  if (exec == kAll<Wave>)
    return Coverage::Wave;
  // Walking all 8-lane chunks for these instead: 3.3 -> 4.1 ns on the cell.
  if (std::popcount(exec) <= 4)
    return Coverage::FewLanes;
  return Coverage::Chunks;
}

/// By EXEC shape. On the census mix these branches mispredict 0.26 times per
/// call; avx2::by_exec stores b32 under a mask instead (0.03).
template <DstFormat F, uint32_t Wave> void by_exec(const Target<F, Wave> &t, uint64_t exec) {
  switch (choose<Wave>(exec)) {
  case Coverage::Wave:
    return store_wave<F, Wave>(t);
  case Coverage::FewLanes:
    return store_active<F, Wave>(t, exec);
  case Coverage::Chunks:
    return store_chunks<F, Wave>(t, exec);
  }
}

// ---------------------------------------------------------------------------
// Level 2: the stores each format allows.
// ---------------------------------------------------------------------------

enum class Form { Block, Pairs, Extend, PerLane };

template <DstFormat F>
constexpr Form kForm = kStride<F> == 4               ? Form::Block
                       : kRegs<F> % 2 == 0           ? Form::Pairs
                       : F.elem_size < 4 && !kD16<F> ? Form::Extend
                                                     : Form::PerLane;

template <DstFormat F, uint32_t Wave, uint32_t N>
[[gnu::always_inline]] inline void block(const Target<F, Wave> &t, uint32_t first);
template <DstFormat F, uint32_t Wave, uint32_t N>
[[gnu::always_inline]] inline void pairs(const Target<F, Wave> &t, uint32_t first);
template <DstFormat F, uint32_t Wave, uint32_t N>
[[gnu::always_inline]] inline void extend(const Target<F, Wave> &t, uint32_t first);

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
    // b96 and d16: 0.11% of the census together.
    for (uint32_t l = 0; l < N; ++l)
      lane<F, Wave>(t, first + l);
  }
}

/// The lanes in `lanes`, one at a time. A branch-free form for up to four
/// lanes (lowest, second lowest, second highest, highest) is slower on the
/// FewLanes cell (3.7 vs 3.3 ns) and within 0.1 ns on the mix.
template <DstFormat F, uint32_t Wave> void store_active(const Target<F, Wave> &t, uint64_t lanes) {
  for (; lanes; lanes &= lanes - 1)
    lane<F, Wave>(t, std::countr_zero(lanes));
}

/// Fully active 8-lane chunks as store_lanes<8>, the rest one lane at a time:
/// the 48-lane prefix is six chunks. store_active over all of EXEC instead:
/// b32 Chunks cell 6.8 -> 10.0 ns.
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
/// A 32-bit copy per register and lane instead: b128 full 29.7 vs 21.0 ns,
/// b64 15.6 vs 10.8. Without full unrolling: b128 28.5, b64 12.7.
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

/// u8, i8, u16, i16: two extended lanes per 64-bit store.
template <DstFormat F, uint32_t Wave, uint32_t N>
void extend(const Target<F, Wave> &t, uint32_t first) {
  const uint8_t *s = src(t, first);
  for (uint32_t l = 0; l < N; l += 2) {
    const uint64_t a = narrow<F>(s + l * F.elem_size, 0, t.ecc);
    const uint64_t b = narrow<F>(s + (l + 1) * F.elem_size, 0, t.ecc);
    store64(t.dst[0] + 4 * (first + l), a | b << 32);
  }
}

// ---------------------------------------------------------------------------
// Level 3: one lane, one value, raw memory.
// ---------------------------------------------------------------------------

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

/// Lane l's data.
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
} // namespace lds_writeback
} // namespace amdgpu
} // namespace rocjitsu

#if RJ_HOST_HAS_AVX2
#include "rocjitsu/vm/amdgpu/lds_writeback_avx2.h"
#endif

#endif // ROCJITSU_VM_AMDGPU_LDS_WRITEBACK_H_
