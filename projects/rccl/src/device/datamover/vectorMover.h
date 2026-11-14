/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/// \file vectorMover.h
/// \brief Data mover that stages every byte through VGPRs.
///
/// Global memory is accessed with global_load/global_store (the address is
/// narrowed to address_space(1), so no flat instructions), LDS with
/// ds_load/ds_store (address_space(3)), and RegTile endpoints are the VGPRs
/// themselves. It handles every pair of spaces on every target, which makes it
/// the backend of last resort in a DataMover chain.
///
/// Memory-to-memory moves pick the widest access the pointers allow: if every
/// source and destination shares the same misalignment modulo W (W <= 16), the
/// head is peeled byte-wise up to a W boundary, the bulk runs at W bytes per
/// lane unrolled `Unroll` deep, and the sub-W tail is byte-wise. Each source is
/// loaded once and stored to all of its fan-out destinations.
///
/// Completion is synchronous from the issuing thread's point of view; the
/// compiler's own load/store counters order everything, so wait() is empty.
/// Cross-thread visibility (e.g. LDS written here, read by another wave) is
/// still the caller's barrier.
#pragma once

#include "dmCore.h"
#include "../tdm/cachePolicy.h"

namespace rccl {
namespace dm {

namespace detail {

template <int W>
struct VecOf;
template <>
struct VecOf<16> {
  using T = u32x4;
};
template <>
struct VecOf<8> {
  using T = u32x2;
};
template <>
struct VecOf<4> {
  using T = uint32_t;
};
template <>
struct VecOf<2> {
  using T = uint16_t;
};
template <>
struct VecOf<1> {
  using T = uint8_t;
};
template <int W>
using Vec = typename VecOf<W>::T;

template <Space S, int W, bool NT>
DM_INLINE Vec<W> ldMem(uintptr_t a) {
  using T = Vec<W>;
  if constexpr (S == Space::Global) {
    auto p = (const __attribute__((address_space(1))) T*)reinterpret_cast<const T*>(a);
    if constexpr (NT) return __builtin_nontemporal_load(p);
    else return *p;
  } else {
    static_assert(S == Space::Lds, "ldMem addresses memory, not registers");
    auto p = (const __attribute__((address_space(3))) T*)reinterpret_cast<const T*>(a);
    return *p;
  }
}

template <Space S, int W, bool NT>
DM_INLINE void stMem(uintptr_t a, Vec<W> v) {
  using T = Vec<W>;
  if constexpr (S == Space::Global) {
    auto p = (__attribute__((address_space(1))) T*)reinterpret_cast<T*>(a);
    if constexpr (NT) __builtin_nontemporal_store(v, p);
    else *p = v;
  } else {
    static_assert(S == Space::Lds, "stMem addresses memory, not registers");
    auto p = (__attribute__((address_space(3))) T*)reinterpret_cast<T*>(a);
    *p = v;
  }
}

// First n (<= 16) bytes at a, assembled into a pack; the rest is zero.
template <Space S>
DM_INLINE u32x4 ldBytes(uintptr_t a, uint32_t n) {
  u32x4 r = {0, 0, 0, 0};
#pragma unroll
  for (int b = 0; b < 16; ++b)
    if (b < n) r[b / 4] |= static_cast<uint32_t>(ldMem<S, 1, false>(a + b)) << (8 * (b % 4));
  return r;
}

template <Space S>
DM_INLINE void stBytes(uintptr_t a, u32x4 v, uint32_t n) {
#pragma unroll
  for (int b = 0; b < 16; ++b)
    if (b < n) stMem<S, 1, false>(a + b, static_cast<uint8_t>(v[b / 4] >> (8 * (b % 4))));
}

// `old` with its first n bytes replaced by those of `nw`.
DM_INLINE u32x4 mergeBytes(u32x4 old, u32x4 nw, uint32_t n) {
  u32x4 r = old;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint32_t lo = 4 * k;
    if (n >= lo + 4) {
      r[k] = nw[k];
    } else if (n > lo) {
      const uint32_t mask = (1u << (8 * (n - lo))) - 1;
      r[k] = (old[k] & ~mask) | (nw[k] & mask);
    }
  }
  return r;
}

// One step of a memory-to-memory copy: U accesses of W bytes per lane, `step`
// bytes apart, for every source and each of its fan-out destinations.
template <Space SS, Space DS, int W, int U, bool NT, int MS, int MD>
DM_INLINE void copyStep(const MemSet<SS, MS>& src, const MemSet<DS, MD>& dst, int fan, size_t off,
                                size_t step) {
#pragma unroll
  for (int s = 0; s < MS; ++s) {
    if (s >= src.n) break;
    Vec<W> v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = ldMem<SS, W, NT>(src.addr[s] + off + u * step);
#pragma unroll
    for (int d = 0; d < MD; ++d) {
      if (!fedBy(d, s, fan)) continue;
#pragma unroll
      for (int u = 0; u < U; ++u) stMem<DS, W, NT>(dst.addr[d] + off + u * step, v[u]);
    }
  }
}

// Copy [begin, end), whose length is a multiple of W and whose every pointer is
// W-aligned at `begin`.
template <Space SS, Space DS, int W, int U, bool NT, int MS, int MD>
DM_INLINE void copyRange(const Team& t, const MemSet<SS, MS>& src, const MemSet<DS, MD>& dst, int fan,
                                 size_t begin, size_t end) {
  const size_t step = static_cast<size_t>(t.nThreads) * W;
  size_t off = begin + static_cast<size_t>(t.tid) * W;
  for (; off + (U - 1) * step < end; off += U * step) copyStep<SS, DS, W, U, NT>(src, dst, fan, off, step);
  if constexpr (U > 1)
    for (; off < end; off += step) copyStep<SS, DS, W, 1, NT>(src, dst, fan, off, step);
}

template <Space SS, Space DS, int W, int U, bool NT, int MS, int MD>
DM_INLINE void copyPeeled(const Team& t, const MemSet<SS, MS>& src, const MemSet<DS, MD>& dst, int fan,
                                  size_t bytes) {
  size_t head = (W - (src.addr[0] & (W - 1))) & (W - 1);
  if (head > bytes) head = bytes;
  const size_t bulkEnd = head + ((bytes - head) & ~static_cast<size_t>(W - 1));
  copyRange<SS, DS, 1, 1, NT>(t, src, dst, fan, 0, head);
  copyRange<SS, DS, W, U, NT>(t, src, dst, fan, head, bulkEnd);
  copyRange<SS, DS, 1, 1, NT>(t, src, dst, fan, bulkEnd, bytes);
}

template <Space SS, Space DS, int U, bool NT, int MS, int MD>
DM_INLINE void copyMem(const Team& t, const MemSet<SS, MS>& src, const MemSet<DS, MD>& dst, int fan,
                               size_t bytes) {
  // Lowest bit in which any two pointers differ = largest W for which they all
  // share one misalignment, capped at a 16-byte pack.
  const uintptr_t a0 = src.addr[0];
  uintptr_t diff = kPackBytes;
#pragma unroll
  for (int s = 1; s < MS; ++s)
    if (s < src.n) diff |= src.addr[s] ^ a0;
#pragma unroll
  for (int d = 0; d < MD; ++d)
    if (d < dst.n) diff |= dst.addr[d] ^ a0;
  switch (diff & (~diff + 1)) {
  case 16: copyPeeled<SS, DS, 16, U, NT>(t, src, dst, fan, bytes); break;
  case 8: copyPeeled<SS, DS, 8, U, NT>(t, src, dst, fan, bytes); break;
  case 4: copyPeeled<SS, DS, 4, U, NT>(t, src, dst, fan, bytes); break;
  case 2: copyPeeled<SS, DS, 2, U, NT>(t, src, dst, fan, bytes); break;
  default: copyRange<SS, DS, 1, U, NT>(t, src, dst, fan, 0, bytes); break;
  }
}

// Bytes of pack i this thread owns in a `bytes`-long tile (0 past the end).
template <int NP, int N>
DM_INLINE uint32_t packBytes(const Team& t, int i, size_t bytes) {
  const size_t off = RegTile<NP, N>::packOffset(t, i);
  if (off >= bytes) return 0;
  return bytes - off < kPackBytes ? static_cast<uint32_t>(bytes - off) : kPackBytes;
}

// A partially filled pack is zero past `bytes`; packs entirely past it are left
// untouched. Misaligned memory takes the byte-wise path, so tiles want
// 16-byte-aligned memory to run at full speed.
template <Space SS, bool NT, int MS, int NP, int N>
DM_INLINE void loadTile(const Team& t, const MemSet<SS, MS>& src, RegTile<NP, N>& dst, int fan,
                                size_t bytes) {
  if (bytes > RegTile<NP, N>::capacity(t)) __builtin_trap();
#pragma unroll
  for (int d = 0; d < N; ++d) {
    const uintptr_t base = pick(src, d / fan);
    const bool aligned = (base & (kPackBytes - 1)) == 0;
#pragma unroll
    for (int i = 0; i < NP; ++i) {
      const uint32_t n = packBytes<NP, N>(t, i, bytes);
      if (n == 0) break;
      const uintptr_t a = base + RegTile<NP, N>::packOffset(t, i);
      dst.v[d][i] = (aligned && n == kPackBytes) ? ldMem<SS, 16, NT>(a) : ldBytes<SS>(a, n);
    }
  }
}

template <Space DS, bool NT, int NP, int N, int MD>
DM_INLINE void storeTile(const Team& t, const RegTile<NP, N>& src, const MemSet<DS, MD>& dst, int fan,
                                 size_t bytes) {
  if (bytes > RegTile<NP, N>::capacity(t)) __builtin_trap();
#pragma unroll
  for (int s = 0; s < N; ++s) {
#pragma unroll
    for (int d = 0; d < MD; ++d) {
      if (!fedBy(d, s, fan)) continue;
      const uintptr_t base = dst.addr[d];
      const bool aligned = (base & (kPackBytes - 1)) == 0;
#pragma unroll
      for (int i = 0; i < NP; ++i) {
        const uint32_t n = packBytes<NP, N>(t, i, bytes);
        if (n == 0) break;
        const uintptr_t a = base + RegTile<NP, N>::packOffset(t, i);
        if (aligned && n == kPackBytes) stMem<DS, 16, NT>(a, src.v[s][i]);
        else stBytes<DS>(a, src.v[s][i], n);
      }
    }
  }
}

// Both tiles were filled by the same team, so they share a layout and the copy
// is lane-local.
template <int NPS, int NS, int NPD, int ND>
DM_INLINE void regToReg(const Team& t, const RegTile<NPS, NS>& src, RegTile<NPD, ND>& dst, int fan,
                                size_t bytes) {
  if (bytes > RegTile<NPS, NS>::capacity(t) || bytes > RegTile<NPD, ND>::capacity(t)) __builtin_trap();
  constexpr int NP = NPS < NPD ? NPS : NPD;
#pragma unroll
  for (int d = 0; d < ND; ++d) {
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      if (s != d / fan) continue;
#pragma unroll
      for (int i = 0; i < NP; ++i) {
        const uint32_t n = packBytes<NP, 1>(t, i, bytes);
        if (n == 0) break;
        dst.v[d][i] = n == kPackBytes ? src.v[s][i] : mergeBytes(dst.v[d][i], src.v[s][i], n);
      }
    }
  }
}

}  // namespace detail

/// \tparam cp     Cache policy; a non-temporal hint selects nontemporal
///                global loads/stores. Scope bits are ignored (plain accesses).
/// \tparam Unroll Packs in flight per lane per source in the bulk loop.
template <CachePolicy cp = DEFAULT_CACHE_POLICY, int Unroll = 8>
struct VectorMover {
  static constexpr bool kSupported = true;
  static constexpr bool kNonTemporal = (cp & 7u) == static_cast<uint32_t>(TemporalHint::NT);

  template <Space S, Space D>
  static constexpr bool handles() {
    return true;
  }
  template <Space S, Space D>
  DM_INLINE bool accepts(const Team&, size_t) const {
    return true;
  }

  template <class Src, class Dst>
  DM_INLINE void issue(const Team& t, Src& src, Dst& dst, size_t bytes) {
    using S = Endpoint<Src>;
    using D = Endpoint<Dst>;
    const int fan = detail::fanOut(src, dst);
    if constexpr (S::space == Space::Reg && D::space == Space::Reg)
      detail::regToReg(t, src, dst, fan, bytes);
    else if constexpr (S::space == Space::Reg)
      detail::storeTile<D::space, kNonTemporal>(t, src, dst, fan, bytes);
    else if constexpr (D::space == Space::Reg)
      detail::loadTile<S::space, kNonTemporal>(t, src, dst, fan, bytes);
    else
      detail::copyMem<S::space, D::space, Unroll, kNonTemporal>(t, src, dst, fan, bytes);
  }

  DM_INLINE void wait(const Team&) {}
};

}  // namespace dm
}  // namespace rccl
