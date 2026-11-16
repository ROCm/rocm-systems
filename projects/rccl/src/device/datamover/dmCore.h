/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/// \file dmCore.h
/// \brief Vocabulary types shared by every data-mover backend: memory spaces,
///        teams of participating threads, and typed endpoints.
///
/// An endpoint's memory space is part of its TYPE, never a runtime property, so
/// a backend can select instructions (global_*, ds_*, tensor_*, SDMA packets)
/// at compile time. Three endpoint kinds exist:
///
///   * MemSet<Space::Global, N> -- up to N pointers into global memory (local
///     HBM, peer HBM mapped over XGMI/IPC, or host-pinned memory).
///   * MemSet<Space::Lds, N>    -- up to N generic pointers into this block's LDS.
///   * RegTile<NPacks, N>       -- N register-resident tiles, distributed across
///     the team (see RegTile for the layout).
///
/// Multiple sources and destinations follow one fan-out rule: with S sources and
/// D destinations, D must be a multiple of S, and source s is copied to the
/// destinations [s*F, (s+1)*F) where F = D/S. So S == D is a batch of
/// independent pairs, S == 1 is a broadcast, and anything in between is a batch
/// of broadcasts. Every source moves the same `bytes`; sources and destinations
/// must not overlap.
#pragma once

#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include <type_traits>

// A RegTile endpoint stays in VGPRs only if every call between the caller and
// the register access inlines; one outlined frame turns the tile into scratch.
#define DM_INLINE __device__ __forceinline__

namespace rccl {
namespace dm {

enum struct Space : uint32_t {
  Global,
  Lds,
  Reg,
};

using u32x2 = __attribute__((ext_vector_type(2))) uint32_t;
using u32x4 = __attribute__((ext_vector_type(4))) uint32_t;

/// Bytes per register pack: one b128 access per lane.
constexpr uint32_t kPackBytes = 16;

DM_INLINE uint32_t waveSize() {
  return __builtin_amdgcn_wavefrontsize();
}

// ============================================================================
//  Team
// ============================================================================

/// The set of threads that cooperatively issue one move. Every member calls the
/// mover with identical arguments; non-members must not call it.
///
/// Backends built on per-wave DMA engines (TDM, async-to-LDS) need the team to
/// be whole waves starting on a wave boundary. A team that is not falls through
/// to the next backend in the DataMover chain.
struct Team {
  uint32_t tid;       // rank of the calling thread within the team
  uint32_t nThreads;  // team size

  /// Every thread of a 1-D block.
  DM_INLINE static Team block() {
    return {threadIdx.x, blockDim.x};
  }
  /// The contiguous waves [firstWave, firstWave + nWaves) of a 1-D block.
  DM_INLINE static Team waves(uint32_t firstWave, uint32_t nWaves) {
    return {threadIdx.x - firstWave * waveSize(), nWaves * waveSize()};
  }
  /// The calling wave alone.
  DM_INLINE static Team wave() {
    return {__lane_id(), waveSize()};
  }

  DM_INLINE uint32_t lane() const {
    return tid % waveSize();
  }
  /// Wave rank within the team, forced into an SGPR so DMA descriptors built
  /// from it stay scalar.
  DM_INLINE uint32_t waveRank() const {
    return __builtin_amdgcn_readfirstlane(tid / waveSize());
  }
  DM_INLINE uint32_t nWaves() const {
    return nThreads / waveSize();
  }
  DM_INLINE bool wholeWaves() const {
    return nThreads % waveSize() == 0 && (threadIdx.x - tid) % waveSize() == 0;
  }
};

// ============================================================================
//  Endpoints
// ============================================================================

/// Up to MaxN pointers, all in memory space S, with a runtime count. Addresses
/// are generic (flat) addresses; backends narrow them to the right address
/// space. The count and every address must be uniform across the team.
template <Space S, int MaxN>
struct MemSet {
  static_assert(S != Space::Reg, "registers are described by RegTile, not MemSet");
  static_assert(MaxN >= 1, "a MemSet holds at least one pointer");
  static constexpr Space space = S;
  static constexpr int kMax = MaxN;

  uintptr_t addr[MaxN];
  int n;

  DM_INLINE int count() const {
    return n;
  }
};

template <int MaxN>
using GlobalSet = MemSet<Space::Global, MaxN>;
template <int MaxN>
using LdsSet = MemSet<Space::Lds, MaxN>;

/// Per-thread register storage for N tiles of NPacks 16-byte packs each.
///
/// Layout is fixed by the team that fills it: byte b of tile c lives in
///   thread (b / 16) % T, pack (b / 16) / T, byte b % 16      (T = team size)
/// i.e. pack i of every thread together covers the contiguous T*16 bytes
/// starting at i*T*16, which keeps every global/LDS access fully coalesced.
/// A tile holds up to capacity(team) = NPacks * 16 * T bytes.
///
/// All indexing into v[][] is compile-time, so the tile stays in VGPRs.
template <int NPacks, int N = 1>
struct RegTile {
  static_assert(NPacks >= 1 && N >= 1, "a RegTile holds at least one pack");
  static constexpr Space space = Space::Reg;
  static constexpr int kMax = N;
  static constexpr int kPacks = NPacks;

  u32x4 v[N][NPacks];

  DM_INLINE constexpr int count() const {
    return N;
  }
  DM_INLINE static size_t capacity(const Team& t) {
    return static_cast<size_t>(NPacks) * kPackBytes * t.nThreads;
  }
  /// Byte offset, within the tile, of this thread's pack i.
  DM_INLINE static size_t packOffset(const Team& t, int i) {
    return (static_cast<size_t>(i) * t.nThreads + t.tid) * kPackBytes;
  }
};

template <class E>
using Endpoint = std::remove_cv_t<std::remove_reference_t<E>>;

// ---- endpoint factories ----------------------------------------------------

/// global(p0, p1, ...): a fixed set of global-memory pointers.
template <class... P>
DM_INLINE GlobalSet<sizeof...(P)> global(P*... p) {
  return {{reinterpret_cast<uintptr_t>(p)...}, static_cast<int>(sizeof...(P))};
}

/// lds(p0, p1, ...): a fixed set of pointers into this block's LDS.
template <class... P>
DM_INLINE LdsSet<sizeof...(P)> lds(P*... p) {
  return {{reinterpret_cast<uintptr_t>(p)...}, static_cast<int>(sizeof...(P))};
}

/// A runtime-sized set of n <= MaxN pointers in space S, e.g. the peer buffers
/// of a ring step.
template <Space S, int MaxN, class P>
DM_INLINE MemSet<S, MaxN> memSet(P* const* ptrs, int n) {
  MemSet<S, MaxN> m;
  m.n = n;
#pragma unroll
  for (int i = 0; i < MaxN; ++i) m.addr[i] = i < n ? reinterpret_cast<uintptr_t>(ptrs[i]) : 0;
  return m;
}

namespace detail {

template <class>
inline constexpr bool kAlwaysFalse = false;

/// Fan-out F = D/S, trapping on a shape that violates the D % S == 0 rule.
template <class Src, class Dst>
DM_INLINE int fanOut(const Src& src, const Dst& dst) {
  const int ns = src.count();
  const int nd = dst.count();
  if (ns <= 0 || nd % ns != 0) __builtin_trap();
  return nd / ns;
}

/// addr[i] for a runtime i, as a select chain so the set stays in registers.
template <Space S, int M>
DM_INLINE uintptr_t pick(const MemSet<S, M>& m, int i) {
  uintptr_t r = m.addr[0];
#pragma unroll
  for (int k = 1; k < M; ++k)
    if (k == i) r = m.addr[k];
  return r;
}

/// True when destination d belongs to source s under fan-out `fan`.
DM_INLINE bool fedBy(int d, int s, int fan) {
  return d >= s * fan && d < (s + 1) * fan;
}

}  // namespace detail

}  // namespace dm
}  // namespace rccl
