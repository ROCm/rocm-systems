/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/// \file ldsDmaMover.h
/// \brief Data movers built on the per-wave global<->LDS DMA engines of gfx1250:
///        the Tensor Data Mover (tensor_load_to_lds / tensor_store_from_lds) and
///        the async-to/from-LDS loads and stores (global_load_async_to_lds_*).
///
/// Both engines move global <-> LDS without touching VGPRs, so they handle:
///   * Global -> Lds and Lds -> Global directly, and
///   * Global -> Global staged through a caller-provided LDS window.
/// Anything touching registers, and Lds -> Lds, is not theirs; a DataMover chain
/// routes those to the next backend (normally VectorMover).
///
/// Work is split across the team's waves in contiguous 256-byte-granular slices;
/// each wave drives its own engine and its own counter, so wait() drains only
/// the calling wave's operations. Every member wave must call wait().
///
/// Global -> Global double-buffers each wave's window: while the stores of
/// chunk j drain from one half, the load of chunk j+1 fills the other, and a
/// single counter-zero wait per chunk covers both the RAW edge (load j+1 before
/// its stores) and the WAR edge (stores j before that half is reloaded). A
/// source's chunk is loaded once and stored to every one of its destinations.
#pragma once

#include "dmCore.h"
#include "../tdm/tdmCopy.h"
#include "../tdm/asyncCopy.h"

namespace rccl {
namespace dm {

// ---- engines: one global<->LDS leg per call, issued by a whole wave --------

struct TdmEngine {
  static constexpr bool kSupported = TDM_SUPPORTED;

  template <CachePolicy cp>
  DM_INLINE static void load([[maybe_unused]] uintptr_t global, [[maybe_unused]] uintptr_t lds,
                              [[maybe_unused]] size_t n) {
#if TDM_SUPPORTED
    tdm::asyncLoadToLDS<SyncPolicy::Async, cp>(reinterpret_cast<const uint8_t*>(global),
                                               reinterpret_cast<uint8_t*>(lds), n);
#endif
  }
  template <CachePolicy cp>
  DM_INLINE static void store([[maybe_unused]] uintptr_t lds, [[maybe_unused]] uintptr_t global,
                               [[maybe_unused]] size_t n) {
#if TDM_SUPPORTED
    tdm::asyncStoreFromLDS<SyncPolicy::Async, cp>(reinterpret_cast<const uint8_t*>(lds),
                                                  reinterpret_cast<uint8_t*>(global), n);
#endif
  }
  DM_INLINE static void wait() {
#if TDM_SUPPORTED
    tdm::tdmWait();
#endif
  }
};

struct AsyncLdsEngine {
  static constexpr bool kSupported = ASYNC_COPY_SUPPORTED;

  template <CachePolicy cp>
  DM_INLINE static void load([[maybe_unused]] uintptr_t global, [[maybe_unused]] uintptr_t lds,
                              [[maybe_unused]] size_t n) {
#if ASYNC_COPY_SUPPORTED
    ::asyncLoadToLDS<SyncPolicy::Async, cp>(reinterpret_cast<const uint8_t*>(global),
                                            reinterpret_cast<uint8_t*>(lds), n);
#endif
  }
  template <CachePolicy cp>
  DM_INLINE static void store([[maybe_unused]] uintptr_t lds, [[maybe_unused]] uintptr_t global,
                               [[maybe_unused]] size_t n) {
#if ASYNC_COPY_SUPPORTED
    ::asyncStoreFromLDS<SyncPolicy::Async, cp>(reinterpret_cast<const uint8_t*>(lds),
                                               reinterpret_cast<uint8_t*>(global), n);
#endif
  }
  DM_INLINE static void wait() {
#if ASYNC_COPY_SUPPORTED
    asyncWait<0>();
#endif
  }
};

/// \tparam Engine TdmEngine or AsyncLdsEngine.
/// \tparam cp     Cache policy baked into every engine instruction.
template <class Engine, CachePolicy cp = DEFAULT_CACHE_POLICY>
struct LdsDmaMover {
  static constexpr bool kSupported = Engine::kSupported;
  /// Slice granularity across waves, and half-window granularity for staging.
  static constexpr uint32_t kGrain = 256;

  /// LDS staging area for Global -> Global, shared by the team's waves. May be
  /// null when only Global <-> Lds moves are issued.
  uintptr_t staging = 0;
  uint32_t stagingBytes = 0;

  __host__ __device__ LdsDmaMover() {}
  __host__ __device__ LdsDmaMover(void* stagingLds, uint32_t stagingLdsBytes)
    : staging(reinterpret_cast<uintptr_t>(stagingLds)), stagingBytes(stagingLdsBytes) {}

  template <Space S, Space D>
  static constexpr bool handles() {
    return (S == Space::Global && (D == Space::Lds || D == Space::Global)) ||
           (S == Space::Lds && D == Space::Global);
  }

  template <Space S, Space D>
  DM_INLINE bool accepts(const Team& t, size_t) const {
    if (!t.wholeWaves()) return false;
    if constexpr (S == Space::Global && D == Space::Global) return windowBytes(t) >= 2 * kGrain;
    else return true;
  }

  template <class Src, class Dst>
  DM_INLINE void issue(const Team& t, Src& src, Dst& dst, size_t bytes) {
    using S = Endpoint<Src>;
    using D = Endpoint<Dst>;
    static_assert(Engine::kSupported, "this LDS DMA engine is not available for the target being compiled");
    static_assert(handles<S::space, D::space>(), "LdsDmaMover moves only Global<->Lds and Global->Global");
    const int fan = detail::fanOut(src, dst);

    size_t lo, n;
    waveSlice(t, bytes, lo, n);
    if (n == 0) return;

    if constexpr (D::space == Space::Global && S::space == Space::Global) {
      stagedCopy(t, src, dst, fan, lo, n);
    } else {
#pragma unroll
      for (int s = 0; s < S::kMax; ++s) {
        if (s >= src.n) break;
#pragma unroll
        for (int d = 0; d < D::kMax; ++d) {
          if (!detail::fedBy(d, s, fan)) continue;
          if constexpr (S::space == Space::Global) Engine::template load<cp>(src.addr[s] + lo, dst.addr[d] + lo, n);
          else Engine::template store<cp>(src.addr[s] + lo, dst.addr[d] + lo, n);
        }
      }
    }
  }

  DM_INLINE void wait(const Team&) {
    Engine::wait();
  }

private:
  // Per-wave staging window: a multiple of two half-windows of kGrain bytes.
  DM_INLINE uint32_t windowBytes(const Team& t) const {
    return (stagingBytes / t.nWaves()) & ~(2 * kGrain - 1);
  }

  DM_INLINE static void waveSlice(const Team& t, size_t bytes, size_t& lo, size_t& n) {
    const size_t nw = t.nWaves();
    const size_t slice = ((bytes + nw - 1) / nw + kGrain - 1) & ~static_cast<size_t>(kGrain - 1);
    lo = static_cast<size_t>(t.waveRank()) * slice;
    if (lo > bytes) lo = bytes;
    n = bytes - lo < slice ? bytes - lo : slice;
  }

  template <Space SS, int MS, Space DS, int MD>
  DM_INLINE void stagedCopy(const Team& t, const MemSet<SS, MS>& src, const MemSet<DS, MD>& dst, int fan,
                             size_t lo, size_t n) {
    const uint32_t half = windowBytes(t) / 2;
    const uintptr_t window = staging + static_cast<uintptr_t>(t.waveRank()) * 2 * half;
    const size_t nChunks = (n + half - 1) / half;
    const size_t nItems = static_cast<size_t>(src.n) * nChunks;

    // Item j is chunk (j % nChunks) of source (j / nChunks), staged in half j & 1.
    auto item = [&](size_t j, uintptr_t& buf, int& s, size_t& off, size_t& len) {
      s = static_cast<int>(j / nChunks);
      off = lo + (j % nChunks) * half;
      len = lo + n - off < half ? lo + n - off : half;
      buf = window + (j & 1) * half;
    };

    uintptr_t buf;
    int s;
    size_t off, len;
    // An earlier un-waited staged copy may still be storing out of this window.
    Engine::wait();
    item(0, buf, s, off, len);
    Engine::template load<cp>(detail::pick(src, s) + off, buf, len);
    for (size_t j = 0; j < nItems; ++j) {
      Engine::wait();
      item(j, buf, s, off, len);
#pragma unroll
      for (int d = 0; d < MD; ++d)
        if (detail::fedBy(d, s, fan)) Engine::template store<cp>(buf, dst.addr[d] + off, len);
      if (j + 1 < nItems) {
        item(j + 1, buf, s, off, len);
        Engine::template load<cp>(detail::pick(src, s) + off, buf, len);
      }
    }
  }
};

template <CachePolicy cp = DEFAULT_CACHE_POLICY>
using TdmMover = LdsDmaMover<TdmEngine, cp>;
template <CachePolicy cp = DEFAULT_CACHE_POLICY>
using AsyncLdsMover = LdsDmaMover<AsyncLdsEngine, cp>;

}  // namespace dm
}  // namespace rccl
