/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/// \file sdmaMover.h
/// \brief Data mover that posts device-initiated SDMA copies to an Anvil queue.
///
/// Handles Global -> Global only (the SDMA engine cannot see LDS or VGPRs), and
/// only for transfers of at least `minBytes`, below which posting a packet costs
/// more than a vector copy; smaller moves fall through the DataMover chain.
///
/// Team rank 0 posts one linear-copy packet per (source, destination) pair,
/// split into gin_sdma::kGinPutSegBytes segments; the other members do nothing.
/// wait() has rank 0 drain the queue. Like the GIN Anvil backend, the queue
/// handle is a device pointer the host set up (one queue per peer/channel).
///
/// \note The engine reads sources from memory, not from the team's caches:
///       barrier the team before issue() if other members just wrote a source.
///       After wait(), barrier again before other members read a destination.
#pragma once

#include "dmCore.h"

#if __has_include("sdma/anvil_device.hpp") && __has_include("nccl_device/gin/anvil_sdma/gin_anvil_sdma_put_policy.h")
#include "sdma/anvil_device.hpp"
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_put_policy.h"
#define DM_SDMA_SUPPORTED 1
#else
#define DM_SDMA_SUPPORTED 0
#endif

namespace rccl {
namespace dm {

struct SdmaMover {
  static constexpr bool kSupported = DM_SDMA_SUPPORTED;

#if DM_SDMA_SUPPORTED
  using Queue = ::sdma_anvil::SdmaQueueDeviceHandle;
#else
  struct Queue;
#endif

  Queue* queue = nullptr;
  size_t minBytes = 0;
  bool pending = false;

  __host__ __device__ SdmaMover() {}
  __host__ __device__ SdmaMover(Queue* q, size_t minSdmaBytes) : queue(q), minBytes(minSdmaBytes) {}

  template <Space S, Space D>
  static constexpr bool handles() {
    return S == Space::Global && D == Space::Global;
  }

  template <Space S, Space D>
  DM_INLINE bool accepts(const Team&, size_t bytes) const {
    return queue != nullptr && bytes >= minBytes;
  }

  template <class Src, class Dst>
  DM_INLINE void issue([[maybe_unused]] const Team& t, [[maybe_unused]] Src& src, [[maybe_unused]] Dst& dst,
                        [[maybe_unused]] size_t bytes) {
    using S = Endpoint<Src>;
    using D = Endpoint<Dst>;
    static_assert(kSupported, "SDMA mover needs sdma/anvil_device.hpp (ENABLE_ROCSHMEM_GIN builds)");
    static_assert(handles<S::space, D::space>(), "SdmaMover moves only Global->Global");
#if DM_SDMA_SUPPORTED
    const int fan = detail::fanOut(src, dst);
    pending = true;
    if (t.tid != 0) return;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    const size_t nSeg = gin_sdma::ginPutSegmentCount(bytes, gin_sdma::kGinPutSegBytes);
#pragma unroll
    for (int s = 0; s < S::kMax; ++s) {
      if (s >= src.n) break;
#pragma unroll
      for (int d = 0; d < D::kMax; ++d) {
        if (!detail::fedBy(d, s, fan)) continue;
        for (size_t i = 0; i < nSeg; ++i) {
          const gin_sdma::PutSegment seg = gin_sdma::ginPutSegmentAt(bytes, gin_sdma::kGinPutSegBytes, i);
          if (seg.bytes == 0) continue;
          ::sdma_anvil::put(*queue, reinterpret_cast<void*>(dst.addr[d] + seg.offset),
                            reinterpret_cast<void*>(src.addr[s] + seg.offset), seg.bytes);
        }
      }
    }
#endif
  }

  DM_INLINE void wait(const Team& t) {
#if DM_SDMA_SUPPORTED
    if (!pending) return;
    pending = false;
    if (t.tid != 0) return;
    ::sdma_anvil::quiet(*queue);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#else
    (void)t;
#endif
  }
};

}  // namespace dm
}  // namespace rccl
