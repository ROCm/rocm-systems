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
/// split into gin_sdma::kGinPutSegBytes segments, and records a completion
/// ticket; the other members do nothing. wait() has rank 0 poll that ticket.
///
/// Queue
///   One SdmaMover targets one destination device: pass the queue for
///   (this device -> the device that owns the destination memory). Anvil keeps
///   one set of queues per destination device for the whole process and the
///   GIN Anvil backend uses the same queues, so:
///     * use the multi-producer handle (SdmaQueue::deviceHandle()) only. The
///       single-producer handle shares the ring but never advances the commit
///       pointer the multi-producer path serializes on, so mixing them on one
///       queue stalls every multi-producer submitter;
///     * packets from other users of the queue execute in commit order with
///       ours, so wait() may also wait for their earlier work (latency only).
///   Inside a GIN kernel, take the handle from the GIN Anvil SDMA context, which
///   already connected a multi-producer queue per peer and channel:
///     namespace ga = nccl::gin::anvil::detail;
///     dm::SdmaMover sdma(ga::queueHandle(rsCtx, peer, blockIdx.x), minBytes, abortFlag);
///   (ga::channelHandle(rsCtx, peer, ch) picks a channel explicitly.)
///
/// Completion
///   issue() reads quietTarget() after rank 0's last put: the queue's highest
///   committed write index, which covers that thread's own packets. wait() polls
///   that ticket with GIN's waitFlushed() instead of quiet(), so it
///   waits only up to this mover's work as of its issue (not for packets
///   committed later) and can observe an abort flag. It then issues a
///   system-scope acquire, because isFlushed() is a relaxed load and the
///   destination was written by an agent outside the CU (RCCL needs a
///   system-scope acquire after copy-engine publication on gfx950, see
///   prims_simple.h).
///
/// Assumptions (to be validated on hardware)
///   * The hardware read pointer passing the end of a copy packet means the
///     copy's writes are complete; rocSHMEM describes isFlushed() as "HW has
///     consumed packets up to upToIndex". RCCL's GIN Anvil path relies on the
///     same reading.
///   * An agent-scope release by rank 0 before the put makes sources written by
///     this GPU's CUs visible to the engine, as in RCCL's GIN Anvil put path.
///
/// \note The engine reads sources from memory, not from the team's caches:
///       barrier the team before issue() if other members just wrote a source.
///       After wait(), barrier again before other members read a destination.
#pragma once

#include "dmCore.h"

// Enabled exactly when the GIN Anvil backend is (same rule as
// gin_device_common.h). Built on the GIN Anvil device helpers, so
// "sdma/anvil_device.hpp" resolves the same way it does for GIN: the rocSHMEM
// copy in RCCL builds, the test stub in the fixtures build.
#if __has_include("nccl_device/gin/rccl_gin_config.h")
#include "nccl_device/gin/rccl_gin_config.h"
#endif
#if defined(NCCL_GIN_ANVIL_SDMA_ENABLE)
#define DM_SDMA_SUPPORTED (NCCL_GIN_ANVIL_SDMA_ENABLE)
#elif defined(NCCL_GIN_ANVIL_SDMA_AVAILABLE) && defined(__HIP_PLATFORM_AMD__) && __cplusplus >= 201703L
#define DM_SDMA_SUPPORTED (NCCL_GIN_ANVIL_SDMA_AVAILABLE)
#else
#define DM_SDMA_SUPPORTED 0
#endif
#if DM_SDMA_SUPPORTED
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma.h"
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
  /// Optional communicator abort flag; wait() stops polling once it is set
  /// (checked every few thousand polls, as GIN does).
  uint32_t* abortFlag = nullptr;
  /// Completion ticket, meaningful on team rank 0: the queue write index that
  /// covers every packet this mover has issued and not yet waited for.
  uint64_t target = 0;
  bool pending = false;
  bool aborted = false;

  __host__ __device__ SdmaMover() {}
  __host__ __device__ SdmaMover(Queue* q, size_t minSdmaBytes, uint32_t* abort = nullptr)
    : queue(q), minBytes(minSdmaBytes), abortFlag(abort) {}

  template <Space S, Space D>
  static constexpr bool handles() {
    return S == Space::Global && D == Space::Global;
  }

  template <Space S, Space D, class Src, class Dst>
  DM_INLINE bool accepts(const Team&, const Src&, const Dst&, size_t bytes) const {
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
    // Read after this thread's own submissions, so the ticket covers them.
    const uint64_t t1 = ::sdma_anvil::quietTarget(*queue);
    if (t1 > target) target = t1;
#endif
  }

  DM_INLINE void wait(const Team& t) {
#if DM_SDMA_SUPPORTED
    if (!pending) return;
    pending = false;
    if (t.tid != 0) return;
    // Same poll as GIN's flush; it returns on abort without saying so, so a
    // ticket still outstanding afterwards means the wait was aborted.
    (void)::nccl::gin::anvil::detail::waitFlushed</*HasTimeout=*/false>(queue, target, abortFlag, 0, 0);
    aborted = !::sdma_anvil::isFlushed(*queue, target);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
#else
    (void)t;
#endif
  }

  /// True on team rank 0 if a wait() stopped because the abort flag was set; the
  /// destinations of the copies it was waiting for are then undefined.
  DM_INLINE bool wasAborted() const {
    return aborted;
  }
};

}  // namespace dm
}  // namespace rccl
