/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Single-GPU ncclGinBarrierSession Put-fence ordering over stub GPI leaves: no thread may signal a peer while
// another thread of the CTA is still flushing (NCCL 2.32.6 "GIN barrier bugfix: add coop sync between flush and signal").

#ifdef __CLANG_RDC__
#error "GinBarrierFlushOrderTests.cpp must stay non-RDC: its stub specializations would ODR-merge with other TUs"
#endif

// Compile only GPI, whose leaves are stubbed below; skip gin_device_api.h since the real backend headers are CUDA-only.
#undef NCCL_GIN_PROXY_ENABLE
#define NCCL_GIN_PROXY_ENABLE 0
#undef NCCL_GIN_GDAKI_ENABLE
#define NCCL_GIN_GDAKI_ENABLE 0
#undef NCCL_GIN_ROCSHMEM_GDA_ENABLE
#define NCCL_GIN_ROCSHMEM_GDA_ENABLE 0
#undef NCCL_GIN_ANVIL_SDMA_ENABLE
#define NCCL_GIN_ANVIL_SDMA_ENABLE 0
#undef NCCL_GIN_EFA_GDA_ENABLE
#define NCCL_GIN_EFA_GDA_ENABLE 0
#undef NCCL_GIN_GPI_ENABLE
#define NCCL_GIN_GPI_ENABLE 1
#define _NCCL_GIN_DEVICE_API_H_

#include <cstdint>

#include "DeviceTestBase.hpp"

#include "nccl_device/coop.h"
#include "nccl_device/gin/gin_device_common.h"
#include "nccl_device/gin/gin_device_host_common.h"

namespace RcclUnitTesting {
namespace {

// ctx.handle points here. Thread 0's flush is held back so the other wavefronts reach the signal loop first.
struct BarrierRecord {
  unsigned int flushesDone;
  unsigned int expectedFlushes;
  unsigned int signals;
  unsigned int earlySignals;  // Signals posted while flushesDone < expectedFlushes.
  uint64_t signalWord;        // Every barrier signal reads as already arrived.
};

NCCL_DEVICE_INLINE void finishFlush(BarrierRecord* rec) {
  if (threadIdx.x == 0) {
    uint64_t t0 = wall_clock64();
    while (wall_clock64() - t0 < 2000000) __builtin_amdgcn_s_sleep(8);  // ~20 ms at the 100 MHz wall clock.
  }
  atomicAdd(&rec->flushesDone, 1u);
}

}  // namespace
}  // namespace RcclUnitTesting

using RcclUnitTesting::BarrierRecord;

template <>
struct ncclGinApi_FlushAsync<NCCL_NET_DEVICE_GIN_GPI> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, uint32_t, ncclGinRequest_t* outRequest, bool, ncclGinDescriptorSmem*,
                                      uint32_t) {
    *outRequest = ncclGinRequest_t{};
  }
};
template <>
struct ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_GPI> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinRequest_t&, bool, ncclGinDescriptorSmem*,
                                      cuda::memory_order, uint32_t*) {
    RcclUnitTesting::finishFlush(static_cast<BarrierRecord*>(ctx.handle));
  }
};
template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_GPI> {
  // The one-context fence: the whole CTA calls in, thread 0 completes the single flush.
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop, bool, ncclGinDescriptorSmem*, cuda::memory_order,
                                      uint32_t*) {
    if (threadIdx.x == 0) RcclUnitTesting::finishFlush(static_cast<BarrierRecord*>(ctx.handle));
  }
};
template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_GPI> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop, int, bool, ncclGinWindow_t, size_t, ncclGinWindow_t,
                                      size_t, size_t, ncclGinSignalDescriptor, ncclGinSignalOp_t, uint64_t, bool,
                                      ncclGinCounter_t, bool, ncclGinDescriptorSmem*, cuda::thread_scope,
                                      cuda::thread_scope, uint32_t = ncclGinOptFlagsDefault) {
    BarrierRecord* rec = static_cast<BarrierRecord*>(ctx.handle);
    atomicAdd(&rec->signals, 1u);
    if (atomicAdd(&rec->flushesDone, 0u) < rec->expectedFlushes) atomicAdd(&rec->earlySignals, 1u);
  }
};
template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_GPI> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, uint32_t) {
    return {&static_cast<BarrierRecord*>(ctx.handle)->signalWord, 0};
  }
};
template <>
struct ncclGinApi_SupportsStrongSignal<NCCL_NET_DEVICE_GIN_GPI> {
  NCCL_DEVICE_INLINE static bool call(ncclGinCtx) { return false; }  // The Put fence must flush before signalling.
};
template <>
struct ncclGinApi_BarrierOptions<NCCL_NET_DEVICE_GIN_GPI> {
  NCCL_DEVICE_INLINE static ncclGinBarrierOptions_t call(ncclGinCtx) { return NCCL_GIN_BARRIER_DEFAULT; }
};

#include "nccl_device/impl/core__funcs.h"
#include "nccl_device/impl/gin__funcs.h"
#include "nccl_device/impl/gin_barrier__funcs.h"

namespace RcclUnitTesting {
namespace {

constexpr int kThreads = 256;

template <bool kAllContexts>
__global__ void kernelPutBarrier(ncclDevComm comm) {
  ncclGinBarrierHandle handle{};
  if (kAllContexts) {
    ncclGinBarrierSession<ncclCoopCta> bar(ncclCoopCta{}, ncclGinAllContexts(comm), ncclTeamWorld(comm), handle, 0);
    bar.sync(ncclCoopCta{}, cuda::memory_order_relaxed, ncclGinFenceLevel::Put);
  } else {
    ncclGinBarrierSession<ncclCoopCta> bar(ncclCoopCta{}, ncclGin(comm, 0), ncclTeamWorld(comm), handle, 0);
    bar.sync(ncclCoopCta{}, cuda::memory_order_relaxed, ncclGinFenceLevel::Put);
  }
}

class GinBarrierFlushOrderTest : public DeviceTestBase {
protected:
  // nCtx contexts x nRanks peers: the all-contexts fence gives one (ctx, peer) flush and signal to each thread.
  BarrierRecord run(bool allContexts, int nRanks, int nCtx, unsigned int expectedFlushes) {
    DeviceBuffer<BarrierRecord> rec(1);
    DeviceBuffer<uint64_t> shadows(nCtx * nRanks);
    DeviceBuffer<uint32_t> abortFlag(1);
    rec.zero();
    shadows.zero();
    abortFlag.zero();
    BarrierRecord init{};
    init.expectedFlushes = expectedFlushes;
    init.signalWord = 1;
    rec.upload(init);
    ncclDevComm comm{};
    comm.rank = 0;
    comm.nRanks = nRanks;
    comm.ginConnectionCount = 1;
    comm.ginConnectionStride = 1;
    comm.ginNetDeviceTypes[0] = NCCL_NET_DEVICE_GIN_GPI;
    comm.ginHandles[0] = rec.ptr;
    comm.ginSignalCount = nRanks;
    comm.ginSignalShadows = shadows.ptr;
    comm.ginContextCount = nCtx;
    comm.abortFlag = abortFlag.ptr;
    if (allContexts) {
      kernelPutBarrier<true><<<1, kThreads>>>(comm);
    } else {
      kernelPutBarrier<false><<<1, kThreads>>>(comm);
    }
    syncAndCheck();
    return rec.copyTo()[0];
  }
};

// The all-contexts fence flushes (ctx, peer) pairs one per thread; its coop.sync must hold every signal until all
// of them, including thread 0's slow one, have completed.
TEST_F(GinBarrierFlushOrderTest, AllContextsPutFenceSignalsOnlyAfterEveryFlush) {
  const BarrierRecord r = run(/*allContexts=*/true, /*nRanks=*/16, /*nCtx=*/16, /*expectedFlushes=*/kThreads);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(r.flushesDone, static_cast<unsigned int>(kThreads));
  EXPECT_EQ(r.signals, static_cast<unsigned int>(kThreads));
  EXPECT_EQ(r.earlySignals, 0u) << "threads signalled peers while thread 0 was still flushing";
}

// Control: the one-context fence goes through ncclGin::flush, whose own coop.sync orders it; same stubs and spread.
TEST_F(GinBarrierFlushOrderTest, ControlOneContextPutFenceSignalsOnlyAfterFlush) {
  const BarrierRecord r = run(/*allContexts=*/false, /*nRanks=*/kThreads, /*nCtx=*/1, /*expectedFlushes=*/1);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(r.flushesDone, 1u);
  EXPECT_EQ(r.signals, static_cast<unsigned int>(kThreads));
  EXPECT_EQ(r.earlySignals, 0u);
}

}  // namespace
}  // namespace RcclUnitTesting
