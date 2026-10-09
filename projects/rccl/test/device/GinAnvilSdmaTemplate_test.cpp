/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Suite H: device template coverage (Put/PutValue/Get/Flush/FlushAsync/Wait/Signal/Counter SDMA, timeouts + edge paths).
// Uses test/device/sdma/anvil_device.hpp stubs (no librocshmem device link).

#include "DeviceTestBase.hpp"

#include "nccl_device/coop.h"
#include "nccl_device/gin/gin_device_host_common.h"
#include "nccl_device/gin/gin_device_common.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h"

#if NCCL_GIN_ANVIL_SDMA_ENABLE
// Count invocations of the Put/PutValue system-scope fence seam (gin_device_common.h).
// Override must precede gin_anvil_sdma.h so the templates expand our counter.
__device__ unsigned long long g_sdmaStubThreadfenceCount = 0;
#undef NCCL_GIN_THREADFENCE_SYSTEM
#define NCCL_GIN_THREADFENCE_SYSTEM() atomicAdd(&g_sdmaStubThreadfenceCount, 1ULL)
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma.h"
#endif

#include <algorithm>
#include <cstring>
#include <vector>

namespace RcclUnitTesting
{

#if NCCL_GIN_ANVIL_SDMA_ENABLE

class GinAnvilSdmaTemplateTest : public DeviceTestBase {};

struct TemplateHarness {
  ncclGinAnvilSdmaGPUContext ctx;
  ncclGinAnvilSdmaMemHandle dstMh;
  ncclGinAnvilSdmaMemHandle srcMh;
  ncclGinAnvilIpcBufEntry ipcEntry;
};

static void uploadHarness(DeviceBuffer<TemplateHarness>* d_h, TemplateHarness* host,
                          DeviceBuffer<uint8_t>* d_src, DeviceBuffer<uint8_t>* d_dst,
                          DeviceBuffer<ncclGinAnvilIpcBufEntry>* d_entry,
                          DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle>* d_q,
                          DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*>* d_handleRow,
                          int threshold) {
  std::memset(host, 0, sizeof(*host));
  host->ctx.layoutMagic = NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC;
  host->ctx.sdmaThreshold = static_cast<uint32_t>(threshold);
  host->ctx.numChannels = 1;
  host->ctx.nRanks = 2;
  host->ctx.rank = 0;
  host->ctx.fusedSdmaSignal = 1;

  sdma_anvil::SdmaQueueDeviceHandle stub{};
  stub.tag = 42;
  d_q->upload(stub);
  sdma_anvil::SdmaQueueDeviceHandle* rowHost[2] = {d_q->ptr, d_q->ptr};
  d_handleRow->copyFrom(rowHost, 2);
  host->ctx.queueHandles = reinterpret_cast<void**>(d_handleRow->ptr);

  host->ipcEntry.local_base = reinterpret_cast<uintptr_t>(d_dst->ptr);
  host->ipcEntry.length = 4096;
  host->ipcEntry.remote_bases[1] = reinterpret_cast<uintptr_t>(d_dst->ptr);
  d_entry->upload(host->ipcEntry);
  host->ctx.ipcTable = d_entry->ptr;
  host->ctx.ipcTableCount = 1;

  host->dstMh.baseAddr = reinterpret_cast<uintptr_t>(d_dst->ptr);
  host->srcMh.baseAddr = reinterpret_cast<uintptr_t>(d_src->ptr);
  d_h->upload(*host);
}

static void resetThreadfenceCount() {
  unsigned long long z = 0;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_sdmaStubThreadfenceCount), &z, sizeof(z)));
}

static unsigned long long readThreadfenceCount() {
  unsigned long long c = 0;
  HIP_EXPECT(hipMemcpyFromSymbol(&c, HIP_SYMBOL(g_sdmaStubThreadfenceCount), sizeof(c)));
  return c;
}

static void resetQuietCount() {
  unsigned long long z = 0;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(sdma_anvil::g_sdmaStubQuietCount), &z, sizeof(z)));
}

static unsigned long long readQuietCount() {
  unsigned long long c = 0;
  HIP_EXPECT(hipMemcpyFromSymbol(&c, HIP_SYMBOL(sdma_anvil::g_sdmaStubQuietCount), sizeof(c)));
  return c;
}

// H1: non-leader thread returns immediately.
__global__ void kernelPutLeaderOnly(TemplateHarness* h, int* executed) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(
      ginCtx, ncclCoopThread{}, 1, true, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, 16, sig, ncclGinSignalInc, 0, false, 0,
      false, nullptr, cuda::thread_scope_system, cuda::thread_scope_system);
  if (threadIdx.x == 0) executed[0] = 1;
}

TEST_F(GinAnvilSdmaTemplateTest, Put_NonLeaderThreadNoOp) {
  DeviceBuffer<uint8_t> d_src(16);
  DeviceBuffer<uint8_t> d_dst(16);
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_src, &d_dst, &d_entry, &d_q, &d_row, 128);
  DeviceBuffer<int> d_executed(1);
  d_executed.zero();
  kernelPutLeaderOnly<<<1, 4>>>(d_h.ptr, d_executed.ptr);
  syncAndCheck();
  EXPECT_EQ(d_executed.download(), 1);
}

// H2: system fence when required==system && given<required (HIP scope ordering).
__global__ void kernelPutScopeFence(TemplateHarness* h) {
  if (threadIdx.x != 0) return;
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(
      ginCtx, ncclCoopThread{}, 1, true, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, 8, sig, ncclGinSignalInc, 0, false, 0,
      false, nullptr, cuda::thread_scope_system, cuda::thread_scope_block);
}

TEST_F(GinAnvilSdmaTemplateTest, Put_ThreadScopeFence) {
  DeviceBuffer<uint8_t> d_src(8);
  DeviceBuffer<uint8_t> d_dst(8);
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_src, &d_dst, &d_entry, &d_q, &d_row, 128);
  resetThreadfenceCount();
  kernelPutScopeFence<<<1, 1>>>(d_h.ptr);
  syncAndCheck();
  EXPECT_EQ(readThreadfenceCount(), 1ULL);
}

// H3: SDMA path (threshold 0) with stub put + markSdmaDirty.
__global__ void kernelPutSdmaPath(TemplateHarness* h, uint64_t* dirtyOut) {
  if (threadIdx.x != 0) return;
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(
      ginCtx, ncclCoopThread{}, 1, true, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, 256, sig, ncclGinSignalInc, 0, false, 0,
      false, nullptr, cuda::thread_scope_system, cuda::thread_scope_system);
  if (h->ctx.sdmaDirty) {
    dirtyOut[0] = __scoped_atomic_load_n(h->ctx.sdmaDirty, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  }
}

TEST_F(GinAnvilSdmaTemplateTest, Put_SdmaPathSetsDirty) {
  DeviceBuffer<uint8_t> d_src(256);
  DeviceBuffer<uint8_t> d_dst(256);
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  DeviceBuffer<uint64_t> d_dirty(1);
  d_dirty.zero();
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_src, &d_dst, &d_entry, &d_q, &d_row, 0);
  host.ctx.sdmaDirty = d_dirty.ptr;
  d_h.upload(host);
  kernelPutSdmaPath<<<1, 1>>>(d_h.ptr, d_dirty.ptr);
  syncAndCheck();
  EXPECT_NE(d_dirty.download(), 0ULL);
}

// H4: SDMA primary resolve fails → IPC fallback via sym lookup.
__global__ void kernelPutSdmaIpcFallback(TemplateHarness* h, uint8_t* dst, const uint8_t* src) {
  if (threadIdx.x != 0) return;
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  // dstMh base points at dst; ipc table maps peer 1 to dst for fallback.
  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(
      ginCtx, ncclCoopThread{}, 1, true, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, 256, sig, ncclGinSignalInc, 0, false, 0,
      false, nullptr, cuda::thread_scope_system, cuda::thread_scope_system);
}

TEST_F(GinAnvilSdmaTemplateTest, Put_SdmaFallbackIpcCopy) {
  constexpr int kN = 64;
  std::vector<uint8_t> pat(kN);
  for (int i = 0; i < kN; ++i) pat[static_cast<size_t>(i)] = static_cast<uint8_t>(0xC0 + i);
  DeviceBuffer<uint8_t> d_src(static_cast<size_t>(kN));
  DeviceBuffer<uint8_t> d_dst(static_cast<size_t>(kN));
  d_src.copyFrom(pat);
  d_dst.zero();
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_src, &d_dst, &d_entry, &d_q, &d_row, 0);
  kernelPutSdmaIpcFallback<<<1, 1>>>(d_h.ptr, d_dst.ptr, d_src.ptr);
  syncAndCheck();
  auto got = d_dst.copyTo();
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(got[static_cast<size_t>(i)], pat[static_cast<size_t>(i)]);
  }
}

// H5: signal + counter on IPC path (fenceBeforeSignal IPC branch).
__global__ void kernelPutSignalCounter(TemplateHarness* h) {
  if (threadIdx.x != 0) return;
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  sig.indexedSignal.signalId = 0;
  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(
      ginCtx, ncclCoopThread{}, 1, true, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, 32, sig, ncclGinSignalAdd, 7, true, 0,
      false, nullptr, cuda::thread_scope_system, cuda::thread_scope_system);
}

TEST_F(GinAnvilSdmaTemplateTest, Put_SignalAndCounterIpc) {
  DeviceBuffer<uint8_t> d_src(32);
  DeviceBuffer<uint8_t> d_dst(32);
  DeviceBuffer<uint64_t> d_counters(1);
  DeviceBuffer<uint64_t> d_signals(2);
  d_counters.zero();
  d_signals.zero();
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_src, &d_dst, &d_entry, &d_q, &d_row, 128);
  host.ctx.counters = d_counters.ptr;
  host.ctx.signals = d_signals.ptr;
  host.ctx.nSignals = 2;
  host.ctx.nCounters = 1;
  d_h.upload(host);
  kernelPutSignalCounter<<<1, 1>>>(d_h.ptr);
  syncAndCheck();
  EXPECT_EQ(d_counters.download(), 1ULL);
}

// H6: fused SDMA signal path when OSS7 + remote signal resolved.
__global__ void kernelPutFusedSignal(TemplateHarness* h, bool* usedFused) {
  if (threadIdx.x != 0) return;
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  sig.indexedSignal.signalId = 0;
  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(
      ginCtx, ncclCoopThread{}, 1, true, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, 512, sig, ncclGinSignalInc, 0, false, 0,
      false, nullptr, cuda::thread_scope_system, cuda::thread_scope_system);
  usedFused[0] = true;
}

TEST_F(GinAnvilSdmaTemplateTest, Put_FusedSdmaSignalPath) {
  DeviceBuffer<uint8_t> d_src(512);
  DeviceBuffer<uint8_t> d_dst(512);
  DeviceBuffer<uint64_t> d_signals(2);
  d_signals.zero();
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  DeviceBuffer<bool> d_fused(1);
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_src, &d_dst, &d_entry, &d_q, &d_row, 0);
  host.ctx.signals = d_signals.ptr;
  host.ctx.fusedSdmaSignal = 1;
  d_h.upload(host);
  kernelPutFusedSignal<<<1, 1>>>(d_h.ptr, d_fused.ptr);
  syncAndCheck();
  EXPECT_TRUE(d_fused.download());
}

// H7: PutValue SDMA scalar path.
__global__ void kernelPutValueSdma(TemplateHarness* h) {
  if (threadIdx.x != 0) return;
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(
      ginCtx, ncclCoopThread{}, 1, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      static_cast<uint64_t>(0xAABBCCDDEEFF0011ULL), sig, ncclGinSignalInc, 0, false, nullptr,
      cuda::thread_scope_system, cuda::thread_scope_system);
}

TEST_F(GinAnvilSdmaTemplateTest, PutValue_SdmaScalar) {
  DeviceBuffer<uint8_t> d_dst(8);
  d_dst.zero();
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_dst, &d_dst, &d_entry, &d_q, &d_row, 0);
  kernelPutValueSdma<<<1, 1>>>(d_h.ptr);
  syncAndCheck();
}

// H8: Flush quiet on dirty queue with non-null handle (stub quiet).
__global__ void kernelFlushQuiet(TemplateHarness* h, uint64_t* dirty) {
  h->ctx.sdmaDirty = dirty;
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, ncclCoopThread{}, false, nullptr,
                                                         cuda::memory_order_seq_cst, nullptr);
}

TEST_F(GinAnvilSdmaTemplateTest, Flush_QuietDirtyQueue) {
  DeviceBuffer<uint8_t> d_src(1);
  DeviceBuffer<uint8_t> d_dst(1);
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  DeviceBuffer<uint64_t> d_dirty(1);
  uint64_t one = 1;
  d_dirty.copyFrom(&one, 1);
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_src, &d_dst, &d_entry, &d_q, &d_row, 128);
  kernelFlushQuiet<<<1, 1>>>(d_h.ptr, d_dirty.ptr);
  syncAndCheck();
  EXPECT_EQ(d_dirty.download(), 0ULL);
}

// H9: getter / reset API specializations.
__global__ void kernelCounterSignalApi(TemplateHarness* h, uint64_t* outCtr, uint64_t* outSig) {
  if (threadIdx.x != 0) return;
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ncclGinOffsetPtr ctrOff = ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, 0);
  if (ctrOff.ptr) ctrOff.ptr[0] = 99;
  ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, 0);
  outCtr[0] = ctrOff.ptr ? ctrOff.ptr[0] : 0;
  ncclGinOffsetPtr sigOff = ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, 0);
  if (sigOff.ptr) sigOff.ptr[0] = 11;
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  desc.indexedSignal.signalId = 0;
  ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, desc);
  outSig[0] = sigOff.ptr ? sigOff.ptr[0] : 0;
}

TEST_F(GinAnvilSdmaTemplateTest, CounterSignal_GetReset) {
  DeviceBuffer<uint8_t> d_src(1);
  DeviceBuffer<uint8_t> d_dst(1);
  DeviceBuffer<uint64_t> d_counters(1);
  DeviceBuffer<uint64_t> d_signals(1);
  d_counters.zero();
  d_signals.zero();
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  DeviceBuffer<uint64_t> d_outCtr(1);
  DeviceBuffer<uint64_t> d_outSig(1);
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_src, &d_dst, &d_entry, &d_q, &d_row, 128);
  host.ctx.counters = d_counters.ptr;
  host.ctx.signals = d_signals.ptr;
  d_h.upload(host);
  kernelCounterSignalApi<<<1, 1>>>(d_h.ptr, d_outCtr.ptr, d_outSig.ptr);
  syncAndCheck();
  EXPECT_EQ(d_outCtr.download(), 0ULL);
  EXPECT_EQ(d_outSig.download(), 0ULL);
}

// H10: invalid ctx on getters returns nullptr / no-op.
__global__ void kernelInvalidCtxApis(bool* ok) {
  ncclGinCtx ginCtx{};
  ncclGinAnvilSdmaGPUContext bad{};
  bad.layoutMagic = 0;
  ginCtx.handle = &bad;
  ok[0] = ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, 0).ptr == nullptr;
  ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, 0);
  ok[1] = true;
}

TEST_F(GinAnvilSdmaTemplateTest, GetReset_InvalidCtx) {
  DeviceBuffer<bool> d_ok(2);
  d_ok.zero();
  kernelInvalidCtxApis<<<1, 1>>>(d_ok.ptr);
  syncAndCheck();
  auto ok = d_ok.copyTo();
  EXPECT_TRUE(ok[0]);
  EXPECT_TRUE(ok[1]);
}

// H11: SDMA path with counter (fenceBeforeSignal quiet branch).
__global__ void kernelPutSdmaCounter(TemplateHarness* h) {
  if (threadIdx.x != 0) return;
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(
      ginCtx, ncclCoopThread{}, 1, true, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, 512, sig, ncclGinSignalInc, 0, true, 0, false,
      nullptr, cuda::thread_scope_system, cuda::thread_scope_system);
}

TEST_F(GinAnvilSdmaTemplateTest, Put_SdmaCounterFence) {
  DeviceBuffer<uint8_t> d_src(512);
  DeviceBuffer<uint8_t> d_dst(512);
  DeviceBuffer<uint64_t> d_counters(1);
  d_counters.zero();
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_src, &d_dst, &d_entry, &d_q, &d_row, 0);
  host.ctx.counters = d_counters.ptr;
  d_h.upload(host);
  resetQuietCount();
  resetThreadfenceCount();
  kernelPutSdmaCounter<<<1, 1>>>(d_h.ptr);
  syncAndCheck();
  EXPECT_EQ(d_counters.download(), 1ULL);
  EXPECT_EQ(readQuietCount(), 1ULL);
  EXPECT_EQ(readThreadfenceCount(), 1ULL);
}

// H12: Flush clears multiple dirty channel bits across peers.
__global__ void kernelFlushMultiDirty(TemplateHarness* h, uint64_t* dirty) {
  h->ctx.sdmaDirty = dirty;
  h->ctx.numChannels = 2;
  h->ctx.nRanks = 2;
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, ncclCoopThread{}, false, nullptr,
                                                         cuda::memory_order_seq_cst, nullptr);
}

TEST_F(GinAnvilSdmaTemplateTest, Flush_MultiDirtyBits) {
  DeviceBuffer<uint8_t> d_src(1);
  DeviceBuffer<uint8_t> d_dst(1);
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(4);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(4);
  DeviceBuffer<TemplateHarness> d_h(1);
  DeviceBuffer<uint64_t> d_dirty(1);
  uint64_t mask = (1ULL << 0) | (1ULL << 1) | (1ULL << 2) | (1ULL << 3);
  d_dirty.copyFrom(&mask, 1);

  sdma_anvil::SdmaQueueDeviceHandle stub{};
  stub.tag = 7;
  d_q.upload(stub);
  sdma_anvil::SdmaQueueDeviceHandle* rowHost[4] = {d_q.ptr, d_q.ptr, d_q.ptr, d_q.ptr};
  d_row.copyFrom(rowHost, 4);

  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_src, &d_dst, &d_entry, &d_q, &d_row, 128);
  host.ctx.numChannels = 2;
  host.ctx.sdmaDirty = d_dirty.ptr;
  host.ctx.queueHandles = reinterpret_cast<void**>(d_row.ptr);
  d_h.upload(host);

  kernelFlushMultiDirty<<<1, 1>>>(d_h.ptr, d_dirty.ptr);
  syncAndCheck();
  EXPECT_EQ(d_dirty.download(), 0ULL);
}

using nccl::gin::anvil::detail::ncclGinAnvilSdmaRequest;

template <typename T>
static ncclGinAnvilIpcBufEntry makeIpcEntry(DeviceBuffer<T>* buf, size_t bytes) {
  ncclGinAnvilIpcBufEntry entry{};
  entry.local_base = reinterpret_cast<uintptr_t>(buf->ptr);
  entry.length = bytes;
  entry.remote_bases[1] = reinterpret_cast<uintptr_t>(buf->ptr);
  return entry;
}

template <typename T>
static void mapIpcTo(TemplateHarness* host, DeviceBuffer<ncclGinAnvilIpcBufEntry>* d_entry,
                     DeviceBuffer<T>* buf, size_t bytes) {
  host->ipcEntry = makeIpcEntry(buf, bytes);
  d_entry->upload(host->ipcEntry);
  host->ctx.ipcTable = d_entry->ptr;
  host->ctx.ipcTableCount = 1;
}

template <typename A, typename B>
static void mapIpcToTwo(TemplateHarness* host, DeviceBuffer<ncclGinAnvilIpcBufEntry>* d_entry,
                        DeviceBuffer<A>* a, size_t aBytes, DeviceBuffer<B>* b, size_t bBytes) {
  ncclGinAnvilIpcBufEntry table[2] = {makeIpcEntry(a, aBytes), makeIpcEntry(b, bBytes)};
  d_entry->copyFrom(table, 2);
  host->ctx.ipcTable = d_entry->ptr;
  host->ctx.ipcTableCount = 2;
}

// H13: Get below the SDMA threshold copies via ipcPut (reverse copy).
__global__ void kernelGetIpc(TemplateHarness* h, size_t bytes) {
  if (threadIdx.x != 0) return;
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinApi_Get<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(
      ginCtx, ncclCoopThread{}, 1, reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0, bytes, false, nullptr);
}

TEST_F(GinAnvilSdmaTemplateTest, Get_IpcCopiesRemoteToLocal) {
  constexpr int kN = 64;
  std::vector<uint8_t> pat(kN);
  for (int i = 0; i < kN; ++i) pat[static_cast<size_t>(i)] = static_cast<uint8_t>(0x40 + i);
  DeviceBuffer<uint8_t> d_src(static_cast<size_t>(kN));
  DeviceBuffer<uint8_t> d_dst(static_cast<size_t>(kN));
  d_src.copyFrom(pat);
  d_dst.zero();
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_src, &d_dst, &d_entry, &d_q, &d_row, 128);
  mapIpcTo(&host, &d_entry, &d_src, static_cast<size_t>(kN));
  d_h.upload(host);
  kernelGetIpc<<<1, 1>>>(d_h.ptr, static_cast<size_t>(kN));
  syncAndCheck();
  auto got = d_dst.copyTo();
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(got[static_cast<size_t>(i)], pat[static_cast<size_t>(i)]);
  }
}

// H14: Get above the threshold goes through the peer queue and marks dirty.
TEST_F(GinAnvilSdmaTemplateTest, Get_SdmaPathSetsDirty) {
  constexpr int kN = 256;
  std::vector<uint8_t> pat(kN);
  for (int i = 0; i < kN; ++i) pat[static_cast<size_t>(i)] = static_cast<uint8_t>(0x80 + i);
  DeviceBuffer<uint8_t> d_src(static_cast<size_t>(kN));
  DeviceBuffer<uint8_t> d_dst(static_cast<size_t>(kN));
  d_src.copyFrom(pat);
  d_dst.zero();
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  DeviceBuffer<uint64_t> d_dirty(1);
  d_dirty.zero();
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_src, &d_dst, &d_entry, &d_q, &d_row, 0);
  mapIpcTo(&host, &d_entry, &d_src, static_cast<size_t>(kN));
  host.ctx.sdmaDirty = d_dirty.ptr;
  d_h.upload(host);
  kernelGetIpc<<<1, 1>>>(d_h.ptr, static_cast<size_t>(kN));
  syncAndCheck();
  EXPECT_EQ(d_dirty.download(), 1ULL << 1);  // peer 1, channel 0
  auto got = d_dst.copyTo();
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(got[static_cast<size_t>(i)], pat[static_cast<size_t>(i)]);
  }
}

// H15: bytes==0 is a no-op even when the IPC table would otherwise hit.
TEST_F(GinAnvilSdmaTemplateTest, Get_ZeroBytesNoOp) {
  constexpr int kN = 16;
  std::vector<uint8_t> pat(kN, 0xAB);
  DeviceBuffer<uint8_t> d_src(static_cast<size_t>(kN));
  DeviceBuffer<uint8_t> d_dst(static_cast<size_t>(kN));
  d_src.copyFrom(pat);
  d_dst.zero();
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  DeviceBuffer<uint64_t> d_dirty(1);
  d_dirty.zero();
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_src, &d_dst, &d_entry, &d_q, &d_row, 128);
  mapIpcTo(&host, &d_entry, &d_src, static_cast<size_t>(kN));
  host.ctx.sdmaDirty = d_dirty.ptr;
  d_h.upload(host);
  kernelGetIpc<<<1, 1>>>(d_h.ptr, 0);
  syncAndCheck();
  auto got = d_dst.copyTo();
  for (uint8_t b : got) EXPECT_EQ(b, 0);
  EXPECT_EQ(d_dirty.download(), 0ULL);
}

// H16: missing queue handle falls back to ipcPut even above the threshold.
TEST_F(GinAnvilSdmaTemplateTest, Get_MissingHandleFallsBackToIpc) {
  constexpr int kN = 64;
  std::vector<uint8_t> pat(kN);
  for (int i = 0; i < kN; ++i) pat[static_cast<size_t>(i)] = static_cast<uint8_t>(0x11 + i);
  DeviceBuffer<uint8_t> d_src(static_cast<size_t>(kN));
  DeviceBuffer<uint8_t> d_dst(static_cast<size_t>(kN));
  d_src.copyFrom(pat);
  d_dst.zero();
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  DeviceBuffer<uint64_t> d_dirty(1);
  d_dirty.zero();
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_src, &d_dst, &d_entry, &d_q, &d_row, 0);
  mapIpcTo(&host, &d_entry, &d_src, static_cast<size_t>(kN));
  host.ctx.sdmaDirty = d_dirty.ptr;
  host.ctx.queueHandles = nullptr;
  d_h.upload(host);
  kernelGetIpc<<<1, 1>>>(d_h.ptr, static_cast<size_t>(kN));
  syncAndCheck();
  auto got = d_dst.copyTo();
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(got[static_cast<size_t>(i)], pat[static_cast<size_t>(i)]);
  }
  EXPECT_EQ(d_dirty.download(), 0ULL);
}

// Each (peer, channel) gets its own stub queue at index peer * numChannels + channel.
struct SdmaEnv {
  static constexpr int kQueues = 4;
  DeviceBuffer<uint8_t> src{1};
  DeviceBuffer<uint8_t> dst{1};
  DeviceBuffer<ncclGinAnvilIpcBufEntry> entry{1};
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> q{kQueues};
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> row{kQueues};
  DeviceBuffer<TemplateHarness> h{1};
  DeviceBuffer<uint64_t> dirty{1};
  TemplateHarness host{};

  explicit SdmaEnv(uint64_t dirtyBits, int numChannels = 1) {
    dirty.copyFrom(&dirtyBits, 1);
    uploadHarness(&h, &host, &src, &dst, &entry, &q, &row, 128);
    q.zero();
    std::vector<sdma_anvil::SdmaQueueDeviceHandle*> rowHost(kQueues);
    for (int i = 0; i < kQueues; ++i) rowHost[i] = q.ptr + i;
    row.copyFrom(rowHost);
    host.ctx.numChannels = numChannels;
    host.ctx.sdmaDirty = dirty.ptr;
    h.upload(host);
  }

  void setQueue(int index, uint64_t writeIndex, uint64_t readLag, uint64_t busyPolls) {
    sdma_anvil::SdmaQueueDeviceHandle handle{};
    handle.writeIndex = writeIndex;
    handle.readLag = readLag;
    handle.busyPolls = busyPolls;
    HIP_CHECK(hipMemcpy(q.ptr + index, &handle, sizeof(handle), hipMemcpyHostToDevice));
  }

  unsigned long long polls(int index) {
    sdma_anvil::SdmaQueueDeviceHandle handle{};
    HIP_EXPECT(hipMemcpy(&handle, q.ptr + index, sizeof(handle), hipMemcpyDeviceToHost));
    return handle.polls;
  }
};

// H17: FlushAsync records the dirty channels and queue target of the requested peer without quieting or clearing them.
__global__ void kernelFlushAsync(TemplateHarness* h, ncclGinRequest_t* req, int peer) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinApi_FlushAsync<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, peer, req, false, nullptr, 0);
}

static ncclGinAnvilSdmaRequest readRequest(DeviceBuffer<ncclGinRequest_t>& d_req) {
  ncclGinRequest_t raw = d_req.download();
  ncclGinAnvilSdmaRequest req;
  std::memcpy(&req, &raw, sizeof(req));
  return req;
}

static ncclGinRequest_t makeRequest(int peer, uint32_t channelMask, uint64_t target = 0) {
  ncclGinAnvilSdmaRequest req{peer, channelMask, target};
  ncclGinRequest_t raw{};
  std::memcpy(&raw, &req, sizeof(req));
  return raw;
}

TEST_F(GinAnvilSdmaTemplateTest, FlushAsync_RecordsDirtyChannelsWithoutQuiet) {
  uint64_t peer1Bit = 1ULL << 1;  // peer 1, channel 0, numChannels=1
  SdmaEnv env(peer1Bit);
  DeviceBuffer<ncclGinRequest_t> d_req(1);
  d_req.zero();
  env.setQueue(/*index=*/1, /*writeIndex=*/5, /*readLag=*/0, /*busyPolls=*/0);
  resetQuietCount();
  kernelFlushAsync<<<1, 1>>>(env.h.ptr, d_req.ptr, /*peer=*/1);
  syncAndCheck();
  ncclGinAnvilSdmaRequest req = readRequest(d_req);
  EXPECT_EQ(req.peer, 1);
  EXPECT_EQ(req.channelMask, 1u);
  EXPECT_EQ(req.target, 5ULL);
  EXPECT_EQ(env.dirty.download(), peer1Bit);
  EXPECT_EQ(readQuietCount(), 0ULL);
}

TEST_F(GinAnvilSdmaTemplateTest, FlushAsync_CleanPeerRecordsNoChannels) {
  uint64_t peer0Bit = 1ULL << 0;  // peer 0 dirty, request peer 1. A bitIdx that ignored peer would record ch 0
  SdmaEnv env(peer0Bit);
  DeviceBuffer<ncclGinRequest_t> d_req(1);
  d_req.zero();
  resetQuietCount();
  kernelFlushAsync<<<1, 1>>>(env.h.ptr, d_req.ptr, /*peer=*/1);
  syncAndCheck();
  EXPECT_EQ(readRequest(d_req).channelMask, 0u);
  EXPECT_EQ(env.dirty.download(), peer0Bit);
  EXPECT_EQ(readQuietCount(), 0ULL);
}

// H18: an invalid ctx records no channels.
TEST_F(GinAnvilSdmaTemplateTest, FlushAsync_InvalidCtxRecordsNoChannels) {
  SdmaEnv env(/*dirtyBits=*/1ULL << 1);
  env.host.ctx.layoutMagic = 0;
  env.h.upload(env.host);
  DeviceBuffer<ncclGinRequest_t> d_req(1);
  d_req.zero();
  resetQuietCount();
  kernelFlushAsync<<<1, 1>>>(env.h.ptr, d_req.ptr, /*peer=*/1);
  syncAndCheck();
  EXPECT_EQ(readRequest(d_req).channelMask, 0u);
  EXPECT_EQ(readQuietCount(), 0ULL);
}

// H19/H20: Wait polls the channel recorded by FlushAsync up to its target, then fences.
__global__ void kernelWait(TemplateHarness* h, ncclGinRequest_t* req, uint32_t* abortFlag) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(
      ginCtx, *req, false, nullptr, cuda::memory_order_acq_rel, abortFlag);
}

TEST_F(GinAnvilSdmaTemplateTest, Wait_PollsRecordedChannel) {
  SdmaEnv env(/*dirtyBits=*/0);
  DeviceBuffer<ncclGinRequest_t> d_req(1);
  d_req.upload(makeRequest(/*peer=*/1, /*channelMask=*/1u));
  resetThreadfenceCount();
  kernelWait<<<1, 1>>>(env.h.ptr, d_req.ptr, nullptr);
  syncAndCheck();
  EXPECT_EQ(env.polls(1), 1ULL);
  EXPECT_EQ(env.polls(0), 0ULL);
  EXPECT_EQ(readThreadfenceCount(), 1ULL);
}

TEST_F(GinAnvilSdmaTemplateTest, Wait_EmptyRequestOnlyFences) {
  SdmaEnv env(/*dirtyBits=*/0);
  DeviceBuffer<ncclGinRequest_t> d_req(1);
  d_req.upload(makeRequest(/*peer=*/0, /*channelMask=*/0u));
  resetQuietCount();
  resetThreadfenceCount();
  kernelWait<<<1, 1>>>(env.h.ptr, d_req.ptr, nullptr);
  syncAndCheck();
  EXPECT_EQ(readQuietCount(), 0ULL);
  EXPECT_EQ(readThreadfenceCount(), 1ULL);
}

// H21/H22: a strong signal still resolves the peer queue so fenceBeforeSignal
// quiets SDMA instead of racing the payload via IPC. hasWins=false is a
// standalone barrier signal; hasWins=true is a windowed sub-threshold put.
__global__ void kernelPutSignalQuiesce(TemplateHarness* h, bool hasWins, size_t bytes, ncclGinSignal_t signalId = 0,
                                       bool hasCounter = false, ncclGinSignalOp_t op = ncclGinSignalInc) {
  if (threadIdx.x != 0) return;
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  sig.indexedSignal.signalId = signalId;
  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(
      ginCtx, ncclCoopThread{}, 1, hasWins, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, bytes, sig, op, 0, hasCounter, 0, false,
      nullptr, cuda::thread_scope_system, cuda::thread_scope_system);
}

TEST_F(GinAnvilSdmaTemplateTest, Put_StandaloneSignalResolvesQueue) {
  DeviceBuffer<uint8_t> d_src(1);
  DeviceBuffer<uint8_t> d_dst(1);
  DeviceBuffer<uint64_t> d_signals(2);
  d_signals.zero();
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_src, &d_dst, &d_entry, &d_q, &d_row, 0);
  host.ctx.signals = d_signals.ptr;
  host.ctx.nSignals = 2;
  mapIpcTo(&host, &d_entry, &d_signals, 2 * sizeof(uint64_t));
  d_h.upload(host);
  resetQuietCount();
  resetThreadfenceCount();
  kernelPutSignalQuiesce<<<1, 1>>>(d_h.ptr, /*hasWins=*/false, /*bytes=*/0);
  syncAndCheck();
  EXPECT_EQ(d_signals.download(), 1ULL);
  EXPECT_EQ(readQuietCount(), 1ULL);
  EXPECT_EQ(readThreadfenceCount(), 1ULL);
}

// H22: a windowed sub-threshold put with a strong signal still resolves the peer
// queue so fenceBeforeSignal quiets in-flight SDMA from earlier large puts.
TEST_F(GinAnvilSdmaTemplateTest, Put_WindowedIpcPutStrongSignalResolvesQueue) {
  constexpr int kN = 64;
  std::vector<uint8_t> pat(kN);
  for (int i = 0; i < kN; ++i) pat[static_cast<size_t>(i)] = static_cast<uint8_t>(0x51 + i);
  DeviceBuffer<uint8_t> d_src(static_cast<size_t>(kN));
  DeviceBuffer<uint8_t> d_dst(static_cast<size_t>(kN));
  d_src.copyFrom(pat);
  d_dst.zero();
  DeviceBuffer<uint64_t> d_signals(2);
  d_signals.zero();
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(2);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_src, &d_dst, &d_entry, &d_q, &d_row, 128);
  host.ctx.signals = d_signals.ptr;
  host.ctx.nSignals = 2;
  mapIpcToTwo(&host, &d_entry, &d_dst, static_cast<size_t>(kN), &d_signals, 2 * sizeof(uint64_t));
  d_h.upload(host);
  resetQuietCount();
  resetThreadfenceCount();
  kernelPutSignalQuiesce<<<1, 1>>>(d_h.ptr, /*hasWins=*/true, static_cast<size_t>(kN));
  syncAndCheck();
  EXPECT_EQ(d_signals.download(), 1ULL);
  EXPECT_EQ(readQuietCount(), 1ULL);
  EXPECT_EQ(readThreadfenceCount(), 1ULL);
  auto got = d_dst.copyTo();
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(got[static_cast<size_t>(i)], pat[static_cast<size_t>(i)]);
  }
}

// H23: windowed PutValue with a strong signal also resolves the peer queue.
__global__ void kernelPutValueIpcSignalQuiesce(TemplateHarness* h, uint64_t value, ncclGinSignal_t signalId = 0) {
  if (threadIdx.x != 0) return;
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  sig.indexedSignal.signalId = signalId;
  ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(
      ginCtx, ncclCoopThread{}, 1, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0, value, sig,
      ncclGinSignalInc, 0, false, nullptr, cuda::thread_scope_system, cuda::thread_scope_system);
}

// H24: counter-only windowed put below the SDMA threshold still resolves the
// peer queue via the || hasCounter disjunct so fenceBeforeSignal can quiet.
__global__ void kernelPutCounterOnlyIpcQuiesce(TemplateHarness* h, size_t bytes) {
  if (threadIdx.x != 0) return;
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(
      ginCtx, ncclCoopThread{}, 1, true, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, bytes, sig, ncclGinSignalInc, 0, true, 0, false,
      nullptr, cuda::thread_scope_system, cuda::thread_scope_system);
}

TEST_F(GinAnvilSdmaTemplateTest, Put_WindowedIpcPutCounterOnlyResolvesQueue) {
  constexpr int kN = 64;
  std::vector<uint8_t> pat(kN);
  for (int i = 0; i < kN; ++i) pat[static_cast<size_t>(i)] = static_cast<uint8_t>(0x33 + i);
  DeviceBuffer<uint8_t> d_src(static_cast<size_t>(kN));
  DeviceBuffer<uint8_t> d_dst(static_cast<size_t>(kN));
  d_src.copyFrom(pat);
  d_dst.zero();
  DeviceBuffer<uint64_t> d_counters(1);
  d_counters.zero();
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_src, &d_dst, &d_entry, &d_q, &d_row, 128);
  host.ctx.counters = d_counters.ptr;
  mapIpcTo(&host, &d_entry, &d_dst, static_cast<size_t>(kN));
  d_h.upload(host);
  resetQuietCount();
  resetThreadfenceCount();
  kernelPutCounterOnlyIpcQuiesce<<<1, 1>>>(d_h.ptr, static_cast<size_t>(kN));
  syncAndCheck();
  EXPECT_EQ(d_counters.download(), 1ULL);
  EXPECT_EQ(readQuietCount(), 1ULL);
  EXPECT_EQ(readThreadfenceCount(), 1ULL);
  auto got = d_dst.copyTo();
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(got[static_cast<size_t>(i)], pat[static_cast<size_t>(i)]);
  }
}

TEST_F(GinAnvilSdmaTemplateTest, PutValue_WindowedIpcPutStrongSignalResolvesQueue) {
  constexpr uint64_t kVal = 0xAABBCCDDEEFF0011ULL;
  DeviceBuffer<uint8_t> d_dst(sizeof(uint64_t));
  d_dst.zero();
  DeviceBuffer<uint64_t> d_signals(2);
  d_signals.zero();
  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(2);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> d_q(1);
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> d_row(2);
  DeviceBuffer<TemplateHarness> d_h(1);
  TemplateHarness host{};
  uploadHarness(&d_h, &host, &d_dst, &d_dst, &d_entry, &d_q, &d_row, 128);
  host.ctx.signals = d_signals.ptr;
  host.ctx.nSignals = 2;
  mapIpcToTwo(&host, &d_entry, &d_dst, sizeof(uint64_t), &d_signals, 2 * sizeof(uint64_t));
  d_h.upload(host);
  resetQuietCount();
  resetThreadfenceCount();
  kernelPutValueIpcSignalQuiesce<<<1, 1>>>(d_h.ptr, kVal);
  syncAndCheck();
  EXPECT_EQ(d_signals.download(), 1ULL);
  EXPECT_EQ(readQuietCount(), 1ULL);
  EXPECT_EQ(readThreadfenceCount(), 1ULL);
  auto got = d_dst.copyTo();
  uint64_t landed = 0;
  std::memcpy(&landed, got.data(), sizeof(landed));
  EXPECT_EQ(landed, kVal);
}

static uint64_t downloadU64(const DeviceBuffer<uint8_t>& buf) {
  auto got = buf.copyTo();
  uint64_t v = 0;
  std::memcpy(&v, got.data(), sizeof(v));
  return v;
}

// Peer 1, one channel, sdmaChannel 0: markSdmaDirty sets bit peer * numCh + effCh.
constexpr uint64_t kPeer1DirtyBit = 1ULL << 1;

// mapDst IPC-maps dst; remoteSignalAddrs routes signals only via signal_remote_addrs and enables fusion on OSS7.
struct SdmaSignalEnv {
  DeviceBuffer<uint8_t> src{1};
  DeviceBuffer<uint8_t> dst{sizeof(uint64_t)};
  DeviceBuffer<uint64_t> signals{2};
  DeviceBuffer<uint64_t> dirty{1};
  DeviceBuffer<uint64_t> counters{1};
  DeviceBuffer<uintptr_t> sigAddrs{2};
  DeviceBuffer<ncclGinAnvilIpcBufEntry> entry{2};
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle> q{1};
  DeviceBuffer<sdma_anvil::SdmaQueueDeviceHandle*> row{2};
  DeviceBuffer<TemplateHarness> h{1};
  TemplateHarness host{};

  SdmaSignalEnv(int threshold, bool mapDst, bool remoteSignalAddrs) {
    dst.zero();
    signals.zero();
    dirty.zero();
    uploadHarness(&h, &host, &src, &dst, &entry, &q, &row, threshold);
    host.ctx.signals = signals.ptr;
    host.ctx.nSignals = 2;
    host.ctx.sdmaDirty = dirty.ptr;
    host.ctx.counters = counters.ptr;
    sigAddrs.copyFrom(std::vector<uintptr_t>{0, reinterpret_cast<uintptr_t>(signals.ptr)});
    host.ctx.signal_remote_addrs = remoteSignalAddrs ? sigAddrs.ptr : nullptr;
    if (remoteSignalAddrs) {
      mapIpcTo(&host, &entry, &dst, sizeof(uint64_t));  // Signals resolve only via signal_remote_addrs[peer].
    } else if (mapDst) {
      mapIpcToTwo(&host, &entry, &dst, sizeof(uint64_t), &signals, 2 * sizeof(uint64_t));
    } else {
      mapIpcTo(&host, &entry, &signals, 2 * sizeof(uint64_t));
    }
    h.upload(host);
    resetQuietCount();
    resetThreadfenceCount();
  }
};

// sizeof(T) > sdmaThreshold takes the SDMA path and marks dirty; at the threshold it stays on IPC and leaves it clean.
TEST_F(GinAnvilSdmaTemplateTest, PutValue_ThresholdPicksSdmaOrIpcPath) {
  for (const int threshold : {0, static_cast<int>(sizeof(uint64_t))}) {
    SCOPED_TRACE(::testing::Message() << "sdmaThreshold=" << threshold);
    SdmaSignalEnv env(threshold, /*mapDst=*/true, /*remoteSignalAddrs=*/false);
    kernelPutValueSdma<<<1, 1>>>(env.h.ptr);
    syncAndCheck();
    EXPECT_EQ(downloadU64(env.dst), 0xAABBCCDDEEFF0011ULL);
    EXPECT_EQ(env.dirty.download(), threshold == 0 ? kPeer1DirtyBit : 0ULL);
  }
}

static void setStubRecordOnly(bool on) {
  sdma_anvil::SdmaStubLog log{};
  log.recordOnly = on;
  HIP_EXPECT(hipMemcpyToSymbol(HIP_SYMBOL(sdma_anvil::g_sdmaStubLog), &log, sizeof(log)));
}

namespace {

// Clears record-only on every scope exit so no early return can leave later Puts in this process skipping their copies.
class StubRecordOnlyScope {
public:
  StubRecordOnlyScope() {
    setStubRecordOnly(true);
  }
  ~StubRecordOnlyScope() {
    setStubRecordOnly(false);
  }
  StubRecordOnlyScope(const StubRecordOnlyScope&) = delete;
  StubRecordOnlyScope& operator=(const StubRecordOnlyScope&) = delete;
};

}  // namespace

static sdma_anvil::SdmaStubLog readStubLog() {
  sdma_anvil::SdmaStubLog log{};
  HIP_EXPECT(hipMemcpyFromSymbol(&log, HIP_SYMBOL(sdma_anvil::g_sdmaStubLog), sizeof(log)));
  return log;
}

__global__ void kernelSdmaIsOss7(int* out) {
  if (threadIdx.x != 0) {
    return;
  }
  out[0] = SDMA_IS_OSS7;
}

// Fused SDMA signals compile only where SDMA_IS_OSS7 (gfx950); elsewhere useSdmaFusedSignal is always false.
static bool sdmaIsOss7() {
  DeviceBuffer<int> d_oss7(1);
  d_oss7.zero();
  kernelSdmaIsOss7<<<1, 1>>>(d_oss7.ptr);
  HIP_EXPECT(hipGetLastError());
  HIP_EXPECT(hipDeviceSynchronize());
  return d_oss7.download() != 0;
}

// H25: Puts above kGinPutSegBytes split at the cap (stub logs only); fused puts the signal on the last segment.
static void checkMultiSegmentSplit(bool fused) {
  constexpr size_t kSeg = gin_sdma::kGinPutSegBytes;
  SdmaSignalEnv env(/*threshold=*/0, /*mapDst=*/true, /*remoteSignalAddrs=*/true);
  env.host.ctx.fusedSdmaSignal = fused ? 1 : 0;
  env.h.upload(env.host);
  const uintptr_t dst = reinterpret_cast<uintptr_t>(env.dst.ptr);
  const uintptr_t src = reinterpret_cast<uintptr_t>(env.src.ptr);

  struct SegCase {
    size_t bytes;
    unsigned nSeg;
  };
  for (const SegCase& c : {SegCase{kSeg, 1}, SegCase{kSeg + 1, 2}, SegCase{2 * kSeg, 2}, SegCase{2 * kSeg + 1, 3}}) {
    SCOPED_TRACE(::testing::Message() << "fused=" << fused << " bytes=" << c.bytes);
    env.signals.zero();
    resetQuietCount();
    sdma_anvil::SdmaStubLog log{};
    {
      StubRecordOnlyScope recordOnly;
      ASSERT_TRUE(readStubLog().recordOnly) << "else the stub would copy up to 2 * kGinPutSegBytes into 8-byte buffers";
      kernelPutSignalQuiesce<<<1, 1>>>(env.h.ptr, /*hasWins=*/true, c.bytes, /*signalId=*/1);
      ASSERT_EQ(hipGetLastError(), hipSuccess);
      ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
      log = readStubLog();
    }
    ASSERT_EQ(log.count, fused ? c.nSeg : c.nSeg + 1);
    for (unsigned i = 0; i < c.nSeg; ++i) {
      const size_t off = i * kSeg;
      const bool last = i + 1 == c.nSeg;
      EXPECT_EQ(log.calls[i].op, (last && fused) ? sdma_anvil::kSdmaStubPutSignal : sdma_anvil::kSdmaStubPut) << i;
      EXPECT_EQ(reinterpret_cast<uintptr_t>(log.calls[i].dst), dst + off) << i;
      EXPECT_EQ(reinterpret_cast<uintptr_t>(log.calls[i].src), src + off) << i;
      EXPECT_EQ(log.calls[i].size, std::min(kSeg, c.bytes - off)) << i;
      EXPECT_EQ(log.calls[i].signal, (last && fused) ? env.signals.ptr + 1 : nullptr) << i;
    }
    if (fused) {
      EXPECT_EQ(readQuietCount(), 0ULL);
      EXPECT_EQ(env.signals.copyTo()[1], 0ULL);
    } else {
      EXPECT_EQ(log.calls[c.nSeg].op, sdma_anvil::kSdmaStubQuiet);
      EXPECT_EQ(readQuietCount(), 1ULL);
      EXPECT_EQ(env.signals.copyTo()[1], 1ULL);
    }
  }
}

TEST_F(GinAnvilSdmaTemplateTest, Put_MultiSegmentSplitsAtSegCap) {
  checkMultiSegmentSplit(/*fused=*/false);
}

TEST_F(GinAnvilSdmaTemplateTest, Put_MultiSegmentFusedSignalsLastSegment) {
  if (!sdmaIsOss7()) {
    GTEST_SKIP() << "fused SDMA signal needs SDMA_IS_OSS7 (gfx950)";
  }
  checkMultiSegmentSplit(/*fused=*/true);
  // Fusion needs SignalInc and no counter: a counter or a SignalAdd keeps one segment unfused (put, then quiet).
  SdmaSignalEnv env(/*threshold=*/0, /*mapDst=*/true, /*remoteSignalAddrs=*/true);
  for (const bool counter : {true, false}) {
    StubRecordOnlyScope recordOnly;
    ASSERT_TRUE(readStubLog().recordOnly);
    kernelPutSignalQuiesce<<<1, 1>>>(env.h.ptr, true, 64, 1, counter, counter ? ncclGinSignalInc : ncclGinSignalAdd);
    syncAndCheck();
    const sdma_anvil::SdmaStubLog log = readStubLog();
    ASSERT_EQ(log.count, 2U) << "counter=" << counter;
    EXPECT_EQ(log.calls[0].op, sdma_anvil::kSdmaStubPut) << "counter=" << counter;
    EXPECT_EQ(log.calls[1].op, sdma_anvil::kSdmaStubQuiet) << "counter=" << counter;
  }
}

// Unfused SDMA-path PutValue: a hit lands, marks dirty and quiets; a dst resolve miss drops the value and skips quiet.
TEST_F(GinAnvilSdmaTemplateTest, PutValue_SdmaPathSignalQuietsOnlyOnHit) {
  constexpr uint64_t kVal = 0x1122334455667788ULL;
  for (const bool mapDst : {true, false}) {
    SCOPED_TRACE(::testing::Message() << "mapDst=" << mapDst);
    SdmaSignalEnv env(/*threshold=*/0, mapDst, /*remoteSignalAddrs=*/false);
    kernelPutValueIpcSignalQuiesce<<<1, 1>>>(env.h.ptr, kVal);
    syncAndCheck();
    EXPECT_EQ(downloadU64(env.dst), mapDst ? kVal : 0ULL);
    EXPECT_EQ(env.dirty.download(), mapDst ? kPeer1DirtyBit : 0ULL);
    EXPECT_EQ(env.signals.download(), 1ULL);
    EXPECT_EQ(readQuietCount(), mapDst ? 1ULL : 0ULL);
    EXPECT_EQ(readThreadfenceCount(), 1ULL);
  }
}

// Fused SDMA-path PutValue issues one putSignal to the remote signal and skips the fence, quiet and signalPeer.
TEST_F(GinAnvilSdmaTemplateTest, PutValue_SdmaFusedSignalSkipsSignalPeer) {
  if (!sdmaIsOss7()) {
    GTEST_SKIP() << "fused SDMA signal needs SDMA_IS_OSS7 (gfx950)";
  }
  constexpr uint64_t kVal = 0x5A5A5A5A5A5A5A5AULL;
  SdmaSignalEnv env(/*threshold=*/0, /*mapDst=*/true, /*remoteSignalAddrs=*/true);
  sdma_anvil::SdmaStubLog log{};
  {
    StubRecordOnlyScope recordOnly;
    ASSERT_TRUE(readStubLog().recordOnly);
    kernelPutValueIpcSignalQuiesce<<<1, 1>>>(env.h.ptr, kVal, /*signalId=*/1);
    syncAndCheck();
    log = readStubLog();
  }
  ASSERT_EQ(log.count, 1U);
  EXPECT_EQ(log.calls[0].op, sdma_anvil::kSdmaStubPutSignal);
  EXPECT_EQ(log.calls[0].dst, static_cast<void*>(env.dst.ptr));
  EXPECT_EQ(log.calls[0].size, sizeof(uint64_t));
  EXPECT_EQ(log.calls[0].signal, env.signals.ptr + 1) << "signal_remote_addrs[peer 1] + signalId * 8";
  EXPECT_EQ(env.dirty.download(), kPeer1DirtyBit);
  EXPECT_EQ(env.signals.copyTo()[1], 0ULL);
  EXPECT_EQ(readQuietCount(), 0ULL);
  EXPECT_EQ(readThreadfenceCount(), 0ULL);
  kernelPutValueIpcSignalQuiesce<<<1, 1>>>(env.h.ptr, kVal, /*signalId=*/1);  // Copying stub: the value must land.
  syncAndCheck();
  EXPECT_EQ(downloadU64(env.dst), kVal);
}

// H25-H30: Flush and Wait poll the queue until it reaches the target, the budget runs out, or abort is set.
constexpr uint64_t kShortBudget = 1000;
constexpr uint64_t kLongBudget = 1ULL << 34;
constexpr unsigned long long kNeverDrains = ~0ULL;

__global__ void kernelFlushTimeout(TemplateHarness* h, uint32_t* abortFlag, uint64_t timeoutCycles,
                                   ncclResult_t* result) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  *result = ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(
      ginCtx, ncclCoopThread{}, false, nullptr, cuda::memory_order_seq_cst, abortFlag, timeoutCycles);
}

__global__ void kernelWaitTimeout(TemplateHarness* h, ncclGinRequest_t* req, uint32_t* abortFlag,
                                  uint64_t timeoutCycles, ncclResult_t* result) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  *result = ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(
      ginCtx, *req, false, nullptr, cuda::memory_order_acq_rel, abortFlag, timeoutCycles);
}

TEST_F(GinAnvilSdmaTemplateTest, FlushTimeout_PollsUntilDrainedAndClearsDirty) {
  SdmaEnv env(/*dirtyBits=*/1ULL << 1);
  DeviceBuffer<ncclResult_t> d_result(1);
  d_result.upload(ncclInternalError);
  env.setQueue(/*index=*/1, /*writeIndex=*/0, /*readLag=*/0, /*busyPolls=*/3);
  kernelFlushTimeout<<<1, 1>>>(env.h.ptr, nullptr, kLongBudget, d_result.ptr);
  syncAndCheck();
  EXPECT_EQ(d_result.download(), ncclSuccess);
  EXPECT_EQ(env.dirty.download(), 0ULL);
  EXPECT_EQ(env.polls(1), 4ULL);
}

TEST_F(GinAnvilSdmaTemplateTest, FlushTimeout_BusyQueueTimesOutAndStaysDirty) {
  SdmaEnv env(/*dirtyBits=*/1ULL << 1);
  DeviceBuffer<ncclResult_t> d_result(1);
  d_result.upload(ncclInternalError);
  env.setQueue(/*index=*/1, /*writeIndex=*/0, /*readLag=*/0, kNeverDrains);
  kernelFlushTimeout<<<1, 1>>>(env.h.ptr, nullptr, kShortBudget, d_result.ptr);
  syncAndCheck();
  EXPECT_EQ(d_result.download(), ncclTimeout);
  EXPECT_EQ(env.dirty.download(), 1ULL << 1);
}

TEST_F(GinAnvilSdmaTemplateTest, WaitTimeout_CoversOnlyWorkBeforeFlushAsync) {
  SdmaEnv env(/*dirtyBits=*/1ULL << 1);
  DeviceBuffer<ncclGinRequest_t> d_req(1);
  DeviceBuffer<ncclResult_t> d_result(1);
  env.setQueue(/*index=*/1, /*writeIndex=*/5, /*readLag=*/0, /*busyPolls=*/0);
  kernelFlushAsync<<<1, 1>>>(env.h.ptr, d_req.ptr, /*peer=*/1);
  syncAndCheck();

  // Work posted after FlushAsync moves the write index past the recorded target.
  d_result.upload(ncclInternalError);
  env.setQueue(/*index=*/1, /*writeIndex=*/9, /*readLag=*/4, /*busyPolls=*/0);
  kernelWaitTimeout<<<1, 1>>>(env.h.ptr, d_req.ptr, nullptr, kShortBudget, d_result.ptr);
  syncAndCheck();
  EXPECT_EQ(d_result.download(), ncclSuccess);

  d_result.upload(ncclInternalError);
  env.setQueue(/*index=*/1, /*writeIndex=*/9, /*readLag=*/5, /*busyPolls=*/0);
  kernelWaitTimeout<<<1, 1>>>(env.h.ptr, d_req.ptr, nullptr, kShortBudget, d_result.ptr);
  syncAndCheck();
  EXPECT_EQ(d_result.download(), ncclTimeout);
}

TEST_F(GinAnvilSdmaTemplateTest, WaitTimeout_BusyQueueTimesOutAfterFence) {
  SdmaEnv env(/*dirtyBits=*/0);
  DeviceBuffer<ncclGinRequest_t> d_req(1);
  d_req.upload(makeRequest(/*peer=*/1, /*channelMask=*/1u, /*target=*/5));
  DeviceBuffer<ncclResult_t> d_result(1);
  d_result.upload(ncclInternalError);
  env.setQueue(/*index=*/1, /*writeIndex=*/5, /*readLag=*/0, kNeverDrains);
  resetThreadfenceCount();
  kernelWaitTimeout<<<1, 1>>>(env.h.ptr, d_req.ptr, nullptr, kShortBudget, d_result.ptr);
  syncAndCheck();
  EXPECT_EQ(d_result.download(), ncclTimeout);
  EXPECT_EQ(readThreadfenceCount(), 1ULL);
}

TEST_F(GinAnvilSdmaTemplateTest, WaitTimeout_AbortReturnsSuccess) {
  SdmaEnv env(/*dirtyBits=*/0);
  DeviceBuffer<ncclGinRequest_t> d_req(1);
  d_req.upload(makeRequest(/*peer=*/1, /*channelMask=*/1u, /*target=*/5));
  DeviceBuffer<ncclResult_t> d_result(1);
  d_result.upload(ncclInternalError);
  DeviceBuffer<uint32_t> d_abort(1);
  uint32_t aborted = 1;
  d_abort.copyFrom(&aborted, 1);
  env.setQueue(/*index=*/1, /*writeIndex=*/5, /*readLag=*/0, kNeverDrains);
  kernelWaitTimeout<<<1, 1>>>(env.h.ptr, d_req.ptr, d_abort.ptr, kLongBudget, d_result.ptr);
  syncAndCheck();
  EXPECT_EQ(d_result.download(), ncclSuccess);
}

TEST_F(GinAnvilSdmaTemplateTest, Wait_BlockingReturnsOnAbort) {
  SdmaEnv env(/*dirtyBits=*/0);
  DeviceBuffer<ncclGinRequest_t> d_req(1);
  d_req.upload(makeRequest(/*peer=*/1, /*channelMask=*/1u, /*target=*/5));
  DeviceBuffer<uint32_t> d_abort(1);
  uint32_t aborted = 1;
  d_abort.copyFrom(&aborted, 1);
  env.setQueue(/*index=*/1, /*writeIndex=*/5, /*readLag=*/0, kNeverDrains);
  resetThreadfenceCount();
  kernelWait<<<1, 1>>>(env.h.ptr, d_req.ptr, d_abort.ptr);
  syncAndCheck();
  EXPECT_GT(env.polls(1), 1ULL);
  EXPECT_EQ(readThreadfenceCount(), 1ULL);
}

// H31: a request spanning several channels records the live-target marker, and Wait drains each channel.
TEST_F(GinAnvilSdmaTemplateTest, Wait_MultiChannelRequestDrainsEachChannel) {
  SdmaEnv env(/*dirtyBits=*/(1ULL << 2) | (1ULL << 3), /*numChannels=*/2);  // peer 1, channels 0 and 1
  DeviceBuffer<ncclGinRequest_t> d_req(1);
  d_req.zero();
  kernelFlushAsync<<<1, 1>>>(env.h.ptr, d_req.ptr, /*peer=*/1);
  syncAndCheck();
  ncclGinAnvilSdmaRequest req = readRequest(d_req);
  EXPECT_EQ(req.channelMask, 3u);
  EXPECT_EQ(req.target, nccl::gin::anvil::detail::kSdmaLiveTarget);
  kernelWait<<<1, 1>>>(env.h.ptr, d_req.ptr, nullptr);
  syncAndCheck();
  EXPECT_EQ(env.polls(2), 1ULL);
  EXPECT_EQ(env.polls(3), 1ULL);
}

// H32: blocking Flush returns once abort is set, even if the queue never drains.
__global__ void kernelFlushBlocking(TemplateHarness* h, uint32_t* abortFlag) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, ncclCoopThread{}, false, nullptr,
                                                         cuda::memory_order_seq_cst, abortFlag);
}

TEST_F(GinAnvilSdmaTemplateTest, Flush_BlockingReturnsOnAbort) {
  SdmaEnv env(/*dirtyBits=*/1ULL << 1);
  DeviceBuffer<uint32_t> d_abort(1);
  uint32_t aborted = 1;
  d_abort.copyFrom(&aborted, 1);
  env.setQueue(/*index=*/1, /*writeIndex=*/0, /*readLag=*/0, kNeverDrains);
  kernelFlushBlocking<<<1, 1>>>(env.h.ptr, d_abort.ptr);
  syncAndCheck();
  EXPECT_GT(env.polls(1), 1ULL);
}

// H33: when one peer drains and the next times out, the timed Flush clears only the drained peer's bits.
TEST_F(GinAnvilSdmaTemplateTest, FlushTimeout_ClearsOnlyDrainedPeers) {
  SdmaEnv env(/*dirtyBits=*/(1ULL << 0) | (1ULL << 1));
  env.setQueue(/*index=*/1, /*writeIndex=*/0, /*readLag=*/0, kNeverDrains);
  DeviceBuffer<ncclResult_t> d_result(1);
  d_result.upload(ncclInternalError);
  kernelFlushTimeout<<<1, 1>>>(env.h.ptr, nullptr, kShortBudget, d_result.ptr);
  syncAndCheck();
  EXPECT_EQ(d_result.download(), ncclTimeout);
  EXPECT_EQ(env.dirty.download(), 1ULL << 1);
  EXPECT_EQ(env.polls(0), 1ULL);
}

#endif  // NCCL_GIN_ANVIL_SDMA_ENABLE

}  // namespace RcclUnitTesting
