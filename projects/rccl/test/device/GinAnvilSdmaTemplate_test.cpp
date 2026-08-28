/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Suite H: device template coverage (Put/PutValue/Flush/Signal/Counter SDMA + edge paths).
// Uses test/device/sdma/anvil_device.hpp stubs (no librocshmem device link).

#include "DeviceTestBase.hpp"

#include "nccl_device/coop.h"
#include "nccl_device/gin/gin_device_host_common.h"
#include "nccl_device/gin/gin_device_common.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h"

#if NCCL_GIN_ANVIL_SDMA_ENABLE
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma.h"
#endif

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

// H2: thread-scope fence when given > required.
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
  kernelPutScopeFence<<<1, 1>>>(d_h.ptr);
  syncAndCheck();
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
    dirtyOut[0] = __hip_atomic_load(h->ctx.sdmaDirty, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
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
  ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, ncclCoopThread{},
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
  uint64_t* ctr = ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, 0);
  if (ctr) ctr[0] = 99;
  ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, 0);
  outCtr[0] = ctr ? ctr[0] : 0;
  uint64_t* sig = ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, 0);
  if (sig) sig[0] = 11;
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  desc.indexedSignal.signalId = 0;
  ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, desc);
  outSig[0] = sig ? sig[0] : 0;
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
  ok[0] = ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, 0) == nullptr;
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
  kernelPutSdmaCounter<<<1, 1>>>(d_h.ptr);
  syncAndCheck();
  EXPECT_EQ(d_counters.download(), 1ULL);
}

// H12: Flush clears multiple dirty channel bits across peers.
__global__ void kernelFlushMultiDirty(TemplateHarness* h, uint64_t* dirty) {
  h->ctx.sdmaDirty = dirty;
  h->ctx.numChannels = 2;
  h->ctx.nRanks = 2;
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, ncclCoopThread{},
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

#endif  // NCCL_GIN_ANVIL_SDMA_ENABLE

}  // namespace RcclUnitTesting
