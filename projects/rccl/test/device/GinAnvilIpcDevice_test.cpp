/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Suites B–E: device IPC resolve, flat copy, detail helpers, and template IPC paths.

#include "DeviceTestBase.hpp"

#include "nccl_device/coop.h"
#include "nccl_device/gin/gin_device_host_common.h"
#include "nccl_device/gin/gin_device_common.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_ipc_table_device.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_ipc_copy.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h"

#if NCCL_GIN_ANVIL_SDMA_ENABLE
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma.h"
#endif

#include <cstring>
#include <vector>

namespace RcclUnitTesting
{

class GinAnvilIpcDeviceTest : public DeviceTestBase {};

using nccl::gin::anvil::detail::ginAnvilResolvePeerVa;

// ---------------------------------------------------------------------------
// Suite B: ginAnvilResolvePeerVa
// ---------------------------------------------------------------------------

struct ResolveCase {
  void* localAddr;
  int peer;
  int count;
  bool expectHit;
  uintptr_t expectRemote;
};

__global__ void kernelResolvePeerVa(const ncclGinAnvilIpcBufEntry* table, const ResolveCase* cases,
                                    void** out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const ResolveCase& c = cases[i];
  out[i] = ginAnvilResolvePeerVa(c.localAddr, c.peer, table, c.count);
}

TEST_F(GinAnvilIpcDeviceTest, ResolvePeerVa_AllCases) {
  ncclGinAnvilIpcBufEntry entry{};
  entry.local_base = 0x50000000ULL;
  entry.length = 0x1000;
  for (int pe = 0; pe < 4; ++pe) {
    entry.remote_bases[pe] = 0x60000000ULL + static_cast<uintptr_t>(pe) * 0x1000;
  }

  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_table(1);
  d_table.upload(entry);

  std::vector<ResolveCase> cases = {
      {reinterpret_cast<void*>(0x50000100ULL), 2, 1, true, 0x60002000ULL + 0x100},
      {reinterpret_cast<void*>(0x50000FFFULL), 0, 1, true, 0x60000FFFULL},  // last byte in range → hit
      {reinterpret_cast<void*>(0x50001000ULL), 0, 1, false, 0},  // one past end → miss
      {reinterpret_cast<void*>(0x40000000ULL), 0, 1, false, 0},  // before base
      {reinterpret_cast<void*>(0x50000000ULL), -1, 1, false, 0},
      {reinterpret_cast<void*>(0x50000000ULL), NCCL_GIN_ANVIL_IPC_MAX_RANKS, 1, false, 0},
      {reinterpret_cast<void*>(0x50000000ULL), 0, 0, false, 0},
  };

  DeviceBuffer<ResolveCase> d_cases(static_cast<size_t>(cases.size()));
  DeviceBuffer<void*> d_out(cases.size());
  d_cases.copyFrom(cases);
  d_out.zero();

  kernelResolvePeerVa<<<gridFor(cases.size()), kDefaultBlockSize>>>(d_table.ptr, d_cases.ptr,
                                                                     d_out.ptr,
                                                                     static_cast<int>(cases.size()));
  syncAndCheck();

  auto results = d_out.copyTo();
  for (size_t i = 0; i < cases.size(); ++i) {
    if (cases[i].expectHit) {
      EXPECT_EQ(reinterpret_cast<uintptr_t>(results[i]), cases[i].expectRemote) << "case " << i;
    } else {
      EXPECT_EQ(results[i], nullptr) << "case " << i;
    }
  }

  // Null table branch.
  DeviceBuffer<void*> d_nullOut(1);
  d_nullOut.zero();
  kernelResolvePeerVa<<<1, 1>>>(nullptr, d_cases.ptr, d_nullOut.ptr, 0);
  syncAndCheck();
}

// ---------------------------------------------------------------------------
// Suite C: ipcPut / ipcPutScalar / ipcPutRemainder
// ---------------------------------------------------------------------------

__global__ void kernelIpcPutRoundtrip(uint8_t* dst, const uint8_t* src, int nbytes) {
  nccl::gin::anvil::ipcPut(dst, src, static_cast<size_t>(nbytes));
}

TEST_F(GinAnvilIpcDeviceTest, IpcPut_RoundtripSizes) {
  constexpr int kMax = 24;
  std::vector<uint8_t> pattern(kMax);
  for (int i = 0; i < kMax; ++i) pattern[static_cast<size_t>(i)] = static_cast<uint8_t>(0xA0 + i);

  DeviceBuffer<uint8_t> d_src(kMax);
  DeviceBuffer<uint8_t> d_dst(kMax);
  d_src.copyFrom(pattern);

  for (int n : {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24}) {
    d_dst.zero();
    kernelIpcPutRoundtrip<<<1, 1>>>(d_dst.ptr, d_src.ptr, n);
    syncAndCheck();
    auto got = d_dst.copyTo();
    for (int i = 0; i < n; ++i) {
      EXPECT_EQ(got[static_cast<size_t>(i)], pattern[static_cast<size_t>(i)]) << "n=" << n << " i=" << i;
    }
    for (int i = n; i < kMax; ++i) {
      EXPECT_EQ(got[static_cast<size_t>(i)], 0) << "tail n=" << n;
    }
  }
}

__global__ void kernelIpcPutScalar(uint8_t* dst, uint64_t val, int nbytes) {
  nccl::gin::anvil::ipcPutScalar(dst, &val, static_cast<size_t>(nbytes));
}

TEST_F(GinAnvilIpcDeviceTest, IpcPutScalar_Sizes) {
  constexpr uint64_t kVal = 0x0123456789ABCDEFULL;
  DeviceBuffer<uint8_t> d_dst(8);
  for (int n = 1; n <= 8; ++n) {
    d_dst.zero();
    kernelIpcPutScalar<<<1, 1>>>(d_dst.ptr, kVal, n);
    syncAndCheck();
    auto got = d_dst.copyTo();
    for (int i = 0; i < n; ++i) {
      uint8_t expect = static_cast<uint8_t>((kVal >> (8 * i)) & 0xFF);
      EXPECT_EQ(got[static_cast<size_t>(i)], expect) << "n=" << n << " byte=" << i;
    }
  }
}

__global__ void kernelIpcAtomicAdd(uint64_t* dst, uint64_t addend) {
  nccl::gin::anvil::detail::ipcFlatAtomicAddSys64(dst, addend);
}

TEST_F(GinAnvilIpcDeviceTest, IpcFlatAtomicAddSys64) {
  DeviceBuffer<uint64_t> d_val(1);
  uint64_t zero = 0;
  d_val.copyFrom(&zero, 1);
  kernelIpcAtomicAdd<<<1, 1>>>(d_val.ptr, 7ULL);
  syncAndCheck();
  EXPECT_EQ(d_val.download(), 7ULL);
  kernelIpcAtomicAdd<<<1, 1>>>(d_val.ptr, 5ULL);
  syncAndCheck();
  EXPECT_EQ(d_val.download(), 12ULL);
}

// ---------------------------------------------------------------------------
// Suite D: detail helpers (included via gin_anvil_sdma.h when enabled)
// ---------------------------------------------------------------------------

#if NCCL_GIN_ANVIL_SDMA_ENABLE

using nccl::gin::anvil::detail::anvilCtxValid;
using nccl::gin::anvil::detail::effectiveChannel;
using nccl::gin::anvil::detail::markSdmaDirty;
using nccl::gin::anvil::detail::anvilSignalPtrOrDummy;
using nccl::gin::anvil::detail::remoteSignalAddr;
using nccl::gin::anvil::detail::useSdmaFusedSignal;
using nccl::gin::anvil::detail::fenceBeforeSignal;

__global__ void kernelDetailHelpers(const ncclGinAnvilSdmaGPUContext* ctx, int blockId,
                                    int* effChOut, bool* fusedOut, uint64_t* dirtyOut) {
  if (threadIdx.x != 0) return;
  ncclGinAnvilSdmaGPUContext* mut =
      const_cast<ncclGinAnvilSdmaGPUContext*>(ctx);
  effChOut[0] = effectiveChannel(mut, blockId);
  fusedOut[0] = useSdmaFusedSignal(mut, true, true, false, ncclGinSignalInc);
  markSdmaDirty(mut, 1, mut->numChannels, effChOut[0]);
  if (mut->sdmaDirty) {
    dirtyOut[0] = __hip_atomic_load(mut->sdmaDirty, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}

TEST_F(GinAnvilIpcDeviceTest, DetailHelpers_ChannelAndDirty) {
  ncclGinAnvilSdmaGPUContext hostCtx{};
  hostCtx.layoutMagic = NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC;
  hostCtx.numChannels = 4;
  hostCtx.sdmaChannel = 0;
  hostCtx.sdmaChannelStride = 1;
  hostCtx.fusedSdmaSignal = 1;
  hostCtx.signals = reinterpret_cast<uint64_t*>(0x1);  // non-null for fused predicate

  DeviceBuffer<uint64_t> d_dirty(1);
  d_dirty.zero();
  hostCtx.sdmaDirty = d_dirty.ptr;

  DeviceBuffer<ncclGinAnvilSdmaGPUContext> d_ctx(1);
  d_ctx.upload(hostCtx);

  DeviceBuffer<int> d_eff(1);
  DeviceBuffer<bool> d_fused(1);
  DeviceBuffer<uint64_t> d_dirtyRead(1);
  d_eff.zero();
  d_fused.zero();
  d_dirtyRead.zero();

  kernelDetailHelpers<<<1, 1>>>(d_ctx.ptr, 128, d_eff.ptr, d_fused.ptr, d_dirtyRead.ptr);
  syncAndCheck();

  EXPECT_EQ(d_eff.download(), 2);  // (0 + 1*(128/64)) % 4 == 2
  hipDeviceProp_t prop{};
  ASSERT_EQ(hipSuccess, hipGetDeviceProperties(&prop, 0));
  const bool oss7 = (std::strstr(prop.gcnArchName, "gfx950") != nullptr);
  EXPECT_EQ(d_fused.download(), oss7);
  EXPECT_NE(d_dirtyRead.download(), 0ULL);

  // stride 0 → always channel 0
  hostCtx.sdmaChannelStride = 0;
  d_ctx.upload(hostCtx);
  kernelDetailHelpers<<<1, 1>>>(d_ctx.ptr, 999, d_eff.ptr, d_fused.ptr, d_dirtyRead.ptr);
  syncAndCheck();
  EXPECT_EQ(d_eff.download(), 0);

  // null dirty guard
  hostCtx.sdmaDirty = nullptr;
  d_ctx.upload(hostCtx);
  kernelDetailHelpers<<<1, 1>>>(d_ctx.ptr, 0, d_eff.ptr, d_fused.ptr, d_dirtyRead.ptr);
  syncAndCheck();
}

__global__ void kernelAnvilCtxValid(bool* outValid, bool* outSignalNonNull) {
  if (threadIdx.x != 0) return;
  outValid[0] = anvilCtxValid(nullptr);

  ncclGinAnvilSdmaGPUContext bad{};
  bad.layoutMagic = 0;
  outValid[1] = anvilCtxValid(&bad);

  ncclGinAnvilSdmaGPUContext good{};
  good.layoutMagic = NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC;
  outValid[2] = anvilCtxValid(&good);

  outSignalNonNull[0] = (anvilSignalPtrOrDummy(nullptr, 0) != nullptr);
  outSignalNonNull[1] = (anvilSignalPtrOrDummy(&good, 0) != nullptr);

  good.signals = nullptr;
  outSignalNonNull[2] = (anvilSignalPtrOrDummy(&good, 0) != nullptr);

  uint64_t sigs[2] = {0, 0};
  good.signals = sigs;
  outSignalNonNull[3] = (anvilSignalPtrOrDummy(&good, 1) == sigs + 1);

  outSignalNonNull[4] = (remoteSignalAddr(&good, 0, 0) == nullptr);
}

TEST_F(GinAnvilIpcDeviceTest, AnvilCtxValid_AndSignalPtr) {
  DeviceBuffer<bool> d_valid(3);
  DeviceBuffer<bool> d_sig(5);
  d_valid.zero();
  d_sig.zero();
  kernelAnvilCtxValid<<<1, 1>>>(d_valid.ptr, d_sig.ptr);
  syncAndCheck();
  auto valid = d_valid.copyTo();
  auto sig = d_sig.copyTo();
  EXPECT_FALSE(valid[0]);
  EXPECT_FALSE(valid[1]);
  EXPECT_TRUE(valid[2]);
  for (bool b : sig) EXPECT_TRUE(b);
  EXPECT_TRUE(sig[4]);  // remoteSignalAddr null
}

__global__ void kernelFencePaths(bool sdmaPath, bool hasCounter) {
  if (threadIdx.x != 0) return;
  fenceBeforeSignal(sdmaPath, nullptr, hasCounter);
}

TEST_F(GinAnvilIpcDeviceTest, FenceBeforeSignal_CompilesAllPaths) {
  for (bool sdma : {false, true}) {
    for (bool ctr : {false, true}) {
      kernelFencePaths<<<1, 1>>>(sdma, ctr);
      syncAndCheck();
    }
  }
}

// ---------------------------------------------------------------------------
// Suite E: template IPC-only Put / Flush (no real SDMA queue)
// ---------------------------------------------------------------------------

struct PutHarness {
  ncclGinAnvilSdmaGPUContext ctx;
  ncclGinAnvilSdmaMemHandle dstMh;
  ncclGinAnvilSdmaMemHandle srcMh;
  ncclGinAnvilIpcBufEntry ipcEntry;
};

__global__ void kernelPutIpcSmall(PutHarness* h, uint8_t* src, uint8_t* dst, int nbytes) {
  if (threadIdx.x != 0) return;
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ginCtx.rank = 0;

  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_NONE;

  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(
      ginCtx, ncclCoopThread{}, 1, true, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, static_cast<size_t>(nbytes), sig,
      ncclGinSignalInc, 0, false, 0, false, nullptr, cuda::thread_scope_system,
      cuda::thread_scope_system);
}

TEST_F(GinAnvilIpcDeviceTest, Put_IpcSmallMessage) {
  constexpr int kBytes = 32;
  std::vector<uint8_t> srcPat(kBytes);
  for (int i = 0; i < kBytes; ++i) srcPat[static_cast<size_t>(i)] = static_cast<uint8_t>(i);

  DeviceBuffer<uint8_t> d_src(static_cast<size_t>(kBytes));
  DeviceBuffer<uint8_t> d_dst(static_cast<size_t>(kBytes));
  d_src.copyFrom(srcPat);
  d_dst.zero();

  PutHarness host{};
  host.ctx.layoutMagic = NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC;
  host.ctx.sdmaThreshold = 128;
  host.ctx.numChannels = 1;
  host.ctx.queueHandles = nullptr;  // forces IPC for large; small uses IPC anyway

  host.ipcEntry.local_base = reinterpret_cast<uintptr_t>(d_dst.ptr);
  host.ipcEntry.length = kBytes;
  host.ipcEntry.remote_bases[1] = reinterpret_cast<uintptr_t>(d_dst.ptr);

  DeviceBuffer<ncclGinAnvilIpcBufEntry> d_entry(1);
  d_entry.upload(host.ipcEntry);
  host.ctx.ipcTable = d_entry.ptr;
  host.ctx.ipcTableCount = 1;

  host.dstMh.baseAddr = reinterpret_cast<uintptr_t>(d_dst.ptr);
  host.srcMh.baseAddr = reinterpret_cast<uintptr_t>(d_src.ptr);

  DeviceBuffer<PutHarness> d_h(1);
  d_h.upload(host);

  kernelPutIpcSmall<<<1, 1>>>(d_h.ptr, d_src.ptr, d_dst.ptr, kBytes);
  syncAndCheck();

  auto got = d_dst.copyTo();
  for (int i = 0; i < kBytes; ++i) {
    EXPECT_EQ(got[static_cast<size_t>(i)], srcPat[static_cast<size_t>(i)]);
  }
}

__global__ void kernelFlushDirty(PutHarness* h, uint64_t* dirty) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  h->ctx.sdmaDirty = dirty;
  h->ctx.queueHandles = nullptr;
  ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>::call(ginCtx, ncclCoopThread{},
                                                         cuda::memory_order_seq_cst, nullptr);
}

TEST_F(GinAnvilIpcDeviceTest, Flush_InvalidAndDirtyPaths) {
  PutHarness host{};
  host.ctx.layoutMagic = 0;  // invalid
  DeviceBuffer<PutHarness> d_h(1);
  d_h.upload(host);

  ncclGinCtx ginCtx{};
  ginCtx.handle = &host.ctx;
  // Invalid ctx flush runs threadfence only — compile-time coverage via kernel:
  kernelFlushDirty<<<1, 1>>>(d_h.ptr, nullptr);
  syncAndCheck();

  host.ctx.layoutMagic = NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC;
  host.ctx.numChannels = 1;
  DeviceBuffer<uint64_t> d_dirty(1);
  uint64_t dirtyVal = 1ULL;
  d_dirty.copyFrom(&dirtyVal, 1);
  d_h.upload(host);
  kernelFlushDirty<<<1, 32>>>(d_h.ptr, d_dirty.ptr);
  syncAndCheck();
  EXPECT_EQ(d_dirty.download(), 0ULL);
}

#endif  // NCCL_GIN_ANVIL_SDMA_ENABLE

}  // namespace RcclUnitTesting
