/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// HIP device tests for nccl_device/gin/anvil/gin_anvil.h (GIN_ANVIL).
// This translation unit is compiled with ENABLE_ROCSHMEM_GIN (see test/CMakeLists.txt)
// so NCCL_GIN_ANVIL_ENABLE is 1 regardless of how librccl.so was built.

#include "DeviceTestBase.hpp"

#include "nccl_device/coop.h"
#include "nccl_device/gin/anvil/gin_anvil.h"

#include <cstdint>
#include <cstring>

namespace RcclUnitTesting
{

class GinAnvilDeviceTest : public DeviceTestBase {};

// --- RankPtr -----------------------------------------------------------------

__global__ void kRankPtr(uint64_t* out, uintptr_t base, uint64_t stride, int rank) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  out[0] = (uint64_t)ncclGinAnvilRankPtr(base, stride, rank);
}

TEST_F(GinAnvilDeviceTest, RankPtr_Stride) {
  DeviceBuffer<uint64_t> d_out(1);
  d_out.zero();
  constexpr uintptr_t kBase = 0x10000;
  constexpr uint64_t kStride = 256;
  kRankPtr<<<1, 1>>>(d_out.ptr, kBase, kStride, 3);
  syncAndCheck();
  EXPECT_EQ(d_out.download(), (uint64_t)(kBase + 3 * kStride));
}

// --- Memcpy ------------------------------------------------------------------

__global__ void kMemcpyPattern(uint8_t* dst, uint8_t const* src, size_t n) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  ncclGinAnvilMemcpy(dst, src, n);
}

TEST_F(GinAnvilDeviceTest, Memcpy_ZeroBytes) {
  std::vector<uint8_t> h_dst(4, 0xAB), h_src(4, 0xCD);
  DeviceBuffer<uint8_t> d_dst(4), d_src(4);
  d_dst.copyFrom(h_dst);
  d_src.copyFrom(h_src);
  kMemcpyPattern<<<1, 1>>>(d_dst.ptr, d_src.ptr, 0);
  syncAndCheck();
  EXPECT_EQ(d_dst.copyTo(), h_dst);
}

TEST_F(GinAnvilDeviceTest, Memcpy_UnalignedSmall) {
  std::vector<uint8_t> h_src = {0x01, 0x02, 0x03};
  std::vector<uint8_t> h_dst(3, 0);
  DeviceBuffer<uint8_t> d_src(3), d_dst(3);
  d_src.copyFrom(h_src);
  d_dst.zero();
  kMemcpyPattern<<<1, 1>>>(d_dst.ptr, d_src.ptr, 3);
  syncAndCheck();
  EXPECT_EQ(d_dst.copyTo(), h_src);
}

TEST_F(GinAnvilDeviceTest, Memcpy_AlignedUint64Path) {
  alignas(8) uint8_t h_src[16], h_dst[16];
  for (int i = 0; i < 16; ++i) h_src[i] = static_cast<uint8_t>(i + 1);
  std::memset(h_dst, 0, sizeof(h_dst));
  DeviceBuffer<uint8_t> d_src(16), d_dst(16);
  d_src.copyFrom(h_src, 16);
  d_dst.copyFrom(h_dst, 16);
  kMemcpyPattern<<<1, 1>>>(d_dst.ptr, d_src.ptr, 16);
  syncAndCheck();
  std::vector<uint8_t> got = d_dst.copyTo();
  for (int i = 0; i < 16; ++i) EXPECT_EQ(got[i], h_src[i]);
}

// --- SdmaPutChunks (bytes == 0 only) -----------------------------------------

__global__ void kSdmaPutChunksNoop(rocshmem::anvil::SdmaQueueDeviceHandle* q) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  ncclGinAnvilSdmaPutChunks(q, nullptr, nullptr, 0, 12345);
}

TEST_F(GinAnvilDeviceTest, SdmaPutChunks_ZeroBytes) {
  DeviceBuffer<rocshmem::anvil::SdmaQueueDeviceHandle> d_q(1);
  d_q.zero();
  kSdmaPutChunksNoop<<<1, 1>>>(d_q.ptr);
  syncAndCheck();
}

// --- SelectSdmaChannel -------------------------------------------------------

__global__ void kSelectChannel(int* out, ncclGinAnvilGPUContext const* ctxArr, ncclGinCtx ctx) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  ncclGinAnvilGPUContext* aCtx = const_cast<ncclGinAnvilGPUContext*>(ctxArr);
  out[0] = ncclGinAnvilSelectSdmaChannel(ctx, aCtx);
}

TEST_F(GinAnvilDeviceTest, SelectSdmaChannel_NumChEdgeCases) {
  DeviceBuffer<int> d_out(2);
  DeviceBuffer<ncclGinAnvilGPUContext> d_ctx(2);
  std::vector<ncclGinAnvilGPUContext> h_ctx(2);
  h_ctx[0].numSdmaChannels = 1;
  h_ctx[1].numSdmaChannels = 4;
  d_ctx.copyFrom(h_ctx);

  ncclGinCtx gin0{};
  gin0.contextId = 0;
  ncclGinCtx gin1{};
  gin1.contextId = 5;

  kSelectChannel<<<1, 1>>>(d_out.ptr, d_ctx.ptr, gin0);
  kSelectChannel<<<1, 1>>>(d_out.ptr + 1, d_ctx.ptr + 1, gin1);
  syncAndCheck();

  std::vector<int> got = d_out.copyTo();
  EXPECT_EQ(got[0], 0);
  EXPECT_EQ(got[1], 1);
}

// --- PeerQueue ---------------------------------------------------------------

__global__ void kPeerQueueAddr(uint64_t* out, ncclGinAnvilGPUContext const* ctxArr, int peer, int ch) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  auto* q = ncclGinAnvilPeerQueue(const_cast<ncclGinAnvilGPUContext*>(ctxArr), peer, ch);
  out[0] = reinterpret_cast<uint64_t>(q);
}

TEST_F(GinAnvilDeviceTest, PeerQueue_NullQueues) {
  DeviceBuffer<uint64_t> d_out(1);
  DeviceBuffer<ncclGinAnvilGPUContext> d_ctx(1);
  ncclGinAnvilGPUContext h{};
  h.queues = nullptr;
  h.numSdmaChannels = 2;
  d_ctx.upload(h);
  kPeerQueueAddr<<<1, 1>>>(d_out.ptr, d_ctx.ptr, 0, 0);
  syncAndCheck();
  EXPECT_EQ(d_out.download(), 0u);
}

TEST_F(GinAnvilDeviceTest, PeerQueue_NumChClampLoadsNullSlot) {
  DeviceBuffer<uint64_t> d_out(1);
  DeviceBuffer<ncclGinAnvilGPUContext> d_ctx(1);
  DeviceBuffer<void*> d_slots(1);
  d_slots.zero();

  ncclGinAnvilGPUContext h{};
  h.queues = d_slots.ptr;
  h.numSdmaChannels = 0;
  d_ctx.upload(h);

  kPeerQueueAddr<<<1, 1>>>(d_out.ptr, d_ctx.ptr, 0, 0);
  syncAndCheck();
  EXPECT_EQ(d_out.download(), 0u);
}

// --- MarkSdmaDirty ------------------------------------------------------------

__global__ void kMarkDirty(ncclGinAnvilGPUContext* ctx, int peer) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  ctx->sdmaDirtyMask = 0;
  ncclGinAnvilMarkSdmaDirty(ctx, peer);
}

TEST_F(GinAnvilDeviceTest, MarkSdmaDirty_PeerBounds) {
  DeviceBuffer<ncclGinAnvilGPUContext> d_ctx(1);
  d_ctx.zero();
  kMarkDirty<<<1, 1>>>(d_ctx.ptr, 3);
  syncAndCheck();
  ncclGinAnvilGPUContext h = d_ctx.download();
  EXPECT_EQ(h.sdmaDirtyMask, 1u << 3);

  d_ctx.zero();
  kMarkDirty<<<1, 1>>>(d_ctx.ptr, 31);
  syncAndCheck();
  h = d_ctx.download();
  EXPECT_EQ(h.sdmaDirtyMask, 1u << 31);

  d_ctx.zero();
  kMarkDirty<<<1, 1>>>(d_ctx.ptr, 32);
  syncAndCheck();
  h = d_ctx.download();
  EXPECT_EQ(h.sdmaDirtyMask, 0u);
}

// --- Local / remote signal ops -----------------------------------------------

__global__ void kLocalSignalOps(uint64_t* sig, int mode) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (mode == 0) {
    ncclGinAnvilLocalSignalOp(nullptr, ncclGinSignalInc, 99);
    return;
  }
  *sig = 10;
  if (mode == 1)
    ncclGinAnvilLocalSignalOp(sig, ncclGinSignalInc, 12345);
  else
    ncclGinAnvilLocalSignalOp(sig, ncclGinSignalAdd, 7);
}

TEST_F(GinAnvilDeviceTest, LocalSignalOp_Branches) {
  DeviceBuffer<uint64_t> d_sig(1);
  d_sig.zero();
  kLocalSignalOps<<<1, 1>>>(d_sig.ptr, 0);
  syncAndCheck();
  EXPECT_EQ(d_sig.download(), 0ULL);

  kLocalSignalOps<<<1, 1>>>(d_sig.ptr, 1);
  syncAndCheck();
  EXPECT_EQ(d_sig.download(), 11ULL);

  kLocalSignalOps<<<1, 1>>>(d_sig.ptr, 2);
  syncAndCheck();
  EXPECT_EQ(d_sig.download(), 18ULL);
}

__global__ void kRemoteSignalOps(uint64_t* sig, int mode) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (mode == 0) {
    ncclGinAnvilRemoteGpuSignalOp(nullptr, ncclGinSignalInc, 1);
    return;
  }
  *sig = 5;
  if (mode == 1)
    ncclGinAnvilRemoteGpuSignalOp(sig, ncclGinSignalInc, 99);
  else
    ncclGinAnvilRemoteGpuSignalOp(sig, ncclGinSignalAdd, 4);
}

TEST_F(GinAnvilDeviceTest, RemoteGpuSignalOp_Branches) {
  DeviceBuffer<uint64_t> d_sig(1);
  kRemoteSignalOps<<<1, 1>>>(d_sig.ptr, 0);
  syncAndCheck();
  EXPECT_EQ(d_sig.download(), 0ULL);

  kRemoteSignalOps<<<1, 1>>>(d_sig.ptr, 1);
  syncAndCheck();
  EXPECT_EQ(d_sig.download(), 6ULL);

  kRemoteSignalOps<<<1, 1>>>(d_sig.ptr, 2);
  syncAndCheck();
  EXPECT_EQ(d_sig.download(), 10ULL);
}

// --- ncclGinApi_Put ----------------------------------------------------------

__device__ static void fillSignalDescNone(ncclGinSignalDescriptor& s) {
  s.type = NCCL_GIN_SIGNAL_TYPE_NONE;
}

__device__ static void fillSignalDescIndexed(ncclGinSignalDescriptor& s, ncclGinSignal_t id) {
  s.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  s.indexedSignal.signalId = id;
}

__global__ void kPut(ncclGinCtx ctx, int peer, bool hasWins, ncclGinWindow_t dstWin, size_t dstOff,
                     ncclGinWindow_t srcWin, size_t srcOff, size_t bytes, int signalMode,
                     ncclGinSignalOp_t signalOp, uint64_t signalOpArg, bool hasCounter,
                     ncclGinCounter_t counterId) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  ncclGinSignalDescriptor sig{};
  if (signalMode == 1) fillSignalDescIndexed(sig, 0);
  else fillSignalDescNone(sig);

  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL>::call(
      ctx,
      ncclCoopCta{},
      peer,
      hasWins,
      dstWin,
      dstOff,
      srcWin,
      srcOff,
      bytes,
      sig,
      signalOp,
      signalOpArg,
      hasCounter,
      counterId,
      false,
      nullptr,
      cuda::thread_scope_device,
      cuda::thread_scope_device,
      ncclGinOptFlagsDefault);
}

TEST_F(GinAnvilDeviceTest, Put_Self_HandleNull) {
  ncclGinCtx h{};
  h.handle = nullptr;
  h.contextId = 0;
  h.rank = 0;
  h.nRanks = 1;
  kPut<<<1, 1>>>(h, 0, false, nullptr, 0, nullptr, 0, 0, 0, ncclGinSignalInc, 1, false, 0);
  syncAndCheck();
}

TEST_F(GinAnvilDeviceTest, Put_Self_NoWins_NoSignal) {
  DeviceBuffer<ncclGinAnvilGPUContext> d_gctx(1);
  d_gctx.zero();
  ncclGinCtx gin{};
  gin.handle = d_gctx.ptr;
  gin.contextId = 0;
  gin.rank = 0;
  gin.nRanks = 1;
  kPut<<<1, 1>>>(gin, 0, false, nullptr, 0, nullptr, 0, 0, 0, ncclGinSignalInc, 1, false, 0);
  syncAndCheck();
}

TEST_F(GinAnvilDeviceTest, Put_Self_NoWins_LocalSignal) {
  DeviceBuffer<uint64_t> d_sig(4);
  d_sig.zero();
  DeviceBuffer<ncclGinAnvilGPUContext> d_gctx(1);
  ncclGinAnvilGPUContext hg{};
  hg.signals = d_sig.ptr;
  d_gctx.upload(hg);

  ncclGinCtx gin{};
  gin.handle = d_gctx.ptr;
  gin.contextId = 0;
  gin.rank = 0;
  gin.nRanks = 1;
  kPut<<<1, 1>>>(gin, 0, false, nullptr, 0, nullptr, 0, 0, 1, ncclGinSignalAdd, 2, false, 0);
  syncAndCheck();
  EXPECT_EQ(d_sig.download(), 2ULL);
}

TEST_F(GinAnvilDeviceTest, Put_Self_Wins_NullWindow) {
  DeviceBuffer<ncclGinAnvilGPUContext> d_gctx(1);
  d_gctx.zero();
  ncclGinCtx gin{};
  gin.handle = d_gctx.ptr;
  gin.contextId = 0;
  gin.rank = 0;
  gin.nRanks = 1;
  kPut<<<1, 1>>>(gin, 0, true, nullptr, 0, nullptr, 0, 4, 0, ncclGinSignalInc, 1, false, 0);
  syncAndCheck();
}

TEST_F(GinAnvilDeviceTest, Put_Self_Wins_InvalidBases) {
  DeviceBuffer<ncclGinAnvilMemHandle> d_dst(1), d_src(1);
  ncclGinAnvilMemHandle mh{};
  mh.lsaRank0Base = 0;
  mh.lsaStrideBytes = 4096;
  d_dst.upload(mh);
  mh.lsaRank0Base = 0x1000;
  d_src.upload(mh);

  DeviceBuffer<ncclGinAnvilGPUContext> d_gctx(1);
  d_gctx.zero();

  ncclGinCtx gin{};
  gin.handle = d_gctx.ptr;
  gin.contextId = 0;
  gin.rank = 0;
  gin.nRanks = 1;
  kPut<<<1, 1>>>(gin, 0, true, d_dst.ptr, 0, d_src.ptr, 0, 4, 0, ncclGinSignalInc, 1, false, 0);
  syncAndCheck();
}

TEST_F(GinAnvilDeviceTest, Put_Self_Wins_MemcpyAndOptionalSignal) {
  constexpr size_t kStride = 8192;
  DeviceBuffer<uint8_t> d_dst(16), d_src(16);
  std::vector<uint8_t> pat(16);
  for (size_t i = 0; i < 16; ++i) pat[i] = static_cast<uint8_t>(0xA0 + i);
  d_src.copyFrom(pat);
  d_dst.zero();

  DeviceBuffer<ncclGinAnvilMemHandle> d_dmh(1), d_smh(1);
  ncclGinAnvilMemHandle dh{};
  dh.lsaRank0Base = reinterpret_cast<uintptr_t>(d_dst.ptr);
  dh.lsaStrideBytes = kStride;
  ncclGinAnvilMemHandle sh{};
  sh.lsaRank0Base = reinterpret_cast<uintptr_t>(d_src.ptr);
  sh.lsaStrideBytes = kStride;
  d_dmh.upload(dh);
  d_smh.upload(sh);

  DeviceBuffer<uint64_t> d_sig(4);
  d_sig.zero();
  DeviceBuffer<uint64_t> d_ctr(4);
  d_ctr.zero();

  DeviceBuffer<ncclGinAnvilGPUContext> d_gctx(1);
  ncclGinAnvilGPUContext hg{};
  hg.signals = d_sig.ptr;
  hg.counters = d_ctr.ptr;
  d_gctx.upload(hg);

  ncclGinCtx gin{};
  gin.handle = d_gctx.ptr;
  gin.contextId = 0;
  gin.rank = 0;
  gin.nRanks = 1;

  kPut<<<1, 1>>>(gin, 0, true, d_dmh.ptr, 0, d_smh.ptr, 0, 16, 1, ncclGinSignalInc, 1, true, 0);
  syncAndCheck();
  EXPECT_EQ(d_dst.copyTo(), pat);
  EXPECT_EQ(d_sig.download(), 1ULL);
  EXPECT_EQ(d_ctr.download(), 1ULL);
}

TEST_F(GinAnvilDeviceTest, IndexedSignalId_OnlyWhenIndexed) {
  DeviceBuffer<uint64_t> d_sigNone(4), d_sigIdx(4);
  d_sigNone.zero();
  d_sigIdx.zero();

  DeviceBuffer<ncclGinAnvilGPUContext> d_gctx(2);
  std::vector<ncclGinAnvilGPUContext> hg(2);
  hg[0].signals = d_sigNone.ptr;
  hg[1].signals = d_sigIdx.ptr;
  d_gctx.copyFrom(hg);

  ncclGinCtx gin0{};
  gin0.handle = d_gctx.ptr;
  gin0.contextId = 0;
  gin0.rank = 0;
  gin0.nRanks = 1;
  kPut<<<1, 1>>>(gin0, 0, false, nullptr, 0, nullptr, 0, 0, 0, ncclGinSignalInc, 1, false, 0);
  syncAndCheck();
  EXPECT_EQ(d_sigNone.download(), 0ULL);

  ncclGinCtx gin1{};
  gin1.handle = d_gctx.ptr;
  gin1.contextId = 1;
  gin1.rank = 0;
  gin1.nRanks = 1;
  kPut<<<1, 1>>>(gin1, 0, false, nullptr, 0, nullptr, 0, 0, 1, ncclGinSignalInc, 1, false, 0);
  syncAndCheck();
  EXPECT_EQ(d_sigIdx.download(), 1ULL);
}

// --- Cross-peer Put (dummy non-null queue for control flow when required) ----

__global__ void kFillPeerQueues(ncclGinAnvilGPUContext* ctx, void** slots, int nRanks, int numCh,
                                uintptr_t dummyQ, int myRank) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (int pe = 0; pe < nRanks; ++pe) {
    for (int ch = 0; ch < numCh; ++ch) {
      int slot = pe * numCh + ch;
      if (pe == myRank) slots[slot] = nullptr;
      else slots[slot] = dummyQ ? reinterpret_cast<void*>(dummyQ) : nullptr;
    }
  }
  ctx->queues = slots;
  ctx->numSdmaChannels = (uint32_t)numCh;
  ctx->rank = myRank;
}

TEST_F(GinAnvilDeviceTest, Put_Peer_QueueNull) {
  constexpr int nRanks = 2;
  constexpr int numCh = 1;
  DeviceBuffer<void*> d_slots((size_t)nRanks * (size_t)numCh);
  d_slots.zero();
  DeviceBuffer<ncclGinAnvilGPUContext> d_gctx(1);
  d_gctx.zero();

  kFillPeerQueues<<<1, 1>>>(d_gctx.ptr, d_slots.ptr, nRanks, numCh, /*dummyQ=*/0, /*myRank=*/0);
  syncAndCheck();

  ncclGinAnvilGPUContext hg = d_gctx.download();
  hg.nRanks = nRanks;
  d_gctx.upload(hg);

  ncclGinCtx gin{};
  gin.handle = d_gctx.ptr;
  gin.contextId = 0;
  gin.rank = 0;
  gin.nRanks = nRanks;

  kPut<<<1, 1>>>(gin, 1, false, nullptr, 0, nullptr, 0, 0, 0, ncclGinSignalInc, 1, false, 0);
  syncAndCheck();
}

TEST_F(GinAnvilDeviceTest, Put_Peer_HasSignal_SigPtrNull) {
  constexpr int nRanks = 2;
  constexpr int numCh = 1;
  DeviceBuffer<void*> d_slots((size_t)nRanks * (size_t)numCh);
  DeviceBuffer<uint64_t*> d_sigBase((size_t)nRanks);
  std::vector<uint64_t*> hb((size_t)nRanks, nullptr);
  d_sigBase.copyFrom(hb);

  DeviceBuffer<ncclGinAnvilGPUContext> d_gctx(1);
  d_gctx.zero();

  uintptr_t kDummy = 0xFEEDu;
  kFillPeerQueues<<<1, 1>>>(d_gctx.ptr, d_slots.ptr, nRanks, numCh, kDummy, /*myRank=*/0);
  syncAndCheck();

  ncclGinAnvilGPUContext hg = d_gctx.download();
  hg.nRanks = nRanks;
  hg.signalsBase = d_sigBase.ptr;
  d_gctx.upload(hg);

  ncclGinCtx gin{};
  gin.handle = d_gctx.ptr;
  gin.contextId = 0;
  gin.rank = 0;
  gin.nRanks = nRanks;

  kPut<<<1, 1>>>(gin, 1, false, nullptr, 0, nullptr, 0, 0, 1, ncclGinSignalInc, 1, false, 0);
  syncAndCheck();
}

TEST_F(GinAnvilDeviceTest, Put_Peer_NoWins_RemoteSignal) {
  constexpr int nRanks = 2;
  constexpr int numCh = 1;
  DeviceBuffer<void*> d_slots((size_t)nRanks * (size_t)numCh);
  DeviceBuffer<uint64_t> d_peerSig(4);
  d_peerSig.zero();
  DeviceBuffer<uint64_t*> d_sigBase((size_t)nRanks);
  uint64_t* hostPtrs[nRanks]{};
  hostPtrs[1] = d_peerSig.ptr;
  d_sigBase.copyFrom(hostPtrs, (size_t)nRanks);

  DeviceBuffer<ncclGinAnvilGPUContext> d_gctx(1);
  d_gctx.zero();

  uintptr_t kDummy = 0xFEEDu;
  kFillPeerQueues<<<1, 1>>>(d_gctx.ptr, d_slots.ptr, nRanks, numCh, kDummy, /*myRank=*/0);
  syncAndCheck();

  ncclGinAnvilGPUContext hg = d_gctx.download();
  hg.nRanks = nRanks;
  hg.signalsBase = d_sigBase.ptr;
  hg.signalsContextOffset = 0;
  d_gctx.upload(hg);

  ncclGinCtx gin{};
  gin.handle = d_gctx.ptr;
  gin.contextId = 0;
  gin.rank = 0;
  gin.nRanks = nRanks;

  kPut<<<1, 1>>>(gin, 1, false, nullptr, 0, nullptr, 0, 0, 1, ncclGinSignalAdd, 3, false, 0);
  syncAndCheck();
  EXPECT_EQ(d_peerSig.download(), 3ULL);
}

TEST_F(GinAnvilDeviceTest, Put_Peer_NoWins_NoSignal) {
  constexpr int nRanks = 2;
  constexpr int numCh = 1;
  DeviceBuffer<void*> d_slots((size_t)nRanks * (size_t)numCh);
  uintptr_t kDummy = 0xFEEDu;
  DeviceBuffer<ncclGinAnvilGPUContext> d_gctx(1);
  d_gctx.zero();
  kFillPeerQueues<<<1, 1>>>(d_gctx.ptr, d_slots.ptr, nRanks, numCh, kDummy, /*myRank=*/0);
  syncAndCheck();
  ncclGinAnvilGPUContext hg = d_gctx.download();
  hg.nRanks = nRanks;
  d_gctx.upload(hg);

  ncclGinCtx gin{};
  gin.handle = d_gctx.ptr;
  gin.contextId = 0;
  gin.rank = 0;
  gin.nRanks = nRanks;
  kPut<<<1, 1>>>(gin, 1, false, nullptr, 0, nullptr, 0, 0, 0, ncclGinSignalInc, 1, false, 0);
  syncAndCheck();
}

TEST_F(GinAnvilDeviceTest, Put_Peer_SubThreshold_Memcpy) {
  constexpr int nRanks = 2;
  constexpr int numCh = 1;
  constexpr size_t kStride = 64;
  DeviceBuffer<void*> d_slots((size_t)nRanks * (size_t)numCh);
  uintptr_t kDummy = 0xFEEDu;
  DeviceBuffer<ncclGinAnvilGPUContext> d_gctx(1);
  d_gctx.zero();
  kFillPeerQueues<<<1, 1>>>(d_gctx.ptr, d_slots.ptr, nRanks, numCh, kDummy, /*myRank=*/0);
  syncAndCheck();

  constexpr int kBytes = 64;
  DeviceBuffer<uint8_t> d_dst(kBytes * nRanks), d_src(kBytes * nRanks);
  std::vector<uint8_t> pat((size_t)kBytes);
  for (int i = 0; i < kBytes; ++i) pat[(size_t)i] = static_cast<uint8_t>(i);
  d_src.copyFrom(pat);
  d_dst.zero();

  DeviceBuffer<ncclGinAnvilMemHandle> d_dmh(1), d_smh(1);
  ncclGinAnvilMemHandle dh{};
  dh.lsaRank0Base = reinterpret_cast<uintptr_t>(d_dst.ptr);
  dh.lsaStrideBytes = kStride;
  ncclGinAnvilMemHandle sh{};
  sh.lsaRank0Base = reinterpret_cast<uintptr_t>(d_src.ptr);
  sh.lsaStrideBytes = kStride;
  d_dmh.upload(dh);
  d_smh.upload(sh);

  DeviceBuffer<uint64_t> d_peerSig(4);
  d_peerSig.zero();
  DeviceBuffer<uint64_t*> d_sigBase((size_t)nRanks);
  std::vector<uint64_t*> hp((size_t)nRanks, nullptr);
  hp[1] = d_peerSig.ptr;
  d_sigBase.copyFrom(hp);

  ncclGinAnvilGPUContext hg = d_gctx.download();
  hg.nRanks = nRanks;
  hg.signalsBase = d_sigBase.ptr;
  hg.signalsContextOffset = 0;
  DeviceBuffer<uint64_t> d_ctr(4);
  d_ctr.zero();
  hg.counters = d_ctr.ptr;
  d_gctx.upload(hg);

  ncclGinCtx gin{};
  gin.handle = d_gctx.ptr;
  gin.contextId = 0;
  gin.rank = 0;
  gin.nRanks = nRanks;

  kPut<<<1, 1>>>(gin, 1, true, d_dmh.ptr, 0, d_smh.ptr, 0, (size_t)kBytes, 1, ncclGinSignalInc, 1, true, 0);
  syncAndCheck();

  std::vector<uint8_t> got = d_dst.copyTo();
  for (int i = 0; i < kBytes; ++i) {
    EXPECT_EQ(got[(size_t)kStride + (size_t)i], pat[(size_t)i]) << "i=" << i;
  }
  EXPECT_EQ(d_peerSig.download(), 1ULL);
  EXPECT_EQ(d_ctr.download(), 1ULL);
}

TEST_F(GinAnvilDeviceTest, Put_Peer_Wins_NullWindow) {
  constexpr int nRanks = 2;
  constexpr int numCh = 1;
  DeviceBuffer<void*> d_slots((size_t)nRanks * (size_t)numCh);
  uintptr_t kDummy = 0xFEEDu;
  DeviceBuffer<ncclGinAnvilGPUContext> d_gctx(1);
  d_gctx.zero();
  kFillPeerQueues<<<1, 1>>>(d_gctx.ptr, d_slots.ptr, nRanks, numCh, kDummy, /*myRank=*/0);
  syncAndCheck();
  ncclGinAnvilGPUContext hg = d_gctx.download();
  hg.nRanks = nRanks;
  d_gctx.upload(hg);

  ncclGinCtx gin{};
  gin.handle = d_gctx.ptr;
  gin.contextId = 0;
  gin.rank = 0;
  gin.nRanks = nRanks;
  kPut<<<1, 1>>>(gin, 1, true, nullptr, 0, nullptr, 0, 8, 0, ncclGinSignalInc, 1, false, 0);
  syncAndCheck();
}

TEST_F(GinAnvilDeviceTest, Put_Peer_Wins_InvalidBases) {
  constexpr int nRanks = 2;
  constexpr int numCh = 1;
  DeviceBuffer<void*> d_slots((size_t)nRanks * (size_t)numCh);
  uintptr_t kDummy = 0xFEEDu;
  DeviceBuffer<ncclGinAnvilGPUContext> d_gctx(1);
  d_gctx.zero();
  kFillPeerQueues<<<1, 1>>>(d_gctx.ptr, d_slots.ptr, nRanks, numCh, kDummy, /*myRank=*/0);
  syncAndCheck();

  DeviceBuffer<ncclGinAnvilMemHandle> d_dst(1), d_src(1);
  ncclGinAnvilMemHandle mh{};
  mh.lsaRank0Base = 0;
  mh.lsaStrideBytes = 4096;
  d_dst.upload(mh);
  mh.lsaRank0Base = 0x2000;
  d_src.upload(mh);

  ncclGinAnvilGPUContext hg = d_gctx.download();
  hg.nRanks = nRanks;
  d_gctx.upload(hg);

  ncclGinCtx gin{};
  gin.handle = d_gctx.ptr;
  gin.contextId = 0;
  gin.rank = 0;
  gin.nRanks = nRanks;
  kPut<<<1, 1>>>(gin, 1, true, d_dst.ptr, 0, d_src.ptr, 0, 8, 0, ncclGinSignalInc, 1, false, 0);
  syncAndCheck();
}

// --- GetCounterPtr / ResetCounter / GetSignalPtr / ResetSignal ---------------

__global__ void kGetCounterPtr(uint64_t** out, ncclGinCtx ctx, ncclGinCounter_t id) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  out[0] = ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_ANVIL>::call(ctx, id);
}

__global__ void kResetCounter(ncclGinCtx ctx, ncclGinCounter_t id) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_ANVIL>::call(ctx, id);
}

TEST_F(GinAnvilDeviceTest, GetCounterPtr_And_ResetCounter) {
  DeviceBuffer<uint64_t*> d_out(3);
  d_out.zero();

  ncclGinCtx ginNull{};
  ginNull.handle = nullptr;
  kGetCounterPtr<<<1, 1>>>(d_out.ptr, ginNull, 0);
  syncAndCheck();
  EXPECT_EQ(d_out.copyTo()[0], nullptr);

  DeviceBuffer<ncclGinAnvilGPUContext> d_gctx(1);
  d_gctx.zero();
  ginNull.handle = d_gctx.ptr;
  kGetCounterPtr<<<1, 1>>>(&d_out.ptr[1], ginNull, 0);
  syncAndCheck();
  EXPECT_EQ(d_out.copyTo()[1], nullptr);

  DeviceBuffer<uint64_t> d_ctr(4);
  d_ctr.zero();
  ncclGinAnvilGPUContext hg{};
  hg.counters = d_ctr.ptr;
  d_gctx.upload(hg);
  ginNull.handle = d_gctx.ptr;
  kGetCounterPtr<<<1, 1>>>(&d_out.ptr[2], ginNull, 1);
  syncAndCheck();
  uint64_t* p = d_out.copyTo()[2];
  ASSERT_NE(p, nullptr);

  kResetCounter<<<1, 1>>>(ginNull, 1);
  syncAndCheck();
  std::vector<uint64_t> words = d_ctr.copyTo();
  EXPECT_EQ(words[1], 0u);
}

__global__ void kGetSignalPtr(uint64_t** out, ncclGinCtx ctx, ncclGinSignal_t id) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  out[0] = ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_ANVIL>::call(ctx, id);
}

__global__ void kResetSignal(ncclGinCtx ctx) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  ncclGinSignalDescriptor s{};
  s.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  s.indexedSignal.signalId = 0;
  ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_ANVIL>::call(ctx, s);
}

TEST_F(GinAnvilDeviceTest, GetSignalPtr) {
  DeviceBuffer<uint64_t*> d_out(2);
  d_out.zero();

  DeviceBuffer<ncclGinAnvilGPUContext> d_gctx(1);
  d_gctx.zero();
  ncclGinCtx gin{};
  gin.handle = d_gctx.ptr;
  kGetSignalPtr<<<1, 1>>>(d_out.ptr, gin, 0);
  syncAndCheck();
  EXPECT_EQ(d_out.copyTo()[0], nullptr);

  DeviceBuffer<uint64_t> d_sig(4);
  d_sig.zero();
  ncclGinAnvilGPUContext hg{};
  hg.signals = d_sig.ptr;
  d_gctx.upload(hg);
  gin.handle = d_gctx.ptr;
  kGetSignalPtr<<<1, 1>>>(&d_out.ptr[1], gin, 1);
  syncAndCheck();
  std::vector<uint64_t*> outs = d_out.copyTo();
  ASSERT_NE(outs[1], nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(outs[1]),
            reinterpret_cast<uintptr_t>(d_sig.ptr + 1));
}

TEST_F(GinAnvilDeviceTest, ResetSignal_IsNoOp) {
  DeviceBuffer<ncclGinAnvilGPUContext> d_gctx(1);
  d_gctx.zero();
  ncclGinCtx gin{};
  gin.handle = d_gctx.ptr;
  kResetSignal<<<1, 1>>>(gin);
  syncAndCheck();
}

// --- Flush -------------------------------------------------------------------

__global__ void kFlush(ncclGinCtx ctx, uint32_t* abortFlag) {
  ncclCoopCta coop;
  ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_ANVIL>::call(ctx, coop, cuda::memory_order_seq_cst, abortFlag);
}

TEST_F(GinAnvilDeviceTest, Flush_NullCtx) {
  DeviceBuffer<uint32_t> d_abort(1);
  d_abort.zero();
  ncclGinCtx gin{};
  gin.handle = nullptr;
  gin.nRanks = 4;
  kFlush<<<1, 1>>>(gin, d_abort.ptr);
  syncAndCheck();
}

TEST_F(GinAnvilDeviceTest, Flush_DirtyMaskZero_Collective) {
  DeviceBuffer<ncclGinAnvilGPUContext> d_gctx(1);
  d_gctx.zero();
  ncclGinAnvilGPUContext hg{};
  hg.numSdmaChannels = 2;
  hg.sdmaDirtyMask = 0;
  d_gctx.upload(hg);

  ncclGinCtx gin{};
  gin.handle = d_gctx.ptr;
  gin.rank = 0;
  gin.nRanks = 4;

  DeviceBuffer<uint32_t> d_abort(1);
  d_abort.zero();
  kFlush<<<1, 1>>>(gin, d_abort.ptr);
  syncAndCheck();
  EXPECT_EQ(d_gctx.download().sdmaDirtyMask, 0u);
}

}  // namespace RcclUnitTesting
