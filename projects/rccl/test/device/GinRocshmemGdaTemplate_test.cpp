/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Suite G: GIN rocSHMEM-GDA device template coverage (Put/PutValue/Flush/
// Signal/Counter + edge paths), the GDA analog of Suite H
// (GinAnvilSdmaTemplate_test.cpp). AllToAll drives these ncclGinApi_* template
// specializations, so this suite unit-tests the GDA AllToAll device path
// without a live network.
//
// The real rocshmem::QueuePair device methods (put_nbi/atomic_add/quiet) are
// only declared in queue_pair_device.h (defined in rocSHMEM device bitcode).
// This translation unit supplies inline stub definitions so the suite links
// against no librocshmem device code, mirroring how Suite H shadows the SDMA
// engine with test/device/sdma/anvil_device.hpp.

#include "DeviceTestBase.hpp"

#include "nccl_device/coop.h"
#include "nccl_device/gin/gin_device_host_common.h"
#include "nccl_device/gin/gin_device_common.h"
#include "nccl_device/gin/rocshmem_gda/gin_rocshmem_device_host_common_gda.h"

#if NCCL_GIN_ROCSHMEM_GDA_ENABLE
// Count invocations of the Put/PutValue system-scope fence seam (gin_device_common.h).
// Override must precede gin_rocshmem_gda.h so the templates expand our counter.
__device__ unsigned long long g_gdaStubThreadfenceCount = 0;
#undef NCCL_GIN_THREADFENCE_SYSTEM
#define NCCL_GIN_THREADFENCE_SYSTEM() atomicAdd(&g_gdaStubThreadfenceCount, 1ULL)
#include "nccl_device/gin/rocshmem_gda/gin_rocshmem_gda.h"
#endif

#include <cstdint>
#include <cstring>
#include <vector>

#if NCCL_GIN_ROCSHMEM_GDA_ENABLE

// --------------------------------------------------------------------------
// Stub definitions for rocshmem::QueuePair device methods.
//   put_nbi       : byte-copy laddr -> raddr (observe data landing at remote VA)
//   atomic_add   : atomic add into the remote signal word (observe signal deliver)
//   quiet         : bump a device-global call counter (observe completion sync)
// Non-RDC build: these must be defined in the same TU as the kernels that
// (via the inlined template) call them.
// --------------------------------------------------------------------------
__device__ unsigned long long g_gdaStubQuietCount = 0;
__device__ unsigned long long g_gdaStubPutNbiCount = 0;

namespace rocshmem {

__device__ void QueuePair::put_nbi(void* raddr, uint32_t /*rkey*/, const void* laddr, uint32_t /*lkey*/,
                                   size_t length, ActiveWFInfo& /*wf_info*/, bool /*ring_db*/) {
  if (raddr == nullptr || laddr == nullptr) return;
  atomicAdd(&g_gdaStubPutNbiCount, 1ULL);
  auto* d = static_cast<char*>(raddr);
  const auto* s = static_cast<const char*>(laddr);
  for (size_t i = 0; i < length; ++i) d[i] = s[i];
}

// Matches queue_pair_device.h: atomic_add(void* raddr, uint32_t rkey, int64_t
// value, ActiveWFInfo&, bool fence). The template's signal path calls this
// (gin_rocshmem_gda.h:54,101); rkey/fence are unused by the byte-accurate stub.
__device__ void QueuePair::atomic_add(void* raddr, uint32_t /*rkey*/, int64_t value, ActiveWFInfo& /*wf_info*/,
                                      bool /*fence*/) {
  if (raddr == nullptr) return;
  atomicAdd(reinterpret_cast<unsigned long long*>(raddr), static_cast<unsigned long long>(value));
}

__device__ void QueuePair::quiet(ActiveWFInfo& /*wf_info*/) {
  atomicAdd(&g_gdaStubQuietCount, 1ULL);
}

}  // namespace rocshmem

namespace RcclUnitTesting
{

class GinRocshmemGdaTemplateTest : public DeviceTestBase {};

struct GdaHarness {
  ncclGinRocshmemGdaGPUContext ctx;
  ncclGinRocshmemGdaMemHandle dstMh;
  ncclGinRocshmemGdaMemHandle srcMh;
};

// Bundles all device-side arrays a GDA context/mem-handle points at, and wires
// them into a single uploaded GdaHarness. Peer 1 is the (self-mapped) target:
// its remote_vas entry points back at the local dst buffer and its signal
// remote-address points back at the local signals array, so a self put/signal
// is observable from the host.
class GdaEnv {
public:
  static constexpr int kNRanks = 2;
  static constexpr int kPeer = 1;
  static constexpr uint32_t kNSignals = 4;
  static constexpr uint32_t kNCounters = 2;

  DeviceBuffer<rocshmem::QueuePair> qp{1};
  DeviceBuffer<rocshmem::QueuePair*> qps{kNRanks};
  DeviceBuffer<uint64_t> signals{kNSignals};
  DeviceBuffer<uint64_t> counters{kNCounters};
  DeviceBuffer<uint32_t> signalRkeys{kNRanks};
  DeviceBuffer<uintptr_t> signalRaddrs{kNRanks};
  DeviceBuffer<uintptr_t> dstRemoteVas{kNRanks};
  DeviceBuffer<uint32_t> dstRkeys{kNRanks};
  DeviceBuffer<uint8_t> dst;
  DeviceBuffer<uint8_t> src;
  DeviceBuffer<GdaHarness> dHarness{1};
  GdaHarness host{};

  explicit GdaEnv(size_t bytes) : dst(bytes ? bytes : 1), src(bytes ? bytes : 1) {}

  void build() {
    std::vector<rocshmem::QueuePair*> qpRow(kNRanks, qp.ptr);
    qps.copyFrom(qpRow.data(), kNRanks);

    signals.zero();
    counters.zero();
    signalRkeys.zero();
    dstRkeys.zero();

    std::vector<uintptr_t> sraddr(kNRanks, 0);
    sraddr[kPeer] = reinterpret_cast<uintptr_t>(signals.ptr);  // signal lands in signals[]
    signalRaddrs.copyFrom(sraddr.data(), kNRanks);

    std::vector<uintptr_t> rvas(kNRanks, 0);
    rvas[kPeer] = reinterpret_cast<uintptr_t>(dst.ptr);  // "remote" dst maps to local dst
    dstRemoteVas.copyFrom(rvas.data(), kNRanks);

    std::memset(&host, 0, sizeof(host));
    host.ctx.qps = qps.ptr;
    host.ctx.signals = signals.ptr;
    host.ctx.counters = counters.ptr;
    host.ctx.signal_rkeys = signalRkeys.ptr;
    host.ctx.signal_raddrs = signalRaddrs.ptr;
    host.ctx.nSignals = kNSignals;
    host.ctx.nCounters = kNCounters;
    host.ctx.nRanks = kNRanks;
    host.ctx.rank = 0;

    host.dstMh.local_va = reinterpret_cast<uintptr_t>(dst.ptr);
    host.dstMh.remote_vas = dstRemoteVas.ptr;
    host.dstMh.lkey = 0;
    host.dstMh.rkeys = dstRkeys.ptr;

    host.srcMh.local_va = reinterpret_cast<uintptr_t>(src.ptr);
    host.srcMh.remote_vas = nullptr;
    host.srcMh.lkey = 0;
    host.srcMh.rkeys = nullptr;

    dHarness.upload(host);
  }
};

static void resetQuietCount() {
  unsigned long long z = 0;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_gdaStubQuietCount), &z, sizeof(z)));
}

static unsigned long long readQuietCount() {
  unsigned long long q = 0;
  HIP_EXPECT(hipMemcpyFromSymbol(&q, HIP_SYMBOL(g_gdaStubQuietCount), sizeof(q)));
  return q;
}

static void resetPutNbiCount() {
  unsigned long long z = 0;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_gdaStubPutNbiCount), &z, sizeof(z)));
}

static unsigned long long readPutNbiCount() {
  unsigned long long n = 0;
  HIP_EXPECT(hipMemcpyFromSymbol(&n, HIP_SYMBOL(g_gdaStubPutNbiCount), sizeof(n)));
  return n;
}

static void resetThreadfenceCount() {
  unsigned long long z = 0;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_gdaStubThreadfenceCount), &z, sizeof(z)));
}

static unsigned long long readThreadfenceCount() {
  unsigned long long c = 0;
  HIP_EXPECT(hipMemcpyFromSymbol(&c, HIP_SYMBOL(g_gdaStubThreadfenceCount), sizeof(c)));
  return c;
}

// G1: Put with data (no signal) copies src -> peer's remote buffer.
__global__ void kernelPutData(GdaHarness* h, size_t bytes) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(
      ginCtx, ncclCoopThread{}, 1, true, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, bytes, sig, ncclGinSignalInc, 0, false, 0,
      false, nullptr, cuda::thread_scope_system, cuda::thread_scope_system);
}

TEST_F(GinRocshmemGdaTemplateTest, Put_DataLandsAtRemote) {
  constexpr int kN = 64;
  std::vector<uint8_t> pat(kN);
  for (int i = 0; i < kN; ++i) pat[static_cast<size_t>(i)] = static_cast<uint8_t>(0xA0 + i);
  GdaEnv env(kN);
  env.src.copyFrom(pat);
  env.dst.zero();
  env.build();
  kernelPutData<<<1, 1>>>(env.dHarness.ptr, kN);
  syncAndCheck();
  auto got = env.dst.copyTo();
  for (int i = 0; i < kN; ++i) EXPECT_EQ(got[static_cast<size_t>(i)], pat[static_cast<size_t>(i)]);
}

// G2: zero-byte Put skips the RDMA write but still delivers the signal.
__global__ void kernelPutZeroBytes(GdaHarness* h) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  sig.indexedSignal.signalId = 0;
  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(
      ginCtx, ncclCoopThread{}, 1, true, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, 0, sig, ncclGinSignalAdd, 5, false, 0,
      false, nullptr, cuda::thread_scope_system, cuda::thread_scope_system);
}

TEST_F(GinRocshmemGdaTemplateTest, Put_ZeroByteSkipsDataStillSignals) {
  constexpr int kN = 32;
  GdaEnv env(kN);
  std::vector<uint8_t> src(kN, 0x5A);
  env.src.copyFrom(src);
  env.dst.zero();
  env.build();
  resetPutNbiCount();
  kernelPutZeroBytes<<<1, 1>>>(env.dHarness.ptr);
  syncAndCheck();
  EXPECT_EQ(readPutNbiCount(), 0ULL);  // production skips put_nbi when bytes==0
  auto got = env.dst.copyTo();
  for (int i = 0; i < kN; ++i) EXPECT_EQ(got[static_cast<size_t>(i)], 0u);  // data write skipped
  auto sigs = env.signals.copyTo();
  EXPECT_EQ(sigs[0], 5ULL);  // signal still delivered
}

// G3: indexed SignalAdd delivers the given argument to the peer signal word.
__global__ void kernelPutSignalAdd(GdaHarness* h) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  sig.indexedSignal.signalId = 1;
  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(
      ginCtx, ncclCoopThread{}, 1, true, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, 32, sig, ncclGinSignalAdd, 7, false, 0,
      false, nullptr, cuda::thread_scope_system, cuda::thread_scope_system);
}

TEST_F(GinRocshmemGdaTemplateTest, Put_SignalAddDeliversArg) {
  GdaEnv env(32);
  env.src.zero();
  env.build();
  kernelPutSignalAdd<<<1, 1>>>(env.dHarness.ptr);
  syncAndCheck();
  auto sigs = env.signals.copyTo();
  EXPECT_EQ(sigs[1], 7ULL);
  EXPECT_EQ(sigs[0], 0ULL);
}

// G4: SignalInc normalizes any signalOpArg to +1.
__global__ void kernelPutSignalInc(GdaHarness* h) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  sig.indexedSignal.signalId = 0;
  // Pass a large arg to prove Inc normalizes to 1.
  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(
      ginCtx, ncclCoopThread{}, 1, true, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, 16, sig, ncclGinSignalInc, 99, false, 0,
      false, nullptr, cuda::thread_scope_system, cuda::thread_scope_system);
}

TEST_F(GinRocshmemGdaTemplateTest, Put_SignalIncNormalizesToOne) {
  GdaEnv env(16);
  env.src.zero();
  env.build();
  kernelPutSignalInc<<<1, 1>>>(env.dHarness.ptr);
  syncAndCheck();
  auto sigs = env.signals.copyTo();
  EXPECT_EQ(sigs[0], 1ULL);
}

// G5: Put with counter (no signal) quiets the QP then bumps the counter.
__global__ void kernelPutCounter(GdaHarness* h) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(
      ginCtx, ncclCoopThread{}, 1, true, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, 32, sig, ncclGinSignalInc, 0, true, 0,
      false, nullptr, cuda::thread_scope_system, cuda::thread_scope_system);
}

TEST_F(GinRocshmemGdaTemplateTest, Put_CounterQuietPath) {
  constexpr int kN = 32;
  std::vector<uint8_t> pat(kN, 0x3C);
  GdaEnv env(kN);
  env.src.copyFrom(pat);
  env.dst.zero();
  env.build();
  resetQuietCount();
  kernelPutCounter<<<1, 1>>>(env.dHarness.ptr);
  syncAndCheck();
  auto ctr = env.counters.copyTo();
  EXPECT_EQ(ctr[0], 1ULL);
  EXPECT_GE(readQuietCount(), 1ULL);  // counter path quiets the QP
  auto got = env.dst.copyTo();
  for (int i = 0; i < kN; ++i) EXPECT_EQ(got[static_cast<size_t>(i)], pat[static_cast<size_t>(i)]);
}

// G6: Put with both signal and counter delivers the signal and bumps counter.
__global__ void kernelPutSignalCounter(GdaHarness* h) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  sig.indexedSignal.signalId = 0;
  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(
      ginCtx, ncclCoopThread{}, 1, true, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, 32, sig, ncclGinSignalAdd, 3, true, 1,
      false, nullptr, cuda::thread_scope_system, cuda::thread_scope_system);
}

TEST_F(GinRocshmemGdaTemplateTest, Put_SignalAndCounter) {
  GdaEnv env(32);
  env.src.zero();
  env.build();
  kernelPutSignalCounter<<<1, 1>>>(env.dHarness.ptr);
  syncAndCheck();
  auto sigs = env.signals.copyTo();
  auto ctr = env.counters.copyTo();
  EXPECT_EQ(sigs[0], 3ULL);
  EXPECT_EQ(ctr[1], 1ULL);
}

// G7: required=system, given=block -> HIP guard fires (given < required) and put completes.
__global__ void kernelPutScopeFence(GdaHarness* h) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(
      ginCtx, ncclCoopThread{}, 1, true, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, 8, sig, ncclGinSignalInc, 0, false, 0,
      false, nullptr, cuda::thread_scope_system, cuda::thread_scope_block);
}

TEST_F(GinRocshmemGdaTemplateTest, Put_WeakerGivenScopeFencesAndPuts) {
  constexpr int kN = 8;
  std::vector<uint8_t> pat(kN);
  for (int i = 0; i < kN; ++i) pat[static_cast<size_t>(i)] = static_cast<uint8_t>(0x11 * (i + 1));
  GdaEnv env(kN);
  env.src.copyFrom(pat);
  env.dst.zero();
  env.build();
  resetThreadfenceCount();
  kernelPutScopeFence<<<1, 1>>>(env.dHarness.ptr);
  syncAndCheck();
  EXPECT_EQ(readThreadfenceCount(), 1ULL);
  auto got = env.dst.copyTo();
  for (int i = 0; i < kN; ++i) EXPECT_EQ(got[static_cast<size_t>(i)], pat[static_cast<size_t>(i)]);
}

// G7b: required=given=system -> guard does not fire (given < required is false) and put completes.
TEST_F(GinRocshmemGdaTemplateTest, Put_EqualScopeTakesNoFence) {
  constexpr int kN = 8;
  std::vector<uint8_t> pat(kN);
  for (int i = 0; i < kN; ++i) pat[static_cast<size_t>(i)] = static_cast<uint8_t>(0x22 + i);
  GdaEnv env(kN);
  env.src.copyFrom(pat);
  env.dst.zero();
  env.build();
  resetThreadfenceCount();
  kernelPutData<<<1, 1>>>(env.dHarness.ptr, kN);
  syncAndCheck();
  EXPECT_EQ(readThreadfenceCount(), 0ULL);
  auto got = env.dst.copyTo();
  for (int i = 0; i < kN; ++i) EXPECT_EQ(got[static_cast<size_t>(i)], pat[static_cast<size_t>(i)]);
}

// G8: PutValue writes an inline scalar to the peer buffer (lkey=0 inline WQE).
__global__ void kernelPutValueScalar(GdaHarness* h, uint64_t val) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(
      ginCtx, ncclCoopThread{}, 1, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0, val, sig,
      ncclGinSignalInc, 0, false, nullptr, cuda::thread_scope_system, cuda::thread_scope_system);
}

TEST_F(GinRocshmemGdaTemplateTest, PutValue_InlineScalar) {
  GdaEnv env(sizeof(uint64_t));
  env.dst.zero();
  env.build();
  const uint64_t kVal = 0xAABBCCDDEEFF0011ULL;
  kernelPutValueScalar<<<1, 1>>>(env.dHarness.ptr, kVal);
  syncAndCheck();
  auto got = env.dst.copyTo();
  uint64_t observed = 0;
  std::memcpy(&observed, got.data(), sizeof(observed));
  EXPECT_EQ(observed, kVal);
}

// G9: PutValue with signal delivers both the scalar and the signal.
__global__ void kernelPutValueSignal(GdaHarness* h, uint32_t val) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  sig.indexedSignal.signalId = 2;
  ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(
      ginCtx, ncclCoopThread{}, 1, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0, val, sig,
      ncclGinSignalAdd, 4, false, nullptr, cuda::thread_scope_system, cuda::thread_scope_system);
}

TEST_F(GinRocshmemGdaTemplateTest, PutValue_WithSignal) {
  GdaEnv env(sizeof(uint32_t));
  env.dst.zero();
  env.build();
  const uint32_t kVal = 0x12345678u;
  kernelPutValueSignal<<<1, 1>>>(env.dHarness.ptr, kVal);
  syncAndCheck();
  auto got = env.dst.copyTo();
  uint32_t observed = 0;
  std::memcpy(&observed, got.data(), sizeof(observed));
  EXPECT_EQ(observed, kVal);
  auto sigs = env.signals.copyTo();
  EXPECT_EQ(sigs[2], 4ULL);
}

// G10: Flush quiets every peer QP (one quiet per rank for a single-thread coop).
__global__ void kernelFlush(GdaHarness* h) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(ginCtx, ncclCoopThread{}, false, nullptr,
                                                           cuda::memory_order_seq_cst, nullptr);
}

TEST_F(GinRocshmemGdaTemplateTest, Flush_QuietsAllPeers) {
  GdaEnv env(1);
  env.build();
  resetQuietCount();
  kernelFlush<<<1, 1>>>(env.dHarness.ptr);
  syncAndCheck();
  EXPECT_EQ(readQuietCount(), static_cast<unsigned long long>(GdaEnv::kNRanks));
}

// G11: GetSignalPtr/ResetSignal and GetCounterPtr/ResetCounter round-trip.
__global__ void kernelGetReset(GdaHarness* h) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ncclGinOffsetPtr sigOff = ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(ginCtx, 0);
  if (sigOff.ptr) sigOff.ptr[0] = 55;
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  desc.indexedSignal.signalId = 0;
  ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(ginCtx, desc);

  ncclGinOffsetPtr ctrOff = ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(ginCtx, 0);
  if (ctrOff.ptr) ctrOff.ptr[0] = 77;
  ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(ginCtx, 0);
}

TEST_F(GinRocshmemGdaTemplateTest, GetReset_SignalAndCounter) {
  GdaEnv env(1);
  env.build();
  kernelGetReset<<<1, 1>>>(env.dHarness.ptr);
  syncAndCheck();
  auto sigs = env.signals.copyTo();
  auto ctr = env.counters.copyTo();
  EXPECT_EQ(sigs[0], 0ULL);  // set to 55 then reset
  EXPECT_EQ(ctr[0], 0ULL);   // set to 77 then reset
}

// G12: ResetSignal with a non-indexed descriptor is a no-op.
__global__ void kernelResetSignalNone(GdaHarness* h) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ncclGinOffsetPtr sigOff = ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(ginCtx, 0);
  if (sigOff.ptr) sigOff.ptr[0] = 42;
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(ginCtx, desc);
}

TEST_F(GinRocshmemGdaTemplateTest, ResetSignal_NoneIsNoOp) {
  GdaEnv env(1);
  env.build();
  kernelResetSignalNone<<<1, 1>>>(env.dHarness.ptr);
  syncAndCheck();
  auto sigs = env.signals.copyTo();
  EXPECT_EQ(sigs[0], 42ULL);  // untouched by non-indexed reset
}

}  // namespace RcclUnitTesting

#endif  // NCCL_GIN_ROCSHMEM_GDA_ENABLE
