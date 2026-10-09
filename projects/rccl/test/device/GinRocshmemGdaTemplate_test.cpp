/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Suite G: GIN rocSHMEM-GDA device template coverage (Put/PutValue/Flush/
// FlushAsync/Wait/Signal/Counter, timeouts + edge paths), the GDA analog of Suite H
// (GinAnvilSdmaTemplate_test.cpp). AllToAll drives these ncclGinApi_* template
// specializations, so this suite unit-tests the GDA AllToAll device path
// without a live network.
//
// rocshmem::QueuePair is declared as a type alias for some QueuePair type by
// the rocSHMEM-GDA header gda/queue_pair_provider.hpp. rocshmem::QueuePairMock
// provides a mock QueuePair type that conforms to rocshmem::QueuePairInterface.
// By defining GDA_QUEUEPAIR_MOCK prior to including queue_pair_provider.hpp,
// rocshmem::QueuePair is aliased to rocshmem::QueuePairMock. The definition of
// the rocshmem::QueuePairMock mock QueuePair object type mirrors how Suite H
// shadows the SDMA engine with test/device/sdma/anvil_device.hpp.
// projects/rccl/test/CMakeLists.txt defines GDA_QUEUEPAIR_MOCK for targets
// rccl-UnitTestsFixtures and rccl-UnitTestsGinAnvilPlugin

#include "DeviceTestBase.hpp"

#include "nccl_device/coop.h"
#include "nccl_device/gin/gin_device_host_common.h"
#include "nccl_device/gin/gin_device_common.h"
#include "nccl_device/gin/rocshmem_gda/gin_rocshmem_device_host_common_gda.h"
#include "nccl_device/gin/rocshmem_gda/gda/queue_pair_provider.hpp"

#if NCCL_GIN_ROCSHMEM_GDA_ENABLE
// Count invocations of the Put/PutValue system-scope fence seam (gin_device_common.h).
// Override must precede gin_rocshmem_gda.h so the templates expand our counter.
__device__ unsigned long long g_gdaStubThreadfenceCount = 0;
// Inline puts already posted when the last fence ran; nonzero means the fence came after the put.
__device__ unsigned long long g_gdaInlinePutsAtFence = 0;
#undef NCCL_GIN_THREADFENCE_SYSTEM
#define NCCL_GIN_THREADFENCE_SYSTEM() \
  (g_gdaInlinePutsAtFence = rocshmem::QueuePairMock::rma_inline_count, atomicAdd(&g_gdaStubThreadfenceCount, 1ULL))
#include "nccl_device/gin/rocshmem_gda/gin_rocshmem_gda.h"
#endif

#include <cstdint>
#include <cstring>
#include <vector>

#if NCCL_GIN_ROCSHMEM_GDA_ENABLE

namespace RcclUnitTesting
{

class GinRocshmemGdaTemplateTest : public DeviceTestBase {
protected:
  // A test that leaves the mock queue busy would stall the tests after it.
  void TearDown() override {
    size_t zero = 0;
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(rocshmem::QueuePairMock::try_quiet_busy), &zero, sizeof(zero)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(rocshmem::QueuePairMock::pending_wqes), &zero, sizeof(zero)));
  }
};

struct GdaHarness {
  ncclGinRocshmemGdaGPUContext ctx;
  ncclGinRocshmemGdaMemHandle dstMh;
  ncclGinRocshmemGdaMemHandle srcMh;
};

// Bundles all device-side arrays a GDA context/mem-handle points at, and wires
// them into a single uploaded GdaHarness. Peer 1 is the (self-mapped) target:
// its remote_vas entry points back at the local dst buffer and its signal
// remote-address points back at the local signals array, so a self put/signal
// is observable from the host. Each peer gets its own mock QP.
class GdaEnv {
public:
  static constexpr int kNRanks = 2;
  static constexpr int kPeer = 1;
  static constexpr uint32_t kNSignals = 4;
  static constexpr uint32_t kNCounters = 2;

  DeviceBuffer<rocshmem::QueuePair> qp{kNRanks};
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
    qp.zero();
    std::vector<rocshmem::QueuePair*> qpRow(kNRanks);
    for (int i = 0; i < kNRanks; ++i) qpRow[i] = qp.ptr + i;
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
  size_t z = 0;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(rocshmem::QueuePairMock::quiet_count), &z, sizeof(z)));
}

static size_t readQuietCount() {
  size_t q = 0;
  HIP_EXPECT(hipMemcpyFromSymbol(&q, HIP_SYMBOL(rocshmem::QueuePairMock::quiet_count), sizeof(q)));
  return q;
}

static void resetPutNbiCount() {
  size_t z = 0;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(rocshmem::QueuePairMock::rma_count), &z, sizeof(z)));
}

static size_t readPutNbiCount() {
  size_t n = 0;
  HIP_EXPECT(hipMemcpyFromSymbol(&n, HIP_SYMBOL(rocshmem::QueuePairMock::rma_count), sizeof(n)));
  return n;
}

static void resetPutValCount() {
  size_t z = 0;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(rocshmem::QueuePairMock::rma_inline_count), &z, sizeof(z)));
}

static size_t readPutValCount() {
  size_t v = 0;
  HIP_EXPECT(hipMemcpyFromSymbol(&v, HIP_SYMBOL(rocshmem::QueuePairMock::rma_inline_count), sizeof(v)));
  return v;
}

static void resetSignalCount() {
  size_t z = 0;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(rocshmem::QueuePairMock::amo_count), &z, sizeof(z)));
}

static size_t readSignalCount() {
  size_t s = 0;
  HIP_EXPECT(hipMemcpyFromSymbol(&s, HIP_SYMBOL(rocshmem::QueuePairMock::amo_count), sizeof(s)));
  return s;
}

static void resetThreadfenceCount() {
  size_t z = 0;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_gdaStubThreadfenceCount), &z, sizeof(z)));
}

static size_t readThreadfenceCount() {
  size_t c = 0;
  HIP_EXPECT(hipMemcpyFromSymbol(&c, HIP_SYMBOL(g_gdaStubThreadfenceCount), sizeof(c)));
  return c;
}

static void resetInlinePutsAtFence() {
  unsigned long long z = 0;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_gdaInlinePutsAtFence), &z, sizeof(z)));
}

static unsigned long long readInlinePutsAtFence() {
  unsigned long long n = 0;
  HIP_EXPECT(hipMemcpyFromSymbol(&n, HIP_SYMBOL(g_gdaInlinePutsAtFence), sizeof(n)));
  return n;
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
  resetSignalCount();
  kernelPutZeroBytes<<<1, 1>>>(env.dHarness.ptr);
  syncAndCheck();
  EXPECT_EQ(readPutNbiCount(), 0ULL);  // production skips put_nbi when bytes==0
  EXPECT_EQ(readSignalCount(), 1ULL);  // signal still dispatched
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
  resetSignalCount();
  kernelPutSignalAdd<<<1, 1>>>(env.dHarness.ptr);
  syncAndCheck();
  EXPECT_EQ(readSignalCount(), 1ULL);  // signal dispatched
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

// G8: PutValue inlines the scalar (lkey=0 WQE) and fences first only when required=system and given < required.
__global__ void kernelPutValueScoped(GdaHarness* h, uint64_t val, cuda::thread_scope required,
                                     cuda::thread_scope given) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(
      ginCtx, ncclCoopThread{}, 1, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0, val, sig,
      ncclGinSignalInc, 0, false, nullptr, required, given);
}

TEST_F(GinRocshmemGdaTemplateTest, PutValue_FencesOnlyForWeakerGivenAtSystemScope) {
  struct ScopeCase {
    cuda::thread_scope required;
    cuda::thread_scope given;
    size_t fences;
  };
  const ScopeCase cases[] = {
      {cuda::thread_scope_system, cuda::thread_scope_thread, 1},
      {cuda::thread_scope_system, cuda::thread_scope_block, 1},
      {cuda::thread_scope_system, cuda::thread_scope_device, 1},
      {cuda::thread_scope_system, cuda::thread_scope_system, 0},
      {cuda::thread_scope_device, cuda::thread_scope_block, 0},  // weaker given, but required is not system
  };
  uint64_t val = 0xC0FFEE0000000000ULL;
  for (const ScopeCase& c : cases) {
    SCOPED_TRACE(::testing::Message() << "required=" << c.required << " given=" << c.given);
    ++val;
    GdaEnv env(sizeof(uint64_t));
    env.dst.zero();
    env.build();
    resetThreadfenceCount();
    resetInlinePutsAtFence();
    resetPutValCount();
    resetSignalCount();
    kernelPutValueScoped<<<1, 1>>>(env.dHarness.ptr, val, c.required, c.given);
    syncAndCheck();
    EXPECT_EQ(readThreadfenceCount(), c.fences);
    if (c.fences != 0) {
      EXPECT_EQ(readInlinePutsAtFence(), 0ULL) << "the fence must precede the inline put";
    }
    EXPECT_EQ(readPutValCount(), 1ULL);
    EXPECT_EQ(readSignalCount(), 0ULL) << "no signal descriptor, so no AMO";
    auto got = env.dst.copyTo();
    uint64_t observed = 0;
    std::memcpy(&observed, got.data(), sizeof(observed));
    EXPECT_EQ(observed, val);
  }
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
  resetPutValCount();
  resetSignalCount();
  const uint32_t kVal = 0x12345678u;
  kernelPutValueSignal<<<1, 1>>>(env.dHarness.ptr, kVal);
  syncAndCheck();
  EXPECT_EQ(readPutValCount(), 1ULL);  // put inlined
  EXPECT_EQ(readSignalCount(), 1ULL);  // signal dispatched
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
  EXPECT_EQ(readQuietCount(), static_cast<size_t>(GdaEnv::kNRanks));
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

using nccl::gin::rocshmem_gda::ncclGinRocshmemGdaRequest;

constexpr uint64_t kShortBudget = 1000;
constexpr uint64_t kLongBudget = 1ULL << 34;
constexpr size_t kNeverDrains = ~size_t{0};

static void setTryQuietBusy(size_t polls) {
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(rocshmem::QueuePairMock::try_quiet_busy), &polls, sizeof(polls)));
}

static void setPendingWqes(size_t wqes) {
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(rocshmem::QueuePairMock::pending_wqes), &wqes, sizeof(wqes)));
}

static size_t readPolls(const GdaEnv& env, int peer) {
  size_t polls = 0;
  HIP_EXPECT(hipMemcpy(&polls, &env.qp.ptr[peer].polls, sizeof(polls), hipMemcpyDeviceToHost));
  return polls;
}

static ncclGinRequest_t makeRequest(int peer) {
  ncclGinRocshmemGdaRequest req{/*target=*/0, peer};
  ncclGinRequest_t raw{};
  std::memcpy(&raw, &req, sizeof(req));
  return raw;
}

static ncclGinRocshmemGdaRequest readRequest(DeviceBuffer<ncclGinRequest_t>& d_req) {
  ncclGinRequest_t raw = d_req.download();
  ncclGinRocshmemGdaRequest req;
  std::memcpy(&req, &raw, sizeof(req));
  return req;
}

// G13: FlushAsync records the peer and its queue target and posts no quiet.
__global__ void kernelFlushAsync(GdaHarness* h, ncclGinRequest_t* req, int peer) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinApi_FlushAsync<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(ginCtx, peer, req, false, nullptr, 0);
}

TEST_F(GinRocshmemGdaTemplateTest, FlushAsync_RecordsQueueTargetWithoutQuiet) {
  GdaEnv env(1);
  env.build();
  DeviceBuffer<ncclGinRequest_t> d_req(1);
  d_req.zero();
  resetPutNbiCount();
  resetSignalCount();
  kernelPutData<<<1, 1>>>(env.dHarness.ptr, 1);
  syncAndCheck();
  resetQuietCount();
  kernelFlushAsync<<<1, 1>>>(env.dHarness.ptr, d_req.ptr, GdaEnv::kPeer);
  syncAndCheck();
  ncclGinRocshmemGdaRequest req = readRequest(d_req);
  EXPECT_EQ(req.peer, GdaEnv::kPeer);
  EXPECT_EQ(req.target, 1ULL);
  EXPECT_EQ(readQuietCount(), 0ULL);
}

// G14: Wait polls only the QP of the recorded peer.
__global__ void kernelWait(GdaHarness* h, ncclGinRequest_t* req) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(ginCtx, *req, false, nullptr, cuda::memory_order_acq_rel,
                                                          nullptr);
}

TEST_F(GinRocshmemGdaTemplateTest, Wait_PollsRecordedPeer) {
  GdaEnv env(1);
  env.build();
  DeviceBuffer<ncclGinRequest_t> d_req(1);
  d_req.upload(makeRequest(GdaEnv::kPeer));
  kernelWait<<<1, 1>>>(env.dHarness.ptr, d_req.ptr);
  syncAndCheck();
  EXPECT_EQ(readPolls(env, GdaEnv::kPeer), 1u);
  EXPECT_EQ(readPolls(env, 0), 0u);
}

// G15-G19: timed Flush and Wait poll the queue until it reaches the target, the budget runs out, or abort is set.
__global__ void kernelFlushTimeout(GdaHarness* h, uint32_t* abortFlag, uint64_t timeoutCycles, ncclResult_t* result) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  *result = ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(
      ginCtx, ncclCoopThread{}, false, nullptr, cuda::memory_order_seq_cst, abortFlag, timeoutCycles);
}

__global__ void kernelWaitTimeout(GdaHarness* h, ncclGinRequest_t* req, uint32_t* abortFlag, uint64_t timeoutCycles,
                                  ncclResult_t* result) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  *result = ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(
      ginCtx, *req, false, nullptr, cuda::memory_order_acq_rel, abortFlag, timeoutCycles);
}

TEST_F(GinRocshmemGdaTemplateTest, FlushTimeout_DrainedQueueNeedsNoBudget) {
  GdaEnv env(1);
  env.build();
  DeviceBuffer<ncclResult_t> d_result(1);
  d_result.upload(ncclInternalError);
  kernelFlushTimeout<<<1, 1>>>(env.dHarness.ptr, nullptr, /*timeoutCycles=*/0, d_result.ptr);
  syncAndCheck();
  EXPECT_EQ(d_result.download(), ncclSuccess);
  EXPECT_EQ(readPolls(env, 0), 1u);
  EXPECT_EQ(readPolls(env, GdaEnv::kPeer), 1u);
}

TEST_F(GinRocshmemGdaTemplateTest, FlushTimeout_BusyQueueTimesOut) {
  GdaEnv env(1);
  env.build();
  DeviceBuffer<ncclResult_t> d_result(1);
  d_result.upload(ncclInternalError);
  setTryQuietBusy(kNeverDrains);
  kernelFlushTimeout<<<1, 1>>>(env.dHarness.ptr, nullptr, kShortBudget, d_result.ptr);
  syncAndCheck();
  EXPECT_EQ(d_result.download(), ncclTimeout);
}

TEST_F(GinRocshmemGdaTemplateTest, WaitTimeout_PollsUntilDrained) {
  GdaEnv env(1);
  env.build();
  DeviceBuffer<ncclGinRequest_t> d_req(1);
  d_req.upload(makeRequest(GdaEnv::kPeer));
  DeviceBuffer<ncclResult_t> d_result(1);
  d_result.upload(ncclInternalError);
  setTryQuietBusy(3);
  resetQuietCount();
  kernelWaitTimeout<<<1, 1>>>(env.dHarness.ptr, d_req.ptr, nullptr, kLongBudget, d_result.ptr);
  syncAndCheck();
  EXPECT_EQ(d_result.download(), ncclSuccess);
  EXPECT_EQ(readQuietCount(), 4ULL);
}

TEST_F(GinRocshmemGdaTemplateTest, WaitTimeout_BusyQueueTimesOut) {
  GdaEnv env(1);
  env.build();
  DeviceBuffer<ncclGinRequest_t> d_req(1);
  d_req.upload(makeRequest(GdaEnv::kPeer));
  DeviceBuffer<ncclResult_t> d_result(1);
  d_result.upload(ncclInternalError);
  setTryQuietBusy(kNeverDrains);
  kernelWaitTimeout<<<1, 1>>>(env.dHarness.ptr, d_req.ptr, nullptr, kShortBudget, d_result.ptr);
  syncAndCheck();
  EXPECT_EQ(d_result.download(), ncclTimeout);
}

TEST_F(GinRocshmemGdaTemplateTest, WaitTimeout_AbortReturnsSuccess) {
  GdaEnv env(1);
  env.build();
  DeviceBuffer<ncclGinRequest_t> d_req(1);
  d_req.upload(makeRequest(GdaEnv::kPeer));
  DeviceBuffer<ncclResult_t> d_result(1);
  d_result.upload(ncclInternalError);
  DeviceBuffer<uint32_t> d_abort(1);
  uint32_t aborted = 1;
  d_abort.copyFrom(&aborted, 1);
  setTryQuietBusy(kNeverDrains);
  kernelWaitTimeout<<<1, 1>>>(env.dHarness.ptr, d_req.ptr, d_abort.ptr, kLongBudget, d_result.ptr);
  syncAndCheck();
  EXPECT_EQ(d_result.download(), ncclSuccess);
}

// G20: Wait covers the WQEs posted before FlushAsync, not the ones posted after it.
__device__ void putOneByte(ncclGinCtx ginCtx, GdaHarness* h) {
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(
      ginCtx, ncclCoopThread{}, GdaEnv::kPeer, true, reinterpret_cast<ncclGinWindow_t>(&h->dstMh), 0,
      reinterpret_cast<ncclGinWindow_t>(&h->srcMh), 0, 1, sig, ncclGinSignalInc, 0, false, 0, false, nullptr,
      cuda::thread_scope_system, cuda::thread_scope_system);
}

__global__ void kernelPutFlushAsyncPutWait(GdaHarness* h, uint64_t timeoutCycles, ncclResult_t* result) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinRequest_t req;
  putOneByte(ginCtx, h);
  ncclGinApi_FlushAsync<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(ginCtx, GdaEnv::kPeer, &req, false, nullptr, 0);
  putOneByte(ginCtx, h);
  *result = ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(
      ginCtx, req, false, nullptr, cuda::memory_order_acq_rel, nullptr, timeoutCycles);
}

TEST_F(GinRocshmemGdaTemplateTest, WaitTimeout_CoversOnlyWqesBeforeFlushAsync) {
  GdaEnv env(1);
  env.build();
  DeviceBuffer<ncclResult_t> d_result(1);

  d_result.upload(ncclInternalError);
  setPendingWqes(1);
  kernelPutFlushAsyncPutWait<<<1, 1>>>(env.dHarness.ptr, kShortBudget, d_result.ptr);
  syncAndCheck();
  EXPECT_EQ(d_result.download(), ncclSuccess);

  d_result.upload(ncclInternalError);
  setPendingWqes(2);
  kernelPutFlushAsyncPutWait<<<1, 1>>>(env.dHarness.ptr, kShortBudget, d_result.ptr);
  syncAndCheck();
  EXPECT_EQ(d_result.download(), ncclTimeout);
}

// G21: blocking Flush and Wait return once abort is set, even if the queue never drains.
__global__ void kernelBlockingFlushWait(GdaHarness* h, ncclGinRequest_t* req, uint32_t* abortFlag) {
  ncclGinCtx ginCtx{};
  ginCtx.handle = &h->ctx;
  ginCtx.nRanks = 2;
  ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(ginCtx, ncclCoopThread{}, false, nullptr,
                                                           cuda::memory_order_seq_cst, abortFlag);
  ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA>::call(ginCtx, *req, false, nullptr, cuda::memory_order_acq_rel,
                                                          abortFlag);
}

TEST_F(GinRocshmemGdaTemplateTest, Blocking_ReturnsOnAbort) {
  GdaEnv env(1);
  env.build();
  DeviceBuffer<ncclGinRequest_t> d_req(1);
  d_req.upload(makeRequest(GdaEnv::kPeer));
  DeviceBuffer<uint32_t> d_abort(1);
  uint32_t aborted = 1;
  d_abort.copyFrom(&aborted, 1);
  setTryQuietBusy(kNeverDrains);
  kernelBlockingFlushWait<<<1, 1>>>(env.dHarness.ptr, d_req.ptr, d_abort.ptr);
  syncAndCheck();
  EXPECT_GT(readPolls(env, 0), 0u);
  EXPECT_GT(readPolls(env, GdaEnv::kPeer), readPolls(env, 0));
}

}  // namespace RcclUnitTesting

#endif  // NCCL_GIN_ROCSHMEM_GDA_ENABLE
