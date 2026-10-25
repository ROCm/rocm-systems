/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// rocSHMEM-GDA GIN plugin host tests (stubbed QP factory, no NIC required).
// Drives ginRocshmemGdaCreateContext / DestroyContext through the plugin vtable
// and reads the uploaded GPU-context array back to check per-logical-context
// signal, counter, and remote-signal striping (AICOMRCCL-2339).

#include "gin_rocshmem_gda_plugin_test_stubs.h"

#include "../common/ScopedEnv.hpp"

#include "gin/gin_host_rocshmem_gda.h"
#include "comm.h"
#include "nccl_device/gin/rocshmem_gda/gin_rocshmem_device_host_common_gda.h"
#include "nccl_device/net_device.h"
#include "plugin/nccl_gin.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <string>
#include <vector>

namespace RcclUnitTesting
{

namespace Stubs = GinRocshmemGdaPluginStubs;

template <typename T>
static std::vector<T> CopyToHost(const T* dev, size_t count) {
  std::vector<T> host(count);
  EXPECT_EQ(hipMemcpy(host.data(), dev, count * sizeof(T), hipMemcpyDeviceToHost), hipSuccess);
  return host;
}

class GinRocshmemGdaPluginTest : public ::testing::Test {
 protected:
  ncclGin_t& plugin_ = ncclGinRocshmemGdaPlugin;
  ncclComm comm_{};
  char bootstrapPlaceholder_{0};
  void* ictx_ = nullptr;
  void* coll_ = nullptr;

  void SetUp() override {
    Stubs::Reset();
    comm_.bootstrap = &bootstrapPlaceholder_;
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev < 1) GTEST_SKIP() << "GPU required by GinRocshmemGdaPlugin fixture";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    setenv("NCCL_GIN_TYPE", std::to_string(static_cast<int>(NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA)).c_str(), 1);
  }

  void TearDown() override {
    if (coll_) plugin_.closeColl(coll_);
    if (ictx_) plugin_.finalize(ictx_);
    unsetenv("NCCL_GIN_TYPE");
    Stubs::Reset();
  }

  void connect(int nranks, int rank) {
    Stubs::SetBootstrapLayout(nranks, rank);
    ASSERT_EQ(plugin_.init(&ictx_, 0, nullptr), ncclSuccess);
    ncclGinRocshmemSetInitContext(ictx_, &comm_);
    void* listen = nullptr;
    char handle[NCCL_NET_HANDLE_MAXSIZE] = {};
    ASSERT_EQ(plugin_.listen(ictx_, 0, handle, &listen), ncclSuccess);
    std::vector<void*> handles(static_cast<size_t>(nranks), handle);
    ASSERT_EQ(plugin_.connect(ictx_, handles.data(), nranks, rank, listen, &coll_), ncclSuccess);
    ASSERT_EQ(plugin_.closeListen(listen), ncclSuccess);
  }

  static std::vector<ncclGinRocshmemGdaGPUContext> readGpuContexts(const ncclNetDeviceHandle_v11_t* devHandle) {
    const size_t n = devHandle->size / sizeof(ncclGinRocshmemGdaGPUContext);
    return CopyToHost(static_cast<const ncclGinRocshmemGdaGPUContext*>(devHandle->handle), n);
  }
};

// AICOMRCCL-3418: GDA stays off unless NCCL_GIN_TYPE=6, or NCCL_GIN_TYPE is unset
// and NCCL_GIN_TYPES lists 6.
TEST_F(GinRocshmemGdaPluginTest, Init_OffByDefault) {
  ScopedEnv ty("NCCL_GIN_TYPE", nullptr);
  ScopedEnv list("NCCL_GIN_TYPES", nullptr);
  EXPECT_EQ(plugin_.init(&ictx_, 0, nullptr), ncclInternalError);
  EXPECT_EQ(ictx_, nullptr);
}

TEST_F(GinRocshmemGdaPluginTest, Init_ListedInGinTypes) {
  ScopedEnv ty("NCCL_GIN_TYPE", nullptr);
  ScopedEnv list("NCCL_GIN_TYPES", "6,7");
  ASSERT_EQ(plugin_.init(&ictx_, 0, nullptr), ncclSuccess);
  EXPECT_NE(ictx_, nullptr);
}

TEST_F(GinRocshmemGdaPluginTest, Init_NotListedInGinTypes) {
  ScopedEnv ty("NCCL_GIN_TYPE", nullptr);
  ScopedEnv list("NCCL_GIN_TYPES", "2,7");
  EXPECT_EQ(plugin_.init(&ictx_, 0, nullptr), ncclInternalError);
  EXPECT_EQ(ictx_, nullptr);
}

TEST_F(GinRocshmemGdaPluginTest, Init_GinTypeOverridesGinTypes) {
  ScopedEnv list("NCCL_GIN_TYPES", "2,7");
  ASSERT_EQ(plugin_.init(&ictx_, 0, nullptr), ncclSuccess);
  EXPECT_NE(ictx_, nullptr);

  ScopedEnv ty("NCCL_GIN_TYPE", "7");
  ScopedEnv listWithGda("NCCL_GIN_TYPES", "6,7");
  void* other = nullptr;
  EXPECT_EQ(plugin_.init(&other, 0, nullptr), ncclInternalError);
  EXPECT_EQ(other, nullptr);
}

TEST_F(GinRocshmemGdaPluginTest, CreateContext_StripesSignalsAndCountersPerContext) {
  constexpr int kRanks = 2, kRank = 1, kContexts = 3, kSignals = 4, kCounters = 2;
  connect(kRanks, kRank);

  ncclGinConfig_t cfg{};
  cfg.nContexts = kContexts;
  cfg.nSignals = kSignals;
  cfg.nCounters = kCounters;
  void* ginCtx = nullptr;
  ncclNetDeviceHandle_v11_t* devHandle = nullptr;
  ASSERT_EQ(plugin_.createContext(coll_, &cfg, &ginCtx, &devHandle), ncclSuccess);
  ASSERT_NE(devHandle, nullptr);
  EXPECT_EQ(devHandle->netDeviceType, NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA);
  EXPECT_EQ(devHandle->netDeviceVersion, NCCL_GIN_ROCSHMEM_VERSION);
  ASSERT_EQ(devHandle->size, kContexts * sizeof(ncclGinRocshmemGdaGPUContext));

  // A single atomic MR covers every context's signal stripe.
  ASSERT_EQ(Stubs::GetRegMrCalls().size(), 1u);
  const Stubs::RegMrCall sigMr = Stubs::GetRegMrCalls()[0];
  EXPECT_EQ(sigMr.size, sizeof(uint64_t) * kSignals * kContexts);
  EXPECT_EQ(sigMr.atomic, 1);

  const auto ctxs = readGpuContexts(devHandle);
  ASSERT_EQ(ctxs.size(), static_cast<size_t>(kContexts));
  uint64_t* const signalsBase = ctxs[0].signals;
  uint64_t* const countersBase = ctxs[0].counters;
  EXPECT_EQ(static_cast<void*>(signalsBase), sigMr.addr);
  ASSERT_NE(countersBase, nullptr);
  ASSERT_NE(ctxs[0].signal_rkeys, nullptr);

  const size_t stripeBytes = sizeof(uint64_t) * kSignals;
  for (int c = 0; c < kContexts; ++c) {
    SCOPED_TRACE("contextId=" + std::to_string(c));
    const ncclGinRocshmemGdaGPUContext& ctx = ctxs[static_cast<size_t>(c)];
    EXPECT_EQ(ctx.nRanks, kRanks);
    EXPECT_EQ(ctx.rank, kRank);
    EXPECT_EQ(ctx.nSignals, static_cast<uint32_t>(kSignals));
    EXPECT_EQ(ctx.nCounters, static_cast<uint32_t>(kCounters));
    EXPECT_EQ(static_cast<void*>(ctx.qps), static_cast<void*>(Stubs::FakeGpuQps()));
    EXPECT_EQ(ctx.signals, signalsBase + c * kSignals);
    EXPECT_EQ(ctx.counters, countersBase + c * kCounters);
    EXPECT_EQ(ctx.signal_rkeys, ctxs[0].signal_rkeys);

    ASSERT_NE(ctx.signal_raddrs, nullptr);
    if (c > 0) EXPECT_NE(ctx.signal_raddrs, ctxs[static_cast<size_t>(c - 1)].signal_raddrs);
    const auto raddrs = CopyToHost(ctx.signal_raddrs, kRanks);
    EXPECT_EQ(raddrs[0], Stubs::PeerAddr(0) + c * stripeBytes);
    EXPECT_EQ(raddrs[kRank], reinterpret_cast<uintptr_t>(signalsBase) + c * stripeBytes);
  }

  const auto rkeys = CopyToHost(ctxs[0].signal_rkeys, kRanks);
  EXPECT_EQ(rkeys[0], Stubs::PeerRkey(0));
  EXPECT_EQ(rkeys[kRank], sigMr.rkey);

  for (uint64_t v : CopyToHost(signalsBase, kSignals * kContexts)) EXPECT_EQ(v, 0u);
  for (uint64_t v : CopyToHost(countersBase, kCounters * kContexts)) EXPECT_EQ(v, 0u);

  EXPECT_EQ(plugin_.destroyContext(ginCtx), ncclSuccess);
}

TEST_F(GinRocshmemGdaPluginTest, DestroyContext_ReleasesEveryContextAllocation) {
  constexpr int kContexts = 3;
  connect(2, 0);
  const int hipAllocsBefore = Stubs::GetLiveHipAllocationCount();

  ncclGinConfig_t cfg{};
  cfg.nContexts = kContexts;
  cfg.nSignals = 4;
  cfg.nCounters = 2;
  void* ginCtx = nullptr;
  ncclNetDeviceHandle_v11_t* devHandle = nullptr;
  ASSERT_EQ(plugin_.createContext(coll_, &cfg, &ginCtx, &devHandle), ncclSuccess);
  const auto ctxs = readGpuContexts(devHandle);
  ASSERT_EQ(ctxs.size(), static_cast<size_t>(kContexts));

  // GPU-context array, signal span, counter span, shared rkeys, one raddrs array per context.
  std::vector<const void*> allocations = {devHandle->handle, ctxs[0].signals, ctxs[0].counters,
                                          ctxs[0].signal_rkeys};
  for (const auto& ctx : ctxs) allocations.push_back(ctx.signal_raddrs);
  for (const void* p : allocations) EXPECT_TRUE(Stubs::IsLiveHipAllocation(p)) << p;
  EXPECT_EQ(Stubs::GetLiveHipAllocationCount() - hipAllocsBefore, static_cast<int>(allocations.size()));
  EXPECT_EQ(Stubs::GetLiveMrCount(), 1);

  EXPECT_EQ(plugin_.destroyContext(ginCtx), ncclSuccess);
  EXPECT_EQ(Stubs::GetLiveMrCount(), 0);
  for (const void* p : allocations) EXPECT_FALSE(Stubs::IsLiveHipAllocation(p)) << p;
  EXPECT_EQ(Stubs::GetLiveHipAllocationCount(), hipAllocsBefore);
  // The QP set is borrowed from collComm and must outlive the context.
  EXPECT_EQ(Stubs::GetDestroyQpsCalls(), 0);
}

TEST_F(GinRocshmemGdaPluginTest, CreateContext_DefaultsToOneContext) {
  connect(2, 0);

  ncclGinConfig_t cfg{};
  cfg.nSignals = 2;
  cfg.nCounters = 1;
  void* ginCtx = nullptr;
  ncclNetDeviceHandle_v11_t* devHandle = nullptr;
  ASSERT_EQ(plugin_.createContext(coll_, &cfg, &ginCtx, &devHandle), ncclSuccess);
  EXPECT_EQ(devHandle->size, sizeof(ncclGinRocshmemGdaGPUContext));
  ASSERT_EQ(Stubs::GetRegMrCalls().size(), 1u);
  EXPECT_EQ(Stubs::GetRegMrCalls()[0].size, sizeof(uint64_t) * 2);

  const auto ctxs = readGpuContexts(devHandle);
  ASSERT_EQ(ctxs.size(), 1u);
  const auto raddrs = CopyToHost(ctxs[0].signal_raddrs, 2);
  EXPECT_EQ(raddrs[0], reinterpret_cast<uintptr_t>(ctxs[0].signals));
  EXPECT_EQ(raddrs[1], Stubs::PeerAddr(1));

  EXPECT_EQ(plugin_.destroyContext(ginCtx), ncclSuccess);
}

TEST_F(GinRocshmemGdaPluginTest, CreateContext_CountersOnlyLeavesSignalFieldsNull) {
  constexpr int kContexts = 2, kCounters = 3;
  connect(2, 0);

  ncclGinConfig_t cfg{};
  cfg.nContexts = kContexts;
  cfg.nCounters = kCounters;
  void* ginCtx = nullptr;
  ncclNetDeviceHandle_v11_t* devHandle = nullptr;
  ASSERT_EQ(plugin_.createContext(coll_, &cfg, &ginCtx, &devHandle), ncclSuccess);
  EXPECT_TRUE(Stubs::GetRegMrCalls().empty());

  const auto ctxs = readGpuContexts(devHandle);
  ASSERT_EQ(ctxs.size(), static_cast<size_t>(kContexts));
  ASSERT_NE(ctxs[0].counters, nullptr);
  for (int c = 0; c < kContexts; ++c) {
    SCOPED_TRACE("contextId=" + std::to_string(c));
    const ncclGinRocshmemGdaGPUContext& ctx = ctxs[static_cast<size_t>(c)];
    EXPECT_EQ(ctx.signals, nullptr);
    EXPECT_EQ(ctx.signal_rkeys, nullptr);
    EXPECT_EQ(ctx.signal_raddrs, nullptr);
    EXPECT_EQ(ctx.counters, ctxs[0].counters + c * kCounters);
  }

  EXPECT_EQ(plugin_.destroyContext(ginCtx), ncclSuccess);
}

TEST_F(GinRocshmemGdaPluginTest, CreateContext_SignalMrFailureReturnsErrorWithoutLeaking) {
  connect(2, 0);
  const int hipAllocsBefore = Stubs::GetLiveHipAllocationCount();
  Stubs::SetRegMrFail(true);

  ncclGinConfig_t cfg{};
  cfg.nContexts = 2;
  cfg.nSignals = 4;
  cfg.nCounters = 2;
  void* ginCtx = nullptr;
  ncclNetDeviceHandle_v11_t* devHandle = nullptr;
  EXPECT_EQ(plugin_.createContext(coll_, &cfg, &ginCtx, &devHandle), ncclSystemError);
  EXPECT_EQ(ginCtx, nullptr);
  EXPECT_EQ(devHandle, nullptr);
  EXPECT_EQ(Stubs::GetLiveMrCount(), 0);
  // The GPU-context array and signal span were allocated before the MR failed.
  EXPECT_EQ(Stubs::GetLiveHipAllocationCount(), hipAllocsBefore);

  Stubs::SetRegMrFail(false);
  ASSERT_EQ(plugin_.createContext(coll_, &cfg, &ginCtx, &devHandle), ncclSuccess);
  EXPECT_EQ(Stubs::GetLiveMrCount(), 1);
  EXPECT_EQ(plugin_.destroyContext(ginCtx), ncclSuccess);
  EXPECT_EQ(Stubs::GetLiveMrCount(), 0);
  EXPECT_EQ(Stubs::GetLiveHipAllocationCount(), hipAllocsBefore);
}

}  // namespace RcclUnitTesting
