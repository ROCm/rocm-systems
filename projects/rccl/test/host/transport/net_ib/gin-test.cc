/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib/gin.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

#include "HostAllocLedger.h"
#include "IbVerbsScene.h"
#include "ScopedHook.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/socket_fakes.h"
#include "gin/gin_host.h"
#include "transport/net_ib/NetIbGlobals.h"
#include "transport/net_ib/NetIbSupportSeams.h"
#include "transport/net_ib/common.h"
#include "transport/net_ib/gin.h"

namespace {

int g_ginSelfCopies = 0;

void* TestGinMemcpy(void* dst, const void* src, size_t size) {
  if (dst == src) {
    ++g_ginSelfCopies;
    return dst;
  }
  return std::memcpy(dst, src, size);
}

constexpr auto kGinHookFails = [](auto&&...) { return ncclSystemError; };
constexpr auto kGinHookNoOp = [](auto&&...) {};

std::function<ncclResult_t(ncclDebugLogger_t, ncclProfilerCallback_t)> g_ginInitDevices = kGinHookFails;
std::function<ncclResult_t()> g_ginFinalizeDevices = kGinHookFails;
std::function<ncclResult_t(void*, int, void*, void**, ncclNetDeviceHandle_t**, int, int)> g_ginConnect =
    kGinHookFails;
std::function<ncclResult_t(void*, void**, ncclNetDeviceHandle_t**, int)> g_ginAccept = kGinHookFails;
std::function<void(void*, int)> g_ginSetTrafficClass = kGinHookNoOp;
std::function<ncclResult_t(ncclIbRecvComm*)> g_ginCreateFlushQp = kGinHookFails;
std::function<ncclResult_t(void*, void*, size_t, int, uint64_t, int, uint64_t, void**)> g_ginRegMrDmaBuf =
    kGinHookFails;

ncclResult_t TestGinInitDevices(ncclDebugLogger_t logger, ncclProfilerCallback_t profiler) {
  return g_ginInitDevices(logger, profiler);
}
ncclResult_t TestGinFinalizeDevices() { return g_ginFinalizeDevices(); }
ncclResult_t TestGinConnect(void* ctx, int dev, void* handle, void** comm, ncclNetDeviceHandle_t** devComm,
                            int nQpsPerDev, int trafficClass) {
  return g_ginConnect(ctx, dev, handle, comm, devComm, nQpsPerDev, trafficClass);
}
ncclResult_t TestGinAccept(void* listenComm, void** comm, ncclNetDeviceHandle_t** devComm, int nQpsPerDev) {
  return g_ginAccept(listenComm, comm, devComm, nQpsPerDev);
}
void TestGinSetTrafficClass(void* ctx, int trafficClass) { g_ginSetTrafficClass(ctx, trafficClass); }
ncclResult_t TestGinCreateFlushQp(ncclIbRecvComm* comm) { return g_ginCreateFlushQp(comm); }
ncclResult_t TestGinRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd,
                                uint64_t flags, void** mhandle) {
  return g_ginRegMrDmaBuf(comm, data, size, type, offset, fd, flags, mhandle);
}
std::function<ncclResult_t(int, ncclNetProperties_t*)> g_ginNetGetProperties = kGinHookFails;
std::function<ncclResult_t(void*, int, void*, void**)> g_ginNetListen = kGinHookFails;
std::function<ncclResult_t(void*, void*, size_t, int, void**)> g_ginNetRegMr = kGinHookFails;
std::function<ncclResult_t(void*, void*)> g_ginNetDeregMr = kGinHookFails;
std::function<ncclResult_t(void*, void*, size_t, int, void*, void*, void**)> g_ginNetIsend = kGinHookFails;
std::function<ncclResult_t(void*, int, void**, size_t*, int*, void**, void**, void**)> g_ginNetIrecv =
    kGinHookFails;
std::function<ncclResult_t(void*, int*, int*)> g_ginNetTest = kGinHookFails;
std::function<ncclResult_t(void*)> g_ginNetCloseSend = kGinHookFails;
std::function<ncclResult_t(void*)> g_ginNetCloseRecv = kGinHookFails;
std::function<ncclResult_t(void*)> g_ginNetCloseListen = kGinHookFails;
std::function<ncclResult_t(ncclGinIbCollComm*, void*, void*, size_t)> g_ginCollAllGather = kGinHookFails;

ncclResult_t GinNetGetProperties(int dev, ncclNetProperties_t* props) {
  return g_ginNetGetProperties(dev, props);
}
ncclResult_t GinNetListen(void* ctx, int dev, void* handle, void** comm) {
  return g_ginNetListen(ctx, dev, handle, comm);
}
ncclResult_t GinNetRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  return g_ginNetRegMr(comm, data, size, type, mhandle);
}
ncclResult_t GinNetDeregMr(void* comm, void* mhandle) { return g_ginNetDeregMr(comm, mhandle); }
ncclResult_t GinNetIsend(void* comm, void* data, size_t size, int tag, void* mhandle, void* phandle,
                        void** request) {
  return g_ginNetIsend(comm, data, size, tag, mhandle, phandle, request);
}
ncclResult_t GinNetIrecv(void* comm, int n, void** data, size_t* sizes, int* tags, void** mhandles,
                        void** phandles, void** request) {
  return g_ginNetIrecv(comm, n, data, sizes, tags, mhandles, phandles, request);
}
ncclResult_t GinNetTest(void* request, int* done, int* sizes) {
  return g_ginNetTest(request, done, sizes);
}
ncclResult_t GinNetCloseSend(void* comm) { return g_ginNetCloseSend(comm); }
ncclResult_t GinNetCloseRecv(void* comm) { return g_ginNetCloseRecv(comm); }
ncclResult_t GinNetCloseListen(void* comm) { return g_ginNetCloseListen(comm); }
ncclResult_t GinCollAllGather(ncclGinIbCollComm* comm, void* src, void* dst, size_t size) {
  return g_ginCollAllGather(comm, src, dst, size);
}

ncclNet_t g_testGinNet{};

void ResetGinHooks() {
  g_ginInitDevices = kGinHookFails;
  g_ginFinalizeDevices = kGinHookFails;
  g_ginConnect = kGinHookFails;
  g_ginAccept = kGinHookFails;
  g_ginSetTrafficClass = kGinHookNoOp;
  g_ginCreateFlushQp = kGinHookFails;
  g_ginRegMrDmaBuf = kGinHookFails;
  g_ginNetGetProperties = kGinHookFails;
  g_ginNetListen = kGinHookFails;
  g_ginNetRegMr = kGinHookFails;
  g_ginNetDeregMr = kGinHookFails;
  g_ginNetIsend = kGinHookFails;
  g_ginNetIrecv = kGinHookFails;
  g_ginNetTest = kGinHookFails;
  g_ginNetCloseSend = kGinHookFails;
  g_ginNetCloseRecv = kGinHookFails;
  g_ginNetCloseListen = kGinHookFails;
  g_ginCollAllGather = kGinHookFails;
  std::memset(&g_testGinNet, 0, sizeof(g_testGinNet));
  g_testGinNet.name = "test-gin-net";
  g_testGinNet.getProperties = GinNetGetProperties;
  g_testGinNet.listen = GinNetListen;
  g_testGinNet.regMr = GinNetRegMr;
  g_testGinNet.deregMr = GinNetDeregMr;
  g_testGinNet.isend = GinNetIsend;
  g_testGinNet.irecv = GinNetIrecv;
  g_testGinNet.test = GinNetTest;
  g_testGinNet.closeSend = GinNetCloseSend;
  g_testGinNet.closeRecv = GinNetCloseRecv;
  g_testGinNet.closeListen = GinNetCloseListen;
}

}  // namespace

#undef ncclIbMalloc
#undef ncclCalloc
#define ncclIbMalloc(...) g_hostAllocs.Malloc(__VA_ARGS__, __FILE__, __LINE__, __func__)
#define ncclCalloc(...) g_hostAllocs.Calloc(__VA_ARGS__, __FILE__, __LINE__, __func__)
#define free(ptr) g_hostAllocs.Free(ptr)
#define ncclNetIb g_testGinNet
#define ncclIbInitDevices TestGinInitDevices
#define ncclIbFinalizeDevices TestGinFinalizeDevices
#define ncclIbGdrSupport NetIbTestGdrSupport
#define ncclIbPeerMemSupport NetIbTestPeerMemSupport
#define ncclIbDmaBufSupport NetIbTestDmaBufSupport
#define ncclIbConnectImpl TestGinConnect
#define ncclIbAcceptImpl TestGinAccept
#define ncclIbSetTrafficClass TestGinSetTrafficClass
#define ncclIbCreateFlushQp TestGinCreateFlushQp
#define ncclIbRegMrDmaBufInternal TestGinRegMrDmaBuf
#define memcpy TestGinMemcpy
#include NET_IB_GIN_CC_PATH
#undef memcpy
#undef ncclIbRegMrDmaBufInternal
#undef ncclIbCreateFlushQp
#undef ncclIbSetTrafficClass
#undef ncclIbAcceptImpl
#undef ncclIbConnectImpl
#undef ncclIbDmaBufSupport
#undef ncclIbPeerMemSupport
#undef ncclIbGdrSupport
#undef ncclIbFinalizeDevices
#undef ncclIbInitDevices
#undef ncclNetIb
#undef free
#undef ncclCalloc
#undef ncclIbMalloc

namespace {

class NetIbGinMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { ResetGinHooks(); }

  void TearDown() override {
    NetIbResetGlobals();
    g_ginSelfCopies = 0;
    ResetGinHooks();
    ResetNetIbSupportSeams();
    ResetSocketFakes();
    ResetNcclFakes();
    EXPECT_TRUE(g_hostAllocs.Clean()) << "GIN unit: " << g_hostAllocs.Summary();
    for (const auto& allocation : g_hostAllocs.live) std::free(allocation.first);
    g_hostAllocs.Reset();
  }
};

TEST_F(NetIbGinMicrotest, GdrSupport_PeerMemoryOrDmaBufMakesGinAvailable) {
  bool supported = false;
  {
    ScopedHook gdr(g_netIbGdrSupport, [] { return ncclSuccess; });
    ASSERT_EQ(ncclGinIbGdrSupport(&supported, false), ncclSuccess);
    EXPECT_TRUE(supported);
  }
  {
    ScopedHook dmaBuf(g_netIbDmaBufSupport, [](int dev) {
      EXPECT_EQ(dev, 0);
      return ncclSuccess;
    });
    ASSERT_EQ(ncclGinIbGdrSupport(&supported, false), ncclSuccess);
    EXPECT_TRUE(supported);
    EXPECT_EQ(dmaBuf.calls, 1);
  }
  ASSERT_EQ(ncclGinIbGdrSupport(&supported, false), ncclSuccess);
  EXPECT_FALSE(supported);
}

TEST_F(NetIbGinMicrotest, DevIsMlx5_RequiresValidMergedDeviceWithOnlyMlx5Members) {
  ncclNMergedIbDevs = 1;
  ncclNIbDevs = 2;
  ncclIbMergedDevs[0].vProps.ndevs = 2;
  ncclIbMergedDevs[0].vProps.devs[0] = 0;
  ncclIbMergedDevs[0].vProps.devs[1] = 1;
  ncclIbDevs[0].ibProvider = IB_PROVIDER_MLX5;
  ncclIbDevs[1].ibProvider = IB_PROVIDER_MLX5;

  EXPECT_FALSE(ncclNetIbDevIsMlx5(-1));
  EXPECT_FALSE(ncclNetIbDevIsMlx5(1));
  EXPECT_TRUE(ncclNetIbDevIsMlx5(0));

  ncclIbDevs[1].ibProvider = IB_PROVIDER_NONE;
  EXPECT_FALSE(ncclNetIbDevIsMlx5(0));
  ncclIbDevs[1].ibProvider = IB_PROVIDER_MLX5;
  ncclIbDevs[2].ibProvider = IB_PROVIDER_MLX5;
  ncclIbMergedDevs[0].vProps.devs[1] = 2;
  EXPECT_FALSE(ncclNetIbDevIsMlx5(0));
}

TEST_F(NetIbGinMicrotest, GdrGpuSupport_DmaBufWithoutPeerMemorySucceeds) {
  ScopedHook dmaBuf(g_netIbDmaBufSupport, [](int) { return ncclSuccess; });

  EXPECT_EQ(ncclGinIbGdrGpuSupport(false), ncclSuccess);
  EXPECT_EQ(dmaBuf.calls, 1);
}

TEST_F(NetIbGinMicrotest, InitType_ValidDeviceAndGdrPublishesDefaultTrafficClassContext) {
  ScopedHook init(g_ginInitDevices, [](ncclDebugLogger_t, ncclProfilerCallback_t) {
    ncclNIbDevs = 1;
    return ncclSuccess;
  });
  ScopedHook gdr(g_netIbGdrSupport, [] { return ncclSuccess; });
  void* context = nullptr;

  ASSERT_EQ(ncclGinIbInitType(&context, 7, nullptr, NCCL_GIN_TYPE_PROXY), ncclSuccess);
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(static_cast<ncclIbNetCtx*>(context)->config.trafficClass, NCCL_NET_TRAFFIC_CLASS_UNDEF);
  EXPECT_EQ(g_hostAllocs.live.size(), 1u);
  g_hostAllocs.Free(context);
}

TEST_F(NetIbGinMicrotest, InitType_NoDevicesOrGdrReturnsErrorWithoutPublishingContext) {
  void* context = nullptr;
  {
    ScopedHook init(g_ginInitDevices, [](ncclDebugLogger_t, ncclProfilerCallback_t) {
      ncclNIbDevs = 0;
      return ncclSuccess;
    });
    EXPECT_EQ(ncclGinIbInitType(&context, 0, nullptr, NCCL_GIN_TYPE_PROXY), ncclInternalError);
    EXPECT_EQ(context, nullptr);
  }
  ScopedHook init(g_ginInitDevices, [](ncclDebugLogger_t, ncclProfilerCallback_t) {
    ncclNIbDevs = 1;
    return ncclSuccess;
  });
  EXPECT_EQ(ncclGinIbInitType(&context, 0, nullptr, NCCL_GIN_TYPE_PROXY), ncclInternalError);
  EXPECT_EQ(context, nullptr);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

TEST_F(NetIbGinMicrotest, ProxyInit_UsesConfiguredTypeAndPublishesContext) {
  ScopedHook init(g_ginInitDevices, [](ncclDebugLogger_t, ncclProfilerCallback_t) {
    ncclNIbDevs = 1;
    return ncclSuccess;
  });
  ScopedHook peerMem(g_netIbPeerMemSupport, [] { return ncclSuccess; });
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "GIN_TYPE") == 0 ? int64_t{NCCL_GIN_TYPE_GDAKI} : fallback;
  });
  void* context = nullptr;

  ASSERT_EQ(ncclRmaIbProxyInit(&context, 42, nullptr), ncclSuccess);
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(peerMem.calls, 1);
  g_hostAllocs.Free(context);
}

TEST_F(NetIbGinMicrotest, Finalize_FreesContextAndPropagatesDeviceFinalize) {
  void* context = nullptr;
  ASSERT_EQ(g_hostAllocs.Malloc(&context, 16), ncclSuccess);
  ScopedHook finalize(g_ginFinalizeDevices, [] { return ncclRemoteError; });

  EXPECT_EQ(ncclGinIbFinalize(context), ncclRemoteError);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

TEST_F(NetIbGinMicrotest, RmaProperties_HipReportsSignalsDoNotFlushAllPuts) {
  ncclGinIbCollComm coll{};
  ncclRmaProperties_t props{};
  props.flushesAllPutsOnAnySignal = true;

  EXPECT_EQ(ncclRmaIbProxyGetRmaProperties(&coll, &props), ncclSuccess);
  EXPECT_FALSE(props.flushesAllPutsOnAnySignal);
}

TEST_F(NetIbGinMicrotest, GetProperties_DelegatesAndMarksGinProxyDevice) {
  ScopedHook getProperties(g_ginNetGetProperties, [](int dev, ncclNetProperties_t* props) {
    EXPECT_EQ(dev, 2);
    props->speed = 123;
    return ncclSuccess;
  });
  ncclNetProperties_t props{};

  ASSERT_EQ(ncclRmaIbProxyGetProperties(2, &props), ncclSuccess);
  EXPECT_EQ(props.speed, 123);
  EXPECT_EQ(props.netDeviceType, NCCL_NET_DEVICE_GIN_PROXY);
}

TEST_F(NetIbGinMicrotest, AllGather_ThreeRankRingCopiesLocalAndTransfersEachSlot) {
  ncclGinIbCollComm coll{};
  coll.rank = 1;
  coll.nranks = 3;
  coll.sendComm = reinterpret_cast<void*>(0x10);
  coll.recvComm = reinterpret_cast<void*>(0x20);
  int sendHandle = 1;
  int recvHandle = 2;
  ScopedHook regMr(g_ginNetRegMr, [&](void* comm, void*, size_t size, int type, void** handle) {
    EXPECT_EQ(size, 3 * sizeof(int));
    EXPECT_EQ(type, NCCL_PTR_HOST);
    *handle = comm == coll.recvComm ? static_cast<void*>(&recvHandle) : static_cast<void*>(&sendHandle);
    return ncclSuccess;
  });
  std::vector<std::pair<void*, void*>> deregistered;
  ScopedHook deregMr(g_ginNetDeregMr, [&](void* comm, void* handle) {
    deregistered.emplace_back(comm, handle);
    return ncclSuccess;
  });
  std::vector<void*> sent;
  int sendRequest = 1;
  ScopedHook isend(g_ginNetIsend, [&](void* comm, void* data, size_t size, int tag, void* mhandle, void*,
                                      void** request) {
    EXPECT_EQ(comm, coll.sendComm);
    EXPECT_EQ(size, sizeof(int));
    EXPECT_EQ(tag, NCCL_GIN_IB_ALLGATHER_TAG);
    EXPECT_EQ(mhandle, static_cast<void*>(&sendHandle));
    sent.push_back(data);
    *request = &sendRequest;
    return ncclSuccess;
  });
  std::vector<void*> received;
  int recvRequest = 2;
  ScopedHook irecv(g_ginNetIrecv, [&](void* comm, int n, void** data, size_t* sizes, int* tags, void** mhandles,
                                      void**, void** request) {
    EXPECT_EQ(comm, coll.recvComm);
    EXPECT_EQ(n, 1);
    EXPECT_EQ(*sizes, sizeof(int));
    EXPECT_EQ(*tags, NCCL_GIN_IB_ALLGATHER_TAG);
    EXPECT_EQ(*mhandles, static_cast<void*>(&recvHandle));
    received.push_back(*data);
    *request = &recvRequest;
    return ncclSuccess;
  });
  ScopedHook test(g_ginNetTest, [](void*, int* done, int*) {
    *done = 1;
    return ncclSuccess;
  });
  int source = 17;
  int gathered[3] = {};

  ASSERT_EQ(ncclGinIbAllGather(&coll, &source, gathered, sizeof(source)), ncclSuccess);
  EXPECT_EQ(gathered[1], source);
  ASSERT_EQ(sent.size(), 2u);
  EXPECT_EQ(sent[0], &gathered[1]);
  EXPECT_EQ(sent[1], &gathered[0]);
  ASSERT_EQ(received.size(), 2u);
  EXPECT_EQ(received[0], &gathered[0]);
  EXPECT_EQ(received[1], &gathered[2]);
  ASSERT_EQ(deregistered.size(), 2u);
  EXPECT_NE(std::find(deregistered.begin(), deregistered.end(), std::make_pair(coll.recvComm,
                                                                               static_cast<void*>(&recvHandle))),
            deregistered.end());
  EXPECT_NE(std::find(deregistered.begin(), deregistered.end(), std::make_pair(coll.sendComm,
                                                                               static_cast<void*>(&sendHandle))),
            deregistered.end());
}

TEST_F(NetIbGinMicrotest, AllGather_NullRequestsAndIncompleteTestsRetryWithoutAdvancingStep) {
  ncclGinIbCollComm coll{};
  coll.rank = 0;
  coll.nranks = 2;
  int registration = 1;
  ScopedHook regMr(g_ginNetRegMr, [&](void*, void*, size_t, int, void** handle) {
    *handle = &registration;
    return ncclSuccess;
  });
  ScopedHook deregMr(g_ginNetDeregMr, [](void*, void*) { return ncclSuccess; });
  // Each side posts its request on the second try and completes on its second test.
  int sendToken = 1;
  int recvToken = 2;
  bool sendRetried = false;
  bool recvRetried = false;
  ScopedHook isend(g_ginNetIsend, [&](void*, void*, size_t, int, void*, void*, void** request) {
    if (sendRetried) *request = &sendToken;
    sendRetried = true;
    return ncclSuccess;
  });
  ScopedHook irecv(g_ginNetIrecv, [&](void*, int, void**, size_t*, int*, void**, void**, void** request) {
    if (recvRetried) *request = &recvToken;
    recvRetried = true;
    return ncclSuccess;
  });
  bool sendTested = false;
  bool recvTested = false;
  ScopedHook test(g_ginNetTest, [&](void* request, int* done, int*) {
    bool& tested = request == &sendToken ? sendTested : recvTested;
    *done = tested;
    tested = true;
    return ncclSuccess;
  });
  int source = 7;
  int gathered[2] = {};

  ASSERT_EQ(ncclGinIbAllGather(&coll, &source, gathered, sizeof(source)), ncclSuccess);
  EXPECT_EQ(gathered[0], source);
  EXPECT_EQ(isend.calls, 2);
  EXPECT_EQ(irecv.calls, 2);
  EXPECT_EQ(test.calls, 4);
}

TEST_F(NetIbGinMicrotest, AllGather_SecondRegistrationFailureReleasesFirstHandle) {
  ncclGinIbCollComm coll{};
  coll.nranks = 2;
  int recvHandle = 1;
  bool registered = false;
  ScopedHook regMr(g_ginNetRegMr, [&](void*, void*, size_t, int, void** handle) {
    if (registered) return ncclSystemError;
    registered = true;
    *handle = &recvHandle;
    return ncclSuccess;
  });
  ScopedHook deregMr(g_ginNetDeregMr, [&](void*, void* handle) {
    EXPECT_EQ(handle, &recvHandle);
    return ncclSuccess;
  });
  int source = 1;
  int gathered[2] = {};

  EXPECT_EQ(ncclGinIbAllGather(&coll, &source, gathered, sizeof(source)), ncclSystemError);
  EXPECT_EQ(regMr.calls, 2);
  EXPECT_EQ(deregMr.calls, 1);
}

TEST_F(NetIbGinMicrotest, AllToAll_ExtractsLocalRankColumnAndFreesScratch) {
  ncclGinIbCollComm coll{};
  coll.rank = 1;
  coll.nranks = 3;
  coll.allGather = [](ncclGinIbCollComm*, void*, void* out, size_t size) {
    EXPECT_EQ(size, 3 * sizeof(int));
    int matrix[9] = {0, 1, 2, 10, 11, 12, 20, 21, 22};
    std::memcpy(out, matrix, sizeof(matrix));
    return ncclSuccess;
  };
  int source[3] = {};
  int received[3] = {};

  ASSERT_EQ(ncclGinIbAllToAll(&coll, source, received, sizeof(int)), ncclSuccess);
  EXPECT_EQ(received[0], 1);
  EXPECT_EQ(received[1], 11);
  EXPECT_EQ(received[2], 21);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

TEST_F(NetIbGinMicrotest, Connect_RetriesNullCommsAndPublishesRingContext) {
  // GIN_IB_TC wins when set; -1 falls back to IB_TC.
  struct TrafficClassCase {
    int64_t ginTc;
    int expected;
  };
  for (TrafficClassCase tcCase : {TrafficClassCase{-1, 4}, TrafficClassCase{9, 9}}) {
    SCOPED_TRACE(tcCase.ginTc);
    ncclIbListenComm listen{};
    listen.dev = 1;
    ncclGinIbCollComm* published = nullptr;
    void* handles[3] = {reinterpret_cast<void*>(0x10), reinterpret_cast<void*>(0x20),
                        reinterpret_cast<void*>(0x30)};
    ScopedHook params(g_loadParam, [tcCase](const char* name, int64_t fallback) {
      if (std::strcmp(name, "GIN_IB_TC") == 0) return tcCase.ginTc;
      if (std::strcmp(name, "IB_TC") == 0) return int64_t{4};
      return fallback;
    });
    // Both directions come up on the second try.
    int sendToken = 1;
    int recvToken = 2;
    bool sendRetried = false;
    bool recvRetried = false;
    ScopedHook connect(g_ginConnect, [&](void*, int dev, void* handle, void** comm, ncclNetDeviceHandle_t**,
                                         int nQps, int tc) {
      EXPECT_EQ(dev, 1);
      EXPECT_EQ(handle, handles[2]);
      EXPECT_EQ(nQps, 1);
      EXPECT_EQ(tc, tcCase.expected);
      if (sendRetried) *comm = &sendToken;
      sendRetried = true;
      return ncclSuccess;
    });
    ScopedHook accept(g_ginAccept, [&](void* seen, void** comm, ncclNetDeviceHandle_t**, int nQps) {
      EXPECT_EQ(seen, &listen);
      EXPECT_EQ(nQps, 1);
      if (recvRetried) *comm = &recvToken;
      recvRetried = true;
      return ncclSuccess;
    });
    ibv_context verbsContext{};
    ibv_pd pd{};
    ncclIbDevs[1].context = &verbsContext;
    ncclIbDevs[1].pd = &pd;

    ASSERT_EQ(ncclGinIbConnect(nullptr, handles, 3, 1, &listen, reinterpret_cast<void**>(&published)), ncclSuccess);
    ASSERT_NE(published, nullptr);
    EXPECT_EQ(connect.calls, 2);
    EXPECT_EQ(accept.calls, 2);
    EXPECT_EQ(published->rank, 1);
    EXPECT_EQ(published->nranks, 3);
    EXPECT_EQ(published->dev, 1);
    EXPECT_EQ(published->sendComm, &sendToken);
    EXPECT_EQ(published->recvComm, &recvToken);
    EXPECT_EQ(published->ib.context, &verbsContext);
    EXPECT_EQ(published->ib.pd, &pd);
    EXPECT_EQ(published->allGather, ncclGinIbAllGather);
    EXPECT_EQ(published->allToAll, ncclGinIbAllToAll);
    g_hostAllocs.Free(published);
  }
}

TEST_F(NetIbGinMicrotest, Connect_AcceptFailureClosesEstablishedSendComm) {
  ncclIbListenComm listen{};
  void* handles[1] = {nullptr};
  int sendToken = 1;
  ScopedHook connect(g_ginConnect, [&](void*, int, void*, void** comm, ncclNetDeviceHandle_t**, int, int) {
    *comm = &sendToken;
    return ncclSuccess;
  });
  ScopedHook accept(g_ginAccept, [](void*, void**, ncclNetDeviceHandle_t**, int) { return ncclSystemError; });
  ScopedHook closeSend(g_ginNetCloseSend, [&](void* comm) {
    EXPECT_EQ(comm, &sendToken);
    return ncclSuccess;
  });
  void* coll = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(ncclGinIbConnect(nullptr, handles, 1, 0, &listen, &coll), ncclSystemError);
  EXPECT_EQ(coll, nullptr);
  EXPECT_EQ(closeSend.calls, 1);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

TEST_F(NetIbGinMicrotest, ProxyConnect_GdrSupportedBuildsProxyConnection) {
  ncclIbListenComm listen{};
  void* handles[1] = {nullptr};
  int sendToken = 1;
  int recvToken = 2;
  ScopedHook gdr(g_netIbGdrSupport, [] { return ncclSuccess; });
  ScopedHook connect(g_ginConnect, [&](void*, int, void*, void** comm, ncclNetDeviceHandle_t**, int, int) {
    *comm = &sendToken;
    return ncclSuccess;
  });
  ScopedHook accept(g_ginAccept, [&](void*, void** comm, ncclNetDeviceHandle_t**, int) {
    *comm = &recvToken;
    return ncclSuccess;
  });
  void* coll = nullptr;

  ASSERT_EQ(ncclRmaIbProxyConnect(nullptr, handles, 1, 0, &listen, &coll), ncclSuccess);
  ASSERT_NE(coll, nullptr);
  g_hostAllocs.Free(coll);
}

TEST_F(NetIbGinMicrotest, ProxyConnect_NoGdrReturnsInvalidUsageBeforeConnecting) {
  ScopedHook connect(g_ginConnect, [](void*, int, void*, void**, ncclNetDeviceHandle_t**, int, int) {
    return ncclSuccess;
  });
  void* coll = nullptr;

  EXPECT_EQ(ncclRmaIbProxyConnect(nullptr, nullptr, 1, 0, nullptr, &coll), ncclInvalidUsage);
  EXPECT_EQ(connect.calls, 0);
  EXPECT_EQ(coll, nullptr);
}

TEST_F(NetIbGinMicrotest, CloseColl_ClosesBothDirectionsAndFreesContext) {
  ncclGinIbCollComm* coll = nullptr;
  ASSERT_EQ(g_hostAllocs.Malloc(reinterpret_cast<void**>(&coll), sizeof(*coll)), ncclSuccess);
  int sendToken = 1;
  int recvToken = 2;
  coll->sendComm = &sendToken;
  coll->recvComm = &recvToken;
  std::vector<void*> closed;
  ScopedHook closeRecv(g_ginNetCloseRecv, [&](void* comm) {
    closed.push_back(comm);
    return ncclSuccess;
  });
  ScopedHook closeSend(g_ginNetCloseSend, [&](void* comm) {
    closed.push_back(comm);
    return ncclSuccess;
  });

  ASSERT_EQ(ncclGinIbCloseColl(coll), ncclSuccess);
  ASSERT_EQ(closed.size(), 2u);
  EXPECT_NE(std::find(closed.begin(), closed.end(), &recvToken), closed.end());
  EXPECT_NE(std::find(closed.begin(), closed.end(), &sendToken), closed.end());
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

// One rank's ncclRmaIbProxyCreateContext. The handle all-gather writes each rank's index into its slot, so every
// connect shows whose handle it was given; the per-peer barriers run the real ring all-gather.
struct GinCreateContextScene {
  GinCreateContextScene(int nranks, int rank) {
    coll.rank = rank;
    coll.nranks = nranks;
    coll.dev = 0;
    coll.ctx = &transportContext;
    coll.allGather = HandleAllGather;
  }
  GinCreateContextScene(const GinCreateContextScene&) = delete;
  GinCreateContextScene& operator=(const GinCreateContextScene&) = delete;

  static ncclResult_t HandleAllGather(ncclGinIbCollComm* comm, void* src, void* dst, size_t size) {
    EXPECT_EQ(size, static_cast<size_t>(NCCL_NET_HANDLE_MAXSIZE));
    EXPECT_EQ(src, static_cast<void*>(static_cast<char*>(dst) + comm->rank * size));
    for (int r = 0; r < comm->nranks; ++r) std::memcpy(static_cast<char*>(dst) + r * size, &r, sizeof(r));
    return ncclSuccess;
  }

  ncclGinIbCollComm coll{};
  int transportContext = 1;
  int listenToken = 1;
  int sendToken = 1;
  int recvToken = 2;
  int registration = 1;
  int barrierRequest = 1;
  int trafficClass = -1;
  std::vector<int> connectPeers;
  std::vector<int> connectTcs;
  std::vector<void**> connectSlots;
  std::vector<void**> acceptSlots;

  HookOf<decltype(g_ginNetListen)> listen{g_ginNetListen, [this](void* ctx, int dev, void*, void** comm) {
    EXPECT_EQ(ctx, &transportContext);
    EXPECT_EQ(dev, 0);
    *comm = &listenToken;
    return ncclSuccess;
  }};
  HookOf<decltype(g_ginNetCloseListen)> closeListen{g_ginNetCloseListen, [this](void* comm) {
    EXPECT_EQ(comm, &listenToken);
    return ncclSuccess;
  }};
  HookOf<decltype(g_ginSetTrafficClass)> setTrafficClass{g_ginSetTrafficClass, [this](void* ctx, int tc) {
    EXPECT_EQ(ctx, &transportContext);
    trafficClass = tc;
  }};
  HookOf<decltype(g_ginConnect)> connect{
      g_ginConnect, [this](void*, int, void* handle, void** comm, ncclNetDeviceHandle_t**, int, int tc) {
        int peer = -1;
        std::memcpy(&peer, handle, sizeof(peer));
        connectPeers.push_back(peer);
        connectTcs.push_back(tc);
        connectSlots.push_back(comm);
        *comm = &sendToken;
        return ncclSuccess;
      }};
  HookOf<decltype(g_ginAccept)> accept{g_ginAccept, [this](void*, void** comm, ncclNetDeviceHandle_t**, int) {
    acceptSlots.push_back(comm);
    *comm = &recvToken;
    return ncclSuccess;
  }};
  HookOf<decltype(g_ginNetRegMr)> regMr{g_ginNetRegMr, [this](void*, void*, size_t, int, void** handle) {
    *handle = &registration;
    return ncclSuccess;
  }};
  HookOf<decltype(g_ginNetDeregMr)> deregMr{g_ginNetDeregMr, [](void*, void*) { return ncclSuccess; }};
  HookOf<decltype(g_ginNetIsend)> isend{
      g_ginNetIsend, [this](void*, void*, size_t, int, void*, void*, void** request) {
        *request = &barrierRequest;
        return ncclSuccess;
      }};
  HookOf<decltype(g_ginNetIrecv)> irecv{
      g_ginNetIrecv, [this](void*, int, void**, size_t*, int*, void**, void**, void** request) {
        *request = &barrierRequest;
        return ncclSuccess;
      }};
  HookOf<decltype(g_ginNetTest)> test{g_ginNetTest, [](void*, int* done, int*) {
    *done = 1;
    return ncclSuccess;
  }};
  HookOf<decltype(g_ginNetCloseSend)> closeSend{g_ginNetCloseSend, [](void*) { return ncclSuccess; }};
  HookOf<decltype(g_ginNetCloseRecv)> closeRecv{g_ginNetCloseRecv, [](void*) { return ncclSuccess; }};
};

TEST_F(NetIbGinMicrotest, CreateContext_SingleRankBuildsContextsAndClampsBatchSize) {
  struct BatchCase {
    int64_t requested;
    int expected;
  };
  for (BatchCase batch :
       {BatchCase{1, NCCL_RMA_MAX_IB_WRS_PER_OP}, BatchCase{1000, NCCL_RMA_IB_WR_BATCHSIZE}}) {
    SCOPED_TRACE(batch.requested);
    GinCreateContextScene scene(/*nranks=*/1, /*rank=*/0);
    ScopedHook params(g_loadParam, [batch](const char* name, int64_t fallback) {
      return std::strcmp(name, "RMA_IB_WR_BATCHSIZE") == 0 ? batch.requested : fallback;
    });
    ncclRmaConfig_t config{};
    config.nContexts = 2;
    config.rankStride = 1;
    config.trafficClass = 7;
    ncclRmaIbProxyCtx* contexts = nullptr;

    ASSERT_EQ(ncclRmaIbProxyCreateContext(&scene.coll, &config, reinterpret_cast<void**>(&contexts)), ncclSuccess);
    ASSERT_NE(contexts, nullptr);
    EXPECT_EQ(scene.listen.calls, 1);
    EXPECT_EQ(scene.closeListen.calls, 1);
    EXPECT_EQ(scene.setTrafficClass.calls, 1);
    EXPECT_EQ(scene.trafficClass, 7);
    EXPECT_EQ(scene.connectPeers.size(), 2u);
    EXPECT_EQ(scene.acceptSlots.size(), 2u);
    EXPECT_EQ(contexts[0].nContexts, 2);
    EXPECT_EQ(contexts[0].nranks, 1);
    EXPECT_EQ(contexts[0].wrBatchsize, batch.expected);
    EXPECT_EQ(contexts[1].wrBatchsize, batch.expected);

    ASSERT_EQ(ncclRmaIbProxyDestroyContext(contexts), ncclSuccess);
    EXPECT_EQ(scene.closeSend.calls, 2);
    EXPECT_EQ(scene.closeRecv.calls, 2);
    // AICOMRCCL-3094: DestroyContext leaves the top-level array to its caller.
    g_hostAllocs.Free(contexts);
  }
}

TEST_F(NetIbGinMicrotest, CreateContext_RankStrideConnectsForwardAndAcceptsBackward) {
  // Six ranks, not four: with a stride of 2, rank + 2 and rank - 2 coincide mod 4, which would hide a swapped peer.
  GinCreateContextScene scene(/*nranks=*/6, /*rank=*/1);
  ncclRmaConfig_t config{};
  config.nContexts = 1;
  config.rankStride = 2;
  ncclRmaIbProxyCtx* contexts = nullptr;

  ASSERT_EQ(ncclRmaIbProxyCreateContext(&scene.coll, &config, reinterpret_cast<void**>(&contexts)), ncclSuccess);
  ASSERT_NE(contexts, nullptr);
  void** sends = contexts[0].fullSendComm;
  void** recvs = contexts[0].fullRecvComm;
  EXPECT_EQ(scene.connectPeers, (std::vector<int>{1, 3, 5}));
  EXPECT_EQ(scene.connectSlots, (std::vector<void**>{&sends[1], &sends[3], &sends[5]}));
  EXPECT_EQ(scene.acceptSlots, (std::vector<void**>{&recvs[1], &recvs[5], &recvs[3]}));
  // A ring barrier of nranks - 1 sends after each of the three peers.
  EXPECT_EQ(scene.isend.calls, 3 * 5);

  ASSERT_EQ(ncclRmaIbProxyDestroyContext(contexts), ncclSuccess);
  EXPECT_EQ(scene.closeSend.calls, 6);
  EXPECT_EQ(scene.closeRecv.calls, 6);
  g_hostAllocs.Free(contexts);
}

TEST_F(NetIbGinMicrotest, CreateContext_NoRankStrideConnectsEveryRank) {
  GinCreateContextScene scene(/*nranks=*/4, /*rank=*/1);
  ncclRmaConfig_t config{};
  config.nContexts = 1;
  config.rankStride = 0;
  ncclRmaIbProxyCtx* contexts = nullptr;

  ASSERT_EQ(ncclRmaIbProxyCreateContext(&scene.coll, &config, reinterpret_cast<void**>(&contexts)), ncclSuccess);
  EXPECT_EQ(scene.connectPeers, (std::vector<int>{1, 2, 3, 0}));
  ASSERT_EQ(ncclRmaIbProxyDestroyContext(contexts), ncclSuccess);
  g_hostAllocs.Free(contexts);
}

TEST_F(NetIbGinMicrotest, CreateContext_PeerConnectionsTakeGinIbTcOverIbTc) {
  struct TrafficClassCase {
    int64_t ginTc;
    int expected;
  };
  for (TrafficClassCase tcCase : {TrafficClassCase{-1, 4}, TrafficClassCase{9, 9}}) {
    SCOPED_TRACE(tcCase.ginTc);
    GinCreateContextScene scene(/*nranks=*/2, /*rank=*/1);
    ScopedHook params(g_loadParam, [tcCase](const char* name, int64_t fallback) {
      if (std::strcmp(name, "GIN_IB_TC") == 0) return tcCase.ginTc;
      if (std::strcmp(name, "IB_TC") == 0) return int64_t{4};
      return fallback;
    });
    ncclRmaConfig_t config{};
    config.nContexts = 1;
    config.rankStride = 1;
    ncclRmaIbProxyCtx* contexts = nullptr;

    ASSERT_EQ(ncclRmaIbProxyCreateContext(&scene.coll, &config, reinterpret_cast<void**>(&contexts)), ncclSuccess);
    EXPECT_EQ(scene.connectTcs, (std::vector<int>{tcCase.expected, tcCase.expected}));
    ASSERT_EQ(ncclRmaIbProxyDestroyContext(contexts), ncclSuccess);
    g_hostAllocs.Free(contexts);
  }
}

TEST_F(NetIbGinMicrotest, CreateContext_NonDivisibleStrideIsRejectedBeforeAllocation) {
  ncclGinIbCollComm coll{};
  coll.nranks = 3;
  ncclRmaConfig_t config{};
  config.nContexts = 1;
  config.rankStride = 2;
  void* context = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(ncclRmaIbProxyCreateContext(&coll, &config, &context), ncclInternalError);
  EXPECT_EQ(g_hostAllocs.calls, 0);
}

// AICOMRCCL-3094: partial context creation must release child arrays allocated
// before a later child allocation fails.
TEST_F(NetIbGinMicrotest, DISABLED_CreateContext_PartialAllocationFailureReleasesEarlierChildren) {
  ncclGinIbCollComm coll{};
  coll.rank = 0;
  coll.nranks = 1;
  coll.allGather = [](ncclGinIbCollComm*, void* src, void* dst, size_t size) {
    TestGinMemcpy(dst, src, size);
    return ncclSuccess;
  };
  int listenToken = 1;
  ScopedHook listen(g_ginNetListen, [&](void*, int, void*, void** comm) {
    *comm = &listenToken;
    return ncclSuccess;
  });
  ScopedHook closeListen(g_ginNetCloseListen, [](void*) { return ncclSuccess; });
  // top-level context, handles, fullSendComm, then fullRecvComm
  g_hostAllocs.failCall = 4;
  ncclRmaConfig_t config{};
  config.nContexts = 1;
  config.rankStride = 1;
  void* context = nullptr;

  EXPECT_EQ(ncclRmaIbProxyCreateContext(&coll, &config, &context), ncclSystemError);
  EXPECT_EQ(context, nullptr);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

// AICOMRCCL-3094: failure after peers are established must close both
// directions and free all context allocations.
TEST_F(NetIbGinMicrotest, DISABLED_CreateContext_BarrierFailureClosesEstablishedPeersAndFreesContext) {
  ncclGinIbCollComm coll{};
  coll.rank = 0;
  coll.nranks = 1;
  coll.allGather = [](ncclGinIbCollComm*, void* src, void* dst, size_t size) {
    TestGinMemcpy(dst, src, size);
    return ncclSuccess;
  };
  int listenToken = 1;
  int sendToken = 2;
  int recvToken = 3;
  ScopedHook listen(g_ginNetListen, [&](void*, int, void*, void** comm) {
    *comm = &listenToken;
    return ncclSuccess;
  });
  ScopedHook closeListen(g_ginNetCloseListen, [](void*) { return ncclSuccess; });
  ScopedHook connect(g_ginConnect, [&](void*, int, void*, void** comm, ncclNetDeviceHandle_t**, int, int) {
    *comm = &sendToken;
    return ncclSuccess;
  });
  ScopedHook accept(g_ginAccept, [&](void*, void** comm, ncclNetDeviceHandle_t**, int) {
    *comm = &recvToken;
    return ncclSuccess;
  });
  ScopedHook closeSend(g_ginNetCloseSend, [&](void* comm) {
    EXPECT_EQ(comm, &sendToken);
    return ncclSuccess;
  });
  ScopedHook closeRecv(g_ginNetCloseRecv, [&](void* comm) {
    EXPECT_EQ(comm, &recvToken);
    return ncclSuccess;
  });
  // top-level context, handles, fullSendComm, fullRecvComm, then barrier dummy
  g_hostAllocs.failCall = 5;
  ncclRmaConfig_t config{};
  config.nContexts = 1;
  config.rankStride = 1;
  void* context = nullptr;

  EXPECT_EQ(ncclRmaIbProxyCreateContext(&coll, &config, &context), ncclSystemError);
  EXPECT_EQ(context, nullptr);
  EXPECT_EQ(closeSend.calls, 1);
  EXPECT_EQ(closeRecv.calls, 1);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

TEST_F(NetIbGinMicrotest, RegMrSym_GathersBaseAddressesAndRkeysThenDeregisters) {
  ncclGinIbCollComm coll{};
  coll.rank = 0;
  coll.nranks = 2;
  int recvToken = 1;
  coll.recvComm = &recvToken;
  ibv_mr mr{};
  mr.rkey = 0x44;
  ncclIbMrHandle ibHandle{};
  ibHandle.mrs[0] = &mr;
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [&](void* comm, void*, size_t, int, uint64_t, int, uint64_t,
                                               void** handle) {
    EXPECT_EQ(comm, &recvToken);
    *handle = &ibHandle;
    return ncclSuccess;
  });
  // The first gather is each rank's base VA, the second its rkey.
  ScopedHook gather(g_ginCollAllGather, [](ncclGinIbCollComm*, void* src, void* dst, size_t size) {
    if (size == sizeof(uintptr_t)) {
      auto* bases = static_cast<uintptr_t*>(dst);
      bases[0] = reinterpret_cast<uintptr_t>(*static_cast<void**>(src));
      bases[1] = 0x2000;
    } else {
      EXPECT_EQ(size, sizeof(uint32_t));
      auto* rkeys = static_cast<uint32_t*>(dst);
      rkeys[0] = *static_cast<uint32_t*>(src);
      rkeys[1] = 0x55;
    }
    return ncclSuccess;
  });
  coll.allGather = GinCollAllGather;
  ScopedHook deregMr(g_ginNetDeregMr, [&](void* comm, void* handle) {
    EXPECT_EQ(comm, &recvToken);
    EXPECT_EQ(handle, &ibHandle);
    return ncclSuccess;
  });
  int data = 0;
  ncclRmaIbProxyMrHandle* handle = nullptr;

  ASSERT_EQ(ncclRmaIbProxyRegMrSym(&coll, &data, sizeof(data), NCCL_PTR_HOST, 0,
                                  reinterpret_cast<void**>(&handle)),
            ncclSuccess);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(gather.calls, 2);
  EXPECT_EQ(handle->base_vas[0], reinterpret_cast<uintptr_t>(&data));
  EXPECT_EQ(handle->base_vas[1], 0x2000u);
  EXPECT_EQ(handle->rkeys[0], 0x44u);
  EXPECT_EQ(handle->rkeys[1], 0x55u);
  ASSERT_EQ(ncclRmaIbProxyDeregMrSym(&coll, handle), ncclSuccess);
  EXPECT_EQ(deregMr.calls, 1);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

TEST_F(NetIbGinMicrotest, RegMrSym_FirstGatherFailureDeregistersAndFreesAllWrappers) {
  ncclGinIbCollComm coll{};
  coll.nranks = 2;
  int recvToken = 1;
  coll.recvComm = &recvToken;
  ibv_mr mr{};
  ncclIbMrHandle ibHandle{};
  ibHandle.mrs[0] = &mr;
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [&](void*, void*, size_t, int, uint64_t, int, uint64_t,
                                               void** handle) {
    *handle = &ibHandle;
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, [](ncclGinIbCollComm*, void*, void*, size_t) { return ncclSystemError; });
  coll.allGather = GinCollAllGather;
  ScopedHook deregMr(g_ginNetDeregMr, [&](void*, void* handle) {
    EXPECT_EQ(handle, &ibHandle);
    return ncclSuccess;
  });
  int data = 0;
  void* handle = nullptr;

  EXPECT_EQ(ncclRmaIbProxyRegMrSym(&coll, &data, sizeof(data), NCCL_PTR_HOST, 0, &handle), ncclSystemError);
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(deregMr.calls, 1);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

struct GinDataScene {
  ncclRmaIbProxyCtx context{};
  std::array<void*, 2> sendComms{};
  std::array<void*, 2> recvComms{};
  ncclIbSendComm send{};
  ncclIbRecvComm recv{};
  IbVerbsScene verbs{"test-hca"};
  ibv_qp* const dataQp = &verbs.qps[0];
  ibv_qp* const flushQp = &verbs.qps[1];
  ibv_mr srcMr{};
  ibv_mr localMr{};
  ibv_mr scratchMr{};
  ncclIbMrHandle srcIbHandle{};
  ncclIbMrHandle localIbHandle{};
  std::array<uintptr_t, 2> srcBases{{0x1000, 0x2000}};
  std::array<uintptr_t, 2> dstBases{{0x3000, 0x4000}};
  std::array<uint32_t, 2> srcRkeys{{0x10, 0x11}};
  std::array<uint32_t, 2> dstRkeys{{0x20, 0x21}};
  ncclRmaIbProxyMrHandle srcHandle{};
  ncclRmaIbProxyMrHandle dstHandle{};

  GinDataScene() {
    context.rank = 0;
    context.nranks = 2;
    context.nContexts = 1;
    context.wrBatchsize = NCCL_RMA_IB_WR_BATCHSIZE;
    context.fullSendComm = sendComms.data();
    context.fullRecvComm = recvComms.data();
    sendComms[1] = &send;
    recvComms[1] = &recv;
    send.base.isSend = true;
    send.base.vProps.ndevs = 1;
    send.base.nqps = 1;
    send.base.nRemDevs = 1;
    send.base.qps[0].qp = dataQp;
    send.base.qps[0].devIndex = 0;
    send.devs[0].base.cq = &verbs.cq;
    send.devs[0].base.pd = &verbs.pd;
    recv.base.isSend = false;
    recv.base.vProps.ndevs = 1;
    recv.base.nRemDevs = 1;
    recv.devs[0].base.cq = &verbs.cq;
    recv.devs[0].base.pd = &verbs.pd;
    recv.flushEnabled = true;
    recv.devs[0].gpuFlush.qp.qp = flushQp;
    recv.devs[0].gpuFlush.qp.devIndex = 0;
    recv.devs[0].gpuFlush.sge.addr = 0x9000;
    recv.devs[0].gpuFlush.sge.length = 8;
    recv.devs[0].gpuFlush.sge.lkey = 0x90;
    scratchMr.lkey = 0x91;
    send.devs[0].putSignalScratchpadMr = &scratchMr;
    srcMr.lkey = 0x31;
    srcMr.rkey = 0x32;
    localMr.lkey = 0x41;
    localMr.rkey = 0x42;
    srcIbHandle.mrs[0] = &srcMr;
    localIbHandle.mrs[0] = &localMr;
    srcHandle = {&srcIbHandle, srcBases.data(), srcRkeys.data()};
    dstHandle = {&localIbHandle, dstBases.data(), dstRkeys.data()};
  }

  // An 8-byte put to rank 1; NULL if the post fails.
  ncclIbRequest* PostPut(uint32_t optFlags = 0) {
    void* request = nullptr;
    EXPECT_EQ(ncclRmaIbProxyIPut(&context, 0, 0, &srcHandle, 8, 0, &dstHandle, 1, optFlags, &request), ncclSuccess);
    return static_cast<ncclIbRequest*>(request);
  }

  void QueueCompletion(ncclIbRequest* req, ibv_wc_status status = IBV_WC_SUCCESS) {
    ibv_wc completion{};
    completion.status = status;
    completion.wr_id = req - req->base->reqs;
    verbs.wcs.push_back(completion);
  }
};

TEST_F(NetIbGinMicrotest, ReserveWrs_QpSwitchPostsPendingChainBeforeChangingQp) {
  GinDataScene scene;
  ibv_qp* otherQp = &scene.verbs.qps[2];
  ibv_send_wr* wr = nullptr;
  ibv_sge* sge = nullptr;
  ASSERT_EQ(ncclRmaIbProxyReserveWrs(&scene.context, scene.dataQp, 1, &wr, &sge), ncclSuccess);
  wr->opcode = IBV_WR_RDMA_WRITE;
  wr->sg_list = sge;
  wr->num_sge = 1;
  ASSERT_EQ(ncclRmaIbProxyCommitWrs(&scene.context, 1, true), ncclSuccess);
  ASSERT_TRUE(scene.verbs.posted.empty());

  ASSERT_EQ(ncclRmaIbProxyReserveWrs(&scene.context, otherQp, 1, &wr, &sge), ncclSuccess);
  ASSERT_EQ(scene.verbs.posted.size(), 1u);
  EXPECT_EQ(scene.context.batch.qp, otherQp);
  EXPECT_EQ(scene.context.batch.nWrs, 0);
}

TEST_F(NetIbGinMicrotest, CommitWrs_AggregatePostsWhenNextMaximumOperationWouldNotFit) {
  GinDataScene scene;
  scene.context.wrBatchsize = 2;
  ibv_send_wr* wr = nullptr;
  ibv_sge* sge = nullptr;
  ASSERT_EQ(ncclRmaIbProxyReserveWrs(&scene.context, scene.dataQp, 1, &wr, &sge), ncclSuccess);
  wr->opcode = IBV_WR_RDMA_WRITE;
  wr->sg_list = sge;
  wr->num_sge = 1;

  ASSERT_EQ(ncclRmaIbProxyCommitWrs(&scene.context, 1, true), ncclSuccess);

  ASSERT_EQ(scene.verbs.posted.size(), 1u);
  EXPECT_EQ(scene.context.batch.nWrs, 0);
  EXPECT_EQ(scene.context.batch.qp, nullptr);
}

TEST_F(NetIbGinMicrotest, CommitWrs_AggregateDefersWhenNextMaximumOperationExactlyFits) {
  GinDataScene scene;
  scene.context.wrBatchsize = 3;
  ibv_send_wr* wr = nullptr;
  ibv_sge* sge = nullptr;
  ASSERT_EQ(ncclRmaIbProxyReserveWrs(&scene.context, scene.dataQp, 1, &wr, &sge), ncclSuccess);
  wr->opcode = IBV_WR_RDMA_WRITE;
  wr->sg_list = sge;
  wr->num_sge = 1;

  ASSERT_EQ(ncclRmaIbProxyCommitWrs(&scene.context, 1, true), ncclSuccess);

  EXPECT_TRUE(scene.verbs.posted.empty());
  EXPECT_EQ(scene.context.batch.nWrs, 1);
  EXPECT_EQ(scene.context.batch.qp, scene.dataQp);
}

TEST_F(NetIbGinMicrotest, DataApis_UnconnectedPeersRejectWithoutRequestOrPost) {
  GinDataScene scene;
  scene.sendComms[1] = nullptr;
  scene.recvComms[1] = nullptr;
  void* request = nullptr;

  EXPECT_EQ(ncclRmaIbProxyIPut(&scene.context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1, 0,
                              &request),
            ncclInvalidUsage);
  EXPECT_EQ(ncclRmaIbProxyIGet(&scene.context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1, 0,
                              &request),
            ncclInvalidUsage);
  EXPECT_EQ(ncclRmaIbProxyIFlush(&scene.context, 0, &scene.dstHandle, 1, &request), ncclInvalidUsage);
  EXPECT_EQ(request, nullptr);
  EXPECT_TRUE(scene.verbs.posted.empty());
  EXPECT_EQ(scene.send.ginSeq.posted, 0u);
  EXPECT_EQ(scene.recv.ginSeq.posted, 0u);
}

TEST_F(NetIbGinMicrotest, IPut_PostsExactWriteAndPublishesRequest) {
  GinDataScene scene;
  void* request = nullptr;

  ASSERT_EQ(ncclRmaIbProxyIPut(&scene.context, 0, 3, &scene.srcHandle, 16, 5, &scene.dstHandle, 1, 0,
                              &request),
            ncclSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(scene.verbs.posted.size(), 1u);
  EXPECT_EQ(scene.verbs.posted[0].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_NE(scene.verbs.posted[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(scene.verbs.posted[0].remoteAddress, 0x4005u);
  EXPECT_EQ(scene.verbs.posted[0].rkey, 0x21u);
  EXPECT_EQ(scene.verbs.posted[0].sge.addr, 0x1003u);
  EXPECT_EQ(scene.verbs.posted[0].sge.length, 16u);
  EXPECT_EQ(scene.verbs.posted[0].sge.lkey, 0x31u);
  auto* req = static_cast<ncclIbRequest*>(request);
  EXPECT_EQ(req->type, NCCL_NET_IB_REQ_GIN_IPUT);
  EXPECT_EQ(req->iput.rank, 1u);
  EXPECT_EQ(req->events[0], 1);
}

TEST_F(NetIbGinMicrotest, IGet_PostsExactReadAndPublishesRequest) {
  GinDataScene scene;
  void* request = nullptr;

  ASSERT_EQ(ncclRmaIbProxyIGet(&scene.context, 0, 7, &scene.srcHandle, 32, 9, &scene.dstHandle, 1, 0,
                              &request),
            ncclSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(scene.verbs.posted.size(), 1u);
  EXPECT_EQ(scene.verbs.posted[0].opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(scene.verbs.posted[0].remoteAddress, 0x2007u);
  EXPECT_EQ(scene.verbs.posted[0].rkey, 0x11u);
  EXPECT_EQ(scene.verbs.posted[0].sge.addr, 0x3009u);
  EXPECT_EQ(scene.verbs.posted[0].sge.length, 32u);
  EXPECT_EQ(scene.verbs.posted[0].sge.lkey, 0x41u);
  EXPECT_EQ(static_cast<ncclIbRequest*>(request)->type, NCCL_NET_IB_REQ_GIN_IGET);
}

TEST_F(NetIbGinMicrotest, IPutSignal_PutAndAddPostOrderedWriteThenAtomic) {
  GinDataScene scene;
  void* request = nullptr;

  ASSERT_EQ(ncclRmaIbProxyIPutSignal(&scene.context, 0, 1, &scene.srcHandle, 8, 2, &scene.dstHandle, 1, 8,
                                    &scene.dstHandle, 9, NCCL_NET_SIGNAL_OP_ADD, false, 0, &request),
            ncclSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(scene.verbs.posted.size(), 2u);
  EXPECT_EQ(scene.verbs.posted[0].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(scene.verbs.posted[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(scene.verbs.posted[0].remoteAddress, 0x4002u);
  EXPECT_EQ(scene.verbs.posted[1].opcode, IBV_WR_ATOMIC_FETCH_AND_ADD);
  EXPECT_EQ(scene.verbs.posted[1].remoteAddress, 0x4008u);
  EXPECT_EQ(scene.verbs.posted[1].rkey, 0x21u);
  EXPECT_EQ(scene.verbs.posted[1].compareAdd, 9u);
  EXPECT_EQ(scene.verbs.posted[1].sge.addr, reinterpret_cast<uintptr_t>(&scene.send.putSignalScratchpad));
  EXPECT_EQ(scene.verbs.posted[1].sge.lkey, 0x91u);
  EXPECT_NE(scene.verbs.posted[1].sendFlags & IBV_SEND_SIGNALED, 0);
}

TEST_F(NetIbGinMicrotest, IPutSignal_SignalOnlyIncrementPostsOneAtomicWithValueOne) {
  GinDataScene scene;
  void* request = nullptr;

  ASSERT_EQ(ncclRmaIbProxyIPutSignal(&scene.context, 0, 0, nullptr, 0, 0, nullptr, 1, 8, &scene.dstHandle, 99,
                                    NCCL_NET_SIGNAL_OP_INC, false, 0, &request),
            ncclSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(scene.verbs.posted.size(), 1u);
  EXPECT_EQ(scene.verbs.posted[0].opcode, IBV_WR_ATOMIC_FETCH_AND_ADD);
  EXPECT_EQ(scene.verbs.posted[0].remoteAddress, 0x4008u);
  EXPECT_EQ(scene.verbs.posted[0].compareAdd, 1u);
  EXPECT_NE(scene.verbs.posted[0].sendFlags & IBV_SEND_SIGNALED, 0);
}

TEST_F(NetIbGinMicrotest, IPutSignal_UnsupportedOperationIsRejectedBeforeRequestOrPost) {
  GinDataScene scene;
  void* request = nullptr;

  EXPECT_EQ(ncclRmaIbProxyIPutSignal(&scene.context, 0, 0, nullptr, 0, 0, nullptr, 1, 8, &scene.dstHandle, 1,
                                    99, false, 0, &request),
            ncclInvalidArgument);
  EXPECT_EQ(request, nullptr);
  EXPECT_TRUE(scene.verbs.posted.empty());
  EXPECT_EQ(scene.send.ginSeq.posted, 0u);
}

// AICOMRCCL-3094: 64-bit RDMA atomics require an 8-byte aligned remote
// address, matching the validation already used by the socket RMA backend.
TEST_F(NetIbGinMicrotest, DISABLED_IPutSignal_UnalignedSignalIsRejectedBeforeRequestOrPost) {
  GinDataScene scene;
  void* request = nullptr;

  EXPECT_EQ(ncclRmaIbProxyIPutSignal(&scene.context, 0, 0, nullptr, 0, 0, nullptr, 1, 4, &scene.dstHandle, 1,
                                    NCCL_NET_SIGNAL_OP_INC, false, 0, &request),
            ncclInvalidArgument);
  EXPECT_EQ(request, nullptr);
  EXPECT_TRUE(scene.verbs.posted.empty());
  EXPECT_EQ(scene.send.ginSeq.posted, 0u);
}

TEST_F(NetIbGinMicrotest, AggregatedPut_DefersDoorbellUntilTestAndCompletesRequest) {
  GinDataScene scene;
  ncclIbRequest* req = scene.PostPut(ncclRmaOptFlagsAggregateRequests);
  ASSERT_NE(req, nullptr);
  EXPECT_TRUE(scene.verbs.posted.empty());
  scene.QueueCompletion(req);
  int done = 0;

  ASSERT_EQ(ncclRmaIbProxyTest(nullptr, req, &done), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(scene.verbs.posted.size(), 1u);
  EXPECT_EQ(req->type, NCCL_NET_IB_REQ_UNUSED);
  EXPECT_EQ(scene.send.ginSeq.completed, req->id);
}

TEST_F(NetIbGinMicrotest, Test_AlreadyCompletedSequenceFreesWithoutPolling) {
  GinDataScene scene;
  ncclIbRequest* req = scene.PostPut();
  ASSERT_NE(req, nullptr);
  scene.send.ginSeq.completed = req->id;
  scene.verbs.pollErr = -EIO;
  int done = 0;

  ASSERT_EQ(ncclRmaIbProxyTest(nullptr, req, &done), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(req->type, NCCL_NET_IB_REQ_UNUSED);
}

TEST_F(NetIbGinMicrotest, Test_ErrorCompletionReturnsRemoteErrorWithoutCompleting) {
  GinDataScene scene;
  ncclIbRequest* req = scene.PostPut();
  ASSERT_NE(req, nullptr);
  scene.QueueCompletion(req, IBV_WC_REM_ACCESS_ERR);
  ScopedHook getAddr(g_ncclSocketGetAddr, [](ncclSocket*, ncclSocketAddress* addr) {
    std::memset(addr, 0, sizeof(*addr));
    return ncclSuccess;
  });
  ScopedHook toString(g_ncclSocketToString, [](const ncclSocketAddress*, char* line, size_t length, int) {
    std::snprintf(line, length, "test-peer");
    return line;
  });
  int done = 0;

  EXPECT_EQ(ncclRmaIbProxyTest(nullptr, req, &done), ncclRemoteError);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(scene.send.ginSeq.completed, 0u);
  ncclIbFreeRequest(req);
}

TEST_F(NetIbGinMicrotest, Test_PollFailurePropagatesWithoutCompletingRequest) {
  GinDataScene scene;
  ncclIbRequest* req = scene.PostPut();
  ASSERT_NE(req, nullptr);
  scene.verbs.pollErr = -EIO;
  int done = 0;

  EXPECT_EQ(ncclRmaIbProxyTest(nullptr, req, &done), ncclSystemError);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(scene.send.ginSeq.completed, 0u);
  ncclIbFreeRequest(req);
}

TEST_F(NetIbGinMicrotest, IFlush_DrainsAggregatedPutsThenReadsLocalRegistrationOnFlushQp) {
  GinDataScene scene;
  ScopedHook createFlushQp(g_ginCreateFlushQp, [&](ncclIbRecvComm* comm) {
    EXPECT_EQ(comm, &scene.recv);
    return ncclSuccess;
  });
  ASSERT_NE(scene.PostPut(ncclRmaOptFlagsAggregateRequests), nullptr);
  ASSERT_TRUE(scene.verbs.posted.empty());
  void* request = nullptr;

  ASSERT_EQ(ncclRmaIbProxyIFlush(&scene.context, 0, &scene.dstHandle, 1, &request), ncclSuccess);
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(createFlushQp.calls, 1);
  ASSERT_EQ(scene.verbs.posted.size(), 2u);
  EXPECT_EQ(scene.verbs.posted[0].qp, scene.dataQp);
  EXPECT_EQ(scene.verbs.posted[0].opcode, IBV_WR_RDMA_WRITE);
  const auto& flush = scene.verbs.posted[1];
  EXPECT_EQ(flush.qp, scene.flushQp);
  EXPECT_EQ(flush.opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(flush.remoteAddress, 0x3000u);
  EXPECT_EQ(flush.rkey, 0x20u);
  EXPECT_EQ(flush.sge.addr, 0x9000u);
  EXPECT_NE(flush.sendFlags & IBV_SEND_SIGNALED, 0);
}

TEST_F(NetIbGinMicrotest, IFlush_DisabledFlushReturnsNoRequest) {
  GinDataScene scene;
  scene.recv.flushEnabled = false;
  void* request = reinterpret_cast<void*>(0x1);

  ASSERT_EQ(ncclRmaIbProxyIFlush(&scene.context, 0, &scene.dstHandle, 1, &request), ncclSuccess);
  EXPECT_EQ(request, nullptr);
  EXPECT_TRUE(scene.verbs.posted.empty());
  EXPECT_EQ(scene.recv.ginSeq.posted, 0u);
}

TEST_F(NetIbGinMicrotest, IFlush_CompletionAdvancesReceiveSequenceAndFreesRequest) {
  GinDataScene scene;
  ScopedHook createFlushQp(g_ginCreateFlushQp, [](ncclIbRecvComm*) { return ncclSuccess; });
  void* request = nullptr;
  ASSERT_EQ(ncclRmaIbProxyIFlush(&scene.context, 0, &scene.dstHandle, 1, &request), ncclSuccess);
  auto* req = static_cast<ncclIbRequest*>(request);
  scene.QueueCompletion(req);
  int done = 0;

  ASSERT_EQ(ncclRmaIbProxyTest(nullptr, request, &done), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(scene.recv.ginSeq.completed, req->id);
  EXPECT_EQ(req->type, NCCL_NET_IB_REQ_UNUSED);
}

// AICOMRCCL-3094: a failed post should release the request, sequence, and
// retained batch state allocated for an operation that never reached hardware.
TEST_F(NetIbGinMicrotest, DISABLED_IPut_PostFailureRollsBackRequestSequenceAndBatch) {
  GinDataScene scene;
  scene.verbs.postSendErr = EIO;
  void* request = nullptr;

  EXPECT_EQ(ncclRmaIbProxyIPut(&scene.context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1, 0,
                              &request),
            ncclSystemError);
  EXPECT_EQ(request, nullptr);
  EXPECT_EQ(scene.send.ginSeq.posted, 0u);
  EXPECT_EQ(scene.send.base.reqs[0].type, NCCL_NET_IB_REQ_UNUSED);
  EXPECT_EQ(scene.context.batch.nWrs, 0);
  EXPECT_EQ(scene.context.batch.qp, nullptr);
}

// AICOMRCCL-3094: every RMA entry point must roll back request ownership and
// sequence/batch state when its hardware post fails.
TEST_F(NetIbGinMicrotest, DISABLED_OtherDataApis_PostFailureRollsBackRequestSequenceAndBatch) {
  {
    GinDataScene scene;
    scene.verbs.postSendErr = EIO;
    void* request = nullptr;
    EXPECT_EQ(ncclRmaIbProxyIGet(&scene.context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1, 0,
                                &request),
              ncclSystemError);
    EXPECT_EQ(request, nullptr);
    EXPECT_EQ(scene.send.ginSeq.posted, 0u);
    EXPECT_EQ(scene.send.base.reqs[0].type, NCCL_NET_IB_REQ_UNUSED);
    EXPECT_EQ(scene.context.batch.nWrs, 0);
    EXPECT_EQ(scene.context.batch.qp, nullptr);
  }
  {
    GinDataScene scene;
    scene.verbs.postSendErr = EIO;
    void* request = nullptr;
    EXPECT_EQ(ncclRmaIbProxyIPutSignal(&scene.context, 0, 0, nullptr, 0, 0, nullptr, 1, 8,
                                      &scene.dstHandle, 1, NCCL_NET_SIGNAL_OP_INC, false, 0, &request),
              ncclSystemError);
    EXPECT_EQ(request, nullptr);
    EXPECT_EQ(scene.send.ginSeq.posted, 0u);
    EXPECT_EQ(scene.send.base.reqs[0].type, NCCL_NET_IB_REQ_UNUSED);
    EXPECT_EQ(scene.context.batch.nWrs, 0);
    EXPECT_EQ(scene.context.batch.qp, nullptr);
  }
  {
    GinDataScene scene;
    scene.verbs.postSendErr = EIO;
    ScopedHook createFlushQp(g_ginCreateFlushQp, [](ncclIbRecvComm*) { return ncclSuccess; });
    void* request = nullptr;
    EXPECT_EQ(ncclRmaIbProxyIFlush(&scene.context, 0, &scene.dstHandle, 1, &request), ncclSystemError);
    EXPECT_EQ(request, nullptr);
    EXPECT_EQ(scene.recv.ginSeq.posted, 0u);
    EXPECT_EQ(scene.recv.base.reqs[0].type, NCCL_NET_IB_REQ_UNUSED);
  }
}

// AICOMRCCL-3094: a failed barrier all-gather should release its
// temporary allocation before returning the dependency error.
TEST_F(NetIbGinMicrotest, DISABLED_P2PBarrier_AllGatherFailureFreesScratch) {
  ncclGinIbCollComm coll{};
  coll.rank = 0;
  coll.nranks = 1;
  ScopedHook regMr(g_ginNetRegMr, [](void*, void*, size_t, int, void**) { return ncclSystemError; });

  EXPECT_EQ(ncclGinIbP2PBarrier(&coll), ncclSystemError);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

// AICOMRCCL-3094: the local all-gather contribution should not invoke memcpy
// with identical source and destination ranges.
TEST_F(NetIbGinMicrotest, DISABLED_P2PBarrier_LocalContributionAvoidsOverlappingMemcpy) {
  ncclGinIbCollComm coll{};
  coll.rank = 0;
  coll.nranks = 1;
  int registration = 1;
  ScopedHook regMr(g_ginNetRegMr, [&](void*, void*, size_t, int, void** handle) {
    *handle = &registration;
    return ncclSuccess;
  });
  ScopedHook deregMr(g_ginNetDeregMr, [](void*, void*) { return ncclSuccess; });

  ASSERT_EQ(ncclGinIbP2PBarrier(&coll), ncclSuccess);
  EXPECT_EQ(g_ginSelfCopies, 0);
}

// AICOMRCCL-3094: the backend context is owned by DestroyContext,
// just like its child arrays and peer communications.
TEST_F(NetIbGinMicrotest, DISABLED_DestroyContext_FreesTopLevelContext) {
  ncclRmaIbProxyCtx* context = nullptr;
  ASSERT_EQ(g_hostAllocs.Calloc(&context, 1), ncclSuccess);
  context->nContexts = 1;
  context->nranks = 0;

  ASSERT_EQ(ncclRmaIbProxyDestroyContext(context), ncclSuccess);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

// AICOMRCCL-3094: disabling the explicit flush must not leave an
// already-aggregated batch indefinitely unposted.
TEST_F(NetIbGinMicrotest, DISABLED_IFlush_DisabledFlushStillDrainsAggregatedBatch) {
  GinDataScene scene;
  scene.recv.flushEnabled = false;
  ASSERT_NE(scene.PostPut(ncclRmaOptFlagsAggregateRequests), nullptr);
  ASSERT_TRUE(scene.verbs.posted.empty());
  void* flushRequest = reinterpret_cast<void*>(0x1);

  ASSERT_EQ(ncclRmaIbProxyIFlush(&scene.context, 0, &scene.dstHandle, 1, &flushRequest), ncclSuccess);
  EXPECT_EQ(flushRequest, nullptr);
  EXPECT_EQ(scene.verbs.posted.size(), 1u);
}

TEST_F(NetIbGinMicrotest, RmaVtable_ExportsProxyBackendContract) {
  EXPECT_STREQ(ncclRmaIbProxy.name, "RMA_IB_PROXY");
  EXPECT_EQ(ncclRmaIbProxy.iput, ncclRmaIbProxyIPut);
  EXPECT_EQ(ncclRmaIbProxy.iputSignal, ncclRmaIbProxyIPutSignal);
  EXPECT_EQ(ncclRmaIbProxy.iget, ncclRmaIbProxyIGet);
  EXPECT_EQ(ncclRmaIbProxy.iflush, ncclRmaIbProxyIFlush);
  EXPECT_EQ(ncclRmaIbProxy.test, ncclRmaIbProxyTest);
  EXPECT_EQ(ncclRmaIbProxy.rmaProgress, nullptr);
  EXPECT_EQ(ncclRmaIbProxy.queryLastError, nullptr);
}

}  // namespace
