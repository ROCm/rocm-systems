/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib_multiseg/gin.cc.
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
#include <fcntl.h>
#include <functional>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "HostAllocLedger.h"
#include "IbVerbsScene.h"
#include "ScopedHook.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/os_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/rccl_mseg_fakes.h"
#include "fakes/socket_fakes.h"
#include "gin/gin_host.h"
#include "transport/net_ib_multiseg/NetIbMsegGlobals.h"
#include "transport/net_ib/NetIbSupportSeams.h"
#include "transport/net_ib_multiseg/common_mseg.h"
#include "transport/net_ib_multiseg/gin_mseg.h"

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
std::function<ncclResult_t(IbMsegRecvComm*)> g_ginCreateFlushQp = kGinHookFails;
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
ncclResult_t TestGinCreateFlushQp(IbMsegRecvComm* comm) { return g_ginCreateFlushQp(comm); }
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
std::function<ncclResult_t(IbMsegGinIbCollComm*, void*, void*, size_t)> g_ginCollAllGather = kGinHookFails;

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
ncclResult_t GinCollAllGather(IbMsegGinIbCollComm* comm, void* src, void* dst, size_t size) {
  return g_ginCollAllGather(comm, src, dst, size);
}

ncclNet_t g_testGinNet{};

void ResetGinHooks() {
  g_hostSegmentFdLookup = [](uintptr_t, size_t, uint64_t*) { return -1; };
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
#define ncclNetIbMseg g_testGinNet
#define IbMsegInitDevices TestGinInitDevices
#define IbMsegFinalizeDevices TestGinFinalizeDevices
#define IbMsegGdrSupport NetIbTestGdrSupport
#define IbMsegPeerMemSupport NetIbTestPeerMemSupport
#define IbMsegDmaBufSupport NetIbTestDmaBufSupport
#define IbMsegConnectImpl TestGinConnect
#define IbMsegAcceptImpl TestGinAccept
#define IbMsegSetTrafficClass TestGinSetTrafficClass
#define IbMsegCreateFlushQp TestGinCreateFlushQp
#define IbMsegRegMrDmaBufInternal TestGinRegMrDmaBuf
#define memcpy TestGinMemcpy
#include NET_IB_MSEG_GIN_CC_PATH
#undef memcpy
#undef IbMsegRegMrDmaBufInternal
#undef IbMsegCreateFlushQp
#undef IbMsegSetTrafficClass
#undef IbMsegAcceptImpl
#undef IbMsegConnectImpl
#undef IbMsegDmaBufSupport
#undef IbMsegPeerMemSupport
#undef IbMsegGdrSupport
#undef IbMsegFinalizeDevices
#undef IbMsegInitDevices
#undef ncclNetIbMseg
#undef free
#undef ncclCalloc
#undef ncclIbMalloc

namespace {

class NetIbMsegGinMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { ResetGinHooks(); }

  void TearDown() override {
    NetIbMsegResetGlobals();
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

TEST(NetIbMsegGinRmaSeqId, ReleaseSeqIdOnlyReturnsNewestId) {
  uint64_t posted = 5;
  rcclRmaReleaseSeqId(/*id=*/5, &posted);
  EXPECT_EQ(posted, uint64_t{4});

  posted = 7;
  rcclRmaReleaseSeqId(/*id=*/6, &posted);
  EXPECT_EQ(posted, uint64_t{7}) << "a later request already holds id 7";
}

TEST_F(NetIbMsegGinMicrotest, GdrSupport_PeerMemoryOrDmaBufMakesGinAvailable) {
  bool supported = false;
  {
    ScopedHook gdr(g_netIbGdrSupport, [] { return ncclSuccess; });
    ASSERT_EQ(IbMsegGinIbGdrSupport(&supported, false), ncclSuccess);
    EXPECT_TRUE(supported);
  }
  {
    ScopedHook dmaBuf(g_netIbDmaBufSupport, [](int dev) {
      EXPECT_EQ(dev, 0);
      return ncclSuccess;
    });
    ASSERT_EQ(IbMsegGinIbGdrSupport(&supported, false), ncclSuccess);
    EXPECT_TRUE(supported);
    EXPECT_EQ(dmaBuf.calls, 1);
  }
  ASSERT_EQ(IbMsegGinIbGdrSupport(&supported, false), ncclSuccess);
  EXPECT_FALSE(supported);
}

TEST_F(NetIbMsegGinMicrotest, DevIsMlx5_RequiresValidMergedDeviceWithOnlyMlx5Members) {
  IbMsegNMergedDevs = 1;
  IbMsegNDevs = 2;
  IbMsegMergedDevs[0].vProps.ndevs = 2;
  IbMsegMergedDevs[0].vProps.devs[0] = 0;
  IbMsegMergedDevs[0].vProps.devs[1] = 1;
  IbMsegDevs[0].ibProvider = IB_PROVIDER_MLX5;
  IbMsegDevs[1].ibProvider = IB_PROVIDER_MLX5;

  EXPECT_FALSE(IbMsegNetDevIsMlx5(-1));
  EXPECT_FALSE(IbMsegNetDevIsMlx5(1));
  EXPECT_TRUE(IbMsegNetDevIsMlx5(0));

  IbMsegDevs[1].ibProvider = IB_PROVIDER_NONE;
  EXPECT_FALSE(IbMsegNetDevIsMlx5(0));
  IbMsegDevs[1].ibProvider = IB_PROVIDER_MLX5;
  IbMsegDevs[2].ibProvider = IB_PROVIDER_MLX5;
  IbMsegMergedDevs[0].vProps.devs[1] = 2;
  EXPECT_FALSE(IbMsegNetDevIsMlx5(0));
}

TEST_F(NetIbMsegGinMicrotest, GdrGpuSupport_DmaBufWithoutPeerMemorySucceeds) {
  ScopedHook dmaBuf(g_netIbDmaBufSupport, [](int) { return ncclSuccess; });

  EXPECT_EQ(IbMsegGinIbGdrGpuSupport(false), ncclSuccess);
  EXPECT_EQ(dmaBuf.calls, 1);
}

TEST_F(NetIbMsegGinMicrotest, InitType_ValidDeviceAndGdrPublishesDefaultTrafficClassContext) {
  ScopedHook init(g_ginInitDevices, [](ncclDebugLogger_t, ncclProfilerCallback_t) {
    IbMsegNDevs = 1;
    return ncclSuccess;
  });
  ScopedHook gdr(g_netIbGdrSupport, [] { return ncclSuccess; });
  void* context = nullptr;

  ASSERT_EQ(IbMsegGinIbInitType(&context, 7, nullptr, NCCL_GIN_TYPE_PROXY), ncclSuccess);
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(static_cast<IbMsegNetCtx*>(context)->config.trafficClass, NCCL_NET_TRAFFIC_CLASS_UNDEF);
  EXPECT_EQ(g_hostAllocs.live.size(), 1u);
  g_hostAllocs.Free(context);
}

TEST_F(NetIbMsegGinMicrotest, InitType_NoDevicesOrGdrReturnsErrorWithoutPublishingContext) {
  void* context = nullptr;
  {
    ScopedHook init(g_ginInitDevices, [](ncclDebugLogger_t, ncclProfilerCallback_t) {
      IbMsegNDevs = 0;
      return ncclSuccess;
    });
    EXPECT_EQ(IbMsegGinIbInitType(&context, 0, nullptr, NCCL_GIN_TYPE_PROXY), ncclInternalError);
    EXPECT_EQ(context, nullptr);
  }
  ScopedHook init(g_ginInitDevices, [](ncclDebugLogger_t, ncclProfilerCallback_t) {
    IbMsegNDevs = 1;
    return ncclSuccess;
  });
  EXPECT_EQ(IbMsegGinIbInitType(&context, 0, nullptr, NCCL_GIN_TYPE_PROXY), ncclInternalError);
  EXPECT_EQ(context, nullptr);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

TEST_F(NetIbMsegGinMicrotest, ProxyInit_UsesConfiguredTypeAndPublishesContext) {
  ScopedHook init(g_ginInitDevices, [](ncclDebugLogger_t, ncclProfilerCallback_t) {
    IbMsegNDevs = 1;
    return ncclSuccess;
  });
  ScopedHook peerMem(g_netIbPeerMemSupport, [] { return ncclSuccess; });
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "GIN_TYPE") == 0 ? int64_t{NCCL_GIN_TYPE_GDAKI} : fallback;
  });
  void* context = nullptr;

  ASSERT_EQ(IbMsegRmaIbProxyInit(&context, 42, nullptr), ncclSuccess);
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(peerMem.calls, 1);
  g_hostAllocs.Free(context);
}

TEST_F(NetIbMsegGinMicrotest, Finalize_FreesContextAndPropagatesDeviceFinalize) {
  void* context = nullptr;
  ASSERT_EQ(g_hostAllocs.Malloc(&context, 16), ncclSuccess);
  ScopedHook finalize(g_ginFinalizeDevices, [] { return ncclRemoteError; });

  EXPECT_EQ(IbMsegGinIbFinalize(context), ncclRemoteError);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

TEST_F(NetIbMsegGinMicrotest, RmaProperties_HipReportsSignalsDoNotFlushAllPuts) {
  IbMsegGinIbCollComm coll{};
  ncclRmaProperties_t props{};
  props.flushesAllPutsOnAnySignal = true;

  EXPECT_EQ(IbMsegRmaIbProxyGetRmaProperties(&coll, &props), ncclSuccess);
  EXPECT_FALSE(props.flushesAllPutsOnAnySignal);
}

TEST_F(NetIbMsegGinMicrotest, GetProperties_DelegatesAndMarksGinProxyDevice) {
  ScopedHook getProperties(g_ginNetGetProperties, [](int dev, ncclNetProperties_t* props) {
    EXPECT_EQ(dev, 2);
    props->speed = 123;
    return ncclSuccess;
  });
  ncclNetProperties_t props{};

  ASSERT_EQ(IbMsegRmaIbProxyGetProperties(2, &props), ncclSuccess);
  EXPECT_EQ(props.speed, 123);
  EXPECT_EQ(props.netDeviceType, NCCL_NET_DEVICE_GIN_PROXY);
}

TEST_F(NetIbMsegGinMicrotest, AllGather_ThreeRankRingCopiesLocalAndTransfersEachSlot) {
  IbMsegGinIbCollComm coll{};
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

  ASSERT_EQ(IbMsegGinIbAllGather(&coll, &source, gathered, sizeof(source)), ncclSuccess);
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

TEST_F(NetIbMsegGinMicrotest, AllGather_NullRequestsAndIncompleteTestsRetryWithoutAdvancingStep) {
  IbMsegGinIbCollComm coll{};
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

  ASSERT_EQ(IbMsegGinIbAllGather(&coll, &source, gathered, sizeof(source)), ncclSuccess);
  EXPECT_EQ(gathered[0], source);
  EXPECT_EQ(isend.calls, 2);
  EXPECT_EQ(irecv.calls, 2);
  EXPECT_EQ(test.calls, 4);
}

TEST_F(NetIbMsegGinMicrotest, AllGather_SecondRegistrationFailureReleasesFirstHandle) {
  IbMsegGinIbCollComm coll{};
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

  EXPECT_EQ(IbMsegGinIbAllGather(&coll, &source, gathered, sizeof(source)), ncclSystemError);
  EXPECT_EQ(regMr.calls, 2);
  EXPECT_EQ(deregMr.calls, 1);
}

TEST_F(NetIbMsegGinMicrotest, AllToAll_ExtractsLocalRankColumnAndFreesScratch) {
  IbMsegGinIbCollComm coll{};
  coll.rank = 1;
  coll.nranks = 3;
  coll.allGather = [](IbMsegGinIbCollComm*, void*, void* out, size_t size) {
    EXPECT_EQ(size, 3 * sizeof(int));
    int matrix[9] = {0, 1, 2, 10, 11, 12, 20, 21, 22};
    std::memcpy(out, matrix, sizeof(matrix));
    return ncclSuccess;
  };
  int source[3] = {};
  int received[3] = {};

  ASSERT_EQ(IbMsegGinIbAllToAll(&coll, source, received, sizeof(int)), ncclSuccess);
  EXPECT_EQ(received[0], 1);
  EXPECT_EQ(received[1], 11);
  EXPECT_EQ(received[2], 21);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

TEST_F(NetIbMsegGinMicrotest, Connect_RetriesNullCommsAndPublishesRingContext) {
  // GIN_IB_TC wins when set; -1 falls back to IB_TC.
  struct TrafficClassCase {
    int64_t ginTc;
    int expected;
  };
  for (TrafficClassCase tcCase : {TrafficClassCase{-1, 4}, TrafficClassCase{9, 9}}) {
    SCOPED_TRACE(tcCase.ginTc);
    IbMsegListenComm listen{};
    listen.dev = 1;
    IbMsegGinIbCollComm* published = nullptr;
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
    IbMsegDevs[1].context = &verbsContext;
    IbMsegDevs[1].pd = &pd;

    volatile uint32_t abortFlag = 0;

    ASSERT_EQ(IbMsegGinIbConnect(nullptr, handles, 3, 1, &listen, reinterpret_cast<void**>(&published), &abortFlag),
              ncclSuccess);
    ASSERT_NE(published, nullptr);
    // The abort flag is kept so the coll comm's all-gather can bail out on abort.
    EXPECT_EQ(published->abortFlag, &abortFlag);
    EXPECT_EQ(connect.calls, 2);
    EXPECT_EQ(accept.calls, 2);
    EXPECT_EQ(published->rank, 1);
    EXPECT_EQ(published->nranks, 3);
    EXPECT_EQ(published->dev, 1);
    EXPECT_EQ(published->sendComm, &sendToken);
    EXPECT_EQ(published->recvComm, &recvToken);
    EXPECT_EQ(published->ib.context, &verbsContext);
    EXPECT_EQ(published->ib.pd, &pd);
    EXPECT_EQ(published->allGather, IbMsegGinIbAllGather);
    EXPECT_EQ(published->allToAll, IbMsegGinIbAllToAll);
    g_hostAllocs.Free(published);
  }
}

TEST_F(NetIbMsegGinMicrotest, Connect_AcceptFailureClosesEstablishedSendComm) {
  IbMsegListenComm listen{};
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

  EXPECT_EQ(IbMsegGinIbConnect(nullptr, handles, 1, 0, &listen, &coll, nullptr), ncclSystemError);
  EXPECT_EQ(coll, nullptr);
  EXPECT_EQ(closeSend.calls, 1);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

TEST_F(NetIbMsegGinMicrotest, Connect_AbortFlagStopsRetryLoopAndClosesEstablishedSendComm) {
  IbMsegListenComm listen{};
  void* handles[1] = {nullptr};
  int sendToken = 1;
  volatile uint32_t abortFlag = 0;
  ScopedHook connect(g_ginConnect, [&](void*, int, void*, void** comm, ncclNetDeviceHandle_t**, int, int) {
    *comm = &sendToken;
    return ncclSuccess;
  });
  // The peer never shows up; the abort is raised while waiting for it.
  ScopedHook accept(g_ginAccept, [&](void*, void**, ncclNetDeviceHandle_t**, int) {
    abortFlag = 1;
    return ncclSuccess;
  });
  ScopedHook closeSend(g_ginNetCloseSend, [&](void* comm) {
    EXPECT_EQ(comm, &sendToken);
    return ncclSuccess;
  });
  void* coll = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(IbMsegGinIbConnect(nullptr, handles, 1, 0, &listen, &coll, &abortFlag), ncclInternalError);
  EXPECT_EQ(coll, nullptr);
  EXPECT_EQ(connect.calls, 1);
  EXPECT_EQ(accept.calls, 1);
  EXPECT_EQ(closeSend.calls, 1);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

TEST_F(NetIbMsegGinMicrotest, ProxyConnect_GdrSupportedBuildsProxyConnection) {
  IbMsegListenComm listen{};
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

  ASSERT_EQ(IbMsegRmaIbProxyConnect(nullptr, handles, 1, 0, &listen, &coll, nullptr), ncclSuccess);
  ASSERT_NE(coll, nullptr);
  g_hostAllocs.Free(coll);
}

// Switch on: connect allocates the per-rank registration records and close frees them.
TEST_F(NetIbMsegGinMicrotest, ProxyConnect_MultiSegAllocatesRegRecordsThatCloseFrees) {
  IbMsegListenComm listen{};
  void* handles[1] = {nullptr};
  int sendToken = 1;
  int recvToken = 2;
  ScopedHook multiSeg(g_multiSegOn, [](bool pluginRegistersPerSegment) { return pluginRegistersPerSegment; });
  ScopedHook gdr(g_netIbGdrSupport, [] { return ncclSuccess; });
  ScopedHook connect(g_ginConnect, [&](void*, int, void*, void** comm, ncclNetDeviceHandle_t**, int, int) {
    *comm = &sendToken;
    return ncclSuccess;
  });
  ScopedHook accept(g_ginAccept, [&](void*, void** comm, ncclNetDeviceHandle_t**, int) {
    *comm = &recvToken;
    return ncclSuccess;
  });
  ScopedHook closeRecv(g_ginNetCloseRecv, [](void*) { return ncclSuccess; });
  ScopedHook closeSend(g_ginNetCloseSend, [](void*) { return ncclSuccess; });
  void* coll = nullptr;

  ASSERT_EQ(IbMsegRmaIbProxyConnect(nullptr, handles, 1, 0, &listen, &coll, nullptr), ncclSuccess);
  ASSERT_NE(coll, nullptr);
  auto* records = static_cast<IbMsegGinIbCollComm*>(coll)->regRecords;
  ASSERT_NE(records, nullptr);
  EXPECT_TRUE(g_hostAllocs.Covers(records, sizeof(rcclIbSegSymRecord)));

  ASSERT_EQ(IbMsegGinIbCloseColl(coll), ncclSuccess);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

// Switch on: a failed record allocation closes both directions and hands back no coll comm.
TEST_F(NetIbMsegGinMicrotest, ProxyConnect_MultiSegRecordAllocFailureClosesCollComm) {
  IbMsegListenComm listen{};
  void* handles[1] = {nullptr};
  int sendToken = 1;
  int recvToken = 2;
  ScopedHook multiSeg(g_multiSegOn, [](bool pluginRegistersPerSegment) { return pluginRegistersPerSegment; });
  ScopedHook gdr(g_netIbGdrSupport, [] { return ncclSuccess; });
  ScopedHook connect(g_ginConnect, [&](void*, int, void*, void** comm, ncclNetDeviceHandle_t**, int, int) {
    *comm = &sendToken;
    return ncclSuccess;
  });
  ScopedHook accept(g_ginAccept, [&](void*, void** comm, ncclNetDeviceHandle_t**, int) {
    *comm = &recvToken;
    return ncclSuccess;
  });
  ScopedHook closeRecv(g_ginNetCloseRecv, [](void*) { return ncclSuccess; });
  ScopedHook closeSend(g_ginNetCloseSend, [](void*) { return ncclSuccess; });
  g_hostAllocs.failCall = 2; // the coll comm is call 1, the records call 2
  void* coll = nullptr;

  EXPECT_EQ(IbMsegRmaIbProxyConnect(nullptr, handles, 1, 0, &listen, &coll, nullptr), ncclSystemError);
  EXPECT_EQ(coll, nullptr);
  EXPECT_EQ(closeRecv.calls, 1);
  EXPECT_EQ(closeSend.calls, 1);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

TEST_F(NetIbMsegGinMicrotest, ProxyConnect_NoGdrReturnsInvalidUsageBeforeConnecting) {
  ScopedHook connect(g_ginConnect, [](void*, int, void*, void**, ncclNetDeviceHandle_t**, int, int) {
    return ncclSuccess;
  });
  void* coll = nullptr;

  EXPECT_EQ(IbMsegRmaIbProxyConnect(nullptr, nullptr, 1, 0, nullptr, &coll, nullptr), ncclInvalidUsage);
  EXPECT_EQ(connect.calls, 0);
  EXPECT_EQ(coll, nullptr);
}

TEST_F(NetIbMsegGinMicrotest, CloseColl_ClosesBothDirectionsAndFreesContext) {
  IbMsegGinIbCollComm* coll = nullptr;
  ASSERT_EQ(g_hostAllocs.Malloc(reinterpret_cast<void**>(&coll), sizeof(*coll)), ncclSuccess);
  int sendToken = 1;
  int recvToken = 2;
  coll->sendComm = &sendToken;
  coll->recvComm = &recvToken;
  ASSERT_EQ(g_hostAllocs.Malloc(reinterpret_cast<void**>(&coll->regRecords), sizeof(rcclIbSegSymRecord)), ncclSuccess);
  std::vector<void*> closed;
  ScopedHook closeRecv(g_ginNetCloseRecv, [&](void* comm) {
    closed.push_back(comm);
    return ncclSuccess;
  });
  ScopedHook closeSend(g_ginNetCloseSend, [&](void* comm) {
    closed.push_back(comm);
    return ncclSuccess;
  });

  ASSERT_EQ(IbMsegGinIbCloseColl(coll), ncclSuccess);
  ASSERT_EQ(closed.size(), 2u);
  EXPECT_NE(std::find(closed.begin(), closed.end(), &recvToken), closed.end());
  EXPECT_NE(std::find(closed.begin(), closed.end(), &sendToken), closed.end());
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

// One rank's IbMsegRmaIbProxyCreateContext. The handle all-gather writes each rank's index into its slot, so every
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

  static ncclResult_t HandleAllGather(IbMsegGinIbCollComm* comm, void* src, void* dst, size_t size) {
    EXPECT_EQ(size, static_cast<size_t>(NCCL_NET_HANDLE_MAXSIZE));
    EXPECT_EQ(src, static_cast<void*>(static_cast<char*>(dst) + comm->rank * size));
    for (int r = 0; r < comm->nranks; ++r) std::memcpy(static_cast<char*>(dst) + r * size, &r, sizeof(r));
    return ncclSuccess;
  }

  IbMsegGinIbCollComm coll{};
  int transportContext = 1;
  // A real listen comm: CreateContext marks it as GIN.
  IbMsegListenComm listenToken{};
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

TEST_F(NetIbMsegGinMicrotest, CreateContext_SingleRankBuildsContextsAndClampsBatchSize) {
  struct BatchCase {
    int64_t requested;
    int expected;
  };
  for (BatchCase batch :
       {BatchCase{1, NCCL_RMA_MAX_IB_WRS_PER_OP}, BatchCase{NCCL_RMA_IB_WR_BATCHSIZE + 1, NCCL_RMA_IB_WR_BATCHSIZE},
        BatchCase{1000, NCCL_RMA_IB_WR_BATCHSIZE}}) {
    SCOPED_TRACE(batch.requested);
    GinCreateContextScene scene(/*nranks=*/1, /*rank=*/0);
    ScopedHook params(g_loadParam, [batch](const char* name, int64_t fallback) {
      return std::strcmp(name, "RMA_IB_WR_BATCHSIZE") == 0 ? batch.requested : fallback;
    });
    ncclRmaConfig_t config{};
    config.nContexts = 2;
    config.trafficClass = 7;
    IbMsegRmaIbProxyCtx* contexts = nullptr;

    ASSERT_EQ(IbMsegRmaIbProxyCreateContext(&scene.coll, &config, reinterpret_cast<void**>(&contexts)), ncclSuccess);
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

    ASSERT_EQ(IbMsegRmaIbProxyDestroyContext(contexts), ncclSuccess);
    EXPECT_EQ(scene.closeSend.calls, 2);
    EXPECT_EQ(scene.closeRecv.calls, 2);
    // AICOMRCCL-3094: DestroyContext leaves the top-level array to its caller.
    g_hostAllocs.Free(contexts);
  }
}

TEST_F(NetIbMsegGinMicrotest, CreateContext_PeerArrayConnectsForwardAndAcceptsBackward) {
  // Six ranks, not four: with peers two apart, rank + 2 and rank - 2 coincide mod 4, which would hide a swapped peer.
  // {1, 3, 5} is the sorted peerArray ncclRmaProxyCreateContext builds for rail peer 1 at rankStride 2.
  GinCreateContextScene scene(/*nranks=*/6, /*rank=*/1);
  int peerArray[] = {1, 3, 5};
  ncclRmaConfig_t config{};
  config.nContexts = 1;
  config.peerArray = peerArray;
  config.peerArrayCount = 3;
  IbMsegRmaIbProxyCtx* contexts = nullptr;

  ASSERT_EQ(IbMsegRmaIbProxyCreateContext(&scene.coll, &config, reinterpret_cast<void**>(&contexts)), ncclSuccess);
  ASSERT_NE(contexts, nullptr);
  void** sends = contexts[0].fullSendComm;
  void** recvs = contexts[0].fullRecvComm;
  EXPECT_EQ(scene.connectPeers, (std::vector<int>{1, 3, 5}));
  EXPECT_EQ(scene.connectSlots, (std::vector<void**>{&sends[1], &sends[3], &sends[5]}));
  EXPECT_EQ(scene.acceptSlots, (std::vector<void**>{&recvs[1], &recvs[5], &recvs[3]}));
  for (int r : {0, 2, 4}) {
    EXPECT_EQ(sends[r], nullptr) << r;
    EXPECT_EQ(recvs[r], nullptr) << r;
  }
  // Every one of the nranks rounds runs a ring barrier of nranks - 1 sends, wanted peer or not, so ranks with
  // different peer counts stay in lockstep.
  EXPECT_EQ(scene.isend.calls, 6 * 5);

  ASSERT_EQ(IbMsegRmaIbProxyDestroyContext(contexts), ncclSuccess);
  EXPECT_EQ(scene.closeSend.calls, 6);
  EXPECT_EQ(scene.closeRecv.calls, 6);
  g_hostAllocs.Free(contexts);
}

TEST_F(NetIbMsegGinMicrotest, CreateContext_FullPeerArrayConnectsEveryRank) {
  GinCreateContextScene scene(/*nranks=*/4, /*rank=*/1);
  int peerArray[] = {0, 1, 2, 3};
  ncclRmaConfig_t config{};
  config.nContexts = 1;
  config.peerArray = peerArray;
  config.peerArrayCount = 4;
  IbMsegRmaIbProxyCtx* contexts = nullptr;

  ASSERT_EQ(IbMsegRmaIbProxyCreateContext(&scene.coll, &config, reinterpret_cast<void**>(&contexts)), ncclSuccess);
  EXPECT_EQ(scene.connectPeers, (std::vector<int>{1, 2, 3, 0}));
  ASSERT_EQ(IbMsegRmaIbProxyDestroyContext(contexts), ncclSuccess);
  g_hostAllocs.Free(contexts);
}

TEST_F(NetIbMsegGinMicrotest, CreateContext_EmptyPeerArrayConnectsOnlySelf) {
  // peerArrayCount 0 means this rank needs no RMA connectivity; only the self QP iFlush relies on is built, but the
  // per-round barriers still run so the other ranks are not left waiting.
  GinCreateContextScene scene(/*nranks=*/4, /*rank=*/1);
  ncclRmaConfig_t config{};
  config.nContexts = 1;
  config.peerArray = nullptr;
  config.peerArrayCount = 0;
  IbMsegRmaIbProxyCtx* contexts = nullptr;

  ASSERT_EQ(IbMsegRmaIbProxyCreateContext(&scene.coll, &config, reinterpret_cast<void**>(&contexts)), ncclSuccess);
  EXPECT_EQ(scene.connectPeers, (std::vector<int>{1}));
  EXPECT_EQ(scene.acceptSlots, (std::vector<void**>{&contexts[0].fullRecvComm[1]}));
  EXPECT_EQ(scene.isend.calls, 4 * 3);
  ASSERT_EQ(IbMsegRmaIbProxyDestroyContext(contexts), ncclSuccess);
  g_hostAllocs.Free(contexts);
}

TEST_F(NetIbMsegGinMicrotest, CreateContext_PeerConnectionsTakeGinIbTcOverIbTc) {
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
    int peerArray[] = {0, 1};
    ncclRmaConfig_t config{};
    config.nContexts = 1;
    config.peerArray = peerArray;
    config.peerArrayCount = 2;
    IbMsegRmaIbProxyCtx* contexts = nullptr;

    ASSERT_EQ(IbMsegRmaIbProxyCreateContext(&scene.coll, &config, reinterpret_cast<void**>(&contexts)), ncclSuccess);
    EXPECT_EQ(scene.connectTcs, (std::vector<int>{tcCase.expected, tcCase.expected}));
    ASSERT_EQ(IbMsegRmaIbProxyDestroyContext(contexts), ncclSuccess);
    g_hostAllocs.Free(contexts);
  }
}

TEST_F(NetIbMsegGinMicrotest, CreateContext_PeerArrayWithoutSelfStillConnectsSelf) {
  // Three ranks with a peer set that neither divides nranks nor names this rank: the self QP (used by iFlush) is
  // always built, and only the listed peer is added to it.
  GinCreateContextScene scene(/*nranks=*/3, /*rank=*/1);
  int peerArray[] = {2};
  ncclRmaConfig_t config{};
  config.nContexts = 1;
  config.peerArray = peerArray;
  config.peerArrayCount = 1;
  IbMsegRmaIbProxyCtx* contexts = nullptr;

  ASSERT_EQ(IbMsegRmaIbProxyCreateContext(&scene.coll, &config, reinterpret_cast<void**>(&contexts)), ncclSuccess);
  ASSERT_NE(contexts, nullptr);
  void** sends = contexts[0].fullSendComm;
  void** recvs = contexts[0].fullRecvComm;
  EXPECT_EQ(scene.connectPeers, (std::vector<int>{1, 2}));
  EXPECT_EQ(scene.connectSlots, (std::vector<void**>{&sends[1], &sends[2]}));
  EXPECT_EQ(scene.acceptSlots, (std::vector<void**>{&recvs[1], &recvs[2]}));
  EXPECT_EQ(sends[0], nullptr);
  EXPECT_EQ(recvs[0], nullptr);
  EXPECT_EQ(scene.isend.calls, 3 * 2);

  ASSERT_EQ(IbMsegRmaIbProxyDestroyContext(contexts), ncclSuccess);
  g_hostAllocs.Free(contexts);
}

// AICOMRCCL-3094: partial context creation must release child arrays allocated
// before a later child allocation fails.
TEST_F(NetIbMsegGinMicrotest, DISABLED_CreateContext_PartialAllocationFailureReleasesEarlierChildren) {
  IbMsegGinIbCollComm coll{};
  coll.rank = 0;
  coll.nranks = 1;
  coll.allGather = [](IbMsegGinIbCollComm*, void* src, void* dst, size_t size) {
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
  void* context = nullptr;

  EXPECT_EQ(IbMsegRmaIbProxyCreateContext(&coll, &config, &context), ncclSystemError);
  EXPECT_EQ(context, nullptr);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

// AICOMRCCL-3094: failure after peers are established must close both
// directions and free all context allocations.
TEST_F(NetIbMsegGinMicrotest, DISABLED_CreateContext_BarrierFailureClosesEstablishedPeersAndFreesContext) {
  IbMsegGinIbCollComm coll{};
  coll.rank = 0;
  coll.nranks = 1;
  coll.allGather = [](IbMsegGinIbCollComm*, void* src, void* dst, size_t size) {
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
  void* context = nullptr;

  EXPECT_EQ(IbMsegRmaIbProxyCreateContext(&coll, &config, &context), ncclSystemError);
  EXPECT_EQ(context, nullptr);
  EXPECT_EQ(closeSend.calls, 1);
  EXPECT_EQ(closeRecv.calls, 1);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

TEST_F(NetIbMsegGinMicrotest, RegMrSym_GathersBaseAddressesAndRkeysThenDeregisters) {
  IbMsegGinIbCollComm coll{};
  coll.rank = 0;
  coll.nranks = 2;
  int recvToken = 1;
  coll.recvComm = &recvToken;
  ibv_mr mr{};
  mr.rkey = 0x44;
  IbMsegMrHandle ibHandle{};
  ibHandle.mrs[0] = &mr;
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [&](void* comm, void*, size_t, int, uint64_t, int, uint64_t,
                                               void** handle) {
    EXPECT_EQ(comm, &recvToken);
    *handle = &ibHandle;
    return ncclSuccess;
  });
  // The first gather is each rank's base VA, the second its rkey.
  ScopedHook gather(g_ginCollAllGather, [](IbMsegGinIbCollComm*, void* src, void* dst, size_t size) {
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
  IbMsegRmaIbProxyMrHandle* handle = nullptr;

  ASSERT_EQ(IbMsegRmaIbProxyRegMrSym(&coll, &data, sizeof(data), NCCL_PTR_HOST, 0,
                                  reinterpret_cast<void**>(&handle)),
            ncclSuccess);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(gather.calls, 2);
  EXPECT_EQ(handle->base_vas[0], reinterpret_cast<uintptr_t>(&data));
  EXPECT_EQ(handle->base_vas[1], 0x2000u);
  EXPECT_EQ(handle->rkeys[0], 0x44u);
  EXPECT_EQ(handle->rkeys[1], 0x55u);
  ASSERT_EQ(IbMsegRmaIbProxyDeregMrSym(&coll, handle), ncclSuccess);
  EXPECT_EQ(deregMr.calls, 1);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

TEST_F(NetIbMsegGinMicrotest, RegMrSym_FirstGatherFailureDeregistersAndFreesAllWrappers) {
  IbMsegGinIbCollComm coll{};
  coll.nranks = 2;
  int recvToken = 1;
  coll.recvComm = &recvToken;
  ibv_mr mr{};
  IbMsegMrHandle ibHandle{};
  ibHandle.mrs[0] = &mr;
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [&](void*, void*, size_t, int, uint64_t, int, uint64_t,
                                               void** handle) {
    *handle = &ibHandle;
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, [](IbMsegGinIbCollComm*, void*, void*, size_t) { return ncclSystemError; });
  coll.allGather = GinCollAllGather;
  ScopedHook deregMr(g_ginNetDeregMr, [&](void*, void* handle) {
    EXPECT_EQ(handle, &ibHandle);
    return ncclSuccess;
  });
  int data = 0;
  void* handle = nullptr;

  EXPECT_EQ(IbMsegRmaIbProxyRegMrSym(&coll, &data, sizeof(data), NCCL_PTR_HOST, 0, &handle), ncclSystemError);
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(deregMr.calls, 1);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

#if NCCL_CUMEM_DMABUF_EXPORT_GATE
// One rank with a two-segment device VMM window of kSegBytes per segment at kSegBase.
struct GinTwoSegmentScene {
  static constexpr uintptr_t kSegBase = 0x40000000;
  static constexpr size_t kSegBytes = 0x200000;
  IbMsegGinIbCollComm coll{};
  IbMsegRecvComm recv{};
  rcclIbSegSymRecord records[2] = {};

  GinTwoSegmentScene() {
    coll.rank = 0;
    coll.nranks = 1;
    coll.recvComm = &recv;
    coll.regRecords = records;
    coll.allGather = GinCollAllGather;
    recv.base.vProps.ndevs = 1;
  }
};

std::function<hipError_t(hipDeviceptr_t*, size_t*, hipDeviceptr_t)> TwoSegmentRanges() {
  return [](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
    uintptr_t segBase = GinTwoSegmentScene::kSegBase;
    if (reinterpret_cast<uintptr_t>(ptr) >= segBase + GinTwoSegmentScene::kSegBytes) {
      segBase += GinTwoSegmentScene::kSegBytes;
    }
    *base = reinterpret_cast<hipDeviceptr_t>(segBase);
    *size = GinTwoSegmentScene::kSegBytes;
    return hipSuccess;
  };
}

// Each segment is kSegBytes long and aligned to it.
std::function<hipError_t(hipDeviceptr_t*, size_t*, hipDeviceptr_t)> AlignedSegmentRanges() {
  return [](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
    *base = reinterpret_cast<hipDeviceptr_t>(reinterpret_cast<uintptr_t>(ptr) & ~(GinTwoSegmentScene::kSegBytes - 1));
    *size = GinTwoSegmentScene::kSegBytes;
    return hipSuccess;
  };
}

// Every export hands back a fresh fd the walk owns.
std::function<hipError_t(void*, hipDeviceptr_t, size_t, hipMemRangeHandleType, unsigned long long)> ExportsDevNull() {
  return [](void* handle, hipDeviceptr_t, size_t, hipMemRangeHandleType, unsigned long long) {
    *static_cast<int*>(handle) = open("/dev/null", O_RDONLY);
    return hipSuccess;
  };
}

// A single-rank allGather.
ncclResult_t CopyGather(IbMsegGinIbCollComm*, void* src, void* dst, size_t size) {
  std::memcpy(dst, src, size);
  return ncclSuccess;
}

ncclResult_t RegSegments(GinTwoSegmentScene& scene, size_t bytes, void** handle) {
  return rcclRmaIbProxyRegMrSymDmaBuf(&scene.coll, reinterpret_cast<void*>(GinTwoSegmentScene::kSegBase), bytes,
                                      NCCL_PTR_CUDA, 0, -1, 0, handle);
}

// Segment locations: retain hands back the segment pointer as the handle, so isHost sees which segment it asks about.
struct GinSegLocations {
  explicit GinSegLocations(std::function<bool(uintptr_t)> isHost)
      : props(g_hipMemGetAllocationPropertiesFromHandle,
              [isHost](hipMemAllocationProp* prop, hipMemGenericAllocationHandle_t handle) {
                *prop = hipMemAllocationProp{};
                prop->location.type = hipMemLocationTypeDevice;
                [[maybe_unused]] const bool host = isHost(reinterpret_cast<uintptr_t>(handle));
#if NCCL_CUMEM_HOST_GATE
                if (host) prop->location.type = hipMemLocationTypeHost;
#endif
                return hipSuccess;
              }) {}
  ScopedHook<hipError_t(hipMemGenericAllocationHandle_t*, void*)> retain{
      g_hipMemRetainAllocationHandle, [](hipMemGenericAllocationHandle_t* handle, void* ptr) {
        *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(ptr);
        return hipSuccess;
      }};
  ScopedHook<hipError_t(hipMemAllocationProp*, hipMemGenericAllocationHandle_t)> props;
  ScopedHook<hipError_t(hipMemGenericAllocationHandle_t)> release{
      g_hipMemRelease, [](hipMemGenericAllocationHandle_t) { return hipSuccess; }};
};

// Branch: nSeg > 1. A published owner fd and offset reach the second segment's DMA-BUF MR; the first is exported.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_TwoSegments_UsesPublishedOwnerFd) {
  constexpr uint64_t kOwnerOffset = 0x3000;
  GinTwoSegmentScene scene;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, TwoSegmentRanges());
  ScopedHook exportFd(g_hipMemGetHandleForAddressRange, ExportsDevNull());
  int ownerFd = open("/dev/null", O_RDONLY);
  ASSERT_GE(ownerFd, 0);
  ScopedHook lookup(g_hostSegmentFdLookup, [ownerFd](uintptr_t base, size_t, uint64_t* offset) {
    if (base != GinTwoSegmentScene::kSegBase + GinTwoSegmentScene::kSegBytes) return -1;
    *offset = kOwnerOffset;
    return ownerFd;
  });
  struct Reg {
    uintptr_t data;
    size_t size;
    uint64_t offset;
    bool fdValid;
  };
  std::vector<Reg> regs;
  std::array<ibv_mr, 2> mrs{};
  std::array<IbMsegMrHandle, 2> ibHandles{};
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [&](void*, void* data, size_t size, int, uint64_t offset, int fd,
                                               uint64_t, void** handle) {
    size_t i = regs.size();
    regs.push_back({reinterpret_cast<uintptr_t>(data), size, offset, fd >= 0});
    ibHandles[i].mrs[0] = &mrs[i];
    *handle = &ibHandles[i];
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, CopyGather);
  ScopedHook deregMr(g_ginNetDeregMr, [](void*, void*) { return ncclSuccess; });
  GinSegLocations locations([](uintptr_t) { return false; });
  void* handle = nullptr;

  ASSERT_EQ(RegSegments(scene, 2 * GinTwoSegmentScene::kSegBytes, &handle), ncclSuccess);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(static_cast<rcclRmaIbProxySegMrHandle*>(handle)->nSegments, 2);
  EXPECT_EQ(static_cast<rcclRmaIbProxySegMrHandle*>(handle)->flushSeg, 0);
  EXPECT_EQ(locations.retain.calls, 1) << "the walk stops asking once it has a device segment";
  EXPECT_EQ(exportFd.calls, 1) << "only the first segment is exported";
  ASSERT_EQ(regs.size(), 2u);
  EXPECT_EQ(regs[0].offset, 0u);
  EXPECT_EQ(regs[1].data, GinTwoSegmentScene::kSegBase + GinTwoSegmentScene::kSegBytes);
  EXPECT_EQ(regs[1].size, GinTwoSegmentScene::kSegBytes);
  EXPECT_EQ(regs[1].offset, kOwnerOffset);
  EXPECT_TRUE(regs[1].fdValid);
  EXPECT_NE(fcntl(ownerFd, F_GETFD), -1) << "the published owner fd is borrowed, not closed";
  ASSERT_EQ(rcclRmaIbProxyDeregMrSym(&scene.coll, handle), ncclSuccess);
  EXPECT_EQ(deregMr.calls, 2);
  close(ownerFd);
}

// The builder has no segment cap: keys come from the window this transfer touches, bounded by maxWr.
TEST_F(NetIbMsegGinMicrotest, BuildSegmentedWrs_ResolvesKeysOverTheTouchedWindowOnly) {
  constexpr int kSeg = 24;
  constexpr size_t kBytes = 0x1000;
  std::vector<size_t> segOff(kSeg + 1);
  std::vector<uintptr_t> bases(kSeg);
  std::vector<uint64_t> rkeys((size_t)kSeg * NCCL_IB_MAX_DEVS_PER_NIC);
  std::vector<ibv_mr> mrs(kSeg);
  std::vector<IbMsegMrHandle> ibHandles(kSeg);
  std::vector<void*> mrHandle(kSeg);
  for (int s = 0; s <= kSeg; s++) segOff[s] = s * kBytes;
  for (int s = 0; s < kSeg; s++) {
    bases[s] = 0x10000000 + s * kBytes;
    rkeys[(size_t)s * NCCL_IB_MAX_DEVS_PER_NIC] = 0x500 + s;
    mrs[s].lkey = 0x900 + s;
    ibHandles[s].mrs[0] = &mrs[s];
    mrHandle[s] = &ibHandles[s];
  }
  rcclRmaIbProxySegMrHandle h{};
  h.nSegments = kSeg;
  h.segOff = segOff.data();
  h.mrHandle = mrHandle.data();
  h.sym.vas = bases.data();
  h.sym.keys = rkeys.data();
  IbMsegQp qp{};
  ibv_send_wr wr[RCCL_RMA_MAX_DATA_WRS] = {};
  ibv_sge sge[RCCL_RMA_MAX_DATA_WRS] = {};
  int nWr = 0;
  const uint64_t off = 20 * kBytes + kBytes / 2;

  ASSERT_EQ(rcclRmaBuildSegmentedWrs(wr, sge, RCCL_RMA_MAX_DATA_WRS, &nWr, IBV_WR_RDMA_WRITE, &qp, &h, 0, off, &h, 0,
                                     off, kBytes),
            ncclSuccess);
  ASSERT_EQ(nWr, 2);
  EXPECT_EQ(wr[0].wr.rdma.rkey, 0x514u);
  EXPECT_EQ(wr[1].wr.rdma.rkey, 0x515u);
  EXPECT_EQ(sge[0].lkey, 0x914u);
  EXPECT_EQ(sge[1].lkey, 0x915u);
  EXPECT_EQ(wr[1].wr.rdma.remote_addr, bases[0] + 21 * kBytes);
  EXPECT_EQ(sge[1].addr, bases[0] + 21 * kBytes);

  EXPECT_EQ(rcclRmaBuildSegmentedWrs(wr, sge, 1, &nWr, IBV_WR_RDMA_WRITE, &qp, &h, 0, off, &h, 0, off, kBytes),
            ncclInternalError)
      << "two touched segments do not fit one WR";
}

// Tables sized from the walked count: two ranks, two segments, so each rank's offsets sit nSeg + 1 apart.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_TwoRanksTwoSegments_SizesTablesFromTheCount) {
  GinTwoSegmentScene scene;
  scene.coll.nranks = 2;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, TwoSegmentRanges());
  ScopedHook exportFd(g_hipMemGetHandleForAddressRange, ExportsDevNull());
  std::array<ibv_mr, 2> mrs{};
  std::array<IbMsegMrHandle, 2> ibHandles{};
  int nReg = 0;
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [&](void*, void*, size_t, int, uint64_t, int, uint64_t, void** handle) {
    mrs[nReg].rkey = 0x100 + nReg;
    ibHandles[nReg].mrs[0] = &mrs[nReg];
    *handle = &ibHandles[nReg++];
    return ncclSuccess;
  });
  // Both ranks contribute the same bytes.
  std::vector<size_t> gatherSizes;
  ScopedHook gather(g_ginCollAllGather, [&](IbMsegGinIbCollComm*, void* src, void* dst, size_t size) {
    gatherSizes.push_back(size);
    std::memcpy(dst, src, size);
    std::memcpy(static_cast<char*>(dst) + size, src, size);
    return ncclSuccess;
  });
  ScopedHook deregMr(g_ginNetDeregMr, [](void*, void*) { return ncclSuccess; });
  GinSegLocations locations([](uintptr_t) { return false; });
  void* handle = nullptr;

  ASSERT_EQ(RegSegments(scene, 2 * GinTwoSegmentScene::kSegBytes, &handle), ncclSuccess);
  auto* h = static_cast<rcclRmaIbProxySegMrHandle*>(handle);
  ASSERT_EQ(h->nSegments, 2);
  ASSERT_GE(gatherSizes.size(), 3u);
  EXPECT_EQ(gatherSizes[2], 3 * sizeof(size_t)) << "the offset gather carries nSeg + 1 entries";
  const size_t want[] = {0, GinTwoSegmentScene::kSegBytes, 2 * GinTwoSegmentScene::kSegBytes};
  for (int r = 0; r < 2; r++) {
    for (int i = 0; i < 3; i++) EXPECT_EQ(h->sym.rankOff[r * 3 + i], want[i]) << "rank " << r << " entry " << i;
  }
  EXPECT_EQ(rcclRmaRankOff(h, 1), h->sym.rankOff + 3);
  EXPECT_EQ(rcclRmaRemoteRkey(h, 1, 1, 0), 0x101u);
  ASSERT_EQ(rcclRmaIbProxyDeregMrSym(&scene.coll, handle), ncclSuccess);
  EXPECT_EQ(deregMr.calls, 2);
}

// Branch: nSeg > 1 with NCCL_DMABUF_ENABLE=0 fails through the status exchange before any MR is registered.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_TwoSegmentsWithDmaBufOff_RejectsCollectively) {
  GinTwoSegmentScene scene;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, TwoSegmentRanges());
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) -> int64_t {
    return std::string(name) == "DMABUF_ENABLE" ? 0 : fallback;
  });
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [](void*, void*, size_t, int, uint64_t, int, uint64_t, void**) {
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, CopyGather);
  void* handle = nullptr;

  EXPECT_EQ(RegSegments(scene, 2 * GinTwoSegmentScene::kSegBytes, &handle), ncclInvalidUsage);
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(regMrDmaBuf.calls, 0);
  EXPECT_EQ(gather.calls, 1) << "the rejection travels in the first status exchange";
}

// A borrowed owner fd marks a host segment without asking the allocation; the first device segment is the flush one.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_OwnerFdSegmentFirst_FlushesTheExportedSegment) {
  GinTwoSegmentScene scene;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, TwoSegmentRanges());
  ScopedHook exportFd(g_hipMemGetHandleForAddressRange, ExportsDevNull());
  int ownerFd = open("/dev/null", O_RDONLY);
  ASSERT_GE(ownerFd, 0);
  ScopedHook lookup(g_hostSegmentFdLookup, [ownerFd](uintptr_t base, size_t, uint64_t* offset) {
    if (base != GinTwoSegmentScene::kSegBase) return -1;
    *offset = 0;
    return ownerFd;
  });
  std::array<ibv_mr, 2> mrs{};
  std::array<IbMsegMrHandle, 2> ibHandles{};
  int nReg = 0;
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [&](void*, void*, size_t, int, uint64_t, int, uint64_t, void** handle) {
    ibHandles[nReg].mrs[0] = &mrs[nReg];
    *handle = &ibHandles[nReg++];
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, CopyGather);
  ScopedHook deregMr(g_ginNetDeregMr, [](void*, void*) { return ncclSuccess; });
  std::vector<uintptr_t> asked;
  GinSegLocations locations([&](uintptr_t ptr) {
    asked.push_back(ptr);
    return false;
  });
  void* handle = nullptr;

  ASSERT_EQ(RegSegments(scene, 2 * GinTwoSegmentScene::kSegBytes, &handle), ncclSuccess);
  EXPECT_EQ(static_cast<rcclRmaIbProxySegMrHandle*>(handle)->flushSeg, 1);
  ASSERT_EQ(asked.size(), 1u) << "only the exported segment's location is asked";
  EXPECT_EQ(asked[0], GinTwoSegmentScene::kSegBase + GinTwoSegmentScene::kSegBytes);
  ASSERT_EQ(rcclRmaIbProxyDeregMrSym(&scene.coll, handle), ncclSuccess);
  close(ownerFd);
}

#if NCCL_CUMEM_HOST_GATE
// Every segment in host memory: no flush segment, so a flush of this window posts nothing.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_AllHostSegments_HasNoFlushSegment) {
  GinTwoSegmentScene scene;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, TwoSegmentRanges());
  ScopedHook exportFd(g_hipMemGetHandleForAddressRange, ExportsDevNull());
  std::array<ibv_mr, 2> mrs{};
  std::array<IbMsegMrHandle, 2> ibHandles{};
  int nReg = 0;
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [&](void*, void*, size_t, int, uint64_t, int, uint64_t, void** handle) {
    ibHandles[nReg].mrs[0] = &mrs[nReg];
    *handle = &ibHandles[nReg++];
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, CopyGather);
  ScopedHook deregMr(g_ginNetDeregMr, [](void*, void*) { return ncclSuccess; });
  GinSegLocations locations([](uintptr_t) { return true; });
  void* handle = nullptr;

  ASSERT_EQ(RegSegments(scene, 2 * GinTwoSegmentScene::kSegBytes, &handle), ncclSuccess);
  EXPECT_EQ(static_cast<rcclRmaIbProxySegMrHandle*>(handle)->flushSeg, -1);
  EXPECT_EQ(locations.retain.calls, 2);
  EXPECT_EQ(locations.release.calls, 2) << "every retained handle is released";
  ASSERT_EQ(rcclRmaIbProxyDeregMrSym(&scene.coll, handle), ncclSuccess);
}
#endif

// A failed location query fails the walk: its fd is closed and no MR is registered.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_LocationQueryFails_FailsBeforeRegistering) {
  GinTwoSegmentScene scene;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, TwoSegmentRanges());
  int exported = -1;
  ScopedHook exportFd(g_hipMemGetHandleForAddressRange,
                      [&](void* handle, hipDeviceptr_t, size_t, hipMemRangeHandleType, unsigned long long) {
                        exported = open("/dev/null", O_RDONLY);
                        *static_cast<int*>(handle) = exported;
                        return hipSuccess;
                      });
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [](void*, void*, size_t, int, uint64_t, int, uint64_t, void**) {
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, CopyGather);
  ScopedHook retain(g_hipMemRetainAllocationHandle,
                    [](hipMemGenericAllocationHandle_t*, void*) { return hipErrorInvalidValue; });
  void* handle = nullptr;

  EXPECT_NE(RegSegments(scene, 2 * GinTwoSegmentScene::kSegBytes, &handle), ncclSuccess);
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(regMrDmaBuf.calls, 0);
  ASSERT_GE(exported, 0);
  EXPECT_EQ(fcntl(exported, F_GETFD), -1) << "the exported fd is closed";
}

// Past the old 16-segment limit: 17 segments register one MR each and agree on the count.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_SeventeenSegments_RegistersEachSegment) {
  constexpr int kSeg = 17;
  GinTwoSegmentScene scene;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, AlignedSegmentRanges());
  ScopedHook exportFd(g_hipMemGetHandleForAddressRange, ExportsDevNull());
  std::vector<ibv_mr> mrs(kSeg);
  std::vector<IbMsegMrHandle> ibHandles(kSeg);
  std::vector<uintptr_t> regData;
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [&](void*, void* data, size_t, int, uint64_t, int, uint64_t, void** handle) {
    size_t i = regData.size();
    regData.push_back(reinterpret_cast<uintptr_t>(data));
    mrs[i].rkey = 0x100 + (uint32_t)i;
    ibHandles[i].mrs[0] = &mrs[i];
    *handle = &ibHandles[i];
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, CopyGather);
  ScopedHook deregMr(g_ginNetDeregMr, [](void*, void*) { return ncclSuccess; });
  GinSegLocations locations([](uintptr_t) { return false; });
  void* handle = nullptr;

  ASSERT_EQ(RegSegments(scene, kSeg * GinTwoSegmentScene::kSegBytes, &handle), ncclSuccess);
  auto* h = static_cast<rcclRmaIbProxySegMrHandle*>(handle);
  ASSERT_EQ(h->nSegments, kSeg);
  EXPECT_EQ(scene.records[0].nSegments, kSeg);
  ASSERT_EQ(regData.size(), (size_t)kSeg);
  for (int s = 0; s < kSeg; s++) {
    EXPECT_EQ(regData[s], GinTwoSegmentScene::kSegBase + s * GinTwoSegmentScene::kSegBytes) << s;
    EXPECT_EQ(rcclRmaRemoteRkey(h, 0, s, 0), 0x100u + s) << s;
  }
  EXPECT_EQ(rcclRmaRankOff(h, 0)[kSeg], kSeg * GinTwoSegmentScene::kSegBytes);
  ASSERT_EQ(rcclRmaIbProxyDeregMrSym(&scene.coll, handle), ncclSuccess);
  EXPECT_EQ(deregMr.calls, kSeg);
}

// Branch: ranks x segments past the table budget declines in the first exchange, before any MR or table.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_TablePastTheBudget_DeclinesBeforeRegistering) {
  constexpr int kRanks = RCCL_RMA_SEG_TABLE_MAX_ENTRIES / RCCL_IB_REG_SEG_DEFAULT + 1;
  GinTwoSegmentScene scene;
  std::vector<rcclIbSegSymRecord> records(kRanks);
  scene.coll.nranks = kRanks;
  scene.coll.regRecords = records.data();
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, AlignedSegmentRanges());
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [](void*, void*, size_t, int, uint64_t, int, uint64_t, void**) {
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, [](IbMsegGinIbCollComm*, void* src, void* dst, size_t size) {
    for (int r = 0; r < kRanks; r++) std::memcpy(static_cast<char*>(dst) + r * size, src, size);
    return ncclSuccess;
  });
  void* handle = nullptr;

  EXPECT_EQ(RegSegments(scene, RCCL_IB_REG_SEG_DEFAULT * GinTwoSegmentScene::kSegBytes, &handle), ncclInvalidUsage);
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(regMrDmaBuf.calls, 0);
  EXPECT_EQ(gather.calls, 1);
  EXPECT_EQ(records[kRanks - 1].nSegments, RCCL_IB_REG_SEG_DEFAULT);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

// Branch: a walk past the supported count declines in the first exchange, before any MR or per-segment table.
TEST_F(NetIbMsegGinMicrotest, RegMrSymDmaBuf_PastTheSupportedCount_DeclinesBeforeSizingTables) {
  GinTwoSegmentScene scene;
  ScopedHook<int()> cuMemEnable{g_cuMemEnable, [] { return 1; }};
  ScopedHook ranges(g_hipMemGetAddressRange, AlignedSegmentRanges());
  ScopedHook regMrDmaBuf(g_ginRegMrDmaBuf, [](void*, void*, size_t, int, uint64_t, int, uint64_t, void**) {
    return ncclSuccess;
  });
  ScopedHook gather(g_ginCollAllGather, CopyGather);
  void* handle = nullptr;

  EXPECT_EQ(RegSegments(scene, (RCCL_IB_REG_SEG_DEFAULT + 1) * GinTwoSegmentScene::kSegBytes, &handle), ncclInvalidUsage);
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(regMrDmaBuf.calls, 0);
  EXPECT_EQ(gather.calls, 1);
  EXPECT_EQ(scene.records[0].nSegments, RCCL_IB_REG_SEG_DEFAULT + 1) << "every rank sees the walked count";
  EXPECT_TRUE(g_hostAllocs.live.empty());
}
#endif

struct GinDataScene {
  IbMsegRmaIbProxyCtx context{};
  std::array<void*, 2> sendComms{};
  std::array<void*, 2> recvComms{};
  IbMsegSendComm send{};
  IbMsegRecvComm recv{};
  IbVerbsScene verbs{"test-hca"};
  ibv_qp* const dataQp = &verbs.qps[0];
  ibv_qp* const flushQp = &verbs.qps[1];
  ibv_mr srcMr{};
  ibv_mr localMr{};
  ibv_mr scratchMr{};
  IbMsegMrHandle srcIbHandle{};
  IbMsegMrHandle localIbHandle{};
  std::array<uintptr_t, 2> srcBases{{0x1000, 0x2000}};
  std::array<uintptr_t, 2> dstBases{{0x3000, 0x4000}};
  std::array<uint32_t, 2> srcRkeys{{0x10, 0x11}};
  std::array<uint32_t, 2> dstRkeys{{0x20, 0x21}};
  IbMsegRmaIbProxyMrHandle srcHandle{};
  IbMsegRmaIbProxyMrHandle dstHandle{};

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
  IbMsegRequest* PostPut(uint32_t optFlags = 0) {
    void* request = nullptr;
    EXPECT_EQ(IbMsegRmaIbProxyIPut(&context, 0, 0, &srcHandle, 8, 0, &dstHandle, 1, optFlags, &request), ncclSuccess);
    return static_cast<IbMsegRequest*>(request);
  }

  void QueueCompletion(IbMsegRequest* req, ibv_wc_status status = IBV_WC_SUCCESS) {
    ibv_wc completion{};
    completion.status = status;
    completion.wr_id = req - req->base->reqs;
    verbs.wcs.push_back(completion);
  }
};

TEST_F(NetIbMsegGinMicrotest, ReserveWrs_QpSwitchPostsPendingChainBeforeChangingQp) {
  GinDataScene scene;
  ibv_qp* otherQp = &scene.verbs.qps[2];
  ibv_send_wr* wr = nullptr;
  ibv_sge* sge = nullptr;
  ASSERT_EQ(IbMsegRmaIbProxyReserveWrs(&scene.context, scene.dataQp, 1, &wr, &sge), ncclSuccess);
  wr->opcode = IBV_WR_RDMA_WRITE;
  wr->sg_list = sge;
  wr->num_sge = 1;
  ASSERT_EQ(IbMsegRmaIbProxyCommitWrs(&scene.context, 1, true), ncclSuccess);
  ASSERT_TRUE(scene.verbs.posted.empty());

  ASSERT_EQ(IbMsegRmaIbProxyReserveWrs(&scene.context, otherQp, 1, &wr, &sge), ncclSuccess);
  ASSERT_EQ(scene.verbs.posted.size(), 1u);
  EXPECT_EQ(scene.context.batch.qp, otherQp);
  EXPECT_EQ(scene.context.batch.nWrs, 0);
}

TEST_F(NetIbMsegGinMicrotest, CommitWrs_AggregatePostsWhenNextMaximumOperationWouldNotFit) {
  GinDataScene scene;
  scene.context.wrBatchsize = 2;
  ibv_send_wr* wr = nullptr;
  ibv_sge* sge = nullptr;
  ASSERT_EQ(IbMsegRmaIbProxyReserveWrs(&scene.context, scene.dataQp, 1, &wr, &sge), ncclSuccess);
  wr->opcode = IBV_WR_RDMA_WRITE;
  wr->sg_list = sge;
  wr->num_sge = 1;

  ASSERT_EQ(IbMsegRmaIbProxyCommitWrs(&scene.context, 1, true), ncclSuccess);

  ASSERT_EQ(scene.verbs.posted.size(), 1u);
  EXPECT_EQ(scene.context.batch.nWrs, 0);
  EXPECT_EQ(scene.context.batch.qp, nullptr);
}

TEST_F(NetIbMsegGinMicrotest, CommitWrs_AggregateDefersWhenNextMaximumOperationExactlyFits) {
  GinDataScene scene;
  scene.context.wrBatchsize = 3;
  ibv_send_wr* wr = nullptr;
  ibv_sge* sge = nullptr;
  ASSERT_EQ(IbMsegRmaIbProxyReserveWrs(&scene.context, scene.dataQp, 1, &wr, &sge), ncclSuccess);
  wr->opcode = IBV_WR_RDMA_WRITE;
  wr->sg_list = sge;
  wr->num_sge = 1;

  ASSERT_EQ(IbMsegRmaIbProxyCommitWrs(&scene.context, 1, true), ncclSuccess);

  EXPECT_TRUE(scene.verbs.posted.empty());
  EXPECT_EQ(scene.context.batch.nWrs, 1);
  EXPECT_EQ(scene.context.batch.qp, scene.dataQp);
}

TEST_F(NetIbMsegGinMicrotest, DataApis_UnconnectedPeersRejectWithoutRequestOrPost) {
  GinDataScene scene;
  scene.sendComms[1] = nullptr;
  scene.recvComms[1] = nullptr;
  void* request = nullptr;

  EXPECT_EQ(IbMsegRmaIbProxyIPut(&scene.context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1, 0,
                              &request),
            ncclInvalidUsage);
  EXPECT_EQ(IbMsegRmaIbProxyIGet(&scene.context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1, 0,
                              &request),
            ncclInvalidUsage);
  EXPECT_EQ(IbMsegRmaIbProxyIFlush(&scene.context, 0, &scene.dstHandle, 1, &request), ncclInvalidUsage);
  EXPECT_EQ(request, nullptr);
  EXPECT_TRUE(scene.verbs.posted.empty());
  EXPECT_EQ(scene.send.ginSeq.posted, 0u);
  EXPECT_EQ(scene.recv.ginSeq.posted, 0u);
}

TEST_F(NetIbMsegGinMicrotest, IPut_PostsExactWriteAndPublishesRequest) {
  GinDataScene scene;
  void* request = nullptr;

  ASSERT_EQ(IbMsegRmaIbProxyIPut(&scene.context, 0, 3, &scene.srcHandle, 16, 5, &scene.dstHandle, 1, 0,
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
  auto* req = static_cast<IbMsegRequest*>(request);
  EXPECT_EQ(req->type, NCCL_NET_IB_REQ_GIN_IPUT);
  EXPECT_EQ(req->iput.rank, 1u);
  EXPECT_EQ(req->events[0], 1);
}

TEST_F(NetIbMsegGinMicrotest, IGet_PostsExactReadAndPublishesRequest) {
  GinDataScene scene;
  void* request = nullptr;

  ASSERT_EQ(IbMsegRmaIbProxyIGet(&scene.context, 0, 7, &scene.srcHandle, 32, 9, &scene.dstHandle, 1, 0,
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
  EXPECT_EQ(static_cast<IbMsegRequest*>(request)->type, NCCL_NET_IB_REQ_GIN_IGET);
}

TEST_F(NetIbMsegGinMicrotest, IPutSignal_PutAndAddPostOrderedWriteThenAtomic) {
  GinDataScene scene;
  void* request = nullptr;

  ASSERT_EQ(IbMsegRmaIbProxyIPutSignal(&scene.context, 0, 1, &scene.srcHandle, 8, 2, &scene.dstHandle, 1, 8,
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

TEST_F(NetIbMsegGinMicrotest, IPutSignal_SignalOnlyIncrementPostsOneAtomicWithValueOne) {
  GinDataScene scene;
  void* request = nullptr;

  ASSERT_EQ(IbMsegRmaIbProxyIPutSignal(&scene.context, 0, 0, nullptr, 0, 0, nullptr, 1, 8, &scene.dstHandle, 99,
                                    NCCL_NET_SIGNAL_OP_INC, false, 0, &request),
            ncclSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(scene.verbs.posted.size(), 1u);
  EXPECT_EQ(scene.verbs.posted[0].opcode, IBV_WR_ATOMIC_FETCH_AND_ADD);
  EXPECT_EQ(scene.verbs.posted[0].remoteAddress, 0x4008u);
  EXPECT_EQ(scene.verbs.posted[0].compareAdd, 1u);
  EXPECT_NE(scene.verbs.posted[0].sendFlags & IBV_SEND_SIGNALED, 0);
}

TEST_F(NetIbMsegGinMicrotest, IPutSignal_UnsupportedOperationIsRejectedBeforeRequestOrPost) {
  GinDataScene scene;
  void* request = nullptr;

  EXPECT_EQ(IbMsegRmaIbProxyIPutSignal(&scene.context, 0, 0, nullptr, 0, 0, nullptr, 1, 8, &scene.dstHandle, 1,
                                    99, false, 0, &request),
            ncclInvalidArgument);
  EXPECT_EQ(request, nullptr);
  EXPECT_TRUE(scene.verbs.posted.empty());
  EXPECT_EQ(scene.send.ginSeq.posted, 0u);
}

// AICOMRCCL-3094: 64-bit RDMA atomics require an 8-byte aligned remote
// address, matching the validation already used by the socket RMA backend.
TEST_F(NetIbMsegGinMicrotest, DISABLED_IPutSignal_UnalignedSignalIsRejectedBeforeRequestOrPost) {
  GinDataScene scene;
  void* request = nullptr;

  EXPECT_EQ(IbMsegRmaIbProxyIPutSignal(&scene.context, 0, 0, nullptr, 0, 0, nullptr, 1, 4, &scene.dstHandle, 1,
                                    NCCL_NET_SIGNAL_OP_INC, false, 0, &request),
            ncclInvalidArgument);
  EXPECT_EQ(request, nullptr);
  EXPECT_TRUE(scene.verbs.posted.empty());
  EXPECT_EQ(scene.send.ginSeq.posted, 0u);
}

TEST_F(NetIbMsegGinMicrotest, AggregatedPut_DefersDoorbellUntilTestAndCompletesRequest) {
  GinDataScene scene;
  IbMsegRequest* req = scene.PostPut(ncclRmaOptFlagsAggregateRequests);
  ASSERT_NE(req, nullptr);
  EXPECT_TRUE(scene.verbs.posted.empty());
  scene.QueueCompletion(req);
  int done = 0;

  ASSERT_EQ(IbMsegRmaIbProxyTest(nullptr, req, &done), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(scene.verbs.posted.size(), 1u);
  EXPECT_EQ(req->type, NCCL_NET_IB_REQ_UNUSED);
  EXPECT_EQ(scene.send.ginSeq.completed, req->id);
}

TEST_F(NetIbMsegGinMicrotest, Test_AlreadyCompletedSequenceFreesWithoutPolling) {
  GinDataScene scene;
  IbMsegRequest* req = scene.PostPut();
  ASSERT_NE(req, nullptr);
  scene.send.ginSeq.completed = req->id;
  scene.verbs.pollErr = -EIO;
  int done = 0;

  ASSERT_EQ(IbMsegRmaIbProxyTest(nullptr, req, &done), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(req->type, NCCL_NET_IB_REQ_UNUSED);
}

TEST_F(NetIbMsegGinMicrotest, Test_ErrorCompletionReturnsRemoteErrorWithoutCompleting) {
  GinDataScene scene;
  IbMsegRequest* req = scene.PostPut();
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

  EXPECT_EQ(IbMsegRmaIbProxyTest(nullptr, req, &done), ncclRemoteError);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(scene.send.ginSeq.completed, 0u);
  IbMsegFreeRequest(req);
}

TEST_F(NetIbMsegGinMicrotest, Test_PollFailurePropagatesWithoutCompletingRequest) {
  GinDataScene scene;
  IbMsegRequest* req = scene.PostPut();
  ASSERT_NE(req, nullptr);
  scene.verbs.pollErr = -EIO;
  int done = 0;

  EXPECT_EQ(IbMsegRmaIbProxyTest(nullptr, req, &done), ncclSystemError);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(scene.send.ginSeq.completed, 0u);
  IbMsegFreeRequest(req);
}

TEST_F(NetIbMsegGinMicrotest, IFlush_DrainsAggregatedPutsThenReadsLocalRegistrationOnFlushQp) {
  GinDataScene scene;
  ScopedHook createFlushQp(g_ginCreateFlushQp, [&](IbMsegRecvComm* comm) {
    EXPECT_EQ(comm, &scene.recv);
    return ncclSuccess;
  });
  ASSERT_NE(scene.PostPut(ncclRmaOptFlagsAggregateRequests), nullptr);
  ASSERT_TRUE(scene.verbs.posted.empty());
  void* request = nullptr;

  ASSERT_EQ(IbMsegRmaIbProxyIFlush(&scene.context, 0, &scene.dstHandle, 1, &request), ncclSuccess);
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

TEST_F(NetIbMsegGinMicrotest, IFlush_DisabledFlushReturnsNoRequest) {
  GinDataScene scene;
  scene.recv.flushEnabled = false;
  void* request = reinterpret_cast<void*>(0x1);

  ASSERT_EQ(IbMsegRmaIbProxyIFlush(&scene.context, 0, &scene.dstHandle, 1, &request), ncclSuccess);
  EXPECT_EQ(request, nullptr);
  EXPECT_TRUE(scene.verbs.posted.empty());
  EXPECT_EQ(scene.recv.ginSeq.posted, 0u);
}

TEST_F(NetIbMsegGinMicrotest, IFlush_CompletionAdvancesReceiveSequenceAndFreesRequest) {
  GinDataScene scene;
  ScopedHook createFlushQp(g_ginCreateFlushQp, [](IbMsegRecvComm*) { return ncclSuccess; });
  void* request = nullptr;
  ASSERT_EQ(IbMsegRmaIbProxyIFlush(&scene.context, 0, &scene.dstHandle, 1, &request), ncclSuccess);
  auto* req = static_cast<IbMsegRequest*>(request);
  scene.QueueCompletion(req);
  int done = 0;

  ASSERT_EQ(IbMsegRmaIbProxyTest(nullptr, request, &done), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(scene.recv.ginSeq.completed, req->id);
  EXPECT_EQ(req->type, NCCL_NET_IB_REQ_UNUSED);
}

// Two ranks, n segments of kSegBytes: rank r's segment s at base + r * 0x100000 + s * kSegBytes, lkey lkey0 + s,
// rkey rkey0 + r * n + s in device slot 0.
struct GinSegHandle {
  static constexpr size_t kSegBytes = 0x100;
  std::vector<size_t> segOff;
  std::vector<uintptr_t> bases;
  std::vector<uint64_t> rkeys;
  std::vector<ibv_mr> mrs;
  std::vector<IbMsegMrHandle> ibHandles;
  std::vector<void*> slots;
  rcclRmaIbProxySegMrHandle h{};

  GinSegHandle(int n, uintptr_t base, uint32_t lkey0, uint32_t rkey0)
      : segOff(n + 1), bases(2 * n), rkeys((size_t)2 * n * NCCL_IB_MAX_DEVS_PER_NIC), mrs(n), ibHandles(n), slots(n) {
    for (int s = 0; s <= n; s++) segOff[s] = s * kSegBytes;
    for (int r = 0; r < 2; r++) {
      for (int s = 0; s < n; s++) {
        bases[r * n + s] = base + r * 0x100000 + s * kSegBytes;
        rkeys[(size_t)(r * n + s) * NCCL_IB_MAX_DEVS_PER_NIC] = rkey0 + r * n + s;
      }
    }
    for (int s = 0; s < n; s++) {
      mrs[s].lkey = lkey0 + s;
      ibHandles[s].mrs[0] = &mrs[s];
      slots[s] = &ibHandles[s];
    }
    h.tag = rcclRmaSegMrTag();
    h.nSegments = n;
    h.segOff = segOff.data();
    h.mrHandle = slots.data();
    h.sym.vas = bases.data();
    h.sym.keys = rkeys.data();
  }
};

// The data scene with the per-peer queues that multi-segment contexts allocate.
struct GinRoundScene : GinDataScene {
  std::array<rcclRmaRoundQueue, 2> sendQueues{};
  GinRoundScene() { context.pending.send = sendQueues.data(); }
  int Test(IbMsegRequest* req, ncclResult_t expect = ncclSuccess) {
    int done = 0;
    EXPECT_EQ(IbMsegRmaIbProxyTest(nullptr, req, &done), expect);
    return done;
  }
  int SignaledCount() const {
    int n = 0;
    for (const auto& p : verbs.posted) n += (p.sendFlags & IBV_SEND_SIGNALED) != 0;
    return n;
  }
};

// A partial round stops at maxWr and resolves keys over the segments it reaches.
TEST_F(NetIbMsegGinMicrotest, BuildSegmentedWrs_CoveredStopsAtMaxWrAndReportsBytes) {
  GinSegHandle seg(24, 0x10000000, 0x900, 0x500);
  IbMsegQp qp{};
  ibv_send_wr wr[4] = {};
  ibv_sge sge[4] = {};
  int nWr = 0;
  uint64_t covered = 0;
  const uint64_t off = 18 * GinSegHandle::kSegBytes;
  ASSERT_EQ(rcclRmaBuildSegmentedWrs(wr, sge, 4, &nWr, IBV_WR_RDMA_WRITE, &qp, &seg.h, 0, off, &seg.h, 1, off,
                                     6 * GinSegHandle::kSegBytes, &covered),
            ncclSuccess);
  ASSERT_EQ(nWr, 4);
  EXPECT_EQ(covered, 4 * GinSegHandle::kSegBytes);
  EXPECT_EQ(sge[3].lkey, 0x900u + 21);
  EXPECT_EQ(wr[3].wr.rdma.rkey, 0x500u + seg.h.nSegments + 21);
  EXPECT_EQ(wr[3].wr.rdma.remote_addr, seg.bases[seg.h.nSegments] + 21 * GinSegHandle::kSegBytes);
}

// A put wider than one round posts 64 WRs at a time. Each round's signaled tail starts the next round and does not
// complete the request; the last round's tail does.
TEST_F(NetIbMsegGinMicrotest, IPut_WiderThanOneRound_PostsTheNextRoundWhenEachTailCompletes) {
  GinRoundScene scene;
  GinSegHandle src(70, 0x10000000, 0xa00, 0xb00);
  GinSegHandle dst(70, 0x20000000, 0xc00, 0xd00);
  // A half-segment skew splits every segment in two: 138 WRs, posted as rounds of 64, 64 and 10.
  const size_t skew = GinSegHandle::kSegBytes / 2;
  const size_t size = 69 * GinSegHandle::kSegBytes;
  void* request = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIPut(&scene.context, 0, skew, &src.h, size, 0, &dst.h, 1, 0, &request), ncclSuccess);
  auto* req = static_cast<IbMsegRequest*>(request);
  ASSERT_NE(req, nullptr);
  EXPECT_EQ(req->events[0], 1);
  const uint64_t wrId = req - scene.send.base.reqs;

  const size_t rounds[] = {64, 128, 138};
  for (int r = 0; r < 3; r++) {
    ASSERT_EQ(scene.verbs.posted.size(), rounds[r]) << "round " << r;
    EXPECT_EQ(scene.SignaledCount(), r + 1);
    EXPECT_NE(scene.verbs.posted.back().sendFlags & IBV_SEND_SIGNALED, 0);
    scene.QueueCompletion(req);
    EXPECT_EQ(scene.Test(req), r == 2 ? 1 : 0) << "round " << r;
    if (r < 2) EXPECT_EQ(scene.send.ginSeq.completed, 0u);
  }
  EXPECT_EQ(scene.send.ginSeq.completed, req->id);
  EXPECT_EQ(scene.sendQueues[1].head, nullptr);
  for (size_t i = 0; i < scene.verbs.posted.size(); i++) {
    const auto& p = scene.verbs.posted[i];
    EXPECT_EQ(p.wrId, wrId) << i;
    EXPECT_EQ(p.sge.length, skew) << i;
    EXPECT_EQ(p.sge.addr, src.bases[0] + skew + i * skew) << i;
    EXPECT_EQ(p.remoteAddress, dst.bases[70] + i * skew) << i;
  }
  rcclIbSegStats stats{};
  ASSERT_EQ(rcclRmaIbProxySegStats(&scene.context, 0, 1, &stats), ncclSuccess);
  EXPECT_EQ(stats.posts, 3u) << "one count per posted round, none for the probe build";
  EXPECT_EQ(stats.wrs, 138u);
  EXPECT_EQ(stats.laterSegPosts, 3u);
  EXPECT_EQ(rcclRmaIbProxySegStats(&scene.context, 0, 2, &stats), ncclInvalidArgument);
}

// An op that arrives while a wider op is queued waits behind it, even one whose handle would take the single-MR
// path, so its completion cannot pass the earlier request.
TEST_F(NetIbMsegGinMicrotest, IPut_BehindAQueuedOp_PostsAfterItsLastRound) {
  GinRoundScene scene;
  GinSegHandle src(70, 0x10000000, 0xa00, 0xb00);
  GinSegHandle dst(70, 0x20000000, 0xc00, 0xd00);
  GinSegHandle one(1, 0x30000000, 0xe00, 0xf00);
  void* wide = nullptr;
  void* small = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIPut(&scene.context, 0, GinSegHandle::kSegBytes / 2, &src.h, 69 * GinSegHandle::kSegBytes,
                               0, &dst.h, 1, 0, &wide),
            ncclSuccess);
  ASSERT_EQ(rcclRmaIbProxyIPut(&scene.context, 0, 0, &one.h, 8, 0, &one.h, 1, 0, &small), ncclSuccess);
  auto* wideReq = static_cast<IbMsegRequest*>(wide);
  auto* smallReq = static_cast<IbMsegRequest*>(small);
  ASSERT_NE(smallReq, nullptr);
  EXPECT_EQ(smallReq->id, wideReq->id + 1);
  EXPECT_EQ(scene.verbs.posted.size(), 64u);
  EXPECT_EQ(scene.Test(smallReq), 0);

  for (int r = 0; r < 2; r++) {
    scene.QueueCompletion(wideReq);
    EXPECT_EQ(scene.Test(smallReq), 0);
  }
  ASSERT_EQ(scene.verbs.posted.size(), 139u) << "the small put follows the wide put's last round";
  EXPECT_EQ(scene.verbs.posted.back().wrId, (uint64_t)(smallReq - scene.send.base.reqs));
  EXPECT_EQ(scene.verbs.posted.back().remoteAddress, one.bases[1]);
  EXPECT_EQ(scene.verbs.posted.back().sge.lkey, 0xe00u);
  scene.QueueCompletion(wideReq);
  scene.QueueCompletion(smallReq);
  EXPECT_EQ(scene.Test(smallReq), 1);
  EXPECT_EQ(scene.Test(wideReq), 1);
}

// The signal rides only the last round, after every data WR of the op.
TEST_F(NetIbMsegGinMicrotest, IPutSignal_WiderThanOneRound_SignalsOnlyInTheLastRound) {
  GinRoundScene scene;
  GinSegHandle src(70, 0x10000000, 0xa00, 0xb00);
  GinSegHandle dst(70, 0x20000000, 0xc00, 0xd00);
  void* request = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIPutSignal(&scene.context, 0, GinSegHandle::kSegBytes / 2, &src.h,
                                     69 * GinSegHandle::kSegBytes, 0, &dst.h, 1, 8, &dst.h, 5,
                                     NCCL_NET_SIGNAL_OP_ADD, false, 0, &request),
            ncclSuccess);
  auto* req = static_cast<IbMsegRequest*>(request);
  ASSERT_NE(req, nullptr);
  for (int r = 0; r < 2; r++) {
    scene.QueueCompletion(req);
    EXPECT_EQ(scene.Test(req), 0);
  }
  ASSERT_EQ(scene.verbs.posted.size(), 139u);
  for (size_t i = 0; i + 1 < scene.verbs.posted.size(); i++) {
    EXPECT_EQ(scene.verbs.posted[i].opcode, IBV_WR_RDMA_WRITE) << i;
  }
  const auto& signal = scene.verbs.posted.back();
  EXPECT_EQ(signal.opcode, IBV_WR_ATOMIC_FETCH_AND_ADD);
  EXPECT_EQ(signal.compareAdd, 5u);
  EXPECT_EQ(signal.remoteAddress, dst.bases[70] + 8);
  EXPECT_EQ(signal.rkey, 0xd00u + 70);
  EXPECT_NE(signal.sendFlags & IBV_SEND_SIGNALED, 0);
}

// A round that fails to post marks its request FAILED, and its next test reports the error.
TEST_F(NetIbMsegGinMicrotest, IPut_LaterRoundPostFailure_FailsTheRequest) {
  GinRoundScene scene;
  GinSegHandle src(70, 0x10000000, 0xa00, 0xb00);
  GinSegHandle dst(70, 0x20000000, 0xc00, 0xd00);
  void* request = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIPut(&scene.context, 0, GinSegHandle::kSegBytes / 2, &src.h, 69 * GinSegHandle::kSegBytes,
                               0, &dst.h, 1, 0, &request),
            ncclSuccess);
  auto* req = static_cast<IbMsegRequest*>(request);
  scene.verbs.postSendFailOnCall = 2;
  scene.QueueCompletion(req);
  EXPECT_EQ(scene.Test(req), 0);
  EXPECT_EQ(req->type, NCCL_NET_IB_REQ_FAILED);
  EXPECT_EQ(scene.sendQueues[1].head, nullptr);
  EXPECT_EQ(scene.Test(req, ncclRemoteError), 1);
}

// A wide op rings the aggregated batch ahead of it first. If that doorbell fails, nothing is published or queued.
TEST_F(NetIbMsegGinMicrotest, IPut_BatchDoorbellFailure_PublishesNoRequest) {
  GinRoundScene scene;
  GinSegHandle src(70, 0x10000000, 0xa00, 0xb00);
  GinSegHandle dst(70, 0x20000000, 0xc00, 0xd00);
  void* batched = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIPut(&scene.context, 0, 0, &src.h, 8, 0, &dst.h, 1, ncclRmaOptFlagsAggregateRequests,
                               &batched),
            ncclSuccess);
  ASSERT_EQ(scene.verbs.posted.size(), 0u);
  scene.verbs.postSendFailOnCall = 1;
  void* request = nullptr;
  EXPECT_NE(rcclRmaIbProxyIPut(&scene.context, 0, GinSegHandle::kSegBytes / 2, &src.h, 69 * GinSegHandle::kSegBytes,
                               0, &dst.h, 1, 0, &request),
            ncclSuccess);
  EXPECT_EQ(request, nullptr);
  EXPECT_EQ(scene.sendQueues[1].head, nullptr);
}

// A multi-segment flush is one signaled read of the flush segment, through that segment's local MR.
TEST_F(NetIbMsegGinMicrotest, IFlush_MultiSegment_ReadsOneByteOfTheFlushSegment) {
  GinRoundScene scene;
  ScopedHook createFlushQp(g_ginCreateFlushQp, [](IbMsegRecvComm*) { return ncclSuccess; });
  GinSegHandle win(40, 0x10000000, 0xa00, 0xb00);
  win.h.flushSeg = 5;
  win.mrs[5].rkey = 0x7705;
  void* request = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIFlush(&scene.context, 0, &win.h, 1, &request), ncclSuccess);
  auto* req = static_cast<IbMsegRequest*>(request);
  ASSERT_NE(req, nullptr);
  ASSERT_EQ(scene.verbs.posted.size(), 1u);
  const auto& flush = scene.verbs.posted[0];
  EXPECT_EQ(flush.qp, scene.flushQp);
  EXPECT_EQ(flush.opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(flush.remoteAddress, win.bases[5]);
  EXPECT_EQ(flush.rkey, 0x7705u);
  EXPECT_EQ(flush.sge.addr, 0x9000u);
  EXPECT_NE(flush.sendFlags & IBV_SEND_SIGNALED, 0);
  scene.QueueCompletion(req);
  EXPECT_EQ(scene.Test(req), 1);
  EXPECT_EQ(scene.recv.ginSeq.completed, scene.recv.ginSeq.posted);
}

// A window with no device segment has nothing for a GPUDirect flush to order.
TEST_F(NetIbMsegGinMicrotest, IFlush_AllHostSegments_PostsNothing) {
  GinRoundScene scene;
  ScopedHook createFlushQp(g_ginCreateFlushQp, [](IbMsegRecvComm*) { return ncclSuccess; });
  GinSegHandle win(3, 0x10000000, 0xa00, 0xb00);
  win.h.flushSeg = -1;
  void* request = reinterpret_cast<void*>(0x1);
  ASSERT_EQ(rcclRmaIbProxyIFlush(&scene.context, 0, &win.h, 1, &request), ncclSuccess);
  EXPECT_EQ(request, nullptr);
  EXPECT_TRUE(scene.verbs.posted.empty());
  EXPECT_EQ(scene.recv.ginSeq.posted, 0u);
  EXPECT_EQ(createFlushQp.calls, 0);
}

// A flush does not wait behind a send round queued to the same peer.
TEST_F(NetIbMsegGinMicrotest, IFlush_BehindAQueuedSend_PostsAtOnce) {
  GinRoundScene scene;
  ScopedHook createFlushQp(g_ginCreateFlushQp, [](IbMsegRecvComm*) { return ncclSuccess; });
  GinSegHandle src(70, 0x10000000, 0xa00, 0xb00);
  GinSegHandle dst(70, 0x20000000, 0xc00, 0xd00);
  void* wide = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIPut(&scene.context, 0, GinSegHandle::kSegBytes / 2, &src.h, 69 * GinSegHandle::kSegBytes,
                               0, &dst.h, 1, 0, &wide),
            ncclSuccess);
  ASSERT_NE(scene.sendQueues[1].head, nullptr);
  void* request = nullptr;
  ASSERT_EQ(rcclRmaIbProxyIFlush(&scene.context, 0, &dst.h, 1, &request), ncclSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(scene.verbs.posted.size(), 65u);
  EXPECT_EQ(scene.verbs.posted.back().qp, scene.flushQp);
  EXPECT_EQ(scene.verbs.posted.back().remoteAddress, dst.bases[0]);
  auto* wideReq = static_cast<IbMsegRequest*>(wide);
  for (int r = 0; r < 3; r++) {
    scene.QueueCompletion(wideReq);
    EXPECT_EQ(scene.Test(wideReq), r == 2 ? 1 : 0) << "round " << r;
  }
  EXPECT_EQ(scene.sendQueues[1].head, nullptr);
}

// AICOMRCCL-3094: a failed post should release the request, sequence, and
// retained batch state allocated for an operation that never reached hardware.
TEST_F(NetIbMsegGinMicrotest, DISABLED_IPut_PostFailureRollsBackRequestSequenceAndBatch) {
  GinDataScene scene;
  scene.verbs.postSendErr = EIO;
  void* request = nullptr;

  EXPECT_EQ(IbMsegRmaIbProxyIPut(&scene.context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1, 0,
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
TEST_F(NetIbMsegGinMicrotest, DISABLED_OtherDataApis_PostFailureRollsBackRequestSequenceAndBatch) {
  {
    GinDataScene scene;
    scene.verbs.postSendErr = EIO;
    void* request = nullptr;
    EXPECT_EQ(IbMsegRmaIbProxyIGet(&scene.context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1, 0,
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
    EXPECT_EQ(IbMsegRmaIbProxyIPutSignal(&scene.context, 0, 0, nullptr, 0, 0, nullptr, 1, 8,
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
    ScopedHook createFlushQp(g_ginCreateFlushQp, [](IbMsegRecvComm*) { return ncclSuccess; });
    void* request = nullptr;
    EXPECT_EQ(IbMsegRmaIbProxyIFlush(&scene.context, 0, &scene.dstHandle, 1, &request), ncclSystemError);
    EXPECT_EQ(request, nullptr);
    EXPECT_EQ(scene.recv.ginSeq.posted, 0u);
    EXPECT_EQ(scene.recv.base.reqs[0].type, NCCL_NET_IB_REQ_UNUSED);
  }
}

// AICOMRCCL-3094: a failed barrier all-gather should release its
// temporary allocation before returning the dependency error.
TEST_F(NetIbMsegGinMicrotest, DISABLED_P2PBarrier_AllGatherFailureFreesScratch) {
  IbMsegGinIbCollComm coll{};
  coll.rank = 0;
  coll.nranks = 1;
  ScopedHook regMr(g_ginNetRegMr, [](void*, void*, size_t, int, void**) { return ncclSystemError; });

  EXPECT_EQ(IbMsegGinIbP2PBarrier(&coll), ncclSystemError);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

// AICOMRCCL-3094: the local all-gather contribution should not invoke memcpy
// with identical source and destination ranges.
TEST_F(NetIbMsegGinMicrotest, DISABLED_P2PBarrier_LocalContributionAvoidsOverlappingMemcpy) {
  IbMsegGinIbCollComm coll{};
  coll.rank = 0;
  coll.nranks = 1;
  int registration = 1;
  ScopedHook regMr(g_ginNetRegMr, [&](void*, void*, size_t, int, void** handle) {
    *handle = &registration;
    return ncclSuccess;
  });
  ScopedHook deregMr(g_ginNetDeregMr, [](void*, void*) { return ncclSuccess; });

  ASSERT_EQ(IbMsegGinIbP2PBarrier(&coll), ncclSuccess);
  EXPECT_EQ(g_ginSelfCopies, 0);
}

// AICOMRCCL-3094: the backend context is owned by DestroyContext,
// just like its child arrays and peer communications.
TEST_F(NetIbMsegGinMicrotest, DISABLED_DestroyContext_FreesTopLevelContext) {
  IbMsegRmaIbProxyCtx* context = nullptr;
  ASSERT_EQ(g_hostAllocs.Calloc(&context, 1), ncclSuccess);
  context->nContexts = 1;
  context->nranks = 0;

  ASSERT_EQ(IbMsegRmaIbProxyDestroyContext(context), ncclSuccess);
  EXPECT_TRUE(g_hostAllocs.live.empty());
}

// AICOMRCCL-3094: disabling the explicit flush must not leave an
// already-aggregated batch indefinitely unposted.
TEST_F(NetIbMsegGinMicrotest, DISABLED_IFlush_DisabledFlushStillDrainsAggregatedBatch) {
  GinDataScene scene;
  scene.recv.flushEnabled = false;
  ASSERT_NE(scene.PostPut(ncclRmaOptFlagsAggregateRequests), nullptr);
  ASSERT_TRUE(scene.verbs.posted.empty());
  void* flushRequest = reinterpret_cast<void*>(0x1);

  ASSERT_EQ(IbMsegRmaIbProxyIFlush(&scene.context, 0, &scene.dstHandle, 1, &flushRequest), ncclSuccess);
  EXPECT_EQ(flushRequest, nullptr);
  EXPECT_EQ(scene.verbs.posted.size(), 1u);
}

TEST_F(NetIbMsegGinMicrotest, RmaVtable_ExportsProxyBackendContract) {
  EXPECT_STREQ(IbMsegRmaIbProxy.name, "RMA_IB_PROXY");
  EXPECT_EQ(IbMsegRmaIbProxy.iput, IbMsegRmaIbProxyIPut);
  EXPECT_EQ(IbMsegRmaIbProxy.iputSignal, IbMsegRmaIbProxyIPutSignal);
  EXPECT_EQ(IbMsegRmaIbProxy.iget, IbMsegRmaIbProxyIGet);
  EXPECT_EQ(IbMsegRmaIbProxy.iflush, IbMsegRmaIbProxyIFlush);
  EXPECT_EQ(IbMsegRmaIbProxy.test, IbMsegRmaIbProxyTest);
  EXPECT_EQ(IbMsegRmaIbProxy.rmaProgress, nullptr);
  EXPECT_EQ(IbMsegRmaIbProxy.queryLastError, nullptr);
}

// One doorbell: every WR but the last is unsignaled; the last is the signaled
// tail that a single completion retires the whole batch with.
void ExpectSignaledTail(const IbVerbsScene::Send& send) {
  ASSERT_FALSE(send.chain.empty());
  for (size_t i = 0; i + 1 < send.chain.size(); i++) {
    EXPECT_EQ(send.chain[i].wr.send_flags & IBV_SEND_SIGNALED, 0) << "wr " << i;
  }
  EXPECT_NE(send.chain.back().wr.send_flags & IBV_SEND_SIGNALED, 0);
}

TEST_F(NetIbMsegGinMicrotest, AggregatePuts_StayUnpostedUntilTheBatchCannotHoldAPutSignal) {
  GinDataScene scene;
  scene.context.wrBatchsize = 4;
  ASSERT_NE(scene.PostPut(ncclRmaOptFlagsAggregateRequests), nullptr);
  ASSERT_NE(scene.PostPut(ncclRmaOptFlagsAggregateRequests), nullptr);
  EXPECT_TRUE(scene.verbs.sends.empty());

  // Two held puts plus a third would leave no room for a put-signal, so the third closes the batch.
  ASSERT_NE(scene.PostPut(ncclRmaOptFlagsAggregateRequests), nullptr);
  ASSERT_EQ(scene.verbs.sends.size(), 1u);
  EXPECT_EQ(scene.verbs.sends[0].qp, scene.dataQp);
  ASSERT_EQ(scene.verbs.sends[0].chain.size(), 3u);
  ExpectSignaledTail(scene.verbs.sends[0]);
}

TEST_F(NetIbMsegGinMicrotest, DefaultPuts_RingTheDoorbellForEachRequest) {
  GinDataScene scene;
  for (size_t i = 0; i < 3; i++) {
    ASSERT_NE(scene.PostPut(), nullptr);
    ASSERT_EQ(scene.verbs.sends.size(), i + 1);
    EXPECT_EQ(scene.verbs.sends.back().qp, scene.dataQp);
    ASSERT_EQ(scene.verbs.sends.back().chain.size(), 1u);
    ExpectSignaledTail(scene.verbs.sends.back());
  }
}

TEST_F(NetIbMsegGinMicrotest, PutSignal_ClosesTheBatchOnceAnotherPutSignalWouldNotFit) {
  GinDataScene scene;
  scene.context.wrBatchsize = 4;
  ASSERT_NE(scene.PostPut(ncclRmaOptFlagsAggregateRequests), nullptr);
  ASSERT_NE(scene.PostPut(ncclRmaOptFlagsAggregateRequests), nullptr);
  EXPECT_TRUE(scene.verbs.sends.empty());

  // The two held WRs plus this put-signal's two fill the batch, leaving no room for another put-signal.
  void* request = nullptr;
  ASSERT_EQ(IbMsegRmaIbProxyIPutSignal(&scene.context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1, 8,
                                    &scene.dstHandle, 5, NCCL_NET_SIGNAL_OP_ADD, false,
                                    ncclRmaOptFlagsAggregateRequests, &request),
            ncclSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(scene.verbs.sends.size(), 1u);
  ASSERT_EQ(scene.verbs.posted.size(), 4u);
  EXPECT_EQ(scene.verbs.posted[2].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(scene.verbs.posted[3].opcode, IBV_WR_ATOMIC_FETCH_AND_ADD);
  ExpectSignaledTail(scene.verbs.sends[0]);
}

TEST_F(NetIbMsegGinMicrotest, PutSignal_StaysHeldUntilTheNextPutRingsTheDoorbell) {
  GinDataScene scene;
  scene.context.wrBatchsize = 6;
  ASSERT_NE(scene.PostPut(ncclRmaOptFlagsAggregateRequests), nullptr);
  ASSERT_NE(scene.PostPut(ncclRmaOptFlagsAggregateRequests), nullptr);
  void* request = nullptr;
  ASSERT_EQ(IbMsegRmaIbProxyIPutSignal(&scene.context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1, 8,
                                    &scene.dstHandle, 5, NCCL_NET_SIGNAL_OP_ADD, false,
                                    ncclRmaOptFlagsAggregateRequests, &request),
            ncclSuccess);
  EXPECT_TRUE(scene.verbs.sends.empty());

  // The default put closes the batch: two aggregated puts, one put-signal, then this put.
  ASSERT_NE(scene.PostPut(), nullptr);
  ASSERT_EQ(scene.verbs.sends.size(), 1u);
  ASSERT_EQ(scene.verbs.posted.size(), 5u);
  EXPECT_EQ(scene.verbs.posted[3].opcode, IBV_WR_ATOMIC_FETCH_AND_ADD);
  EXPECT_EQ(scene.verbs.posted[4].opcode, IBV_WR_RDMA_WRITE);
  ExpectSignaledTail(scene.verbs.sends[0]);
}

TEST_F(NetIbMsegGinMicrotest, IGet_AggregatesAReadTheSameWay) {
  GinDataScene scene;
  scene.context.wrBatchsize = 4;
  void* request = nullptr;
  ASSERT_EQ(IbMsegRmaIbProxyIGet(&scene.context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1,
                              ncclRmaOptFlagsAggregateRequests, &request),
            ncclSuccess);
  ASSERT_NE(request, nullptr);
  EXPECT_TRUE(scene.verbs.sends.empty());

  // A default read closes the batch: the held read plus this one, posted as one doorbell.
  void* closing = nullptr;
  ASSERT_EQ(IbMsegRmaIbProxyIGet(&scene.context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1, 0, &closing),
            ncclSuccess);
  ASSERT_NE(closing, nullptr);
  ASSERT_EQ(scene.verbs.sends.size(), 1u);
  ASSERT_EQ(scene.verbs.posted.size(), 2u);
  EXPECT_EQ(scene.verbs.posted[0].opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(scene.verbs.posted[1].opcode, IBV_WR_RDMA_READ);
  ExpectSignaledTail(scene.verbs.sends[0]);
}

TEST_F(NetIbMsegGinMicrotest, Test_PostsTheHeldBatchAndOneTailCompletionRetiresEveryEarlierRequest) {
  GinDataScene scene;
  scene.context.wrBatchsize = 4;
  IbMsegRequest* reqs[4];
  for (IbMsegRequest*& req : reqs) {
    req = scene.PostPut(ncclRmaOptFlagsAggregateRequests);
    ASSERT_NE(req, nullptr);
  }
  // Three puts fill the first batch and post it; the fourth stays held.
  ASSERT_EQ(scene.verbs.sends.size(), 1u);
  ASSERT_EQ(scene.verbs.sends[0].chain.size(), 3u);
  ExpectSignaledTail(scene.verbs.sends[0]);

  // Only the posted batch's tail completes; a single completion covers the earlier requests.
  scene.QueueCompletion(reqs[2]);
  int done = 0;
  ASSERT_EQ(IbMsegRmaIbProxyTest(nullptr, reqs[0], &done), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(scene.verbs.sends.size(), 2u);  // the held fourth put was posted by the test
  for (int i = 1; i <= 2; i++) {
    ASSERT_EQ(IbMsegRmaIbProxyTest(nullptr, reqs[i], &done), ncclSuccess);
    EXPECT_EQ(done, 1) << "request " << i;
  }
  done = 0;
  ASSERT_EQ(IbMsegRmaIbProxyTest(nullptr, reqs[3], &done), ncclSuccess);
  EXPECT_EQ(done, 0);

  scene.QueueCompletion(reqs[3]);
  ASSERT_EQ(IbMsegRmaIbProxyTest(nullptr, reqs[3], &done), ncclSuccess);
  EXPECT_EQ(done, 1);
}

TEST_F(NetIbMsegGinMicrotest, DestroyContext_PostsAHeldBatchBeforeReleasingTheComms) {
  GinDataScene scene;
  // DestroyContext frees the comm arrays with the ledgered free(), so they come from the ledger too.
  IbMsegRmaIbProxyCtx* context = nullptr;
  ASSERT_EQ(g_hostAllocs.Calloc(&context, 1), ncclSuccess);
  context->rank = 0;
  context->nranks = 2;
  context->nContexts = 1;
  context->wrBatchsize = NCCL_RMA_IB_WR_BATCHSIZE;
  void** sendComms = nullptr;
  void** recvComms = nullptr;
  ASSERT_EQ(g_hostAllocs.Calloc(&sendComms, 2), ncclSuccess);
  ASSERT_EQ(g_hostAllocs.Calloc(&recvComms, 2), ncclSuccess);
  context->fullSendComm = sendComms;
  context->fullRecvComm = recvComms;
  sendComms[1] = &scene.send;
  recvComms[1] = &scene.recv;
  HookOf<decltype(g_ginNetCloseSend)> closeSend{g_ginNetCloseSend, [](void*) { return ncclSuccess; }};
  HookOf<decltype(g_ginNetCloseRecv)> closeRecv{g_ginNetCloseRecv, [](void*) { return ncclSuccess; }};

  void* put = nullptr;
  ASSERT_EQ(IbMsegRmaIbProxyIPut(context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1,
                              ncclRmaOptFlagsAggregateRequests, &put),
            ncclSuccess);
  ASSERT_NE(put, nullptr);
  EXPECT_TRUE(scene.verbs.sends.empty());

  ASSERT_EQ(IbMsegRmaIbProxyDestroyContext(context), ncclSuccess);
  ASSERT_EQ(scene.verbs.sends.size(), 1u);
  EXPECT_EQ(scene.verbs.sends[0].qp, scene.dataQp);
  ASSERT_EQ(scene.verbs.sends[0].chain.size(), 1u);
  ExpectSignaledTail(scene.verbs.sends[0]);
  EXPECT_EQ(context->fullSendComm, nullptr);
  EXPECT_EQ(context->fullRecvComm, nullptr);
  g_hostAllocs.Free(context);
}

// The RMA proxy's GIN listen marks the handle and the listen comm, so its classic comms skip the multi-segment
// layout. The GDAKI listen marks them too, but it is not built on AMD.
TEST_F(NetIbMsegGinMicrotest, GinListens_MarkHandleAndListenCommAsGin) {
  ScopedHook init(g_ncclSocketInit, [](ncclSocket*, const ncclSocketAddress*, uint64_t, ncclSocketType,
                                       volatile uint32_t*, int, int) { return ncclSuccess; });
  ScopedHook socketListen(g_ncclSocketListen, [](ncclSocket*) { return ncclSuccess; });
  ScopedHook address(g_ncclSocketGetAddr, [](ncclSocket*, ncclSocketAddress* out) {
    std::memset(out, 0, sizeof(*out));
    out->sa.sa_family = AF_INET;
    return ncclSuccess;
  });
  ScopedHook close(g_ncclSocketClose, [](ncclSocket*, bool) { return ncclSuccess; });
  unsigned char classicHandle[NCCL_NET_HANDLE_MAXSIZE] = {};
  unsigned char ginHandle[NCCL_NET_HANDLE_MAXSIZE] = {};
  void* classic = nullptr;
  void* gin = nullptr;

  ASSERT_EQ(IbMsegListen(nullptr, 0, classicHandle, &classic), ncclSuccess);
  ASSERT_EQ(IbMsegRmaIbProxy.listen(nullptr, 0, ginHandle, &gin), ncclSuccess);
  EXPECT_FALSE(static_cast<IbMsegListenComm*>(classic)->isGin);
  EXPECT_TRUE(static_cast<IbMsegListenComm*>(gin)->isGin);
  EXPECT_NE(std::memcmp(classicHandle, ginHandle, sizeof(ginHandle)), 0);
  EXPECT_EQ(IbMsegCloseListen(classic), ncclSuccess);
  EXPECT_EQ(IbMsegCloseListen(gin), ncclSuccess);
}

TEST_F(NetIbMsegGinMicrotest, CreateContext_MarksTheRmaListenCommAsGin) {
  GinCreateContextScene scene(/*nranks=*/1, /*rank=*/0);
  ncclRmaConfig_t config{};
  config.nContexts = 1;
  IbMsegRmaIbProxyCtx* contexts = nullptr;

  ASSERT_EQ(IbMsegRmaIbProxyCreateContext(&scene.coll, &config, reinterpret_cast<void**>(&contexts)), ncclSuccess);
  EXPECT_TRUE(scene.listenToken.isGin);
  ASSERT_EQ(IbMsegRmaIbProxyDestroyContext(contexts), ncclSuccess);
  g_hostAllocs.Free(contexts);
}

// Branch: multi-segment on swaps in every MR and data entry; connection, test and teardown stay.
TEST_F(NetIbMsegGinMicrotest, RmaVtable_MultiSegmentInstallReplacesMrAndDataOps) {
  ncclRma_t rma = IbMsegRmaIbProxy;
  rcclRmaIbProxyInstallMultiSeg(&rma);
  EXPECT_EQ(rma.regMrSym, rcclRmaIbProxyRegMrSym);
  EXPECT_EQ(rma.regMrSymDmaBuf, rcclRmaIbProxyRegMrSymDmaBuf);
  EXPECT_EQ(rma.deregMrSym, rcclRmaIbProxyDeregMrSym);
  EXPECT_EQ(rma.iput, rcclRmaIbProxyIPut);
  EXPECT_EQ(rma.iputSignal, rcclRmaIbProxyIPutSignal);
  EXPECT_EQ(rma.iget, rcclRmaIbProxyIGet);
  EXPECT_EQ(rma.iflush, rcclRmaIbProxyIFlush);
  EXPECT_EQ(rma.init, IbMsegRmaIbProxy.init);
  EXPECT_EQ(rma.connect, IbMsegRmaIbProxy.connect);
  EXPECT_EQ(rma.test, IbMsegRmaIbProxy.test);
  EXPECT_EQ(rma.closeColl, IbMsegRmaIbProxy.closeColl);
  EXPECT_EQ(IbMsegRmaIbProxy.iput, IbMsegRmaIbProxyIPut) << "the shared table is untouched";
}

}  // namespace
