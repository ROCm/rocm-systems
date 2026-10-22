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
#include <cstring>
#include <deque>
#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>

#include <unistd.h>

#include "ScopedHook.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/socket_fakes.h"
#include "gin/gin_host.h"
#include "transport/net_ib/common.h"
#include "transport/net_ib/gin.h"

namespace {

std::unordered_set<void*> g_ginAllocations;
int g_ginAllocCalls = 0;
int g_ginAllocFailCall = -1;
int g_ginSelfCopies = 0;

ncclResult_t TestGinMalloc(void** ptr, size_t size) {
  ++g_ginAllocCalls;
  if (g_ginAllocCalls == g_ginAllocFailCall) return ncclSystemError;
  if (size == 0) {
    *ptr = nullptr;
    return ncclSuccess;
  }
  const size_t pageSize = static_cast<size_t>(::getpagesize());
  const size_t alignedSize = ((size + pageSize - 1) / pageSize) * pageSize;
  void* allocation = nullptr;
  if (posix_memalign(&allocation, pageSize, alignedSize) != 0) return ncclSystemError;
  std::memset(allocation, 0, size);
  *ptr = allocation;
  g_ginAllocations.insert(allocation);
  return ncclSuccess;
}

template <typename T>
ncclResult_t TestGinCalloc(T** ptr, size_t count) {
  ++g_ginAllocCalls;
  if (g_ginAllocCalls == g_ginAllocFailCall) return ncclSystemError;
  if (count == 0) {
    *ptr = nullptr;
    return ncclSuccess;
  }
  T* allocation = static_cast<T*>(std::calloc(count, sizeof(T)));
  if (allocation == nullptr) return ncclSystemError;
  *ptr = allocation;
  g_ginAllocations.insert(allocation);
  return ncclSuccess;
}

void TestGinFree(void* ptr) {
  if (ptr == nullptr) return;
  if (g_ginAllocations.erase(ptr) != 1) {
    ADD_FAILURE() << "GIN unit freed an unowned or already-freed pointer " << ptr;
    return;
  }
  std::free(ptr);
}

void ReleaseTrackedAllocationForFixture(void* ptr) {
  if (ptr != nullptr && g_ginAllocations.erase(ptr) == 1) std::free(ptr);
}

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
std::function<ncclResult_t()> g_ginGdrSupport = kGinHookFails;
std::function<ncclResult_t()> g_ginPeerMemSupport = kGinHookFails;
std::function<ncclResult_t(int)> g_ginDmaBufSupport = kGinHookFails;
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
ncclResult_t TestGinGdrSupport() { return g_ginGdrSupport(); }
ncclResult_t TestGinPeerMemSupport() { return g_ginPeerMemSupport(); }
ncclResult_t TestGinDmaBufSupport(int dev) { return g_ginDmaBufSupport(dev); }
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
  g_ginGdrSupport = kGinHookFails;
  g_ginPeerMemSupport = kGinHookFails;
  g_ginDmaBufSupport = kGinHookFails;
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

struct GinPostedWr {
  ibv_wr_opcode opcode{};
  int sendFlags = 0;
  uint64_t wrId = 0;
  uint64_t remoteAddress = 0;
  uint32_t rkey = 0;
  uint64_t compareAdd = 0;
  ibv_sge sge{};
};

std::vector<GinPostedWr> g_ginPostedWrs;
std::deque<ibv_wc> g_ginCompletions;
int g_ginPostResult = 0;
int g_ginPollResult = 0;

int GinPostSend(ibv_qp*, ibv_send_wr* wr, ibv_send_wr** badWr) {
  if (g_ginPostResult != 0) {
    if (badWr) *badWr = wr;
    return g_ginPostResult;
  }
  for (auto* current = wr; current != nullptr; current = current->next) {
    GinPostedWr copy{};
    copy.opcode = current->opcode;
    copy.sendFlags = current->send_flags;
    copy.wrId = current->wr_id;
    if (current->opcode == IBV_WR_ATOMIC_FETCH_AND_ADD) {
      copy.remoteAddress = current->wr.atomic.remote_addr;
      copy.rkey = current->wr.atomic.rkey;
      copy.compareAdd = current->wr.atomic.compare_add;
    } else {
      copy.remoteAddress = current->wr.rdma.remote_addr;
      copy.rkey = current->wr.rdma.rkey;
    }
    if (current->num_sge == 1 && current->sg_list) copy.sge = *current->sg_list;
    g_ginPostedWrs.push_back(copy);
  }
  return 0;
}

int GinPollCq(ibv_cq*, int count, ibv_wc* completions) {
  if (g_ginPollResult < 0) return g_ginPollResult;
  int n = std::min<int>(count, g_ginCompletions.size());
  for (int i = 0; i < n; ++i) {
    completions[i] = g_ginCompletions.front();
    g_ginCompletions.pop_front();
  }
  return n;
}

}  // namespace

#undef ncclIbMalloc
#undef ncclCalloc
#define ncclIbMalloc TestGinMalloc
#define ncclCalloc TestGinCalloc
#define free TestGinFree
#define ncclNetIb g_testGinNet
#define ncclIbInitDevices TestGinInitDevices
#define ncclIbFinalizeDevices TestGinFinalizeDevices
#define ncclIbGdrSupport TestGinGdrSupport
#define ncclIbPeerMemSupport TestGinPeerMemSupport
#define ncclIbDmaBufSupport TestGinDmaBufSupport
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
  void SetUp() override {
    ResetGinHooks();
    savedNIbDevs_ = ncclNIbDevs;
    savedNMergedIbDevs_ = ncclNMergedIbDevs;
    savedMergedDev_ = ncclIbMergedDevs[0];
    for (size_t i = 0; i < savedProviders_.size(); ++i) savedProviders_[i] = ncclIbDevs[i].ibProvider;
    savedDevContext_ = ncclIbDevs[0].context;
    savedDevPd_ = ncclIbDevs[0].pd;
  }

  void TearDown() override {
    ncclNIbDevs = savedNIbDevs_;
    ncclNMergedIbDevs = savedNMergedIbDevs_;
    ncclIbMergedDevs[0] = savedMergedDev_;
    for (size_t i = 0; i < savedProviders_.size(); ++i) ncclIbDevs[i].ibProvider = savedProviders_[i];
    ncclIbDevs[0].context = savedDevContext_;
    ncclIbDevs[0].pd = savedDevPd_;
    g_ginPostedWrs.clear();
    g_ginCompletions.clear();
    g_ginPostResult = 0;
    g_ginPollResult = 0;
    g_ginAllocCalls = 0;
    g_ginAllocFailCall = -1;
    g_ginSelfCopies = 0;
    ResetGinHooks();
    ResetSocketFakes();
    ResetNcclFakes();
    if (!g_ginAllocations.empty()) {
      ADD_FAILURE() << "GIN unit leaked " << g_ginAllocations.size() << " tracked allocation(s)";
    }
    for (void* ptr : g_ginAllocations) std::free(ptr);
    g_ginAllocations.clear();
  }

 private:
  int savedNIbDevs_ = 0;
  int savedNMergedIbDevs_ = 0;
  ncclIbMergedDev savedMergedDev_{};
  std::array<ncclIbProvider, 3> savedProviders_{};
  ibv_context* savedDevContext_ = nullptr;
  ibv_pd* savedDevPd_ = nullptr;
};

TEST_F(NetIbGinMicrotest, GdrSupport_PeerMemoryOrDmaBufMakesGinAvailable) {
  bool supported = false;
  g_ginGdrSupport = [] { return ncclSuccess; };
  g_ginDmaBufSupport = [](int) { return ncclSystemError; };
  ASSERT_EQ(ncclGinIbGdrSupport(&supported, false), ncclSuccess);
  EXPECT_TRUE(supported);

  g_ginGdrSupport = [] { return ncclSystemError; };
  g_ginDmaBufSupport = [](int dev) {
    EXPECT_EQ(dev, 0);
    return ncclSuccess;
  };
  ASSERT_EQ(ncclGinIbGdrSupport(&supported, false), ncclSuccess);
  EXPECT_TRUE(supported);

  g_ginDmaBufSupport = [](int) { return ncclSystemError; };
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

TEST_F(NetIbGinMicrotest, GdrGpuSupport_NoPeerMemoryOrDmaBufReturnsInvalidUsage) {
  g_ginGdrSupport = [] { return ncclSystemError; };
  g_ginDmaBufSupport = [](int) { return ncclSystemError; };

  EXPECT_EQ(ncclGinIbGdrGpuSupport(false), ncclInvalidUsage);

  g_ginDmaBufSupport = [](int) { return ncclSuccess; };
  EXPECT_EQ(ncclGinIbGdrGpuSupport(false), ncclSuccess);
}

TEST_F(NetIbGinMicrotest, InitType_ValidDeviceAndGdrPublishesDefaultTrafficClassContext) {
  g_ginInitDevices = [](ncclDebugLogger_t, ncclProfilerCallback_t) {
    ncclNIbDevs = 1;
    return ncclSuccess;
  };
  g_ginGdrSupport = [] { return ncclSuccess; };
  void* context = nullptr;

  ASSERT_EQ(ncclGinIbInitType(&context, 7, nullptr, NCCL_GIN_TYPE_PROXY), ncclSuccess);
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(static_cast<ncclNetCommConfig_t*>(context)->trafficClass, NCCL_NET_TRAFFIC_CLASS_UNDEF);
  EXPECT_EQ(g_ginAllocations.size(), 1u);
  TestGinFree(context);
  EXPECT_TRUE(g_ginAllocations.empty());
}

TEST_F(NetIbGinMicrotest, InitType_NoDevicesOrGdrReturnsErrorWithoutPublishingContext) {
  g_ginInitDevices = [](ncclDebugLogger_t, ncclProfilerCallback_t) {
    ncclNIbDevs = 0;
    return ncclSuccess;
  };
  void* context = nullptr;
  EXPECT_EQ(ncclGinIbInitType(&context, 0, nullptr, NCCL_GIN_TYPE_PROXY), ncclInternalError);
  EXPECT_EQ(context, nullptr);

  g_ginInitDevices = [](ncclDebugLogger_t, ncclProfilerCallback_t) {
    ncclNIbDevs = 1;
    return ncclSuccess;
  };
  g_ginGdrSupport = [] { return ncclSystemError; };
  g_ginDmaBufSupport = [](int) { return ncclSystemError; };
  EXPECT_EQ(ncclGinIbInitType(&context, 0, nullptr, NCCL_GIN_TYPE_PROXY), ncclInternalError);
  EXPECT_EQ(context, nullptr);
  EXPECT_TRUE(g_ginAllocations.empty());
}

TEST_F(NetIbGinMicrotest, ProxyInit_UsesConfiguredTypeAndPublishesContext) {
  g_ginInitDevices = [](ncclDebugLogger_t, ncclProfilerCallback_t) {
    ncclNIbDevs = 1;
    return ncclSuccess;
  };
  g_ginGdrSupport = [] { return ncclSystemError; };
  g_ginPeerMemSupport = [] { return ncclSuccess; };
  g_ginDmaBufSupport = [](int) { return ncclSystemError; };
  g_loadParam = [](const char* name, int64_t fallback) {
    return std::strcmp(name, "GIN_TYPE") == 0 ? int64_t{NCCL_GIN_TYPE_GDAKI} : fallback;
  };
  void* context = nullptr;

  ASSERT_EQ(ncclRmaIbProxyInit(&context, 42, nullptr), ncclSuccess);
  ASSERT_NE(context, nullptr);
  TestGinFree(context);
  EXPECT_TRUE(g_ginAllocations.empty());
}

TEST_F(NetIbGinMicrotest, Finalize_FreesContextAndPropagatesDeviceFinalize) {
  void* context = nullptr;
  ASSERT_EQ(TestGinMalloc(&context, 16), ncclSuccess);
  g_ginFinalizeDevices = [] { return ncclRemoteError; };

  EXPECT_EQ(ncclGinIbFinalize(context), ncclRemoteError);
  EXPECT_TRUE(g_ginAllocations.empty());
}

TEST_F(NetIbGinMicrotest, RmaProperties_HipReportsSignalsDoNotFlushAllPuts) {
  ncclGinIbCollComm coll{};
  ncclRmaProperties_t props{};
  props.flushesAllPutsOnAnySignal = true;

  EXPECT_EQ(ncclRmaIbProxyGetRmaProperties(&coll, &props), ncclSuccess);
  EXPECT_FALSE(props.flushesAllPutsOnAnySignal);
}

TEST_F(NetIbGinMicrotest, GetProperties_DelegatesAndMarksGinProxyDevice) {
  g_ginNetGetProperties = [](int dev, ncclNetProperties_t* props) {
    EXPECT_EQ(dev, 2);
    props->speed = 123;
    return ncclSuccess;
  };
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
  g_ginNetRegMr = [&](void* comm, void*, size_t size, int type, void** handle) {
    EXPECT_EQ(size, 3 * sizeof(int));
    EXPECT_EQ(type, NCCL_PTR_HOST);
    *handle = comm == coll.recvComm ? static_cast<void*>(&recvHandle) : static_cast<void*>(&sendHandle);
    return ncclSuccess;
  };
  std::vector<std::pair<void*, void*>> deregistered;
  g_ginNetDeregMr = [&](void* comm, void* handle) {
    deregistered.emplace_back(comm, handle);
    return ncclSuccess;
  };
  std::vector<void*> sent;
  int sendRequest = 1;
  g_ginNetIsend = [&](void*, void* data, size_t size, int tag, void*, void*, void** request) {
    EXPECT_EQ(size, sizeof(int));
    EXPECT_EQ(tag, NCCL_GIN_IB_ALLGATHER_TAG);
    sent.push_back(data);
    *request = &sendRequest;
    return ncclSuccess;
  };
  std::vector<void*> received;
  int recvRequest = 2;
  g_ginNetIrecv = [&](void*, int n, void** data, size_t* sizes, int* tags, void**, void**, void** request) {
    EXPECT_EQ(n, 1);
    EXPECT_EQ(*sizes, sizeof(int));
    EXPECT_EQ(*tags, NCCL_GIN_IB_ALLGATHER_TAG);
    received.push_back(*data);
    *request = &recvRequest;
    return ncclSuccess;
  };
  g_ginNetTest = [](void*, int* done, int*) {
    *done = 1;
    return ncclSuccess;
  };
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
  g_ginNetRegMr = [&](void*, void*, size_t, int, void** handle) {
    *handle = &registration;
    return ncclSuccess;
  };
  g_ginNetDeregMr = [](void*, void*) { return ncclSuccess; };
  int sendToken = 1;
  int recvToken = 2;
  int sendCalls = 0;
  int recvCalls = 0;
  g_ginNetIsend = [&](void*, void*, size_t, int, void*, void*, void** request) {
    if (++sendCalls == 2) *request = &sendToken;
    return ncclSuccess;
  };
  g_ginNetIrecv = [&](void*, int, void**, size_t*, int*, void**, void**, void** request) {
    if (++recvCalls == 2) *request = &recvToken;
    return ncclSuccess;
  };
  int sendTests = 0;
  int recvTests = 0;
  g_ginNetTest = [&](void* request, int* done, int*) {
    int& calls = request == &sendToken ? sendTests : recvTests;
    *done = ++calls == 2;
    return ncclSuccess;
  };
  int source = 7;
  int gathered[2] = {};

  ASSERT_EQ(ncclGinIbAllGather(&coll, &source, gathered, sizeof(source)), ncclSuccess);
  EXPECT_EQ(gathered[0], source);
  EXPECT_EQ(sendCalls, 2);
  EXPECT_EQ(recvCalls, 2);
  EXPECT_EQ(sendTests, 2);
  EXPECT_EQ(recvTests, 2);
}

TEST_F(NetIbGinMicrotest, AllGather_SecondRegistrationFailureReleasesFirstHandle) {
  ncclGinIbCollComm coll{};
  coll.nranks = 2;
  int recvHandle = 1;
  int calls = 0;
  g_ginNetRegMr = [&](void*, void*, size_t, int, void** handle) {
    if (++calls == 1) {
      *handle = &recvHandle;
      return ncclSuccess;
    }
    return ncclSystemError;
  };
  int deregistered = 0;
  g_ginNetDeregMr = [&](void*, void* handle) {
    EXPECT_EQ(handle, &recvHandle);
    ++deregistered;
    return ncclSuccess;
  };
  int source = 1;
  int gathered[2] = {};

  EXPECT_EQ(ncclGinIbAllGather(&coll, &source, gathered, sizeof(source)), ncclSystemError);
  EXPECT_EQ(deregistered, 1);
}

TEST_F(NetIbGinMicrotest, AllToAll_ExtractsLocalRankColumnAndFreesScratch) {
  ncclGinIbCollComm coll{};
  coll.rank = 1;
  coll.nranks = 3;
  coll.allGather = [](ncclGinIbCollComm*, void*, void* out, size_t) {
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
  EXPECT_TRUE(g_ginAllocations.empty());
}

TEST_F(NetIbGinMicrotest, Connect_RetriesNullCommsAndPublishesRingContext) {
  ncclIbListenComm listen{};
  listen.dev = 0;
  ncclGinIbCollComm* published = nullptr;
  void* handles[3] = {reinterpret_cast<void*>(0x10), reinterpret_cast<void*>(0x20),
                      reinterpret_cast<void*>(0x30)};
  int sendToken = 1;
  int recvToken = 2;
  int connectCalls = 0;
  int acceptCalls = 0;
  g_loadParam = [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "GIN_IB_TC") == 0) return int64_t{-1};
    if (std::strcmp(name, "IB_TC") == 0) return int64_t{4};
    return fallback;
  };
  g_ginConnect = [&](void*, int dev, void* handle, void** comm, ncclNetDeviceHandle_t**, int nQps, int tc) {
    EXPECT_EQ(dev, 0);
    EXPECT_EQ(handle, handles[2]);
    EXPECT_EQ(nQps, 1);
    EXPECT_EQ(tc, 4);
    if (++connectCalls == 2) *comm = &sendToken;
    return ncclSuccess;
  };
  g_ginAccept = [&](void* seen, void** comm, ncclNetDeviceHandle_t**, int nQps) {
    EXPECT_EQ(seen, &listen);
    EXPECT_EQ(nQps, 1);
    if (++acceptCalls == 2) *comm = &recvToken;
    return ncclSuccess;
  };
  ibv_context verbsContext{};
  ibv_pd pd{};
  ncclIbDevs[0].context = &verbsContext;
  ncclIbDevs[0].pd = &pd;

  ASSERT_EQ(ncclGinIbConnect(nullptr, handles, 3, 1, &listen, reinterpret_cast<void**>(&published)), ncclSuccess);
  ASSERT_NE(published, nullptr);
  EXPECT_EQ(connectCalls, 2);
  EXPECT_EQ(acceptCalls, 2);
  EXPECT_EQ(published->rank, 1);
  EXPECT_EQ(published->nranks, 3);
  EXPECT_EQ(published->sendComm, &sendToken);
  EXPECT_EQ(published->recvComm, &recvToken);
  EXPECT_EQ(published->ib.context, &verbsContext);
  EXPECT_EQ(published->ib.pd, &pd);
  TestGinFree(published);
}

TEST_F(NetIbGinMicrotest, Connect_AcceptFailureClosesEstablishedSendComm) {
  ncclIbListenComm listen{};
  void* handles[1] = {nullptr};
  int sendToken = 1;
  g_ginConnect = [&](void*, int, void*, void** comm, ncclNetDeviceHandle_t**, int, int) {
    *comm = &sendToken;
    return ncclSuccess;
  };
  g_ginAccept = [](void*, void**, ncclNetDeviceHandle_t**, int) { return ncclSystemError; };
  int closed = 0;
  g_ginNetCloseSend = [&](void* comm) {
    EXPECT_EQ(comm, &sendToken);
    ++closed;
    return ncclSuccess;
  };
  void* coll = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(ncclGinIbConnect(nullptr, handles, 1, 0, &listen, &coll), ncclSystemError);
  EXPECT_EQ(coll, nullptr);
  EXPECT_EQ(closed, 1);
  EXPECT_TRUE(g_ginAllocations.empty());
}

TEST_F(NetIbGinMicrotest, ProxyConnect_GdrSupportedBuildsProxyConnection) {
  ncclIbListenComm listen{};
  listen.dev = 0;
  void* handles[1] = {nullptr};
  int sendToken = 1;
  int recvToken = 2;
  g_ginGdrSupport = [] { return ncclSuccess; };
  g_ginConnect = [&](void*, int, void*, void** comm, ncclNetDeviceHandle_t**, int, int) {
    *comm = &sendToken;
    return ncclSuccess;
  };
  g_ginAccept = [&](void*, void** comm, ncclNetDeviceHandle_t**, int) {
    *comm = &recvToken;
    return ncclSuccess;
  };
  ibv_context verbsContext{};
  ibv_pd pd{};
  ncclIbDevs[0].context = &verbsContext;
  ncclIbDevs[0].pd = &pd;
  void* coll = nullptr;

  ASSERT_EQ(ncclRmaIbProxyConnect(nullptr, handles, 1, 0, &listen, &coll), ncclSuccess);
  ASSERT_NE(coll, nullptr);
  EXPECT_EQ(static_cast<ncclGinIbCollComm*>(coll)->sendComm, &sendToken);
  EXPECT_EQ(static_cast<ncclGinIbCollComm*>(coll)->recvComm, &recvToken);
  TestGinFree(coll);
}

TEST_F(NetIbGinMicrotest, ProxyConnect_NoGdrReturnsInvalidUsageBeforeConnecting) {
  g_ginGdrSupport = [] { return ncclSystemError; };
  g_ginDmaBufSupport = [](int) { return ncclSystemError; };
  int connectCalls = 0;
  g_ginConnect = [&](void*, int, void*, void**, ncclNetDeviceHandle_t**, int, int) {
    ++connectCalls;
    return ncclSuccess;
  };
  void* coll = nullptr;

  EXPECT_EQ(ncclRmaIbProxyConnect(nullptr, nullptr, 1, 0, nullptr, &coll), ncclInvalidUsage);
  EXPECT_EQ(connectCalls, 0);
  EXPECT_EQ(coll, nullptr);
}

TEST_F(NetIbGinMicrotest, CloseColl_ClosesBothDirectionsAndFreesContext) {
  auto* coll = static_cast<ncclGinIbCollComm*>(nullptr);
  ASSERT_EQ(TestGinMalloc(reinterpret_cast<void**>(&coll), sizeof(*coll)), ncclSuccess);
  int sendToken = 1;
  int recvToken = 2;
  coll->sendComm = &sendToken;
  coll->recvComm = &recvToken;
  std::vector<void*> closed;
  g_ginNetCloseRecv = [&](void* comm) {
    closed.push_back(comm);
    return ncclSuccess;
  };
  g_ginNetCloseSend = [&](void* comm) {
    closed.push_back(comm);
    return ncclSuccess;
  };

  ASSERT_EQ(ncclGinIbCloseColl(coll), ncclSuccess);
  ASSERT_EQ(closed.size(), 2u);
  EXPECT_NE(std::find(closed.begin(), closed.end(), &recvToken), closed.end());
  EXPECT_NE(std::find(closed.begin(), closed.end(), &sendToken), closed.end());
  EXPECT_TRUE(g_ginAllocations.empty());
}

TEST_F(NetIbGinMicrotest, CreateContext_SingleRankBuildsContextsAndClampsBatchSize) {
  ncclGinIbCollComm coll{};
  coll.rank = 0;
  coll.nranks = 1;
  coll.dev = 0;
  int transportContext = 1;
  coll.ctx = &transportContext;
  coll.allGather = [](ncclGinIbCollComm*, void* src, void* dst, size_t size) {
    TestGinMemcpy(dst, src, size);
    return ncclSuccess;
  };
  int listenToken = 1;
  g_ginNetListen = [&](void* ctx, int dev, void*, void** listen) {
    EXPECT_EQ(ctx, &transportContext);
    EXPECT_EQ(dev, 0);
    *listen = &listenToken;
    return ncclSuccess;
  };
  g_ginNetCloseListen = [&](void* listen) {
    EXPECT_EQ(listen, &listenToken);
    return ncclSuccess;
  };
  int sendToken[2] = {};
  int recvToken[2] = {};
  int connectCalls = 0;
  int acceptCalls = 0;
  g_ginConnect = [&](void*, int, void*, void** comm, ncclNetDeviceHandle_t**, int, int) {
    *comm = &sendToken[connectCalls++];
    return ncclSuccess;
  };
  g_ginAccept = [&](void*, void** comm, ncclNetDeviceHandle_t**, int) {
    *comm = &recvToken[acceptCalls++];
    return ncclSuccess;
  };
  int registration = 1;
  g_ginNetRegMr = [&](void*, void*, size_t, int, void** handle) {
    *handle = &registration;
    return ncclSuccess;
  };
  g_ginNetDeregMr = [](void*, void*) { return ncclSuccess; };
  g_ginSetTrafficClass = [&](void* ctx, int tc) {
    EXPECT_EQ(ctx, &transportContext);
    EXPECT_EQ(tc, 7);
  };
  g_loadParam = [](const char* name, int64_t fallback) {
    return std::strcmp(name, "RMA_IB_WR_BATCHSIZE") == 0 ? int64_t{1} : fallback;
  };
  ncclRmaConfig_t config{};
  config.nContexts = 2;
  config.rankStride = 1;
  config.trafficClass = 7;
  ncclRmaIbProxyCtx* contexts = nullptr;

  ASSERT_EQ(ncclRmaIbProxyCreateContext(&coll, &config, reinterpret_cast<void**>(&contexts)), ncclSuccess);
  ASSERT_NE(contexts, nullptr);
  EXPECT_EQ(connectCalls, 2);
  EXPECT_EQ(acceptCalls, 2);
  EXPECT_EQ(contexts[0].nContexts, 2);
  EXPECT_EQ(contexts[0].nranks, 1);
  EXPECT_EQ(contexts[0].wrBatchsize, 2);
  EXPECT_EQ(contexts[1].wrBatchsize, 2);

  g_ginNetCloseRecv = [](void*) { return ncclSuccess; };
  g_ginNetCloseSend = [](void*) { return ncclSuccess; };
  ASSERT_EQ(ncclRmaIbProxyDestroyContext(contexts), ncclSuccess);
  ReleaseTrackedAllocationForFixture(contexts);
  EXPECT_TRUE(g_ginAllocations.empty());
}

TEST_F(NetIbGinMicrotest, CreateContext_NonDivisibleStrideIsRejectedBeforeAllocation) {
  ncclGinIbCollComm coll{};
  coll.nranks = 3;
  ncclRmaConfig_t config{};
  config.nContexts = 1;
  config.rankStride = 2;
  void* context = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(ncclRmaIbProxyCreateContext(&coll, &config, &context), ncclInternalError);
  EXPECT_TRUE(g_ginAllocations.empty());
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
  g_ginNetListen = [&](void*, int, void*, void** listen) {
    *listen = &listenToken;
    return ncclSuccess;
  };
  g_ginNetCloseListen = [](void*) { return ncclSuccess; };
  g_ginAllocFailCall = 4;
  ncclRmaConfig_t config{};
  config.nContexts = 1;
  config.rankStride = 1;
  void* context = nullptr;

  EXPECT_EQ(ncclRmaIbProxyCreateContext(&coll, &config, &context), ncclSystemError);
  EXPECT_EQ(context, nullptr);
  EXPECT_TRUE(g_ginAllocations.empty());
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
  g_ginNetListen = [&](void*, int, void*, void** listen) {
    *listen = &listenToken;
    return ncclSuccess;
  };
  g_ginNetCloseListen = [](void*) { return ncclSuccess; };
  g_ginConnect = [&](void*, int, void*, void** comm, ncclNetDeviceHandle_t**, int, int) {
    *comm = &sendToken;
    return ncclSuccess;
  };
  g_ginAccept = [&](void*, void** comm, ncclNetDeviceHandle_t**, int) {
    *comm = &recvToken;
    return ncclSuccess;
  };
  int sendCloses = 0;
  int recvCloses = 0;
  g_ginNetCloseSend = [&](void* comm) {
    EXPECT_EQ(comm, &sendToken);
    ++sendCloses;
    return ncclSuccess;
  };
  g_ginNetCloseRecv = [&](void* comm) {
    EXPECT_EQ(comm, &recvToken);
    ++recvCloses;
    return ncclSuccess;
  };
  // top-level context, handles, fullSendComm, fullRecvComm, then barrier dummy
  g_ginAllocFailCall = 5;
  ncclRmaConfig_t config{};
  config.nContexts = 1;
  config.rankStride = 1;
  void* context = nullptr;

  EXPECT_EQ(ncclRmaIbProxyCreateContext(&coll, &config, &context), ncclSystemError);
  EXPECT_EQ(context, nullptr);
  EXPECT_EQ(sendCloses, 1);
  EXPECT_EQ(recvCloses, 1);
  EXPECT_TRUE(g_ginAllocations.empty());
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
  g_ginRegMrDmaBuf = [&](void* comm, void*, size_t, int, uint64_t, int, uint64_t, void** handle) {
    EXPECT_EQ(comm, &recvToken);
    *handle = &ibHandle;
    return ncclSuccess;
  };
  int gatherCall = 0;
  g_ginCollAllGather = [&](ncclGinIbCollComm*, void* src, void* dst, size_t size) {
    if (gatherCall++ == 0) {
      auto local = *static_cast<void**>(src);
      auto* bases = static_cast<uintptr_t*>(dst);
      bases[0] = reinterpret_cast<uintptr_t>(local);
      bases[1] = 0x2000;
      EXPECT_EQ(size, sizeof(uintptr_t));
    } else {
      auto* rkeys = static_cast<uint32_t*>(dst);
      rkeys[0] = *static_cast<uint32_t*>(src);
      rkeys[1] = 0x55;
      EXPECT_EQ(size, sizeof(uint32_t));
    }
    return ncclSuccess;
  };
  coll.allGather = GinCollAllGather;
  int deregistered = 0;
  g_ginNetDeregMr = [&](void* comm, void* handle) {
    EXPECT_EQ(comm, &recvToken);
    EXPECT_EQ(handle, &ibHandle);
    ++deregistered;
    return ncclSuccess;
  };
  int data = 0;
  ncclRmaIbProxyMrHandle* handle = nullptr;

  ASSERT_EQ(ncclRmaIbProxyRegMrSym(&coll, &data, sizeof(data), NCCL_PTR_HOST, 0,
                                  reinterpret_cast<void**>(&handle)),
            ncclSuccess);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->base_vas[0], reinterpret_cast<uintptr_t>(&data));
  EXPECT_EQ(handle->base_vas[1], 0x2000u);
  EXPECT_EQ(handle->rkeys[0], 0x44u);
  EXPECT_EQ(handle->rkeys[1], 0x55u);
  ASSERT_EQ(ncclRmaIbProxyDeregMrSym(&coll, handle), ncclSuccess);
  EXPECT_EQ(deregistered, 1);
  EXPECT_TRUE(g_ginAllocations.empty());
}

TEST_F(NetIbGinMicrotest, RegMrSym_FirstGatherFailureDeregistersAndFreesAllWrappers) {
  ncclGinIbCollComm coll{};
  coll.nranks = 2;
  int recvToken = 1;
  coll.recvComm = &recvToken;
  ibv_mr mr{};
  ncclIbMrHandle ibHandle{};
  ibHandle.mrs[0] = &mr;
  g_ginRegMrDmaBuf = [&](void*, void*, size_t, int, uint64_t, int, uint64_t, void** handle) {
    *handle = &ibHandle;
    return ncclSuccess;
  };
  g_ginCollAllGather = [](ncclGinIbCollComm*, void*, void*, size_t) { return ncclSystemError; };
  coll.allGather = GinCollAllGather;
  int deregistered = 0;
  g_ginNetDeregMr = [&](void*, void* handle) {
    EXPECT_EQ(handle, &ibHandle);
    ++deregistered;
    return ncclSuccess;
  };
  int data = 0;
  void* handle = nullptr;

  EXPECT_EQ(ncclRmaIbProxyRegMrSym(&coll, &data, sizeof(data), NCCL_PTR_HOST, 0, &handle), ncclSystemError);
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(deregistered, 1);
  EXPECT_TRUE(g_ginAllocations.empty());
}

struct GinDataScene {
  ncclRmaIbProxyCtx context{};
  std::array<void*, 2> sendComms{};
  std::array<void*, 2> recvComms{};
  ncclIbSendComm send{};
  ncclIbRecvComm recv{};
  ibv_device device{};
  ibv_context verbsContext{};
  ibv_pd pd{};
  ibv_qp dataQp{};
  ibv_qp flushQp{};
  ibv_cq sendCq{};
  ibv_cq recvCq{};
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
    std::strncpy(device.name, "test-hca", sizeof(device.name) - 1);
    verbsContext.device = &device;
    verbsContext.ops.post_send = GinPostSend;
    verbsContext.ops.poll_cq = GinPollCq;
    pd.context = &verbsContext;
    dataQp.context = &verbsContext;
    flushQp.context = &verbsContext;
    sendCq.context = &verbsContext;
    recvCq.context = &verbsContext;
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
    send.base.qps[0].qp = &dataQp;
    send.base.qps[0].devIndex = 0;
    send.devs[0].base.cq = &sendCq;
    send.devs[0].base.pd = &pd;
    recv.base.isSend = false;
    recv.base.vProps.ndevs = 1;
    recv.base.nRemDevs = 1;
    recv.devs[0].base.cq = &recvCq;
    recv.devs[0].base.pd = &pd;
    recv.flushEnabled = true;
    recv.devs[0].gpuFlush.qp.qp = &flushQp;
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
};

TEST_F(NetIbGinMicrotest, ReserveWrs_QpSwitchPostsPendingChainBeforeChangingQp) {
  GinDataScene scene;
  ibv_qp otherQp{};
  otherQp.context = &scene.verbsContext;
  ibv_send_wr* wr = nullptr;
  ibv_sge* sge = nullptr;
  ASSERT_EQ(ncclRmaIbProxyReserveWrs(&scene.context, &scene.dataQp, 1, &wr, &sge), ncclSuccess);
  wr->opcode = IBV_WR_RDMA_WRITE;
  wr->sg_list = sge;
  wr->num_sge = 1;
  ASSERT_EQ(ncclRmaIbProxyCommitWrs(&scene.context, 1, true), ncclSuccess);
  ASSERT_TRUE(g_ginPostedWrs.empty());

  ASSERT_EQ(ncclRmaIbProxyReserveWrs(&scene.context, &otherQp, 1, &wr, &sge), ncclSuccess);
  ASSERT_EQ(g_ginPostedWrs.size(), 1u);
  EXPECT_EQ(scene.context.batch.qp, &otherQp);
  EXPECT_EQ(scene.context.batch.nWrs, 0);
}

TEST_F(NetIbGinMicrotest, CommitWrs_AggregatePostsWhenNextMaximumOperationWouldNotFit) {
  GinDataScene scene;
  scene.context.wrBatchsize = 2;
  ibv_send_wr* wr = nullptr;
  ibv_sge* sge = nullptr;
  ASSERT_EQ(ncclRmaIbProxyReserveWrs(&scene.context, &scene.dataQp, 1, &wr, &sge), ncclSuccess);
  wr->opcode = IBV_WR_RDMA_WRITE;
  wr->sg_list = sge;
  wr->num_sge = 1;

  ASSERT_EQ(ncclRmaIbProxyCommitWrs(&scene.context, 1, true), ncclSuccess);

  ASSERT_EQ(g_ginPostedWrs.size(), 1u);
  EXPECT_EQ(scene.context.batch.nWrs, 0);
  EXPECT_EQ(scene.context.batch.qp, nullptr);
}

TEST_F(NetIbGinMicrotest, CommitWrs_AggregateDefersWhenNextMaximumOperationExactlyFits) {
  GinDataScene scene;
  scene.context.wrBatchsize = 3;
  ibv_send_wr* wr = nullptr;
  ibv_sge* sge = nullptr;
  ASSERT_EQ(ncclRmaIbProxyReserveWrs(&scene.context, &scene.dataQp, 1, &wr, &sge), ncclSuccess);
  wr->opcode = IBV_WR_RDMA_WRITE;
  wr->sg_list = sge;
  wr->num_sge = 1;

  ASSERT_EQ(ncclRmaIbProxyCommitWrs(&scene.context, 1, true), ncclSuccess);

  EXPECT_TRUE(g_ginPostedWrs.empty());
  EXPECT_EQ(scene.context.batch.nWrs, 1);
  EXPECT_EQ(scene.context.batch.qp, &scene.dataQp);
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
  EXPECT_TRUE(g_ginPostedWrs.empty());
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
  ASSERT_EQ(g_ginPostedWrs.size(), 1u);
  EXPECT_EQ(g_ginPostedWrs[0].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_NE(g_ginPostedWrs[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(g_ginPostedWrs[0].remoteAddress, 0x4005u);
  EXPECT_EQ(g_ginPostedWrs[0].rkey, 0x21u);
  EXPECT_EQ(g_ginPostedWrs[0].sge.addr, 0x1003u);
  EXPECT_EQ(g_ginPostedWrs[0].sge.length, 16u);
  EXPECT_EQ(g_ginPostedWrs[0].sge.lkey, 0x31u);
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
  ASSERT_EQ(g_ginPostedWrs.size(), 1u);
  EXPECT_EQ(g_ginPostedWrs[0].opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(g_ginPostedWrs[0].remoteAddress, 0x2007u);
  EXPECT_EQ(g_ginPostedWrs[0].rkey, 0x11u);
  EXPECT_EQ(g_ginPostedWrs[0].sge.addr, 0x3009u);
  EXPECT_EQ(g_ginPostedWrs[0].sge.length, 32u);
  EXPECT_EQ(g_ginPostedWrs[0].sge.lkey, 0x41u);
  EXPECT_EQ(static_cast<ncclIbRequest*>(request)->type, NCCL_NET_IB_REQ_GIN_IGET);
}

TEST_F(NetIbGinMicrotest, IPutSignal_PutAndAddPostOrderedWriteThenAtomic) {
  GinDataScene scene;
  void* request = nullptr;

  ASSERT_EQ(ncclRmaIbProxyIPutSignal(&scene.context, 0, 1, &scene.srcHandle, 8, 2, &scene.dstHandle, 1, 8,
                                    &scene.dstHandle, 9, NCCL_NET_SIGNAL_OP_ADD, false, 0, &request),
            ncclSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(g_ginPostedWrs.size(), 2u);
  EXPECT_EQ(g_ginPostedWrs[0].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(g_ginPostedWrs[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(g_ginPostedWrs[0].remoteAddress, 0x4002u);
  EXPECT_EQ(g_ginPostedWrs[1].opcode, IBV_WR_ATOMIC_FETCH_AND_ADD);
  EXPECT_EQ(g_ginPostedWrs[1].remoteAddress, 0x4008u);
  EXPECT_EQ(g_ginPostedWrs[1].rkey, 0x21u);
  EXPECT_EQ(g_ginPostedWrs[1].compareAdd, 9u);
  EXPECT_EQ(g_ginPostedWrs[1].sge.addr, reinterpret_cast<uintptr_t>(&scene.send.putSignalScratchpad));
  EXPECT_NE(g_ginPostedWrs[1].sendFlags & IBV_SEND_SIGNALED, 0);
}

TEST_F(NetIbGinMicrotest, IPutSignal_SignalOnlyIncrementPostsOneAtomicWithValueOne) {
  GinDataScene scene;
  void* request = nullptr;

  ASSERT_EQ(ncclRmaIbProxyIPutSignal(&scene.context, 0, 0, nullptr, 0, 0, nullptr, 1, 8, &scene.dstHandle, 99,
                                    NCCL_NET_SIGNAL_OP_INC, false, 0, &request),
            ncclSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(g_ginPostedWrs.size(), 1u);
  EXPECT_EQ(g_ginPostedWrs[0].opcode, IBV_WR_ATOMIC_FETCH_AND_ADD);
  EXPECT_EQ(g_ginPostedWrs[0].remoteAddress, 0x4008u);
  EXPECT_EQ(g_ginPostedWrs[0].compareAdd, 1u);
  EXPECT_NE(g_ginPostedWrs[0].sendFlags & IBV_SEND_SIGNALED, 0);
}

TEST_F(NetIbGinMicrotest, IPutSignal_UnsupportedOperationIsRejectedBeforeRequestOrPost) {
  GinDataScene scene;
  void* request = nullptr;

  EXPECT_EQ(ncclRmaIbProxyIPutSignal(&scene.context, 0, 0, nullptr, 0, 0, nullptr, 1, 8, &scene.dstHandle, 1,
                                    99, false, 0, &request),
            ncclInvalidArgument);
  EXPECT_EQ(request, nullptr);
  EXPECT_TRUE(g_ginPostedWrs.empty());
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
  EXPECT_TRUE(g_ginPostedWrs.empty());
  EXPECT_EQ(scene.send.ginSeq.posted, 0u);
}

TEST_F(NetIbGinMicrotest, AggregatedPut_DefersDoorbellUntilTestAndCompletesRequest) {
  GinDataScene scene;
  void* request = nullptr;
  ASSERT_EQ(ncclRmaIbProxyIPut(&scene.context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1,
                              ncclRmaOptFlagsAggregateRequests, &request),
            ncclSuccess);
  EXPECT_TRUE(g_ginPostedWrs.empty());
  auto* req = static_cast<ncclIbRequest*>(request);
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  completion.wr_id = req - scene.send.base.reqs;
  g_ginCompletions.push_back(completion);
  int done = 0;

  ASSERT_EQ(ncclRmaIbProxyTest(nullptr, request, &done), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(g_ginPostedWrs.size(), 1u);
  EXPECT_EQ(req->type, NCCL_NET_IB_REQ_UNUSED);
  EXPECT_EQ(scene.send.ginSeq.completed, req->id);
}

TEST_F(NetIbGinMicrotest, Test_AlreadyCompletedSequenceFreesWithoutPolling) {
  GinDataScene scene;
  void* request = nullptr;
  ASSERT_EQ(ncclRmaIbProxyIPut(&scene.context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1, 0,
                              &request),
            ncclSuccess);
  auto* req = static_cast<ncclIbRequest*>(request);
  scene.send.ginSeq.completed = req->id;
  g_ginPollResult = -EIO;
  int done = 0;

  ASSERT_EQ(ncclRmaIbProxyTest(nullptr, request, &done), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(req->type, NCCL_NET_IB_REQ_UNUSED);
}

TEST_F(NetIbGinMicrotest, Test_ErrorCompletionReturnsRemoteErrorWithoutCompleting) {
  GinDataScene scene;
  void* request = nullptr;
  ASSERT_EQ(ncclRmaIbProxyIPut(&scene.context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1, 0,
                              &request),
            ncclSuccess);
  auto* req = static_cast<ncclIbRequest*>(request);
  ibv_wc completion{};
  completion.status = IBV_WC_REM_ACCESS_ERR;
  completion.wr_id = req - scene.send.base.reqs;
  g_ginCompletions.push_back(completion);
  g_ncclSocketGetAddr = [](ncclSocket*, ncclSocketAddress* addr) {
    std::memset(addr, 0, sizeof(*addr));
    return ncclSuccess;
  };
  g_ncclSocketToString = [](const ncclSocketAddress*, char* line, size_t length, int) {
    std::snprintf(line, length, "test-peer");
    return line;
  };
  int done = 0;

  EXPECT_EQ(ncclRmaIbProxyTest(nullptr, request, &done), ncclRemoteError);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(scene.send.ginSeq.completed, 0u);
  ncclIbFreeRequest(req);
}

TEST_F(NetIbGinMicrotest, Test_PollFailurePropagatesWithoutCompletingRequest) {
  GinDataScene scene;
  void* request = nullptr;
  ASSERT_EQ(ncclRmaIbProxyIPut(&scene.context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1, 0,
                              &request),
            ncclSuccess);
  auto* req = static_cast<ncclIbRequest*>(request);
  g_ginPollResult = -EIO;
  int done = 0;

  EXPECT_EQ(ncclRmaIbProxyTest(nullptr, request, &done), ncclSystemError);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(scene.send.ginSeq.completed, 0u);
  ncclIbFreeRequest(req);
}

TEST_F(NetIbGinMicrotest, IFlush_UsesLocalRankRegistrationAndFlushQp) {
  GinDataScene scene;
  g_ginCreateFlushQp = [&](ncclIbRecvComm* comm) {
    EXPECT_EQ(comm, &scene.recv);
    return ncclSuccess;
  };
  void* request = nullptr;

  ASSERT_EQ(ncclRmaIbProxyIFlush(&scene.context, 0, &scene.dstHandle, 1, &request), ncclSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(g_ginPostedWrs.size(), 1u);
  EXPECT_EQ(g_ginPostedWrs[0].opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(g_ginPostedWrs[0].remoteAddress, 0x3000u);
  EXPECT_EQ(g_ginPostedWrs[0].rkey, 0x20u);
  EXPECT_EQ(g_ginPostedWrs[0].sge.addr, 0x9000u);
  EXPECT_NE(g_ginPostedWrs[0].sendFlags & IBV_SEND_SIGNALED, 0);
}

TEST_F(NetIbGinMicrotest, IFlush_CompletionAdvancesReceiveSequenceAndFreesRequest) {
  GinDataScene scene;
  g_ginCreateFlushQp = [](ncclIbRecvComm*) { return ncclSuccess; };
  void* request = nullptr;
  ASSERT_EQ(ncclRmaIbProxyIFlush(&scene.context, 0, &scene.dstHandle, 1, &request), ncclSuccess);
  auto* req = static_cast<ncclIbRequest*>(request);
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  completion.wr_id = req - scene.recv.base.reqs;
  g_ginCompletions.push_back(completion);
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
  g_ginPostResult = EIO;
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
  g_ginPostResult = EIO;
  {
    GinDataScene scene;
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
    g_ginCreateFlushQp = [](ncclIbRecvComm*) { return ncclSuccess; };
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
  g_ginNetRegMr = [](void*, void*, size_t, int, void**) { return ncclSystemError; };

  EXPECT_EQ(ncclGinIbP2PBarrier(&coll), ncclSystemError);
  EXPECT_TRUE(g_ginAllocations.empty());
}

// AICOMRCCL-3094: the local all-gather contribution should not invoke memcpy
// with identical source and destination ranges.
TEST_F(NetIbGinMicrotest, DISABLED_P2PBarrier_LocalContributionAvoidsOverlappingMemcpy) {
  ncclGinIbCollComm coll{};
  coll.rank = 0;
  coll.nranks = 1;
  int registration = 1;
  g_ginNetRegMr = [&](void*, void*, size_t, int, void** handle) {
    *handle = &registration;
    return ncclSuccess;
  };
  g_ginNetDeregMr = [](void*, void*) { return ncclSuccess; };

  ASSERT_EQ(ncclGinIbP2PBarrier(&coll), ncclSuccess);
  EXPECT_EQ(g_ginSelfCopies, 0);
}

// AICOMRCCL-3094: the backend context is owned by DestroyContext,
// just like its child arrays and peer communications.
TEST_F(NetIbGinMicrotest, DISABLED_DestroyContext_FreesTopLevelContext) {
  ncclRmaIbProxyCtx* context = nullptr;
  ASSERT_EQ(TestGinCalloc(&context, 1), ncclSuccess);
  context->nContexts = 1;
  context->nranks = 0;

  ASSERT_EQ(ncclRmaIbProxyDestroyContext(context), ncclSuccess);
  EXPECT_TRUE(g_ginAllocations.empty());
}

// AICOMRCCL-3094: disabling the explicit flush must not leave an
// already-aggregated batch indefinitely unposted.
TEST_F(NetIbGinMicrotest, DISABLED_IFlush_DisabledFlushStillDrainsAggregatedBatch) {
  GinDataScene scene;
  scene.recv.flushEnabled = false;
  void* putRequest = nullptr;
  ASSERT_EQ(ncclRmaIbProxyIPut(&scene.context, 0, 0, &scene.srcHandle, 8, 0, &scene.dstHandle, 1,
                              ncclRmaOptFlagsAggregateRequests, &putRequest),
            ncclSuccess);
  ASSERT_TRUE(g_ginPostedWrs.empty());
  void* flushRequest = reinterpret_cast<void*>(0x1);

  ASSERT_EQ(ncclRmaIbProxyIFlush(&scene.context, 0, &scene.dstHandle, 1, &flushRequest), ncclSuccess);
  EXPECT_EQ(flushRequest, nullptr);
  EXPECT_EQ(g_ginPostedWrs.size(), 1u);
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
