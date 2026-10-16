/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/coll_net.cc, driven through a fake ncclCollNet_t.
// This TU #includes the hipified coll_net.cc (COLL_NET_CC_PATH) so its file-static helpers
// are reachable. No GPU, no librccl.so, no network.

#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "ScopedHook.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/rocmwrap_fakes.h"
#include "fakes/topo_net_stubs.h"
#include "fakes/topo_stubs.h"

// Routes every NCCL_PARAM / RCCL_PARAM in coll_net.cc through g_loadParam on each call.
#include "fakes/param_redirect.h"

#include COLL_NET_CC_PATH

// init.cc data symbol: a non-null ncclGdrCopy selects coll_net.cc's GDRCopy arms.
gdr_t ncclGdrCopy = nullptr;

namespace {

// Only listen takes a context, so the trampolines reach the fake through the instance
// the fixture installs.
struct FakeCollNet;
FakeCollNet* g_activeCollNet = nullptr;

struct FakeCollNet {
  ncclNetProperties_t props = DefaultProps();
  ncclResult_t propsResult = ncclSuccess;
  int lastPropsDev = -1;

  ncclResult_t listenResult = ncclSuccess;
  int listenCalls = 0;
  int lastListenDev = -1;
  void* lastListenCtx = nullptr;
  char listenComm = 0;  // its address is the listen comm handed out

  // connect: hands out &collComm unless connectResult fails.
  ncclResult_t connectResult = ncclSuccess;
  int connectCalls = 0;
  std::vector<void*> lastHandles;
  int lastConnectRanks = -1, lastConnectRank = -1;
  void* lastConnectListen = nullptr;
  char collComm = 0;

  ncclResult_t closeListenResult = ncclSuccess, closeCollResult = ncclSuccess;
  std::vector<void*> closedListens, closedColls;

  struct MrCall {
    void* comm;
    void* data;
    size_t size;
    int type;
    uint64_t offset;
    int fd;
  };
  std::vector<MrCall> regMrs, regMrDmaBufs;
  ncclResult_t regMrResult = ncclSuccess, regMrDmaBufResult = ncclSuccess;
  std::vector<std::pair<void*, void*>> deregMrs;  // (comm, mhandle)
  ncclResult_t deregMrResult = ncclSuccess;

  static ncclNetProperties_t DefaultProps() {
    ncclNetProperties_t p{};
    p.ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;
    p.maxCollBytes = MAX_NET_SIZE;
    return p;
  }

  ncclResult_t getProperties(int dev, ncclNetProperties_t* out) {
    lastPropsDev = dev;
    *out = props;
    return propsResult;
  }
  ncclResult_t listen(void* ctx, int dev, void* handle, void** comm) {
    ++listenCalls;
    lastListenCtx = ctx;
    lastListenDev = dev;
    if (listenResult != ncclSuccess) return listenResult;
    memset(handle, 0x5A, NCCL_NET_HANDLE_MAXSIZE);
    *comm = &listenComm;
    return ncclSuccess;
  }
  ncclResult_t connect(void* handles[], int nranks, int rank, void* listen, void** comm) {
    ++connectCalls;
    lastHandles.assign(handles, handles + nranks);
    lastConnectRanks = nranks;
    lastConnectRank = rank;
    lastConnectListen = listen;
    if (connectResult != ncclSuccess) return connectResult;
    *comm = &collComm;
    return ncclSuccess;
  }
  ncclResult_t closeListen(void* comm) {
    closedListens.push_back(comm);
    return closeListenResult;
  }
  ncclResult_t closeColl(void* comm) {
    closedColls.push_back(comm);
    return closeCollResult;
  }

  // An mhandle is the registered address, so tests can tell which buffer it came from.
  ncclResult_t regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
    regMrs.push_back({comm, data, size, type, 0, -1});
    *mhandle = data;
    return regMrResult;
  }
  ncclResult_t regMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
    regMrDmaBufs.push_back({comm, data, size, type, offset, fd});
    *mhandle = data;
    return regMrDmaBufResult;
  }
  ncclResult_t deregMr(void* comm, void* mhandle) {
    deregMrs.push_back({comm, mhandle});
    return deregMrResult;
  }

  ncclCollNet_t vtable() {
    ncclCollNet_t v{};
    v.name = "FakeCollNet";
    v.getProperties = &FakeCollNet::TrampGetProperties;
    v.listen = &FakeCollNet::TrampListen;
    v.connect = &FakeCollNet::TrampConnect;
    v.closeListen = &FakeCollNet::TrampCloseListen;
    v.closeColl = &FakeCollNet::TrampCloseColl;
    v.regMr = &FakeCollNet::TrampRegMr;
    v.regMrDmaBuf = &FakeCollNet::TrampRegMrDmaBuf;
    v.deregMr = &FakeCollNet::TrampDeregMr;
    return v;
  }

 private:
  static ncclResult_t TrampGetProperties(int dev, ncclNetProperties_t* props) {
    return g_activeCollNet->getProperties(dev, props);
  }
  static ncclResult_t TrampListen(void* ctx, int dev, void* handle, void** comm) {
    return g_activeCollNet->listen(ctx, dev, handle, comm);
  }
  static ncclResult_t TrampConnect(void* handles[], int nranks, int rank, void* listen, void** comm) {
    return g_activeCollNet->connect(handles, nranks, rank, listen, comm);
  }
  static ncclResult_t TrampCloseListen(void* comm) { return g_activeCollNet->closeListen(comm); }
  static ncclResult_t TrampCloseColl(void* comm) { return g_activeCollNet->closeColl(comm); }
  static ncclResult_t TrampRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
    return g_activeCollNet->regMr(comm, data, size, type, mhandle);
  }
  static ncclResult_t TrampRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd,
                                       void** mhandle) {
    return g_activeCollNet->regMrDmaBuf(comm, data, size, type, offset, fd, mhandle);
  }
  static ncclResult_t TrampDeregMr(void* comm, void* mhandle) { return g_activeCollNet->deregMr(comm, mhandle); }
};

// Every device and host buffer goes through this ledger, so TearDown releases exactly
// what production left allocated.
class CollNetMicrotestBase : public ::testing::Test {
 protected:
  FakeCollNet net_;
  ncclCollNet_t vtable_{};
  ncclProxyState proxyState_{};
  std::set<void*> live_;
  int hostAllocs_ = 0, devAllocs_ = 0, hostFrees_ = 0, devFrees_ = 0;
  unsigned lastDevFlags_ = 0;

  void* Alloc(size_t n) {
    void* p = calloc(1, n);
    live_.insert(p);
    return p;
  }
  void Release(void* p) {
    if (live_.erase(p)) free(p);
  }

  void SetUp() override {
    g_activeCollNet = &net_;
    // A healthy device for alloc.h's calloc/free paths: capture-mode swaps, side streams,
    // memsets and syncs succeed, and every allocation is its own base address.
    g_hipAsyncOpsResult = hipSuccess;
    g_hipStreamCreateResult = hipSuccess;
    g_hipStreamSynchronize = [](hipStream_t) { return hipSuccess; };
    g_hipMemGetAddressRange = [](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
      *base = ptr;
      *size = 0;
      return hipSuccess;
    };
    g_hipHostMalloc = [this](void** p, size_t n, unsigned) {
      ++hostAllocs_;
      *p = Alloc(n);
      return hipSuccess;
    };
    g_hipExtMallocWithFlags = [this](void** p, size_t n, unsigned flags) {
      ++devAllocs_;
      lastDevFlags_ = flags;
      *p = Alloc(n);
      return hipSuccess;
    };
    g_hipHostFree = [this](void* p) {
      ++hostFrees_;
      Release(p);
      return hipSuccess;
    };
    g_hipFree = [this](void* p) {
      ++devFrees_;
      Release(p);
      return hipSuccess;
    };
    vtable_ = net_.vtable();
    proxyState_.ncclCollNet = &vtable_;
    proxyState_.collNetContext = &net_;
  }

  void TearDown() override {
    for (void* p : live_) free(p);
    g_activeCollNet = nullptr;
    ncclGdrCopy = nullptr;
    ResetNcclFakes();
    ResetHipFakes();
    ResetTopoStubs();
  }
};

// ---------------------------------------------------------------------------
// Helpers (canConnect, getHandleForAddressRangeFlags, collNetDumpMap)
// ---------------------------------------------------------------------------

TEST(CollNetHelpersMicrotest, CanConnect_AnyPeers_ReportsNotConnectable) {
  int ret = 1;

  EXPECT_EQ(canConnect(&ret, nullptr, nullptr, nullptr, nullptr), ncclSuccess);
  EXPECT_EQ(ret, 0);
}

TEST(CollNetHelpersMicrotest, HandleForAddressRangeFlags_OnlyPciGdrCanForcePcieMapping) {
#if CUDA_VERSION >= 12080 || NCCL_CUMEM_DMABUF_EXPORT_GATE
  EXPECT_EQ(getHandleForAddressRangeFlags(ncclTopoGdrModePci), hipMemRangeFlagDmaBufMappingTypePcie);
#else
  EXPECT_EQ(getHandleForAddressRangeFlags(ncclTopoGdrModePci), 0);
#endif
  EXPECT_EQ(getHandleForAddressRangeFlags(ncclTopoGdrModeDefault), 0);
  EXPECT_EQ(getHandleForAddressRangeFlags(ncclTopoGdrModeDisable), 0);
}

TEST(CollNetHelpersMicrotest, DumpMap_PrintsEveryBankAndOffset) {
  connectMap map{};
  char host[64], dev[8], sharedHost[8], sharedDev[8];
  map.mems[NCCL_NET_MAP_HOSTMEM] = {host, host + 1, 0};
  map.mems[NCCL_NET_MAP_DEVMEM] = {dev, dev + 1, 0x8};
  map.mems[NCCL_NET_MAP_SHARED_HOSTMEM] = {sharedHost, sharedHost + 1, 0x20};
  map.mems[NCCL_NET_MAP_SHARED_DEVMEM] = {sharedDev, sharedDev + 1, 0x40};
  NCCL_NET_MAP_ADD_POINTER(&map, 0, 0, 16, sendMem);
  NCCL_NET_MAP_ADD_POINTER(&map, 0, 0, 32, recvMem);
  NCCL_NET_MAP_ADD_POINTER(&map, 1, 1, 0x40, buffs[NCCL_PROTO_SIMPLE]);
  testing::internal::CaptureStdout();

  EXPECT_EQ(collNetDumpMap(&map), ncclSuccess);

  char want[1024];
  void *nil = nullptr, *h = host, *h1 = host + 1;
  snprintf(want, sizeof(want),
           "Dump map\n"
           "Mem 0: Host mem (30 B) CPU %p GPU %p\n"
           "Mem 1: Vid  mem CPU (8 B) %p GPU %p\n"
           "Mem 2: Shared Host mem (20 B) CPU %p GPU %p\n"
           "Mem 3: Shared Vid  (40 B) mem CPU %p GPU %p\n"
           "SendMem -> Used 1 Bank 0 Offset 0, cpu %p gpu %p\n"
           "RecvMem -> Used 1 Bank 0 Offset 10, cpu %p gpu %p\n"
           "Proto 0 -> Used 0 Bank 0 Offset 0, cpu %p, gpu %p\n"
           "Proto 1 -> Used 0 Bank 0 Offset 0, cpu %p, gpu %p\n"
           "Proto 2 -> Used 1 Bank 3 Offset 0, cpu %p, gpu %p\n"
           "End of dump\n",
           h1, h, (void*)(dev + 1), (void*)dev, (void*)(sharedHost + 1), (void*)sharedHost, (void*)(sharedDev + 1),
           (void*)sharedDev, h1, h, (void*)(host + 17), (void*)(host + 16), nil, nil, nil, nil, (void*)(sharedDev + 1),
           (void*)sharedDev);
  EXPECT_EQ(testing::internal::GetCapturedStdout(), want);
}

// ---------------------------------------------------------------------------
// Proxy-side setup (sendProxySetup, recvProxySetup)
// ---------------------------------------------------------------------------

class CollNetProxySetupMicrotest : public CollNetMicrotestBase {
 protected:
  ncclProxyConnection conn_{};
  ncclCollNetSharedRes shared_{};
  setupReq req_{};
  collNetHandle_t handle_{};
  int done_ = 0;

  void SetUp() override {
    CollNetMicrotestBase::SetUp();
    req_.netDev = 2;
    req_.collNet = &shared_;
  }

  void TearDown() override {
    free(conn_.transportResources);
    free(shared_.resources);
    CollNetMicrotestBase::TearDown();
  }

  ncclResult_t SendSetup(int reqSize = sizeof(setupReq)) {
    return sendProxySetup(&conn_, &proxyState_, &req_, reqSize, nullptr, 0, &done_);
  }
  ncclResult_t RecvSetup(int reqSize = sizeof(setupReq), int respSize = sizeof(collNetHandle_t)) {
    return recvProxySetup(&conn_, &proxyState_, &req_, reqSize, &handle_, respSize, &done_);
  }
  sendResources* SendRes() { return static_cast<sendResources*>(conn_.transportResources); }
  recvResources* RecvRes() { return static_cast<recvResources*>(conn_.transportResources); }
  void ResetConn() {
    free(conn_.transportResources);
    conn_ = {};
  }

  // Runs one side's setup on a fresh connection and reports its DMA-BUF choice.
  int UseDmaBuf(bool send) {
    ResetConn();
    EXPECT_EQ(send ? SendSetup() : RecvSetup(), ncclSuccess);
    if (!conn_.transportResources) return -1;
    return send ? SendRes()->useDmaBuf : RecvRes()->useDmaBuf;
  }
};

TEST_F(CollNetProxySetupMicrotest, SendSetup_ValidRequest_RecordsRequestAndCollLimit) {
  req_.useGdr = ncclTopoGdrModePci;
  net_.props.maxCollBytes = 1 << 20;

  ASSERT_EQ(SendSetup(), ncclSuccess);

  EXPECT_EQ(net_.lastPropsDev, 2);
  EXPECT_EQ(conn_.shared, 1);
  EXPECT_EQ(conn_.collNet, &shared_);
  sendResources* r = SendRes();
  ASSERT_NE(r, nullptr);
  EXPECT_EQ(r->netDev, 2);
  EXPECT_EQ(r->useGdr, ncclTopoGdrModePci);
  EXPECT_EQ(r->maxCollBytes, (size_t)1 << 20);
  EXPECT_EQ(net_.listenCalls, 0);
}

TEST_F(CollNetProxySetupMicrotest, RecvSetup_ValidRequest_RecordsRequestAndListens) {
  req_.useGdr = ncclTopoGdrModeDefault;
  req_.needFlush = ncclTopoFlushAlways;
  net_.props.maxCollBytes = 1 << 20;

  ASSERT_EQ(RecvSetup(), ncclSuccess);

  EXPECT_EQ(net_.lastPropsDev, 2);
  EXPECT_EQ(conn_.shared, 1);
  EXPECT_EQ(conn_.collNet, &shared_);
  recvResources* r = RecvRes();
  ASSERT_NE(r, nullptr);
  EXPECT_EQ(r->netDev, 2);
  EXPECT_EQ(r->useGdr, ncclTopoGdrModeDefault);
  EXPECT_EQ(r->needFlush, ncclTopoFlushAlways);
  EXPECT_EQ(r->maxCollBytes, (size_t)1 << 20);
  EXPECT_EQ(net_.listenCalls, 1);
  EXPECT_EQ(net_.lastListenDev, 2);
  EXPECT_EQ(net_.lastListenCtx, &net_);
  EXPECT_EQ(handle_[0], 0x5A);  // the plugin's listen handle is the response
}

TEST_F(CollNetProxySetupMicrotest, Setup_WrongRequestSize_RejectedWithoutResources) {
  EXPECT_EQ(SendSetup(sizeof(setupReq) - 1), ncclInternalError);
  EXPECT_EQ(RecvSetup(sizeof(setupReq) + 1), ncclInternalError);
  EXPECT_EQ(conn_.transportResources, nullptr);
}

TEST_F(CollNetProxySetupMicrotest, Setup_PropertiesFail_ReturnsError) {
  net_.propsResult = ncclSystemError;

  EXPECT_EQ(SendSetup(), ncclSystemError);
  ResetConn();
  EXPECT_EQ(RecvSetup(), ncclSystemError);
  EXPECT_EQ(net_.listenCalls, 0);
}

TEST_F(CollNetProxySetupMicrotest, Setup_InvalidMaxCollBytes_Rejected) {
  for (size_t bad : {(size_t)0, (size_t)NCCL_MAX_NET_SIZE_BYTES + 1}) {
    SCOPED_TRACE(bad);
    net_.props.maxCollBytes = bad;
    ResetConn();
    EXPECT_EQ(SendSetup(), ncclInternalError);
    ResetConn();
    EXPECT_EQ(RecvSetup(), ncclInternalError);
  }
  EXPECT_EQ(net_.listenCalls, 0);
}

TEST_F(CollNetProxySetupMicrotest, Setup_MaxCollBytesAtLimit_Accepted) {
  net_.props.maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;

  EXPECT_EQ(SendSetup(), ncclSuccess);
  ResetConn();
  EXPECT_EQ(RecvSetup(), ncclSuccess);
}

TEST_F(CollNetProxySetupMicrotest, RecvSetup_WrongResponseSize_RejectedBeforeListening) {
  EXPECT_EQ(RecvSetup(sizeof(setupReq), sizeof(collNetHandle_t) - 1), ncclInternalError);
  EXPECT_EQ(net_.listenCalls, 0);
}

TEST_F(CollNetProxySetupMicrotest, RecvSetup_ListenFails_ReturnsError) {
  net_.listenResult = ncclSystemError;

  EXPECT_EQ(RecvSetup(), ncclSystemError);
}

TEST_F(CollNetProxySetupMicrotest, Setup_GdrProxyAndPluginSupportDmaBuf_UsesDmaBuf) {
  req_.useGdr = ncclTopoGdrModeDefault;
  proxyState_.dmaBufSupport = true;
  net_.props.ptrSupport = NCCL_PTR_CUDA | NCCL_PTR_DMABUF;

  EXPECT_EQ(UseDmaBuf(true), 1);
  EXPECT_EQ(UseDmaBuf(false), 1);
}

TEST_F(CollNetProxySetupMicrotest, Setup_AnyDmaBufPrerequisiteMissing_SkipsDmaBuf) {
  req_.useGdr = ncclTopoGdrModeDisable;
  proxyState_.dmaBufSupport = true;
  net_.props.ptrSupport = NCCL_PTR_CUDA | NCCL_PTR_DMABUF;
  EXPECT_EQ(UseDmaBuf(true), 0);
  EXPECT_EQ(UseDmaBuf(false), 0);

  req_.useGdr = ncclTopoGdrModeDefault;
  proxyState_.dmaBufSupport = false;
  EXPECT_EQ(UseDmaBuf(true), 0);
  EXPECT_EQ(UseDmaBuf(false), 0);

  proxyState_.dmaBufSupport = true;
  net_.props.ptrSupport = NCCL_PTR_CUDA;
  EXPECT_EQ(UseDmaBuf(true), 0);
  EXPECT_EQ(UseDmaBuf(false), 0);
}

// ---------------------------------------------------------------------------
// Shared collective comms (sharedListen, sharedConnect, sharedFree)
// ---------------------------------------------------------------------------

class CollNetSharedCommMicrotest : public CollNetMicrotestBase {
 protected:
  static constexpr int kDev = 3;
  static constexpr int kRanks = 4;
  ncclCollNetSharedRes shared_{};
  collNetHandle_t handle_{};
  ncclConnect connectInfos_[kRanks]{};
  void* comm_ = reinterpret_cast<void*>(0x1);

  void TearDown() override {
    free(shared_.resources);
    CollNetMicrotestBase::TearDown();
  }

  sharedResources* Res() { return static_cast<sharedResources*>(shared_.resources); }
  ncclResult_t Listen(int dev = kDev) { return sharedListen(&proxyState_, dev, &shared_, &handle_); }
  ncclResult_t Connect(int dev = kDev, int rank = 1) {
    return sharedConnect(&proxyState_, dev, connectInfos_, kRanks, rank, &shared_, &comm_);
  }
};

TEST_F(CollNetSharedCommMicrotest, Listen_FirstUse_CreatesTableAndListensOnDevice) {
  ASSERT_EQ(Listen(), ncclSuccess);

  ASSERT_NE(Res(), nullptr);
  EXPECT_EQ(net_.listenCalls, 1);
  EXPECT_EQ(net_.lastListenCtx, &net_);
  EXPECT_EQ(net_.lastListenDev, kDev);
  EXPECT_EQ(Res()->collNetListenComms[kDev], &net_.listenComm);
  EXPECT_EQ(handle_[0], 0x5A);
}

TEST_F(CollNetSharedCommMicrotest, Listen_SecondDevice_ReusesTable) {
  ASSERT_EQ(Listen(kDev), ncclSuccess);
  sharedResources* first = Res();

  ASSERT_EQ(Listen(kDev + 1), ncclSuccess);

  EXPECT_EQ(Res(), first);
  EXPECT_EQ(net_.listenCalls, 2);
  EXPECT_EQ(Res()->collNetListenComms[kDev + 1], &net_.listenComm);
}

TEST_F(CollNetSharedCommMicrotest, Listen_DeviceAlreadyConnected_SkipsListen) {
  ASSERT_EQ(Listen(), ncclSuccess);
  Res()->collNetComms[kDev] = &net_.collComm;

  ASSERT_EQ(Listen(), ncclSuccess);

  EXPECT_EQ(net_.listenCalls, 1);
}

TEST_F(CollNetSharedCommMicrotest, Listen_PluginFails_ReturnsError) {
  net_.listenResult = ncclSystemError;

  EXPECT_EQ(Listen(), ncclSystemError);
}

TEST_F(CollNetSharedCommMicrotest, Connect_FirstUse_HandsEveryRanksHandleToPluginAndClosesListen) {
  ASSERT_EQ(Listen(), ncclSuccess);

  ASSERT_EQ(Connect(kDev, /*rank=*/2), ncclSuccess);

  EXPECT_EQ(net_.connectCalls, 1);
  ASSERT_EQ(net_.lastHandles.size(), (size_t)kRanks);
  for (int i = 0; i < kRanks; i++) {
    EXPECT_EQ(net_.lastHandles[i], &reinterpret_cast<collNetRecvConnectInfo*>(connectInfos_ + i)->collNetHandle);
  }
  EXPECT_EQ(net_.lastConnectRanks, kRanks);
  EXPECT_EQ(net_.lastConnectRank, 2);
  EXPECT_EQ(net_.lastConnectListen, &net_.listenComm);
  EXPECT_EQ(net_.closedListens, std::vector<void*>{&net_.listenComm});
  EXPECT_EQ(comm_, &net_.collComm);
  EXPECT_EQ(Res()->collNetComms[kDev], &net_.collComm);
  EXPECT_EQ(Res()->commRefCount[kDev], 1);
}

TEST_F(CollNetSharedCommMicrotest, Connect_AlreadyConnected_ReusesCommAndCountsReference) {
  ASSERT_EQ(Listen(), ncclSuccess);
  ASSERT_EQ(Connect(), ncclSuccess);
  comm_ = nullptr;

  ASSERT_EQ(Connect(), ncclSuccess);

  EXPECT_EQ(net_.connectCalls, 1);
  EXPECT_EQ(comm_, &net_.collComm);
  EXPECT_EQ(Res()->commRefCount[kDev], 2);
}

TEST_F(CollNetSharedCommMicrotest, Connect_PluginFails_ReturnsNoCommAndDropsListen) {
  ASSERT_EQ(Listen(), ncclSuccess);
  net_.connectResult = ncclSystemError;

  ASSERT_EQ(Connect(), ncclSuccess);

  EXPECT_EQ(comm_, nullptr);
  EXPECT_EQ(Res()->collNetListenComms[kDev], nullptr);
  EXPECT_EQ(Res()->commRefCount[kDev], 0);
}

TEST_F(CollNetSharedCommMicrotest, Connect_CloseListenFails_ReturnsError) {
  ASSERT_EQ(Listen(), ncclSuccess);
  net_.closeListenResult = ncclSystemError;

  EXPECT_EQ(Connect(), ncclSystemError);
}

TEST_F(CollNetSharedCommMicrotest, Free_OtherUsersRemain_KeepsComm) {
  ASSERT_EQ(Listen(), ncclSuccess);
  ASSERT_EQ(Connect(), ncclSuccess);
  ASSERT_EQ(Connect(), ncclSuccess);

  ASSERT_EQ(sharedFree(&proxyState_, &shared_, kDev), ncclSuccess);

  EXPECT_TRUE(net_.closedColls.empty());
  ASSERT_NE(Res(), nullptr);
  EXPECT_EQ(Res()->commRefCount[kDev], 1);
}

TEST_F(CollNetSharedCommMicrotest, Free_LastUser_ClosesCommAndReleasesTable) {
  ASSERT_EQ(Listen(), ncclSuccess);
  ASSERT_EQ(Connect(), ncclSuccess);

  ASSERT_EQ(sharedFree(&proxyState_, &shared_, kDev), ncclSuccess);

  EXPECT_EQ(net_.closedColls, std::vector<void*>{&net_.collComm});
  EXPECT_EQ(shared_.resources, nullptr);
}

TEST_F(CollNetSharedCommMicrotest, Free_OtherDeviceStillConnected_KeepsTable) {
  ASSERT_EQ(Listen(kDev), ncclSuccess);
  ASSERT_EQ(Connect(kDev), ncclSuccess);
  ASSERT_EQ(Listen(kDev + 1), ncclSuccess);
  ASSERT_EQ(Connect(kDev + 1), ncclSuccess);

  ASSERT_EQ(sharedFree(&proxyState_, &shared_, kDev), ncclSuccess);

  EXPECT_EQ(net_.closedColls.size(), 1u);
  ASSERT_NE(Res(), nullptr);
  EXPECT_EQ(Res()->commRefCount[kDev + 1], 1);
}

TEST_F(CollNetSharedCommMicrotest, Free_CloseFails_ReturnsError) {
  ASSERT_EQ(Listen(), ncclSuccess);
  ASSERT_EQ(Connect(), ncclSuccess);
  net_.closeCollResult = ncclSystemError;

  EXPECT_EQ(sharedFree(&proxyState_, &shared_, kDev), ncclSystemError);
}

// ---------------------------------------------------------------------------
// Shared buffers (sharedBuffersInit, sharedBuffersDestroy)
// ---------------------------------------------------------------------------

class CollNetSharedBuffersMicrotest : public CollNetMicrotestBase {
 protected:
  static constexpr int kChannels = 2;
  static constexpr int kBuffSize = 1 << 12;
  ncclCollNetSharedRes shared_{};
  char *gpu_ = nullptr, *cpu_ = nullptr;
  int size_ = -1;

  void SetUp() override {
    CollNetMicrotestBase::SetUp();
    shared_.nChannels = kChannels;
    shared_.buffSize = kBuffSize;
  }

  ncclResult_t Init(int cuda) { return sharedBuffersInit(&shared_, cuda, &gpu_, &cpu_, &size_, nullptr); }
};

TEST_F(CollNetSharedBuffersMicrotest, Init_FirstHostUse_SizesPoolForBothDirections) {
  ASSERT_EQ(Init(/*cuda=*/0), ncclSuccess);

  EXPECT_EQ(size_, 2 * kChannels * kBuffSize);
  EXPECT_EQ(shared_.size, size_);
  EXPECT_EQ(hostAllocs_, 1);
  EXPECT_EQ(devAllocs_, 0);
  ASSERT_NE(shared_.hostBuff, nullptr);
  EXPECT_EQ(cpu_, shared_.hostBuff);
  EXPECT_EQ(gpu_, shared_.hostBuff);
}

TEST_F(CollNetSharedBuffersMicrotest, Init_FirstDeviceUse_AllocatesFineGrainedDeviceMemory) {
  ASSERT_EQ(Init(/*cuda=*/1), ncclSuccess);

  EXPECT_EQ(devAllocs_, 1);
  EXPECT_EQ(hostAllocs_, 0);
#if defined(HIP_UNCACHED_MEMORY)
  EXPECT_EQ(lastDevFlags_, (unsigned)hipDeviceMallocUncached);
#else
  EXPECT_EQ(lastDevFlags_, (unsigned)hipDeviceMallocFinegrained);
#endif
  ASSERT_NE(shared_.cudaBuff, nullptr);
  EXPECT_EQ(cpu_, shared_.cudaBuff);
  EXPECT_EQ(gpu_, shared_.cudaBuff);
}

TEST_F(CollNetSharedBuffersMicrotest, Init_LaterUses_ReuseEachBufferAtTheFirstSize) {
  ASSERT_EQ(Init(1), ncclSuccess);
  char* dev = cpu_;
  shared_.nChannels = 8;

  ASSERT_EQ(Init(1), ncclSuccess);
  EXPECT_EQ(cpu_, dev);
  ASSERT_EQ(Init(0), ncclSuccess);
  char* host = cpu_;
  ASSERT_EQ(Init(0), ncclSuccess);
  EXPECT_EQ(cpu_, host);

  EXPECT_EQ(devAllocs_, 1);
  EXPECT_EQ(hostAllocs_, 1);
  EXPECT_EQ(size_, 2 * kChannels * kBuffSize);
  EXPECT_EQ(cpu_, shared_.hostBuff);
}

TEST_F(CollNetSharedBuffersMicrotest, Init_AllocationFails_ReturnsError) {
  g_hipExtMallocWithFlags = [](void**, size_t, unsigned) { return hipErrorOutOfMemory; };
  g_hipHostMalloc = [](void**, size_t, unsigned) { return hipErrorOutOfMemory; };

  EXPECT_NE(Init(1), ncclSuccess);
  EXPECT_NE(Init(0), ncclSuccess);
}

TEST_F(CollNetSharedBuffersMicrotest, Destroy_NeverSized_FreesNothing) {
  ASSERT_EQ(sharedBuffersDestroy(&shared_, &proxyState_), ncclSuccess);

  EXPECT_EQ(hostFrees_ + devFrees_, 0);
}

TEST_F(CollNetSharedBuffersMicrotest, Destroy_RepeatedAcrossConnections_FreesBothBuffersOnce) {
  ASSERT_EQ(Init(1), ncclSuccess);
  ASSERT_EQ(Init(0), ncclSuccess);

  ASSERT_EQ(sharedBuffersDestroy(&shared_, &proxyState_), ncclSuccess);
  ASSERT_EQ(sharedBuffersDestroy(&shared_, &proxyState_), ncclSuccess);

  EXPECT_EQ(devFrees_, 1);
  EXPECT_EQ(hostFrees_, 1);
  EXPECT_EQ(shared_.size, 0);
  EXPECT_TRUE(live_.empty());
}

TEST_F(CollNetSharedBuffersMicrotest, Destroy_FreeFails_ReturnsError) {
  ASSERT_EQ(Init(1), ncclSuccess);
  ScopedHook freeFails(g_hipFree, [](void*) { return hipErrorInvalidValue; });

  EXPECT_NE(sharedBuffersDestroy(&shared_, &proxyState_), ncclSuccess);
  EXPECT_EQ(shared_.size, 2 * kChannels * kBuffSize);
}

// ---------------------------------------------------------------------------
// Client-side setup (sendSetup, recvSetup)
// ---------------------------------------------------------------------------

class CollNetClientMicrotestBase : public CollNetMicrotestBase {
 protected:
  static constexpr int kRanks = 4;
  static constexpr int kMyRank = 1;

  ncclTopoSystem* topo_ = reinterpret_cast<ncclTopoSystem*>(0x70);  // only passed through
  ncclComm comm_{};
  ncclPeerInfo peerInfo_[kRanks]{};
  int topParentLocalRanks_[kRanks] = {20, 21, 22, 23};
  ncclCollNetSharedRes* shared_ = static_cast<ncclCollNetSharedRes*>(calloc(1, sizeof(ncclCollNetSharedRes)));
  ncclConnector conn_{};
  ncclConnect connectInfo_{};

  void SetUp() override {
    CollNetMicrotestBase::SetUp();
    for (int r = 0; r < kRanks; r++) peerInfo_[r].rank = r;
    comm_.rank = kMyRank;
    comm_.nRanks = kRanks;
    comm_.localRank = 2;
    comm_.topo = topo_;
    comm_.topParentLocalRanks = topParentLocalRanks_;
    comm_.collNetSharedRes = shared_;
    comm_.ncclCollNet = &vtable_;
  }

  void TearDown() override {
    free(shared_);
    CollNetMicrotestBase::TearDown();
  }
};

class CollNetSetupMicrotest : public CollNetClientMicrotestBase {
 protected:
  static constexpr int kChannel = 3;
  ncclTopoGraph graph_{};
  ncclTopoGdrMode gdr_ = ncclTopoGdrModeDefault;
  ncclTopoFlushType flush_ = ncclTopoFlushAlways;

  ncclTopoGraph* getNetDevGraph_ = nullptr;
  int getNetDevRank_ = -1, getNetDevChannel_ = -1, getNetDevPeer_ = 0;
  int checkGdrRank_ = -1, checkGdrRead_ = -1;
  int64_t checkGdrNetId_ = -1;
  int flushCalls_ = 0, flushNetDev_ = -1, flushRank_ = -1;
  int64_t flushNetId_ = -1;
  int connectTransport_ = -1, connectSend_ = -1, connectProxyRank_ = -1;
  ncclProxyConnector* connectProxyConn_ = nullptr;
  int callType_ = -1, callRespSize_ = -1;
  int sharedRefsAtCall_ = -1;
  ncclProxyConnector* callProxyConn_ = nullptr;
  void* callResp_ = nullptr;
  setupReq req_{};

  void SetUp() override {
    CollNetClientMicrotestBase::SetUp();
    g_ncclTopoGetNetDev = [this](ncclComm*, int rank, ncclTopoGraph* graph, int channelId, int peerRank,
                                 int64_t* id, int* dev, int* proxyRank) {
      getNetDevGraph_ = graph;
      getNetDevRank_ = rank;
      getNetDevChannel_ = channelId;
      getNetDevPeer_ = peerRank;
      *id = 0x77;
      *dev = 5;
      *proxyRank = 9;
      return ncclSuccess;
    };
    g_ncclTopoCheckGdr = [this](ncclTopoSystem* topo, int rank, int64_t netId, int read, ncclTopoGdrMode* mode) {
      EXPECT_EQ(topo, topo_);
      checkGdrRank_ = rank;
      checkGdrNetId_ = netId;
      checkGdrRead_ = read;
      *mode = gdr_;
      return ncclSuccess;
    };
    g_ncclTopoNeedFlush = [this](ncclComm*, int64_t netId, int netDev, int rank, bool netManaged,
                                 ncclTopoFlushType* flush) {
      ++flushCalls_;
      flushNetId_ = netId;
      flushNetDev_ = netDev;
      flushRank_ = rank;
      EXPECT_FALSE(netManaged);
      *flush = flush_;
      return ncclSuccess;
    };
    g_proxyConnect = [this](ncclComm*, int transport, int send, int proxyRank, ncclProxyConnector* proxyConn) {
      connectTransport_ = transport;
      connectSend_ = send;
      connectProxyRank_ = proxyRank;
      connectProxyConn_ = proxyConn;
      return ncclSuccess;
    };
    g_proxyCallBlocking = [this](ncclComm*, ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize,
                                 void* resp, int respSize) {
      callProxyConn_ = proxyConn;
      callType_ = type;
      callResp_ = resp;
      callRespSize_ = respSize;
      sharedRefsAtCall_ = shared_->refCount;
      EXPECT_EQ(reqSize, (int)sizeof(setupReq));
      memcpy(&req_, reqBuff, sizeof(req_));
      if (resp) memset(resp, 0x5A, respSize);
      return ncclSuccess;
    };
  }

  ncclResult_t SendSetup() {
    return sendSetup(&comm_, &graph_, &peerInfo_[kMyRank], &peerInfo_[0], &connectInfo_, &conn_, kChannel, 1);
  }
  ncclResult_t RecvSetup() {
    return recvSetup(&comm_, &graph_, &peerInfo_[kMyRank], &peerInfo_[0], &connectInfo_, &conn_, kChannel, 1);
  }
};

TEST_F(CollNetSetupMicrotest, SendSetup_Gdr_AsksProxyToSetUpOnTopologyNicAndMarksDirectNic) {
  ASSERT_EQ(SendSetup(), ncclSuccess);

  EXPECT_EQ(getNetDevGraph_, &graph_);
  EXPECT_EQ(getNetDevRank_, kMyRank);
  EXPECT_EQ(getNetDevChannel_, kChannel);
  EXPECT_EQ(getNetDevPeer_, -1);
  EXPECT_EQ(checkGdrRank_, kMyRank);
  EXPECT_EQ(checkGdrNetId_, 0x77);
  EXPECT_EQ(checkGdrRead_, 1);
  EXPECT_EQ(flushCalls_, 0);
  EXPECT_EQ(conn_.conn.flags & NCCL_DIRECT_NIC, NCCL_DIRECT_NIC);
  EXPECT_EQ(conn_.proxyConn.tpLocalRank, 22);
  EXPECT_EQ(connectTransport_, TRANSPORT_COLLNET);
  EXPECT_EQ(connectSend_, 1);
  EXPECT_EQ(connectProxyRank_, kMyRank);
  EXPECT_EQ(connectProxyConn_, &conn_.proxyConn);
  EXPECT_EQ(callProxyConn_, &conn_.proxyConn);
  EXPECT_EQ(callType_, ncclProxyMsgSetup);
  EXPECT_EQ(callResp_, nullptr);
  EXPECT_EQ(callRespSize_, 0);
  EXPECT_EQ(req_.netDev, 5);
  EXPECT_EQ(req_.useGdr, ncclTopoGdrModeDefault);
  EXPECT_EQ(req_.collNet, shared_);
  EXPECT_EQ(sharedRefsAtCall_, 1);  // the proxy's reference is taken before it can see the request
}

TEST_F(CollNetSetupMicrotest, RecvSetup_Gdr_AsksTopologyToFlushAndReturnsListenHandle) {
  flush_ = ncclTopoFlushAlways;

  ASSERT_EQ(RecvSetup(), ncclSuccess);

  EXPECT_EQ(checkGdrRead_, 0);
  EXPECT_EQ(flushCalls_, 1);
  EXPECT_EQ(flushNetId_, 0x77);
  EXPECT_EQ(flushNetDev_, 5);
  EXPECT_EQ(flushRank_, kMyRank);
  EXPECT_EQ(conn_.conn.flags & NCCL_DIRECT_NIC, NCCL_DIRECT_NIC);
  EXPECT_EQ(conn_.proxyConn.tpLocalRank, 22);
  EXPECT_EQ(connectTransport_, TRANSPORT_COLLNET);
  EXPECT_EQ(connectSend_, 0);
  EXPECT_EQ(connectProxyRank_, kMyRank);
  EXPECT_EQ(callType_, ncclProxyMsgSetup);
  EXPECT_EQ(req_.netDev, 5);
  EXPECT_EQ(req_.needFlush, ncclTopoFlushAlways);
  EXPECT_EQ(req_.collNet, shared_);
  EXPECT_EQ(sharedRefsAtCall_, 1);
  auto* info = reinterpret_cast<collNetRecvConnectInfo*>(&connectInfo_);
  EXPECT_EQ(callResp_, &info->collNetHandle);
  EXPECT_EQ(callRespSize_, (int)sizeof(collNetHandle_t));
  EXPECT_EQ((unsigned char)info->collNetHandle[0], 0x5A);
}

TEST_F(CollNetSetupMicrotest, Setup_NoGdr_LeavesDirectNicClearAndSkipsFlushQuery) {
  gdr_ = ncclTopoGdrModeDisable;

  ASSERT_EQ(SendSetup(), ncclSuccess);
  EXPECT_EQ(conn_.conn.flags & NCCL_DIRECT_NIC, 0);
  conn_ = {};
  ASSERT_EQ(RecvSetup(), ncclSuccess);
  EXPECT_EQ(conn_.conn.flags & NCCL_DIRECT_NIC, 0);
  EXPECT_EQ(flushCalls_, 0);
  EXPECT_EQ(req_.needFlush, ncclTopoFlushNone);
}

TEST_F(CollNetSetupMicrotest, Setup_EachConnectionTakesSharedResourceReference) {
  ASSERT_EQ(SendSetup(), ncclSuccess);
  ASSERT_EQ(RecvSetup(), ncclSuccess);

  EXPECT_EQ(shared_->refCount, 2);
}

TEST_F(CollNetSetupMicrotest, Setup_DependencyFails_ReturnsError) {
  for (int failing = 0; failing < 5; failing++) {
    SCOPED_TRACE(failing);
    ScopedHook netDev(g_ncclTopoGetNetDev, [&, inner = g_ncclTopoGetNetDev](
                                               ncclComm* c, int r, ncclTopoGraph* g, int ch, int p, int64_t* id,
                                               int* dev, int* pr) {
      return failing == 0 ? ncclSystemError : inner(c, r, g, ch, p, id, dev, pr);
    });
    ScopedHook gdr(g_ncclTopoCheckGdr, [&, inner = g_ncclTopoCheckGdr](ncclTopoSystem* t, int r, int64_t id, int rd,
                                                                         ncclTopoGdrMode* m) {
      return failing == 1 ? ncclSystemError : inner(t, r, id, rd, m);
    });
    ScopedHook flush(g_ncclTopoNeedFlush, [&, inner = g_ncclTopoNeedFlush](ncclComm* c, int64_t id, int d, int r,
                                                                             bool nm, ncclTopoFlushType* f) {
      return failing == 2 ? ncclSystemError : inner(c, id, d, r, nm, f);
    });
    ScopedHook connect(g_proxyConnect, [&, inner = g_proxyConnect](ncclComm* c, int t, int s, int pr,
                                                                   ncclProxyConnector* pc) {
      return failing == 3 ? ncclSystemError : inner(c, t, s, pr, pc);
    });
    ScopedHook call(g_proxyCallBlocking, [&, inner = g_proxyCallBlocking](ncclComm* c, ncclProxyConnector* pc, int t,
                                                                          void* rq, int rqs, void* rs, int rss) {
      return failing == 4 ? ncclSystemError : inner(c, pc, t, rq, rqs, rs, rss);
    });

    if (failing != 2) EXPECT_EQ(SendSetup(), ncclSystemError);
    EXPECT_EQ(RecvSetup(), ncclSystemError);
  }
}

// ---------------------------------------------------------------------------
// Client-side connect and free (sendConnect, recvConnect, sendFree, recvFree)
// ---------------------------------------------------------------------------

class CollNetConnectMicrotest : public CollNetClientMicrotestBase {
 protected:
  connectMap map_{};
  connectMap* resp_ = &map_;  // what the proxy answers the connect request with
  std::vector<char> hostMem_;
  char shared_buff_[64] = {};
  uint64_t gdcWord_ = 0;
  ncclResult_t callResult_ = ncclSuccess;
  int callType_ = -1;
  collNetConnectArgs args_{};

  void SetUp() override {
    CollNetClientMicrotestBase::SetUp();
    NCCL_NET_MAP_ADD_POINTER(&map_, 0, 0, sizeof(ncclSendMem), sendMem);
    NCCL_NET_MAP_ADD_POINTER(&map_, 0, 0, sizeof(ncclRecvMem), recvMem);
    NCCL_NET_MAP_ADD_POINTER(&map_, 1, 1, sizeof(shared_buff_), buffs[NCCL_PROTO_SIMPLE]);
    hostMem_.assign(map_.mems[NCCL_NET_MAP_HOSTMEM].size, 0);
    map_.mems[NCCL_NET_MAP_HOSTMEM].cpuPtr = map_.mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = hostMem_.data();
    map_.mems[NCCL_NET_MAP_SHARED_DEVMEM] = {shared_buff_, shared_buff_ + 1, (int)sizeof(shared_buff_)};  // {gpu, cpu}
    g_proxyCallBlocking = [this](ncclComm*, ncclProxyConnector* proxyConn, int type, void* req, int reqSize,
                                 void* resp, int respSize) {
      EXPECT_EQ(proxyConn, &conn_.proxyConn);
      callType_ = type;
      EXPECT_EQ(reqSize, (int)sizeof(collNetConnectArgs));
      memcpy(&args_, req, sizeof(args_));
      EXPECT_EQ(respSize, (int)sizeof(connectMap*));
      memcpy(resp, &resp_, sizeof(resp_));
      return callResult_;
    };
  }

  ncclResult_t SendConnect() { return sendConnect(&comm_, &connectInfo_, kRanks, kMyRank, &conn_); }
  ncclResult_t RecvConnect() { return recvConnect(&comm_, &connectInfo_, kRanks, kMyRank, &conn_); }
  void UseGdcWord() { map_.mems[NCCL_NET_MAP_GDCMEM] = {(char*)&gdcWord_, (char*)&gdcWord_, sizeof(gdcWord_)}; }

  ncclSendMem* SendMem() { return reinterpret_cast<ncclSendMem*>(hostMem_.data()); }
  ncclRecvMem* RecvMem() { return reinterpret_cast<ncclRecvMem*>(hostMem_.data() + sizeof(ncclSendMem)); }
};

TEST_F(CollNetConnectMicrotest, SendConnect_ProxyMap_WiresConnectorAndResetsFifo) {
  for (int i = 0; i < NCCL_STEPS; i++) {
    RecvMem()->connFifo[i].mode = NCCL_MODE_NORMAL;
    RecvMem()->connFifo[i].size = 456;
  }
  void* mhandle = reinterpret_cast<void*>(0x1234);
  conn_.conn.mhandles[0] = mhandle;

  ASSERT_EQ(SendConnect(), ncclSuccess);

  EXPECT_EQ(callType_, ncclProxyMsgConnect);
  EXPECT_EQ(args_.rank, kMyRank);
  EXPECT_EQ(args_.nranks, kRanks);
  EXPECT_EQ(args_.connectInfos, &connectInfo_);
  EXPECT_EQ(conn_.conn.head, &SendMem()->head);
  EXPECT_EQ(conn_.conn.tail, &RecvMem()->tail);
  EXPECT_EQ(conn_.conn.connFifo, RecvMem()->connFifo);
  for (int i = 0; i < NCCL_STEPS; i++) {
    EXPECT_EQ(RecvMem()->connFifo[i].size, -1);
    EXPECT_EQ(RecvMem()->connFifo[i].mode, NCCL_MODE_OFFSET);
  }
  EXPECT_EQ(conn_.conn.buffs[NCCL_PROTO_SIMPLE], shared_buff_);  // the GPU-side pointer
  EXPECT_EQ(conn_.conn.buffs[NCCL_PROTO_LL], nullptr);
  EXPECT_EQ(conn_.conn.buffs[NCCL_PROTO_LL128], nullptr);
  EXPECT_EQ(conn_.proxyConn.proxyProgress, sendProxyProgress);
  EXPECT_EQ(RecvMem()->flush, 0);  // nothing past the FIFO or the buffers is written
  EXPECT_EQ(conn_.conn.mhandles[0], mhandle);
}

TEST_F(CollNetConnectMicrotest, RecvConnect_ProxyMap_WiresConnectorAndKeepsFifoSizes) {
  for (int i = 0; i < NCCL_STEPS; i++) {
    RecvMem()->connFifo[i].mode = NCCL_MODE_NORMAL;
    RecvMem()->connFifo[i].size = 456;
  }
  void* mhandle = reinterpret_cast<void*>(0x1234);
  conn_.conn.mhandles[0] = mhandle;

  ASSERT_EQ(RecvConnect(), ncclSuccess);

  EXPECT_EQ(callType_, ncclProxyMsgConnect);
  EXPECT_EQ(args_.rank, kMyRank);
  EXPECT_EQ(args_.nranks, kRanks);
  EXPECT_EQ(args_.connectInfos, &connectInfo_);
  EXPECT_EQ(conn_.conn.head, &SendMem()->head);
  EXPECT_EQ(conn_.conn.tail, &RecvMem()->tail);
  EXPECT_EQ(conn_.conn.connFifo, RecvMem()->connFifo);
  for (int i = 0; i < NCCL_STEPS; i++) {
    EXPECT_EQ(RecvMem()->connFifo[i].size, 456);
    EXPECT_EQ(RecvMem()->connFifo[i].mode, NCCL_MODE_OFFSET);
  }
  EXPECT_EQ(conn_.conn.buffs[NCCL_PROTO_SIMPLE], shared_buff_);
  EXPECT_EQ(conn_.conn.buffs[NCCL_PROTO_LL], nullptr);
  EXPECT_EQ(conn_.proxyConn.proxyProgress, recvProxyProgress);
  EXPECT_EQ(RecvMem()->flush, 0);
  EXPECT_EQ(conn_.conn.mhandles[0], mhandle);
}

TEST_F(CollNetConnectMicrotest, SendConnect_GdcSyncWord_UsedAsHead) {
  UseGdcWord();

  ASSERT_EQ(SendConnect(), ncclSuccess);

  EXPECT_EQ(conn_.conn.head, &gdcWord_);
  EXPECT_EQ(conn_.conn.tail, &RecvMem()->tail);
}

TEST_F(CollNetConnectMicrotest, RecvConnect_GdcSyncWord_UsedAsTail) {
  UseGdcWord();

  ASSERT_EQ(RecvConnect(), ncclSuccess);

  EXPECT_EQ(conn_.conn.tail, &gdcWord_);
  EXPECT_EQ(conn_.conn.head, &SendMem()->head);
}

TEST_F(CollNetConnectMicrotest, Connect_ProxyReturnsNoMap_FailsSoCallerFallsBack) {
  resp_ = nullptr;

  EXPECT_EQ(SendConnect(), ncclSystemError);
  EXPECT_EQ(RecvConnect(), ncclSystemError);
  EXPECT_EQ(conn_.proxyConn.proxyProgress, nullptr);
}

TEST_F(CollNetConnectMicrotest, Connect_ProxyCallFails_ReturnsError) {
  callResult_ = ncclRemoteError;

  EXPECT_EQ(SendConnect(), ncclRemoteError);
  EXPECT_EQ(RecvConnect(), ncclRemoteError);
  EXPECT_EQ(conn_.proxyConn.proxyProgress, nullptr);
}

TEST_F(CollNetConnectMicrotest, Free_EitherSide_ReleasesNothing) {
  EXPECT_EQ(sendFree(&comm_, &conn_), ncclSuccess);
  EXPECT_EQ(recvFree(&comm_, &conn_), ncclSuccess);
  EXPECT_EQ(hostFrees_ + devFrees_, 0);
}

// ---------------------------------------------------------------------------
// Proxy-side connect and free (sendProxyConnect, recvProxyConnect, sendProxyFree, recvProxyFree)
// ---------------------------------------------------------------------------

// Each export gets a higher fd number than any before it, so a stale fd can't match by reuse.
std::vector<int> g_exportedFds;
int g_nextExportFd = 200;
hsa_status_t FakeExportDmaBuf(const void*, size_t, int* fd, uint64_t* offset) {
  int devNull = open("/dev/null", O_RDONLY);
  *fd = fcntl(devNull, F_DUPFD, g_nextExportFd++);
  close(devNull);
  *offset = 0x40;
  g_exportedFds.push_back(*fd);
  return HSA_STATUS_SUCCESS;
}
hsa_status_t FailExportDmaBuf(const void*, size_t, int*, uint64_t*) { return HSA_STATUS_ERROR; }
bool FdOpen(int fd) { return fcntl(fd, F_GETFD) != -1; }

#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
// Builds with the cuMem DMA-BUF export: each export hands back a fresh fd, recorded here.
struct CuMemExport {
  void* ptr;
  size_t size;
  unsigned long long flags;
  int fd;
};
struct CuMemExportRecorder : std::vector<CuMemExport> {
  hipError_t result = hipSuccess;
  void Install() {
    g_hipMemGetHandleForAddressRange = [this](void* handle, hipDeviceptr_t ptr, size_t size, hipMemRangeHandleType,
                                              unsigned long long flags) {
      if (result != hipSuccess) return result;
      int devNull = open("/dev/null", O_RDONLY);
      int fd = fcntl(devNull, F_DUPFD, g_nextExportFd++);  // never a number something else reuses
      close(devNull);
      *static_cast<int*>(handle) = fd;
      push_back({ptr, size, flags, fd});
      return hipSuccess;
    };
  }
  void CloseOpen() {
    for (const auto& e : *this) {
      if (FdOpen(e.fd)) close(e.fd);
    }
  }
};
#endif

// Per-side entry points, so one typed suite covers both proxy connects and frees.
struct SendSide {
  using Res = sendResources;
  static constexpr int kAppendSlot = 0;
  static ncclResult_t Connect(ncclProxyConnection* c, ncclProxyState* ps, void* req, int reqSize, void* resp,
                              int respSize, int* done) {
    return sendProxyConnect(c, ps, req, reqSize, resp, respSize, done);
  }
  static ncclResult_t Free(ncclProxyConnection* c, ncclProxyState* ps) { return sendProxyFree(c, ps); }
  static void** Mhandles(Res* r) { return r->sendMhandles; }
};
struct RecvSide {
  using Res = recvResources;
  static constexpr int kAppendSlot = 1;
  static ncclResult_t Connect(ncclProxyConnection* c, ncclProxyState* ps, void* req, int reqSize, void* resp,
                              int respSize, int* done) {
    return recvProxyConnect(c, ps, req, reqSize, resp, respSize, done);
  }
  static ncclResult_t Free(ncclProxyConnection* c, ncclProxyState* ps) { return recvProxyFree(c, ps); }
  static void** Mhandles(Res* r) { return r->mhandles; }
};

// The connection's proxy setup already ran: its resources name the NIC, and the shared
// table holds the listen comm the recv setup created.
template <typename Side>
class CollNetProxyConnectMicrotest : public CollNetMicrotestBase {
 protected:
  using Res = typename Side::Res;
  static constexpr int kDev = 2;
  static constexpr int kRanks = 4;
  static constexpr int kRank = 1;
  static constexpr int kChannels = 2;
  static constexpr int kBuffSize = 1 << 12;
  static constexpr int kPoolSize = 2 * kChannels * kBuffSize;

  ncclProxyConnection conn_{};
  ncclCollNetSharedRes* shared_ = static_cast<ncclCollNetSharedRes*>(calloc(1, sizeof(ncclCollNetSharedRes)));
  ncclConnect connectInfos_[kRanks]{};
  collNetConnectArgs args_{kRank, kRanks, connectInfos_};
  connectMap* resp_ = reinterpret_cast<connectMap*>(0x1);
  int done_ = -1;
  void* reqFifo_ = reinterpret_cast<void*>(0x4000);
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
  CuMemExportRecorder exports_;
#endif

  void SetUp() override {
    CollNetMicrotestBase::SetUp();
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
    exports_.Install();
#endif
    shared_->refCount = 2;  // this connection's and the test's, so a free never releases it unasked
    shared_->nChannels = kChannels;
    shared_->buffSize = kBuffSize;
    auto* table = static_cast<sharedResources*>(calloc(1, sizeof(sharedResources)));
    table->collNetListenComms[kDev] = &net_.listenComm;
    shared_->resources = table;
    conn_.collNet = shared_;
    NewConn();
    // What the recv side published for the send side.
    auto* info = reinterpret_cast<collNetSendConnectInfo*>(connectInfos_ + kRank);
    info->reqFifo = reqFifo_;
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) info->mhandles[p] = reinterpret_cast<void*>(0x100 + p);
    g_exportedFds.clear();
    g_nextExportFd = 200;
  }

  void TearDown() override {
    DropConn();
    if (shared_) {
      free(shared_->resources);
      free(shared_);
    }
    for (int fd : g_exportedFds) {
      if (FdOpen(fd)) close(fd);
    }
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
    exports_.CloseOpen();
#endif
    ResetRocmWrapFakes();
    CollNetMicrotestBase::TearDown();
  }

  // Fresh resources for a connection on the same shared state; keeps the GDR choice.
  void NewConn() {
    auto* r = static_cast<Res*>(calloc(1, sizeof(Res)));
    r->netDev = kDev;
    if (Res* old = R()) {
      r->useGdr = old->useGdr;
      r->useDmaBuf = old->useDmaBuf;
    }
    DropConn();
    conn_.transportResources = r;
  }
  void DropConn() {
    if (Res* r = R()) free(r->gdrDesc);
    free(conn_.transportResources);
    conn_.transportResources = nullptr;
  }

  Res* R() { return static_cast<Res*>(conn_.transportResources); }
  void UseGdr(bool dmaBuf = false) {
    R()->useGdr = ncclTopoGdrModePci;
    R()->useDmaBuf = dmaBuf;
  }
  char* Pool() { return NCCL_NET_MAP_GET_POINTER(&R()->map, cpu, buffs[NCCL_PROTO_SIMPLE]); }
  void* PoolMhandle() { return Side::Mhandles(R())[NCCL_PROTO_SIMPLE]; }

  ncclResult_t Connect(int reqSize = sizeof(collNetConnectArgs), int respSize = sizeof(connectMap*)) {
    return Side::Connect(&conn_, &proxyState_, &args_, reqSize, &resp_, respSize, &done_);
  }
  // On success the connection's resources are gone.
  ncclResult_t Free() {
    ncclResult_t ret = Side::Free(&conn_, &proxyState_);
    if (ret == ncclSuccess) conn_.transportResources = nullptr;
    return ret;
  }
};

using CollNetSides = ::testing::Types<SendSide, RecvSide>;
template <typename Side>
using CollNetProxyConnectBothMicrotest = CollNetProxyConnectMicrotest<Side>;
TYPED_TEST_SUITE(CollNetProxyConnectBothMicrotest, CollNetSides);

TYPED_TEST(CollNetProxyConnectBothMicrotest, WrongRequestSize_RejectedBeforeConnecting) {
  EXPECT_EQ(this->Connect(sizeof(collNetConnectArgs) - 1), ncclInternalError);
  EXPECT_EQ(this->net_.connectCalls, 0);
}

TYPED_TEST(CollNetProxyConnectBothMicrotest, HostBuffers_MapsControlWordsAndRegistersSharedPool) {
  ASSERT_EQ(this->Connect(), ncclSuccess);

  auto* r = this->R();
  EXPECT_EQ(this->net_.connectCalls, 1);
  EXPECT_EQ(this->net_.lastConnectRank, this->kRank);
  EXPECT_EQ(this->net_.lastConnectRanks, this->kRanks);
  EXPECT_EQ(r->collNetComm, &this->net_.collComm);
  EXPECT_EQ(r->collNetRank, this->kRank);
  EXPECT_EQ(this->conn_.proxyAppendPtr, this->shared_->proxyAppend + 2 * this->kDev + TypeParam::kAppendSlot);
  EXPECT_EQ(this->resp_, &r->map);
  // ncclSendMem then ncclRecvMem in one host allocation the GPU sees at the same address.
  char* host = r->map.mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  EXPECT_EQ((char*)r->sendMem, host);
  EXPECT_EQ((char*)r->recvMem, host + sizeof(ncclSendMem));
  EXPECT_EQ(r->map.mems[NCCL_NET_MAP_HOSTMEM].gpuPtr, host);
  EXPECT_EQ(r->map.mems[NCCL_NET_MAP_HOSTMEM].size, (int)(sizeof(ncclSendMem) + sizeof(ncclRecvMem)));
  EXPECT_EQ(r->gdcSync, nullptr);
  EXPECT_EQ(this->Pool(), this->shared_->hostBuff);
  EXPECT_FALSE(NCCL_NET_MAP_DEV_MEM(&r->map, buffs[NCCL_PROTO_SIMPLE]));
  ASSERT_EQ(this->net_.regMrs.size(), 1u);
  EXPECT_EQ(this->net_.regMrs[0].comm, &this->net_.collComm);
  EXPECT_EQ(this->net_.regMrs[0].data, this->shared_->hostBuff);
  EXPECT_EQ(this->net_.regMrs[0].size, (size_t)this->kPoolSize);
  EXPECT_EQ(this->net_.regMrs[0].type, NCCL_PTR_HOST);
  EXPECT_EQ(this->PoolMhandle(), this->shared_->hostBuff);
}

TYPED_TEST(CollNetProxyConnectBothMicrotest, Gdr_RegistersDevicePoolAsCuda) {
  this->UseGdr();

  ASSERT_EQ(this->Connect(), ncclSuccess);

  EXPECT_EQ(this->Pool(), this->shared_->cudaBuff);
  EXPECT_TRUE(NCCL_NET_MAP_DEV_MEM(&this->R()->map, buffs[NCCL_PROTO_SIMPLE]));
  ASSERT_EQ(this->net_.regMrs.size(), 1u);
  EXPECT_EQ(this->net_.regMrs[0].data, this->shared_->cudaBuff);
  EXPECT_EQ(this->net_.regMrs[0].type, NCCL_PTR_CUDA);
  EXPECT_EQ(this->PoolMhandle(), this->shared_->cudaBuff);
}

TYPED_TEST(CollNetProxyConnectBothMicrotest, CollNetConnectFails_AnswersNoMapWithoutBuffers) {
  this->net_.connectResult = ncclSystemError;

  ASSERT_EQ(this->Connect(), ncclSuccess);

  EXPECT_EQ(this->resp_, nullptr);
  EXPECT_EQ(this->hostAllocs_, 0);
  EXPECT_EQ(this->conn_.proxyAppendPtr, nullptr);
}

TYPED_TEST(CollNetProxyConnectBothMicrotest, WrongResponseSize_RejectedWithoutBuffers) {
  EXPECT_EQ(this->Connect(sizeof(collNetConnectArgs), sizeof(connectMap)), ncclInternalError);
  EXPECT_EQ(this->hostAllocs_, 0);
}

TYPED_TEST(CollNetProxyConnectBothMicrotest, DependencyFails_ReturnsError) {
  this->net_.regMrResult = ncclSystemError;
  EXPECT_EQ(this->Connect(), ncclSystemError);
  EXPECT_EQ(this->resp_, reinterpret_cast<connectMap*>(0x1));

  this->net_.regMrResult = ncclSuccess;
  this->NewConn();
  g_hipHostMalloc = [](void**, size_t, unsigned) { return hipErrorOutOfMemory; };
  EXPECT_NE(this->Connect(), ncclSuccess);
}

TYPED_TEST(CollNetProxyConnectBothMicrotest, CloseListenFails_ReturnsError) {
  this->net_.closeListenResult = ncclSystemError;

  EXPECT_EQ(this->Connect(), ncclSystemError);
}

TYPED_TEST(CollNetProxyConnectBothMicrotest, HsaDmaBuf_RegistersPoolByExportedFd) {
  this->UseGdr(/*dmaBuf=*/true);
  pfn_hsa_amd_portable_export_dmabuf = FakeExportDmaBuf;

  ASSERT_EQ(this->Connect(), ncclSuccess);

  EXPECT_TRUE(this->net_.regMrs.empty());
  ASSERT_EQ(this->net_.regMrDmaBufs.size(), 1u);
  const auto& mr = this->net_.regMrDmaBufs[0];
  EXPECT_EQ(mr.comm, &this->net_.collComm);
  EXPECT_EQ(mr.data, this->shared_->cudaBuff);
  EXPECT_EQ(mr.size, (size_t)this->kPoolSize);
  EXPECT_EQ(mr.type, NCCL_PTR_CUDA);
  ASSERT_EQ(g_exportedFds.size(), 1u);
  EXPECT_EQ(mr.fd, g_exportedFds[0]);
  EXPECT_EQ(this->PoolMhandle(), this->shared_->cudaBuff);
}

TYPED_TEST(CollNetProxyConnectBothMicrotest, HsaDmaBufFails_FallsBackToRegMr) {
  this->UseGdr(/*dmaBuf=*/true);
  pfn_hsa_amd_portable_export_dmabuf = FailExportDmaBuf;
  ASSERT_EQ(this->Connect(), ncclSuccess);
  EXPECT_EQ(this->net_.regMrs.size(), 1u);

  this->NewConn();
  pfn_hsa_amd_portable_export_dmabuf = FakeExportDmaBuf;
  this->net_.regMrDmaBufResult = ncclSystemError;
  ASSERT_EQ(this->Connect(), ncclSuccess);

  ASSERT_EQ(this->net_.regMrs.size(), 2u);
  EXPECT_EQ(this->net_.regMrs[1].type, NCCL_PTR_CUDA);
  EXPECT_EQ(this->PoolMhandle(), this->shared_->cudaBuff);
}

TYPED_TEST(CollNetProxyConnectBothMicrotest, DmaBufNotUsable_RegistersWithRegMrWithoutExporting) {
  pfn_hsa_amd_portable_export_dmabuf = FakeExportDmaBuf;
  this->UseGdr(/*dmaBuf=*/false);
  ASSERT_EQ(this->Connect(), ncclSuccess);

  this->NewConn();
  this->R()->useGdr = ncclTopoGdrModeDisable;
  this->R()->useDmaBuf = 1;
  ASSERT_EQ(this->Connect(), ncclSuccess);

  this->NewConn();
  this->UseGdr(/*dmaBuf=*/true);
  this->vtable_.regMrDmaBuf = nullptr;
  ASSERT_EQ(this->Connect(), ncclSuccess);

  this->NewConn();
  pfn_hsa_amd_portable_export_dmabuf = nullptr;
  this->vtable_ = this->net_.vtable();
  ASSERT_EQ(this->Connect(), ncclSuccess);

  EXPECT_EQ(this->net_.regMrs.size(), 4u);
  EXPECT_TRUE(this->net_.regMrDmaBufs.empty());
  EXPECT_TRUE(g_exportedFds.empty());
}

#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
TYPED_TEST(CollNetProxyConnectBothMicrotest, CuMemDmaBuf_RegistersPoolByCuMemFdAndClosesIt) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  this->UseGdr(/*dmaBuf=*/true);
  pfn_hsa_amd_portable_export_dmabuf = FakeExportDmaBuf;  // must not be needed
  this->shared_->size = this->kPoolSize;                   // pool already allocated
  this->shared_->cudaBuff = static_cast<char*>(this->Alloc(this->kPoolSize));

  ASSERT_EQ(this->Connect(), ncclSuccess);

  ASSERT_EQ(this->exports_.size(), 1u);
  EXPECT_EQ(this->exports_[0].ptr, this->shared_->cudaBuff);
  EXPECT_EQ(this->exports_[0].size, (size_t)this->kPoolSize);
  EXPECT_EQ(this->exports_[0].flags, (unsigned long long)getHandleForAddressRangeFlags(ncclTopoGdrModePci));
  ASSERT_EQ(this->net_.regMrDmaBufs.size(), 1u);
  EXPECT_EQ(this->net_.regMrDmaBufs[0].fd, this->exports_[0].fd);
  EXPECT_EQ(this->net_.regMrDmaBufs[0].offset, 0u);
  EXPECT_EQ(this->net_.regMrDmaBufs[0].type, NCCL_PTR_CUDA);
  EXPECT_FALSE(FdOpen(this->exports_[0].fd));
  EXPECT_TRUE(this->net_.regMrs.empty());
  EXPECT_TRUE(g_exportedFds.empty());
  EXPECT_EQ(this->PoolMhandle(), this->shared_->cudaBuff);
}

TYPED_TEST(CollNetProxyConnectBothMicrotest, CuMemDmaBufFails_ClosesFdAndFallsBack) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  this->UseGdr(/*dmaBuf=*/true);
  this->shared_->size = this->kPoolSize;
  this->shared_->cudaBuff = static_cast<char*>(this->Alloc(this->kPoolSize));
  this->exports_.result = hipErrorNotSupported;
  pfn_hsa_amd_portable_export_dmabuf = FakeExportDmaBuf;
  ASSERT_EQ(this->Connect(), ncclSuccess);  // export refused: the HSA exporter takes over
  EXPECT_EQ(g_exportedFds.size(), 1u);

  this->NewConn();
  this->exports_.result = hipSuccess;
  pfn_hsa_amd_portable_export_dmabuf = nullptr;
  this->net_.regMrDmaBufResult = ncclSystemError;
  ASSERT_EQ(this->Connect(), ncclSuccess);  // registration refused: plain regMr takes over

  ASSERT_EQ(this->exports_.size(), 1u);
  EXPECT_FALSE(FdOpen(this->exports_[0].fd));
  ASSERT_EQ(this->net_.regMrs.size(), 1u);
  EXPECT_EQ(this->net_.regMrs[0].type, NCCL_PTR_CUDA);
}

TYPED_TEST(CollNetProxyConnectBothMicrotest, CuMemDmaBufAndRegMrFail_LeavesOtherDescriptorsOpen) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  this->UseGdr(/*dmaBuf=*/true);
  this->shared_->size = this->kPoolSize;
  this->shared_->cudaBuff = static_cast<char*>(this->Alloc(this->kPoolSize));
  this->net_.regMrDmaBufResult = ncclSystemError;
  // The plugin opens a descriptor after the export fd is closed, so it can reuse that number.
  static int pluginFd;
  pluginFd = -1;
  this->vtable_.regMr = [](void*, void*, size_t, int, void**) {
    pluginFd = open("/dev/null", O_RDONLY);
    return ncclSystemError;
  };

  EXPECT_EQ(this->Connect(), ncclSystemError);

  ASSERT_GE(pluginFd, 0);
  EXPECT_TRUE(FdOpen(pluginFd));
  close(pluginFd);
}
#endif

// Side-specific connect behaviour.
using CollNetSendProxyConnectMicrotest = CollNetProxyConnectMicrotest<SendSide>;
using CollNetRecvProxyConnectMicrotest = CollNetProxyConnectMicrotest<RecvSide>;

TEST_F(CollNetSendProxyConnectMicrotest, Connect_TakesRecvSidesFifoAndHandles_WithheldCredits) {
  ASSERT_EQ(Connect(), ncclSuccess);

  EXPECT_EQ((void*)R()->reqFifo, reqFifo_);
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) EXPECT_EQ(R()->recvMhandles[p], reinterpret_cast<void*>(0x100 + p));
  EXPECT_EQ(R()->sendMem->head, (uint64_t)-NCCL_STEPS);  // no credits until the recv side posts
  EXPECT_EQ(R()->step, 0u);
}

TEST_F(CollNetSendProxyConnectMicrotest, GdrCopy_HeadCreditsLiveInGdcWord) {
  ncclGdrCopy = reinterpret_cast<gdr_t>(0x1);

  ASSERT_EQ(Connect(), ncclSuccess);

  sendResources* r = R();
  ASSERT_NE(r->gdcSync, nullptr);
  EXPECT_EQ(*r->gdcSync, (uint64_t)-NCCL_STEPS);
  EXPECT_EQ(r->sendMem->head, 0u);
  EXPECT_EQ(r->map.mems[NCCL_NET_MAP_GDCMEM].cpuPtr, (char*)r->gdcSync);
  EXPECT_EQ(r->map.mems[NCCL_NET_MAP_GDCMEM].gpuPtr, (char*)r->gdcSync);  // AMD maps device memory 1:1
  EXPECT_EQ(r->map.mems[NCCL_NET_MAP_GDCMEM].size, (int)sizeof(uint64_t));
  EXPECT_NE(r->gdrDesc, nullptr);
}

TEST_F(CollNetSendProxyConnectMicrotest, GdrCopySyncDisabled_KeepsHeadInHostMemory) {
  ncclGdrCopy = reinterpret_cast<gdr_t>(0x1);
  ScopedHook param(g_loadParam, [](const char* env, int64_t deft) {
    return strcmp(env, "GDRCOPY_SYNC_ENABLE") == 0 ? 0 : deft;
  });

  ASSERT_EQ(Connect(), ncclSuccess);

  EXPECT_EQ(R()->gdcSync, nullptr);
  EXPECT_EQ(R()->gdrDesc, nullptr);
  EXPECT_EQ(R()->sendMem->head, (uint64_t)-NCCL_STEPS);
  EXPECT_EQ(R()->map.mems[NCCL_NET_MAP_GDCMEM].gpuPtr, nullptr);
}

TEST_F(CollNetRecvProxyConnectMicrotest, Connect_PublishesFifoAndPoolHandleForSendSide) {
  ASSERT_EQ(Connect(), ncclSuccess);

  auto* info = reinterpret_cast<collNetSendConnectInfo*>(connectInfos_ + kRank);
  EXPECT_EQ(info->reqFifo, (void*)R()->reqFifo);
  EXPECT_EQ(info->mhandles[NCCL_PROTO_SIMPLE], shared_->hostBuff);
  EXPECT_EQ(info->mhandles[NCCL_PROTO_LL], nullptr);
  EXPECT_EQ(info->mhandles[NCCL_PROTO_LL128], nullptr);
  EXPECT_EQ(R()->recvMem->tail, 0u);
}

TEST_F(CollNetRecvProxyConnectMicrotest, GdrCopy_TailInGdcWordAndFlushWordWhenEnabled) {
  ncclGdrCopy = reinterpret_cast<gdr_t>(0x1);
  R()->needFlush = ncclTopoFlushAlways;
  ASSERT_EQ(Connect(), ncclSuccess);
  recvResources* r = R();
  ASSERT_NE(r->gdcSync, nullptr);
  EXPECT_EQ(r->map.mems[NCCL_NET_MAP_GDCMEM].cpuPtr, (char*)r->gdcSync);
  EXPECT_EQ(r->map.mems[NCCL_NET_MAP_GDCMEM].gpuPtr, (char*)r->gdcSync);
  EXPECT_EQ(r->map.mems[NCCL_NET_MAP_GDCMEM].size, (int)sizeof(uint64_t));
  EXPECT_EQ(r->gdcFlush, nullptr);  // GDRCOPY_FLUSH_ENABLE defaults off
  EXPECT_EQ(r->needFlush, ncclTopoFlushAlways);

  NewConn();
  ScopedHook param(g_loadParam, [](const char* env, int64_t deft) {
    if (strcmp(env, "GDRCOPY_FLUSH_ENABLE") == 0) return (int64_t)1;
    return strcmp(env, "GDRCOPY_SYNC_ENABLE") == 0 ? 0 : deft;
  });
  ASSERT_EQ(Connect(), ncclSuccess);
  r = R();
  EXPECT_EQ(r->gdcSync, nullptr);
  EXPECT_EQ(r->map.mems[NCCL_NET_MAP_GDCMEM].gpuPtr, nullptr);
  ASSERT_NE(r->gdrDesc, nullptr);
  auto* desc = static_cast<gdr_mem_desc_t*>(r->gdrDesc);
  EXPECT_EQ(r->gdcFlush, reinterpret_cast<uint64_t*>(desc->gdrDevMem) + 1);
}

// ---------------------------------------------------------------------------
// Proxy-side free (sendProxyFree, recvProxyFree)
// ---------------------------------------------------------------------------

TYPED_TEST(CollNetProxyConnectBothMicrotest, Free_NoResources_Succeeds) {
  this->DropConn();

  EXPECT_EQ(this->Free(), ncclSuccess);
  EXPECT_EQ(this->shared_->refCount, 2);
}

TYPED_TEST(CollNetProxyConnectBothMicrotest, Free_ConnectedConnection_ReleasesEverythingItHolds) {
  ncclGdrCopy = reinterpret_cast<gdr_t>(0x1);
  this->UseGdr();
  ASSERT_EQ(this->Connect(), ncclSuccess);
  void* pool = this->PoolMhandle();
  void* userBuf = reinterpret_cast<void*>(0x7000);
  auto* registered = static_cast<proxyMemHandle*>(calloc(1, sizeof(proxyMemHandle)));
  registered->handle = userBuf;
  ncclIntruQueueEnqueue(&this->conn_.proxyMemHandleQueue, registered);
  this->R()->step = 42;  // it has run ops

  ASSERT_EQ(this->Free(), ncclSuccess);

  std::vector<std::pair<void*, void*>> want = {{&this->net_.collComm, userBuf}, {&this->net_.collComm, pool}};
  EXPECT_EQ(this->net_.deregMrs, want);
  EXPECT_TRUE(ncclIntruQueueEmpty(&this->conn_.proxyMemHandleQueue));
  EXPECT_TRUE(this->live_.empty());  // control words, GDC word and pool
  EXPECT_EQ(this->net_.closedColls, std::vector<void*>{&this->net_.collComm});  // its comm reference dropped
  EXPECT_EQ(this->shared_->refCount, 1);
}

TYPED_TEST(CollNetProxyConnectBothMicrotest, Free_LastReference_ReleasesSharedState) {
  ASSERT_EQ(this->Connect(), ncclSuccess);
  this->shared_->refCount = 1;

  ASSERT_EQ(this->Free(), ncclSuccess);
  this->shared_ = nullptr;  // released by the free

  EXPECT_TRUE(this->live_.empty());
}

TYPED_TEST(CollNetProxyConnectBothMicrotest, Free_DeregistrationFails_ReturnsError) {
  ASSERT_EQ(this->Connect(), ncclSuccess);
  this->net_.deregMrResult = ncclSystemError;
  EXPECT_EQ(this->Free(), ncclSystemError);

  auto* registered = static_cast<proxyMemHandle*>(calloc(1, sizeof(proxyMemHandle)));
  ncclIntruQueueEnqueue(&this->conn_.proxyMemHandleQueue, registered);
  EXPECT_EQ(this->Free(), ncclSystemError);
  free(registered);  // dequeued before the failing deregistration
  this->net_.deregMrResult = ncclSuccess;
  ASSERT_EQ(this->Free(), ncclSuccess);
}

}  // namespace
