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
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "ScopedHook.h"
#include "fakes/dev_runtime_micro_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/rocmwrap_fakes.h"
#include "fakes/topo_net_stubs.h"
#include "fakes/topo_stubs.h"
#include "fakes/transport_stubs.h"

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

  // Collectives. A request is a token address; issueReady=false models a busy plugin.
  struct CollCall {
    void* comm;
    void* send;
    void* recv;
    size_t count;
    ncclDataType_t dtype;
    ncclRedOp_t op;
    void* sendMhandle;
    void* recvMhandle;
    std::vector<ncclNetSGE_t> parts;
    size_t bytesPerRank, windowOffset, windowBytes;
  };
  std::vector<CollCall> iallreduces, iallgathers, ireducescatters;
  bool issueReady = true;
  ncclResult_t issueResult = ncclSuccess;
  char requestTokens[64] = {};
  int nextToken = 0;
  int testDone = 1;
  std::set<void*> pending;  // requests that stay incomplete whatever testDone says
  ncclResult_t testResult = ncclSuccess;
  std::vector<void*> tested;

  struct FlushCall {
    void* comm;
    void* data;
    int size;
    void* mhandle;
  };
  std::vector<FlushCall> iflushes;
  ncclResult_t iflushResult = ncclSuccess;

  std::set<std::pair<int, int>> supported;  // (dtype, op) this plugin reduces
  std::set<std::pair<int, int>> asked;
  ncclResult_t reduceSupportResult = ncclSuccess;

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

  void* Issue(void** request) {
    *request = issueReady ? &requestTokens[nextToken++ % 64] : nullptr;
    return *request;
  }
  ncclResult_t iallreduce(void* comm, void* send, void* recv, size_t count, ncclDataType_t dtype, ncclRedOp_t op,
                          void* sendMh, void* recvMh, void** request) {
    iallreduces.push_back({comm, send, recv, count, dtype, op, sendMh, recvMh, {}, 0, 0, 0});
    Issue(request);
    return issueResult;
  }
  ncclResult_t iallgather(void* comm, void* send, int nParts, ncclNetSGE_t* parts, size_t bytesPerRank,
                          size_t windowOffset, size_t windowBytes, void* sendMh, void** request) {
    iallgathers.push_back({comm, send, nullptr, 0, ncclNumTypes, ncclNumOps, sendMh, nullptr,
                           std::vector<ncclNetSGE_t>(parts, parts + nParts), bytesPerRank, windowOffset, windowBytes});
    Issue(request);
    return issueResult;
  }
  ncclResult_t ireducescatter(void* comm, int nParts, ncclNetSGE_t* parts, void* recv, size_t bytesPerRank,
                              size_t windowOffset, size_t windowBytes, ncclDataType_t dtype, ncclRedOp_t op,
                              void* recvMh, void** request) {
    ireducescatters.push_back({comm, nullptr, recv, 0, dtype, op, nullptr, recvMh,
                               std::vector<ncclNetSGE_t>(parts, parts + nParts), bytesPerRank, windowOffset,
                               windowBytes});
    Issue(request);
    return issueResult;
  }

  ncclResult_t reduceSupport(ncclDataType_t dtype, ncclRedOp_t op, int* out) {
    asked.insert({dtype, op});
    *out = supported.count({dtype, op});
    return reduceSupportResult;
  }
  ncclResult_t iflush(void* comm, void* data, int size, void* mhandle, void** request) {
    iflushes.push_back({comm, data, size, mhandle});
    Issue(request);
    return iflushResult;
  }
  ncclResult_t test(void* request, int* done, int* size) {
    if (request == nullptr) ADD_FAILURE() << "test() on a null request";
    tested.push_back(request);
    *done = testDone && !pending.count(request);
    if (size) *size = 0;
    return testResult;
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
    v.iallreduce = &FakeCollNet::TrampIallreduce;
    v.iallgather = &FakeCollNet::TrampIallgather;
    v.ireducescatter = &FakeCollNet::TrampIreducescatter;
    v.iflush = &FakeCollNet::TrampIflush;
    v.reduceSupport = &FakeCollNet::TrampReduceSupport;
    v.test = &FakeCollNet::TrampTest;
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
  static ncclResult_t TrampIallreduce(void* comm, void* send, void* recv, size_t count, ncclDataType_t dtype,
                                      ncclRedOp_t op, void* sendMh, void* recvMh, void** request) {
    return g_activeCollNet->iallreduce(comm, send, recv, count, dtype, op, sendMh, recvMh, request);
  }
  static ncclResult_t TrampIallgather(void* comm, void* send, int nParts, ncclNetSGE_t* parts, size_t bytesPerRank,
                                      size_t windowOffset, size_t windowBytes, void* sendMh, void** request) {
    return g_activeCollNet->iallgather(comm, send, nParts, parts, bytesPerRank, windowOffset, windowBytes, sendMh,
                                       request);
  }
  static ncclResult_t TrampReduceSupport(ncclDataType_t dtype, ncclRedOp_t op, int* supported) {
    return g_activeCollNet->reduceSupport(dtype, op, supported);
  }
  static ncclResult_t TrampIflush(void* comm, void* data, int size, void* mhandle, void** request) {
    return g_activeCollNet->iflush(comm, data, size, mhandle, request);
  }
  static ncclResult_t TrampTest(void* request, int* done, int* size) {
    return g_activeCollNet->test(request, done, size);
  }
  static ncclResult_t TrampIreducescatter(void* comm, int nParts, ncclNetSGE_t* parts, void* recv,
                                          size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                                          ncclDataType_t dtype, ncclRedOp_t op, void* recvMh, void** request) {
    return g_activeCollNet->ireducescatter(comm, nParts, parts, recv, bytesPerRank, windowOffset, windowBytes, dtype,
                                           op, recvMh, request);
  }
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
    ResetTransportStubs();
    ResetDevRuntimeMicroFakes();
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

TEST(CollNetHelpersMicrotest, Transport_WiresEachSidesEntryPoints) {
  EXPECT_STREQ(collNetTransport.name, "COL");
  EXPECT_EQ(collNetTransport.canConnect, canConnect);
  const ncclTransportComm& send = collNetTransport.send;
  EXPECT_EQ(send.setup, sendSetup);
  EXPECT_EQ(send.connect, sendConnect);
  EXPECT_EQ(send.free, sendFree);
  EXPECT_EQ(send.proxySharedInit, nullptr);
  EXPECT_EQ(send.proxySetup, sendProxySetup);
  EXPECT_EQ(send.proxyConnect, sendProxyConnect);
  EXPECT_EQ(send.proxyFree, sendProxyFree);
  EXPECT_EQ(send.proxyProgress, sendProxyProgress);
  EXPECT_EQ(send.proxyRegister, sendProxyRegBuffer);
  EXPECT_EQ(send.proxyDeregister, sendProxyDeregBuffer);
  const ncclTransportComm& recv = collNetTransport.recv;
  EXPECT_EQ(recv.setup, recvSetup);
  EXPECT_EQ(recv.connect, recvConnect);
  EXPECT_EQ(recv.free, recvFree);
  EXPECT_EQ(recv.proxySharedInit, nullptr);
  EXPECT_EQ(recv.proxySetup, recvProxySetup);
  EXPECT_EQ(recv.proxyConnect, recvProxyConnect);
  EXPECT_EQ(recv.proxyFree, recvProxyFree);
  EXPECT_EQ(recv.proxyProgress, recvProxyProgress);
  EXPECT_EQ(recv.proxyRegister, recvProxyRegBuffer);
  EXPECT_EQ(recv.proxyDeregister, recvProxyDeregBuffer);
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
  static ncclResult_t RegBuffer(ncclProxyConnection* c, ncclProxyState* ps, void* req, int reqSize, void* resp,
                                int respSize, int* done) {
    return sendProxyRegBuffer(c, ps, req, reqSize, resp, respSize, done);
  }
  static ncclResult_t DeregBuffer(ncclProxyConnection* c, ncclProxyState* ps, void* req, int reqSize, int* done) {
    return sendProxyDeregBuffer(c, ps, req, reqSize, done);
  }
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
  static ncclResult_t RegBuffer(ncclProxyConnection* c, ncclProxyState* ps, void* req, int reqSize, void* resp,
                                int respSize, int* done) {
    return recvProxyRegBuffer(c, ps, req, reqSize, resp, respSize, done);
  }
  static ncclResult_t DeregBuffer(ncclProxyConnection* c, ncclProxyState* ps, void* req, int reqSize, int* done) {
    return recvProxyDeregBuffer(c, ps, req, reqSize, done);
  }
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
  // Both take the lowest free fd, so the plugin's descriptor reuses the closed export fd's number.
  static int pluginFd, exportFd;
  pluginFd = exportFd = -1;
  g_hipMemGetHandleForAddressRange = [](void* handle, hipDeviceptr_t, size_t, hipMemRangeHandleType,
                                        unsigned long long) {
    exportFd = open("/dev/null", O_RDONLY);
    *static_cast<int*>(handle) = exportFd;
    return hipSuccess;
  };
  this->vtable_.regMr = [](void*, void*, size_t, int, void**) {
    pluginFd = open("/dev/null", O_RDONLY);
    return ncclSystemError;
  };

  EXPECT_EQ(this->Connect(), ncclSystemError);

  ASSERT_GE(exportFd, 0);
  ASSERT_EQ(pluginFd, exportFd);
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

// ---------------------------------------------------------------------------
// Proxy-side buffer registration (sendProxyRegBuffer, recvProxyRegBuffer, *ProxyDeregBuffer)
// ---------------------------------------------------------------------------

template <typename Side>
class CollNetProxyRegMicrotest : public CollNetProxyConnectMicrotest<Side> {
 protected:
  char user_[256] = {};
  collnetRegInfo info_{reinterpret_cast<uintptr_t>(user_), sizeof(user_)};
  void* handle_ = reinterpret_cast<void*>(0x1);

  void SetUp() override {
    CollNetProxyConnectMicrotest<Side>::SetUp();
    this->R()->collNetComm = &this->net_.collComm;
  }

  void TearDown() override {
    while (!ncclIntruQueueEmpty(&this->conn_.proxyMemHandleQueue)) {
      free(ncclIntruQueueDequeue(&this->conn_.proxyMemHandleQueue));
    }
    CollNetProxyConnectMicrotest<Side>::TearDown();
  }

  ncclResult_t Reg(int reqSize = sizeof(collnetRegInfo), int respSize = sizeof(void*)) {
    this->done_ = 0;
    return Side::RegBuffer(&this->conn_, &this->proxyState_, &info_, reqSize, &handle_, respSize, &this->done_);
  }
  ncclResult_t Dereg(void* handle, int reqSize = sizeof(void*)) {
    this->done_ = 0;
    return Side::DeregBuffer(&this->conn_, &this->proxyState_, &handle, reqSize, &this->done_);
  }
  std::vector<void*> Tracked() {
    std::vector<void*> out;
    for (proxyMemHandle* h = ncclIntruQueueHead(&this->conn_.proxyMemHandleQueue); h; h = h->next) {
      out.push_back(h->handle);
    }
    return out;
  }
};

template <typename Side>
using CollNetProxyRegBothMicrotest = CollNetProxyRegMicrotest<Side>;
TYPED_TEST_SUITE(CollNetProxyRegBothMicrotest, CollNetSides);

TYPED_TEST(CollNetProxyRegBothMicrotest, WrongSizes_RejectedWithoutRegistering) {
  EXPECT_EQ(this->Reg(sizeof(collnetRegInfo) - 1), ncclInternalError);
  EXPECT_EQ(this->Reg(sizeof(collnetRegInfo), sizeof(void*) + 1), ncclInternalError);
  EXPECT_EQ(this->Dereg(nullptr, sizeof(void*) - 1), ncclInternalError);
  EXPECT_TRUE(this->net_.regMrs.empty());
  EXPECT_TRUE(this->net_.deregMrs.empty());
  EXPECT_EQ(this->done_, 0);
}

TYPED_TEST(CollNetProxyRegBothMicrotest, Reg_UserBuffer_RegistersAsCudaAndTracksHandle) {
  ASSERT_EQ(this->Reg(), ncclSuccess);

  ASSERT_EQ(this->net_.regMrs.size(), 1u);
  EXPECT_EQ(this->net_.regMrs[0].comm, &this->net_.collComm);
  EXPECT_EQ(this->net_.regMrs[0].data, this->user_);
  EXPECT_EQ(this->net_.regMrs[0].size, sizeof(this->user_));
  EXPECT_EQ(this->net_.regMrs[0].type, NCCL_PTR_CUDA);
  EXPECT_EQ(this->handle_, this->user_);
  EXPECT_EQ(this->Tracked(), std::vector<void*>{this->user_});
  EXPECT_EQ(this->done_, 1);
}

TYPED_TEST(CollNetProxyRegBothMicrotest, Reg_PluginFails_AnswersNoHandle) {
  this->net_.regMrResult = ncclSystemError;

  ASSERT_EQ(this->Reg(), ncclSuccess);

  EXPECT_EQ(this->handle_, nullptr);
  EXPECT_TRUE(this->Tracked().empty());
  EXPECT_EQ(this->done_, 1);
}

TYPED_TEST(CollNetProxyRegBothMicrotest, Reg_HsaDmaBuf_RegistersByExportedFd) {
  this->UseGdr(/*dmaBuf=*/true);
  pfn_hsa_amd_portable_export_dmabuf = FakeExportDmaBuf;
  ASSERT_EQ(this->Reg(), ncclSuccess);
  ASSERT_EQ(this->net_.regMrDmaBufs.size(), 1u);
  EXPECT_EQ(this->net_.regMrDmaBufs[0].data, this->user_);
  EXPECT_EQ(this->net_.regMrDmaBufs[0].size, sizeof(this->user_));
  EXPECT_EQ(this->net_.regMrDmaBufs[0].type, NCCL_PTR_CUDA);
  EXPECT_TRUE(this->net_.regMrs.empty());
  ASSERT_EQ(g_exportedFds.size(), 1u);
  EXPECT_EQ(this->net_.regMrDmaBufs[0].fd, g_exportedFds[0]);
  EXPECT_EQ(this->handle_, this->user_);

  this->net_.regMrDmaBufResult = ncclSystemError;  // falls back to regMr
  ASSERT_EQ(this->Reg(), ncclSuccess);
  EXPECT_EQ(this->net_.regMrs.size(), 1u);
  EXPECT_EQ(this->Tracked().size(), 2u);
}

TYPED_TEST(CollNetProxyRegBothMicrotest, Reg_DmaBufNotUsable_RegistersWithRegMr) {
  pfn_hsa_amd_portable_export_dmabuf = FakeExportDmaBuf;
  this->UseGdr(/*dmaBuf=*/false);
  ASSERT_EQ(this->Reg(), ncclSuccess);
  this->R()->useGdr = ncclTopoGdrModeDisable;
  this->R()->useDmaBuf = 1;
  ASSERT_EQ(this->Reg(), ncclSuccess);
  this->UseGdr(/*dmaBuf=*/true);
  this->vtable_.regMrDmaBuf = nullptr;
  ASSERT_EQ(this->Reg(), ncclSuccess);
  this->vtable_ = this->net_.vtable();
  pfn_hsa_amd_portable_export_dmabuf = nullptr;
  ASSERT_EQ(this->Reg(), ncclSuccess);

  EXPECT_EQ(this->net_.regMrs.size(), 4u);
  EXPECT_TRUE(g_exportedFds.empty());
}

#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
TYPED_TEST(CollNetProxyRegBothMicrotest, Reg_CuMemDmaBuf_RegistersByCuMemFdWithFallbacks) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  this->UseGdr(/*dmaBuf=*/true);
  ASSERT_EQ(this->Reg(), ncclSuccess);
  ASSERT_EQ(this->exports_.size(), 1u);
  EXPECT_EQ(this->exports_[0].ptr, this->user_);
  EXPECT_EQ(this->exports_[0].size, sizeof(this->user_));
  EXPECT_EQ(this->exports_[0].flags, (unsigned long long)getHandleForAddressRangeFlags(ncclTopoGdrModePci));
  ASSERT_EQ(this->net_.regMrDmaBufs.size(), 1u);
  EXPECT_EQ(this->net_.regMrDmaBufs[0].fd, this->exports_[0].fd);
  EXPECT_EQ(this->net_.regMrDmaBufs[0].offset, 0u);
  EXPECT_FALSE(FdOpen(this->exports_[0].fd));
  EXPECT_TRUE(this->net_.regMrs.empty());

  this->net_.regMrDmaBufResult = ncclSystemError;  // registration refused: plain regMr
  ASSERT_EQ(this->Reg(), ncclSuccess);
  ASSERT_EQ(this->exports_.size(), 2u);
  EXPECT_FALSE(FdOpen(this->exports_[1].fd));
  EXPECT_EQ(this->net_.regMrs.size(), 1u);

  this->exports_.result = hipErrorNotSupported;  // export refused: the HSA exporter
  this->net_.regMrDmaBufResult = ncclSuccess;
  pfn_hsa_amd_portable_export_dmabuf = FakeExportDmaBuf;
  ASSERT_EQ(this->Reg(), ncclSuccess);
  EXPECT_EQ(g_exportedFds.size(), 1u);
  EXPECT_EQ(this->net_.regMrDmaBufs.size(), 3u);
  EXPECT_EQ(this->net_.regMrs.size(), 1u);
  EXPECT_EQ(this->Tracked().size(), 3u);
}
#endif

TYPED_TEST(CollNetProxyRegBothMicrotest, Dereg_TrackedHandle_DeregistersAndForgetsOnlyIt) {
  ASSERT_EQ(this->Reg(), ncclSuccess);
  this->info_.buffer += 128;
  ASSERT_EQ(this->Reg(), ncclSuccess);

  ASSERT_EQ(this->Dereg(this->user_), ncclSuccess);

  EXPECT_EQ(this->Tracked(), std::vector<void*>{this->user_ + 128});
  std::vector<std::pair<void*, void*>> want = {{&this->net_.collComm, this->user_}};
  EXPECT_EQ(this->net_.deregMrs, want);
  EXPECT_EQ(this->done_, 1);
}

TYPED_TEST(CollNetProxyRegBothMicrotest, Dereg_NullHandle_StillAsksPlugin) {
  ASSERT_EQ(this->Reg(), ncclSuccess);

  ASSERT_EQ(this->Dereg(nullptr), ncclSuccess);

  EXPECT_EQ(this->Tracked().size(), 1u);
  std::vector<std::pair<void*, void*>> want = {{&this->net_.collComm, nullptr}};
  EXPECT_EQ(this->net_.deregMrs, want);
}

TYPED_TEST(CollNetProxyRegBothMicrotest, Dereg_PluginFails_ReturnsError) {
  this->net_.deregMrResult = ncclSystemError;

  EXPECT_EQ(this->Dereg(this->user_), ncclSystemError);
  EXPECT_EQ(this->done_, 0);
}

// ---------------------------------------------------------------------------
// Client-side buffer registration (collnetRegisterBuffer, ncclCollnetLocalRegisterBuffer,
// ncclCollnetGraphRegisterBuffer, ncclCollnetDeregBuffer)
// ---------------------------------------------------------------------------

class CollNetRegisterMicrotest : public CollNetMicrotestBase {
 protected:
  static constexpr int kRanks = 4;
  static constexpr size_t kSize = 1 << 12;

  ncclComm comm_{};
  ncclChannelPeer collNetPeer_{};
  ncclChannelPeer* peers_[kRanks + 1] = {};
  ncclReg reg_{};
  ncclReg* slots_[1] = {&reg_};
  alignas(64) char buf_[2 * kSize] = {};
  char* user_ = buf_ + 64;
  void* handle_ = reinterpret_cast<void*>(0x9000);  // what the proxy hands back
  int flag_ = -1;
  void* out_ = reinterpret_cast<void*>(0x1);
  ncclProxyConnector* callConn_ = nullptr;
  int callType_ = -1;
  collnetRegInfo callInfo_{};
  ncclResult_t callResult_ = ncclSuccess;

  void SetUp() override {
    CollNetMicrotestBase::SetUp();
    comm_.nRanks = kRanks;
    peers_[kRanks] = &collNetPeer_;
    comm_.channels[0].peers = peers_;
    comm_.regCache.slots = slots_;
    comm_.regCache.population = 1;
    reg_.begAddr = reinterpret_cast<uintptr_t>(buf_);
    reg_.endAddr = reinterpret_cast<uintptr_t>(buf_) + sizeof(buf_);
    DirectNic(collNetRecv);
    DirectNic(collNetSend);
    g_proxyCallBlocking = [this](ncclComm*, ncclProxyConnector* conn, int type, void* req, int reqSize, void* resp,
                                 int respSize) {
      callConn_ = conn;
      callType_ = type;
      EXPECT_EQ(reqSize, (int)sizeof(collnetRegInfo));
      EXPECT_EQ(respSize, (int)sizeof(void*));
      memcpy(&callInfo_, req, sizeof(callInfo_));
      memcpy(resp, &handle_, sizeof(void*));
      return callResult_;
    };
  }

  void DirectNic(int type) {
    ncclConnector& c = type == collNetRecv ? collNetPeer_.recv[type] : collNetPeer_.send[type];
    c.conn.flags |= NCCL_DIRECT_NIC;
  }
  ncclProxyConnector* ProxyConn(int type) {
    return type == collNetRecv ? &collNetPeer_.recv[type].proxyConn : &collNetPeer_.send[type].proxyConn;
  }
  ncclResult_t Register(int type, ncclReg* reg) {
    return collnetRegisterBuffer(&comm_, user_, kSize, type, reg, &flag_, &out_);
  }
};

TEST_F(CollNetRegisterMicrotest, Register_DirectNic_RegistersRecordRangeThroughSidesProxy) {
  for (int type : {collNetRecv, collNetSend}) {
    SCOPED_TRACE(type);
    reg_.state = 0;

    ASSERT_EQ(Register(type, &reg_), ncclSuccess);

    EXPECT_EQ(callConn_, ProxyConn(type));
    EXPECT_EQ(callType_, ncclProxyMsgRegister);
    EXPECT_EQ(callInfo_.buffer, reg_.begAddr);
    EXPECT_EQ(callInfo_.size, sizeof(buf_));
    EXPECT_EQ(flag_, 1);
    EXPECT_EQ(out_, handle_);
    EXPECT_EQ(reg_.collnetHandle, handle_);
    EXPECT_EQ(reg_.collnetProxyconn, ProxyConn(type));
    EXPECT_TRUE(reg_.state & COLLNET_REG_COMPLETE);
  }
}

TEST_F(CollNetRegisterMicrotest, Register_AlreadyRegistered_ReusesHandleWithoutProxyCall) {
  reg_.state = COLLNET_REG_COMPLETE;
  reg_.collnetHandle = handle_;

  ASSERT_EQ(Register(collNetSend, &reg_), ncclSuccess);

  EXPECT_EQ(flag_, 2);
  EXPECT_EQ(out_, handle_);
  EXPECT_EQ(callType_, -1);
}

TEST_F(CollNetRegisterMicrotest, Register_NoRecord_LeavesOutputs) {
  ASSERT_EQ(Register(collNetRecv, nullptr), ncclSuccess);

  EXPECT_EQ(flag_, -1);
  EXPECT_EQ(out_, reinterpret_cast<void*>(0x1));
  EXPECT_EQ(callType_, -1);
}

TEST_F(CollNetRegisterMicrotest, Register_NoDirectNic_ReportsUnregistered) {
  collNetPeer_.send[collNetSend].conn.flags = 0;

  ASSERT_EQ(Register(collNetSend, &reg_), ncclSuccess);

  EXPECT_EQ(flag_, 0);
  EXPECT_EQ(out_, nullptr);
  EXPECT_EQ(callType_, -1);
  EXPECT_FALSE(reg_.state & COLLNET_REG_COMPLETE);
}

TEST_F(CollNetRegisterMicrotest, Register_ProxyAnswersNoHandle_LeavesRecordUnregistered) {
  handle_ = nullptr;

  ASSERT_EQ(Register(collNetRecv, &reg_), ncclSuccess);

  EXPECT_EQ(flag_, -1);  // callers preset the outputs; nothing to report
  EXPECT_EQ(out_, reinterpret_cast<void*>(0x1));
  EXPECT_FALSE(reg_.state & COLLNET_REG_COMPLETE);
  EXPECT_EQ(reg_.collnetProxyconn, nullptr);
}

TEST_F(CollNetRegisterMicrotest, Register_ProxyFails_ReturnsErrorAndClearsOutputs) {
  callResult_ = ncclRemoteError;

  EXPECT_EQ(Register(collNetRecv, &reg_), ncclRemoteError);
  EXPECT_EQ(flag_, 0);
  EXPECT_EQ(out_, nullptr);
  EXPECT_FALSE(reg_.state & COLLNET_REG_COMPLETE);
}

TEST_F(CollNetRegisterMicrotest, LocalRegister_ValidRecordCoveringBuffer_Registers) {
  ScopedHook valid(g_regLocalIsValid, [this](ncclReg* reg, bool* isValid) {
    EXPECT_EQ(reg, &reg_);
    *isValid = true;
    return ncclSuccess;
  });
  g_hipMemGetAddressRange = [this](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
    EXPECT_EQ(ptr, (hipDeviceptr_t)user_);
    *base = buf_;
    *size = sizeof(buf_);
    return hipSuccess;
  };

  ASSERT_EQ(ncclCollnetLocalRegisterBuffer(&comm_, user_, kSize, collNetSend, &flag_, &out_), ncclSuccess);

  EXPECT_EQ(flag_, 1);
  EXPECT_EQ(out_, handle_);
}

TEST_F(CollNetRegisterMicrotest, LocalRegister_AllocationShorterThanBuffer_SkipsRegistration) {
  ScopedHook valid(g_regLocalIsValid, [](ncclReg*, bool* isValid) {
    *isValid = true;
    return ncclSuccess;
  });
  g_hipMemGetAddressRange = [this](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t) {
    *base = buf_;
    *size = 64 + kSize - 1;
    return hipSuccess;
  };

  ASSERT_EQ(ncclCollnetLocalRegisterBuffer(&comm_, user_, kSize, collNetSend, &flag_, &out_), ncclSuccess);
  EXPECT_EQ(flag_, 0);
  EXPECT_EQ(out_, nullptr);
  EXPECT_EQ(callType_, -1);

  g_hipMemGetAddressRange = [this](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t) {
    *base = buf_;
    *size = 64 + kSize;  // exactly covers it
    return hipSuccess;
  };
  ASSERT_EQ(ncclCollnetLocalRegisterBuffer(&comm_, user_, kSize, collNetSend, &flag_, &out_), ncclSuccess);
  EXPECT_EQ(flag_, 1);
}

TEST_F(CollNetRegisterMicrotest, LocalRegister_RecordNotLocallyValid_RegistersWithoutRangeCheck) {
  g_hipMemGetAddressRange = [](hipDeviceptr_t*, size_t*, hipDeviceptr_t) { return hipErrorInvalidValue; };

  ASSERT_EQ(ncclCollnetLocalRegisterBuffer(&comm_, user_, kSize, collNetRecv, &flag_, &out_), ncclSuccess);

  EXPECT_EQ(flag_, 1);
}

TEST_F(CollNetRegisterMicrotest, LocalRegister_NoBufferOrNoRecord_ReportsUnregistered) {
  EXPECT_EQ(ncclCollnetLocalRegisterBuffer(&comm_, nullptr, kSize, collNetRecv, &flag_, &out_), ncclSuccess);
  EXPECT_EQ(flag_, 0);
  EXPECT_EQ(out_, nullptr);
  flag_ = -1;
  EXPECT_EQ(ncclCollnetLocalRegisterBuffer(&comm_, user_, 0, collNetRecv, &flag_, &out_), ncclSuccess);
  EXPECT_EQ(flag_, 0);
  flag_ = -1;
  EXPECT_EQ(ncclCollnetLocalRegisterBuffer(nullptr, user_, kSize, collNetRecv, &flag_, &out_), ncclSuccess);
  EXPECT_EQ(flag_, 0);
  comm_.regCache.population = 0;
  EXPECT_EQ(ncclCollnetLocalRegisterBuffer(&comm_, user_, kSize, collNetRecv, &flag_, &out_), ncclSuccess);
  EXPECT_EQ(flag_, 0);
  EXPECT_EQ(callType_, -1);
}

TEST_F(CollNetRegisterMicrotest, LocalRegister_DependencyFails_ReturnsErrorUnregistered) {
  ScopedHook valid(g_regLocalIsValid, [](ncclReg*, bool*) { return ncclSystemError; });
  EXPECT_EQ(ncclCollnetLocalRegisterBuffer(&comm_, user_, kSize, collNetRecv, &flag_, &out_), ncclSystemError);
  EXPECT_EQ(flag_, 0);

  g_regLocalIsValid = [](ncclReg*, bool* isValid) {
    *isValid = true;
    return ncclSuccess;
  };
  g_hipMemGetAddressRange = [](hipDeviceptr_t*, size_t*, hipDeviceptr_t) { return hipErrorInvalidValue; };
  EXPECT_NE(ncclCollnetLocalRegisterBuffer(&comm_, user_, kSize, collNetRecv, &flag_, &out_), ncclSuccess);
  EXPECT_EQ(flag_, 0);

  callResult_ = ncclRemoteError;
  ScopedHook unchecked(g_regLocalIsValid, [](ncclReg*, bool* isValid) {
    *isValid = false;
    return ncclSuccess;
  });
  flag_ = 1;
  EXPECT_EQ(ncclCollnetLocalRegisterBuffer(&comm_, user_, kSize, collNetRecv, &flag_, &out_), ncclRemoteError);
  EXPECT_EQ(flag_, 0);
}

class CollNetGraphRegisterMicrotest : public CollNetRegisterMicrotest {
 protected:
  ncclIntruQueue<ncclCommCallback, &ncclCommCallback::next> cleanup_{};
  int nCleanup_ = 3;
  std::vector<ncclReg*> deregistered_;
  size_t allocSize_ = sizeof(buf_);  // the allocation user_ sits in starts at buf_

  void SetUp() override {
    CollNetRegisterMicrotest::SetUp();
    g_hipMemGetAddressRange = [this](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
      EXPECT_EQ(ptr, (hipDeviceptr_t)user_);
      *base = buf_;
      *size = allocSize_;
      return hipSuccess;
    };
    g_commGraphRegister = [this](ncclComm* comm, void* buff, size_t size, void** handle) {
      EXPECT_EQ(comm, &comm_);
      EXPECT_EQ(buff, buf_);
      EXPECT_EQ(size, allocSize_);
      *handle = &reg_;
      return ncclSuccess;
    };
    g_commGraphDeregister = [this](ncclComm* comm, ncclReg* reg) {
      EXPECT_EQ(comm, &comm_);
      deregistered_.push_back(reg);
      return ncclSuccess;
    };
  }

  void TearDown() override {
    while (!ncclIntruQueueEmpty(&cleanup_)) free(ncclIntruQueueDequeue(&cleanup_));
    CollNetRegisterMicrotest::TearDown();
  }

  ncclResult_t GraphRegister(int type = collNetSend, const void* buf = nullptr, size_t size = kSize) {
    return ncclCollnetGraphRegisterBuffer(&comm_, buf ? buf : user_, size, type, &flag_, &out_, &cleanup_,
                                          &nCleanup_);
  }
};

TEST_F(CollNetGraphRegisterMicrotest, Registered_QueuesCleanupThatDeregistersRecord) {
  ASSERT_EQ(GraphRegister(), ncclSuccess);

  EXPECT_EQ(flag_, 1);
  EXPECT_EQ(out_, handle_);
  EXPECT_EQ(nCleanup_, 4);
  ASSERT_FALSE(ncclIntruQueueEmpty(&cleanup_));
  ncclCommCallback* cb = ncclIntruQueueDequeue(&cleanup_);
  EXPECT_TRUE(ncclIntruQueueEmpty(&cleanup_));
  EXPECT_TRUE(deregistered_.empty());

  ASSERT_EQ(cb->fn(nullptr, cb), ncclSuccess);  // deregisters on the comm it recorded, then frees cb
  EXPECT_EQ(deregistered_, std::vector<ncclReg*>{&reg_});
}

TEST_F(CollNetGraphRegisterMicrotest, CleanupDeregisterFails_ReturnsError) {
  ASSERT_EQ(GraphRegister(), ncclSuccess);
  ncclCommCallback* cb = ncclIntruQueueDequeue(&cleanup_);
  g_commGraphDeregister = [](ncclComm*, ncclReg*) { return ncclSystemError; };

  EXPECT_EQ(cb->fn(&comm_, cb), ncclSystemError);
  free(cb);
}

TEST_F(CollNetGraphRegisterMicrotest, NotRegistered_DropsGraphRecordAtOnce) {
  collNetPeer_.send[collNetSend].conn.flags = 0;

  ASSERT_EQ(GraphRegister(), ncclSuccess);

  EXPECT_EQ(flag_, 0);
  EXPECT_EQ(nCleanup_, 3);
  EXPECT_TRUE(ncclIntruQueueEmpty(&cleanup_));
  EXPECT_EQ(deregistered_, std::vector<ncclReg*>{&reg_});
}

TEST_F(CollNetGraphRegisterMicrotest, AllocationShorterThanBuffer_SkipsRegistration) {
  allocSize_ = 64 + kSize - 1;

  ASSERT_EQ(GraphRegister(), ncclSuccess);
  EXPECT_EQ(flag_, 0);
  EXPECT_EQ(out_, reinterpret_cast<void*>(0x1));
  EXPECT_TRUE(deregistered_.empty());
  EXPECT_EQ(callType_, -1);

  allocSize_ = 64 + kSize;  // exactly covers it
  ASSERT_EQ(GraphRegister(), ncclSuccess);
  EXPECT_EQ(flag_, 1);
}

TEST_F(CollNetGraphRegisterMicrotest, NoBuffer_ReportsUnregistered) {
  EXPECT_EQ(ncclCollnetGraphRegisterBuffer(&comm_, nullptr, kSize, collNetSend, &flag_, &out_, &cleanup_, &nCleanup_),
            ncclSuccess);
  EXPECT_EQ(flag_, 0);
  flag_ = -1;
  EXPECT_EQ(GraphRegister(collNetSend, user_, 0), ncclSuccess);
  EXPECT_EQ(flag_, 0);
  flag_ = -1;
  EXPECT_EQ(ncclCollnetGraphRegisterBuffer(nullptr, user_, kSize, collNetSend, &flag_, &out_, &cleanup_, &nCleanup_),
            ncclSuccess);
  EXPECT_EQ(flag_, 0);
  EXPECT_EQ(callType_, -1);
}

TEST_F(CollNetGraphRegisterMicrotest, DependencyFails_ReturnsErrorUnregistered) {
  callResult_ = ncclRemoteError;
  EXPECT_EQ(GraphRegister(), ncclRemoteError);
  EXPECT_EQ(flag_, 0);
  EXPECT_EQ(out_, nullptr);

  callResult_ = ncclSuccess;
  collNetPeer_.send[collNetSend].conn.flags = 0;
  g_commGraphDeregister = [](ncclComm*, ncclReg*) { return ncclSystemError; };
  EXPECT_EQ(GraphRegister(), ncclSystemError);

  g_commGraphRegister = [](ncclComm*, void*, size_t, void**) { return ncclSystemError; };
  EXPECT_EQ(GraphRegister(), ncclSystemError);

  g_hipMemGetAddressRange = [](hipDeviceptr_t*, size_t*, hipDeviceptr_t) { return hipErrorInvalidValue; };
  out_ = reinterpret_cast<void*>(0x1);
  EXPECT_NE(GraphRegister(), ncclSuccess);
  EXPECT_EQ(out_, nullptr);
  EXPECT_EQ(nCleanup_, 3);
}

TEST_F(CollNetRegisterMicrotest, Dereg_AsksRegisteringProxyToDropHandle) {
  ncclProxyConnector* conn = ProxyConn(collNetSend);
  void* sent = nullptr;
  g_proxyCallBlocking = [&](ncclComm*, ncclProxyConnector* c, int type, void* req, int reqSize, void* resp,
                            int respSize) {
    EXPECT_EQ(c, conn);
    EXPECT_EQ(type, ncclProxyMsgDeregister);
    EXPECT_EQ(reqSize, (int)sizeof(void*));
    memcpy(&sent, req, sizeof(sent));
    EXPECT_EQ(resp, nullptr);
    EXPECT_EQ(respSize, 0);
    return callResult_;
  };

  ASSERT_EQ(ncclCollnetDeregBuffer(&comm_, conn, handle_), ncclSuccess);
  EXPECT_EQ(sent, handle_);

  callResult_ = ncclRemoteError;
  EXPECT_EQ(ncclCollnetDeregBuffer(&comm_, conn, handle_), ncclRemoteError);
}

// ---------------------------------------------------------------------------
// Shared-buffer offsets (calcAlgoOffset, calcRegionOffset)
// ---------------------------------------------------------------------------

// 3 subs of 128-byte chunks; 4 nodes of 512 bytes per rank, this one node 1; a 2-channel
// pool of 8 KiB per direction and channel, so 1 KiB per step slot.
class CollNetOffsetsMicrotest : public ::testing::Test {
 protected:
  ncclCollNetSharedRes shared_{};
  ncclProxyConnection conn_{};
  ncclProxyArgs args_{};

  void SetUp() override {
    shared_.buffSize = 8 << 10;
    shared_.nChannels = 2;
    conn_.collNet = &shared_;
    args_.subs[0].connection = &conn_;
    args_.nsubs = 3;
    args_.chunkSize = 128;
    args_.specifics.collnetDirect.nNodes = 4;
    args_.specifics.collnetDirect.node = 1;
    args_.specifics.collnetDirect.sizePerRank = 512;
  }
  ssize_t Region(ncclFunc_t coll, int isRecv, int sub, uint64_t step, int side) {
    args_.coll = coll;
    return calcRegionOffset(&args_, isRecv, sub, step, side);
  }
};

TEST_F(CollNetOffsetsMicrotest, AlgoOffset_WholeWindow_StepMajorChunksCappedAtWindowEnd) {
  EXPECT_EQ(calcAlgoOffset(&args_, 1, /*sub=*/1, /*step=*/2), 896u);  // (2*3+1)*128
  EXPECT_EQ(calcAlgoOffset(&args_, 1, 2, 5), 2048u);                  // 2176 capped at 4*512
}

TEST_F(CollNetOffsetsMicrotest, AlgoOffset_OwnRankWindow_ClampedToThisNodesSlice) {
  EXPECT_EQ(calcAlgoOffset(&args_, 0, 1, 0), 512u);   // 128 raised to node 1's start
  EXPECT_EQ(calcAlgoOffset(&args_, 0, 2, 1), 640u);   // inside
  EXPECT_EQ(calcAlgoOffset(&args_, 0, 2, 2), 1024u);  // exactly the end
  EXPECT_EQ(calcAlgoOffset(&args_, 0, 0, 3), 1024u);  // 1152 lowered to node 1's end
}

TEST_F(CollNetOffsetsMicrotest, RegionOffset_AllReduce_ChunkSlotsWithinStepSlot) {
  EXPECT_EQ(Region(ncclFuncAllReduce, 0, /*sub=*/1, /*step=*/9, /*side=*/0), 2176);  // slot 1: 1*2*1024 + 128
  EXPECT_EQ(Region(ncclFuncAllReduce, 0, 1, 9, 1), 2304);
  EXPECT_EQ(Region(ncclFuncAllReduce, 1, 1, 9, 0), 18560);  // recv half starts NCCL_STEPS slots in
}

TEST_F(CollNetOffsetsMicrotest, RegionOffset_GatherAndScatter_SendOwnSliceRecvWholeWindowOrBack) {
  EXPECT_EQ(Region(ncclFuncAllGather, 0, 1, 1, 0), 2048);       // send: own slice, empty below 512
  EXPECT_EQ(Region(ncclFuncAllGather, 0, 1, 1, 1), 2176);
  EXPECT_EQ(Region(ncclFuncAllGather, 1, 1, 1, 1), 18688);      // recv: whole window
  EXPECT_EQ(Region(ncclFuncReduceScatter, 0, 1, 1, 1), 2304);   // send: whole window
  EXPECT_EQ(Region(ncclFuncReduceScatter, 1, 1, 1, 1), 18560);  // recv: own slice
}

TEST_F(CollNetOffsetsMicrotest, RegionOffset_SecondGroup_StartsAtItsFirstSubsSlot) {
  args_.nsubs = 10;

  EXPECT_EQ(Region(ncclFuncAllGather, 1, /*sub=*/9, 0, 0), 24704);  // 16384 + 8*1024 + (1152-1024)
  EXPECT_EQ(Region(ncclFuncAllGather, 1, 9, 0, 1), 24832);
}

// ---------------------------------------------------------------------------
// Collective issue helpers (collNet[Reg]Iallreduce, collNet[Reg]Iallgather,
// collNet[Reg]Ireducescatter)
// ---------------------------------------------------------------------------

class CollNetIssueMicrotest : public CollNetMicrotestBase {
 protected:
  sendResources res_{};
  ncclProxyArgs args_{};
  ncclProxySubArgs* sub_ = &args_.subs[0];
  alignas(16) char region_[1 << 12] = {};
  alignas(16) uint8_t send_[1 << 12] = {}, recv_[1 << 12] = {};
  void* request_ = reinterpret_cast<void*>(0x1);
  void* poolSendMh_ = reinterpret_cast<void*>(0x51);
  void* poolRecvMh_ = reinterpret_cast<void*>(0x52);
  void* userSendMh_ = reinterpret_cast<void*>(0x61);
  void* userRecvMh_ = reinterpret_cast<void*>(0x62);

  void SetUp() override {
    CollNetMicrotestBase::SetUp();
    res_.collNetComm = &net_.collComm;
    res_.maxCollBytes = 1024;
    res_.sendMhandles[NCCL_PROTO_SIMPLE] = poolSendMh_;
    res_.recvMhandles[NCCL_PROTO_SIMPLE] = poolRecvMh_;
    NCCL_NET_MAP_ADD_POINTER(&res_.map, 1, 1, sizeof(region_), buffs[NCCL_PROTO_SIMPLE]);
    res_.map.mems[NCCL_NET_MAP_SHARED_DEVMEM] = {region_, nullptr, (int)sizeof(region_)};  // {gpu, cpu}
    args_.dtype = ncclFloat32;
    args_.redOp = ncclMax;
    args_.chunkSize = 256;
    args_.specifics.collnetDirect.sizePerRank = 1024;
    args_.specifics.collnetDirect.node = 1;
    sub_->sendbuff = send_;
    sub_->recvbuff = recv_;
    sub_->sendMhandle = userSendMh_;
    sub_->recvMhandle = userRecvMh_;
  }
};

TEST_F(CollNetIssueMicrotest, Iallreduce_ReducesPoolSlicesInPlaceOfElements) {
  ASSERT_EQ(collNetIallreduce(&proxyState_, &res_, &args_, sub_, /*nBytes=*/512, /*sendBeg=*/64, /*recvBeg=*/2048,
                              &request_),
            ncclSuccess);

  ASSERT_EQ(net_.iallreduces.size(), 1u);
  const auto& c = net_.iallreduces[0];
  EXPECT_EQ(c.comm, &net_.collComm);
  EXPECT_EQ(c.send, region_ + 64);
  EXPECT_EQ(c.recv, region_ + 2048);
  EXPECT_EQ(c.count, 128u);
  EXPECT_EQ(c.dtype, ncclFloat32);
  EXPECT_EQ(c.op, ncclMax);
  EXPECT_EQ(c.sendMhandle, poolSendMh_);
  EXPECT_EQ(c.recvMhandle, poolRecvMh_);
  EXPECT_EQ(request_, &net_.requestTokens[0]);
}

TEST_F(CollNetIssueMicrotest, RegIallreduce_OneRankPerNode_ReducesUserBuffersUpToMaxAndAdvances) {
  sub_->isOneRPN = 1;
  sub_->nbytes = 1500;
  ssize_t nBytes = 7;  // ignored: the whole remaining user buffer, capped at maxCollBytes

  ASSERT_EQ(collNetRegIallreduce(&proxyState_, &res_, &args_, sub_, /*groupStart=*/0, &nBytes, &request_),
            ncclSuccess);

  ASSERT_EQ(net_.iallreduces.size(), 1u);
  const auto& c = net_.iallreduces[0];
  EXPECT_EQ(c.send, send_);
  EXPECT_EQ(c.recv, recv_);
  EXPECT_EQ(c.count, 256u);
  EXPECT_EQ(c.sendMhandle, userSendMh_);
  EXPECT_EQ(c.recvMhandle, userRecvMh_);
  EXPECT_EQ(nBytes, 1024);
  EXPECT_EQ(sub_->nbytes, 476);
  EXPECT_EQ(sub_->sendbuff, send_ + 1024);
  EXPECT_EQ(sub_->recvbuff, recv_ + 1024);
}

TEST_F(CollNetIssueMicrotest, RegIallreduce_MultiRank_ReducesGroupWindowAndAdvancesByLoop) {
  sub_->nbytes = 3000;
  sub_->loopOffset = 512;
  sub_->loopSize = 2048;
  ssize_t nBytes = 2048;

  ASSERT_EQ(collNetRegIallreduce(&proxyState_, &res_, &args_, sub_, /*groupStart=*/8, &nBytes, &request_),
            ncclSuccess);

  ASSERT_EQ(net_.iallreduces.size(), 1u);
  EXPECT_EQ(net_.iallreduces[0].send, send_ + 512 + 8 * 256);
  EXPECT_EQ(net_.iallreduces[0].recv, recv_ + 512 + 8 * 256);
  EXPECT_EQ(net_.iallreduces[0].count, (3000u - 2560u) / 4);  // what is left past the window start
  EXPECT_EQ(nBytes, 440);
  EXPECT_EQ(sub_->nbytes, 3000 - 2048);
  EXPECT_EQ(sub_->sendbuff, send_ + 2048);
  EXPECT_EQ(sub_->recvbuff, recv_ + 2048);
}

TEST_F(CollNetIssueMicrotest, RegIallreduce_NothingLeftOrPluginBusy_LeavesBuffers) {
  sub_->nbytes = 1024;
  sub_->loopSize = 2048;
  ssize_t nBytes = 2048;
  ASSERT_EQ(collNetRegIallreduce(&proxyState_, &res_, &args_, sub_, /*groupStart=*/4, &nBytes, &request_),
            ncclSuccess);
  EXPECT_TRUE(net_.iallreduces.empty());  // the group's window starts at the end of the buffer
  EXPECT_EQ(nBytes, 0);
  EXPECT_EQ(request_, reinterpret_cast<void*>(0x1));

  net_.issueReady = false;
  sub_->isOneRPN = 1;
  ASSERT_EQ(collNetRegIallreduce(&proxyState_, &res_, &args_, sub_, 0, &nBytes, &request_), ncclSuccess);
  EXPECT_EQ(net_.iallreduces.size(), 1u);
  EXPECT_EQ(request_, nullptr);
  EXPECT_EQ(sub_->nbytes, 1024);
  EXPECT_EQ(sub_->sendbuff, send_);
  EXPECT_EQ(sub_->recvbuff, recv_);
}

TEST_F(CollNetIssueMicrotest, Iallgather_GathersPoolSliceIntoPoolWindow) {
  ASSERT_EQ(collNetIallgather(&proxyState_, &res_, &args_, sub_, /*nBytes=*/768, /*allBeg=*/1280, /*sendBeg=*/64,
                              /*recvBeg=*/2048, poolSendMh_, poolRecvMh_, &request_),
            ncclSuccess);

  ASSERT_EQ(net_.iallgathers.size(), 1u);
  const auto& c = net_.iallgathers[0];
  EXPECT_EQ(c.comm, &net_.collComm);
  EXPECT_EQ(c.send, region_ + 64);
  EXPECT_EQ(c.sendMhandle, poolSendMh_);
  ASSERT_EQ(c.parts.size(), 1u);
  EXPECT_EQ(c.parts[0].mhandle, poolRecvMh_);
  EXPECT_EQ(c.parts[0].address, region_ + 2048);
  EXPECT_EQ(c.parts[0].size, 768u);
  EXPECT_EQ(c.bytesPerRank, 1024u);
  EXPECT_EQ(c.windowOffset, 1280u);
  EXPECT_EQ(c.windowBytes, 768u);
  EXPECT_NE(request_, nullptr);
}

TEST_F(CollNetIssueMicrotest, RegIallgather_OneRankPerNode_GathersStraightIntoUserBufferAndAdvances) {
  sub_->isOneRPN = 1;
  sub_->nbytes = 1500;
  sub_->offset = 1100;  // node 1's window: send from 76 bytes into the user send buffer

  ASSERT_EQ(collNetRegIallgather(&proxyState_, &res_, &args_, sub_, /*nBytesIn=*/9, /*allBeg=*/9, /*recvBeg=*/9,
                                 poolRecvMh_, &request_),
            ncclSuccess);

  ASSERT_EQ(net_.iallgathers.size(), 1u);
  const auto& c = net_.iallgathers[0];
  EXPECT_EQ(c.send, send_ + 76);
  EXPECT_EQ(c.sendMhandle, userSendMh_);
  ASSERT_EQ(c.parts.size(), 1u);
  EXPECT_EQ(c.parts[0].mhandle, userRecvMh_);
  EXPECT_EQ(c.parts[0].address, recv_);
  EXPECT_EQ(c.parts[0].size, 1024u);
  EXPECT_EQ(c.windowOffset, 1100u);
  EXPECT_EQ(c.windowBytes, 1024u);
  EXPECT_EQ(sub_->recvbuff, recv_ + 1024);
  EXPECT_EQ(sub_->nbytes, 476);
  EXPECT_EQ(sub_->offset, 2124u);
  EXPECT_EQ(sub_->sendbuff, send_);
}

TEST_F(CollNetIssueMicrotest, RegIallgather_MultiRank_GathersIntoPoolWithoutAdvancing) {
  sub_->nbytes = 1500;

  ASSERT_EQ(collNetRegIallgather(&proxyState_, &res_, &args_, sub_, /*nBytesIn=*/512, /*allBeg=*/2304,
                                 /*recvBeg=*/4000 - 1500, poolRecvMh_, &request_),
            ncclSuccess);

  const auto& c = net_.iallgathers.at(0);
  EXPECT_EQ(c.send, send_);  // window starts on another node: send from the start
  EXPECT_EQ(c.parts[0].mhandle, poolRecvMh_);
  EXPECT_EQ(c.parts[0].address, region_ + 2500);
  EXPECT_EQ(c.parts[0].size, 512u);
  EXPECT_EQ(c.windowOffset, 2304u);
  EXPECT_EQ(sub_->nbytes, 1500);
  EXPECT_EQ(sub_->recvbuff, recv_);
}

TEST_F(CollNetIssueMicrotest, Ireducescatter_ScattersPoolWindowIntoPoolSlice) {
  ASSERT_EQ(collNetIreducescatter(&proxyState_, &res_, &args_, sub_, /*nBytes=*/768, /*allBeg=*/1280,
                                  /*sendBeg=*/64, /*recvBeg=*/2048, poolSendMh_, poolRecvMh_, &request_),
            ncclSuccess);

  ASSERT_EQ(net_.ireducescatters.size(), 1u);
  const auto& c = net_.ireducescatters[0];
  EXPECT_EQ(c.comm, &net_.collComm);
  ASSERT_EQ(c.parts.size(), 1u);
  EXPECT_EQ(c.parts[0].mhandle, poolSendMh_);
  EXPECT_EQ(c.parts[0].address, region_ + 64);
  EXPECT_EQ(c.parts[0].size, 768u);
  EXPECT_EQ(c.recv, region_ + 2048);
  EXPECT_EQ(c.recvMhandle, poolRecvMh_);
  EXPECT_EQ(c.bytesPerRank, 1024u);
  EXPECT_EQ(c.windowOffset, 1280u);
  EXPECT_EQ(c.windowBytes, 768u);
  EXPECT_EQ(c.dtype, ncclFloat32);
  EXPECT_EQ(c.op, ncclMax);
}

TEST_F(CollNetIssueMicrotest, RegIreducescatter_OneRankPerNode_ScattersStraightFromUserBufferAndAdvances) {
  sub_->isOneRPN = 1;
  sub_->nbytes = 1500;
  sub_->offset = 1100;

  ASSERT_EQ(collNetRegIreducescatter(&proxyState_, &res_, &args_, sub_, /*nBytesIn=*/9, /*allBeg=*/9,
                                     /*sendBeg=*/9, poolSendMh_, &request_),
            ncclSuccess);

  const auto& c = net_.ireducescatters.at(0);
  EXPECT_EQ(c.parts[0].mhandle, userSendMh_);
  EXPECT_EQ(c.parts[0].address, send_);
  EXPECT_EQ(c.parts[0].size, 1024u);
  EXPECT_EQ(c.recv, recv_ + 76);
  EXPECT_EQ(c.recvMhandle, userRecvMh_);
  EXPECT_EQ(c.windowOffset, 1100u);
  EXPECT_EQ(c.windowBytes, 1024u);
  EXPECT_EQ(c.dtype, ncclFloat32);
  EXPECT_EQ(c.op, ncclMax);
  EXPECT_EQ(sub_->sendbuff, send_ + 1024);
  EXPECT_EQ(sub_->nbytes, 476);
  EXPECT_EQ(sub_->offset, 2124u);
  EXPECT_EQ(sub_->recvbuff, recv_);
}

TEST_F(CollNetIssueMicrotest, RegIreducescatter_MultiRank_ScattersFromPoolWithoutAdvancing) {
  sub_->nbytes = 1500;

  ASSERT_EQ(collNetRegIreducescatter(&proxyState_, &res_, &args_, sub_, /*nBytesIn=*/512, /*allBeg=*/2304,
                                     /*sendBeg=*/96, poolSendMh_, &request_),
            ncclSuccess);

  const auto& c = net_.ireducescatters.at(0);
  EXPECT_EQ(c.parts[0].mhandle, poolSendMh_);
  EXPECT_EQ(c.parts[0].address, region_ + 96);
  EXPECT_EQ(c.parts[0].size, 512u);
  EXPECT_EQ(c.recv, recv_);
  EXPECT_EQ(c.windowOffset, 2304u);
  EXPECT_EQ(sub_->nbytes, 1500);
  EXPECT_EQ(sub_->sendbuff, send_);
}

TEST_F(CollNetIssueMicrotest, RegGatherScatter_PluginBusy_LeavesUserBuffers) {
  net_.issueReady = false;
  sub_->isOneRPN = 1;
  sub_->nbytes = 1500;

  ASSERT_EQ(collNetRegIallgather(&proxyState_, &res_, &args_, sub_, 0, 0, 0, poolRecvMh_, &request_), ncclSuccess);
  ASSERT_EQ(collNetRegIreducescatter(&proxyState_, &res_, &args_, sub_, 0, 0, 0, poolSendMh_, &request_),
            ncclSuccess);

  EXPECT_EQ(request_, nullptr);
  EXPECT_EQ(sub_->nbytes, 1500);
  EXPECT_EQ(sub_->offset, 0u);
  EXPECT_EQ(sub_->sendbuff, send_);
  EXPECT_EQ(sub_->recvbuff, recv_);
}

TEST_F(CollNetIssueMicrotest, Issue_PluginFails_ReturnsError) {
  net_.issueResult = ncclSystemError;
  sub_->isOneRPN = 1;
  sub_->nbytes = 64;
  ssize_t nBytes = 64;

  EXPECT_EQ(collNetIallreduce(&proxyState_, &res_, &args_, sub_, 64, 0, 0, &request_), ncclSystemError);
  EXPECT_EQ(collNetRegIallreduce(&proxyState_, &res_, &args_, sub_, 0, &nBytes, &request_), ncclSystemError);
  EXPECT_EQ(collNetIallgather(&proxyState_, &res_, &args_, sub_, 64, 0, 0, 0, nullptr, nullptr, &request_),
            ncclSystemError);
  EXPECT_EQ(collNetRegIallgather(&proxyState_, &res_, &args_, sub_, 64, 0, 0, nullptr, &request_), ncclSystemError);
  EXPECT_EQ(collNetIreducescatter(&proxyState_, &res_, &args_, sub_, 64, 0, 0, 0, nullptr, nullptr, &request_),
            ncclSystemError);
  EXPECT_EQ(collNetRegIreducescatter(&proxyState_, &res_, &args_, sub_, 64, 0, 0, nullptr, &request_),
            ncclSystemError);
}

// ---------------------------------------------------------------------------
// Send proxy progress (sendProxyProgress)
// ---------------------------------------------------------------------------

// Up to kMaxSubs subs, each on its own channel's connection over one shared pool of 128-byte
// chunks: 2 channels x 8 KiB per direction, so 1 KiB per step slot.
class CollNetProgressMicrotestBase : public CollNetMicrotestBase {
 protected:
  static constexpr int kMaxSubs = 9;
  static constexpr int kChunk = 128;
  static constexpr int kSlot = 1024;
  static constexpr uint64_t kBase = 16;  // where each connection's steps start

  ncclCollNetSharedRes shared_{};
  ncclProxyConnection conns_[kMaxSubs]{};
  std::unique_ptr<ncclSendMem[]> sendMems_{new ncclSendMem[kMaxSubs]()};
  std::unique_ptr<ncclRecvMem[]> recvMems_{new ncclRecvMem[kMaxSubs]()};
  std::vector<char> region_ = std::vector<char>(2 * NCCL_STEPS * 2 * kSlot);
  ncclProxyArgs args_{};
  int gpuSteps_[kMaxSubs] = {};

  void SetUp() override {
    CollNetMicrotestBase::SetUp();
    shared_.buffSize = NCCL_STEPS * kSlot;
    shared_.nChannels = 2;
    args_.state = ncclProxyOpReady;
    args_.sliceSteps = 1;
    args_.chunkSteps = 1;
    args_.chunkSize = kChunk;
    args_.dtype = ncclFloat32;
    args_.redOp = ncclSum;
    args_.coll = ncclFuncAllReduce;
    args_.specifics.collnetDirect.nNodes = 4;
    args_.specifics.collnetDirect.node = 1;
    args_.specifics.collnetDirect.sizePerRank = 4 * kChunk;
    for (int s = 0; s < kMaxSubs; s++) {
      conns_[s].collNet = &shared_;
      args_.subs[s].connection = &conns_[s];
      for (auto& f : recvMems_[s].connFifo) f.size = -1;  // as sendConnect leaves it
    }
  }

  void Subs(int n, int nsteps) {
    args_.nsubs = n;
    for (int s = 0; s < n; s++) args_.subs[s].nsteps = nsteps;
  }
  ncclRecvMem* RecvMem(int s) { return &recvMems_[s]; }
  int Slot(int s, uint64_t step) { return (args_.subs[s].base + step) % NCCL_STEPS; }
  // The GPU hands the proxy its next step: the bytes in the FIFO slot, then the tail.
  void GpuFills(int s, ssize_t bytes) {
    int step = gpuSteps_[s]++;
    RecvMem(s)->connFifo[(args_.subs[s].base + step) % NCCL_STEPS].size = bytes;
    RecvMem(s)->tail = args_.subs[s].base + gpuSteps_[s];
  }
};

class CollNetSendProgressMicrotest : public CollNetProgressMicrotestBase {
 protected:
  sendResources res_[kMaxSubs]{};
  reqSlot fifo_[kMaxSubs][COLLNET_MAX_GROUPS][NCCL_STEPS]{};
  void* poolSendMh_ = reinterpret_cast<void*>(0x51);
  void* poolRecvMh_ = reinterpret_cast<void*>(0x52);

  void SetUp() override {
    CollNetProgressMicrotestBase::SetUp();
    for (int s = 0; s < kMaxSubs; s++) {
      sendResources& r = res_[s];
      r.collNetComm = &net_.collComm;
      r.sendMem = &sendMems_[s];
      r.recvMem = &recvMems_[s];
      r.reqFifo = fifo_[s];
      r.step = kBase;
      r.maxCollBytes = 1024;
      r.sendMhandles[NCCL_PROTO_SIMPLE] = poolSendMh_;
      r.recvMhandles[NCCL_PROTO_SIMPLE] = poolRecvMh_;
      NCCL_NET_MAP_ADD_POINTER(&r.map, 1, 1, (int)region_.size(), buffs[NCCL_PROTO_SIMPLE]);
      r.map.mems[NCCL_NET_MAP_SHARED_DEVMEM].gpuPtr = region_.data();
      conns_[s].transportResources = &r;
    }
  }

  ncclResult_t Progress() { return sendProxyProgress(&proxyState_, &args_); }
  // The recv proxy hands a step's shared slot to the send side.
  void RecvTurn(int s, int group, uint64_t step) { fifo_[s][group][Slot(s, step)].turnIsSendNotRecv = true; }
  // Drives every sub's next step from the GPU to the plugin.
  void FillAndTurn(int n, uint64_t step, ssize_t bytes = kChunk) {
    for (int s = 0; s < n; s++) GpuFills(s, bytes);
    for (int s = 0; s < n; s++) RecvTurn(s, s / COLLNET_GROUP_NSUBS, step);
  }
};

TEST_F(CollNetSendProgressMicrotest, Ready_StartsEachSubAtNextChunkBoundaryAndReservesSteps) {
  Subs(2, 3);
  res_[1].step = 21;
  args_.chunkSteps = 4;
  args_.hdp_flushed = 7;
  args_.subs[0].received = args_.subs[0].transmitted = args_.subs[0].done = 5;

  ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(args_.state, ncclProxyOpProgress);
  EXPECT_EQ(args_.hdp_flushed, 0u);
  EXPECT_EQ(args_.subs[0].base, kBase);
  EXPECT_EQ(res_[0].step, kBase + 3);
  EXPECT_EQ(args_.subs[1].base, 24u);
  EXPECT_EQ(res_[1].step, 27u);
  EXPECT_EQ(args_.subs[0].received, 0u);
  EXPECT_EQ(args_.subs[0].transmitted, 0u);
  EXPECT_EQ(args_.subs[0].done, 0u);
  EXPECT_EQ(args_.subs[0].posted, 1u);  // and the first slot goes straight out
}

TEST_F(CollNetSendProgressMicrotest, Ready_RegisteredOneRankPerNode_OneStepPerMaxCollBytes) {
  Subs(3, 1);
  for (int s = 0; s < 3; s++) args_.subs[s].nbytes = 2049;
  args_.subs[0].reg = args_.subs[0].isOneRPN = 1;
  args_.subs[1].reg = 1;       // several ranks per node
  args_.subs[2].isOneRPN = 1;  // not registered

  ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(args_.subs[0].nsteps, 3);
  EXPECT_EQ(args_.subs[1].nsteps, 1);
  EXPECT_EQ(args_.subs[2].nsteps, 1);
}

TEST_F(CollNetSendProgressMicrotest, Post_FillsWindowWithSendSlotsAndCredits) {
  Subs(2, 12);
  args_.subs[0].isOneRPN = args_.subs[1].isOneRPN = 1;  // unregistered: a credit per step regardless

  ASSERT_EQ(Progress(), ncclSuccess);
  EXPECT_EQ(sendMems_[1].head, kBase + 1 - NCCL_STEPS);
  EXPECT_EQ(args_.idle, 1);  // posting alone is not progress

  for (int i = 0; i < 10; i++) ASSERT_EQ(Progress(), ncclSuccess);

  for (int s = 0; s < 2; s++) {
    EXPECT_EQ(args_.subs[s].posted, (uint64_t)NCCL_STEPS);  // a full window ahead of done
    EXPECT_EQ(sendMems_[s].head, kBase);
    for (int step = 0; step < NCCL_STEPS; step++) {
      EXPECT_EQ(RecvMem(s)->connFifo[Slot(s, step)].offset, (ssize_t)(step * 2 * kSlot + s * kChunk));
    }
  }
}

TEST_F(CollNetSendProgressMicrotest, Post_GdcSync_CreditsGoToGdcWord) {
  Subs(1, 4);
  uint64_t gdc = 0;
  res_[0].gdcSync = &gdc;

  ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(gdc, kBase + 1 - NCCL_STEPS);
  EXPECT_EQ(sendMems_[0].head, 0u);
}

TEST_F(CollNetSendProgressMicrotest, Post_RegisteredOneRankPerNode_SingleCreditNoOffsets) {
  Subs(1, 1);
  args_.subs[0].reg = 1;
  args_.subs[0].isOneRPN = 1;
  args_.subs[0].nbytes = 3 * 1024;
  for (auto& f : RecvMem(0)->connFifo) f.offset = -7;

  for (int i = 0; i < 3; i++) ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(args_.subs[0].posted, 3u);
  EXPECT_EQ(sendMems_[0].head, kBase + 1 - NCCL_STEPS);
  for (auto& f : RecvMem(0)->connFifo) EXPECT_EQ(f.offset, -7);

  args_.state = ncclProxyOpReady;
  args_.coll = ncclFuncReduceScatter;
  args_.subs[0].nsteps = 1;
  ASSERT_EQ(Progress(), ncclSuccess);
  for (auto& f : RecvMem(0)->connFifo) EXPECT_EQ(f.offset, -7);
}

TEST_F(CollNetSendProgressMicrotest, Post_RegisteredSeveralRanksPerNode_CreditsEveryStepOffsetsOnlyForScatter) {
  Subs(1, 2);
  args_.subs[0].reg = 1;
  for (auto& f : RecvMem(0)->connFifo) f.offset = -7;
  for (int i = 0; i < 2; i++) ASSERT_EQ(Progress(), ncclSuccess);
  EXPECT_EQ(sendMems_[0].head, kBase + 2 - NCCL_STEPS);
  EXPECT_EQ(RecvMem(0)->connFifo[Slot(0, 1)].offset, -7);

  args_.state = ncclProxyOpReady;
  args_.coll = ncclFuncReduceScatter;
  ASSERT_EQ(Progress(), ncclSuccess);
  EXPECT_EQ(RecvMem(0)->connFifo[Slot(0, 0)].offset, 0);  // node 1's slice, empty before it
}

TEST_F(CollNetSendProgressMicrotest, Receive_GpuFilledSlot_ConsumesItAndFreesSlot) {
  Subs(1, 4);
  ASSERT_EQ(Progress(), ncclSuccess);
  GpuFills(0, 999);  // AllReduce sizes are not checked

  ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(args_.subs[0].received, 1u);
  EXPECT_EQ(RecvMem(0)->connFifo[Slot(0, 0)].size, -1);
  EXPECT_EQ(args_.idle, 0);
}

TEST_F(CollNetSendProgressMicrotest, Receive_StepNeverPosted_Ignored) {
  Subs(1, 1);
  ASSERT_EQ(Progress(), ncclSuccess);
  GpuFills(0, kChunk);
  GpuFills(0, kChunk);  // more than the op has

  for (int i = 0; i < 3; i++) ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(args_.subs[0].posted, 1u);
  EXPECT_EQ(args_.subs[0].received, 1u);
  EXPECT_EQ(sendMems_[0].head, kBase + 1 - NCCL_STEPS);
}

TEST_F(CollNetSendProgressMicrotest, Receive_RegisteredSeveralRanksPerNode_TakesOnlyStepsGpuSignalled) {
  Subs(1, 4);
  args_.subs[0].reg = 1;
  for (int i = 0; i < 4; i++) ASSERT_EQ(Progress(), ncclSuccess);
  RecvMem(0)->tail = kBase + 1;

  for (int i = 0; i < 3; i++) ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(args_.subs[0].received, 1u);
}

TEST_F(CollNetSendProgressMicrotest, Receive_SlotNotPublishedOrTailNotPast_Waits) {
  Subs(1, 4);
  ASSERT_EQ(Progress(), ncclSuccess);
  RecvMem(0)->connFifo[Slot(0, 0)].size = -1;
  RecvMem(0)->tail = kBase + 1;
  ASSERT_EQ(Progress(), ncclSuccess);
  EXPECT_EQ(args_.subs[0].received, 0u);

  RecvMem(0)->connFifo[Slot(0, 0)].size = kChunk;
  RecvMem(0)->tail = kBase;
  ASSERT_EQ(Progress(), ncclSuccess);
  EXPECT_EQ(args_.subs[0].received, 0u);
  EXPECT_EQ(args_.idle, 1);

  args_.subs[0].isOneRPN = 1;  // unregistered: still one tail step per step
  RecvMem(0)->tail = kBase + 1;
  RecvMem(0)->connFifo[Slot(0, 1)].size = kChunk;  // step 1 sized, its tail not yet bumped
  for (int i = 0; i < 3; i++) ASSERT_EQ(Progress(), ncclSuccess);
  EXPECT_EQ(args_.subs[0].received, 1u);
}

TEST_F(CollNetSendProgressMicrotest, Receive_GatherSizeNotTheSendSlot_ReturnsInternalError) {
  Subs(1, 4);
  args_.coll = ncclFuncAllGather;
  args_.specifics.collnetDirect.node = 0;  // step 0 sends a whole chunk of this node's slice
  ASSERT_EQ(Progress(), ncclSuccess);
  GpuFills(0, kChunk - 1);
  EXPECT_EQ(Progress(), ncclInternalError);

  RecvMem(0)->connFifo[Slot(0, 0)].size = kChunk;
  ASSERT_EQ(Progress(), ncclSuccess);
  EXPECT_EQ(args_.subs[0].received, 1u);
}

TEST_F(CollNetSendProgressMicrotest, Receive_Registered_WaitsOnTailAlone) {
  Subs(1, 1);
  args_.subs[0].reg = 1;
  args_.subs[0].isOneRPN = 1;
  args_.subs[0].nbytes = 3 * 1024;
  args_.coll = ncclFuncAllGather;
  for (int i = 0; i < 3; i++) ASSERT_EQ(Progress(), ncclSuccess);
  RecvMem(0)->tail = kBase + 1;  // the device signals a registered op once, with no FIFO size

  for (int i = 0; i < 3; i++) ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(args_.subs[0].received, 3u);
}

TEST_F(CollNetSendProgressMicrotest, Transmit_RecvProxyHoldsSlot_Waits) {
  Subs(1, 2);
  ASSERT_EQ(Progress(), ncclSuccess);
  GpuFills(0, kChunk);

  for (int i = 0; i < 3; i++) ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(args_.subs[0].transmitted, 0u);
  EXPECT_TRUE(net_.iallreduces.empty());
}

TEST_F(CollNetSendProgressMicrotest, Transmit_AllReduce_LastSubIssuesWholeGroupAndSizesRecvSlot) {
  Subs(2, 2);
  ASSERT_EQ(Progress(), ncclSuccess);
  FillAndTurn(2, 0);

  ASSERT_EQ(Progress(), ncclSuccess);

  ASSERT_EQ(net_.iallreduces.size(), 1u);
  const auto& c = net_.iallreduces[0];
  EXPECT_EQ(c.comm, &net_.collComm);
  EXPECT_EQ(c.send, region_.data());                                  // step 0, subs 0..1
  EXPECT_EQ(c.recv, region_.data() + NCCL_STEPS * 2 * kSlot);
  EXPECT_EQ(c.count, 2u * kChunk / sizeof(float));
  EXPECT_EQ(c.sendMhandle, poolSendMh_);
  EXPECT_EQ(c.recvMhandle, poolRecvMh_);
  EXPECT_EQ(fifo_[1][0][Slot(1, 0)].size, 2 * kChunk);
  EXPECT_EQ(args_.subs[0].transmitted, 1u);
  EXPECT_EQ(args_.subs[1].transmitted, 1u);
  EXPECT_EQ(args_.subs[1].requests[Slot(1, 0)], &net_.requestTokens[0]);
}

TEST_F(CollNetSendProgressMicrotest, Transmit_LaterSubAheadOfEarlierOne_WaitsForIt) {
  Subs(2, 2);
  ASSERT_EQ(Progress(), ncclSuccess);
  GpuFills(1, kChunk);
  RecvTurn(1, 0, 0);

  ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(args_.subs[1].received, 1u);
  EXPECT_EQ(args_.subs[1].transmitted, 0u);
  EXPECT_TRUE(net_.iallreduces.empty());
}

TEST_F(CollNetSendProgressMicrotest, Transmit_PluginBusy_RetriesOnNextCall) {
  Subs(1, 2);
  ASSERT_EQ(Progress(), ncclSuccess);
  FillAndTurn(1, 0);
  net_.issueReady = false;
  ASSERT_EQ(Progress(), ncclSuccess);
  EXPECT_EQ(args_.subs[0].transmitted, 0u);

  net_.issueReady = true;
  ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(net_.iallreduces.size(), 2u);
  EXPECT_EQ(args_.subs[0].transmitted, 1u);
  EXPECT_EQ(args_.idle, 0);
}

TEST_F(CollNetSendProgressMicrotest, Transmit_TwoGroups_EachLastSubIssuesItsGroup) {
  Subs(9, 1);
  ASSERT_EQ(Progress(), ncclSuccess);
  FillAndTurn(9, 0);

  ASSERT_EQ(Progress(), ncclSuccess);

  ASSERT_EQ(net_.iallreduces.size(), 2u);
  EXPECT_EQ(net_.iallreduces[0].count, 8u * kChunk / sizeof(float));
  EXPECT_EQ(net_.iallreduces[1].send, region_.data() + 8 * kChunk);
  EXPECT_EQ(net_.iallreduces[1].count, 1u * kChunk / sizeof(float));
  EXPECT_EQ(fifo_[7][0][Slot(7, 0)].size, 8 * kChunk);
  EXPECT_EQ(fifo_[8][1][Slot(8, 0)].size, kChunk);
}

TEST_F(CollNetSendProgressMicrotest, Transmit_AllGather_GathersOwnSliceIntoWindow) {
  args_.coll = ncclFuncAllGather;
  args_.specifics.collnetDirect.node = 0;
  Subs(2, 2);
  ASSERT_EQ(Progress(), ncclSuccess);
  for (uint64_t step = 0; step < 2; step++) {
    FillAndTurn(2, step);
    for (int i = 0; i < 3; i++) ASSERT_EQ(Progress(), ncclSuccess);
  }

  EXPECT_TRUE(net_.ireducescatters.empty());
  ASSERT_EQ(net_.iallgathers.size(), 2u);
  const auto& g = net_.iallgathers[1];  // step 1: window bytes 256..512
  EXPECT_EQ(g.send, region_.data() + 2 * kSlot);  // slots are relative to the group start
  EXPECT_EQ(g.sendMhandle, poolSendMh_);
  EXPECT_EQ(g.parts.at(0).address, region_.data() + (NCCL_STEPS + 1) * 2 * kSlot);
  EXPECT_EQ(g.parts.at(0).mhandle, poolRecvMh_);
  EXPECT_EQ(g.parts.at(0).size, 2u * kChunk);
  EXPECT_EQ(g.windowOffset, 2u * kChunk);
  EXPECT_EQ(g.windowBytes, 2u * kChunk);
  EXPECT_EQ(g.bytesPerRank, 4u * kChunk);
}

TEST_F(CollNetSendProgressMicrotest, Transmit_ReduceScatter_ScattersWindowIntoOwnSlice) {
  args_.coll = ncclFuncReduceScatter;
  args_.specifics.collnetDirect.node = 0;
  Subs(2, 2);
  ASSERT_EQ(Progress(), ncclSuccess);
  for (uint64_t step = 0; step < 2; step++) {
    FillAndTurn(2, step);
    for (int i = 0; i < 3; i++) ASSERT_EQ(Progress(), ncclSuccess);
  }

  EXPECT_TRUE(net_.iallgathers.empty());
  ASSERT_EQ(net_.ireducescatters.size(), 2u);
  const auto& r = net_.ireducescatters[1];
  EXPECT_EQ(r.parts.at(0).address, region_.data() + 2 * kSlot);
  EXPECT_EQ(r.parts.at(0).mhandle, poolSendMh_);
  EXPECT_EQ(r.parts.at(0).size, 2u * kChunk);
  EXPECT_EQ(r.recv, region_.data() + (NCCL_STEPS + 1) * 2 * kSlot);
  EXPECT_EQ(r.recvMhandle, poolRecvMh_);
  EXPECT_EQ(r.windowOffset, 2u * kChunk);
  EXPECT_EQ(r.windowBytes, 2u * kChunk);
  EXPECT_EQ(r.dtype, ncclFloat32);
  EXPECT_EQ(r.op, ncclSum);
}

TEST_F(CollNetSendProgressMicrotest, Transmit_Registered_IssuesFromUserBuffers) {
  alignas(16) static uint8_t user[4096];
  for (ncclFunc_t coll : {ncclFuncAllReduce, ncclFuncAllGather, ncclFuncReduceScatter}) {
    SCOPED_TRACE(coll);
    args_.state = ncclProxyOpReady;
    args_.coll = coll;
    gpuSteps_[0] = 0;
    memset(fifo_, 0, sizeof(fifo_));
    Subs(1, 1);
    args_.subs[0].reg = 1;
    args_.subs[0].isOneRPN = 1;
    args_.subs[0].nbytes = 512;
    args_.subs[0].offset = 0;
    args_.subs[0].sendbuff = args_.subs[0].recvbuff = user;
    ASSERT_EQ(Progress(), ncclSuccess);
    FillAndTurn(1, 0);
    ASSERT_EQ(Progress(), ncclSuccess);
    EXPECT_EQ(args_.subs[0].transmitted, 1u);
  }

  EXPECT_EQ(net_.iallreduces.at(0).send, user);
  EXPECT_EQ(net_.iallgathers.at(0).parts.at(0).address, user);
  EXPECT_EQ(net_.ireducescatters.at(0).parts.at(0).address, user);
}

TEST_F(CollNetSendProgressMicrotest, Transmit_EmptyWindow_CompletesWithoutPlugin) {
  Subs(1, 1);
  args_.coll = ncclFuncAllGather;
  args_.specifics.collnetDirect.sizePerRank = 0;
  ASSERT_EQ(Progress(), ncclSuccess);
  FillAndTurn(1, 0, /*bytes=*/0);

  ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_TRUE(net_.iallgathers.empty());
  EXPECT_EQ(args_.subs[0].transmitted, 1u);
  EXPECT_EQ(args_.subs[0].requests[Slot(0, 0)], nullptr);
  ASSERT_EQ(Progress(), ncclSuccess);
  EXPECT_TRUE(net_.tested.empty());
  EXPECT_EQ(args_.state, ncclProxyOpNone);
}

TEST_F(CollNetSendProgressMicrotest, Transmit_RegisteredNothingLeft_CompletesWithoutPlugin) {
  Subs(1, 1);
  args_.subs[0].reg = 1;
  args_.subs[0].nbytes = 0;
  ASSERT_EQ(Progress(), ncclSuccess);
  FillAndTurn(1, 0);

  ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_TRUE(net_.iallreduces.empty());
  EXPECT_EQ(args_.subs[0].transmitted, 1u);
}

TEST_F(CollNetSendProgressMicrotest, Complete_PluginDone_HandsSlotBackAndRetiresGroup) {
  Subs(2, 2);
  ASSERT_EQ(Progress(), ncclSuccess);
  FillAndTurn(2, 0);
  ASSERT_EQ(Progress(), ncclSuccess);
  net_.testDone = 0;
  ASSERT_EQ(Progress(), ncclSuccess);
  EXPECT_EQ(args_.subs[1].done, 0u);
  EXPECT_TRUE(fifo_[1][0][Slot(1, 0)].turnIsSendNotRecv);

  net_.testDone = 1;
  ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(net_.tested.back(), &net_.requestTokens[0]);
  EXPECT_EQ(args_.subs[1].requests[Slot(1, 0)], nullptr);
  EXPECT_FALSE(fifo_[1][0][Slot(1, 0)].turnIsSendNotRecv);
  EXPECT_EQ(args_.subs[0].done, 1u);
  EXPECT_EQ(args_.subs[1].done, 1u);
  EXPECT_EQ(args_.state, ncclProxyOpProgress);
  EXPECT_EQ(args_.idle, 0);
}

TEST_F(CollNetSendProgressMicrotest, Complete_EverySubDone_StopsOp) {
  Subs(2, 2);
  args_.subs[2].nsteps = 99;  // a stale sub past nsubs, left by an earlier op
  ASSERT_EQ(Progress(), ncclSuccess);
  for (uint64_t step = 0; step < 2; step++) {
    FillAndTurn(2, step);
    for (int i = 0; i < 3 && args_.state != ncclProxyOpNone; i++) ASSERT_EQ(Progress(), ncclSuccess);
    EXPECT_EQ(args_.state, step == 1 ? ncclProxyOpNone : ncclProxyOpProgress);
  }

  EXPECT_EQ(net_.iallreduces.size(), 2u);
  EXPECT_EQ(args_.subs[0].done, 2u);
}

TEST_F(CollNetSendProgressMicrotest, WholeOp_LongerThanWindow_RunsEveryStepInSlotOrder) {
  constexpr int kSteps = NCCL_STEPS + 3;
  Subs(1, kSteps);
  ASSERT_EQ(Progress(), ncclSuccess);
  for (uint64_t step = 0; step < kSteps; step++) {
    FillAndTurn(1, step);
    for (int i = 0; i < 3 && args_.subs[0].done <= step; i++) ASSERT_EQ(Progress(), ncclSuccess);
    ASSERT_EQ(args_.subs[0].done, step + 1);
    EXPECT_FALSE(fifo_[0][0][Slot(0, step)].turnIsSendNotRecv);
  }

  EXPECT_EQ(args_.state, ncclProxyOpNone);
  ASSERT_EQ(net_.iallreduces.size(), (size_t)kSteps);
  for (int step = 0; step < kSteps; step++) {
    EXPECT_EQ(net_.iallreduces[step].send, region_.data() + (step % NCCL_STEPS) * 2 * kSlot);
  }
  EXPECT_EQ(sendMems_[0].head, kBase + kSteps - NCCL_STEPS);
}

TEST_F(CollNetSendProgressMicrotest, PluginFails_ReturnsError) {
  Subs(1, 2);
  ASSERT_EQ(Progress(), ncclSuccess);
  FillAndTurn(1, 0);
  net_.issueResult = ncclSystemError;
  EXPECT_EQ(Progress(), ncclSystemError);

  net_.issueResult = ncclSuccess;
  ASSERT_EQ(Progress(), ncclSuccess);
  net_.testResult = ncclSystemError;
  EXPECT_EQ(Progress(), ncclSystemError);
}

TEST_F(CollNetSendProgressMicrotest, NotStarted_DoesNothing) {
  Subs(1, 2);
  args_.state = ncclProxyOpNone;

  ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(args_.subs[0].posted, 0u);
  EXPECT_EQ(args_.idle, 1);
}

// ---------------------------------------------------------------------------
// Recv-side flush (collNetRecvFlush)
// ---------------------------------------------------------------------------

class CollNetRecvFlushMicrotest : public CollNetMicrotestBase {
 protected:
  recvResources res_{};
  ncclProxyArgs args_{};
  ncclProxySubArgs* sub_ = &args_.subs[0];
  alignas(16) char region_[1 << 12] = {};
  alignas(16) uint8_t recv_[1 << 13] = {};
  void* request_ = reinterpret_cast<void*>(0x1);
  void* poolMh_ = reinterpret_cast<void*>(0x52);
  void* userMh_ = reinterpret_cast<void*>(0x62);

  void SetUp() override {
    CollNetMicrotestBase::SetUp();
    res_.collNetComm = &net_.collComm;
    res_.maxCollBytes = 1024;
    res_.mhandles[NCCL_PROTO_SIMPLE] = poolMh_;
    NCCL_NET_MAP_ADD_POINTER(&res_.map, 1, 1, sizeof(region_), buffs[NCCL_PROTO_SIMPLE]);
    res_.map.mems[NCCL_NET_MAP_SHARED_DEVMEM] = {region_, nullptr, (int)sizeof(region_)};  // {gpu, cpu}
    args_.coll = ncclFuncAllReduce;
    args_.chunkSize = 256;
    args_.specifics.collnetDirect.sizePerRank = 1000;
    args_.specifics.collnetDirect.node = 2;
    sub_->recvbuff = recv_;
    sub_->recvMhandle = userMh_;
    sub_->reg = 1;
  }

  ncclResult_t Flush(int groupStart, ssize_t nBytesIn, ssize_t recvBeg = 0) {
    return collNetRecvFlush(&proxyState_, &res_, &args_, sub_, groupStart, nBytesIn, recvBeg, &request_);
  }
  void ExpectFlush(void* data, int size, void* mhandle) {
    ASSERT_EQ(net_.iflushes.size(), 1u);
    EXPECT_EQ(net_.iflushes[0].comm, &net_.collComm);
    EXPECT_EQ(net_.iflushes[0].data, data);
    EXPECT_EQ(net_.iflushes[0].size, size);
    EXPECT_EQ(net_.iflushes[0].mhandle, mhandle);
  }
};

TEST_F(CollNetRecvFlushMicrotest, Unregistered_FlushesGroupsPoolRange) {
  sub_->reg = 0;

  ASSERT_EQ(Flush(/*groupStart=*/8, /*nBytesIn=*/768, /*recvBeg=*/2048), ncclSuccess);

  ExpectFlush(region_ + 2048, 768, poolMh_);
  EXPECT_EQ(request_, &net_.requestTokens[0]);
}

TEST_F(CollNetRecvFlushMicrotest, RegisteredGatherSeveralRanksPerNode_FlushesPoolRange) {
  args_.coll = ncclFuncAllGather;

  ASSERT_EQ(Flush(0, 768, 2048), ncclSuccess);

  ExpectFlush(region_ + 2048, 768, poolMh_);
}

TEST_F(CollNetRecvFlushMicrotest, RegisteredGatherOneRankPerNode_FlushesUserBuffer) {
  args_.coll = ncclFuncAllGather;
  sub_->isOneRPN = 1;
  sub_->nbytes = 512;
  sub_->offset = 128;

  ASSERT_EQ(Flush(0, 9, 2048), ncclSuccess);

  ExpectFlush(recv_ + 128, 512, userMh_);
}

TEST_F(CollNetRecvFlushMicrotest, RegisteredOneRankPerNode_FlushesUserBufferUpToMaxAndAdvances) {
  sub_->isOneRPN = 1;
  sub_->nbytes = 1500;
  sub_->offset = 128;
  sub_->loopOffset = 64;

  ASSERT_EQ(Flush(/*groupStart=*/2, /*nBytesIn=*/9), ncclSuccess);

  ExpectFlush(recv_ + 128 + 2 * 256 + 64, 1024, userMh_);
  EXPECT_EQ(sub_->nbytes, 476);
  EXPECT_EQ(sub_->offset, 128u + 1024u);
}

TEST_F(CollNetRecvFlushMicrotest, RegisteredAllReduceSeveralRanksPerNode_FlushesGroupWindowAndAdvancesByLoop) {
  sub_->nbytes = 3000;
  sub_->loopOffset = 2500;
  sub_->loopSize = 2048;

  ASSERT_EQ(Flush(/*groupStart=*/1, /*nBytesIn=*/768), ncclSuccess);

  ExpectFlush(recv_ + 256 + 2500, 500, userMh_);  // what is left past the loop offset
  EXPECT_EQ(sub_->nbytes, 3000 - 2048);
  EXPECT_EQ(sub_->offset, 2048u);
}

TEST_F(CollNetRecvFlushMicrotest, RegisteredNothingLeftOrPluginBusy_DoesNotAdvance) {
  sub_->nbytes = 1000;
  sub_->loopOffset = 1000;
  sub_->loopSize = 2048;
  ASSERT_EQ(Flush(0, 768), ncclSuccess);
  EXPECT_TRUE(net_.iflushes.empty());
  EXPECT_EQ(request_, reinterpret_cast<void*>(0x1));

  sub_->loopOffset = 0;
  net_.issueReady = false;
  ASSERT_EQ(Flush(0, 768), ncclSuccess);
  EXPECT_EQ(net_.iflushes.size(), 1u);
  EXPECT_EQ(request_, nullptr);
  EXPECT_EQ(sub_->nbytes, 1000);
  EXPECT_EQ(sub_->offset, 0u);
}

// ReduceScatter: this node (2) receives only its 1000-byte slice of the group's window.
TEST_F(CollNetRecvFlushMicrotest, RegisteredScatter_FlushesOnlyThisNodesPartOfWindow) {
  args_.coll = ncclFuncReduceScatter;
  sub_->isOneRPN = 1;
  sub_->loopSize = 4096;
  struct Case {
    size_t offset;
    ssize_t nbytes;
    size_t wantOffset;
    int wantSize;
  } cases[] = {
      {2100, 1024, 100, 900},  // starts in this node's slice
      {2100, 500, 100, 500},   // and ends there
      {1500, 1024, 0, 524},    // ends in it
      {1500, 2000, 0, 1000},   // covers it
      {500, 1024, 0, 1024},    // ends before it: a dummy flush
      {3100, 500, 0, 500},     // starts after it: a dummy flush
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.offset);
    net_.iflushes.clear();
    sub_->offset = c.offset - 2 * 256;  // the same windows, reached through a later group
    sub_->nbytes = c.nbytes;
    res_.maxCollBytes = c.nbytes;

    ASSERT_EQ(Flush(/*groupStart=*/2, 0), ncclSuccess);

    ExpectFlush(recv_ + c.wantOffset, c.wantSize, userMh_);
  }
  for (const Case& c : cases) {
    SCOPED_TRACE(c.offset);
    net_.iflushes.clear();
    sub_->offset = c.offset;
    sub_->nbytes = c.nbytes;
    res_.maxCollBytes = c.nbytes;

    ASSERT_EQ(Flush(0, 0), ncclSuccess);

    ExpectFlush(recv_ + c.wantOffset, c.wantSize, userMh_);
  }
}

TEST_F(CollNetRecvFlushMicrotest, PluginFails_ReturnsError) {
  net_.iflushResult = ncclSystemError;
  sub_->nbytes = 64;
  EXPECT_EQ(Flush(0, 64), ncclSystemError);
  sub_->reg = 0;
  EXPECT_EQ(Flush(0, 64), ncclSystemError);
}

// ---------------------------------------------------------------------------
// Recv proxy progress (recvProxyProgress)
// ---------------------------------------------------------------------------

class CollNetRecvProgressMicrotest : public CollNetProgressMicrotestBase {
 protected:
  recvResources res_[kMaxSubs]{};
  void* poolMh_ = reinterpret_cast<void*>(0x52);

  void SetUp() override {
    CollNetProgressMicrotestBase::SetUp();
    for (int s = 0; s < kMaxSubs; s++) {
      recvResources& r = res_[s];
      r.collNetComm = &net_.collComm;
      r.sendMem = &sendMems_[s];
      r.recvMem = &recvMems_[s];
      r.step = kBase;
      r.maxCollBytes = 1024;
      r.mhandles[NCCL_PROTO_SIMPLE] = poolMh_;
      NCCL_NET_MAP_ADD_POINTER(&r.map, 1, 1, (int)region_.size(), buffs[NCCL_PROTO_SIMPLE]);
      r.map.mems[NCCL_NET_MAP_SHARED_DEVMEM].gpuPtr = region_.data();
      conns_[s].transportResources = &r;
    }
  }

  ncclResult_t Progress() { return recvProxyProgress(&proxyState_, &args_); }
  // Progresses until a call makes no progress.
  void Run() {
    for (int i = 0; i < 64; i++) {
      ASSERT_EQ(Progress(), ncclSuccess);
      if (args_.idle) return;
    }
    FAIL() << "never went idle";
  }
  reqSlot& Fifo(int s, uint64_t step) { return res_[s].reqFifo[s / COLLNET_GROUP_NSUBS][Slot(s, step)]; }
  // The send proxy finished the collective in a slot the last sub of s's group handed it.
  void SendDone(int s, uint64_t step, int recvBytes = kChunk) {
    Fifo(s, step).size = recvBytes;
    Fifo(s, step).turnIsSendNotRecv = false;
  }
  void UseGdrFlush() {
    for (auto& r : res_) {
      r.useGdr = ncclTopoGdrModeDefault;
      r.needFlush = ncclTopoFlushAlways;
    }
  }
  // The GPU consumed the data up to this step.
  void GpuConsumed(int n, uint64_t steps) {
    for (int s = 0; s < n; s++) sendMems_[s].head = args_.subs[s].base + steps;
  }
};

TEST_F(CollNetRecvProgressMicrotest, Ready_StartsAtNextChunkBoundaryAndClearsRequests) {
  Subs(2, 3);
  res_[1].step = 21;
  args_.chunkSteps = 4;
  args_.subs[0].flushed = args_.subs[0].transmitted = args_.subs[0].done = 5;
  args_.subs[1].requests[3] = reinterpret_cast<void*>(0x9);

  ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(args_.state, ncclProxyOpProgress);
  EXPECT_EQ(args_.subs[0].base, kBase);
  EXPECT_EQ(res_[0].step, kBase + 3);
  EXPECT_EQ(args_.subs[1].base, 24u);
  EXPECT_EQ(res_[1].step, 27u);
  EXPECT_EQ(args_.subs[0].flushed, 0u);
  EXPECT_EQ(args_.subs[0].transmitted, 0u);
  EXPECT_EQ(args_.subs[0].done, 0u);
  EXPECT_EQ(args_.subs[1].requests[3], nullptr);
  EXPECT_EQ(args_.subs[1].posted, 1u);
}

TEST_F(CollNetRecvProgressMicrotest, Ready_RegisteredOneRankPerNode_OneStepPerMaxCollBytes) {
  Subs(3, 1);
  for (int s = 0; s < 3; s++) args_.subs[s].nbytes = 2049;
  args_.subs[0].reg = args_.subs[0].isOneRPN = 1;
  args_.subs[1].reg = 1;       // several ranks per node
  args_.subs[2].isOneRPN = 1;  // not registered

  ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(args_.subs[0].nsteps, 3);
  EXPECT_EQ(args_.subs[1].nsteps, 1);
  EXPECT_EQ(args_.subs[2].nsteps, 1);
}

TEST_F(CollNetRecvProgressMicrotest, Post_LastSubOfGroupHandsSlotsToSendProxyWindowAhead) {
  Subs(2, 12);

  for (int i = 0; i < 12; i++) ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(args_.subs[0].posted, 0u);
  EXPECT_EQ(args_.subs[1].posted, (uint64_t)NCCL_STEPS);
  for (int step = 0; step < NCCL_STEPS; step++) EXPECT_TRUE(Fifo(1, step).turnIsSendNotRecv);
  EXPECT_EQ(args_.idle, 1);  // a full window, nothing else to do
}

TEST_F(CollNetRecvProgressMicrotest, Post_ShortOp_PostsOnlyItsSteps) {
  Subs(1, 2);

  Run();

  EXPECT_EQ(args_.subs[0].posted, 2u);
  EXPECT_FALSE(Fifo(0, 2).turnIsSendNotRecv);
}

TEST_F(CollNetRecvProgressMicrotest, Receive_SendProxyStillOwnsSlot_Waits) {
  Subs(1, 2);
  Run();

  EXPECT_EQ(args_.subs[0].received, 0u);
  EXPECT_EQ(args_.subs[0].transmitted, 0u);
}

TEST_F(CollNetRecvProgressMicrotest, Transmit_EarlierSubCatchingUp_CountsAsProgress) {
  Subs(2, 1);
  ASSERT_EQ(Progress(), ncclSuccess);  // posts
  SendDone(1, 0);
  ASSERT_EQ(Progress(), ncclSuccess);  // receives
  ASSERT_EQ(Progress(), ncclSuccess);  // flushes the group, publishes the last sub
  ASSERT_EQ(args_.subs[0].transmitted, 0u);

  ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(args_.subs[0].transmitted, 1u);
  EXPECT_EQ(args_.idle, 0);
}

TEST_F(CollNetRecvProgressMicrotest, Receive_NoFlushNeeded_PublishesSlotAndTailToGpu) {
  Subs(2, 2);
  args_.subs[0].isOneRPN = args_.subs[1].isOneRPN = 1;  // unregistered: still one GPU step per step
  ASSERT_EQ(Progress(), ncclSuccess);
  SendDone(1, 0);

  Run();

  EXPECT_TRUE(net_.iflushes.empty());
  for (int s = 0; s < 2; s++) {
    EXPECT_EQ(args_.subs[s].flushed, 1u);
    EXPECT_EQ(args_.subs[s].transmitted, 1u);
    EXPECT_EQ(RecvMem(s)->connFifo[Slot(s, 0)].offset, (ssize_t)(NCCL_STEPS * 2 * kSlot + s * kChunk));
    EXPECT_EQ(RecvMem(s)->tail, kBase + 1);
  }
  EXPECT_EQ(args_.subs[1].received, 1u);

  SendDone(1, 1);
  GpuConsumed(2, 1);
  Run();
  EXPECT_EQ(RecvMem(1)->tail, kBase + 2);
  EXPECT_EQ(args_.subs[1].done, 1u);  // the GPU has not consumed step 1
}

TEST_F(CollNetRecvProgressMicrotest, Receive_TwoGroups_EachFlushCompletesOnlyItsGroup) {
  UseGdrFlush();
  Subs(9, 1);
  ASSERT_EQ(Progress(), ncclSuccess);
  SendDone(7, 0);
  SendDone(8, 0);
  net_.pending.insert(&net_.requestTokens[0]);  // group 0's flush

  Run();

  ASSERT_EQ(net_.iflushes.size(), 2u);
  EXPECT_EQ(net_.iflushes[0].size, 8 * kChunk);
  EXPECT_EQ(net_.iflushes[1].data, region_.data() + NCCL_STEPS * 2 * kSlot + 8 * kChunk);
  EXPECT_EQ(net_.iflushes[1].size, kChunk);
  for (int s = 0; s < 8; s++) EXPECT_EQ(args_.subs[s].flushed, 0u);
  EXPECT_EQ(args_.subs[8].flushed, 1u);
}

TEST_F(CollNetRecvProgressMicrotest, Receive_GdrNeedsFlush_FlushesGroupRangeBeforeGpuSeesIt) {
  UseGdrFlush();
  Subs(2, 2);
  ASSERT_EQ(Progress(), ncclSuccess);
  SendDone(1, 0);
  net_.testDone = 0;

  Run();

  ASSERT_EQ(net_.iflushes.size(), 1u);
  EXPECT_EQ(net_.iflushes[0].data, region_.data() + NCCL_STEPS * 2 * kSlot);
  EXPECT_EQ(net_.iflushes[0].size, 2 * kChunk);
  EXPECT_EQ(net_.iflushes[0].mhandle, poolMh_);
  EXPECT_EQ(args_.subs[1].flushed, 0u);
  EXPECT_EQ(RecvMem(1)->tail, 0u);

  net_.testDone = 1;
  ASSERT_EQ(Progress(), ncclSuccess);
  EXPECT_EQ(net_.tested.back(), &net_.requestTokens[0]);
  EXPECT_EQ(args_.subs[0].flushed, 1u);
  EXPECT_EQ(args_.subs[1].flushed, 1u);
  EXPECT_EQ(args_.subs[1].requests[Slot(1, 0)], nullptr);
  EXPECT_EQ(args_.idle, 0);
}

TEST_F(CollNetRecvProgressMicrotest, Receive_NothingReceivedOrNoFlushRule_SkipsFlush) {
  UseGdrFlush();
  Subs(1, 4);
  ASSERT_EQ(Progress(), ncclSuccess);
  SendDone(0, 0, /*recvBytes=*/0);
  Run();

  res_[0].needFlush = ncclTopoFlushNone;
  SendDone(0, 1);
  Run();

  res_[0].needFlush = ncclTopoFlushAlways;
  res_[0].useGdr = ncclTopoGdrModeDisable;
  SendDone(0, 2);
  Run();

  EXPECT_TRUE(net_.iflushes.empty());
  EXPECT_EQ(args_.subs[0].transmitted, 3u);
}

TEST_F(CollNetRecvProgressMicrotest, Receive_GdcFlushWord_ReadInsteadOfPluginFlush) {
  UseGdrFlush();
  uint64_t flushWord = 0;
  res_[0].gdcFlush = &flushWord;
  Subs(1, 2);
  ASSERT_EQ(Progress(), ncclSuccess);
  SendDone(0, 0);

  Run();

  EXPECT_TRUE(net_.iflushes.empty());
  EXPECT_EQ(args_.subs[0].transmitted, 1u);
}

TEST_F(CollNetRecvProgressMicrotest, Receive_RegisteredOneRankPerNode_FlushesUserBuffer) {
  UseGdrFlush();
  alignas(16) static uint8_t user[4096];
  Subs(1, 1);
  args_.subs[0].reg = 1;
  args_.subs[0].isOneRPN = 1;
  args_.subs[0].nbytes = 512;
  args_.subs[0].recvbuff = user;
  ASSERT_EQ(Progress(), ncclSuccess);
  SendDone(0, 0, /*recvBytes=*/0);  // a registered step flushes whatever the slot size says

  ASSERT_EQ(Progress(), ncclSuccess);

  ASSERT_EQ(net_.iflushes.size(), 1u);
  EXPECT_EQ(net_.iflushes[0].data, user);
  EXPECT_EQ(net_.iflushes[0].size, 512);
}

TEST_F(CollNetRecvProgressMicrotest, Transmit_GdcSync_TailGoesToGdcWord) {
  uint64_t gdc = 0;
  res_[0].gdcSync = &gdc;
  Subs(1, 2);
  ASSERT_EQ(Progress(), ncclSuccess);
  SendDone(0, 0);

  Run();

  EXPECT_EQ(gdc, kBase + 1);
  EXPECT_EQ(RecvMem(0)->tail, 0u);
}

TEST_F(CollNetRecvProgressMicrotest, Transmit_RegisteredOneRankPerNode_SignalsGpuOnceAtTheEnd) {
  Subs(1, 1);
  args_.subs[0].reg = 1;
  args_.subs[0].isOneRPN = 1;
  args_.subs[0].nbytes = 2 * 1024;
  for (auto& f : RecvMem(0)->connFifo) f.offset = -7;
  ASSERT_EQ(Progress(), ncclSuccess);
  ASSERT_EQ(Progress(), ncclSuccess);
  SendDone(0, 0);
  Run();
  EXPECT_EQ(args_.subs[0].transmitted, 1u);
  EXPECT_EQ(RecvMem(0)->tail, 0u);

  SendDone(0, 1);
  Run();

  EXPECT_EQ(RecvMem(0)->tail, kBase + 1);
  for (auto& f : RecvMem(0)->connFifo) EXPECT_EQ(f.offset, -7);

  args_.state = ncclProxyOpReady;  // AllGather lands in the user buffer too
  args_.coll = ncclFuncAllGather;
  args_.subs[0].nsteps = 1;
  args_.subs[0].nbytes = 512;
  ASSERT_EQ(Progress(), ncclSuccess);
  SendDone(0, 0);
  Run();
  ASSERT_EQ(args_.subs[0].transmitted, 1u);
  for (auto& f : RecvMem(0)->connFifo) EXPECT_EQ(f.offset, -7);
}

TEST_F(CollNetRecvProgressMicrotest, Transmit_RegisteredSeveralRanksPerNode_StepByStepOffsetsOnlyForGather) {
  Subs(1, 2);
  args_.subs[0].reg = 1;
  for (auto& f : RecvMem(0)->connFifo) f.offset = -7;
  ASSERT_EQ(Progress(), ncclSuccess);
  SendDone(0, 0);
  Run();
  EXPECT_EQ(RecvMem(0)->connFifo[Slot(0, 0)].offset, -7);
  EXPECT_EQ(RecvMem(0)->tail, kBase + 1);  // per step, not once at the end
  SendDone(0, 1);
  GpuConsumed(1, 1);
  Run();
  EXPECT_EQ(args_.subs[0].done, 1u);  // the GPU consumed only step 0

  args_.state = ncclProxyOpReady;
  args_.coll = ncclFuncAllGather;
  ASSERT_EQ(Progress(), ncclSuccess);
  SendDone(0, 0);
  Run();
  EXPECT_EQ(RecvMem(0)->connFifo[Slot(0, 0)].offset, (ssize_t)(NCCL_STEPS * 2 * kSlot));
}

TEST_F(CollNetRecvProgressMicrotest, Done_GpuConsumed_RetiresStepsInSubOrder) {
  Subs(2, 2);
  ASSERT_EQ(Progress(), ncclSuccess);
  SendDone(1, 0);
  Run();
  EXPECT_EQ(args_.subs[0].done, 0u);  // the GPU still holds the slot

  GpuConsumed(2, 0);  // head at base: nothing consumed yet
  Run();
  EXPECT_EQ(args_.subs[0].done, 0u);

  sendMems_[1].head = args_.subs[1].base + 1;  // only the later sub's GPU side consumed it
  ASSERT_EQ(Progress(), ncclSuccess);
  EXPECT_EQ(args_.subs[1].done, 0u);

  GpuConsumed(2, 1);
  ASSERT_EQ(Progress(), ncclSuccess);
  EXPECT_EQ(args_.subs[0].done, 1u);
  EXPECT_EQ(args_.subs[1].done, 1u);
  EXPECT_EQ(args_.idle, 0);
  EXPECT_TRUE(Fifo(1, 1).turnIsSendNotRecv);  // done frees the window: the next slot goes out

  GpuConsumed(2, 5);  // the GPU can't consume what was never handed over
  Run();
  EXPECT_EQ(args_.subs[1].done, 1u);
}

TEST_F(CollNetRecvProgressMicrotest, Done_EarlierSubFinishesFirst_KeepsOpRunning) {
  Subs(2, 1);
  ASSERT_EQ(Progress(), ncclSuccess);
  SendDone(1, 0);
  Run();
  sendMems_[0].head = args_.subs[0].base + 1;  // only sub 0's GPU side has consumed its step

  Run();

  EXPECT_EQ(args_.subs[0].done, 1u);
  EXPECT_EQ(args_.subs[1].done, 0u);
  EXPECT_EQ(args_.state, ncclProxyOpProgress);
}

TEST_F(CollNetRecvProgressMicrotest, Done_RegisteredOneRankPerNode_GpuSignalsOnceForAllSteps) {
  Subs(1, 1);
  args_.subs[0].reg = 1;
  args_.subs[0].isOneRPN = 1;
  args_.subs[0].nbytes = 2 * 1024;
  ASSERT_EQ(Progress(), ncclSuccess);
  ASSERT_EQ(Progress(), ncclSuccess);
  for (uint64_t step = 0; step < 2; step++) {
    SendDone(0, step);
    Run();
  }
  GpuConsumed(1, 1);

  Run();

  EXPECT_EQ(args_.subs[0].done, 2u);
  EXPECT_EQ(args_.state, ncclProxyOpNone);
}

TEST_F(CollNetRecvProgressMicrotest, WholeOp_LongerThanWindow_RunsEveryStepThenStops) {
  constexpr int kSteps = NCCL_STEPS + 3;
  UseGdrFlush();
  Subs(2, kSteps);
  args_.subs[2].nsteps = 99;  // a stale sub past nsubs, left mid-op by an earlier one
  args_.subs[2].flushed = 1;
  ASSERT_EQ(Progress(), ncclSuccess);
  for (uint64_t step = 0; step < kSteps; step++) {
    ASSERT_TRUE(Fifo(1, step).turnIsSendNotRecv);
    SendDone(1, step);
    Run();
    ASSERT_EQ(args_.subs[1].transmitted, step + 1);
    EXPECT_EQ(RecvMem(0)->connFifo[Slot(0, step)].offset, (ssize_t)((NCCL_STEPS + step % NCCL_STEPS) * 2 * kSlot));
    GpuConsumed(2, step + 1);
    ASSERT_EQ(Progress(), ncclSuccess);
    ASSERT_EQ(args_.subs[1].done, step + 1);
  }

  EXPECT_EQ(args_.state, ncclProxyOpNone);
  ASSERT_EQ(net_.iflushes.size(), (size_t)kSteps);
  EXPECT_EQ(net_.iflushes[NCCL_STEPS + 1].data, region_.data() + (NCCL_STEPS + 1) * 2 * kSlot);
  EXPECT_EQ(net_.tested.size(), (size_t)kSteps);  // every flush waited for
  EXPECT_EQ(res_[2].step, kBase);
  EXPECT_EQ(RecvMem(2)->tail, 0u);
}

TEST_F(CollNetRecvProgressMicrotest, PluginFails_ReturnsError) {
  UseGdrFlush();
  Subs(1, 1);
  ASSERT_EQ(Progress(), ncclSuccess);
  SendDone(0, 0);
  net_.iflushResult = ncclSystemError;
  EXPECT_EQ(Progress(), ncclSystemError);

  args_.state = ncclProxyOpReady;
  net_.iflushResult = ncclSuccess;
  ASSERT_EQ(Progress(), ncclSuccess);
  SendDone(0, 0);
  ASSERT_EQ(Progress(), ncclSuccess);
  net_.testResult = ncclSystemError;
  EXPECT_EQ(Progress(), ncclSystemError);
}

TEST_F(CollNetRecvProgressMicrotest, NotStarted_DoesNothing) {
  Subs(1, 2);
  args_.state = ncclProxyOpNone;

  ASSERT_EQ(Progress(), ncclSuccess);

  EXPECT_EQ(args_.subs[0].posted, 0u);
  EXPECT_EQ(args_.idle, 1);
}

// ---------------------------------------------------------------------------
// Comm-level setup (ncclCollNetChainBufferSetup, ncclCollNetDirectBufferSetup,
// isCollNetChainHeadsSubset, ncclCollNetSetup)
// ---------------------------------------------------------------------------

// Rank 0 of 4: two nodes of two local ranks, two channels.
class CollNetCommMicrotest : public CollNetMicrotestBase {
 protected:
  static constexpr int kRanks = 4;
  static constexpr int kChannels = 2;
  ncclComm comm_{};
  int rankToNode_[kRanks] = {0, 0, 1, 1};
  int rankToLocalRank_[kRanks] = {0, 1, 0, 1};
  int topParentRanks_[kRanks] = {10, 11, 12, 13};
  ncclTopoGraph direct_{}, chain_{}, nvls_{};
  ncclTopoGraph* graphs_[NCCL_NUM_ALGORITHMS] = {};
  char bootstrap_ = 0;

  struct Connect {
    int channelId, nrecv, nsend, connIndex;
    std::vector<int> recv, send;
  };
  std::vector<Connect> connects_;
  std::vector<std::pair<ncclTopoGraph*, int>> setups_;  // (graph, connIndex)

  void SetUp() override {
    CollNetMicrotestBase::SetUp();
    comm_.rank = 0;
    comm_.nRanks = kRanks;
    comm_.node = 0;
    comm_.nNodes = 2;
    comm_.localRanks = 2;
    comm_.rankToNode = rankToNode_;
    comm_.rankToLocalRank = rankToLocalRank_;
    comm_.topParentRanks = topParentRanks_;
    comm_.nChannels = kChannels;
    comm_.config.collnetEnable = 1;
    comm_.collNetChainSupport = 1;
    comm_.buffSizes[NCCL_PROTO_SIMPLE] = 1 << 16;
    comm_.bootstrap = &bootstrap_;
    comm_.ncclCollNet = &vtable_;
    // Channel 0's head is rank 0, channel 1's is rank 1.
    direct_.nChannels = kChannels;
    direct_.intra[0] = 0;
    direct_.intra[1] = 1;
    direct_.intra[2] = 1;
    direct_.intra[3] = 0;
    chain_.nChannels = 1;
    chain_.intra[0] = 1;
    graphs_[NCCL_ALGO_COLLNET_DIRECT] = &direct_;
    graphs_[NCCL_ALGO_COLLNET_CHAIN] = &chain_;
    graphs_[NCCL_ALGO_NVLS] = &nvls_;
    g_ncclTransportP2pConnect = [this](ncclComm*, int c, int nrecv, int* recv, int nsend, int* send, int connIndex) {
      connects_.push_back({c, nrecv, nsend, connIndex, std::vector<int>(recv, recv + nrecv),
                           std::vector<int>(send, send + nsend)});
      return ncclSuccess;
    };
    g_ncclTransportP2pSetup = [this](ncclComm* comm, ncclTopoGraph* graph, int connIndex, bool*) {
      EXPECT_EQ(comm, &comm_);
      setups_.push_back({graph, connIndex});
      return ncclSuccess;
    };
  }

  void TearDown() override {
    free(comm_.collNetHeads);
    free(comm_.collNetSharedRes);
    CollNetMicrotestBase::TearDown();
  }
};

TEST_F(CollNetCommMicrotest, ChainBufferSetup_ConnectsEachChannelsChainBothWays) {
  for (int c = 0; c < kChannels; c++) {
    comm_.channels[c].collnetChain.up = 10 + c;
    comm_.channels[c].collnetChain.down[0] = 20 + c;
  }

  ASSERT_EQ(ncclCollNetChainBufferSetup(&comm_), ncclSuccess);

  ASSERT_EQ(connects_.size(), 2u * kChannels);
  for (int c = 0; c < kChannels; c++) {
    SCOPED_TRACE(c);
    const Connect& up = connects_[c];  // recv from up, send down
    EXPECT_EQ(up.channelId, c);
    EXPECT_EQ(up.recv, std::vector<int>{10 + c});
    EXPECT_EQ(up.send, std::vector<int>{20 + c});
    EXPECT_EQ(up.connIndex, 0);
    const Connect& down = connects_[kChannels + c];
    EXPECT_EQ(down.recv, std::vector<int>{20 + c});
    EXPECT_EQ(down.send, std::vector<int>{10 + c});
    EXPECT_EQ(down.connIndex, 1);
  }
  ncclTopoGraph* graph = &comm_.graphs[NCCL_ALGO_COLLNET_CHAIN];
  std::vector<std::pair<ncclTopoGraph*, int>> want = {{graph, 0}, {graph, 1}};
  EXPECT_EQ(setups_, want);
}

TEST_F(CollNetCommMicrotest, DirectBufferSetup_ConnectsEachChannelsHeadsBothWays) {
  for (int c = 0; c < kChannels; c++) {
    for (int i = 0; i < NCCL_MAX_DIRECT_ARITY; i++) {
      comm_.channels[c].collnetDirect.up[i] = 100 * c + i;
      comm_.channels[c].collnetDirect.down[i] = 100 * c + 50 + i;
    }
  }

  ASSERT_EQ(ncclCollNetDirectBufferSetup(&comm_), ncclSuccess);

  ASSERT_EQ(connects_.size(), 2u * kChannels);
  for (int c = 0; c < kChannels; c++) {
    SCOPED_TRACE(c);
    const Connect& in = connects_[c];
    EXPECT_EQ(in.channelId, c);
    ASSERT_EQ(in.nrecv, NCCL_MAX_DIRECT_ARITY);
    ASSERT_EQ(in.nsend, NCCL_MAX_DIRECT_ARITY);
    EXPECT_EQ(in.recv.back(), 100 * c + NCCL_MAX_DIRECT_ARITY - 1);
    EXPECT_EQ(in.send.front(), 100 * c + 50);
    EXPECT_EQ(in.connIndex, 0);
    const Connect& out = connects_[kChannels + c];
    EXPECT_EQ(out.recv.front(), 100 * c + 50);
    EXPECT_EQ(out.send.back(), 100 * c + NCCL_MAX_DIRECT_ARITY - 1);
    EXPECT_EQ(out.connIndex, 1);
  }
  ncclTopoGraph* graph = &comm_.graphs[NCCL_ALGO_COLLNET_DIRECT];
  std::vector<std::pair<ncclTopoGraph*, int>> want = {{graph, 0}, {graph, 1}};
  EXPECT_EQ(setups_, want);
}

TEST_F(CollNetCommMicrotest, BufferSetups_CollNetOff_ConnectNothing) {
  comm_.config.collnetEnable = 0;
  EXPECT_EQ(ncclCollNetChainBufferSetup(&comm_), ncclSuccess);
  EXPECT_EQ(ncclCollNetDirectBufferSetup(&comm_), ncclSuccess);

  comm_.config.collnetEnable = 1;
  comm_.collNetChainSupport = 0;
  EXPECT_EQ(ncclCollNetChainBufferSetup(&comm_), ncclSuccess);

  EXPECT_TRUE(connects_.empty());
  EXPECT_TRUE(setups_.empty());
}

TEST_F(CollNetCommMicrotest, BufferSetups_TransportFails_ReturnsError) {
  for (int failing = 0; failing < 4; failing++) {
    SCOPED_TRACE(failing);
    int connects = 0, setups = 0;
    g_ncclTransportP2pConnect = [&](ncclComm*, int, int, int*, int, int*, int) {
      return ++connects == (failing == 0 ? 1 : failing == 1 ? kChannels + 1 : 0) ? ncclSystemError : ncclSuccess;
    };
    g_ncclTransportP2pSetup = [&](ncclComm*, ncclTopoGraph*, int, bool*) {
      return ++setups == (failing == 2 ? 1 : failing == 3 ? 2 : 0) ? ncclSystemError : ncclSuccess;
    };
    EXPECT_EQ(ncclCollNetChainBufferSetup(&comm_), ncclSystemError);
    connects = setups = 0;
    EXPECT_EQ(ncclCollNetDirectBufferSetup(&comm_), ncclSystemError);
  }
}

TEST_F(CollNetCommMicrotest, ChainHeadsSubset_OnlyWhenEveryChainHeadIsACollNetHead) {
  int heads[] = {0, 1};
  comm_.collNetHeads = heads;
  comm_.collNetHeadsNum = 1;
  chain_.intra[0] = 0;
  EXPECT_TRUE(isCollNetChainHeadsSubset(&comm_, &chain_));
  chain_.intra[0] = 1;
  EXPECT_FALSE(isCollNetChainHeadsSubset(&comm_, &chain_));
  comm_.collNetHeadsNum = 2;
  chain_.nChannels = 2;
  chain_.intra[2] = 0;  // channel 1 starts localRanks entries later
  EXPECT_TRUE(isCollNetChainHeadsSubset(&comm_, &chain_));
  comm_.collNetHeadsNum = 1;
  EXPECT_FALSE(isCollNetChainHeadsSubset(&comm_, &chain_));

  heads[0] = 1;  // channel 1's chain head, rank 0, is not a CollNet head
  EXPECT_FALSE(isCollNetChainHeadsSubset(&comm_, &chain_));
  chain_.nChannels = 1;  // and past the last channel nothing counts
  EXPECT_TRUE(isCollNetChainHeadsSubset(&comm_, &chain_));
  comm_.collNetHeads = nullptr;
}

class CollNetSetupCommMicrotest : public CollNetCommMicrotest {
 protected:
  struct SetupCall {
    int channel, head, h, type;
    ncclConnect* connect;
  };
  std::vector<SetupCall> collNetSetups_;
  std::set<ncclTopoGraph*> setupGraphs_;
  std::vector<int> checks_, initChannels_;
  int frees_ = 0;
  std::vector<int> rankMap_;
  bool failRecvOnChannel1_ = false;
  // What the other ranks report for each (op, type); rank 2 is the other node's head.
  uint8_t remote_[kRanks][4][ncclNumTypes] = {};

  // Drops what the last ncclCollNetSetup allocated, except a shared state it does not own.
  void ResetComm(ncclCollNetSharedRes* notOwned = nullptr) {
    free(comm_.collNetHeads);
    if (comm_.collNetSharedRes != notOwned) free(comm_.collNetSharedRes);
    comm_.collNetHeads = nullptr;
    comm_.collNetSharedRes = nullptr;
    comm_.collNetHeadsNum = 0;
    comm_.config.collnetEnable = 1;
  }

  void SetUp() override {
    CollNetCommMicrotest::SetUp();
    g_ncclTransportCollNetSetup = [this](ncclComm* comm, ncclTopoGraph* graph, ncclChannel* channel, int master,
                                         int peer, int id, int type, ncclConnect* connect) {
      setupGraphs_.insert(graph);
      EXPECT_EQ(master, peer);
      int c = channel - comm->channels;
      collNetSetups_.push_back({c, master, id, type, connect});
      return failRecvOnChannel1_ && c == 1 && type == collNetRecv;
    };
    g_ncclTransportCollNetCheck = [this](ncclComm*, int fail) {
      checks_.push_back(fail);
      return ncclSuccess;
    };
    g_ncclTransportCollNetFree = [this](ncclComm*) {
      ++frees_;
      return ncclSuccess;
    };
    g_ncclTransportInitRankMap = [this](ncclComm*, int nHeads, const int* heads) {
      rankMap_.assign(heads, heads + nHeads);
      return ncclSuccess;
    };
    g_initCollnetChannel = [this](ncclComm* comm, int c, ncclComm* parent, bool share) {
      EXPECT_EQ(comm, &comm_);
      EXPECT_EQ(parent, nullptr);
      EXPECT_FALSE(share);
      initChannels_.push_back(c);
      return ncclSuccess;
    };
    remote_[2][ncclSum][ncclFloat32] = 1 << 1;
    remote_[2][ncclMax][ncclInt32] = 1 << 0;
    remote_[3][ncclMin][ncclInt8] = 1 << 1;
    g_devrBootstrapAllGather = [this](void* state, void* all, int size) {
      EXPECT_EQ(state, &bootstrap_);
      EXPECT_EQ(size, (int)sizeof(remote_[0]));
      for (int r = 1; r < kRanks; r++) memcpy(static_cast<char*>(all) + r * size, remote_[r], size);
      return ncclSuccess;
    };
    net_.supported = {{ncclFloat32, ncclSum}, {ncclInt32, ncclMax}};
  }

  ncclResult_t Setup(ncclComm* parent = nullptr) { return ncclCollNetSetup(&comm_, parent, graphs_); }
};

TEST_F(CollNetSetupCommMicrotest, Setup_ConnectsEveryHeadRecvThenSendOnEveryChannel) {
  ASSERT_EQ(Setup(), ncclSuccess);

  EXPECT_EQ(comm_.collNetHeadsNum, 2);
  ASSERT_NE(comm_.collNetHeads, nullptr);
  EXPECT_EQ(comm_.collNetHeads[0], 0);
  EXPECT_EQ(comm_.collNetHeads[1], 1);
  EXPECT_EQ(rankMap_, (std::vector<int>{0, 1}));
  ASSERT_NE(comm_.collNetSharedRes, nullptr);
  EXPECT_EQ(comm_.collNetSharedRes->nChannels, kChannels);
  EXPECT_EQ(comm_.collNetSharedRes->buffSize, (size_t)1 << 16);
  EXPECT_EQ(initChannels_, (std::vector<int>{0, 1}));
  ASSERT_EQ(collNetSetups_.size(), 8u);
  int i = 0;
  for (int c = 0; c < kChannels; c++) {
    for (int h = 0; h < 2; h++) {
      for (int type : {collNetRecv, collNetSend}) {
        SCOPED_TRACE(i);
        const SetupCall& s = collNetSetups_[i++];
        EXPECT_EQ(s.channel, c);
        EXPECT_EQ(s.head, h);
        EXPECT_EQ(s.h, h);
        EXPECT_EQ(s.type, type);
      }
    }
  }
  EXPECT_EQ(setupGraphs_, std::set<ncclTopoGraph*>{&direct_});
  EXPECT_EQ(collNetSetups_[0].connect, collNetSetups_[1].connect);  // send reuses recv's connect info
  EXPECT_EQ(checks_, (std::vector<int>{0, 0}));  // after channel 0, then after all
  EXPECT_EQ(frees_, 0);
  EXPECT_EQ(comm_.config.collnetEnable, 1);
  EXPECT_EQ(comm_.collNetChainSupport, 1);
}

TEST_F(CollNetSetupCommMicrotest, Setup_HeadFails_SkipsLaterSendsAndReportsFailure) {
  failRecvOnChannel1_ = true;

  ASSERT_EQ(Setup(), ncclSuccess);

  ASSERT_EQ(collNetSetups_.size(), 6u);  // channel 1: both recvs, no send
  EXPECT_EQ(collNetSetups_[4].type, collNetRecv);
  EXPECT_EQ(collNetSetups_[5].type, collNetRecv);
  EXPECT_EQ(checks_, (std::vector<int>{0, 1}));
}

TEST_F(CollNetSetupCommMicrotest, Setup_HeadsShareLocalRank_ConnectsEachOnce) {
  direct_.intra[2] = 0;  // channel 1's head is rank 0 again

  ASSERT_EQ(Setup(), ncclSuccess);

  EXPECT_EQ(comm_.collNetHeadsNum, 1);
  EXPECT_EQ(collNetSetups_.size(), 4u);
  EXPECT_EQ(comm_.collNetChainSupport, 0);  // the chain's head, rank 1, gets no CollNet resources
}

TEST_F(CollNetSetupCommMicrotest, Setup_NvlsTransport_TakesHeadsFromNvls) {
  comm_.nvlsSupport = 1;
  comm_.channels[0].nvls.nHeads = 1;
  comm_.nvlsHeads[0] = 1;
  nvls_.nChannels = 1;

  ASSERT_EQ(Setup(), ncclSuccess);

  EXPECT_EQ(comm_.collNetHeadsNum, 1);
  EXPECT_EQ(comm_.collNetHeads[0], 1);
  EXPECT_NE(comm_.collNetHeads, comm_.nvlsHeads);
  ASSERT_EQ(collNetSetups_.size(), 4u);
  EXPECT_EQ(collNetSetups_[0].head, 1);
  EXPECT_EQ(setupGraphs_, std::set<ncclTopoGraph*>{&nvls_});
}

TEST_F(CollNetSetupCommMicrotest, Setup_NvlsTransportDisabled_UsesCollNetGraph) {
  comm_.nvlsSupport = 1;
  comm_.config.nvlsHostMode = ncclNvlsHostModeDisableTransport;

  ASSERT_EQ(Setup(), ncclSuccess);

  EXPECT_EQ(comm_.collNetHeadsNum, 2);
  EXPECT_EQ(setupGraphs_, std::set<ncclTopoGraph*>{&direct_});
}

TEST_F(CollNetSetupCommMicrotest, Setup_SupportMatrix_SupportedOnlyWhereSomeHeadSaysYesAndNoneSaysNo) {
  ASSERT_EQ(Setup(), ncclSuccess);

  EXPECT_EQ(net_.asked.size(), 4u * ncclNumTypes);  // every type with sum, prod, max and min
  EXPECT_LT(net_.asked.rbegin()->first, (int)ncclNumTypes);
  for (const auto& [ty, op] : net_.asked) EXPECT_LT(op, 4);

  EXPECT_EQ(comm_.collNetSupportMatrix[ncclSum][ncclFloat32], 1);  // both heads support it
  EXPECT_EQ(comm_.collNetSupportMatrix[ncclMax][ncclInt32], 0);    // rank 2 does not
  EXPECT_EQ(comm_.collNetSupportMatrix[ncclMin][ncclInt8], 0);     // we do not
  EXPECT_EQ(comm_.collNetSupportMatrix[ncclProd][ncclFloat64], 0);
}

TEST_F(CollNetSetupCommMicrotest, Setup_AnotherRankHeadsEveryChannel_OnlyItIsSetUp) {
  direct_.intra[0] = 1;
  direct_.intra[2] = 1;
  net_.reduceSupportResult = ncclSystemError;  // not a head: never asked

  ASSERT_EQ(Setup(), ncclSuccess);

  EXPECT_EQ(comm_.collNetHeadsNum, 1);
  EXPECT_EQ(comm_.collNetHeads[0], 1);
  EXPECT_EQ(collNetSetups_.size(), 4u);
  EXPECT_TRUE(net_.asked.empty());
}

TEST_F(CollNetSetupCommMicrotest, Setup_NotAHead_TakesMatrixFromOtherRanksClaims) {
  comm_.rank = 1;
  direct_.intra[2] = 0;  // the only head is rank 0
  remote_[0][ncclSum][ncclFloat32] = 1 << 1;
  g_devrBootstrapAllGather = [this](void*, void* all, int size) {
    for (int r = 0; r < kRanks; r++) {
      if (r == 1) {
        static const uint8_t none[4][ncclNumTypes] = {};
        EXPECT_EQ(memcmp(static_cast<char*>(all) + r * size, none, size), 0);  // no claims either way
      } else {
        memcpy(static_cast<char*>(all) + r * size, remote_[r], size);
      }
    }
    return ncclSuccess;
  };
  net_.reduceSupportResult = ncclSystemError;  // never asked

  ASSERT_EQ(Setup(), ncclSuccess);

  EXPECT_EQ(comm_.collNetSupportMatrix[ncclSum][ncclFloat32], 1);
  EXPECT_EQ(comm_.collNetSupportMatrix[ncclMin][ncclInt8], 1);
  EXPECT_EQ(comm_.collNetSupportMatrix[ncclMax][ncclInt32], 0);
}

TEST_F(CollNetSetupCommMicrotest, Setup_HeadOnAnotherNode_DisablesCollNetWithError) {
  direct_.intra[2] = 2;

  EXPECT_EQ(Setup(), ncclInternalError);

  EXPECT_EQ(frees_, 1);
  EXPECT_EQ(comm_.config.collnetEnable, 0);
  EXPECT_TRUE(collNetSetups_.empty());
}

TEST_F(CollNetSetupCommMicrotest, Setup_DependencyFails_DisablesCollNetWithError) {
  for (int failing = 0; failing < 5; failing++) {
    SCOPED_TRACE(failing);
    ResetComm();
    frees_ = 0;
    ScopedHook initChannel(g_initCollnetChannel, [&](ncclComm*, int, ncclComm*, bool) {
      return failing == 0 ? ncclSystemError : ncclSuccess;
    });
    ScopedHook check(g_ncclTransportCollNetCheck,
                     [&](ncclComm*, int) { return failing == 1 ? ncclSystemError : ncclSuccess; });
    ScopedHook rankMap(g_ncclTransportInitRankMap,
                       [&](ncclComm*, int, const int*) { return failing == 2 ? ncclSystemError : ncclSuccess; });
    ScopedHook gather(g_devrBootstrapAllGather, [&](void*, void*, int) {
      return failing == 3 ? ncclSystemError : ncclSuccess;
    });
    net_.reduceSupportResult = failing == 4 ? ncclSystemError : ncclSuccess;

    EXPECT_EQ(Setup(), ncclSystemError);
    EXPECT_EQ(frees_, 1);
    EXPECT_EQ(comm_.config.collnetEnable, 0);
  }
}

class CollNetShareCommMicrotest : public CollNetSetupCommMicrotest {
 protected:
  ncclComm parent_{};
  ncclCollNetSharedRes parentShared_{};
  int parentHeads_[2] = {0, 1};
  int parentTopRanks_[kRanks] = {10, 11, 12, 13};
  std::vector<std::pair<int, bool>> sharedChannels_;
  int otherHeadPosition_ = 0;  // what rank 2, the other node's first head, reports

  void SetUp() override {
    CollNetSetupCommMicrotest::SetUp();
    parent_.config.collnetEnable = 1;
    parent_.nNodes = 2;
    parent_.nRanks = 8;
    parent_.shareResources = true;
    parent_.collNetHeads = parentHeads_;
    parent_.collNetHeadsNum = 2;
    parent_.topParentRanks = parentTopRanks_;
    parent_.collNetSharedRes = &parentShared_;
    parent_.collNetSupportMatrix[ncclProd][ncclBfloat16] = 1;
    direct_.intra[2] = 0;  // one head: rank 0
    g_initCollnetChannel = [this](ncclComm*, int c, ncclComm* parent, bool share) {
      EXPECT_EQ(parent, &parent_);
      sharedChannels_.push_back({c, share});
      return ncclSuccess;
    };
    // Each head reports where it sits among the parent's heads.
    g_devrBootstrapAllGather = [this](void*, void* all, int size) {
      struct Info {
        int headPosition, isMaster;
      };
      EXPECT_EQ(size, (int)sizeof(Info));
      Info* infos = static_cast<Info*>(all);
      const Info others[kRanks] = {{0, 1}, {7, 0}, {otherHeadPosition_, 1}, {7, 0}};  // non-heads are ignored
      for (int r = 0; r < kRanks; r++) {
        if (r != comm_.rank) infos[r] = others[r];
      }
      return ncclSuccess;
    };
  }

  void TearDown() override {
    comm_.collNetSharedRes = nullptr;  // the parent's
    CollNetSetupCommMicrotest::TearDown();
  }
};

TEST_F(CollNetShareCommMicrotest, HeadsMatchParents_SharesParentResourcesAndMatrix) {
  ASSERT_EQ(Setup(&parent_), ncclSuccess);

  EXPECT_EQ(comm_.collNetSharedRes, &parentShared_);
  EXPECT_EQ(sharedChannels_, (std::vector<std::pair<int, bool>>{{0, true}, {1, true}}));
  EXPECT_TRUE(collNetSetups_.empty());
  EXPECT_EQ(rankMap_, std::vector<int>{0});
  EXPECT_EQ(comm_.collNetSupportMatrix[ncclProd][ncclBfloat16], 1);
  EXPECT_EQ(checks_, std::vector<int>{0});
  EXPECT_EQ(comm_.config.collnetEnable, 1);
}

TEST_F(CollNetShareCommMicrotest, NonMasterRank_JoinsWithoutSharingResources) {
  comm_.rank = 1;

  ASSERT_EQ(Setup(&parent_), ncclSuccess);

  EXPECT_EQ(comm_.collNetSharedRes, nullptr);
  EXPECT_TRUE(sharedChannels_.empty());
  EXPECT_EQ(comm_.config.collnetEnable, 1);
}

TEST_F(CollNetShareCommMicrotest, HeadsSitElsewhereInParent_DisablesCollNet) {
  otherHeadPosition_ = 1;

  ASSERT_EQ(Setup(&parent_), ncclSuccess);

  EXPECT_EQ(frees_, 1);
  EXPECT_EQ(comm_.config.collnetEnable, 0);
  EXPECT_TRUE(sharedChannels_.empty());
}

TEST_F(CollNetShareCommMicrotest, HeadNotAParentHead_DisablesCollNet) {
  topParentRanks_[0] = 99;
  for (int other : {0, -1}) {  // whether or not the other node's head is a parent head
    SCOPED_TRACE(other);
    otherHeadPosition_ = other;
    ResetComm(&parentShared_);

    ASSERT_EQ(Setup(&parent_), ncclSuccess);

    EXPECT_EQ(comm_.config.collnetEnable, 0);
    EXPECT_TRUE(sharedChannels_.empty());
  }
}

TEST_F(CollNetShareCommMicrotest, ParentCannotShare_SetsUpOwnResources) {
  auto own = [this](ncclComm* parent, bool share) {
    EXPECT_EQ(parent, &parent_);
    EXPECT_FALSE(share);
  };
  g_initCollnetChannel = [&](ncclComm*, int, ncclComm* parent, bool share) {
    own(parent, share);
    return ncclSuccess;
  };
  g_devrBootstrapAllGather = [](void*, void*, int) { return ncclSuccess; };
  for (int variant = 0; variant < 3; variant++) {
    SCOPED_TRACE(variant);
    parent_.config.collnetEnable = variant != 0;
    parent_.nNodes = variant == 1 ? 3 : 2;
    parent_.shareResources = variant != 2;
    collNetSetups_.clear();
    ResetComm(&parentShared_);

    ASSERT_EQ(Setup(&parent_), ncclSuccess);

    EXPECT_NE(comm_.collNetSharedRes, &parentShared_);
    EXPECT_EQ(collNetSetups_.size(), 4u);
  }
  ResetComm(&parentShared_);
}

TEST_F(CollNetShareCommMicrotest, Share_DependencyFails_DisablesCollNetWithError) {
  g_devrBootstrapAllGather = [](void*, void*, int) { return ncclSystemError; };
  EXPECT_EQ(Setup(&parent_), ncclSystemError);
  EXPECT_EQ(comm_.config.collnetEnable, 0);

  ResetComm(&parentShared_);
  g_devrBootstrapAllGather = [](void*, void*, int) { return ncclSuccess; };
  g_initCollnetChannel = [](ncclComm*, int, ncclComm*, bool) { return ncclSystemError; };
  EXPECT_EQ(Setup(&parent_), ncclSystemError);
  EXPECT_EQ(comm_.config.collnetEnable, 0);
}

}  // namespace
