/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/net/net_v6.cc, the adapter that presents a v6 net/collnet plugin as the
// current ncclNet_t / ncclCollNet_t: one-time init behind a refcount, int-sized calls and defaulted properties.

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "nccl_net.h"
#include "os.h"

namespace {

void* const kLib = reinterpret_cast<void*>(0x6);
ncclNet_v6_t* g_exportNet = nullptr;
ncclCollNet_v6_t* g_exportCollNet = nullptr;
std::vector<std::string> g_dlsyms;

void* FakeDlsym(void* lib, const char* symbol) {
  g_dlsyms.push_back(symbol);
  if (lib != kLib) return nullptr;
  if (!std::strcmp(symbol, "ncclNetPlugin_v6")) return g_exportNet;
  if (!std::strcmp(symbol, "ncclCollNetPlugin_v6")) return g_exportCollNet;
  return nullptr;
}

}  // namespace

#define ncclOsDlsym FakeDlsym
#include NET_V6_CC_PATH
#undef ncclOsDlsym

namespace {

// What the v6 plugin was last called with.
struct Rec {
  int initCalls = 0;
  ncclResult_t initResult = ncclSuccess;
  ncclDebugLogger_t logfn = nullptr;
  ncclResult_t propsResult = ncclSuccess;
  int dev = -1;
  void* handle = nullptr;
  void* comm = nullptr;
  int size = 0;
  int tag = 0;
  int n = 0;
  int sizes[NCCL_PROXY_MAX_SUBS] = {};
  void* request = nullptr;
};
Rec g_net;
Rec g_coll;

ncclNetProperties_v6_t Props6() {
  ncclNetProperties_v6_t p{};
  p.name = const_cast<char*>("mlx5_0");
  p.pciPath = const_cast<char*>("/sys/pci0");
  p.guid = 0x1234;
  p.ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;
  p.speed = 400000;
  p.port = 1;
  p.latency = 1.5f;
  p.maxComms = 64;
  p.maxRecvs = 8;
  return p;
}

ncclNet_v6_t MakeNet() {
  ncclNet_v6_t n{};
  n.name = "Fake6";
  n.init = [](ncclDebugLogger_t logfn) {
    ++g_net.initCalls;
    g_net.logfn = logfn;
    return g_net.initResult;
  };
  n.devices = [](int*) { return ncclSuccess; };
  n.getProperties = [](int, ncclNetProperties_v6_t* p) {
    *p = Props6();
    return g_net.propsResult;
  };
  n.listen = [](int dev, void* handle, void** listenComm) {
    g_net.dev = dev;
    g_net.handle = handle;
    *listenComm = &g_net;
    return ncclSuccess;
  };
  n.connect = [](int dev, void* handle, void** sendComm) {
    g_net.dev = dev;
    g_net.handle = handle;
    *sendComm = &g_net;
    return ncclSuccess;
  };
  n.accept = [](void* listenComm, void** recvComm) {
    g_net.comm = listenComm;
    *recvComm = &g_net;
    return ncclSuccess;
  };
  n.regMr = [](void* comm, void*, int size, int, void**) {
    g_net.comm = comm;
    g_net.size = size;
    return ncclSuccess;
  };
  n.regMrDmaBuf = [](void*, void*, size_t, int, uint64_t, int, void**) { return ncclSuccess; };
  n.deregMr = [](void*, void*) { return ncclSuccess; };
  n.isend = [](void* comm, void*, int size, int tag, void*, void** request) {
    g_net.comm = comm;
    g_net.size = size;
    g_net.tag = tag;
    g_net.request = *request;
    return ncclSuccess;
  };
  n.irecv = [](void* comm, int count, void**, int* sizes, int*, void**, void** request) {
    g_net.comm = comm;
    g_net.n = count;
    for (int i = 0; i < count; i++) g_net.sizes[i] = sizes[i];
    g_net.request = *request;
    return ncclSuccess;
  };
  n.iflush = [](void*, int, void**, int*, void**, void**) { return ncclSuccess; };
  n.test = [](void*, int*, int*) { return ncclSuccess; };
  n.closeSend = [](void*) { return ncclSuccess; };
  n.closeRecv = [](void*) { return ncclSuccess; };
  n.closeListen = [](void*) { return ncclSuccess; };
  return n;
}

ncclCollNet_v6_t MakeCollNet() {
  ncclCollNet_v6_t c{};
  c.name = "FakeColl6";
  c.init = [](ncclDebugLogger_t logfn) {
    ++g_coll.initCalls;
    g_coll.logfn = logfn;
    return g_coll.initResult;
  };
  c.devices = [](int*) { return ncclSuccess; };
  c.getProperties = [](int, ncclNetProperties_v6_t* p) {
    *p = Props6();
    return g_coll.propsResult;
  };
  c.listen = [](int dev, void* handle, void** listenComm) {
    g_coll.dev = dev;
    g_coll.handle = handle;
    *listenComm = &g_coll;
    return ncclSuccess;
  };
  c.connect = [](void**, int, int, void*, void**) { return ncclSuccess; };
  c.reduceSupport = [](ncclDataType_t, ncclRedOp_t, int*) { return ncclSuccess; };
  c.regMr = [](void* comm, void*, int size, int, void**) {
    g_coll.comm = comm;
    g_coll.size = size;
    return ncclSuccess;
  };
  c.regMrDmaBuf = [](void*, void*, size_t, int, uint64_t, int, void**) { return ncclSuccess; };
  c.deregMr = [](void*, void*) { return ncclSuccess; };
  c.iallreduce = [](void* comm, void*, void*, int count, ncclDataType_t, ncclRedOp_t, void*, void*, void**) {
    g_coll.comm = comm;
    g_coll.size = count;
    return ncclSuccess;
  };
  c.iflush = [](void*, void*, int, void*, void**) { return ncclSuccess; };
  c.test = [](void*, int*, int*) { return ncclSuccess; };
  c.closeColl = [](void*) { return ncclSuccess; };
  c.closeListen = [](void*) { return ncclSuccess; };
  return c;
}

void TestLogger(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}

// The fields v6 cannot report are filled with fixed defaults.
void ExpectProps(const ncclNetProperties_t& props, int dev, ncclNetDeviceType type, int version) {
  EXPECT_STREQ(props.name, "mlx5_0");
  EXPECT_STREQ(props.pciPath, "/sys/pci0");
  EXPECT_EQ(props.guid, 0x1234u);
  EXPECT_EQ(props.ptrSupport, NCCL_PTR_HOST | NCCL_PTR_CUDA);
  EXPECT_EQ(props.regIsGlobal, 0);
  EXPECT_EQ(props.forceFlush, 0);
  EXPECT_EQ(props.speed, 400000);
  EXPECT_EQ(props.port, 1);
  EXPECT_EQ(props.maxComms, 64);
  EXPECT_EQ(props.maxRecvs, 8);
  EXPECT_FLOAT_EQ(props.latency, 1.5f);
  EXPECT_EQ(props.netDeviceType, type);
  EXPECT_EQ(props.netDeviceVersion, version);
  EXPECT_EQ(props.vProps.ndevs, 1);
  EXPECT_EQ(props.vProps.devs[0], dev);
  EXPECT_EQ(props.maxP2pBytes, size_t(MAX_NET_SIZE));
  EXPECT_EQ(props.maxCollBytes, size_t(MAX_COLLNET_SIZE));
  EXPECT_EQ(props.maxMultiRequestSize, 1);
  EXPECT_EQ(props.railId, NCCL_NET_ID_UNDEF);
  EXPECT_EQ(props.planeId, NCCL_NET_ID_UNDEF);
}

class PluginNetV6Microtest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_net = Rec{};
    g_coll = Rec{};
    g_dlsyms.clear();
    v6_ = MakeNet();
    coll6_ = MakeCollNet();
    g_exportNet = &v6_;
    g_exportCollNet = &coll6_;
    refCount[NET_INDEX] = refCount[COLLNET_INDEX] = 0;
  }
  void TearDown() override { refCount[NET_INDEX] = refCount[COLLNET_INDEX] = 0; }

  ncclNet_t* InitNet() {
    ncclNet_t* net = getNcclNet_v6(kLib);
    ncclNetCommConfig_t config{};
    void* ctx = nullptr;
    EXPECT_EQ(net->init(&ctx, 1, &config, TestLogger, nullptr), ncclSuccess);
    return net;
  }
  ncclCollNet_t* InitCollNet() {
    ncclCollNet_t* coll = getNcclCollNet_v6(kLib);
    void* ctx = nullptr;
    EXPECT_EQ(coll->init(&ctx, 1, TestLogger), ncclSuccess);
    return coll;
  }

  ncclNet_v6_t v6_;
  ncclCollNet_v6_t coll6_;
};

TEST_F(PluginNetV6Microtest, GettersNeedTheV6Symbols) {
  EXPECT_EQ(getNcclNet_v6(nullptr), nullptr);
  EXPECT_EQ(getNcclCollNet_v6(nullptr), nullptr);
  EXPECT_EQ(g_dlsyms, (std::vector<std::string>{"ncclNetPlugin_v6", "ncclCollNetPlugin_v6"}));

  ncclNet_t* net = getNcclNet_v6(kLib);
  ASSERT_NE(net, nullptr);
  EXPECT_STREQ(net->name, "Fake6");
  EXPECT_NE(net->init, nullptr);
  ncclCollNet_t* coll = getNcclCollNet_v6(kLib);
  ASSERT_NE(coll, nullptr);
  EXPECT_STREQ(coll->name, "FakeColl6");
}

// v6 plugins expect a single init: later comms share it until every comm has finalized.
TEST_F(PluginNetV6Microtest, NetInitRunsOnceWhileAnyCommHoldsIt) {
  ncclNet_t* net = InitNet();
  InitNet();
  EXPECT_EQ(g_net.initCalls, 1);
  EXPECT_EQ(g_net.logfn, &TestLogger);

  EXPECT_EQ(net->finalize(nullptr), ncclSuccess);
  InitNet();
  EXPECT_EQ(g_net.initCalls, 1);
  EXPECT_EQ(net->finalize(nullptr), ncclSuccess);
  EXPECT_EQ(net->finalize(nullptr), ncclSuccess);
  InitNet();
  EXPECT_EQ(g_net.initCalls, 2);
}

TEST_F(PluginNetV6Microtest, AFailedNetInitIsRetriedByTheNextComm) {
  ncclNet_t* net = getNcclNet_v6(kLib);
  ncclNetCommConfig_t config{};
  void* ctx = nullptr;
  g_net.initResult = ncclSystemError;
  EXPECT_EQ(net->init(&ctx, 1, &config, TestLogger, nullptr), ncclSystemError);
  g_net.initResult = ncclSuccess;
  EXPECT_EQ(net->init(&ctx, 1, &config, TestLogger, nullptr), ncclSuccess);
  EXPECT_EQ(g_net.initCalls, 2);
}

TEST_F(PluginNetV6Microtest, NetInitWiresPassThroughsAndLeavesNewerEntriesNull) {
  ncclNet_t* net = InitNet();
  EXPECT_EQ(net->devices, v6_.devices);
  EXPECT_EQ(net->regMrDmaBuf, v6_.regMrDmaBuf);
  EXPECT_EQ(net->deregMr, v6_.deregMr);
  EXPECT_EQ(net->iflush, v6_.iflush);
  EXPECT_EQ(net->test, v6_.test);
  EXPECT_EQ(net->closeSend, v6_.closeSend);
  EXPECT_EQ(net->closeRecv, v6_.closeRecv);
  EXPECT_EQ(net->closeListen, v6_.closeListen);
  EXPECT_EQ(net->getDeviceMr, nullptr);
  EXPECT_EQ(net->irecvConsumed, nullptr);
  EXPECT_EQ(net->makeVDevice, nullptr);
  EXPECT_EQ(net->setNetAttr, nullptr);
}

TEST_F(PluginNetV6Microtest, GetPropertiesFillsWhatV6CannotReport) {
  ncclNet_t* net = InitNet();
  ncclNetProperties_t props;
  std::memset(&props, 0xff, sizeof(props));
  ASSERT_EQ(net->getProperties(3, &props), ncclSuccess);

  ExpectProps(props, 3, NCCL_NET_DEVICE_HOST, NCCL_NET_DEVICE_INVALID_VERSION);

  g_net.propsResult = ncclSystemError;
  EXPECT_EQ(net->getProperties(3, &props), ncclSystemError);
}

TEST_F(PluginNetV6Microtest, ConnectionCallsDropTheContextAndDeviceHandle) {
  ncclNet_t* net = InitNet();
  char handle[8];
  void* comm = nullptr;
  ASSERT_EQ(net->listen(&g_coll, 2, handle, &comm), ncclSuccess);
  EXPECT_EQ(g_net.dev, 2);
  EXPECT_EQ(g_net.handle, handle);
  EXPECT_EQ(comm, &g_net);

  g_net = Rec{};
  comm = nullptr;
  ASSERT_EQ(net->connect(&g_coll, 4, handle, &comm, nullptr), ncclSuccess);
  EXPECT_EQ(g_net.dev, 4);
  EXPECT_EQ(g_net.handle, handle);
  EXPECT_EQ(comm, &g_net);

  comm = nullptr;
  ASSERT_EQ(net->accept(handle, &comm, nullptr), ncclSuccess);
  EXPECT_EQ(g_net.comm, handle);
  EXPECT_EQ(comm, &g_net);
}

TEST_F(PluginNetV6Microtest, RegMrRejectsSizesThatDoNotFitAnInt) {
  ncclNet_t* net = InitNet();
  char comm;
  EXPECT_EQ(net->regMr(&comm, nullptr, size_t(1) << 31, NCCL_PTR_HOST, nullptr), ncclInternalError);
  EXPECT_EQ(g_net.comm, nullptr);
  ASSERT_EQ(net->regMr(&comm, nullptr, (size_t(1) << 31) - 1, NCCL_PTR_HOST, nullptr), ncclSuccess);
  EXPECT_EQ(g_net.comm, &comm);
  EXPECT_EQ(g_net.size, 0x7fffffff);
}

TEST_F(PluginNetV6Microtest, IsendCapsAtMaxNetSize) {
  ncclNet_t* net = InitNet();
  char comm;
  void* request = &comm;
  EXPECT_EQ(net->isend(&comm, nullptr, MAX_NET_SIZE + 1, 7, nullptr, nullptr, &request), ncclInternalError);
  EXPECT_EQ(g_net.comm, nullptr);
  ASSERT_EQ(net->isend(&comm, nullptr, MAX_NET_SIZE, 7, nullptr, nullptr, &request), ncclSuccess);
  EXPECT_EQ(g_net.comm, &comm);
  EXPECT_EQ(g_net.size, MAX_NET_SIZE);
  EXPECT_EQ(g_net.tag, 7);
  EXPECT_EQ(g_net.request, &comm);
}

// v6 has no optional receive completion, so the marker must not reach the plugin as a request pointer.
TEST_F(PluginNetV6Microtest, IrecvClearsTheOptionalCompletionMarkerAndNarrowsSizes) {
  ncclNet_t* net = InitNet();
  char comm;
  size_t sizes[2] = {16, MAX_NET_SIZE};
  void* request = reinterpret_cast<void*>(NCCL_NET_OPTIONAL_RECV_COMPLETION);
  ASSERT_EQ(net->irecv(&comm, 2, nullptr, sizes, nullptr, nullptr, nullptr, &request), ncclSuccess);
  EXPECT_EQ(g_net.n, 2);
  EXPECT_EQ(g_net.sizes[0], 16);
  EXPECT_EQ(g_net.sizes[1], MAX_NET_SIZE);
  EXPECT_EQ(g_net.request, nullptr);

  request = &comm;
  ASSERT_EQ(net->irecv(&comm, 1, nullptr, sizes, nullptr, nullptr, nullptr, &request), ncclSuccess);
  EXPECT_EQ(g_net.request, &comm);

  g_net = Rec{};
  sizes[1] = MAX_NET_SIZE + 1;
  EXPECT_EQ(net->irecv(&comm, 2, nullptr, sizes, nullptr, nullptr, nullptr, &request), ncclInternalError);
  EXPECT_EQ(g_net.n, 0);
}

// The proxy sizes its own arrays by NCCL_PROXY_MAX_SUBS, so that is the largest n the shim's int copy must hold.
TEST_F(PluginNetV6Microtest, IrecvNarrowsAFullSetOfSubs) {
  ncclNet_t* net = InitNet();
  char comm;
  void* request = &comm;
  size_t sizes[NCCL_PROXY_MAX_SUBS];
  for (int i = 0; i < NCCL_PROXY_MAX_SUBS; i++) sizes[i] = 100 + i;
  sizes[NCCL_PROXY_MAX_SUBS - 1] = MAX_NET_SIZE;
  ASSERT_EQ(net->irecv(&comm, NCCL_PROXY_MAX_SUBS, nullptr, sizes, nullptr, nullptr, nullptr, &request), ncclSuccess);
  EXPECT_EQ(g_net.n, NCCL_PROXY_MAX_SUBS);
  for (int i = 0; i < NCCL_PROXY_MAX_SUBS - 1; i++) EXPECT_EQ(g_net.sizes[i], 100 + i) << i;
  EXPECT_EQ(g_net.sizes[NCCL_PROXY_MAX_SUBS - 1], MAX_NET_SIZE);

  g_net = Rec{};
  sizes[NCCL_PROXY_MAX_SUBS - 1] = MAX_NET_SIZE + 1;
  EXPECT_EQ(net->irecv(&comm, NCCL_PROXY_MAX_SUBS, nullptr, sizes, nullptr, nullptr, nullptr, &request),
            ncclInternalError);
  EXPECT_EQ(g_net.n, 0);
}

TEST_F(PluginNetV6Microtest, CollNetInitRunsOnceWhileAnyCommHoldsIt) {
  ncclCollNet_t* coll = InitCollNet();
  InitCollNet();
  EXPECT_EQ(g_coll.initCalls, 1);
  EXPECT_EQ(g_coll.logfn, &TestLogger);
  EXPECT_EQ(coll->finalize(nullptr), ncclSuccess);
  EXPECT_EQ(coll->finalize(nullptr), ncclSuccess);
  InitCollNet();
  EXPECT_EQ(g_coll.initCalls, 2);

  coll->finalize(nullptr);
  g_coll.initResult = ncclSystemError;
  void* ctx = nullptr;
  EXPECT_EQ(coll->init(&ctx, 1, TestLogger), ncclSystemError);
  EXPECT_EQ(refCount[COLLNET_INDEX], 0);
}

TEST_F(PluginNetV6Microtest, CollNetInitWiresPassThroughsAndLeavesNewerEntriesNull) {
  ncclCollNet_t* coll = InitCollNet();
  EXPECT_EQ(coll->devices, coll6_.devices);
  EXPECT_EQ(coll->connect, coll6_.connect);
  EXPECT_EQ(coll->reduceSupport, coll6_.reduceSupport);
  EXPECT_EQ(coll->regMrDmaBuf, coll6_.regMrDmaBuf);
  EXPECT_EQ(coll->deregMr, coll6_.deregMr);
  EXPECT_EQ(coll->iflush, coll6_.iflush);
  EXPECT_EQ(coll->test, coll6_.test);
  EXPECT_EQ(coll->closeColl, coll6_.closeColl);
  EXPECT_EQ(coll->closeListen, coll6_.closeListen);
  EXPECT_EQ(coll->iallgather, nullptr);
  EXPECT_EQ(coll->ireducescatter, nullptr);
  EXPECT_EQ(coll->makeVDevice, nullptr);
}

TEST_F(PluginNetV6Microtest, CollNetGetPropertiesFillsWhatV6CannotReport) {
  ncclCollNet_t* coll = InitCollNet();
  ncclNetProperties_t props;
  std::memset(&props, 0xff, sizeof(props));
  ASSERT_EQ(coll->getProperties(5, &props), ncclSuccess);

  ExpectProps(props, 5, NCCL_NET_DEVICE_HOST, NCCL_NET_DEVICE_INVALID_VERSION);

  g_coll.propsResult = ncclSystemError;
  EXPECT_EQ(coll->getProperties(5, &props), ncclSystemError);
}

TEST_F(PluginNetV6Microtest, CollNetCallsNarrowToInt) {
  ncclCollNet_t* coll = InitCollNet();
  char handle[8], comm;
  void* listenComm = nullptr;
  ASSERT_EQ(coll->listen(&g_net, 6, handle, &listenComm), ncclSuccess);
  EXPECT_EQ(g_coll.dev, 6);
  EXPECT_EQ(g_coll.handle, handle);
  EXPECT_EQ(listenComm, &g_coll);

  EXPECT_EQ(coll->regMr(&comm, nullptr, size_t(1) << 31, NCCL_PTR_HOST, nullptr), ncclInternalError);
  EXPECT_EQ(g_coll.comm, nullptr);
  ASSERT_EQ(coll->regMr(&comm, nullptr, 4096, NCCL_PTR_HOST, nullptr), ncclSuccess);
  EXPECT_EQ(g_coll.comm, &comm);
  EXPECT_EQ(g_coll.size, 4096);

  g_coll = Rec{};
  EXPECT_EQ(coll->iallreduce(&comm, nullptr, nullptr, MAX_NET_SIZE + 1, ncclFloat, ncclSum, nullptr, nullptr, nullptr),
            ncclInternalError);
  EXPECT_EQ(g_coll.comm, nullptr);
  ASSERT_EQ(coll->iallreduce(&comm, nullptr, nullptr, MAX_NET_SIZE, ncclFloat, ncclSum, nullptr, nullptr, nullptr),
            ncclSuccess);
  EXPECT_EQ(g_coll.comm, &comm);
  EXPECT_EQ(g_coll.size, MAX_NET_SIZE);
}

}  // namespace
