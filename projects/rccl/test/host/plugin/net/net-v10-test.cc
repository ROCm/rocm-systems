/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/net/net_v10.cc, the adapter that presents a v10 net/collnet plugin as the
// current ncclNet_t / ncclCollNet_t: one-time init behind a refcount, with the per-comm config the v10 connect()
// takes carried in the context.

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "alloc.h"
#include "checks.h"
#include "nccl_net.h"
#include "os.h"
#include "proxy.h"

namespace {

void* const kLib = reinterpret_cast<void*>(0x10);
ncclNet_v10_t* g_exportNet = nullptr;
ncclCollNet_v10_t* g_exportCollNet = nullptr;
std::vector<std::string> g_dlsyms;

void* FakeDlsym(void* lib, const char* symbol) {
  g_dlsyms.push_back(symbol);
  if (lib != kLib) return nullptr;
  if (!std::strcmp(symbol, "ncclNetPlugin_v10")) return g_exportNet;
  if (!std::strcmp(symbol, "ncclCollNetPlugin_v10")) return g_exportCollNet;
  return nullptr;
}

// The per-comm configs the shim allocates, so a test can see each one freed.
bool g_callocFails = false;
int g_liveConfigs = 0;

template <typename T>
ncclResult_t FakeCalloc(T** ptr, size_t nelem) {
  if (g_callocFails) return ncclSystemError;
  *ptr = static_cast<T*>(calloc(nelem, sizeof(T)));
  ++g_liveConfigs;
  return ncclSuccess;
}

void FakeFree(void* ptr) {
  if (ptr) --g_liveConfigs;
  free(ptr);
}

}  // namespace

#define ncclOsDlsym FakeDlsym
#undef ncclCalloc
#define ncclCalloc FakeCalloc
#define free FakeFree
#include NET_V10_CC_PATH
#undef free
#undef ncclCalloc
#undef ncclOsDlsym

namespace {

// What the v10 plugin was last called with.
struct Rec {
  int initCalls = 0;
  ncclResult_t initResult = ncclSuccess;
  ncclDebugLogger_t logfn = nullptr;
  ncclProfilerCallback_t proffn = nullptr;
  ncclResult_t propsResult = ncclSuccess;
  int dev = -1;
  void* handle = nullptr;
  void* comm = nullptr;
  size_t size = 0;
  int tag = 0;
  int n = 0;
  size_t sizes[3] = {};
  void* ptr = nullptr;
};
Rec g_net;
Rec g_coll;

ncclNetProperties_v10_t Props10() {
  ncclNetProperties_v10_t p{};
  p.name = const_cast<char*>("mlx5_1");
  p.pciPath = const_cast<char*>("/sys/pci1");
  p.guid = 0x99;
  p.ptrSupport = NCCL_PTR_HOST | NCCL_PTR_DMABUF;
  p.regIsGlobal = 1;
  p.forceFlush = 1;
  p.speed = 200000;
  p.port = 2;
  p.latency = 2.5f;
  p.maxComms = 32;
  p.maxRecvs = 4;
  p.netDeviceType = NCCL_NET_DEVICE_UNPACK;
  p.netDeviceVersion = 7;
  p.vProps.ndevs = 2;
  p.vProps.devs[0] = 5;
  p.vProps.devs[1] = 6;
  p.maxP2pBytes = size_t(1) << 40;
  p.maxCollBytes = size_t(1) << 33;
  return p;
}

ncclResult_t MakeVDevice(Rec& rec, int* d, ncclNetVDeviceProps_v10_t* props) {
  rec.ptr = props;
  rec.n = props->ndevs;
  *d = 42;
  return ncclRemoteError;
}

ncclNet_v10_t MakeNet() {
  ncclNet_v10_t n{};
  n.name = "Fake10";
  n.init = [](ncclDebugLogger_t logfn, ncclProfilerCallback_t proffn) {
    ++g_net.initCalls;
    g_net.logfn = logfn;
    g_net.proffn = proffn;
    return g_net.initResult;
  };
  n.devices = [](int*) { return ncclSuccess; };
  n.getProperties = [](int, ncclNetProperties_v10_t* p) {
    *p = Props10();
    return g_net.propsResult;
  };
  n.listen = [](int dev, void* handle, void** listenComm) {
    g_net.dev = dev;
    g_net.handle = handle;
    *listenComm = &g_net;
    return ncclSuccess;
  };
  n.connect = [](int dev, ncclNetCommConfig_v10_t* config, void* handle, void** sendComm,
                 ncclNetDeviceHandle_v10_t** sendDevComm) {
    g_net.dev = dev;
    g_net.tag = config->trafficClass;
    g_net.handle = handle;
    g_net.ptr = sendDevComm;
    *sendComm = &g_net;
    return ncclSuccess;
  };
  n.accept = [](void*, void**, ncclNetDeviceHandle_v10_t**) { return ncclSuccess; };
  n.regMr = [](void*, void*, size_t, int, void**) { return ncclSuccess; };
  n.regMrDmaBuf = [](void*, void*, size_t, int, uint64_t, int, void**) { return ncclSuccess; };
  n.deregMr = [](void*, void*) { return ncclSuccess; };
  n.isend = [](void*, void*, size_t, int, void*, void*, void**) { return ncclSuccess; };
  n.irecv = [](void*, int, void**, size_t*, int*, void**, void**, void**) { return ncclSuccess; };
  n.iflush = [](void*, int, void**, int*, void**, void**) { return ncclSuccess; };
  n.test = [](void*, int*, int*) { return ncclSuccess; };
  n.closeSend = [](void*) { return ncclSuccess; };
  n.closeRecv = [](void*) { return ncclSuccess; };
  n.closeListen = [](void*) { return ncclSuccess; };
  n.getDeviceMr = [](void*, void*, void**) { return ncclSuccess; };
  n.irecvConsumed = [](void*, int, void*) { return ncclSuccess; };
  n.makeVDevice = [](int* d, ncclNetVDeviceProps_v10_t* props) { return MakeVDevice(g_net, d, props); };
  return n;
}

ncclCollNet_v10_t MakeCollNet() {
  ncclCollNet_v10_t c{};
  c.name = "FakeColl10";
  c.init = [](ncclDebugLogger_t logfn) {
    ++g_coll.initCalls;
    g_coll.logfn = logfn;
    return g_coll.initResult;
  };
  c.devices = [](int*) { return ncclSuccess; };
  c.getProperties = [](int, ncclNetProperties_v10_t* p) {
    *p = Props10();
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
  c.regMr = [](void*, void*, size_t, int, void**) { return ncclSuccess; };
  c.regMrDmaBuf = [](void*, void*, size_t, int, uint64_t, int, void**) { return ncclSuccess; };
  c.deregMr = [](void*, void*) { return ncclSuccess; };
  c.iallreduce = [](void*, void*, void*, size_t, ncclDataType_t, ncclRedOp_t, void*, void*, void**) {
    return ncclSuccess;
  };
  c.iallgather = [](void* comm, void*, int nParts, ncclNetSGE_v10_t* parts, size_t bytesPerRank, size_t offset,
                    size_t windowBytes, void*, void**) {
    g_coll.comm = comm;
    g_coll.n = nParts;
    g_coll.ptr = parts;
    g_coll.sizes[0] = bytesPerRank;
    g_coll.sizes[1] = offset;
    g_coll.sizes[2] = windowBytes;
    return ncclSuccess;
  };
  c.ireducescatter = [](void* comm, int nParts, ncclNetSGE_v10_t* parts, void*, size_t bytesPerRank, size_t offset,
                        size_t windowBytes, ncclDataType_t, ncclRedOp_t, void*, void**) {
    g_coll.comm = comm;
    g_coll.n = nParts;
    g_coll.ptr = parts;
    g_coll.sizes[0] = bytesPerRank;
    g_coll.sizes[1] = offset;
    g_coll.sizes[2] = windowBytes;
    return ncclSuccess;
  };
  c.iflush = [](void*, void*, int, void*, void**) { return ncclSuccess; };
  c.test = [](void*, int*, int*) { return ncclSuccess; };
  c.closeColl = [](void*) { return ncclSuccess; };
  c.closeListen = [](void*) { return ncclSuccess; };
  c.makeVDevice = [](int* d, ncclNetVDeviceProps_v10_t* props) { return MakeVDevice(g_coll, d, props); };
  return c;
}

void TestLogger(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}
ncclResult_t ProfilerCallback(void**, int, void*, int64_t, void*) { return ncclSuccess; }

void ExpectProps10(const ncclNetProperties_t& props) {
  EXPECT_STREQ(props.name, "mlx5_1");
  EXPECT_STREQ(props.pciPath, "/sys/pci1");
  EXPECT_EQ(props.guid, 0x99u);
  EXPECT_EQ(props.ptrSupport, NCCL_PTR_HOST | NCCL_PTR_DMABUF);
  EXPECT_EQ(props.regIsGlobal, 1);
  EXPECT_EQ(props.forceFlush, 1);
  EXPECT_EQ(props.speed, 200000);
  EXPECT_EQ(props.port, 2);
  EXPECT_FLOAT_EQ(props.latency, 2.5f);
  EXPECT_EQ(props.maxComms, 32);
  EXPECT_EQ(props.maxRecvs, 4);
  EXPECT_EQ(props.netDeviceType, NCCL_NET_DEVICE_UNPACK);
  EXPECT_EQ(props.netDeviceVersion, 7);
  EXPECT_EQ(props.vProps.ndevs, 2);
  EXPECT_EQ(props.vProps.devs[0], 5);
  EXPECT_EQ(props.vProps.devs[1], 6);
  EXPECT_EQ(props.vProps.devs[2], -1);  // only ndevs entries are copied
  EXPECT_EQ(props.maxP2pBytes, size_t(1) << 40);
  EXPECT_EQ(props.maxCollBytes, size_t(1) << 33);
  EXPECT_EQ(props.maxMultiRequestSize, 1);
  EXPECT_EQ(props.railId, NCCL_NET_ID_UNDEF);
  EXPECT_EQ(props.planeId, NCCL_NET_ID_UNDEF);
}

class PluginNetV10Microtest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_net = Rec{};
    g_coll = Rec{};
    g_dlsyms.clear();
    g_liveConfigs = 0;
    v10_ = MakeNet();
    coll10_ = MakeCollNet();
    g_exportNet = &v10_;
    g_exportCollNet = &coll10_;
    refCount[NET_INDEX] = refCount[COLLNET_INDEX] = 0;
  }
  void TearDown() override {
    for (void* ctx : ctxs_) ncclNet_finalize(ctx);
    refCount[NET_INDEX] = refCount[COLLNET_INDEX] = 0;
  }

  ncclNet_t* InitNet(int trafficClass = 3) {
    ncclNet_t* net = getNcclNet_v10(kLib);
    ncclNetCommConfig_t config{trafficClass};
    void* ctx = nullptr;
    EXPECT_EQ(net->init(&ctx, 1, &config, TestLogger, ProfilerCallback), ncclSuccess);
    ctxs_.push_back(ctx);
    return net;
  }
  ncclCollNet_t* InitCollNet() {
    ncclCollNet_t* coll = getNcclCollNet_v10(kLib);
    void* ctx = nullptr;
    EXPECT_EQ(coll->init(&ctx, 1, TestLogger), ncclSuccess);
    return coll;
  }

  ncclNet_v10_t v10_;
  ncclCollNet_v10_t coll10_;
  std::vector<void*> ctxs_;
};

TEST_F(PluginNetV10Microtest, GettersNeedTheV10Symbols) {
  EXPECT_EQ(getNcclNet_v10(nullptr), nullptr);
  EXPECT_EQ(getNcclCollNet_v10(nullptr), nullptr);
  EXPECT_EQ(g_dlsyms, (std::vector<std::string>{"ncclNetPlugin_v10", "ncclCollNetPlugin_v10"}));

  ncclNet_t* net = getNcclNet_v10(kLib);
  ASSERT_NE(net, nullptr);
  EXPECT_STREQ(net->name, "Fake10");
  EXPECT_NE(net->init, nullptr);
  ncclCollNet_t* coll = getNcclCollNet_v10(kLib);
  ASSERT_NE(coll, nullptr);
  EXPECT_STREQ(coll->name, "FakeColl10");
}

TEST_F(PluginNetV10Microtest, InitRunsOnceWhileAnyCommHoldsIt) {
  ncclNet_t* net = getNcclNet_v10(kLib);
  ncclNetCommConfig_t config{};
  void* ctx = &config;
  g_net.initResult = ncclSystemError;
  EXPECT_EQ(net->init(&ctx, 1, &config, TestLogger, nullptr), ncclSystemError);
  EXPECT_EQ(ctx, nullptr);
  EXPECT_EQ(g_liveConfigs, 0);
  g_net.initResult = ncclSuccess;
  InitNet();
  InitNet();
  EXPECT_EQ(g_net.initCalls, 2);
  EXPECT_EQ(g_net.logfn, &TestLogger);
  EXPECT_EQ(g_net.proffn, &ProfilerCallback);
  EXPECT_EQ(g_liveConfigs, 2);
  EXPECT_EQ(net->finalize(ctxs_[0]), ncclSuccess);
  EXPECT_EQ(net->finalize(ctxs_[1]), ncclSuccess);
  EXPECT_EQ(g_liveConfigs, 0);
  ctxs_.clear();
  InitNet();
  EXPECT_EQ(g_net.initCalls, 3);

  ncclCollNet_t* coll = getNcclCollNet_v10(kLib);
  g_coll.initResult = ncclSystemError;
  EXPECT_EQ(coll->init(&ctx, 1, TestLogger), ncclSystemError);
  g_coll.initResult = ncclSuccess;
  InitCollNet();
  InitCollNet();
  EXPECT_EQ(g_coll.initCalls, 2);
  EXPECT_EQ(g_coll.logfn, &TestLogger);
  EXPECT_EQ(coll->finalize(nullptr), ncclSuccess);
  EXPECT_EQ(coll->finalize(nullptr), ncclSuccess);
  InitCollNet();
  EXPECT_EQ(g_coll.initCalls, 3);
}

TEST_F(PluginNetV10Microtest, NetInitWiresPassThroughs) {
  ncclNet_t* net = InitNet();
  EXPECT_EQ(net->devices, v10_.devices);
  EXPECT_EQ(net->accept, v10_.accept);
  EXPECT_EQ(net->isend, v10_.isend);
  EXPECT_EQ(net->irecv, v10_.irecv);
  EXPECT_EQ(net->regMr, v10_.regMr);
  EXPECT_EQ(net->regMrDmaBuf, v10_.regMrDmaBuf);
  EXPECT_EQ(net->deregMr, v10_.deregMr);
  EXPECT_EQ(net->iflush, v10_.iflush);
  EXPECT_EQ(net->test, v10_.test);
  EXPECT_EQ(net->closeSend, v10_.closeSend);
  EXPECT_EQ(net->closeRecv, v10_.closeRecv);
  EXPECT_EQ(net->closeListen, v10_.closeListen);
  EXPECT_EQ(net->getDeviceMr, v10_.getDeviceMr);
  EXPECT_EQ(net->irecvConsumed, v10_.irecvConsumed);
  EXPECT_EQ(net->setNetAttr, nullptr);
}

TEST_F(PluginNetV10Microtest, MakeVDeviceIsWrappedOnlyWhenThePluginHasOne) {
  ncclNet_t* net = InitNet();
  ncclNetVDeviceProps_t vprops{};
  vprops.ndevs = 3;
  int d = -1;
  ASSERT_NE(net->makeVDevice, nullptr);
  EXPECT_EQ(net->makeVDevice(&d, &vprops), ncclRemoteError);
  EXPECT_EQ(d, 42);
  EXPECT_EQ(g_net.ptr, &vprops);
  EXPECT_EQ(g_net.n, 3);

  ncclCollNet_t* coll = InitCollNet();
  ASSERT_NE(coll->makeVDevice, nullptr);
  EXPECT_EQ(coll->makeVDevice(&d, &vprops), ncclRemoteError);
  EXPECT_EQ(g_coll.ptr, &vprops);

  refCount[NET_INDEX] = refCount[COLLNET_INDEX] = 0;
  v10_.makeVDevice = nullptr;
  coll10_.makeVDevice = nullptr;
  EXPECT_EQ(InitNet()->makeVDevice, nullptr);
  EXPECT_EQ(InitCollNet()->makeVDevice, nullptr);
}

TEST_F(PluginNetV10Microtest, GetPropertiesCopiesEveryV10Field) {
  ncclNetProperties_t props;
  std::memset(&props, 0xff, sizeof(props));
  ASSERT_EQ(InitNet()->getProperties(0, &props), ncclSuccess);
  ExpectProps10(props);

  std::memset(&props, 0xff, sizeof(props));
  ASSERT_EQ(InitCollNet()->getProperties(0, &props), ncclSuccess);
  ExpectProps10(props);

  g_net.propsResult = g_coll.propsResult = ncclSystemError;
  EXPECT_EQ(InitNet()->getProperties(0, &props), ncclSystemError);
  EXPECT_EQ(InitCollNet()->getProperties(0, &props), ncclSystemError);
}

// Each comm's own traffic class reaches connect(), even though only the first comm ran the plugin's init.
TEST_F(PluginNetV10Microtest, ConnectGetsTheTrafficClassItsCommWasInitializedWith) {
  ncclNet_t* net = InitNet(5);
  InitNet(7);
  char handle[8];
  void* out = nullptr;
  ASSERT_EQ(net->listen(ctxs_[0], 2, handle, &out), ncclSuccess);
  EXPECT_EQ(g_net.dev, 2);
  EXPECT_EQ(g_net.handle, handle);
  EXPECT_EQ(out, &g_net);

  ncclNetDeviceHandle_t* devComm = nullptr;
  ASSERT_EQ(net->connect(ctxs_[1], 4, handle, &out, &devComm), ncclSuccess);
  EXPECT_EQ(g_net.dev, 4);
  EXPECT_EQ(g_net.tag, 7);
  EXPECT_EQ(g_net.ptr, &devComm);
  ASSERT_EQ(net->connect(ctxs_[0], 4, handle, &out, &devComm), ncclSuccess);
  EXPECT_EQ(g_net.tag, 5);
}

TEST_F(PluginNetV10Microtest, AFailedConfigAllocationFailsInit) {
  ncclNet_t* net = getNcclNet_v10(kLib);
  ncclNetCommConfig_t config{};
  void* ctx = &config;
  g_callocFails = true;
  EXPECT_EQ(net->init(&ctx, 1, &config, TestLogger, nullptr), ncclSystemError);
  g_callocFails = false;
  EXPECT_EQ(ctx, nullptr);
  EXPECT_EQ(g_net.initCalls, 0);
  EXPECT_EQ(refCount[NET_INDEX], 0);
}

TEST_F(PluginNetV10Microtest, CollNetInitWiresPassThroughs) {
  ncclCollNet_t* coll = InitCollNet();
  EXPECT_EQ(coll->devices, coll10_.devices);
  EXPECT_EQ(coll->connect, coll10_.connect);
  EXPECT_EQ(coll->reduceSupport, coll10_.reduceSupport);
  EXPECT_EQ(coll->regMr, coll10_.regMr);
  EXPECT_EQ(coll->regMrDmaBuf, coll10_.regMrDmaBuf);
  EXPECT_EQ(coll->deregMr, coll10_.deregMr);
  EXPECT_EQ(coll->iallreduce, coll10_.iallreduce);
  EXPECT_EQ(coll->iflush, coll10_.iflush);
  EXPECT_EQ(coll->test, coll10_.test);
  EXPECT_EQ(coll->closeColl, coll10_.closeColl);
  EXPECT_EQ(coll->closeListen, coll10_.closeListen);
}

TEST_F(PluginNetV10Microtest, CollNetCallsForwardEverything) {
  ncclCollNet_t* coll = InitCollNet();
  char handle[8], comm;
  void* out = nullptr;
  ASSERT_EQ(coll->listen(&g_net, 6, handle, &out), ncclSuccess);
  EXPECT_EQ(g_coll.dev, 6);
  EXPECT_EQ(g_coll.handle, handle);
  EXPECT_EQ(out, &g_coll);

  ncclNetSGE_t parts[2] = {};
  ASSERT_EQ(coll->iallgather(&comm, nullptr, 2, parts, 64, 128, 256, nullptr, nullptr), ncclSuccess);
  EXPECT_EQ(g_coll.comm, &comm);
  EXPECT_EQ(g_coll.n, 2);
  EXPECT_EQ(g_coll.ptr, parts);
  EXPECT_EQ(g_coll.sizes[0], 64u);
  EXPECT_EQ(g_coll.sizes[1], 128u);
  EXPECT_EQ(g_coll.sizes[2], 256u);

  g_coll = Rec{};
  ASSERT_EQ(coll->ireducescatter(&comm, 2, parts, nullptr, 64, 128, 256, ncclFloat, ncclSum, nullptr, nullptr),
            ncclSuccess);
  EXPECT_EQ(g_coll.comm, &comm);
  EXPECT_EQ(g_coll.n, 2);
  EXPECT_EQ(g_coll.ptr, parts);
  EXPECT_EQ(g_coll.sizes[0], 64u);
  EXPECT_EQ(g_coll.sizes[1], 128u);
  EXPECT_EQ(g_coll.sizes[2], 256u);
}

}  // namespace
