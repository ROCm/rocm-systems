/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/net/net_v9.cc, the adapter that presents a v9 net/collnet plugin as the
// current ncclNet_t / ncclCollNet_t: one-time init behind a refcount, context-free calls and optional makeVDevice.

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "nccl_net.h"
#include "os.h"

namespace {

void* const kLib = reinterpret_cast<void*>(0x9);
ncclNet_v9_t* g_exportNet = nullptr;
ncclCollNet_v9_t* g_exportCollNet = nullptr;
std::vector<std::string> g_dlsyms;

void* FakeDlsym(void* lib, const char* symbol) {
  g_dlsyms.push_back(symbol);
  if (lib != kLib) return nullptr;
  if (!std::strcmp(symbol, "ncclNetPlugin_v9")) return g_exportNet;
  if (!std::strcmp(symbol, "ncclCollNetPlugin_v9")) return g_exportCollNet;
  return nullptr;
}

}  // namespace

#define ncclOsDlsym FakeDlsym
#include NET_V9_CC_PATH
#undef ncclOsDlsym

namespace {

// What the v9 plugin was last called with.
struct Rec {
  int initCalls = 0;
  ncclResult_t initResult = ncclSuccess;
  ncclDebugLogger_t logfn = nullptr;
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

ncclNetProperties_v9_t Props9() {
  ncclNetProperties_v9_t p{};
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

ncclResult_t MakeVDevice(Rec& rec, int* d, ncclNetVDeviceProps_v9_t* props) {
  rec.ptr = props;
  rec.n = props->ndevs;
  *d = 42;
  return ncclRemoteError;
}

ncclNet_v9_t MakeNet() {
  ncclNet_v9_t n{};
  n.name = "Fake9";
  n.init = [](ncclDebugLogger_t logfn) {
    ++g_net.initCalls;
    g_net.logfn = logfn;
    return g_net.initResult;
  };
  n.devices = [](int*) { return ncclSuccess; };
  n.getProperties = [](int, ncclNetProperties_v9_t* p) {
    *p = Props9();
    return g_net.propsResult;
  };
  n.listen = [](int dev, void* handle, void** listenComm) {
    g_net.dev = dev;
    g_net.handle = handle;
    *listenComm = &g_net;
    return ncclSuccess;
  };
  n.connect = [](int dev, void* handle, void** sendComm, ncclNetDeviceHandle_v9_t** sendDevComm) {
    g_net.dev = dev;
    g_net.handle = handle;
    g_net.ptr = sendDevComm;
    *sendComm = &g_net;
    return ncclSuccess;
  };
  n.accept = [](void*, void**, ncclNetDeviceHandle_v9_t**) { return ncclSuccess; };
  n.regMr = [](void*, void*, size_t, int, void**) { return ncclSuccess; };
  n.regMrDmaBuf = [](void*, void*, size_t, int, uint64_t, int, void**) { return ncclSuccess; };
  n.deregMr = [](void*, void*) { return ncclSuccess; };
  n.isend = [](void* comm, void*, size_t size, int tag, void*, void** request) {
    g_net.comm = comm;
    g_net.size = size;
    g_net.tag = tag;
    g_net.ptr = request;
    return ncclSuccess;
  };
  n.irecv = [](void* comm, int count, void**, size_t* sizes, int*, void**, void** request) {
    g_net.comm = comm;
    g_net.n = count;
    g_net.sizes[0] = sizes[0];
    g_net.ptr = request;
    return ncclSuccess;
  };
  n.iflush = [](void*, int, void**, int*, void**, void**) { return ncclSuccess; };
  n.test = [](void*, int*, int*) { return ncclSuccess; };
  n.closeSend = [](void*) { return ncclSuccess; };
  n.closeRecv = [](void*) { return ncclSuccess; };
  n.closeListen = [](void*) { return ncclSuccess; };
  n.getDeviceMr = [](void*, void*, void**) { return ncclSuccess; };
  n.irecvConsumed = [](void*, int, void*) { return ncclSuccess; };
  n.makeVDevice = [](int* d, ncclNetVDeviceProps_v9_t* props) { return MakeVDevice(g_net, d, props); };
  return n;
}

ncclCollNet_v9_t MakeCollNet() {
  ncclCollNet_v9_t c{};
  c.name = "FakeColl9";
  c.init = [](ncclDebugLogger_t logfn) {
    ++g_coll.initCalls;
    g_coll.logfn = logfn;
    return g_coll.initResult;
  };
  c.devices = [](int*) { return ncclSuccess; };
  c.getProperties = [](int, ncclNetProperties_v9_t* p) {
    *p = Props9();
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
  c.iallgather = [](void* comm, void*, int nParts, ncclNetSGE_v9_t* parts, size_t bytesPerRank, size_t offset,
                    size_t windowBytes, void*, void**) {
    g_coll.comm = comm;
    g_coll.n = nParts;
    g_coll.ptr = parts;
    g_coll.sizes[0] = bytesPerRank;
    g_coll.sizes[1] = offset;
    g_coll.sizes[2] = windowBytes;
    return ncclSuccess;
  };
  c.ireducescatter = [](void* comm, int nParts, ncclNetSGE_v9_t* parts, void*, size_t bytesPerRank, size_t offset,
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
  c.makeVDevice = [](int* d, ncclNetVDeviceProps_v9_t* props) { return MakeVDevice(g_coll, d, props); };
  return c;
}

void TestLogger(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}

void ExpectProps9(const ncclNetProperties_t& props) {
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

class PluginNetV9Microtest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_net = Rec{};
    g_coll = Rec{};
    g_dlsyms.clear();
    v9_ = MakeNet();
    coll9_ = MakeCollNet();
    g_exportNet = &v9_;
    g_exportCollNet = &coll9_;
    refCount[NET_INDEX] = refCount[COLLNET_INDEX] = 0;
  }
  void TearDown() override { refCount[NET_INDEX] = refCount[COLLNET_INDEX] = 0; }

  ncclNet_t* InitNet() {
    ncclNet_t* net = getNcclNet_v9(kLib);
    ncclNetCommConfig_t config{};
    void* ctx = nullptr;
    EXPECT_EQ(net->init(&ctx, 1, &config, TestLogger, nullptr), ncclSuccess);
    return net;
  }
  ncclCollNet_t* InitCollNet() {
    ncclCollNet_t* coll = getNcclCollNet_v9(kLib);
    void* ctx = nullptr;
    EXPECT_EQ(coll->init(&ctx, 1, TestLogger), ncclSuccess);
    return coll;
  }

  ncclNet_v9_t v9_;
  ncclCollNet_v9_t coll9_;
};

TEST_F(PluginNetV9Microtest, GettersNeedTheV9Symbols) {
  EXPECT_EQ(getNcclNet_v9(nullptr), nullptr);
  EXPECT_EQ(getNcclCollNet_v9(nullptr), nullptr);
  EXPECT_EQ(g_dlsyms, (std::vector<std::string>{"ncclNetPlugin_v9", "ncclCollNetPlugin_v9"}));

  ncclNet_t* net = getNcclNet_v9(kLib);
  ASSERT_NE(net, nullptr);
  EXPECT_STREQ(net->name, "Fake9");
  EXPECT_NE(net->init, nullptr);
  ncclCollNet_t* coll = getNcclCollNet_v9(kLib);
  ASSERT_NE(coll, nullptr);
  EXPECT_STREQ(coll->name, "FakeColl9");
}

TEST_F(PluginNetV9Microtest, InitRunsOnceWhileAnyCommHoldsIt) {
  ncclNet_t* net = getNcclNet_v9(kLib);
  ncclNetCommConfig_t config{};
  void* ctx = nullptr;
  g_net.initResult = ncclSystemError;
  EXPECT_EQ(net->init(&ctx, 1, &config, TestLogger, nullptr), ncclSystemError);
  g_net.initResult = ncclSuccess;
  InitNet();
  InitNet();
  EXPECT_EQ(g_net.initCalls, 2);
  EXPECT_EQ(g_net.logfn, &TestLogger);
  EXPECT_EQ(net->finalize(nullptr), ncclSuccess);
  EXPECT_EQ(net->finalize(nullptr), ncclSuccess);
  InitNet();
  EXPECT_EQ(g_net.initCalls, 3);

  ncclCollNet_t* coll = getNcclCollNet_v9(kLib);
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

TEST_F(PluginNetV9Microtest, NetInitWiresPassThroughs) {
  ncclNet_t* net = InitNet();
  EXPECT_EQ(net->devices, v9_.devices);
  EXPECT_EQ(net->accept, v9_.accept);
  EXPECT_EQ(net->regMr, v9_.regMr);
  EXPECT_EQ(net->regMrDmaBuf, v9_.regMrDmaBuf);
  EXPECT_EQ(net->deregMr, v9_.deregMr);
  EXPECT_EQ(net->iflush, v9_.iflush);
  EXPECT_EQ(net->test, v9_.test);
  EXPECT_EQ(net->closeSend, v9_.closeSend);
  EXPECT_EQ(net->closeRecv, v9_.closeRecv);
  EXPECT_EQ(net->closeListen, v9_.closeListen);
  EXPECT_EQ(net->getDeviceMr, v9_.getDeviceMr);
  EXPECT_EQ(net->irecvConsumed, v9_.irecvConsumed);
  EXPECT_EQ(net->setNetAttr, nullptr);
}

TEST_F(PluginNetV9Microtest, MakeVDeviceIsWrappedOnlyWhenThePluginHasOne) {
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
  v9_.makeVDevice = nullptr;
  coll9_.makeVDevice = nullptr;
  EXPECT_EQ(InitNet()->makeVDevice, nullptr);
  EXPECT_EQ(InitCollNet()->makeVDevice, nullptr);
}

TEST_F(PluginNetV9Microtest, GetPropertiesCopiesEveryV9Field) {
  ncclNetProperties_t props;
  std::memset(&props, 0xff, sizeof(props));
  ASSERT_EQ(InitNet()->getProperties(0, &props), ncclSuccess);
  ExpectProps9(props);

  std::memset(&props, 0xff, sizeof(props));
  ASSERT_EQ(InitCollNet()->getProperties(0, &props), ncclSuccess);
  ExpectProps9(props);

  g_net.propsResult = g_coll.propsResult = ncclSystemError;
  EXPECT_EQ(InitNet()->getProperties(0, &props), ncclSystemError);
  EXPECT_EQ(InitCollNet()->getProperties(0, &props), ncclSystemError);
}

TEST_F(PluginNetV9Microtest, NetCallsDropTheContextAndProfilerHandles) {
  ncclNet_t* net = InitNet();
  char handle[8], comm;
  void* out = nullptr;
  ASSERT_EQ(net->listen(&g_coll, 2, handle, &out), ncclSuccess);
  EXPECT_EQ(g_net.dev, 2);
  EXPECT_EQ(g_net.handle, handle);
  EXPECT_EQ(out, &g_net);

  ncclNetDeviceHandle_t* devComm = nullptr;
  ASSERT_EQ(net->connect(&g_coll, 4, handle, &out, &devComm), ncclSuccess);
  EXPECT_EQ(g_net.dev, 4);
  EXPECT_EQ(g_net.ptr, &devComm);

  void* request = nullptr;
  ASSERT_EQ(net->isend(&comm, nullptr, size_t(3) << 31, 9, nullptr, &g_coll, &request), ncclSuccess);
  EXPECT_EQ(g_net.comm, &comm);
  EXPECT_EQ(g_net.size, size_t(3) << 31);
  EXPECT_EQ(g_net.tag, 9);
  EXPECT_EQ(g_net.ptr, &request);

  size_t sizes[1] = {size_t(5) << 31};
  ASSERT_EQ(net->irecv(&comm, 1, nullptr, sizes, nullptr, nullptr, nullptr, &request), ncclSuccess);
  EXPECT_EQ(g_net.n, 1);
  EXPECT_EQ(g_net.sizes[0], size_t(5) << 31);
  EXPECT_EQ(g_net.ptr, &request);
}

TEST_F(PluginNetV9Microtest, CollNetInitWiresPassThroughs) {
  ncclCollNet_t* coll = InitCollNet();
  EXPECT_EQ(coll->devices, coll9_.devices);
  EXPECT_EQ(coll->connect, coll9_.connect);
  EXPECT_EQ(coll->reduceSupport, coll9_.reduceSupport);
  EXPECT_EQ(coll->regMr, coll9_.regMr);
  EXPECT_EQ(coll->regMrDmaBuf, coll9_.regMrDmaBuf);
  EXPECT_EQ(coll->deregMr, coll9_.deregMr);
  EXPECT_EQ(coll->iallreduce, coll9_.iallreduce);
  EXPECT_EQ(coll->iflush, coll9_.iflush);
  EXPECT_EQ(coll->test, coll9_.test);
  EXPECT_EQ(coll->closeColl, coll9_.closeColl);
  EXPECT_EQ(coll->closeListen, coll9_.closeListen);
}

TEST_F(PluginNetV9Microtest, CollNetCallsForwardEverything) {
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
