/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/net/net_v11.cc, the adapter that presents a v11 net/collnet plugin as the
// current ncclNet_t / ncclCollNet_t: per-comm init and finalize, and wrappers for the optional entry points.

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "nccl_net.h"
#include "os.h"

namespace {

void* const kLib = reinterpret_cast<void*>(0x11);
ncclNet_v11_t* g_exportNet = nullptr;
ncclCollNet_v11_t* g_exportCollNet = nullptr;
std::vector<std::string> g_dlsyms;

void* FakeDlsym(void* lib, const char* symbol) {
  g_dlsyms.push_back(symbol);
  if (lib != kLib) return nullptr;
  if (!std::strcmp(symbol, "ncclNetPlugin_v11")) return g_exportNet;
  if (!std::strcmp(symbol, "ncclCollNetPlugin_v11")) return g_exportCollNet;
  return nullptr;
}

}  // namespace

#define ncclOsDlsym FakeDlsym
#include NET_V11_CC_PATH
#undef ncclOsDlsym

namespace {

// What the v11 plugin was last called with.
struct Rec {
  int initCalls = 0;
  ncclResult_t initResult = ncclSuccess;
  void** ctx = nullptr;
  uint64_t commId = 0;
  void* config = nullptr;
  ncclDebugLogger_t logfn = nullptr;
  ncclProfilerCallback_t proffn = nullptr;
  ncclResult_t propsResult = ncclSuccess;
  int n = 0;
  void* comm = nullptr;
  void* ptr = nullptr;
  size_t sizes[3] = {};
};
Rec g_net;
Rec g_coll;

ncclNetProperties_v11_t Props11() {
  ncclNetProperties_v11_t p{};
  p.name = const_cast<char*>("ionic_0");
  p.pciPath = const_cast<char*>("/sys/pci2");
  p.guid = 0x11;
  p.ptrSupport = NCCL_PTR_CUDA;
  p.regIsGlobal = 1;
  p.forceFlush = 1;
  p.speed = 800000;
  p.port = 3;
  p.latency = 0.5f;
  p.maxComms = 16;
  p.maxRecvs = 2;
  p.netDeviceType = NCCL_NET_DEVICE_UNPACK;
  p.netDeviceVersion = 7;
  p.vProps.ndevs = 1;
  p.vProps.devs[0] = 9;
  p.maxP2pBytes = size_t(1) << 41;
  p.maxCollBytes = size_t(1) << 34;
  p.maxMultiRequestSize = 6;
  return p;
}

ncclResult_t MakeVDevice(Rec& rec, int* d, ncclNetVDeviceProps_v11_t* props) {
  rec.ptr = props;
  rec.n = props->ndevs;
  *d = 42;
  return ncclRemoteError;
}

ncclNet_v11_t MakeNet() {
  ncclNet_v11_t n{};
  n.name = "Fake11";
  n.init = [](void** ctx, uint64_t commId, ncclNetCommConfig_v11_t* config, ncclDebugLogger_t logfn,
              ncclProfilerCallback_t proffn) {
    ++g_net.initCalls;
    g_net.ctx = ctx;
    g_net.commId = commId;
    g_net.config = config;
    g_net.logfn = logfn;
    g_net.proffn = proffn;
    return g_net.initResult;
  };
  n.devices = [](int*) { return ncclSuccess; };
  n.getProperties = [](int, ncclNetProperties_v11_t* p) {
    *p = Props11();
    return g_net.propsResult;
  };
  n.listen = [](void*, int, void*, void**) { return ncclSuccess; };
  n.connect = [](void*, int, void*, void**, ncclNetDeviceHandle_v11_t**) { return ncclSuccess; };
  n.accept = [](void*, void**, ncclNetDeviceHandle_v11_t**) { return ncclSuccess; };
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
  n.makeVDevice = [](int* d, ncclNetVDeviceProps_v11_t* props) { return MakeVDevice(g_net, d, props); };
  n.finalize = [](void*) { return ncclSuccess; };
  n.setNetAttr = [](void* ctx, ncclNetAttr_v11_t* attr) {
    g_net.ptr = attr;
    g_net.n = int(attr->op);
    g_net.ctx = static_cast<void**>(ctx);
    return ncclInvalidUsage;
  };
  return n;
}

ncclCollNet_v11_t MakeCollNet() {
  ncclCollNet_v11_t c{};
  c.name = "FakeColl11";
  c.init = [](void** ctx, uint64_t commId, ncclDebugLogger_t logfn) {
    ++g_coll.initCalls;
    g_coll.ctx = ctx;
    g_coll.commId = commId;
    g_coll.logfn = logfn;
    return g_coll.initResult;
  };
  c.devices = [](int*) { return ncclSuccess; };
  c.getProperties = [](int, ncclNetProperties_v11_t* p) {
    *p = Props11();
    return g_coll.propsResult;
  };
  c.listen = [](void*, int, void*, void**) { return ncclSuccess; };
  c.connect = [](void**, int, int, void*, void**) { return ncclSuccess; };
  c.reduceSupport = [](ncclDataType_t, ncclRedOp_t, int*) { return ncclSuccess; };
  c.regMr = [](void*, void*, size_t, int, void**) { return ncclSuccess; };
  c.regMrDmaBuf = [](void*, void*, size_t, int, uint64_t, int, void**) { return ncclSuccess; };
  c.deregMr = [](void*, void*) { return ncclSuccess; };
  c.iallreduce = [](void*, void*, void*, size_t, ncclDataType_t, ncclRedOp_t, void*, void*, void**) {
    return ncclSuccess;
  };
  c.iallgather = [](void* comm, void*, int nParts, ncclNetSGE_v11_t* parts, size_t bytesPerRank, size_t offset,
                    size_t windowBytes, void*, void**) {
    g_coll.comm = comm;
    g_coll.n = nParts;
    g_coll.ptr = parts;
    g_coll.sizes[0] = bytesPerRank;
    g_coll.sizes[1] = offset;
    g_coll.sizes[2] = windowBytes;
    return ncclSuccess;
  };
  c.ireducescatter = [](void* comm, int nParts, ncclNetSGE_v11_t* parts, void*, size_t bytesPerRank, size_t offset,
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
  c.makeVDevice = [](int* d, ncclNetVDeviceProps_v11_t* props) { return MakeVDevice(g_coll, d, props); };
  c.finalize = [](void*) { return ncclSuccess; };
  return c;
}

void TestLogger(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}
ncclResult_t ProfilerCallback(void**, int, void*, int64_t, void*) { return ncclSuccess; }

void ExpectProps11(const ncclNetProperties_t& props) {
  EXPECT_STREQ(props.name, "ionic_0");
  EXPECT_STREQ(props.pciPath, "/sys/pci2");
  EXPECT_EQ(props.guid, 0x11u);
  EXPECT_EQ(props.ptrSupport, NCCL_PTR_CUDA);
  EXPECT_EQ(props.regIsGlobal, 1);
  EXPECT_EQ(props.forceFlush, 1);
  EXPECT_EQ(props.speed, 800000);
  EXPECT_EQ(props.port, 3);
  EXPECT_FLOAT_EQ(props.latency, 0.5f);
  EXPECT_EQ(props.maxComms, 16);
  EXPECT_EQ(props.maxRecvs, 2);
  EXPECT_EQ(props.netDeviceType, NCCL_NET_DEVICE_UNPACK);
  EXPECT_EQ(props.netDeviceVersion, 7);
  EXPECT_EQ(props.vProps.ndevs, 1);
  EXPECT_EQ(props.vProps.devs[0], 9);
  EXPECT_EQ(props.vProps.devs[1], -1);  // only ndevs entries are copied
  EXPECT_EQ(props.maxP2pBytes, size_t(1) << 41);
  EXPECT_EQ(props.maxCollBytes, size_t(1) << 34);
  EXPECT_EQ(props.maxMultiRequestSize, 6);
  EXPECT_EQ(props.railId, NCCL_NET_ID_UNDEF);
  EXPECT_EQ(props.planeId, NCCL_NET_ID_UNDEF);
}

class PluginNetV11Microtest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_net = Rec{};
    g_coll = Rec{};
    g_dlsyms.clear();
    v11_ = MakeNet();
    coll11_ = MakeCollNet();
    g_exportNet = &v11_;
    g_exportCollNet = &coll11_;
  }

  ncclNet_t* InitNet() {
    ncclNet_t* net = getNcclNet_v11(kLib);
    EXPECT_EQ(net->init(&ctx_, 77, &config_, TestLogger, ProfilerCallback), ncclSuccess);
    return net;
  }
  ncclCollNet_t* InitCollNet() {
    ncclCollNet_t* coll = getNcclCollNet_v11(kLib);
    EXPECT_EQ(coll->init(&ctx_, 78, TestLogger), ncclSuccess);
    return coll;
  }

  ncclNet_v11_t v11_;
  ncclCollNet_v11_t coll11_;
  ncclNetCommConfig_t config_{4};
  void* ctx_ = nullptr;
};

TEST_F(PluginNetV11Microtest, GettersNeedTheV11Symbols) {
  EXPECT_EQ(getNcclNet_v11(nullptr), nullptr);
  EXPECT_EQ(getNcclCollNet_v11(nullptr), nullptr);
  EXPECT_EQ(g_dlsyms, (std::vector<std::string>{"ncclNetPlugin_v11", "ncclCollNetPlugin_v11"}));

  ncclNet_t* net = getNcclNet_v11(kLib);
  ASSERT_NE(net, nullptr);
  EXPECT_STREQ(net->name, "Fake11");
  EXPECT_NE(net->init, nullptr);
  ncclCollNet_t* coll = getNcclCollNet_v11(kLib);
  ASSERT_NE(coll, nullptr);
  EXPECT_STREQ(coll->name, "FakeColl11");
}

// v11 plugins keep per-comm state, so every comm's init reaches the plugin with that comm's context and config.
TEST_F(PluginNetV11Microtest, EveryInitReachesThePlugin) {
  InitNet();
  InitNet();
  EXPECT_EQ(g_net.initCalls, 2);
  EXPECT_EQ(g_net.ctx, &ctx_);
  EXPECT_EQ(g_net.commId, 77u);
  EXPECT_EQ(g_net.config, &config_);
  EXPECT_EQ(g_net.logfn, &TestLogger);
  EXPECT_EQ(g_net.proffn, &ProfilerCallback);

  InitCollNet();
  InitCollNet();
  EXPECT_EQ(g_coll.initCalls, 2);
  EXPECT_EQ(g_coll.ctx, &ctx_);
  EXPECT_EQ(g_coll.commId, 78u);
  EXPECT_EQ(g_coll.logfn, &TestLogger);

  g_net.initResult = g_coll.initResult = ncclSystemError;
  EXPECT_EQ(getNcclNet_v11(kLib)->init(&ctx_, 1, &config_, TestLogger, nullptr), ncclSystemError);
  EXPECT_EQ(getNcclCollNet_v11(kLib)->init(&ctx_, 1, TestLogger), ncclSystemError);
}

TEST_F(PluginNetV11Microtest, NetInitWiresPassThroughs) {
  ncclNet_t* net = InitNet();
  EXPECT_EQ(net->devices, v11_.devices);
  EXPECT_EQ(net->listen, v11_.listen);
  EXPECT_EQ(net->connect, v11_.connect);
  EXPECT_EQ(net->accept, v11_.accept);
  EXPECT_EQ(net->regMr, v11_.regMr);
  EXPECT_EQ(net->regMrDmaBuf, v11_.regMrDmaBuf);
  EXPECT_EQ(net->deregMr, v11_.deregMr);
  EXPECT_EQ(net->isend, v11_.isend);
  EXPECT_EQ(net->irecv, v11_.irecv);
  EXPECT_EQ(net->iflush, v11_.iflush);
  EXPECT_EQ(net->test, v11_.test);
  EXPECT_EQ(net->closeSend, v11_.closeSend);
  EXPECT_EQ(net->closeRecv, v11_.closeRecv);
  EXPECT_EQ(net->closeListen, v11_.closeListen);
  EXPECT_EQ(net->getDeviceMr, v11_.getDeviceMr);
  EXPECT_EQ(net->irecvConsumed, v11_.irecvConsumed);
  EXPECT_EQ(net->finalize, v11_.finalize);
}

TEST_F(PluginNetV11Microtest, OptionalEntryPointsAreWrappedOnlyWhenPresent) {
  ncclNet_t* net = InitNet();
  ncclNetVDeviceProps_t vprops{};
  vprops.ndevs = 2;
  int d = -1;
  ASSERT_NE(net->makeVDevice, nullptr);
  EXPECT_EQ(net->makeVDevice(&d, &vprops), ncclRemoteError);
  EXPECT_EQ(d, 42);
  EXPECT_EQ(g_net.ptr, &vprops);
  EXPECT_EQ(g_net.n, 2);

  ncclNetAttr_t attr = NCCL_NET_ATTR_INIT;
  attr.op = 3;
  ASSERT_NE(net->setNetAttr, nullptr);
  EXPECT_EQ(net->setNetAttr(&ctx_, &attr), ncclInvalidUsage);
  EXPECT_EQ(g_net.ptr, &attr);
  EXPECT_EQ(g_net.n, 3);
  EXPECT_EQ(g_net.ctx, &ctx_);

  ncclCollNet_t* coll = InitCollNet();
  ASSERT_NE(coll->makeVDevice, nullptr);
  EXPECT_EQ(coll->makeVDevice(&d, &vprops), ncclRemoteError);
  EXPECT_EQ(g_coll.ptr, &vprops);

  v11_.makeVDevice = nullptr;
  v11_.setNetAttr = nullptr;
  coll11_.makeVDevice = nullptr;
  net = InitNet();
  EXPECT_EQ(net->makeVDevice, nullptr);
  EXPECT_EQ(net->setNetAttr, nullptr);
  EXPECT_EQ(InitCollNet()->makeVDevice, nullptr);
}

TEST_F(PluginNetV11Microtest, GetPropertiesCopiesEveryV11Field) {
  ncclNetProperties_t props;
  std::memset(&props, 0xff, sizeof(props));
  ASSERT_EQ(InitNet()->getProperties(0, &props), ncclSuccess);
  ExpectProps11(props);

  std::memset(&props, 0xff, sizeof(props));
  ASSERT_EQ(InitCollNet()->getProperties(0, &props), ncclSuccess);
  ExpectProps11(props);

  g_net.propsResult = g_coll.propsResult = ncclSystemError;
  EXPECT_EQ(InitNet()->getProperties(0, &props), ncclSystemError);
  EXPECT_EQ(InitCollNet()->getProperties(0, &props), ncclSystemError);
}

TEST_F(PluginNetV11Microtest, CollNetInitWiresPassThroughs) {
  ncclCollNet_t* coll = InitCollNet();
  EXPECT_EQ(coll->devices, coll11_.devices);
  EXPECT_EQ(coll->listen, coll11_.listen);
  EXPECT_EQ(coll->connect, coll11_.connect);
  EXPECT_EQ(coll->reduceSupport, coll11_.reduceSupport);
  EXPECT_EQ(coll->regMr, coll11_.regMr);
  EXPECT_EQ(coll->regMrDmaBuf, coll11_.regMrDmaBuf);
  EXPECT_EQ(coll->deregMr, coll11_.deregMr);
  EXPECT_EQ(coll->iallreduce, coll11_.iallreduce);
  EXPECT_EQ(coll->iflush, coll11_.iflush);
  EXPECT_EQ(coll->test, coll11_.test);
  EXPECT_EQ(coll->closeColl, coll11_.closeColl);
  EXPECT_EQ(coll->closeListen, coll11_.closeListen);
  EXPECT_EQ(coll->finalize, coll11_.finalize);
}

TEST_F(PluginNetV11Microtest, CollNetSgeCallsForwardEverything) {
  ncclCollNet_t* coll = InitCollNet();
  char comm;
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
