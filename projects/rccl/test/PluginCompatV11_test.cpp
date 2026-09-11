/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Regression test for AICOMRCCL-1532: the v11 network-plugin compat layer must
// initialize lazily. getNcclNet_v11() must populate only ncclNet.name/.init;
// the remaining entry points must stay null until ncclNet.init() runs (the old
// bug wired them up eagerly, exposing an uninitialized plugin).
//
// Uses a purely in-process mock: it exports the ncclNetPlugin_v11 /
// ncclCollNetPlugin_v11 symbols the compat layer resolves via dlsym, hands it a
// dlopen(NULL) handle, and checks that only ->init is set before init() and the
// rest become non-null after. No GPU or real plugin required.

#include <gtest/gtest.h>

#include <dlfcn.h>

#include "nccl.h"
#include "nccl_net.h"

// Compat-layer entry points from librccl.so (net_v11.cc); global-namespace C++
// functions, so this prototype matches the (Debug) library's mangled symbol.
ncclNet_t* getNcclNet_v11(void* lib);
ncclCollNet_t* getNcclCollNet_v11(void* lib);

namespace RcclUnitTesting {
namespace {

// Mock v11 net plugin: each entry point is a distinct non-null stub so we can
// tell "wired up" from "still null"; init() just returns success.
ncclResult_t mockNetInit(void** ctx, uint64_t, ncclNetCommConfig_v11_t*,
                         ncclDebugLogger_t, ncclProfilerCallback_t) {
  if (ctx) { *ctx = nullptr; }
  return ncclSuccess;
}
ncclResult_t mockNetDevices(int* ndev) { if (ndev) { *ndev = 1; } return ncclSuccess; }
ncclResult_t mockNetGetProperties(int, ncclNetProperties_v11_t*) { return ncclSuccess; }
ncclResult_t mockNetListen(void*, int, void*, void**) { return ncclSuccess; }
ncclResult_t mockNetConnect(void*, int, void*, void**, ncclNetDeviceHandle_v11_t**) { return ncclSuccess; }
ncclResult_t mockNetAccept(void*, void**, ncclNetDeviceHandle_v11_t**) { return ncclSuccess; }
ncclResult_t mockNetRegMr(void*, void*, size_t, int, void**) { return ncclSuccess; }
ncclResult_t mockNetRegMrDmaBuf(void*, void*, size_t, int, uint64_t, int, void**) { return ncclSuccess; }
ncclResult_t mockNetDeregMr(void*, void*) { return ncclSuccess; }
ncclResult_t mockNetIsend(void*, void*, size_t, int, void*, void*, void**) { return ncclSuccess; }
ncclResult_t mockNetIrecv(void*, int, void**, size_t*, int*, void**, void**, void**) { return ncclSuccess; }
ncclResult_t mockNetIflush(void*, int, void**, int*, void**, void**) { return ncclSuccess; }
ncclResult_t mockNetTest(void*, int*, int*) { return ncclSuccess; }
ncclResult_t mockNetCloseSend(void*) { return ncclSuccess; }
ncclResult_t mockNetCloseRecv(void*) { return ncclSuccess; }
ncclResult_t mockNetCloseListen(void*) { return ncclSuccess; }
ncclResult_t mockNetGetDeviceMr(void*, void*, void**) { return ncclSuccess; }
ncclResult_t mockNetIrecvConsumed(void*, int, void*) { return ncclSuccess; }
ncclResult_t mockNetMakeVDevice(int*, ncclNetVDeviceProps_v11_t*) { return ncclSuccess; }
ncclResult_t mockNetFinalize(void*) { return ncclSuccess; }
ncclResult_t mockNetSetNetAttr(void*, ncclNetAttr_v11_t*) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// Mock v11 collnet plugin (same lazy-init contract as the net plugin).
// ---------------------------------------------------------------------------
ncclResult_t mockCollInit(void** ctx, uint64_t, ncclDebugLogger_t) {
  if (ctx) { *ctx = nullptr; }
  return ncclSuccess;
}
ncclResult_t mockCollDevices(int* ndev) { if (ndev) { *ndev = 1; } return ncclSuccess; }
ncclResult_t mockCollGetProperties(int, ncclNetProperties_v11_t*) { return ncclSuccess; }
ncclResult_t mockCollListen(void*, int, void*, void**) { return ncclSuccess; }
ncclResult_t mockCollConnect(void**, int, int, void*, void**) { return ncclSuccess; }
ncclResult_t mockCollReduceSupport(ncclDataType_t, ncclRedOp_t, int*) { return ncclSuccess; }
ncclResult_t mockCollRegMr(void*, void*, size_t, int, void**) { return ncclSuccess; }
ncclResult_t mockCollRegMrDmaBuf(void*, void*, size_t, int, uint64_t, int, void**) { return ncclSuccess; }
ncclResult_t mockCollDeregMr(void*, void*) { return ncclSuccess; }
ncclResult_t mockCollIallreduce(void*, void*, void*, size_t, ncclDataType_t, ncclRedOp_t, void*, void*, void**) { return ncclSuccess; }
ncclResult_t mockCollIallgather(void*, void*, int, ncclNetSGE_v11_t*, size_t, size_t, size_t, void*, void**) { return ncclSuccess; }
ncclResult_t mockCollIreducescatter(void*, int, ncclNetSGE_v11_t*, void*, size_t, size_t, size_t, ncclDataType_t, ncclRedOp_t, void*, void**) { return ncclSuccess; }
ncclResult_t mockCollIflush(void*, void*, int, void*, void**) { return ncclSuccess; }
ncclResult_t mockCollTest(void*, int*, int*) { return ncclSuccess; }
ncclResult_t mockCollCloseColl(void*) { return ncclSuccess; }
ncclResult_t mockCollCloseListen(void*) { return ncclSuccess; }
ncclResult_t mockCollMakeVDevice(int*, ncclNetVDeviceProps_v11_t*) { return ncclSuccess; }
ncclResult_t mockCollFinalize(void*) { return ncclSuccess; }

} // namespace
} // namespace RcclUnitTesting

// These symbols are what getNcclNet_v11()/getNcclCollNet_v11() look up via
// dlsym(). The CMake target exports them (--export-dynamic-symbol) so they land
// in the test executable's dynamic symbol table; they must be named exactly, so
// we place them at global scope with C linkage and default visibility.
extern "C" {

__attribute__((visibility("default")))
ncclNet_v11_t ncclNetPlugin_v11 = {
  "mock_v11_net",                        // name
  RcclUnitTesting::mockNetInit,          // init
  RcclUnitTesting::mockNetDevices,       // devices
  RcclUnitTesting::mockNetGetProperties, // getProperties
  RcclUnitTesting::mockNetListen,        // listen
  RcclUnitTesting::mockNetConnect,       // connect
  RcclUnitTesting::mockNetAccept,        // accept
  RcclUnitTesting::mockNetRegMr,         // regMr
  RcclUnitTesting::mockNetRegMrDmaBuf,   // regMrDmaBuf
  RcclUnitTesting::mockNetDeregMr,       // deregMr
  RcclUnitTesting::mockNetIsend,         // isend
  RcclUnitTesting::mockNetIrecv,         // irecv
  RcclUnitTesting::mockNetIflush,        // iflush
  RcclUnitTesting::mockNetTest,          // test
  RcclUnitTesting::mockNetCloseSend,     // closeSend
  RcclUnitTesting::mockNetCloseRecv,     // closeRecv
  RcclUnitTesting::mockNetCloseListen,   // closeListen
  RcclUnitTesting::mockNetGetDeviceMr,   // getDeviceMr
  RcclUnitTesting::mockNetIrecvConsumed, // irecvConsumed
  RcclUnitTesting::mockNetMakeVDevice,   // makeVDevice
  RcclUnitTesting::mockNetFinalize,      // finalize
  RcclUnitTesting::mockNetSetNetAttr,    // setNetAttr
};

__attribute__((visibility("default")))
ncclCollNet_v11_t ncclCollNetPlugin_v11 = {
  "mock_v11_collnet",                       // name
  RcclUnitTesting::mockCollInit,            // init
  RcclUnitTesting::mockCollDevices,         // devices
  RcclUnitTesting::mockCollGetProperties,   // getProperties
  RcclUnitTesting::mockCollListen,          // listen
  RcclUnitTesting::mockCollConnect,         // connect
  RcclUnitTesting::mockCollReduceSupport,   // reduceSupport
  RcclUnitTesting::mockCollRegMr,           // regMr
  RcclUnitTesting::mockCollRegMrDmaBuf,     // regMrDmaBuf
  RcclUnitTesting::mockCollDeregMr,         // deregMr
  RcclUnitTesting::mockCollIallreduce,      // iallreduce
  RcclUnitTesting::mockCollIallgather,      // iallgather
  RcclUnitTesting::mockCollIreducescatter,  // ireducescatter
  RcclUnitTesting::mockCollIflush,          // iflush
  RcclUnitTesting::mockCollTest,            // test
  RcclUnitTesting::mockCollCloseColl,       // closeColl
  RcclUnitTesting::mockCollCloseListen,     // closeListen
  RcclUnitTesting::mockCollMakeVDevice,     // makeVDevice
  RcclUnitTesting::mockCollFinalize,        // finalize
};

} // extern "C"

namespace RcclUnitTesting {

// getNcclNet_v11() must hand back a real vtable with ONLY ->init populated;
// every other entry point stays null until init() runs, and becomes non-null
// afterwards. This is the core AICOMRCCL-1532 contract - the pre-fix code
// populated the whole vtable eagerly and would fail the "before init()" block.
TEST(PluginCompatV11, NetLazyInitialization) {
  void* self = dlopen(nullptr, RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(self, nullptr) << "dlopen(NULL) failed: " << dlerror();

  ncclNet_t* net = getNcclNet_v11(self);
  ASSERT_NE(net, nullptr)
    << "getNcclNet_v11 returned null; the in-process ncclNetPlugin_v11 symbol "
       "was not resolved (is the test linked with -rdynamic?)";

  // init() is the single entry point the compat layer is allowed to populate up
  // front, because it is the callback that performs the lazy initialization.
  EXPECT_NE(net->init, nullptr) << "compat layer must set ncclNet.init";

  // Everything else must still be null: the plugin has not been init()'d, so no
  // per-device entry point may be exposed yet.
  EXPECT_EQ(net->devices, nullptr)       << "devices set before init()";
  EXPECT_EQ(net->getProperties, nullptr) << "getProperties set before init()";
  EXPECT_EQ(net->listen, nullptr)        << "listen set before init()";
  EXPECT_EQ(net->connect, nullptr)       << "connect set before init()";
  EXPECT_EQ(net->accept, nullptr)        << "accept set before init()";
  EXPECT_EQ(net->regMr, nullptr)         << "regMr set before init()";
  EXPECT_EQ(net->regMrDmaBuf, nullptr)   << "regMrDmaBuf set before init()";
  EXPECT_EQ(net->deregMr, nullptr)       << "deregMr set before init()";
  EXPECT_EQ(net->isend, nullptr)         << "isend set before init()";
  EXPECT_EQ(net->irecv, nullptr)         << "irecv set before init()";
  EXPECT_EQ(net->iflush, nullptr)        << "iflush set before init()";
  EXPECT_EQ(net->test, nullptr)          << "test set before init()";
  EXPECT_EQ(net->closeSend, nullptr)     << "closeSend set before init()";
  EXPECT_EQ(net->closeRecv, nullptr)     << "closeRecv set before init()";
  EXPECT_EQ(net->closeListen, nullptr)   << "closeListen set before init()";
  EXPECT_EQ(net->getDeviceMr, nullptr)   << "getDeviceMr set before init()";
  EXPECT_EQ(net->irecvConsumed, nullptr) << "irecvConsumed set before init()";
  EXPECT_EQ(net->makeVDevice, nullptr)   << "makeVDevice set before init()";
  EXPECT_EQ(net->finalize, nullptr)      << "finalize set before init()";
  EXPECT_EQ(net->setNetAttr, nullptr)    << "setNetAttr set before init()";

  // Run the lazy init. The compat layer now wires up the remaining entry points
  // from the (mock) v11 plugin struct.
  void* ctx = nullptr;
  ASSERT_EQ(net->init(&ctx, /*commId=*/0, /*config=*/nullptr,
                      /*logFunction=*/nullptr, /*profFunction=*/nullptr),
            ncclSuccess);

  EXPECT_NE(net->devices, nullptr)       << "devices null after init()";
  EXPECT_NE(net->getProperties, nullptr) << "getProperties null after init()";
  EXPECT_NE(net->listen, nullptr)        << "listen null after init()";
  EXPECT_NE(net->connect, nullptr)       << "connect null after init()";
  EXPECT_NE(net->accept, nullptr)        << "accept null after init()";
  EXPECT_NE(net->regMr, nullptr)         << "regMr null after init()";
  EXPECT_NE(net->regMrDmaBuf, nullptr)   << "regMrDmaBuf null after init()";
  EXPECT_NE(net->deregMr, nullptr)       << "deregMr null after init()";
  EXPECT_NE(net->isend, nullptr)         << "isend null after init()";
  EXPECT_NE(net->irecv, nullptr)         << "irecv null after init()";
  EXPECT_NE(net->iflush, nullptr)        << "iflush null after init()";
  EXPECT_NE(net->test, nullptr)          << "test null after init()";
  EXPECT_NE(net->closeSend, nullptr)     << "closeSend null after init()";
  EXPECT_NE(net->closeRecv, nullptr)     << "closeRecv null after init()";
  EXPECT_NE(net->closeListen, nullptr)   << "closeListen null after init()";
  EXPECT_NE(net->getDeviceMr, nullptr)   << "getDeviceMr null after init()";
  EXPECT_NE(net->irecvConsumed, nullptr) << "irecvConsumed null after init()";
  EXPECT_NE(net->makeVDevice, nullptr)   << "makeVDevice null after init()";
  EXPECT_NE(net->finalize, nullptr)      << "finalize null after init()";
  EXPECT_NE(net->setNetAttr, nullptr)    << "setNetAttr null after init()";

  dlclose(self);
}

// Same lazy-init contract for the collnet compat layer.
TEST(PluginCompatV11, CollNetLazyInitialization) {
  void* self = dlopen(nullptr, RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(self, nullptr) << "dlopen(NULL) failed: " << dlerror();

  ncclCollNet_t* coll = getNcclCollNet_v11(self);
  ASSERT_NE(coll, nullptr)
    << "getNcclCollNet_v11 returned null; the in-process ncclCollNetPlugin_v11 "
       "symbol was not resolved (is the test linked with -rdynamic?)";

  EXPECT_NE(coll->init, nullptr) << "compat layer must set ncclCollNet.init";

  EXPECT_EQ(coll->devices, nullptr)        << "devices set before init()";
  EXPECT_EQ(coll->getProperties, nullptr)  << "getProperties set before init()";
  EXPECT_EQ(coll->listen, nullptr)         << "listen set before init()";
  EXPECT_EQ(coll->connect, nullptr)        << "connect set before init()";
  EXPECT_EQ(coll->reduceSupport, nullptr)  << "reduceSupport set before init()";
  EXPECT_EQ(coll->regMr, nullptr)          << "regMr set before init()";
  EXPECT_EQ(coll->regMrDmaBuf, nullptr)    << "regMrDmaBuf set before init()";
  EXPECT_EQ(coll->deregMr, nullptr)        << "deregMr set before init()";
  EXPECT_EQ(coll->iallreduce, nullptr)     << "iallreduce set before init()";
  EXPECT_EQ(coll->iallgather, nullptr)     << "iallgather set before init()";
  EXPECT_EQ(coll->ireducescatter, nullptr) << "ireducescatter set before init()";
  EXPECT_EQ(coll->iflush, nullptr)         << "iflush set before init()";
  EXPECT_EQ(coll->test, nullptr)           << "test set before init()";
  EXPECT_EQ(coll->closeColl, nullptr)      << "closeColl set before init()";
  EXPECT_EQ(coll->closeListen, nullptr)    << "closeListen set before init()";
  EXPECT_EQ(coll->makeVDevice, nullptr)    << "makeVDevice set before init()";
  EXPECT_EQ(coll->finalize, nullptr)       << "finalize set before init()";

  void* ctx = nullptr;
  ASSERT_EQ(coll->init(&ctx, /*commId=*/0, /*logFunction=*/nullptr), ncclSuccess);

  EXPECT_NE(coll->devices, nullptr)        << "devices null after init()";
  EXPECT_NE(coll->getProperties, nullptr)  << "getProperties null after init()";
  EXPECT_NE(coll->listen, nullptr)         << "listen null after init()";
  EXPECT_NE(coll->connect, nullptr)        << "connect null after init()";
  EXPECT_NE(coll->reduceSupport, nullptr)  << "reduceSupport null after init()";
  EXPECT_NE(coll->regMr, nullptr)          << "regMr null after init()";
  EXPECT_NE(coll->regMrDmaBuf, nullptr)    << "regMrDmaBuf null after init()";
  EXPECT_NE(coll->deregMr, nullptr)        << "deregMr null after init()";
  EXPECT_NE(coll->iallreduce, nullptr)     << "iallreduce null after init()";
  EXPECT_NE(coll->iallgather, nullptr)     << "iallgather null after init()";
  EXPECT_NE(coll->ireducescatter, nullptr) << "ireducescatter null after init()";
  EXPECT_NE(coll->iflush, nullptr)         << "iflush null after init()";
  EXPECT_NE(coll->test, nullptr)           << "test null after init()";
  EXPECT_NE(coll->closeColl, nullptr)      << "closeColl null after init()";
  EXPECT_NE(coll->closeListen, nullptr)    << "closeListen null after init()";
  EXPECT_NE(coll->makeVDevice, nullptr)    << "makeVDevice null after init()";
  EXPECT_NE(coll->finalize, nullptr)       << "finalize null after init()";

  dlclose(self);
}

} // namespace RcclUnitTesting
