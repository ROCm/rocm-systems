/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Regression test for AICOMRCCL-1891 (NCCL 2.29.7 net.cc fix): RCCL must not call
// ncclNet.finalize() after a failed ncclNet_v10->init().
//
// The v10 plugin ABI has no finalize() at all, so the compat layer in
// src/plugin/net/net_v10.cc synthesizes one and stores it in ncclNet.finalize
// together with the rest of the vtable, but only once ncclNet_v10->init() has
// succeeded. A failed init therefore leaves ncclNet.finalize null, and the pre-fix
// ncclNetPluginInit() called it unconditionally on its failure path: a call through
// a null function pointer.
//
// Uses the same in-process mock approach as PluginCompatV11_test: it exports the
// ncclNetPlugin_v10 symbol the compat layer resolves via dlsym, hands it a
// dlopen(NULL) handle, and drives init() directly. No GPU or real plugin required.

#include <gtest/gtest.h>

#include <dlfcn.h>

#include "nccl.h"
#include "nccl_net.h"

// Compat-layer entry point from librccl.so (net_v10.cc); a global-namespace C++
// function, so this prototype matches the (Debug) library's mangled symbol.
ncclNet_t* getNcclNet_v10(void* lib);

namespace RcclUnitTesting {
namespace {

// Flipped by the test to make the mock plugin's init() fail on demand.
bool gMockV10InitFails = false;
int gMockV10InitCalls = 0;

ncclResult_t mockNetInit(ncclDebugLogger_t, ncclProfilerCallback_t) {
  ++gMockV10InitCalls;
  return gMockV10InitFails ? ncclSystemError : ncclSuccess;
}

// Distinct non-null stubs, so "wired up" can be told apart from "still null".
ncclResult_t mockNetDevices(int* ndev) { if (ndev) { *ndev = 1; } return ncclSuccess; }
ncclResult_t mockNetGetProperties(int, ncclNetProperties_v10_t*) { return ncclSuccess; }
ncclResult_t mockNetListen(int, void*, void**) { return ncclSuccess; }
ncclResult_t mockNetConnect(int, ncclNetCommConfig_v10_t*, void*, void**, ncclNetDeviceHandle_v10_t**) { return ncclSuccess; }
ncclResult_t mockNetAccept(void*, void**, ncclNetDeviceHandle_v10_t**) { return ncclSuccess; }
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
ncclResult_t mockNetMakeVDevice(int*, ncclNetVDeviceProps_v10_t*) { return ncclSuccess; }

} // namespace
} // namespace RcclUnitTesting

// This symbol is what getNcclNet_v10() looks up via dlsym(). The CMake target
// exports it (--export-dynamic-symbol) so it lands in the test executable's
// dynamic symbol table; it must be named exactly, so we place it at global scope
// with C linkage and default visibility.
extern "C" {

__attribute__((visibility("default")))
ncclNet_v10_t ncclNetPlugin_v10 = {
  "mock_v10_net",                        // name
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
};

} // extern "C"

namespace RcclUnitTesting {

// The compat layer keeps one process-global vtable and refcount, so the whole
// contract is walked in a single test rather than split across tests that would
// depend on each other's execution order.
TEST(PluginCompatV10, FinalizeIsNullUntilInitSucceeds) {
  void* self = dlopen(nullptr, RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(self, nullptr) << "dlopen(NULL) failed: " << dlerror();

  ncclNet_t* net = getNcclNet_v10(self);
  ASSERT_NE(net, nullptr)
    << "getNcclNet_v10 returned null; the in-process ncclNetPlugin_v10 symbol "
       "was not resolved (is the test linked with -rdynamic?)";

  // init() is the only entry point the compat layer may publish up front: it is
  // the callback that performs the lazy initialization.
  ASSERT_NE(net->init, nullptr) << "compat layer must set ncclNet.init";
  ASSERT_EQ(net->finalize, nullptr)
    << "finalize must stay null before init(); the v10 ABI has no finalize, so the "
       "compat layer can only publish its own once init() has succeeded";

  // The compat layer dereferences config, so a real one is required here.
  ncclNetCommConfig_t config = {};
  config.trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;

  gMockV10InitFails = true;
  gMockV10InitCalls = 0;

  void* ctx = nullptr;
  EXPECT_NE(net->init(&ctx, /*commId=*/0, &config,
                      /*logFunction=*/nullptr, /*profFunction=*/nullptr),
            ncclSuccess)
    << "compat layer must propagate the plugin's init() failure";
  EXPECT_EQ(gMockV10InitCalls, 1) << "the underlying v10 init() must have been called";

  // The crux of AICOMRCCL-1891: after a failed init there is still no finalize to
  // call, so ncclNetPluginInit()'s failure path must not attempt one. Without the
  // guard in src/plugin/net.cc that call segfaults here.
  EXPECT_EQ(net->finalize, nullptr) << "finalize must stay null after a failed init()";
  EXPECT_EQ(net->devices, nullptr) << "no entry point may be published by a failed init()";

  // A failed init must not have counted towards the compat layer's refcount: if it
  // had, this retry would short-circuit and leave the vtable unpopulated.
  gMockV10InitFails = false;

  ASSERT_EQ(net->init(&ctx, /*commId=*/0, &config,
                      /*logFunction=*/nullptr, /*profFunction=*/nullptr),
            ncclSuccess);
  EXPECT_EQ(gMockV10InitCalls, 2)
    << "a failed init() must not leave the plugin marked as initialized";

  EXPECT_NE(net->finalize, nullptr) << "finalize null after a successful init()";
  EXPECT_NE(net->devices, nullptr) << "devices null after a successful init()";

  // Release the context the successful init() allocated and rebalance the refcount.
  // Note that finalize() drops the refcount but leaves the process-global vtable
  // populated, so the "finalize is null after a failed init" check above only holds
  // on the first run in a process; this test relies on the harness forking a fresh
  // process per case and must not be run twice in one process (e.g. --gtest_repeat).
  if (net->finalize) EXPECT_EQ(net->finalize(ctx), ncclSuccess);

  dlclose(self);
}

} // namespace RcclUnitTesting
