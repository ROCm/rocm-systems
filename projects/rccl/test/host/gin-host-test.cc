/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/gin/gin_host.cc -- every function in the file
// (AICOMRCCL-2642). Each suite is named GinHost<EntryPoint>Microtest after the
// entry point it covers and is introduced by its own banner comment below; the
// suites are not listed here, so there is one place to keep current rather than
// a copy of the file's own table of contents.
//
// Own binary (rccl-UnitTestsMicroGinHost): gin-plugin-init-test.cc already defines
// ncclParamGinEnable in rccl-UnitTestsMicro, and gin_fakes.cc supplies
// ncclGinQueryLastError to rccl-UnitTestsMicroInit. This TU #includes the hipified
// gin_host.cc (GIN_HOST_CC_PATH) so file-scope helpers and ncclGinProgress are
// reachable. No GPU, no librccl.so, no network.

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>

#include <sched.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "nccl.h"
#include "comm.h"
#include "graph.h"
#include "nccl_device/core.h"
#include "plugin/nccl_net.h"
#include "gin/gin_host.h"

#include "JoinThreadsOnExit.h"
#include "ScopeExit.h"
#include "ScriptedFailure.h"
#include "WaitUntil.h"
#include "fakes/bootstrap_stubs.h"
#include "fakes/nccl_device_core_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/os_fakes.h"

namespace {

int g_nLocalGinDevs = 1;
int g_peerGinCommCount = -1;  // -1: AllGather is a no-op copy; else fill other ranks
int g_peerGinCommCountRanks = 2;  // ranks the AllGather fake fills when g_peerGinCommCount >= 0
int64_t g_paramGinType = -1;
int g_railStride = 1;  // stride reported by the ncclTeamRail fake
int g_railNRanks = 1;
ScriptedFailure g_failTopoGetLocalGinDevs;
ScriptedFailure g_failBootstrapAllGather;
int g_topoGetLocalGinDevsCalls = 0;
int g_bootstrapAllGatherCalls = 0;

// Behaviour installed on bootstrap_stubs.cc's g_bootstrapAllGather seam:
// ncclGinConnectOnce's only AllGather exchanges the per-rank GIN connection
// count, and the test needs to say both "this call fails" and "the peers
// reported N".
ncclResult_t GinCommCountAllGather(void*, void* allData, int size) {
  ncclResult_t ret = g_failBootstrapAllGather.at(++g_bootstrapAllGatherCalls);
  if (ret != ncclSuccess) return ret;
  if (g_peerGinCommCount >= 0 && size == static_cast<int>(sizeof(int))) {
    int* counts = static_cast<int*>(allData);
    // The local rank has already written its own (nonzero) slot; every other
    // rank in the job reports g_peerGinCommCount.
    for (int r = 0; r < g_peerGinCommCountRanks; r++) {
      if (counts[r] == 0) counts[r] = g_peerGinCommCount;
    }
  }
  return ncclSuccess;
}

// Every fixture calls this from SetUp: the knobs are file-scope, so a test that
// moves one would otherwise reach the next fixture in the same binary.
void ResetGinHostGlobals() {
  g_nLocalGinDevs = 1;
  g_peerGinCommCount = -1;
  g_peerGinCommCountRanks = 2;
  g_paramGinType = -1;
  g_railStride = 1;
  g_railNRanks = 1;
  g_failTopoGetLocalGinDevs = {};
  g_failBootstrapAllGather = {};
  g_topoGetLocalGinDevsCalls = 0;
  g_bootstrapAllGatherCalls = 0;
  // bootstrap_stubs.cc owns the symbol and defaults it to fail-loud, so the
  // behaviour the GIN connection-count exchange needs is installed here rather
  // than defined again in this TU.
  g_bootstrapAllGather = GinCommCountAllGather;
  // Same for the team accessors nccl_device_core_fakes.cc owns. Its world-team
  // default already describes this comm, but its rail default is the zero team
  // ("no rail"), and gin_host.cc reads the rail stride on every setup.
  ResetNcclDeviceCoreFakes();
  g_ncclTeamRail = [](ncclComm_t) { return ncclTeam_t{g_railNRanks, 0, g_railStride}; };
}

}  // namespace

// The two externals with no owning fakes file. ncclParamGinType is
// NCCL_PARAM(GinType) in src/transport/net_ib/gin.cc and ncclTopoGetLocalGinDevs
// lives in src/graph/topo.cc; neither subsystem's fakes file can link into this
// binary, so they are defined here. Both production declarations ARE visible --
// gin/gin_host.h:63 and graph.h:120, both included above -- so a signature change
// upstream makes these definitions an unrelated overload and the real symbol goes
// undefined at link. That is the same guarantee ASSERT_HOOK_MATCHES_PROD gives the
// std::function seams in fakes/, which is why these do not need one.
int64_t ncclParamGinType() { return g_paramGinType; }

ncclResult_t ncclTopoGetLocalGinDevs(struct ncclComm*, int* localGinDevs, int* localGinCount) {
  ncclResult_t ret = g_failTopoGetLocalGinDevs.at(++g_topoGetLocalGinDevsCalls);
  if (ret != ncclSuccess) return ret;
  if (localGinCount) *localGinCount = g_nLocalGinDevs;
  if (localGinDevs) {
    for (int i = 0; i < g_nLocalGinDevs; i++) localGinDevs[i] = i;
  }
  return ncclSuccess;
}

// The os_fakes.cc affinity seams (src/os/linux.cc) do double duty here.
// ncclOsCpuCount is the first thing every progress thread calls and the progress
// loop has no counter of its own while writePending is set, so
// g_ncclOsCpuCountCalls is how a test proves a worker started; and its return
// value selects the pin branch -- 0 (the default) makes the thread skip the pin,
// non-zero makes it apply ginState->cpuAffinity into g_ncclOsSetAffinityMasks.

#include "fakes/param_redirect.h"

#include GIN_HOST_CC_PATH

namespace {

struct FakeGin;

// The opaque handles production threads back into the plugin: the per-comm
// instance (ginInstance), the listen comm, the coll comm and the GIN context.
// Each one carries the fake it came from and the role it was handed out for, so
// a handle that arrives at the wrong entry point fails the test instead of
// being accepted silently.
struct FakeHandle {
  enum class Role { Instance, ListenComm, CollComm, GinCtx };
  FakeHandle() = default;
  FakeHandle(FakeGin* f, Role r) : fake(f), role(r) {}
  FakeGin* fake = nullptr;
  Role role = Role::Instance;
};

// The plugin's ginCtx: a tagged handle in its own right, plus the per-context
// state a test reads back.
struct FakeSlot : FakeHandle {
  int idx = 0;
  std::atomic<int> progressCalls{0};
  ncclResult_t progressResult = ncclSuccess;
  ncclNetDeviceHandle_t handle{};
};

// Which half of the plugin's createContext contract to violate. Production
// rejects each of these separately, so one enumerator per `||` arm.
enum class BadContext { None, NullGinCtx, NullDevHandle, NullHandle };

struct RegMrCall {
  void* collComm;
  void* address;
  size_t size;
  int memType;
  uint64_t mrFlags;
};

struct FakeGin {
  int ndev = 1;
  bool needsProxyProgress = true;
  std::atomic<int> totalProgressCalls{0};
  std::atomic<int> destroyCalls{0};

  ScriptedFailure failDevices;
  ScriptedFailure failListen;
  ScriptedFailure failGetProperties;
  ScriptedFailure failConnect;
  ScriptedFailure failCreateContext;
  ScriptedFailure failDestroyContext;
  ScriptedFailure failCloseColl;
  ScriptedFailure failCloseListen;
  ScriptedFailure failRegMrSym;
  ScriptedFailure failDeregMrSym;
  ScriptedFailure failQueryLastError;

  int devicesCalls = 0;
  int listenCalls = 0;
  int getPropertiesCalls = 0;
  int connectCalls = 0;
  int createContextCalls = 0;
  int closeCollCalls = 0;
  int closeListenCalls = 0;
  int queryCalls = 0;

  // Team arguments of the last connect(), to check the strided-team math.
  int lastConnectNRanks = -1;
  int lastConnectRank = -1;
  // Every ginConfig handed to createContext, in call order.
  std::vector<ncclGinConfig_t> createdConfigs;
  BadContext badContext = BadContext::None;

  std::vector<RegMrCall> regMrCalls;
  std::vector<std::pair<void*, void*>> deregMrCalls;  // (collComm, mhandle)
  bool regMrSymReturnsNullWindow = false;
  // Call number (1-based) whose queryLastError reports an error. 0: none.
  int queryErrorOnCall = 0;
  // Nonzero: ginProgress spins until cleared. Lets a test hold a worker inside
  // the call so HostFinalize's join is observable (the post-join memset of
  // ginState would otherwise stop progress even if the join were deleted).
  std::atomic<int> holdProgress{0};
  std::atomic<int> progressHolders{0};
  std::vector<std::unique_ptr<FakeSlot>> slots;
  std::vector<void*> destroyed;

  // --- the handles this fake hands out --------------------------------------
  // The instance and listen comm are single objects (production keeps one of
  // each per backend); coll comms are per connection, created on demand so a
  // test that fills ginComms[] by hand uses the same tagged handles connect()
  // would have returned.
  FakeHandle instanceHandle{this, FakeHandle::Role::Instance};
  FakeHandle listenHandle{this, FakeHandle::Role::ListenComm};
  std::vector<std::unique_ptr<FakeHandle>> collHandles;

  void* collCommHandle(int index) {
    while (static_cast<int>(collHandles.size()) <= index) {
      collHandles.emplace_back(new FakeHandle(this, FakeHandle::Role::CollComm));
    }
    return collHandles[index].get();
  }

  FakeSlot* addSlot() {
    std::unique_ptr<FakeSlot> slot(new FakeSlot());
    slot->fake = this;
    slot->role = FakeHandle::Role::GinCtx;
    slot->idx = static_cast<int>(slots.size());
    FakeSlot* raw = slot.get();
    slots.push_back(std::move(slot));
    return raw;
  }

  // The vtable's devices() and getProperties() are handed no handle at all, so
  // those two -- and only those two -- recover the fake from here. Every other
  // entry point recovers it from the handle production passed back.
  static FakeGin*& handlelessInstance() {
    static FakeGin* p = nullptr;
    return p;
  }
  static void setCurrent(FakeGin* p) { handlelessInstance() = p; }

  // --- test-facing behaviour: the logic lives here, in member functions ------
  ncclResult_t devices(int* ndevOut) {
    ++devicesCalls;
    if (ndevOut) *ndevOut = ndev;
    return failDevices.at(devicesCalls);
  }
  // A plugin that fails hands back no listenComm, so production's fail path sees
  // the NULL slot and skips closeListen for it.
  ncclResult_t listen(void** listenComm) {
    ++listenCalls;
    ncclResult_t ret = failListen.at(listenCalls);
    if (ret == ncclSuccess) *listenComm = &listenHandle;
    return ret;
  }
  ncclResult_t getProperties(ncclNetProperties_t* props) {
    ++getPropertiesCalls;
    if (props) std::memset(props, 0, sizeof(*props));
    return failGetProperties.at(getPropertiesCalls);
  }
  ncclResult_t connect(int nRanks, int rank, void** collComm) {
    ++connectCalls;
    lastConnectNRanks = nRanks;
    lastConnectRank = rank;
    ncclResult_t ret = failConnect.at(connectCalls);
    // One handle per connection, so a later call can be traced to the
    // connection it was made on.
    if (ret == ncclSuccess) *collComm = collCommHandle(connectCalls - 1);
    return ret;
  }
  ncclResult_t closeListen() {
    ++closeListenCalls;
    return failCloseListen.at(closeListenCalls);
  }
  ncclResult_t closeColl() {
    ++closeCollCalls;
    return failCloseColl.at(closeCollCalls);
  }

  ncclResult_t createContext(ncclGinConfig_t* config, void** ginCtx, ncclNetDeviceHandle_t** devHandle) {
    ++createContextCalls;
    if (config) createdConfigs.push_back(*config);
    ncclResult_t ret = failCreateContext.at(createContextCalls);
    if (ret != ncclSuccess) return ret;
    FakeSlot* slot = addSlot();
    slot->handle.netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
    slot->handle.handle = badContext == BadContext::NullHandle ? nullptr : slot;
    slot->handle.needsProxyProgress = needsProxyProgress ? 1 : 0;
    *devHandle = badContext == BadContext::NullDevHandle ? nullptr : &slot->handle;
    *ginCtx = badContext == BadContext::NullGinCtx ? nullptr : static_cast<FakeHandle*>(slot);
    return ncclSuccess;
  }

  ncclResult_t destroyContext(void* ginCtx) {
    ++destroyCalls;
    destroyed.push_back(ginCtx);
    return failDestroyContext.at(destroyCalls);
  }

  ncclResult_t regMrSym(void* collComm, void* address, size_t size, int memType, uint64_t mrFlags,
                        void** mhandle, void** ginHandle) {
    regMrCalls.push_back(RegMrCall{collComm, address, size, memType, mrFlags});
    const int callNumber = static_cast<int>(regMrCalls.size());
    ncclResult_t ret = failRegMrSym.at(callNumber);
    if (ret != ncclSuccess) return ret;
    // Distinct per call so a test can tell the slots apart.
    if (mhandle) {
      *mhandle = regMrSymReturnsNullWindow ? nullptr
                                           : reinterpret_cast<void*>(0x1000 + (uintptr_t)callNumber);
    }
    if (ginHandle) *ginHandle = reinterpret_cast<void*>(0x2000 + (uintptr_t)callNumber);
    return ncclSuccess;
  }

  ncclResult_t deregMrSym(void* collComm, void* mhandle) {
    deregMrCalls.emplace_back(collComm, mhandle);
    return failDeregMrSym.at(static_cast<int>(deregMrCalls.size()));
  }

  ncclResult_t progress(FakeSlot* slot) {
    if (holdProgress.load(std::memory_order_acquire) != 0) {
      progressHolders.fetch_add(1, std::memory_order_release);
      while (holdProgress.load(std::memory_order_acquire) != 0) {
        std::this_thread::yield();
      }
      progressHolders.fetch_sub(1, std::memory_order_release);
    }
    slot->progressCalls.fetch_add(1);
    totalProgressCalls.fetch_add(1);
    return slot->progressResult;
  }

  ncclResult_t queryLastError(bool* hasError) {
    ++queryCalls;
    // A plugin that fails the query reports nothing, so the out-parameter is
    // left as production set it -- otherwise the fake, not production, would be
    // what clears it.
    ncclResult_t ret = failQueryLastError.at(queryCalls);
    if (ret != ncclSuccess) return ret;
    if (hasError) *hasError = (queryErrorOnCall != 0 && queryCalls == queryErrorOnCall);
    return ncclSuccess;
  }

  // --- handle recovery ------------------------------------------------------
  // A handle that reaches an entry point it was not handed out for is a
  // production bug the fake must not paper over, so the tag is checked on every
  // recovery.
  static FakeGin* fakeFrom(void* h, FakeHandle::Role want) {
    auto* handle = static_cast<FakeHandle*>(h);
    if (handle == nullptr) {
      ADD_FAILURE() << "GIN plugin entry point called with a null handle";
      return handlelessInstance();
    }
    EXPECT_EQ(static_cast<int>(want), static_cast<int>(handle->role))
        << "GIN handle passed to the wrong plugin entry point";
    return handle->fake;
  }
  static FakeSlot* slotFrom(void* ginCtx) {
    fakeFrom(ginCtx, FakeHandle::Role::GinCtx);
    return static_cast<FakeSlot*>(static_cast<FakeHandle*>(ginCtx));
  }

  // --- vtable-facing adapters: recover + forward, no logic ------------------
  static ncclResult_t Devices(int* ndev) { return handlelessInstance()->devices(ndev); }
  static ncclResult_t GetProperties(int, ncclNetProperties_t* props) {
    return handlelessInstance()->getProperties(props);
  }
  static ncclResult_t Listen(void* ctx, int, void*, void** listenComm) {
    return fakeFrom(ctx, FakeHandle::Role::Instance)->listen(listenComm);
  }
  static ncclResult_t Connect(void* ctx, void**, int nRanks, int rank, void* listenComm, void** collComm) {
    fakeFrom(listenComm, FakeHandle::Role::ListenComm);
    return fakeFrom(ctx, FakeHandle::Role::Instance)->connect(nRanks, rank, collComm);
  }
  static ncclResult_t CloseListen(void* listenComm) {
    return fakeFrom(listenComm, FakeHandle::Role::ListenComm)->closeListen();
  }
  static ncclResult_t CloseColl(void* collComm) {
    return fakeFrom(collComm, FakeHandle::Role::CollComm)->closeColl();
  }
  static ncclResult_t CreateContext(void* collComm, ncclGinConfig_t* config, void** ginCtx,
                                    ncclNetDeviceHandle_t** devHandle) {
    return fakeFrom(collComm, FakeHandle::Role::CollComm)->createContext(config, ginCtx, devHandle);
  }
  static ncclResult_t DestroyContext(void* ginCtx) {
    return fakeFrom(ginCtx, FakeHandle::Role::GinCtx)->destroyContext(ginCtx);
  }
  static ncclResult_t RegMrSym(void* collComm, void* address, size_t size, int memType, uint64_t mrFlags,
                               void** mhandle, void** ginHandle) {
    return fakeFrom(collComm, FakeHandle::Role::CollComm)
        ->regMrSym(collComm, address, size, memType, mrFlags, mhandle, ginHandle);
  }
  static ncclResult_t DeregMrSym(void* collComm, void* mhandle) {
    return fakeFrom(collComm, FakeHandle::Role::CollComm)->deregMrSym(collComm, mhandle);
  }
  static ncclResult_t Progress(void* ginCtx) {
    return fakeFrom(ginCtx, FakeHandle::Role::GinCtx)->progress(slotFrom(ginCtx));
  }
  static ncclResult_t QueryLastError(void* ginCtx, bool* hasError) {
    return fakeFrom(ginCtx, FakeHandle::Role::GinCtx)->queryLastError(hasError);
  }

  ncclGin_t vtable() {
    ncclGin_t gin{};
    gin.name = "GinProxyNthreadsStub";
    gin.devices = &Devices;
    gin.listen = &Listen;
    gin.getProperties = &GetProperties;
    gin.connect = &Connect;
    gin.createContext = &CreateContext;
    gin.destroyContext = &DestroyContext;
    gin.closeColl = &CloseColl;
    gin.closeListen = &CloseListen;
    gin.ginProgress = &Progress;
    gin.queryLastError = &QueryLastError;
    gin.regMrSym = &RegMrSym;
    gin.deregMrSym = &DeregMrSym;
    return gin;
  }
};

class GinHostFixture : public ::testing::Test {
 protected:
  FakeGin fake_;
  ncclGin_t vtable_{};
  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclSharedResources> sr_;
  int64_t nthreadsParam_ = 1;
  int64_t nconnParam_ = -2;

  ncclGinState* gin() { return &sr_->ginState; }
  ncclComm* comm() { return comm_.get(); }

  void SetUp() override {
    FakeGin::setCurrent(&fake_);
    vtable_ = fake_.vtable();
    ResetGinHostGlobals();
    ResetOsFakes();
    fake_.holdProgress.store(0);
    fake_.progressHolders.store(0);
    nthreadsParam_ = 1;
    nconnParam_ = -2;

    g_loadParam = [this](const char* env, int64_t deft) -> int64_t {
      if (std::strcmp(env, "GIN_PROXY_NTHREADS") == 0) return nthreadsParam_;
      if (std::strcmp(env, "GIN_NCONNECTIONS") == 0) return nconnParam_;
      if (std::strcmp(env, "GIN_ENABLE") == 0) return 1;
      return deft;
    };

    comm_ = std::make_unique<ncclComm>();
    sr_ = std::make_unique<ncclSharedResources>();
    comm_->sharedRes = sr_.get();
    comm_->nRanks = 1;
    comm_->rank = 0;
    comm_->cudaDev = 0;
    comm_->symmetricSupport = true;
    comm_->globalGinSupport = NCCL_GIN_CONNECTION_FULL;
    comm_->contiguousRanksPerHost = 1;
    comm_->bootstrap = reinterpret_cast<void*>(0x1);
    comm_->config.trafficClass = NCCL_CONFIG_UNDEF_INT;

    auto* gs = gin();
    gs->supported = true;
    gs->connected = false;
    gs->numActiveBackends = 1;
    gs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    gs->backends[0].ginType = NCCL_GIN_TYPE_PROXY;
    gs->backends[0].ncclGin = &vtable_;
    gs->backends[0].supportsStrongSignals = true;
    gs->backends[0].supportsVASignals = true;
    gs->backends[0].ginInstance = &fake_.instanceHandle;
  }

  // ncclGinHostFinalize memsets the GIN state, which ends the C++ lifetime of
  // its std::thread and std::atomic members. Any test that lets a finalize run
  // to completion has to start a fresh state in place before the fixture's
  // teardown (and ~ncclSharedResources) touch those members again.
  void restoreGinStateAfterFinalize() { new (gin()) ncclGinState{}; }

  void joinProgressThreads() {
    auto* gs = gin();
    gs->proxyThreadStopSignal.store(true);
    for (int t = 0; t < NCCL_GIN_MAX_CONNECTIONS; t++) {
      if (gs->thread[t].joinable()) gs->thread[t].join();
    }
  }

  void TearDown() override {
    fake_.holdProgress.store(0, std::memory_order_release);
    joinProgressThreads();
    FakeGin::setCurrent(nullptr);
    g_loadParam = [](const char*, int64_t deft) { return deft; };
  }

  ncclResult_t connectOnce() { return ncclGinConnectOnce(comm()); }

  ncclDevCommRequirements proxyReqs() {
    ncclDevCommRequirements r = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    r.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    r.ginType = NCCL_GIN_TYPE_PROXY;
    r.ginContextCount = 1;
    r.ginSignalCount = 0;
    r.ginCounterCount = 0;
    r.ginStrongSignalsRequired = false;
    r.ginVaSignalsRequired = false;
    return r;
  }

  // A one-devComm progress list, built by hand and without worker threads.
  // Only the ncclGinProgress tests use this: they need a different
  // needsProxyProgress per connection, which the real setup path -- where the
  // plugin answers the same way for every context it creates -- cannot express.
  // Every other suite builds its list through ncclGinDevCommSetup.
  void attachProgressList(int ginCommCount, int proxyNthreads, const std::vector<int>& needsProxy) {
    auto* gs = gin();
    gs->proxyNthreads = proxyNthreads;
    gs->backends[0].ginCommCount = ginCommCount;
    fake_.slots.clear();

    auto* dc = static_cast<ncclGinStateDevComm*>(std::calloc(1, sizeof(ncclGinStateDevComm)));
    ASSERT_NE(nullptr, dc) << "calloc ncclGinStateDevComm";
    dc->backendIndex = 0;
    // The slot vector is how a test tells which context each ginProgress call
    // landed on.
    for (int i = 0; i < ginCommCount; i++) {
      FakeSlot* slot = fake_.addSlot();
      slot->handle.netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
      slot->handle.handle = slot;
      slot->handle.needsProxyProgress = (i < static_cast<int>(needsProxy.size()) && needsProxy[i]) ? 1 : 0;
      dc->devHandles[i] = &slot->handle;
      dc->ginCtx[i] = static_cast<FakeHandle*>(slot);
    }
    gs->devComms = dc;
  }

  void freeProgressList() {
    auto* gs = gin();
    while (gs->devComms) {
      auto* n = gs->devComms->next;
      std::free(gs->devComms);
      gs->devComms = n;
    }
  }
};

void stopProgress(ncclGinState* gs) { gs->proxyThreadStopSignal.store(true); }

// Test-spawned ncclGinProgress workers: they only leave the progress loop when
// the stop signal is raised, so that is the stop action their joiner needs.
JoinThreadsOnExit progressWorkers(ncclGinState* gs) {
  return JoinThreadsOnExit([gs] { stopProgress(gs); });
}

////////////////////////////////////////////////////////////////////////////////
// FakeGin's own contract -- the handles it hands out are tagged

class GinHostFakeHandleMicrotest : public GinHostFixture {};

// The point of tagging: every suite below trusts that a handle production
// threads to the wrong entry point is caught, so that detection is pinned here
// rather than assumed.
TEST_F(GinHostFakeHandleMicrotest, FakeGin_HandleDeliveredToTheWrongEntryPoint_FailsTheTest) {
  // EXPECT_NONFATAL_FAILURE's body may not name locals or fixture members, so
  // what it calls is staged in statics first.
  static ncclGin_t* plugin;
  static void* collComm;
  plugin = &vtable_;
  collComm = fake_.collCommHandle(0);

  EXPECT_NONFATAL_FAILURE(plugin->closeListen(collComm), "wrong plugin entry point");
}

////////////////////////////////////////////////////////////////////////////////
// ncclGinConnectOnce -- GIN_PROXY_NTHREADS (NVIDIA/nccl#2279, AICOMRCCL-2017)

class GinHostProxyNthreadsMicrotest : public GinHostFixture {};

// Unset GIN_PROXY_NTHREADS → proxyNthreads and ginCommCount stay 1.
TEST_F(GinHostProxyNthreadsMicrotest, ConnectOnce_ProxyNthreadsUnset_UsesOneThreadAndOneConnection) {
  g_loadParam = [](const char* env, int64_t deft) -> int64_t {
    if (std::strcmp(env, "GIN_ENABLE") == 0) return 1;
    return deft;
  };
  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(1, gin()->proxyNthreads);
  EXPECT_EQ(1, gin()->backends[0].ginCommCount);
}

// GIN_PROXY_NTHREADS<=0 is treated as 1; no extra progress threads.
TEST_F(GinHostProxyNthreadsMicrotest, ConnectOnce_ProxyNthreadsNotPositive_UsesOneThread) {
  nthreadsParam_ = 0;
  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(1, gin()->proxyNthreads);

  gin()->connected = false;
  nthreadsParam_ = -3;
  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(1, gin()->proxyNthreads);
}

// GIN_PROXY_NTHREADS>4 is clamped to NCCL_GIN_MAX_CONNECTIONS (4).
TEST_F(GinHostProxyNthreadsMicrotest, ConnectOnce_ProxyNthreadsAboveMax_ClampsToMaxConnections) {
  nthreadsParam_ = 100;
  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(NCCL_GIN_MAX_CONNECTIONS, gin()->proxyNthreads);
  EXPECT_EQ(NCCL_GIN_MAX_CONNECTIONS, gin()->backends[0].ginCommCount);
}

// nthreads=4 with 1 local GIN dev → ginCommCount raised to 4 before AllGather.
TEST_F(GinHostProxyNthreadsMicrotest, ConnectOnce_FewerLocalDevsThanThreads_RaisesGinCommCountToNthreads) {
  nthreadsParam_ = 4;
  g_nLocalGinDevs = 1;
  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(4, gin()->proxyNthreads);
  EXPECT_EQ(4, gin()->backends[0].ginCommCount);
}

// GIN_NCONNECTIONS=2 then GIN_PROXY_NTHREADS=4 → ginCommCount is 4, not 2.
TEST_F(GinHostProxyNthreadsMicrotest, ConnectOnce_NconnectionsBelowNthreads_GinCommCountFollowsNthreads) {
  nthreadsParam_ = 4;
  nconnParam_ = 2;
  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(4, gin()->proxyNthreads);
  EXPECT_EQ(4, gin()->backends[0].ginCommCount);
}

////////////////////////////////////////////////////////////////////////////////
// ncclGinProgress -- which connections a progress thread polls

class GinHostProgressMicrotest : public GinHostFixture {};

// 2 threads × 4 connections: thread t only progresses connections t, t+2, …
TEST_F(GinHostProgressMicrotest, Progress_MultipleThreads_EachThreadOwnsItsStridedConnections) {
  attachProgressList(/*ginCommCount=*/4, /*proxyNthreads=*/2, {1, 1, 1, 1});
  auto* gs = gin();
  {
    auto workers = progressWorkers(gs);
    workers.spawn([gs] { ncclGinProgress(gs, 0); });
    workers.spawn([gs] { ncclGinProgress(gs, 1); });
    ASSERT_TRUE(waitUntil([&] {
      return fake_.slots[0]->progressCalls.load() > 0 && fake_.slots[1]->progressCalls.load() > 0 &&
             fake_.slots[2]->progressCalls.load() > 0 && fake_.slots[3]->progressCalls.load() > 0;
    }));
  }

  // Sequential check of the stride formula: thread 0 never owns odd connections.
  for (auto& s : fake_.slots) s->progressCalls.store(0);
  fake_.totalProgressCalls.store(0);
  gs->proxyThreadStopSignal.store(false);
  {
    auto only0 = progressWorkers(gs);
    only0.spawn([gs] { ncclGinProgress(gs, 0); });
    ASSERT_TRUE(waitUntil([&] {
      return fake_.slots[0]->progressCalls.load() > 0 && fake_.slots[2]->progressCalls.load() > 0;
    }));
  }
  EXPECT_EQ(0, fake_.slots[1]->progressCalls.load());
  EXPECT_EQ(0, fake_.slots[3]->progressCalls.load());
  freeProgressList();
}

// needsProxyProgress=0 (GDA/Anvil) slot is never passed to ginProgress.
TEST_F(GinHostProgressMicrotest, Progress_ConnectionWithoutProxyProgress_IsNeverProgressed) {
  attachProgressList(2, 1, {1, 0});
  auto* gs = gin();
  {
    auto worker = progressWorkers(gs);
    worker.spawn([gs] { ncclGinProgress(gs, 0); });
    ASSERT_TRUE(waitUntil([&] { return fake_.slots[0]->progressCalls.load() > 0; }));
  }
  EXPECT_EQ(0, fake_.slots[1]->progressCalls.load());
  freeProgressList();
}

// writePending=true: progress thread yields (0 ginProgress) until the flag clears.
TEST_F(GinHostProgressMicrotest, Progress_WritePendingSet_StopsProgressingUntilItClears) {
  attachProgressList(1, 1, {1});
  auto* gs = gin();
  gs->writePending.store(true);
  g_ncclOsCpuCountCalls.store(0);
  {
    auto worker = progressWorkers(gs);
    worker.spawn([gs] { ncclGinProgress(gs, 0); });
    // Entry is the positive control: the 50 ms window starts only after the
    // worker has reached ncclGinProgress, so an ignored writePending cannot pass as 0 == 0.
    ASSERT_TRUE(waitUntil([&] { return g_ncclOsCpuCountCalls.load() > 0; }))
        << "progress thread never entered ncclGinProgress";
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(0, fake_.totalProgressCalls.load());
    gs->writePending.store(false);
    ASSERT_TRUE(waitUntil([&] { return fake_.totalProgressCalls.load() > 0; }));
  }
  freeProgressList();
}

// ginProgress returning ncclSystemError stores it in asyncResult and the thread exits.
TEST_F(GinHostProgressMicrotest, Progress_PluginProgressFails_StoresTheStatusInAsyncResultAndExits) {
  attachProgressList(1, 1, {1});
  fake_.slots[0]->progressResult = ncclSystemError;
  auto* gs = gin();
  {
    // On the test thread a missed error return spins in while(1) until the
    // whole binary's timeout. A worker plus waitUntil fails this case instead.
    auto worker = progressWorkers(gs);
    worker.spawn([gs] { ncclGinProgress(gs, 0); });
    ASSERT_TRUE(waitUntil([&] { return gs->asyncResult == ncclSystemError; }))
        << "ginProgress error did not stop the worker";
  }
  EXPECT_EQ(ncclSystemError, gs->asyncResult);
  freeProgressList();
}

////////////////////////////////////////////////////////////////////////////////
// Progress-thread lifecycle across ncclGinDevCommSetup / ncclGinDevCommFree

class GinHostProgressThreadsMicrotest : public GinHostFixture {};

// First PROXY DevCommCreate starts exactly nthreads joinable progress threads.
TEST_F(GinHostProgressThreadsMicrotest, DevCommSetup_FirstProxyDevComm_StartsOneJoinableThreadPerProxyNthread) {
  nthreadsParam_ = 2;
  ASSERT_EQ(ncclSuccess, connectOnce());
  ncclDevComm devComm{};
  auto reqs = proxyReqs();
  reqs.ginContextCount = gin()->backends[0].ginCommCount;
  ASSERT_EQ(ncclSuccess, ncclGinDevCommSetup(comm(), &reqs, &devComm, NCCL_VERSION_CODE));
  EXPECT_TRUE(gin()->proxyThreadsCreated);
  EXPECT_TRUE(gin()->thread[0].joinable());
  EXPECT_TRUE(gin()->thread[1].joinable());
  EXPECT_FALSE(gin()->thread[2].joinable());
  ASSERT_EQ(ncclSuccess, ncclGinDevCommFree(comm(), &devComm));
}

// GDA DevComm first (no threads), then PROXY DevComm starts them (kgioioso hang).
TEST_F(GinHostProgressThreadsMicrotest, DevCommSetup_EarlierDevCommNeededNoProxyProgress_StartsThreadsOnTheFirstProxyDevComm) {
  nthreadsParam_ = 1;
  ASSERT_EQ(ncclSuccess, connectOnce());

  fake_.needsProxyProgress = false;
  ncclDevComm first{};
  auto reqs = proxyReqs();
  reqs.ginContextCount = gin()->backends[0].ginCommCount;
  ASSERT_EQ(ncclSuccess, ncclGinDevCommSetup(comm(), &reqs, &first, NCCL_VERSION_CODE));
  EXPECT_FALSE(gin()->proxyThreadsCreated);

  fake_.needsProxyProgress = true;
  ncclDevComm second{};
  ASSERT_EQ(ncclSuccess, ncclGinDevCommSetup(comm(), &reqs, &second, NCCL_VERSION_CODE));
  EXPECT_TRUE(gin()->proxyThreadsCreated);
  EXPECT_TRUE(gin()->thread[0].joinable());

  ASSERT_EQ(ncclSuccess, ncclGinDevCommFree(comm(), &first));
  ASSERT_EQ(ncclSuccess, ncclGinDevCommFree(comm(), &second));
}

// DevCommFree unlinks that ctx; ginProgress stops on it while the sibling still runs.
TEST_F(GinHostProgressThreadsMicrotest, DevCommFree_OneOfTwoDevComms_StopsProgressingOnlyTheFreedContext) {
  nthreadsParam_ = 1;
  ASSERT_EQ(ncclSuccess, connectOnce());
  auto reqs = proxyReqs();
  reqs.ginContextCount = gin()->backends[0].ginCommCount;

  ncclDevComm keep{};
  ncclDevComm drop{};
  ASSERT_EQ(ncclSuccess, ncclGinDevCommSetup(comm(), &reqs, &keep, NCCL_VERSION_CODE));
  ASSERT_EQ(ncclSuccess, ncclGinDevCommSetup(comm(), &reqs, &drop, NCCL_VERSION_CODE));
  ASSERT_EQ(2u, fake_.slots.size());
  FakeSlot* dropSlot = fake_.slots.back().get();
  ASSERT_TRUE(waitUntil([&] { return dropSlot->progressCalls.load() > 0; }))
      << "dropped ctx was never progressed";
  ASSERT_EQ(ncclSuccess, ncclGinDevCommFree(comm(), &drop));
  EXPECT_GE(fake_.destroyCalls.load(), 1);

  const int afterFree = dropSlot->progressCalls.load();
  // The worker walks keep before drop, so front()->progressCalls > 0 is already
  // true here. Wait for a new call so the freeze window below has a live worker.
  const int keepBefore = fake_.slots.front()->progressCalls.load();
  ASSERT_TRUE(waitUntil([&] { return fake_.slots.front()->progressCalls.load() > keepBefore; }))
      << "kept ctx stopped progressing after the sibling was freed";
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_EQ(afterFree, dropSlot->progressCalls.load()) << "freed ctx kept receiving ginProgress";

  ASSERT_EQ(ncclSuccess, ncclGinDevCommFree(comm(), &keep));
}

// nthreads=4 but AllGather ginCommCount=2: threads 2 and 3 never call ginProgress.
TEST_F(GinHostProgressMicrotest, Progress_ThreadIndexBeyondTheConnectionCount_ProgressesNothing) {
  nthreadsParam_ = 4;
  comm_->nRanks = 2;
  g_peerGinCommCount = 2;
  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(4, gin()->proxyNthreads);
  EXPECT_EQ(2, gin()->backends[0].ginCommCount);

  attachProgressList(gin()->backends[0].ginCommCount, gin()->proxyNthreads, {1, 1});
  auto* gs = gin();
  g_ncclOsCpuCountCalls.store(0);
  {
    auto idle = progressWorkers(gs);
    idle.spawn([gs] { ncclGinProgress(gs, 2); });
    idle.spawn([gs] { ncclGinProgress(gs, 3); });
    ASSERT_TRUE(waitUntil([&] { return g_ncclOsCpuCountCalls.load() >= 2; }))
        << "idle progress threads never entered ncclGinProgress";
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(0, fake_.totalProgressCalls.load());
  }
  freeProgressList();
}

////////////////////////////////////////////////////////////////////////////////
// ncclGinProgress -- the proxy-thread CPU affinity pin (AICOMRCCL-1859). The
// comm->cpuAffinity stash that feeds it is in GinHostDevCommSetupMicrotest.

class GinHostProxyAffinityMicrotest : public ::testing::Test {
protected:
  ncclGinState ginState_;

  void SetUp() override {
    ResetGinHostGlobals();
    ResetOsFakes();
    CPU_ZERO(&ginState_.cpuAffinity);
    ginState_.proxyThreadStopSignal.store(true);  // exit at the top of the loop
    ginState_.writePending.store(false);
  }

  void TearDown() override { ResetOsFakes(); }

  // Run ncclGinProgress on a fresh thread (so the affinity apply, if the fake
  // were real, would land on a throwaway thread rather than the test runner)
  // and wait for it to return.
  void RunProgressOnce() {
    std::thread t([this] { ncclGinProgress(&ginState_, /*threadIdx=*/0); });
    t.join();
  }
};

TEST_F(GinHostProxyAffinityMicrotest, NonEmptyAffinity_PinsProxyThreadToThatCpuSet) {
  CPU_SET(3, &ginState_.cpuAffinity);
  g_ncclOsCpuCountValue = 1;

  RunProgressOnce();

  ASSERT_EQ(1u, g_ncclOsSetAffinityMasks.size());
  EXPECT_TRUE(CPU_ISSET(3, &g_ncclOsSetAffinityMasks[0]));
  EXPECT_EQ(1, CPU_COUNT(&g_ncclOsSetAffinityMasks[0]));
}

TEST_F(GinHostProxyAffinityMicrotest, EmptyAffinity_LeavesProxyThreadAffinityUnchanged) {
  g_ncclOsCpuCountValue = 0;

  RunProgressOnce();

  EXPECT_EQ(1, g_ncclOsCpuCountCalls.load());
  EXPECT_TRUE(g_ncclOsSetAffinityMasks.empty());
}

////////////////////////////////////////////////////////////////////////////////
// ncclGetGinType / ncclGetRailedGinType

class GinHostTypeQueryMicrotest : public GinHostFixture {};

// The accessor rejects a missing comm or a missing out-parameter rather than
// dereferencing them.
TEST_F(GinHostTypeQueryMicrotest, GetGinType_NullCommOrOutParameter_ReturnsInternalError) {
  ncclGinType_t type = NCCL_GIN_TYPE_NONE;
  EXPECT_EQ(ncclInternalError, ncclGetGinType(nullptr, &type));
  EXPECT_EQ(ncclInternalError, ncclGetGinType(comm(), nullptr));
}

TEST_F(GinHostTypeQueryMicrotest, GetRailedGinType_NullCommOrOutParameter_ReturnsInternalError) {
  ncclGinType_t type = NCCL_GIN_TYPE_NONE;
  EXPECT_EQ(ncclInternalError, ncclGetRailedGinType(nullptr, &type));
  EXPECT_EQ(ncclInternalError, ncclGetRailedGinType(comm(), nullptr));
}

// ncclGetGinType answers for all-to-all reachability, which a FULL comm has.
TEST_F(GinHostTypeQueryMicrotest, GetGinType_FullyConnectedComm_ReportsTheBackendType) {
  gin()->backends[0].ginType = NCCL_GIN_TYPE_ANVIL_SDMA;
  comm_->globalGinSupport = NCCL_GIN_CONNECTION_FULL;

  ncclGinType_t type = NCCL_GIN_TYPE_NONE;
  ASSERT_EQ(ncclSuccess, ncclGetGinType(comm(), &type));
  EXPECT_EQ(NCCL_GIN_TYPE_ANVIL_SDMA, type);
}

// A rail-only comm cannot reach every peer, so it reports no GIN at all.
TEST_F(GinHostTypeQueryMicrotest, GetGinType_RailOnlyComm_ReportsNoGin) {
  gin()->backends[0].ginType = NCCL_GIN_TYPE_ANVIL_SDMA;
  comm_->globalGinSupport = NCCL_GIN_CONNECTION_RAIL;

  ncclGinType_t type = NCCL_GIN_TYPE_NONE;
  ASSERT_EQ(ncclSuccess, ncclGetGinType(comm(), &type));
  EXPECT_EQ(NCCL_GIN_TYPE_NONE, type);
}

// ncclGetRailedGinType answers for rail reachability, so it reports the
// backend's type for the RAIL case the accessor above hides.
TEST_F(GinHostTypeQueryMicrotest, GetRailedGinType_RailConnectedComm_ReportsTheBackendType) {
  gin()->backends[0].ginType = NCCL_GIN_TYPE_GDAKI;
  comm_->globalGinSupport = NCCL_GIN_CONNECTION_RAIL;

  ncclGinType_t type = NCCL_GIN_TYPE_NONE;
  ASSERT_EQ(ncclSuccess, ncclGetRailedGinType(comm(), &type));
  EXPECT_EQ(NCCL_GIN_TYPE_GDAKI, type);
}

// A comm with no GIN connectivity at all has no rail reachability either.
TEST_F(GinHostTypeQueryMicrotest, GetRailedGinType_UnconnectedComm_ReportsNoGin) {
  gin()->backends[0].ginType = NCCL_GIN_TYPE_GDAKI;
  comm_->globalGinSupport = NCCL_GIN_CONNECTION_NONE;

  ncclGinType_t type = NCCL_GIN_TYPE_NONE;
  ASSERT_EQ(ncclSuccess, ncclGetRailedGinType(comm(), &type));
  EXPECT_EQ(NCCL_GIN_TYPE_NONE, type);
}

////////////////////////////////////////////////////////////////////////////////
// ncclGinValidateSignalRequest

class GinHostSignalRequestMicrotest : public GinHostFixture {};

// A backend is only a candidate if it supports every signal flavour the
// requirements insist on.
TEST_F(GinHostSignalRequestMicrotest, ValidateSignalRequest_BackendMissingARequiredSignalFlavour_ReturnsInvalidUsage) {
  auto* backend = &gin()->backends[0];
  auto reqs = proxyReqs();

  reqs.ginStrongSignalsRequired = true;
  backend->supportsStrongSignals = false;
  EXPECT_EQ(ncclInvalidUsage, ncclGinValidateSignalRequest(&reqs, backend));

  backend->supportsStrongSignals = true;
  reqs.ginVaSignalsRequired = true;
  backend->supportsVASignals = false;
  EXPECT_EQ(ncclInvalidUsage, ncclGinValidateSignalRequest(&reqs, backend));
}

// The mirror image of the refusals above: a backend that has every required
// flavour is a candidate.
TEST_F(GinHostSignalRequestMicrotest, ValidateSignalRequest_BackendSupportsEveryRequiredFlavour_ReturnsSuccess) {
  auto* backend = &gin()->backends[0];
  backend->supportsStrongSignals = true;
  backend->supportsVASignals = true;

  auto reqs = proxyReqs();
  reqs.ginStrongSignalsRequired = true;
  reqs.ginVaSignalsRequired = true;
  EXPECT_EQ(ncclSuccess, ncclGinValidateSignalRequest(&reqs, backend));
}

// Signals that are not required are never checked, so a backend without them
// still qualifies.
TEST_F(GinHostSignalRequestMicrotest, ValidateSignalRequest_SignalsNotRequired_IgnoresTheBackendsLackOfThem) {
  auto* backend = &gin()->backends[0];
  backend->supportsStrongSignals = false;
  backend->supportsVASignals = false;

  auto reqs = proxyReqs();  // both *Required flags already false
  EXPECT_EQ(ncclSuccess, ncclGinValidateSignalRequest(&reqs, backend));
}

////////////////////////////////////////////////////////////////////////////////
// ncclGinConnectOnce -- refusals, plugin failures, and the strided team

class GinHostConnectOnceMicrotest : public GinHostFixture {};

// The second connect returns early: it never reaches the plugin again.
TEST_F(GinHostConnectOnceMicrotest, ConnectOnce_AlreadyConnected_DoesNotReachThePlugin) {
  ASSERT_EQ(ncclSuccess, connectOnce());
  const int devicesCalls = fake_.devicesCalls;
  ASSERT_GE(devicesCalls, 1);

  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(devicesCalls, fake_.devicesCalls);
}

// NCCL_GIN_ENABLE=0 is a hard refusal, not a silent fallback.
TEST_F(GinHostConnectOnceMicrotest, ConnectOnce_GinDisabledByParam_ReturnsInternalError) {
  g_loadParam = [](const char* env, int64_t deft) -> int64_t {
    if (std::strcmp(env, "GIN_ENABLE") == 0) return 0;
    return deft;
  };
  EXPECT_EQ(ncclInternalError, connectOnce());
  EXPECT_FALSE(gin()->connected);
  EXPECT_EQ(0, fake_.devicesCalls);
}

// No backend was loaded for this comm: an invalid request, not an internal fault.
TEST_F(GinHostConnectOnceMicrotest, ConnectOnce_NoBackendLoaded_ReturnsInvalidUsage) {
  gin()->supported = false;
  EXPECT_EQ(ncclInvalidUsage, connectOnce());
  EXPECT_FALSE(gin()->connected);
}

// GIN windows are symmetric memory, so a comm without symmetric support cannot
// host them.
TEST_F(GinHostConnectOnceMicrotest, ConnectOnce_CommWithoutSymmetricSupport_ReturnsInternalError) {
  comm_->symmetricSupport = false;
  EXPECT_EQ(ncclInternalError, connectOnce());
  EXPECT_FALSE(gin()->connected);
}

// A plugin that reports zero devices is rejected before any listen/connect.
TEST_F(GinHostConnectOnceMicrotest, ConnectOnce_PluginReportsZeroDevices_ReturnsInternalErrorBeforeListening) {
  fake_.ndev = 0;
  EXPECT_EQ(ncclInternalError, connectOnce());
  EXPECT_EQ(0, fake_.listenCalls);
  EXPECT_FALSE(gin()->connected);
}

// A failing devices() surfaces the plugin's own status, and stops the connect
// before the backend is sized or anything is opened.
TEST_F(GinHostConnectOnceMicrotest, ConnectOnce_DevicesFails_PropagatesTheStatusAndOpensNothing) {
  gin()->backends[0].ginCommCount = 7;  // poison: only a connect that got past devices() rewrites this
  fake_.failDevices = {ncclSystemError, 1};

  EXPECT_EQ(ncclSystemError, connectOnce());

  EXPECT_FALSE(gin()->connected);
  EXPECT_EQ(7, gin()->backends[0].ginCommCount);
  EXPECT_EQ(0, fake_.listenCalls);
  EXPECT_EQ(0, fake_.connectCalls);
}

// More local GIN devices than connections: the extra devices are dropped rather
// than overrunning the fixed-size connection arrays.
TEST_F(GinHostConnectOnceMicrotest, ConnectOnce_MoreLocalDevsThanMaxConnections_ClampsToMaxConnections) {
  g_nLocalGinDevs = NCCL_GIN_MAX_CONNECTIONS + 2;
  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(NCCL_GIN_MAX_CONNECTIONS, gin()->backends[0].ginCommCount);
  EXPECT_EQ(NCCL_GIN_MAX_CONNECTIONS, fake_.connectCalls);
}

// A failed connect releases the listen comm it was handed and leaves the state
// disconnected, so a later retry starts clean.
TEST_F(GinHostConnectOnceMicrotest, ConnectOnce_ConnectFails_ClosesTheListenComm) {
  fake_.failConnect = {ncclSystemError, 1};
  EXPECT_EQ(ncclSystemError, connectOnce());
  EXPECT_EQ(1, fake_.closeListenCalls);
  EXPECT_EQ(0, fake_.closeCollCalls);
  EXPECT_FALSE(gin()->connected);
}

// Failing partway through closes the connections that already succeeded and
// clears their slots.
TEST_F(GinHostConnectOnceMicrotest, ConnectOnce_SecondConnectFails_ClosesTheEarlierConnection) {
  nthreadsParam_ = 2;  // two connections, so there is an earlier one to close
  fake_.failConnect = {ncclSystemError, 2};
  EXPECT_EQ(ncclSystemError, connectOnce());
  EXPECT_EQ(1, fake_.closeCollCalls);
  EXPECT_EQ(nullptr, gin()->backends[0].ginComms[0]);
  EXPECT_FALSE(gin()->connected);
}

// A listen that fails hands back no listen comm, so the fail path must not try
// to close one.
TEST_F(GinHostConnectOnceMicrotest, ConnectOnce_ListenFails_DoesNotCloseAListenComm) {
  fake_.failListen = {ncclSystemError, 1};
  EXPECT_EQ(ncclSystemError, connectOnce());
  EXPECT_EQ(0, fake_.closeListenCalls);
}

// A topology query that fails stops the connect before the plugin is touched.
TEST_F(GinHostConnectOnceMicrotest, ConnectOnce_TopologyQueryFails_PropagatesTheStatusBeforeTouchingThePlugin) {
  gin()->backends[0].ginCommCount = 7;  // poison, as in the devices() failure above
  g_failTopoGetLocalGinDevs = {ncclSystemError, 1};

  EXPECT_EQ(ncclSystemError, connectOnce());

  EXPECT_EQ(0, fake_.devicesCalls);
  EXPECT_FALSE(gin()->connected);
  EXPECT_EQ(7, gin()->backends[0].ginCommCount);
  EXPECT_EQ(0, g_bootstrapAllGatherCalls);
}

// A failed property query releases the listen comm that was opened for the same
// connection.
TEST_F(GinHostConnectOnceMicrotest, ConnectOnce_GetPropertiesFails_ClosesTheListenCommWithoutConnecting) {
  fake_.failGetProperties = {ncclInternalError, 1};
  EXPECT_EQ(ncclInternalError, connectOnce());
  EXPECT_EQ(1, fake_.closeListenCalls);
  EXPECT_EQ(0, fake_.connectCalls);
}

// Ranks agree on the connection count before connecting, so a failed exchange
// stops the connect.
TEST_F(GinHostConnectOnceMicrotest, ConnectOnce_ConnectionCountExchangeFails_PropagatesTheStatusAndOpensNothing) {
  g_failBootstrapAllGather = {ncclSystemError, 1};

  EXPECT_EQ(ncclSystemError, connectOnce());

  EXPECT_FALSE(gin()->connected);
  // The exchange decides how many connections to open, so nothing was opened
  // yet and the cleanup path has nothing to close.
  EXPECT_EQ(0, fake_.listenCalls);
  EXPECT_EQ(0, fake_.connectCalls);
  EXPECT_EQ(0, fake_.closeListenCalls);
  EXPECT_EQ(0, fake_.closeCollCalls);
  EXPECT_EQ(nullptr, gin()->backends[0].ginComms[0]);
}

// Cleanup failures must not mask the failure that caused the cleanup: the
// caller still sees why the connect failed.
TEST_F(GinHostConnectOnceMicrotest, ConnectOnce_CleanupAlsoFails_ReportsTheOriginalFailure) {
  fake_.failConnect = {ncclSystemError, 1};
  fake_.failCloseListen = {ncclInternalError, 1};
  EXPECT_EQ(ncclSystemError, connectOnce());
  EXPECT_EQ(1, fake_.closeListenCalls);
}

// A comm that is only rail-connected connects a strided team of one rank per
// host instead of the world team.
TEST_F(GinHostConnectOnceMicrotest, ConnectOnce_RailOnlyComm_ConnectsOneRankPerHostTeam) {
  comm_->nRanks = 4;
  comm_->rank = 2;
  comm_->contiguousRanksPerHost = 2;
  comm_->globalGinSupport = NCCL_GIN_CONNECTION_RAIL;
  g_peerGinCommCount = NCCL_GIN_MAX_CONNECTIONS;  // peers do not lower ginCommCount
  g_peerGinCommCountRanks = 4;

  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(NCCL_GIN_CONNECTION_RAIL, gin()->ginConnectionType);
  EXPECT_EQ(2, fake_.lastConnectNRanks);  // 4 ranks / 2 ranks per host
  EXPECT_EQ(1, fake_.lastConnectRank);    // rank 2 is the second host
}

////////////////////////////////////////////////////////////////////////////////
// ncclGinDevCommSetup -- backend selection

class GinHostBackendSelectMicrotest : public GinHostFixture {
 protected:
  ncclDevComm devComm_{};

  void SetUp() override {
    GinHostFixture::SetUp();
    ASSERT_EQ(ncclSuccess, connectOnce());
  }

  ncclResult_t setup(ncclDevCommRequirements const& reqs) {
    return ncclGinDevCommSetup(comm(), &reqs, &devComm_, NCCL_VERSION_CODE);
  }

  void TearDown() override {
    if (gin()->devComms != nullptr) EXPECT_EQ(ncclSuccess, ncclGinDevCommFree(comm(), &devComm_));
    GinHostFixture::TearDown();
  }
};

// A type outside the enum is rejected before any backend is considered.
TEST_F(GinHostBackendSelectMicrotest, DevCommSetup_GinTypeOutOfRange_ReturnsInvalidUsageBeforeAnyBackendIsConsidered) {
  auto reqs = proxyReqs();
  reqs.ginType = NCCL_GIN_MAX_TYPES;
  EXPECT_EQ(ncclInvalidUsage, setup(reqs));
  EXPECT_EQ(0, fake_.createContextCalls);
}

// NCCL_GIN_TYPE picks the backend when the requirements express no preference.
TEST_F(GinHostBackendSelectMicrotest, DevCommSetup_NeutralRequestAndEnvTypeMatchingALoadedBackend_SelectsThatBackend) {
  auto reqs = proxyReqs();
  reqs.ginType = NCCL_GIN_TYPE_NONE;
  g_paramGinType = NCCL_GIN_TYPE_PROXY;  // matches the loaded backend

  EXPECT_EQ(ncclSuccess, setup(reqs));
  EXPECT_EQ(1, fake_.createContextCalls);
}

// NCCL_GIN_TYPE is a demand, not a hint: with no backend of that type loaded
// the setup fails rather than falling back to the one that is.
TEST_F(GinHostBackendSelectMicrotest, DevCommSetup_NeutralRequestAndEnvTypeWithNoLoadedBackend_ReturnsInternalError) {
  auto reqs = proxyReqs();
  reqs.ginType = NCCL_GIN_TYPE_NONE;
  g_paramGinType = NCCL_GIN_TYPE_GDAKI;  // no GDAKI backend is loaded

  EXPECT_EQ(ncclInternalError, setup(reqs));
  EXPECT_EQ(0, fake_.createContextCalls);
}

// With no preference from either the requirements or the environment, the first
// loaded backend is taken whatever its type.
TEST_F(GinHostBackendSelectMicrotest, DevCommSetup_NeutralRequestAndNoEnvType_SelectsTheFirstLoadedBackend) {
  ASSERT_EQ(-1, g_paramGinType);  // NCCL_GIN_TYPE unset
  gin()->backends[0].ginType = NCCL_GIN_TYPE_ANVIL_SDMA;

  auto reqs = proxyReqs();
  reqs.ginType = NCCL_GIN_TYPE_NONE;
  EXPECT_EQ(ncclSuccess, setup(reqs));
  EXPECT_EQ(1, fake_.createContextCalls);
}

// A backend that cannot provide a required signal flavour is skipped, not failed
// into.
TEST_F(GinHostBackendSelectMicrotest, DevCommSetup_OnlyBackendFailsSignalValidation_ReturnsInternalErrorWithoutCreatingContexts) {
  gin()->backends[0].supportsStrongSignals = false;
  auto reqs = proxyReqs();
  reqs.ginStrongSignalsRequired = true;

  EXPECT_EQ(ncclInternalError, setup(reqs));
  EXPECT_EQ(0, fake_.createContextCalls);
}

// When the first candidate's setup fails, the next backend gets a turn and the
// devComm records the one that worked.
TEST_F(GinHostBackendSelectMicrotest, DevCommSetup_FirstBackendsSetupFails_SucceedsOnTheNextBackend) {
  auto* gs = gin();
  gs->numActiveBackends = 2;
  gs->backends[1] = gs->backends[0];
  gs->backends[1].ginComms[0] = fake_.collCommHandle(1);
  fake_.failCreateContext = {ncclSystemError, 1};  // only the first backend's attempt

  auto reqs = proxyReqs();
  ASSERT_EQ(ncclSuccess, setup(reqs));
  EXPECT_EQ(1, devComm_.backendIndex);
  EXPECT_EQ(2, fake_.createContextCalls);
}

////////////////////////////////////////////////////////////////////////////////
// ncclGinDevCommSetup -- config, strides, and failure cleanup

// Drives the public entry point against a hand-built backend, so no connect
// plumbing stands between the test and the branch under test while the type and
// signal gating in front of the per-backend setup still runs. The refusals whose
// status the public entry point collapses reach the file-static setup instead --
// see setupWithBackend below.
class GinHostDevCommSetupMicrotest : public GinHostFixture {
 protected:
  ncclDevComm devComm_{};

  void SetUp() override {
    GinHostFixture::SetUp();
    auto* gs = gin();
    gs->proxyNthreads = 1;
    gs->proxyThreadStopSignal.store(true);  // any spawned worker exits immediately
    gs->backends[0].ginCommCount = 1;
    gs->backends[0].ginComms[0] = fake_.collCommHandle(0);
  }

  void TearDown() override {
    if (gin()->devComms != nullptr) EXPECT_EQ(ncclSuccess, ncclGinDevCommFree(comm(), &devComm_));
    GinHostFixture::TearDown();
  }

  ncclResult_t setup(ncclDevCommRequirements const& reqs, ncclDevComm* devComm,
                     uint32_t deviceCodeVersion = NCCL_VERSION_CODE) {
    return ncclGinDevCommSetup(comm(), &reqs, devComm, deviceCodeVersion);
  }

  // One setup plus the matching free, so a table-driven test can loop over
  // variants without leaking a context per iteration.
  ncclResult_t setupAndFree(ncclDevCommRequirements const& reqs, uint32_t deviceCodeVersion) {
    ncclDevComm devComm{};
    ncclResult_t ret = setup(reqs, &devComm, deviceCodeVersion);
    if (ret == ncclSuccess) ret = ncclGinDevCommFree(comm(), &devComm);
    return ret;
  }

  // Call the file-static setup directly: ncclGinDevCommSetup collapses every
  // per-backend failure into ncclInternalError, so a refusal that has to be seen
  // as its own status -- or that returns before the `end:` cleanup -- is only
  // observable here. Every other test in this suite goes through setup() above.
  ncclResult_t setupWithBackend(ncclDevCommRequirements const& reqs, ncclDevComm* devComm,
                                uint32_t deviceCodeVersion = NCCL_VERSION_CODE, int backendIdx = 0) {
    return ginDevCommSetupWithBackend(comm(), &reqs, devComm, deviceCodeVersion, &gin()->backends[backendIdx]);
  }

  const ncclGinConfig_t& lastConfig() { return fake_.createdConfigs.back(); }
};

// The plugin is told which device-code ABI to speak, derived from the device
// code's NCCL version and the backend's own compatibility table.
TEST_F(GinHostDevCommSetupMicrotest, DevCommSetup_DeviceCodeVersionAndBackendType_SelectTheBackendAbiVersion) {
  struct Case {
    const char* label;
    ncclGinType_t ginType;
    uint32_t deviceCodeVersion;
    int expectedBackendVersion;
  };
  // Proxy is the only backend with more than one revision on both sides of a
  // boundary, so it carries the boundary cases.
  const Case cases[] = {
    {"proxy below the first revision", NCCL_GIN_TYPE_PROXY, NCCL_VERSION(2, 30, 2), 0},
    {"proxy at the first revision", NCCL_GIN_TYPE_PROXY, NCCL_VERSION(2, 30, 3), 1},
    {"proxy at the second revision", NCCL_GIN_TYPE_PROXY, NCCL_VERSION(2, 30, 5), 2},
    {"proxy at the third revision", NCCL_GIN_TYPE_PROXY, NCCL_VERSION(2, 32, 0), 3},
    {"gdaki tops out at its last revision", NCCL_GIN_TYPE_GDAKI, NCCL_VERSION(2, 32, 0), 2},
    {"gpi tops out at its last revision", NCCL_GIN_TYPE_GPI, NCCL_VERSION(2, 32, 0), 1},
    {"efa gda at its first revision", NCCL_GIN_TYPE_EFA_GDA, NCCL_VERSION(2, 31, 0), 1},
    // AMD device-initiated backends ignore the host-provided version entirely.
    {"rocshmem gda is version-less", NCCL_GIN_TYPE_ROCSHMEM_GDA, NCCL_VERSION(2, 32, 0), 0},
    {"anvil sdma is version-less", NCCL_GIN_TYPE_ANVIL_SDMA, NCCL_VERSION(2, 32, 0), 0},
  };

  auto reqs = proxyReqs();
  reqs.ginContextCount = 1;
  for (const Case& c : cases) {
    SCOPED_TRACE(c.label);
    gin()->backends[0].ginType = c.ginType;
    reqs.ginType = c.ginType;  // so backend selection lands on the case's backend
    ASSERT_EQ(ncclSuccess, setupAndFree(reqs, c.deviceCodeVersion));
    EXPECT_EQ(c.expectedBackendVersion, lastConfig().backendVersion);
  }
}

// A backend whose type has no compatibility table cannot be configured at all.
TEST_F(GinHostDevCommSetupMicrotest, SetupWithBackend_BackendTypeWithoutAVersionTable_ReturnsInternalError) {
  gin()->backends[0].ginType = NCCL_GIN_TYPE_NONE;  // the sentinel, never a real backend
  auto reqs = proxyReqs();
  EXPECT_EQ(ncclInternalError, setupWithBackend(reqs, &devComm_));
  EXPECT_EQ(0, fake_.createContextCalls);
  // This is the one refusal that returns before the `end:` label, so unlike
  // every other failure it leaves the half-filled devComm fields alone. Nothing
  // asserts them here: the next candidate backend overwrites them, and pinning
  // the current values would make that inconsistency the specification.
}

// Contexts are spread evenly over the connections, so the requested count is
// rounded up to a whole number per connection.
TEST_F(GinHostDevCommSetupMicrotest, DevCommSetup_ContextCountNotAMultipleOfTheConnections_RoundsUpToWholeConnections) {
  gin()->backends[0].ginCommCount = 4;
  for (int i = 1; i < 4; i++) gin()->backends[0].ginComms[i] = fake_.collCommHandle(i);

  auto reqs = proxyReqs();
  reqs.ginContextCount = 5;
  ASSERT_EQ(ncclSuccess, setup(reqs, &devComm_));

  EXPECT_EQ(8u, devComm_.ginContextCount);  // 5 rounded up to a multiple of 4
  EXPECT_EQ(2, lastConfig().nContexts);     // ... which is 2 per connection
  EXPECT_EQ(4, devComm_.ginConnectionCount);
}

// Exclusive contexts are the point of the flag: a second request gets contexts
// of its own from the plugin rather than being handed the first devComm's.
TEST_F(GinHostDevCommSetupMicrotest, DevCommSetup_ExclusiveContextsRequestedTwice_GivesEachDevCommItsOwnContexts) {
  auto reqs = proxyReqs();
  reqs.ginContextCount = 1;
  reqs.ginExclusiveContexts = true;

  ncclDevComm first{};
  ASSERT_EQ(ncclSuccess, setup(reqs, &first));
  ASSERT_EQ(ncclSuccess, setup(reqs, &devComm_));

  EXPECT_EQ(2, fake_.createContextCalls);
  EXPECT_EQ(1u, devComm_.ginContextCount);
  EXPECT_NE(first.ginHandles[0], devComm_.ginHandles[0]);

  EXPECT_EQ(ncclSuccess, ncclGinDevCommFree(comm(), &first));
}

// The traffic class in the requirements is what the plugin is configured with.
TEST_F(GinHostDevCommSetupMicrotest, DevCommSetup_TrafficClassInTheRequirements_PassesItToThePlugin) {
  comm_->config.trafficClass = 7;
  auto reqs = proxyReqs();
  reqs.ginTrafficClass = 3;

  ASSERT_EQ(ncclSuccess, setupAndFree(reqs, NCCL_VERSION_CODE));
  EXPECT_EQ(3, lastConfig().trafficClass);
}

// With no traffic class of its own the request inherits the comm's.
TEST_F(GinHostDevCommSetupMicrotest, DevCommSetup_TrafficClassUnset_FallsBackToTheCommConfig) {
  comm_->config.trafficClass = 7;
  auto reqs = proxyReqs();
  reqs.ginTrafficClass = NCCL_CONFIG_UNDEF_INT;

  ASSERT_EQ(ncclSuccess, setupAndFree(reqs, NCCL_VERSION_CODE));
  EXPECT_EQ(7, lastConfig().trafficClass);
}

// Signal, counter and queue-depth counts are passed through, and legacy signals
// default to the strength the requirements asked for.
TEST_F(GinHostDevCommSetupMicrotest, DevCommSetup_SignalCounterAndQueueDepthRequested_ReachTheDevCommAndThePlugin) {
  auto reqs = proxyReqs();
  reqs.ginSignalCount = 5;
  reqs.ginCounterCount = 6;
  reqs.ginQueueDepth = 64;
  reqs.ginStrongSignalsRequired = true;

  ASSERT_EQ(ncclSuccess, setup(reqs, &devComm_));

  EXPECT_EQ(5, devComm_.ginSignalCount);
  EXPECT_EQ(6, devComm_.ginCounterCount);
  EXPECT_TRUE(devComm_.ginStrongLegacySignals);
  EXPECT_EQ(5, lastConfig().nSignals);
  EXPECT_EQ(6, lastConfig().nCounters);
  EXPECT_EQ(64, lastConfig().queueDepth);
}

// A rail request takes its stride from the rail team.
TEST_F(GinHostDevCommSetupMicrotest, DevCommSetup_RailRequestOnAFullyConnectedComm_UsesTheRailStride) {
  g_railStride = 2;
  auto reqs = proxyReqs();
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;

  ASSERT_EQ(ncclSuccess, setup(reqs, &devComm_));

  EXPECT_EQ(2, devComm_.ginContextStride);
  EXPECT_EQ(1, devComm_.ginConnectionStride);  // the comm itself is FULL-connected
  EXPECT_EQ(2, lastConfig().rankStride);
}

// A rail-connected comm's connections already span whole hosts, so the config
// stride is expressed relative to that.
TEST_F(GinHostDevCommSetupMicrotest, DevCommSetup_CustomStrideOnARailConnectedComm_ExpressesTheStrideRelativeToTheConnection) {
  gin()->ginConnectionType = NCCL_GIN_CONNECTION_RAIL;  // connectedStride = ranks per host
  comm_->contiguousRanksPerHost = 2;
  g_railStride = 4;

  auto reqs = proxyReqs();
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_CUSTOM_STRIDE;
  reqs.ginCustomStride = 4;
  ASSERT_EQ(ncclSuccess, setup(reqs, &devComm_));

  EXPECT_EQ(2, devComm_.ginConnectionStride);
  EXPECT_EQ(4, devComm_.ginContextStride);
  EXPECT_EQ(2, lastConfig().rankStride);  // 4 world ranks / 2 ranks per connection
}

// A zero stride would make every rank its own peer; it is rejected with the
// hint to disable GIN instead.
TEST_F(GinHostDevCommSetupMicrotest, SetupWithBackend_ZeroCustomStride_ReturnsInvalidUsage) {
  auto reqs = proxyReqs();
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_CUSTOM_STRIDE;
  reqs.ginCustomStride = 0;

  EXPECT_EQ(ncclInvalidUsage, setupWithBackend(reqs, &devComm_));
  EXPECT_EQ(0, fake_.createContextCalls);
}

// Hierarchical barriers assume GIN reaches at least the rail team, so a wider
// stride than the rail's is rejected.
TEST_F(GinHostDevCommSetupMicrotest, SetupWithBackend_CustomStrideWiderThanTheRailTeam_ReturnsInvalidUsage) {
  g_railStride = 2;
  auto reqs = proxyReqs();
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_CUSTOM_STRIDE;
  reqs.ginCustomStride = 4;

  EXPECT_EQ(ncclInvalidUsage, setupWithBackend(reqs, &devComm_));
  EXPECT_EQ(0, fake_.createContextCalls);
}

// The requested stride has to be reachable by stepping whole connections.
TEST_F(GinHostDevCommSetupMicrotest, SetupWithBackend_CustomStrideNotAMultipleOfTheConnectedStride_ReturnsInvalidUsage) {
  gin()->ginConnectionType = NCCL_GIN_CONNECTION_RAIL;  // connectedStride = 2 below
  comm_->contiguousRanksPerHost = 2;
  g_railStride = 4;

  auto reqs = proxyReqs();
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_CUSTOM_STRIDE;
  reqs.ginCustomStride = 3;

  EXPECT_EQ(ncclInvalidUsage, setupWithBackend(reqs, &devComm_));
  EXPECT_EQ(0, fake_.createContextCalls);
}

// A rejected setup leaves no GIN state behind on the devComm, so the caller can
// retry with another backend against the same object.
TEST_F(GinHostDevCommSetupMicrotest, SetupWithBackend_SetupFails_ClearsTheGinFieldsOfTheDevComm) {
  auto reqs = proxyReqs();
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_CUSTOM_STRIDE;
  reqs.ginCustomStride = 0;

  devComm_.ginConnectionCount = 7;  // stale values a failed setup must not keep
  devComm_.ginHandles[0] = reinterpret_cast<void*>(0x99);
  devComm_.ginNetDeviceTypes[0] = NCCL_NET_DEVICE_GIN_PROXY;

  ASSERT_EQ(ncclInvalidUsage, setupWithBackend(reqs, &devComm_));

  EXPECT_EQ(0, devComm_.ginConnectionCount);
  EXPECT_EQ(0u, devComm_.ginContextCount);
  EXPECT_EQ(0, devComm_.ginConnectionStride);
  EXPECT_EQ(0u, devComm_.ginConnectionStride_rcp32);
  EXPECT_EQ(0, devComm_.ginContextStride);
  EXPECT_EQ(nullptr, devComm_.ginHandles[0]);
  EXPECT_EQ(0, devComm_.ginNetDeviceTypes[0]);
}

// Each half of the plugin's createContext contract is checked, because a NULL
// anywhere here would surface as a device-side fault instead.
TEST_F(GinHostDevCommSetupMicrotest, DevCommSetup_IncompleteContextFromThePlugin_ReturnsInternalError) {
  const BadContext cases[] = {BadContext::NullGinCtx, BadContext::NullDevHandle, BadContext::NullHandle};
  auto reqs = proxyReqs();
  for (BadContext bad : cases) {
    SCOPED_TRACE(static_cast<int>(bad));
    fake_.badContext = bad;
    ncclDevComm devComm{};
    EXPECT_EQ(ncclInternalError, setup(reqs, &devComm));
    EXPECT_EQ(0, devComm.ginConnectionCount);
  }
}

// A context that was created before the failure is destroyed again rather than
// left behind in the plugin.
TEST_F(GinHostDevCommSetupMicrotest, SetupWithBackend_SecondCreateContextFails_DestroysTheEarlierContext) {
  gin()->backends[0].ginCommCount = 2;
  gin()->backends[0].ginComms[1] = fake_.collCommHandle(1);
  fake_.failCreateContext = {ncclSystemError, 2};

  auto reqs = proxyReqs();
  reqs.ginContextCount = 2;
  EXPECT_EQ(ncclSystemError, setupWithBackend(reqs, &devComm_));
  EXPECT_EQ(1, fake_.destroyCalls.load());
  EXPECT_EQ(0, devComm_.ginConnectionCount);
}

// The producer half of the proxy-affinity pin (AICOMRCCL-1859): the branch that
// first spawns the progress threads stashes comm->cpuAffinity into
// ginState->cpuAffinity, which is the only mask the workers above ever see.
TEST_F(GinHostDevCommSetupMicrotest, DevCommSetup_SpawningTheProxyThreads_StashesCommAffinityForTheWorkers) {
  CPU_ZERO(&comm_->cpuAffinity);
  CPU_SET(5, &comm_->cpuAffinity);  // a distinctive mask, to spot the copy
  g_ncclOsCpuCountValue = 1;        // so the spawned worker takes the pin branch

  ASSERT_EQ(ncclSuccess, setup(proxyReqs(), &devComm_));
  ASSERT_TRUE(gin()->proxyThreadsCreated);  // it took the spawn branch
  EXPECT_TRUE(CPU_EQUAL(&comm_->cpuAffinity, &gin()->cpuAffinity));  // stashed verbatim

  // The worker writes the mask from its own thread, so read it only once joined.
  joinProgressThreads();
  ASSERT_EQ(1u, g_ncclOsSetAffinityMasks.size());
  EXPECT_TRUE(CPU_EQUAL(&comm_->cpuAffinity, &g_ncclOsSetAffinityMasks[0]));
}

// Progress threads are started once; a later devComm joins the existing list
// under the writer lock instead of spawning a second set.
TEST_F(GinHostDevCommSetupMicrotest, DevCommSetup_ProgressThreadsAlreadyRunning_LinksTheDevCommWithoutSpawningMore) {
  auto reqs = proxyReqs();
  ncclDevComm first{};
  ncclDevComm second{};
  ASSERT_EQ(ncclSuccess, setup(reqs, &first));
  ASSERT_TRUE(gin()->proxyThreadsCreated);
  ASSERT_EQ(ncclSuccess, setup(reqs, &second));
  // The third devComm has to walk past both of its predecessors to find the tail.
  ASSERT_EQ(ncclSuccess, setup(reqs, &devComm_));

  EXPECT_FALSE(gin()->thread[1].joinable());  // still just the one worker
  EXPECT_FALSE(gin()->writePending.load());   // the writer lock was released
  int listLength = 0;
  for (auto* dc = gin()->devComms; dc != nullptr; dc = dc->next) listLength++;
  EXPECT_EQ(3, listLength);

  EXPECT_EQ(ncclSuccess, ncclGinDevCommFree(comm(), &first));
  EXPECT_EQ(ncclSuccess, ncclGinDevCommFree(comm(), &second));
}

////////////////////////////////////////////////////////////////////////////////
// ncclGinDevCommFree -- lookup failures

// The devComms this suite frees are built by ncclGinDevCommSetup, so the list a
// free walks is the one production linked rather than one the test modelled.
class GinHostDevCommFreeMicrotest : public GinHostFixture {
 protected:
  void SetUp() override {
    GinHostFixture::SetUp();
    ASSERT_EQ(ncclSuccess, connectOnce());
    gin()->proxyThreadStopSignal.store(true);  // any spawned worker exits immediately
  }

  ncclDevComm makeDevComm() {
    ncclDevComm devComm{};
    auto reqs = proxyReqs();
    reqs.ginContextCount = gin()->backends[0].ginCommCount;
    EXPECT_EQ(ncclSuccess, ncclGinDevCommSetup(comm(), &reqs, &devComm, NCCL_VERSION_CODE));
    return devComm;
  }
};

// Freeing against an empty list is an internal error, not a crash on a NULL
// list head.
TEST_F(GinHostDevCommFreeMicrotest, DevCommFree_EmptyDevCommList_ReturnsInternalError) {
  ASSERT_EQ(nullptr, gin()->devComms);
  ncclDevComm devComm{};
  EXPECT_EQ(ncclInternalError, ncclGinDevCommFree(comm(), &devComm));
}

// The devComm is looked up by its GIN handle; an unknown handle walks the whole
// list and then fails, leaving the devComms that are on it untouched.
TEST_F(GinHostDevCommFreeMicrotest, DevCommFree_UnknownHandle_ReturnsInternalErrorAndLeavesTheListIntact) {
  ncclDevComm first = makeDevComm();
  ncclDevComm second = makeDevComm();

  ncclDevComm devComm{};
  devComm.ginHandles[0] = reinterpret_cast<void*>(0x999);  // belongs to no devComm
  EXPECT_EQ(ncclInternalError, ncclGinDevCommFree(comm(), &devComm));
  EXPECT_EQ(0, fake_.destroyCalls.load());
  int listLength = 0;
  for (auto* dc = gin()->devComms; dc != nullptr; dc = dc->next) listLength++;
  EXPECT_EQ(2, listLength) << "a failed lookup unlinked a devComm";

  EXPECT_EQ(ncclSuccess, ncclGinDevCommFree(comm(), &first));
  EXPECT_EQ(ncclSuccess, ncclGinDevCommFree(comm(), &second));
}

// A plugin that fails to destroy a context surfaces that status to the caller,
// and the devComm stays off the list: the free is not retryable.
TEST_F(GinHostDevCommFreeMicrotest, DevCommFree_DestroyContextFails_PropagatesTheStatusAndStillUnlinks) {
  ncclDevComm devComm = makeDevComm();
  fake_.failDestroyContext = {ncclSystemError, 1};

  EXPECT_EQ(ncclSystemError, ncclGinDevCommFree(comm(), &devComm));
  // The devComm was already unlinked, so the list -- not this test -- owns nothing.
  EXPECT_EQ(nullptr, gin()->devComms);
}

////////////////////////////////////////////////////////////////////////////////
// ncclGinHostFinalize

class GinHostFinalizeMicrotest : public GinHostFixture {};

// Finalizing a comm that never connected touches no plugin state.
TEST_F(GinHostFinalizeMicrotest, HostFinalize_CommNeverConnected_TouchesNoPluginState) {
  ASSERT_FALSE(gin()->connected);
  EXPECT_EQ(ncclSuccess, ncclGinHostFinalize(comm()));
  EXPECT_EQ(0, fake_.closeCollCalls);
}

// Finalize closes every connection and wipes the GIN state, so a later
// ncclGinConnectOnce starts from scratch.
TEST_F(GinHostFinalizeMicrotest, HostFinalize_ConnectedComm_ClosesEveryConnectionAndClearsTheState) {
  nthreadsParam_ = 2;
  ASSERT_EQ(ncclSuccess, connectOnce());
  ASSERT_EQ(2, gin()->backends[0].ginCommCount);

  ASSERT_EQ(ncclSuccess, ncclGinHostFinalize(comm()));

  EXPECT_EQ(2, fake_.closeCollCalls);
  EXPECT_FALSE(gin()->connected);
  EXPECT_EQ(0, gin()->numActiveBackends);
  restoreGinStateAfterFinalize();
}

// A progress-thread slot that was never spawned is skipped rather than joined.
TEST_F(GinHostFinalizeMicrotest, HostFinalize_ProgressThreadSlotNeverSpawned_SkipsTheJoin) {
  ASSERT_EQ(ncclSuccess, connectOnce());
  gin()->proxyThreadsCreated = true;  // ... but no std::thread was ever started
  ASSERT_FALSE(gin()->thread[0].joinable());

  EXPECT_EQ(ncclSuccess, ncclGinHostFinalize(comm()));
  restoreGinStateAfterFinalize();
}

// A connection slot that is already empty is skipped instead of being closed a
// second time.
TEST_F(GinHostFinalizeMicrotest, HostFinalize_ConnectionSlotAlreadyClosed_SkipsIt) {
  nthreadsParam_ = 2;
  ASSERT_EQ(ncclSuccess, connectOnce());
  gin()->backends[0].ginComms[1] = nullptr;

  ASSERT_EQ(ncclSuccess, ncclGinHostFinalize(comm()));
  EXPECT_EQ(1, fake_.closeCollCalls);
  restoreGinStateAfterFinalize();
}

// A close that fails surfaces to the caller instead of being swallowed by the
// teardown path, and finalize stops there rather than wiping the state behind
// a connection it could not release.
TEST_F(GinHostFinalizeMicrotest, HostFinalize_CloseCollFails_PropagatesTheStatusAndLeavesTheStateIntact) {
  nthreadsParam_ = 2;  // a second connection, to show the walk stopped
  ASSERT_EQ(ncclSuccess, connectOnce());
  ASSERT_EQ(2, gin()->backends[0].ginCommCount);
  fake_.failCloseColl = {ncclSystemError, 1};

  EXPECT_EQ(ncclSystemError, ncclGinHostFinalize(comm()));

  EXPECT_EQ(1, fake_.closeCollCalls);
  EXPECT_NE(nullptr, gin()->backends[0].ginComms[0]);  // the slot is not cleared
  EXPECT_TRUE(gin()->connected);                       // ... and the state is not wiped
  EXPECT_EQ(1, gin()->numActiveBackends);
}

// HostFinalize joins every progress thread before it returns. A worker held
// inside ginProgress makes that join visible: deleting the join would let
// finalize return while the worker is still in the call. The post-join memset
// of ginState cannot be used instead, because it stops progress even if the
// threads were never joined.
TEST_F(GinHostFinalizeMicrotest, HostFinalize_WorkerInsideGinProgress_WaitsForEveryProgressThread) {
  nthreadsParam_ = 2;
  ASSERT_EQ(ncclSuccess, connectOnce());
  ncclDevComm devComm{};
  auto reqs = proxyReqs();
  reqs.ginContextCount = gin()->backends[0].ginCommCount;
  ASSERT_EQ(ncclSuccess, ncclGinDevCommSetup(comm(), &reqs, &devComm, NCCL_VERSION_CODE));
  ASSERT_TRUE(gin()->thread[0].joinable());
  ASSERT_TRUE(gin()->thread[1].joinable());

  struct ncclGinStateDevComm* dc = gin()->devComms;
  gin()->connected = true;
  fake_.holdProgress.store(1, std::memory_order_release);
  ASSERT_TRUE(waitUntil([&] { return fake_.progressHolders.load(std::memory_order_acquire) > 0; }))
      << "no progress thread entered ginProgress";

  std::atomic<bool> finalizeDone{false};
  ncclResult_t finalizeSt = ncclInternalError;
  std::thread fin([&] {
    finalizeSt = ncclGinHostFinalize(comm());
    finalizeDone.store(true, std::memory_order_release);
  });
  // Release the held worker and join finalize on every exit, including a fatal
  // ASSERT.
  ScopeExit finalizeGuard([&] {
    fake_.holdProgress.store(0, std::memory_order_release);
    if (fin.joinable()) fin.join();
    restoreGinStateAfterFinalize();
    std::free(dc);
  });

  ASSERT_TRUE(waitUntil([&] { return gin()->proxyThreadStopSignal.load(); }))
      << "HostFinalize did not reach the progress-thread join";
  EXPECT_FALSE(finalizeDone.load(std::memory_order_acquire))
      << "HostFinalize returned while a worker was still inside ginProgress";
  fake_.holdProgress.store(0, std::memory_order_release);
  ASSERT_TRUE(waitUntil([&] { return finalizeDone.load(std::memory_order_acquire); }))
      << "HostFinalize did not return after the worker left ginProgress";
  EXPECT_EQ(ncclSuccess, finalizeSt);
}

////////////////////////////////////////////////////////////////////////////////
// ncclGinRegister / ncclGinDeregister

class GinHostRegisterMicrotest : public GinHostFixture {
 protected:
  void* hostWins_[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS] = {};
  ncclGinWindow_t devWins_[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS] = {};
  char buffer_[64] = {};

  void SetUp() override {
    GinHostFixture::SetUp();
    configureBackend(0, 2);
  }

  // One backend with `ginCommCount` connections, each with its own collComm so a
  // test can tell which connection a registration landed on.
  void configureBackend(int backendIdx, int ginCommCount) {
    auto& backend = gin()->backends[backendIdx];
    backend.ginType = NCCL_GIN_TYPE_PROXY;
    backend.ncclGin = &vtable_;
    backend.ginCommCount = ginCommCount;
    for (int i = 0; i < ginCommCount; i++) {
      backend.ginComms[i] = collCommFor(backendIdx, i);
      backend.ginProps[i].ptrSupport = NCCL_PTR_CUDA;
    }
  }

  // A tagged collComm handle per (backend, connection), from the fake's own
  // pool, so the plugin entry points recover the fake from it the same way they
  // would from a handle connect() handed out.
  void* collCommFor(int backendIdx, int commIdx) {
    return fake_.collCommHandle(backendIdx * NCCL_GIN_MAX_CONNECTIONS + commIdx);
  }

  ncclResult_t registerWindow(int winFlags = 0, bool multiSegment = false, int memType = NCCL_PTR_CUDA) {
    return ncclGinRegister(comm(), buffer_, sizeof(buffer_), hostWins_, devWins_, winFlags, multiSegment, memType);
  }
};

// Every connection of every active backend gets its own registration, and the
// handles land in that connection's slot.
TEST_F(GinHostRegisterMicrotest, Register_MultipleActiveBackends_RegistersOnEveryConnectionOfEachOne) {
  gin()->numActiveBackends = 2;
  configureBackend(1, 1);

  ASSERT_EQ(ncclSuccess, registerWindow());

  ASSERT_EQ(3u, fake_.regMrCalls.size());
  EXPECT_EQ(collCommFor(0, 0), fake_.regMrCalls[0].collComm);
  EXPECT_EQ(collCommFor(0, 1), fake_.regMrCalls[1].collComm);
  EXPECT_EQ(collCommFor(1, 0), fake_.regMrCalls[2].collComm);
  EXPECT_EQ(buffer_, fake_.regMrCalls[0].address);
  EXPECT_EQ(sizeof(buffer_), fake_.regMrCalls[0].size);
  // Backend 1's single connection is addressed by its own block of slots.
  EXPECT_NE(nullptr, hostWins_[0]);
  EXPECT_NE(nullptr, hostWins_[1]);
  EXPECT_NE(nullptr, hostWins_[NCCL_GIN_MAX_CONNECTIONS]);
  EXPECT_NE(nullptr, devWins_[NCCL_GIN_MAX_CONNECTIONS]);
  EXPECT_EQ(nullptr, hostWins_[2]);  // unused connections stay empty
}

// The memory type the caller registered is handed to the plugin unchanged.
TEST_F(GinHostRegisterMicrotest, Register_MemoryTypeGiven_PassesItToThePluginUnchanged) {
  ASSERT_EQ(ncclSuccess, registerWindow(/*winFlags=*/0, /*multiSegment=*/false, NCCL_PTR_HOST));
  ASSERT_FALSE(fake_.regMrCalls.empty());
  EXPECT_EQ(NCCL_PTR_HOST, fake_.regMrCalls[0].memType);
}

// A strictly-ordered window must be registered with strong ordering forced on
// the NIC.
TEST_F(GinHostRegisterMicrotest, Register_StrictOrderingWindow_ForcesStrongOrdering) {
  ASSERT_EQ(ncclSuccess, registerWindow(NCCL_WIN_STRICT_ORDERING));
  ASSERT_FALSE(fake_.regMrCalls.empty());
  for (const auto& call : fake_.regMrCalls) EXPECT_EQ(NCCL_NET_MR_FLAG_FORCE_SO, call.mrFlags);
}

// Without that flag no ordering constraint is imposed.
TEST_F(GinHostRegisterMicrotest, Register_OrdinaryWindow_RegistersWithoutMrFlags) {
  ASSERT_EQ(ncclSuccess, registerWindow());
  ASSERT_FALSE(fake_.regMrCalls.empty());
  for (const auto& call : fake_.regMrCalls) EXPECT_EQ(0u, call.mrFlags);
}

// A multi-segment buffer needs DMABUF on every connection; one connection
// without it rejects the whole registration before anything is registered.
TEST_F(GinHostRegisterMicrotest, Register_MultiSegmentWithoutDmabufOnEveryConnection_ReturnsInvalidArgumentBeforeRegistering) {
  gin()->backends[0].ginProps[0].ptrSupport |= NCCL_PTR_DMABUF;  // only the first connection

  EXPECT_EQ(ncclInvalidArgument, registerWindow(/*winFlags=*/0, /*multiSegment=*/true));
  EXPECT_TRUE(fake_.regMrCalls.empty());
}

// With DMABUF everywhere the multi-segment registration proceeds normally.
TEST_F(GinHostRegisterMicrotest, Register_MultiSegmentWithDmabufEverywhere_RegistersEveryConnection) {
  for (int i = 0; i < 2; i++) gin()->backends[0].ginProps[i].ptrSupport |= NCCL_PTR_DMABUF;

  EXPECT_EQ(ncclSuccess, registerWindow(/*winFlags=*/0, /*multiSegment=*/true));
  EXPECT_EQ(2u, fake_.regMrCalls.size());
}

// A plugin that reports success but hands back no window is treated as a
// failure rather than storing a NULL window.
TEST_F(GinHostRegisterMicrotest, Register_PluginReturnsNoWindow_ReturnsSystemError) {
  fake_.regMrSymReturnsNullWindow = true;
  EXPECT_EQ(ncclSystemError, registerWindow());
  EXPECT_EQ(1u, fake_.regMrCalls.size());  // stops at the first bad connection
}

// A failing registration surfaces the plugin's status and stops there: the
// remaining connections are left unregistered rather than half-populated.
TEST_F(GinHostRegisterMicrotest, Register_RegistrationFails_PropagatesTheStatusAndStopsAtThatConnection) {
  void* const poison = reinterpret_cast<void*>(0xDEAD);
  for (auto& win : hostWins_) win = poison;
  fake_.failRegMrSym = {ncclInternalError, 1};

  EXPECT_EQ(ncclInternalError, registerWindow());

  EXPECT_EQ(1u, fake_.regMrCalls.size());  // the second connection is never asked
  EXPECT_EQ(poison, hostWins_[1]);         // ... so its slot is untouched
}

// Deregistration mirrors registration: one call per populated slot.
TEST_F(GinHostRegisterMicrotest, Deregister_PopulatedSlots_DeregistersEachOne) {
  ASSERT_EQ(ncclSuccess, registerWindow());
  void* const win0 = hostWins_[0];
  void* const win1 = hostWins_[1];

  ASSERT_EQ(ncclSuccess, ncclGinDeregister(comm(), hostWins_));

  ASSERT_EQ(2u, fake_.deregMrCalls.size());
  EXPECT_EQ(std::make_pair(collCommFor(0, 0), win0), fake_.deregMrCalls[0]);
  EXPECT_EQ(std::make_pair(collCommFor(0, 1), win1), fake_.deregMrCalls[1]);
}

// A connection that was never registered has an empty slot, which is skipped
// rather than deregistered as a NULL window.
TEST_F(GinHostRegisterMicrotest, Deregister_EmptySlot_IsNotDeregistered) {
  ASSERT_EQ(ncclSuccess, registerWindow());
  void* const win1 = hostWins_[1];
  hostWins_[0] = nullptr;

  ASSERT_EQ(ncclSuccess, ncclGinDeregister(comm(), hostWins_));

  ASSERT_EQ(1u, fake_.deregMrCalls.size());
  EXPECT_EQ(win1, fake_.deregMrCalls[0].second);
}

// A failing deregistration surfaces the plugin's status and stops the walk
// there, same as registration.
TEST_F(GinHostRegisterMicrotest, Deregister_DeregistrationFails_PropagatesTheStatusAndStopsAtThatSlot) {
  ASSERT_EQ(ncclSuccess, registerWindow());
  fake_.failDeregMrSym = {ncclSystemError, 1};

  EXPECT_EQ(ncclSystemError, ncclGinDeregister(comm(), hostWins_));

  EXPECT_EQ(1u, fake_.deregMrCalls.size());  // the second slot is left registered
}

////////////////////////////////////////////////////////////////////////////////
// ncclGinQueryLastError

// Like the free suite, the list walked here is built by ncclGinDevCommSetup --
// two GIN connections per devComm, so "asked every context" and "stopped early"
// are distinguishable.
class GinHostQueryLastErrorMicrotest : public GinHostFixture {
 protected:
  std::vector<std::unique_ptr<ncclDevComm>> devComms_;

  void SetUp() override {
    GinHostFixture::SetUp();
    nthreadsParam_ = 2;
    ASSERT_EQ(ncclSuccess, connectOnce());
    ASSERT_EQ(2, gin()->backends[0].ginCommCount);
    gin()->proxyThreadStopSignal.store(true);  // any spawned worker exits immediately
  }

  void addDevComm() {
    devComms_.push_back(std::make_unique<ncclDevComm>());
    auto reqs = proxyReqs();
    reqs.ginContextCount = gin()->backends[0].ginCommCount;
    ASSERT_EQ(ncclSuccess, ncclGinDevCommSetup(comm(), &reqs, devComms_.back().get(), NCCL_VERSION_CODE));
  }

  void TearDown() override {
    for (auto& dc : devComms_) EXPECT_EQ(ncclSuccess, ncclGinDevCommFree(comm(), dc.get()));
    GinHostFixture::TearDown();
  }
};

// With nothing registered there is nothing to ask, and the answer is "no error".
TEST_F(GinHostQueryLastErrorMicrotest, QueryLastError_NoDevComms_ReportsNoError) {
  ASSERT_EQ(nullptr, gin()->devComms);
  bool hasError = true;
  EXPECT_EQ(ncclSuccess, ncclGinQueryLastError(gin(), &hasError));
  EXPECT_FALSE(hasError);
  EXPECT_EQ(0, fake_.queryCalls);
}

// Every context of every devComm is asked, including the ones that do not need
// proxy progress -- that is the only way a device-initiated backend reports.
TEST_F(GinHostQueryLastErrorMicrotest, QueryLastError_NoContextReportsAnError_AsksEveryContextOfEveryDevComm) {
  addDevComm();
  fake_.needsProxyProgress = false;  // a device-initiated devComm, never progressed
  addDevComm();

  bool hasError = true;
  EXPECT_EQ(ncclSuccess, ncclGinQueryLastError(gin(), &hasError));
  EXPECT_FALSE(hasError);
  EXPECT_EQ(4, fake_.queryCalls);  // 2 devComms x 2 connections
}

// The first context reporting an error ends the walk: the caller only needs to
// know that something failed.
TEST_F(GinHostQueryLastErrorMicrotest, QueryLastError_AContextReportsAnError_StopsAtThatContext) {
  addDevComm();
  addDevComm();
  fake_.queryErrorOnCall = 2;

  bool hasError = false;
  EXPECT_EQ(ncclSuccess, ncclGinQueryLastError(gin(), &hasError));
  EXPECT_TRUE(hasError);
  EXPECT_EQ(2, fake_.queryCalls);  // the remaining three contexts are never asked
}

// A query that fails outright is a different thing from a query that reports an
// error: the plugin's status surfaces, the walk stops, and the caller is not
// told an error was found.
TEST_F(GinHostQueryLastErrorMicrotest, QueryLastError_QueryFails_PropagatesTheStatusAndReportsNoError) {
  addDevComm();
  fake_.failQueryLastError = {ncclSystemError, 1};

  bool hasError = true;  // pre-poisoned, so "left false" is a real assertion
  EXPECT_EQ(ncclSystemError, ncclGinQueryLastError(gin(), &hasError));
  EXPECT_FALSE(hasError);
  EXPECT_EQ(1, fake_.queryCalls);  // the devComm's second connection is never asked
}

}  // namespace
