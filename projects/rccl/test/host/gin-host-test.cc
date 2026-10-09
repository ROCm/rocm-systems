/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/gin/gin_host.cc -- every function in the file
// (AICOMRCCL-2642). The suites, in file order:
//
//   GinHostTest                        GIN_PROXY_NTHREADS (NVIDIA/nccl#2279,
//                                      AICOMRCCL-2017): per-thread endpoint
//                                      assignment, ginCommCount bump,
//                                      writer-priority list lock, and
//                                      progress-thread lifecycle.
//   GinHostProxyAffinity*Microtest     proxy-thread CPU affinity
//                                      (AICOMRCCL-1859): the comm->cpuAffinity
//                                      stash and the pin each worker applies.
//   GinHostTypeQueryMicrotest          ncclGetGinType / ncclGetRailedGinType.
//   GinHostSignalRequestMicrotest      ncclGinValidateSignalRequest.
//   GinHostConnectOnceMicrotest        ncclGinConnectOnce refusals, plugin
//                                      failures, and the strided team.
//   GinHostBackendSelectMicrotest      ncclGinDevCommSetup backend selection.
//   GinHostDevCommSetupMicrotest       ginDevCommSetupWithBackend config,
//                                      strides, and failure cleanup.
//   GinHostDevCommFreeMicrotest        ncclGinDevCommFree lookup failures.
//   GinHostFinalizeMicrotest           ncclGinHostFinalize.
//   GinHostRegisterMicrotest           ncclGinRegister / ncclGinDeregister.
//   GinHostQueryLastErrorMicrotest     ncclGinQueryLastError.
//
// Own binary (rccl-UnitTestsMicroGinHost): gin-plugin-init-test.cc already defines
// ncclParamGinEnable in rccl-UnitTestsMicro, and gin_fakes.cc supplies
// ncclGinQueryLastError to rccl-UnitTestsMicroInit. This TU #includes the hipified
// gin_host.cc (GIN_HOST_CC_PATH) so file-scope helpers and ncclGinProgress are
// reachable. No GPU, no librccl.so, no network.

#include <gtest/gtest.h>

#include <sched.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
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

#include "fakes/nccl_fakes.h"

namespace {

// Scripted failure for one faked entry point: return `result` on call number
// `onCall` (1-based), or on every call when `onCall` is 0. Lets a test say "fail
// the second connection" without a bespoke counter per entry point.
struct ScriptedFailure {
  ncclResult_t result = ncclSuccess;
  int onCall = 0;

  ncclResult_t at(int callNumber) const {
    if (result == ncclSuccess) return ncclSuccess;
    if (onCall == 0 || onCall == callNumber) return result;
    return ncclSuccess;
  }
};

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
}

}  // namespace

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

ncclResult_t bootstrapAllGather(void*, void* allData, int size) {
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

ncclTeam_t ncclTeamWorld(ncclComm_t comm) {
  ncclTeam_t t{};
  t.nRanks = comm->nRanks;
  t.rank = comm->rank;
  t.stride = 1;
  return t;
}

ncclTeam_t ncclTeamRail(ncclComm_t) {
  ncclTeam_t t{};
  t.nRanks = g_railNRanks;
  t.rank = 0;
  t.stride = g_railStride;
  return t;
}

int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}

// Counts entries into ncclGinProgress. The progress loop itself has no counter
// while writePending is set, so tests use this to prove a worker started.
// ncclOsCpuCount is the first thing every progress thread calls, so it doubles as
// the entry counter and as the affinity seam: returning 0 (the default) makes the
// thread skip the pin, non-zero makes it apply ginState->cpuAffinity.
std::atomic<int> g_progressEntries{0};
std::atomic<int> g_osCpuCountValue{0};
// Every mask handed to ncclOsSetAffinity, in call order. Written from the spawned
// progress threads, so it is mutex-guarded; read only after those threads are joined.
std::mutex g_osAffinityMutex;
std::vector<ncclAffinity> g_osSetAffinityMasks;

void ResetAffinityFakes() {
  g_progressEntries.store(0);
  g_osCpuCountValue.store(0);
  std::lock_guard<std::mutex> lock(g_osAffinityMutex);
  g_osSetAffinityMasks.clear();
}

int ncclOsCpuCount(const ncclAffinity&) {
  g_progressEntries.fetch_add(1, std::memory_order_relaxed);
  return g_osCpuCountValue.load(std::memory_order_relaxed);
}
ncclResult_t ncclOsSetAffinity(const ncclAffinity& affinity) {
  std::lock_guard<std::mutex> lock(g_osAffinityMutex);
  g_osSetAffinityMasks.push_back(affinity);
  return ncclSuccess;
}
void ncclSetThreadName(std::thread&, const char*, ...) {}

#include "fakes/param_redirect.h"

#include GIN_HOST_CC_PATH

namespace {

constexpr auto kWait = std::chrono::milliseconds(2000);

template <typename Pred>
bool waitUntil(Pred pred, std::chrono::milliseconds budget = kWait) {
  const auto deadline = std::chrono::steady_clock::now() + budget;
  while (!pred()) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::yield();
  }
  return true;
}

struct FakeSlot {
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

  static FakeGin*& currentPtr() {
    static FakeGin* p = nullptr;
    return p;
  }
  static FakeGin& current() { return *currentPtr(); }
  static void setCurrent(FakeGin* p) { currentPtr() = p; }

  FakeSlot* slotFor(void* ginCtx) {
    return static_cast<FakeSlot*>(ginCtx);
  }

  static ncclResult_t Devices(int* ndev) {
    FakeGin& self = current();
    ++self.devicesCalls;
    if (ndev) *ndev = self.ndev;
    return self.failDevices.at(self.devicesCalls);
  }
  // A plugin that fails hands back no listenComm, so production's fail path sees
  // the NULL slot and skips closeListen for it.
  static ncclResult_t Listen(void*, int, void*, void** listenComm) {
    FakeGin& self = current();
    ++self.listenCalls;
    ncclResult_t ret = self.failListen.at(self.listenCalls);
    if (ret == ncclSuccess) *listenComm = reinterpret_cast<void*>(0x11);
    return ret;
  }
  static ncclResult_t GetProperties(int, ncclNetProperties_t* props) {
    FakeGin& self = current();
    ++self.getPropertiesCalls;
    if (props) std::memset(props, 0, sizeof(*props));
    return self.failGetProperties.at(self.getPropertiesCalls);
  }
  static ncclResult_t Connect(void*, void**, int nRanks, int rank, void*, void** collComm) {
    FakeGin& self = current();
    ++self.connectCalls;
    self.lastConnectNRanks = nRanks;
    self.lastConnectRank = rank;
    ncclResult_t ret = self.failConnect.at(self.connectCalls);
    if (ret == ncclSuccess) *collComm = reinterpret_cast<void*>(0x22);
    return ret;
  }
  static ncclResult_t CloseListen(void*) {
    FakeGin& self = current();
    ++self.closeListenCalls;
    return self.failCloseListen.at(self.closeListenCalls);
  }
  static ncclResult_t CloseColl(void*) {
    FakeGin& self = current();
    ++self.closeCollCalls;
    return self.failCloseColl.at(self.closeCollCalls);
  }

  static ncclResult_t CreateContext(void*, ncclGinConfig_t* config, void** ginCtx,
                                    ncclNetDeviceHandle_t** devHandle) {
    FakeGin& self = current();
    ++self.createContextCalls;
    if (config) self.createdConfigs.push_back(*config);
    ncclResult_t ret = self.failCreateContext.at(self.createContextCalls);
    if (ret != ncclSuccess) return ret;
    auto slot = std::make_unique<FakeSlot>();
    slot->idx = static_cast<int>(self.slots.size());
    slot->handle.netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
    slot->handle.handle = self.badContext == BadContext::NullHandle ? nullptr : slot.get();
    slot->handle.needsProxyProgress = self.needsProxyProgress ? 1 : 0;
    *devHandle = self.badContext == BadContext::NullDevHandle ? nullptr : &slot->handle;
    *ginCtx = self.badContext == BadContext::NullGinCtx ? nullptr : slot.get();
    self.slots.push_back(std::move(slot));
    return ncclSuccess;
  }

  static ncclResult_t DestroyContext(void* ginCtx) {
    FakeGin& self = current();
    ++self.destroyCalls;
    self.destroyed.push_back(ginCtx);
    return self.failDestroyContext.at(self.destroyCalls);
  }

  static ncclResult_t RegMrSym(void* collComm, void* address, size_t size, int memType, uint64_t mrFlags,
                               void** mhandle, void** ginHandle) {
    FakeGin& self = current();
    self.regMrCalls.push_back(RegMrCall{collComm, address, size, memType, mrFlags});
    const int callNumber = static_cast<int>(self.regMrCalls.size());
    ncclResult_t ret = self.failRegMrSym.at(callNumber);
    if (ret != ncclSuccess) return ret;
    // Distinct per call so a test can tell the slots apart.
    if (mhandle) {
      *mhandle = self.regMrSymReturnsNullWindow ? nullptr
                                                : reinterpret_cast<void*>(0x1000 + (uintptr_t)callNumber);
    }
    if (ginHandle) *ginHandle = reinterpret_cast<void*>(0x2000 + (uintptr_t)callNumber);
    return ncclSuccess;
  }

  static ncclResult_t DeregMrSym(void* collComm, void* mhandle) {
    FakeGin& self = current();
    self.deregMrCalls.emplace_back(collComm, mhandle);
    return self.failDeregMrSym.at(static_cast<int>(self.deregMrCalls.size()));
  }

  static ncclResult_t Progress(void* ginCtx) {
    FakeGin& self = current();
    if (self.holdProgress.load(std::memory_order_acquire) != 0) {
      self.progressHolders.fetch_add(1, std::memory_order_release);
      while (self.holdProgress.load(std::memory_order_acquire) != 0) {
        std::this_thread::yield();
      }
      self.progressHolders.fetch_sub(1, std::memory_order_release);
    }
    FakeSlot* slot = self.slotFor(ginCtx);
    slot->progressCalls.fetch_add(1);
    self.totalProgressCalls.fetch_add(1);
    return slot->progressResult;
  }

  static ncclResult_t QueryLastError(void*, bool* hasError) {
    FakeGin& self = current();
    ++self.queryCalls;
    if (hasError) *hasError = (self.queryErrorOnCall != 0 && self.queryCalls == self.queryErrorOnCall);
    return self.failQueryLastError.at(self.queryCalls);
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

class GinHostTest : public ::testing::Test {
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
    ResetAffinityFakes();
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
    gs->backends[0].ginInstance = reinterpret_cast<void*>(0x33);
  }

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

  // Call the file-static setup directly: ncclGinDevCommSetup collapses every
  // per-backend failure into ncclInternalError, so the status a single backend
  // produced is only observable here.
  ncclResult_t setupWithBackend(ncclDevCommRequirements const& reqs, ncclDevComm* devComm,
                                uint32_t deviceCodeVersion = NCCL_VERSION_CODE, int backendIdx = 0) {
    return ginDevCommSetupWithBackend(comm(), &reqs, devComm, deviceCodeVersion, &gin()->backends[backendIdx]);
  }

  // One setup plus the matching free, so a table-driven test can loop over
  // variants without leaking a context per iteration.
  ncclResult_t setupAndFree(ncclDevCommRequirements const& reqs, uint32_t deviceCodeVersion) {
    ncclDevComm devComm{};
    ncclResult_t ret = setupWithBackend(reqs, &devComm, deviceCodeVersion);
    if (ret == ncclSuccess) ret = ncclGinDevCommFree(comm(), &devComm);
    return ret;
  }

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

  // Build a progress list without spawning worker threads.
  void attachProgressList(int ginCommCount, int proxyNthreads, const std::vector<int>& needsProxy) {
    auto* gs = gin();
    gs->proxyNthreads = proxyNthreads;
    gs->backends[0].ginCommCount = ginCommCount;
    fake_.slots.clear();
    gs->devComms = nullptr;
    appendDevComm(ginCommCount, needsProxy);
  }

  // Append one more devComm to the list, reusing the slot vector so the caller
  // can tell which context each ginProgress / queryLastError call landed on.
  ncclGinStateDevComm* appendDevComm(int ginCommCount, const std::vector<int>& needsProxy) {
    auto* dc = static_cast<ncclGinStateDevComm*>(std::calloc(1, sizeof(ncclGinStateDevComm)));
    if (dc == nullptr) {
      ADD_FAILURE() << "calloc ncclGinStateDevComm";
      return nullptr;
    }
    dc->backendIndex = 0;
    for (int i = 0; i < ginCommCount; i++) {
      auto slot = std::make_unique<FakeSlot>();
      slot->idx = i;
      slot->handle.netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
      slot->handle.handle = slot.get();
      slot->handle.needsProxyProgress = (i < static_cast<int>(needsProxy.size()) && needsProxy[i]) ? 1 : 0;
      dc->devHandles[i] = &slot->handle;
      dc->ginCtx[i] = slot.get();
      fake_.slots.push_back(std::move(slot));
    }
    auto* gs = gin();
    if (gs->devComms == nullptr) {
      gs->devComms = dc;
    } else {
      auto* last = gs->devComms;
      while (last->next) last = last->next;
      last->next = dc;
    }
    return dc;
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

// Joins every spawned worker on scope exit, including a fatal ASSERT return.
// Destroying a still-joinable std::thread calls std::terminate().
class JoinProgressThreads {
 public:
  explicit JoinProgressThreads(ncclGinState* gs) : gs_(gs) {}
  ~JoinProgressThreads() {
    stopProgress(gs_);
    for (auto& t : threads_) {
      if (t.joinable()) t.join();
    }
  }
  template <class Fn>
  void spawn(Fn&& fn) {
    threads_.emplace_back(std::forward<Fn>(fn));
  }
  JoinProgressThreads(const JoinProgressThreads&) = delete;
  JoinProgressThreads& operator=(const JoinProgressThreads&) = delete;

 private:
  ncclGinState* gs_;
  std::vector<std::thread> threads_;
};

// Unset GIN_PROXY_NTHREADS → proxyNthreads and ginCommCount stay 1.
TEST_F(GinHostTest, DefaultNthreadsIsOne) {
  g_loadParam = [](const char* env, int64_t deft) -> int64_t {
    if (std::strcmp(env, "GIN_ENABLE") == 0) return 1;
    return deft;
  };
  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(1, gin()->proxyNthreads);
  EXPECT_EQ(1, gin()->backends[0].ginCommCount);
}

// GIN_PROXY_NTHREADS<=0 is treated as 1; no extra progress threads.
TEST_F(GinHostTest, ZeroAndNegativeNthreadsStayAtOne) {
  nthreadsParam_ = 0;
  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(1, gin()->proxyNthreads);

  gin()->connected = false;
  nthreadsParam_ = -3;
  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(1, gin()->proxyNthreads);
}

// GIN_PROXY_NTHREADS>4 is clamped to NCCL_GIN_MAX_CONNECTIONS (4).
TEST_F(GinHostTest, ClampNthreadsToMaxConnections) {
  nthreadsParam_ = 100;
  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(NCCL_GIN_MAX_CONNECTIONS, gin()->proxyNthreads);
  EXPECT_EQ(NCCL_GIN_MAX_CONNECTIONS, gin()->backends[0].ginCommCount);
}

// nthreads=4 with 1 local GIN dev → ginCommCount raised to 4 before AllGather.
TEST_F(GinHostTest, BumpsGinCommCountToMatchNthreads) {
  nthreadsParam_ = 4;
  g_nLocalGinDevs = 1;
  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(4, gin()->proxyNthreads);
  EXPECT_EQ(4, gin()->backends[0].ginCommCount);
}

// GIN_NCONNECTIONS=2 then GIN_PROXY_NTHREADS=4 → ginCommCount is 4, not 2.
TEST_F(GinHostTest, NthreadsWinsOverGinNconnections) {
  nthreadsParam_ = 4;
  nconnParam_ = 2;
  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(4, gin()->proxyNthreads);
  EXPECT_EQ(4, gin()->backends[0].ginCommCount);
}

// 2 threads × 4 connections: thread t only progresses connections t, t+2, …
TEST_F(GinHostTest, RoundRobinOwnership) {
  attachProgressList(/*ginCommCount=*/4, /*proxyNthreads=*/2, {1, 1, 1, 1});
  auto* gs = gin();
  {
    JoinProgressThreads workers(gs);
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
    JoinProgressThreads only0(gs);
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
TEST_F(GinHostTest, SkipsConnectionsThatDoNotNeedProxyProgress) {
  attachProgressList(2, 1, {1, 0});
  auto* gs = gin();
  {
    JoinProgressThreads worker(gs);
    worker.spawn([gs] { ncclGinProgress(gs, 0); });
    ASSERT_TRUE(waitUntil([&] { return fake_.slots[0]->progressCalls.load() > 0; }));
  }
  EXPECT_EQ(0, fake_.slots[1]->progressCalls.load());
  freeProgressList();
}

// writePending=true: progress thread yields (0 ginProgress) until the flag clears.
TEST_F(GinHostTest, WritePendingBacksOffReaders) {
  attachProgressList(1, 1, {1});
  auto* gs = gin();
  gs->writePending.store(true);
  g_progressEntries.store(0);
  {
    JoinProgressThreads worker(gs);
    worker.spawn([gs] { ncclGinProgress(gs, 0); });
    // Entry is the positive control: the 50 ms window starts only after the
    // worker has reached ncclGinProgress, so an ignored writePending cannot pass as 0 == 0.
    ASSERT_TRUE(waitUntil([&] { return g_progressEntries.load() > 0; }))
        << "progress thread never entered ncclGinProgress";
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(0, fake_.totalProgressCalls.load());
    gs->writePending.store(false);
    ASSERT_TRUE(waitUntil([&] { return fake_.totalProgressCalls.load() > 0; }));
  }
  freeProgressList();
}

// ginProgress returning ncclSystemError stores it in asyncResult and the thread exits.
TEST_F(GinHostTest, ProgressErrorSetsAsyncResultAndExits) {
  attachProgressList(1, 1, {1});
  fake_.slots[0]->progressResult = ncclSystemError;
  auto* gs = gin();
  {
    // On the test thread a missed error return spins in while(1) until the
    // whole binary's timeout. A worker plus waitUntil fails this case instead.
    JoinProgressThreads worker(gs);
    worker.spawn([gs] { ncclGinProgress(gs, 0); });
    ASSERT_TRUE(waitUntil([&] { return gs->asyncResult == ncclSystemError; }))
        << "ginProgress error did not stop the worker";
  }
  EXPECT_EQ(ncclSystemError, gs->asyncResult);
  freeProgressList();
}

// First PROXY DevCommCreate starts exactly nthreads joinable progress threads.
TEST_F(GinHostTest, FirstProxyDevCommStartsThreads) {
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
TEST_F(GinHostTest, GdaThenProxyStartsThreadsOnSecondDevComm) {
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
TEST_F(GinHostTest, FreeUnlinksAndStopsProgressingDestroyedCtx) {
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

// HostFinalize joins every progress thread before it returns. A worker held
// inside ginProgress makes that join visible: deleting the join would let
// finalize return while the worker is still in the call. The post-join memset
// of ginState cannot be used instead, because it stops progress even if the
// threads were never joined.
TEST_F(GinHostTest, FinalizeJoinsAllProgressThreads) {
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
  // ASSERT. HostFinalize memsets ginState, so restore C++ lifetime afterwards.
  auto restoreAfterFinalize = [&]() {
    fake_.holdProgress.store(0, std::memory_order_release);
    if (fin.joinable()) fin.join();
    new (gin()) ncclGinState{};
    std::free(dc);
  };
  struct FinalizeGuard {
    decltype(restoreAfterFinalize)* restore;
    ~FinalizeGuard() { (*restore)(); }
  } finalizeGuard{&restoreAfterFinalize};

  ASSERT_TRUE(waitUntil([&] { return gin()->proxyThreadStopSignal.load(); }))
      << "HostFinalize did not reach the progress-thread join";
  EXPECT_FALSE(finalizeDone.load(std::memory_order_acquire))
      << "HostFinalize returned while a worker was still inside ginProgress";
  fake_.holdProgress.store(0, std::memory_order_release);
  ASSERT_TRUE(waitUntil([&] { return finalizeDone.load(std::memory_order_acquire); }))
      << "HostFinalize did not return after the worker left ginProgress";
  EXPECT_EQ(ncclSuccess, finalizeSt);
}

// nthreads=4 but AllGather ginCommCount=2: threads 2 and 3 never call ginProgress.
TEST_F(GinHostTest, IdleExtraThreadsNeverCallGinProgress) {
  nthreadsParam_ = 4;
  comm_->nRanks = 2;
  g_peerGinCommCount = 2;
  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(4, gin()->proxyNthreads);
  EXPECT_EQ(2, gin()->backends[0].ginCommCount);

  attachProgressList(gin()->backends[0].ginCommCount, gin()->proxyNthreads, {1, 1});
  auto* gs = gin();
  g_progressEntries.store(0);
  {
    JoinProgressThreads idle(gs);
    idle.spawn([gs] { ncclGinProgress(gs, 2); });
    idle.spawn([gs] { ncclGinProgress(gs, 3); });
    ASSERT_TRUE(waitUntil([&] { return g_progressEntries.load() >= 2; }))
        << "idle progress threads never entered ncclGinProgress";
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(0, fake_.totalProgressCalls.load());
  }
  freeProgressList();
}

class GinHostProxyAffinityMicrotest : public ::testing::Test {
protected:
  ncclGinState ginState_;

  void SetUp() override {
    ResetGinHostGlobals();
    ResetAffinityFakes();
    CPU_ZERO(&ginState_.cpuAffinity);
    ginState_.proxyThreadStopSignal.store(true);  // exit at the top of the loop
    ginState_.writePending.store(false);
  }

  void TearDown() override { ResetAffinityFakes(); }

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
  g_osCpuCountValue.store(1);

  RunProgressOnce();

  ASSERT_EQ(1u, g_osSetAffinityMasks.size());
  EXPECT_TRUE(CPU_ISSET(3, &g_osSetAffinityMasks[0]));
  EXPECT_EQ(1, CPU_COUNT(&g_osSetAffinityMasks[0]));
}

TEST_F(GinHostProxyAffinityMicrotest, EmptyAffinity_LeavesProxyThreadAffinityUnchanged) {
  g_osCpuCountValue.store(0);

  RunProgressOnce();

  EXPECT_EQ(1, g_progressEntries.load());
  EXPECT_TRUE(g_osSetAffinityMasks.empty());
}

struct FakeGinBackend {
  ncclNetDeviceHandle_t devHandle{};
  void* ginCtx = reinterpret_cast<void*>(0x1);
  int createContextCalls = 0;
};
FakeGinBackend* g_fakeGinBackend = nullptr;

ncclResult_t FakeCreateContext(void* /*collComm*/, ncclGinConfig_t* /*config*/, void** ginCtx,
                               ncclNetDeviceHandle_t** devHandle) {
  g_fakeGinBackend->createContextCalls++;
  g_fakeGinBackend->devHandle.handle = reinterpret_cast<void*>(0x2);
  g_fakeGinBackend->devHandle.needsProxyProgress = 1;
  *ginCtx = g_fakeGinBackend->ginCtx;
  *devHandle = &g_fakeGinBackend->devHandle;
  return ncclSuccess;
}

ncclResult_t FakeDestroyContext(void* /*ginCtx*/) { return ncclSuccess; }

// The producer: ginDevCommSetupWithBackend copies comm->cpuAffinity into
// ginState->cpuAffinity (gin_host.cc:393) on the branch that first spawns the
// progress threads, so the consumer above has a populated set to read on a real
// comm. Drive that setup path with a scripted backend and assert the copy.
class GinHostProxyAffinitySetupMicrotest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> comm_ = std::make_unique<ncclComm>();
  std::unique_ptr<ncclSharedResources> sr_ = std::make_unique<ncclSharedResources>();
  FakeGinBackend fakeBackend_;
  ncclGin_t vtable_{};
  ncclDevComm devComm_{};

  void SetUp() override {
    ResetGinHostGlobals();
    ResetAffinityFakes();
    g_fakeGinBackend = &fakeBackend_;

    comm_->sharedRes = sr_.get();
    struct ncclGinState& ginState = sr_->ginState;
    ginState.ginConnectionType = NCCL_GIN_CONNECTION_FULL;  // connectedStride == 1
    ginState.proxyNthreads = 1;
    ginState.proxyThreadsCreated = false;                  // so needsStart is true
    ginState.proxyThreadStopSignal.store(true);            // spawned threads exit immediately
    ginState.writePending.store(false);

    vtable_.name = "fake-gin";
    vtable_.createContext = &FakeCreateContext;
    vtable_.destroyContext = &FakeDestroyContext;

    struct ncclGinBackendState& backend = ginState.backends[0];
    backend.ginType = NCCL_GIN_TYPE_PROXY;
    backend.ncclGin = &vtable_;
    backend.ginCommCount = 1;
    backend.ginComms[0] = reinterpret_cast<void*>(0x10);
  }

  void TearDown() override {
    struct ncclGinState& ginState = sr_->ginState;
    for (int t = 0; t < ginState.proxyNthreads; t++) {
      if (ginState.thread[t].joinable()) ginState.thread[t].join();
    }

    if (ginState.devComms != nullptr) {
      EXPECT_EQ(ncclSuccess, ncclGinDevCommFree(comm_.get(), &devComm_));
    }
    g_fakeGinBackend = nullptr;
    ResetAffinityFakes();
  }
};

TEST_F(GinHostProxyAffinitySetupMicrotest, StashesCommAffinityBeforeSpawningProxyThreads) {
  CPU_ZERO(&comm_->cpuAffinity);
  CPU_SET(5, &comm_->cpuAffinity);  // a distinctive mask to spot the copy
  g_osCpuCountValue.store(1);        // so the spawned worker takes the pin branch

  struct ncclGinState& ginState = sr_->ginState;
  ncclDevCommRequirements reqs{};
  reqs.ginContextCount = 1;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_NONE;      // requestedStride stays 1
  reqs.ginTrafficClass = NCCL_CONFIG_UNDEF_INT;

  ncclResult_t ret =
    ginDevCommSetupWithBackend(comm_.get(), &reqs, &devComm_, /*deviceCodeVersion=*/0, &ginState.backends[0]);

  ASSERT_EQ(ncclSuccess, ret);
  EXPECT_EQ(1, fakeBackend_.createContextCalls);         // the setup path really ran
  EXPECT_TRUE(ginState.proxyThreadsCreated);             // it took the spawn branch
  EXPECT_TRUE(CPU_EQUAL(&comm_->cpuAffinity, &ginState.cpuAffinity));  // comm mask stashed verbatim

  ginState.thread[0].join();
  ASSERT_EQ(1u, g_osSetAffinityMasks.size());
  EXPECT_TRUE(CPU_EQUAL(&comm_->cpuAffinity, &g_osSetAffinityMasks[0]));
}

////////////////////////////////////////////////////////////////////////////////
// ncclGetGinType / ncclGetRailedGinType

class GinHostTypeQueryMicrotest : public GinHostTest {};

// Both accessors reject a missing comm or a missing out-parameter rather than
// dereferencing them.
TEST_F(GinHostTypeQueryMicrotest, NullArgumentsRejected) {
  ncclGinType_t type = NCCL_GIN_TYPE_NONE;
  EXPECT_EQ(ncclInternalError, ncclGetGinType(nullptr, &type));
  EXPECT_EQ(ncclInternalError, ncclGetGinType(comm(), nullptr));
  EXPECT_EQ(ncclInternalError, ncclGetRailedGinType(nullptr, &type));
  EXPECT_EQ(ncclInternalError, ncclGetRailedGinType(comm(), nullptr));
}

// ncclGetGinType answers for all-to-all reachability: only a FULL comm has it.
TEST_F(GinHostTypeQueryMicrotest, GinTypeNeedsFullConnectivity) {
  gin()->backends[0].ginType = NCCL_GIN_TYPE_ANVIL_SDMA;

  ncclGinType_t type = NCCL_GIN_TYPE_NONE;
  comm_->globalGinSupport = NCCL_GIN_CONNECTION_FULL;
  ASSERT_EQ(ncclSuccess, ncclGetGinType(comm(), &type));
  EXPECT_EQ(NCCL_GIN_TYPE_ANVIL_SDMA, type);

  // A rail-only comm cannot reach every peer, so it reports no GIN at all.
  comm_->globalGinSupport = NCCL_GIN_CONNECTION_RAIL;
  ASSERT_EQ(ncclSuccess, ncclGetGinType(comm(), &type));
  EXPECT_EQ(NCCL_GIN_TYPE_NONE, type);
}

// ncclGetRailedGinType answers for rail reachability, so anything but NONE
// reports the backend's type -- including the RAIL case the accessor above hides.
TEST_F(GinHostTypeQueryMicrotest, RailedGinTypeAcceptsRailConnectivity) {
  gin()->backends[0].ginType = NCCL_GIN_TYPE_GDAKI;

  ncclGinType_t type = NCCL_GIN_TYPE_NONE;
  comm_->globalGinSupport = NCCL_GIN_CONNECTION_RAIL;
  ASSERT_EQ(ncclSuccess, ncclGetRailedGinType(comm(), &type));
  EXPECT_EQ(NCCL_GIN_TYPE_GDAKI, type);

  comm_->globalGinSupport = NCCL_GIN_CONNECTION_NONE;
  ASSERT_EQ(ncclSuccess, ncclGetRailedGinType(comm(), &type));
  EXPECT_EQ(NCCL_GIN_TYPE_NONE, type);
}

////////////////////////////////////////////////////////////////////////////////
// ncclGinValidateSignalRequest

class GinHostSignalRequestMicrotest : public GinHostTest {};

// A backend is only a candidate if it supports every signal flavour the
// requirements insist on.
TEST_F(GinHostSignalRequestMicrotest, RejectsBackendMissingARequiredSignalFlavour) {
  auto* backend = &gin()->backends[0];
  auto reqs = proxyReqs();

  reqs.ginStrongSignalsRequired = true;
  backend->supportsStrongSignals = false;
  EXPECT_EQ(ncclInvalidUsage, ncclGinValidateSignalRequest(&reqs, backend));

  backend->supportsStrongSignals = true;
  reqs.ginVaSignalsRequired = true;
  backend->supportsVASignals = false;
  EXPECT_EQ(ncclInvalidUsage, ncclGinValidateSignalRequest(&reqs, backend));

  backend->supportsVASignals = true;
  EXPECT_EQ(ncclSuccess, ncclGinValidateSignalRequest(&reqs, backend));
}

// Signals that are not required are never checked, so a backend without them
// still qualifies.
TEST_F(GinHostSignalRequestMicrotest, IgnoresUnsupportedSignalsThatAreNotRequired) {
  auto* backend = &gin()->backends[0];
  backend->supportsStrongSignals = false;
  backend->supportsVASignals = false;

  auto reqs = proxyReqs();  // both *Required flags already false
  EXPECT_EQ(ncclSuccess, ncclGinValidateSignalRequest(&reqs, backend));
}

////////////////////////////////////////////////////////////////////////////////
// ncclGinConnectOnce -- refusals, plugin failures, and the strided team

class GinHostConnectOnceMicrotest : public GinHostTest {};

// Connecting twice is a no-op: the second call never reaches the plugin.
TEST_F(GinHostConnectOnceMicrotest, SecondCallIsANoOp) {
  ASSERT_EQ(ncclSuccess, connectOnce());
  const int devicesCalls = fake_.devicesCalls;
  ASSERT_GE(devicesCalls, 1);

  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(devicesCalls, fake_.devicesCalls);
}

// NCCL_GIN_ENABLE=0 is a hard refusal, not a silent fallback.
TEST_F(GinHostConnectOnceMicrotest, DisabledByParamReturnsInternalError) {
  g_loadParam = [](const char* env, int64_t deft) -> int64_t {
    if (std::strcmp(env, "GIN_ENABLE") == 0) return 0;
    return deft;
  };
  EXPECT_EQ(ncclInternalError, connectOnce());
  EXPECT_FALSE(gin()->connected);
  EXPECT_EQ(0, fake_.devicesCalls);
}

// No backend was loaded for this comm: an invalid request, not an internal fault.
TEST_F(GinHostConnectOnceMicrotest, UnsupportedCommReturnsInvalidUsage) {
  gin()->supported = false;
  EXPECT_EQ(ncclInvalidUsage, connectOnce());
  EXPECT_FALSE(gin()->connected);
}

// GIN windows are symmetric memory, so a comm without symmetric support cannot
// host them.
TEST_F(GinHostConnectOnceMicrotest, NoSymmetricSupportReturnsInternalError) {
  comm_->symmetricSupport = false;
  EXPECT_EQ(ncclInternalError, connectOnce());
  EXPECT_FALSE(gin()->connected);
}

// A plugin that reports zero devices is rejected before any listen/connect.
TEST_F(GinHostConnectOnceMicrotest, ZeroDevicesReturnsInternalError) {
  fake_.ndev = 0;
  EXPECT_EQ(ncclInternalError, connectOnce());
  EXPECT_EQ(0, fake_.listenCalls);
  EXPECT_FALSE(gin()->connected);
}

// A failing devices() surfaces the plugin's own status.
TEST_F(GinHostConnectOnceMicrotest, DevicesFailurePropagates) {
  fake_.failDevices = {ncclSystemError, 1};
  EXPECT_EQ(ncclSystemError, connectOnce());
  EXPECT_FALSE(gin()->connected);
}

// More local GIN devices than connections: the extra devices are dropped rather
// than overrunning the fixed-size connection arrays.
TEST_F(GinHostConnectOnceMicrotest, ClampsLocalDevicesToMaxConnections) {
  g_nLocalGinDevs = NCCL_GIN_MAX_CONNECTIONS + 2;
  ASSERT_EQ(ncclSuccess, connectOnce());
  EXPECT_EQ(NCCL_GIN_MAX_CONNECTIONS, gin()->backends[0].ginCommCount);
  EXPECT_EQ(NCCL_GIN_MAX_CONNECTIONS, fake_.connectCalls);
}

// A failed connect releases the listen comm it was handed and leaves the state
// disconnected, so a later retry starts clean.
TEST_F(GinHostConnectOnceMicrotest, ConnectFailureClosesListenComm) {
  fake_.failConnect = {ncclSystemError, 1};
  EXPECT_EQ(ncclSystemError, connectOnce());
  EXPECT_EQ(1, fake_.closeListenCalls);
  EXPECT_EQ(0, fake_.closeCollCalls);
  EXPECT_FALSE(gin()->connected);
}

// Failing partway through closes the connections that already succeeded and
// clears their slots.
TEST_F(GinHostConnectOnceMicrotest, ConnectFailureClosesEarlierConnections) {
  nthreadsParam_ = 2;  // two connections, so there is an earlier one to close
  fake_.failConnect = {ncclSystemError, 2};
  EXPECT_EQ(ncclSystemError, connectOnce());
  EXPECT_EQ(1, fake_.closeCollCalls);
  EXPECT_EQ(nullptr, gin()->backends[0].ginComms[0]);
  EXPECT_FALSE(gin()->connected);
}

// A listen that fails hands back no listen comm, so the fail path must not try
// to close one.
TEST_F(GinHostConnectOnceMicrotest, ListenFailureSkipsCloseListen) {
  fake_.failListen = {ncclSystemError, 1};
  EXPECT_EQ(ncclSystemError, connectOnce());
  EXPECT_EQ(0, fake_.closeListenCalls);
}

// A topology query that fails stops the connect before the plugin is touched.
TEST_F(GinHostConnectOnceMicrotest, TopologyQueryFailurePropagates) {
  g_failTopoGetLocalGinDevs = {ncclSystemError, 1};
  EXPECT_EQ(ncclSystemError, connectOnce());
  EXPECT_EQ(0, fake_.devicesCalls);
  EXPECT_FALSE(gin()->connected);
}

// A failed property query releases the listen comm that was opened for the same
// connection.
TEST_F(GinHostConnectOnceMicrotest, GetPropertiesFailureClosesListenComm) {
  fake_.failGetProperties = {ncclInternalError, 1};
  EXPECT_EQ(ncclInternalError, connectOnce());
  EXPECT_EQ(1, fake_.closeListenCalls);
  EXPECT_EQ(0, fake_.connectCalls);
}

// Ranks agree on the connection count before connecting, so a failed exchange
// stops the connect.
TEST_F(GinHostConnectOnceMicrotest, ConnectionCountExchangeFailurePropagates) {
  g_failBootstrapAllGather = {ncclSystemError, 1};
  EXPECT_EQ(ncclSystemError, connectOnce());
  EXPECT_EQ(0, fake_.connectCalls);
  EXPECT_FALSE(gin()->connected);
}

// Cleanup failures must not mask the failure that caused the cleanup: the
// caller still sees why the connect failed.
TEST_F(GinHostConnectOnceMicrotest, OriginalFailureSurvivesAFailingCleanup) {
  fake_.failConnect = {ncclSystemError, 1};
  fake_.failCloseListen = {ncclInternalError, 1};
  EXPECT_EQ(ncclSystemError, connectOnce());
  EXPECT_EQ(1, fake_.closeListenCalls);
}

// A comm that is only rail-connected connects a strided team of one rank per
// host instead of the world team.
TEST_F(GinHostConnectOnceMicrotest, RailOnlyCommConnectsStridedTeam) {
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

class GinHostBackendSelectMicrotest : public GinHostTest {
 protected:
  ncclDevComm devComm_{};

  void SetUp() override {
    GinHostTest::SetUp();
    ASSERT_EQ(ncclSuccess, connectOnce());
  }

  ncclResult_t setup(ncclDevCommRequirements const& reqs) {
    return ncclGinDevCommSetup(comm(), &reqs, &devComm_, NCCL_VERSION_CODE);
  }

  void TearDown() override {
    if (gin()->devComms != nullptr) EXPECT_EQ(ncclSuccess, ncclGinDevCommFree(comm(), &devComm_));
    GinHostTest::TearDown();
  }
};

// A type outside the enum is rejected before any backend is considered.
TEST_F(GinHostBackendSelectMicrotest, OutOfRangeGinTypeRejected) {
  auto reqs = proxyReqs();
  reqs.ginType = NCCL_GIN_MAX_TYPES;
  EXPECT_EQ(ncclInvalidUsage, setup(reqs));
  EXPECT_EQ(0, fake_.createContextCalls);
}

// NCCL_GIN_TYPE picks the backend when the requirements express no preference.
TEST_F(GinHostBackendSelectMicrotest, EnvGinTypeSelectsBackendWhenRequirementsAreNeutral) {
  auto reqs = proxyReqs();
  reqs.ginType = NCCL_GIN_TYPE_NONE;

  g_paramGinType = NCCL_GIN_TYPE_GDAKI;  // no GDAKI backend is loaded
  EXPECT_EQ(ncclInternalError, setup(reqs));
  EXPECT_EQ(0, fake_.createContextCalls);

  g_paramGinType = NCCL_GIN_TYPE_PROXY;  // matches the loaded backend
  EXPECT_EQ(ncclSuccess, setup(reqs));
  EXPECT_EQ(1, fake_.createContextCalls);
}

// With no preference from either the requirements or the environment, the first
// loaded backend is taken whatever its type.
TEST_F(GinHostBackendSelectMicrotest, NeutralRequestAcceptsTheFirstLoadedBackend) {
  ASSERT_EQ(-1, g_paramGinType);  // NCCL_GIN_TYPE unset
  gin()->backends[0].ginType = NCCL_GIN_TYPE_ANVIL_SDMA;

  auto reqs = proxyReqs();
  reqs.ginType = NCCL_GIN_TYPE_NONE;
  EXPECT_EQ(ncclSuccess, setup(reqs));
  EXPECT_EQ(1, fake_.createContextCalls);
}

// A backend that cannot provide a required signal flavour is skipped, not failed
// into.
TEST_F(GinHostBackendSelectMicrotest, SkipsBackendThatFailsSignalValidation) {
  gin()->backends[0].supportsStrongSignals = false;
  auto reqs = proxyReqs();
  reqs.ginStrongSignalsRequired = true;

  EXPECT_EQ(ncclInternalError, setup(reqs));
  EXPECT_EQ(0, fake_.createContextCalls);
}

// When the first candidate's setup fails, the next backend gets a turn and the
// devComm records the one that worked.
TEST_F(GinHostBackendSelectMicrotest, FallsBackToTheNextBackendOnSetupFailure) {
  auto* gs = gin();
  gs->numActiveBackends = 2;
  gs->backends[1] = gs->backends[0];
  gs->backends[1].ginComms[0] = reinterpret_cast<void*>(0x55);
  fake_.failCreateContext = {ncclSystemError, 1};  // only the first backend's attempt

  auto reqs = proxyReqs();
  ASSERT_EQ(ncclSuccess, setup(reqs));
  EXPECT_EQ(1, devComm_.backendIndex);
  EXPECT_EQ(2, fake_.createContextCalls);
}

////////////////////////////////////////////////////////////////////////////////
// ginDevCommSetupWithBackend -- config, strides, and failure cleanup

// Drives the file-static setup directly against a hand-built backend, so no
// connect plumbing stands between the test and the branch under test.
class GinHostDevCommSetupMicrotest : public GinHostTest {
 protected:
  ncclDevComm devComm_{};

  void SetUp() override {
    GinHostTest::SetUp();
    auto* gs = gin();
    gs->proxyNthreads = 1;
    gs->proxyThreadStopSignal.store(true);  // any spawned worker exits immediately
    gs->backends[0].ginCommCount = 1;
    gs->backends[0].ginComms[0] = reinterpret_cast<void*>(0x44);
  }

  void TearDown() override {
    if (gin()->devComms != nullptr) EXPECT_EQ(ncclSuccess, ncclGinDevCommFree(comm(), &devComm_));
    GinHostTest::TearDown();
  }

  const ncclGinConfig_t& lastConfig() { return fake_.createdConfigs.back(); }
};

// The plugin is told which device-code ABI to speak, derived from the device
// code's NCCL version and the backend's own compatibility table.
TEST_F(GinHostDevCommSetupMicrotest, BackendVersionFollowsDeviceCodeVersionPerBackend) {
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
    ASSERT_EQ(ncclSuccess, setupAndFree(reqs, c.deviceCodeVersion));
    EXPECT_EQ(c.expectedBackendVersion, lastConfig().backendVersion);
  }
}

// A backend whose type has no compatibility table cannot be configured at all.
TEST_F(GinHostDevCommSetupMicrotest, UnknownBackendTypeRejected) {
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
TEST_F(GinHostDevCommSetupMicrotest, ContextCountRoundsUpToWholeConnections) {
  gin()->backends[0].ginCommCount = 4;
  for (int i = 1; i < 4; i++) gin()->backends[0].ginComms[i] = reinterpret_cast<void*>(0x44 + i);

  auto reqs = proxyReqs();
  reqs.ginContextCount = 5;
  ASSERT_EQ(ncclSuccess, setupWithBackend(reqs, &devComm_));

  EXPECT_EQ(8u, devComm_.ginContextCount);  // 5 rounded up to a multiple of 4
  EXPECT_EQ(2, lastConfig().nContexts);     // ... which is 2 per connection
  EXPECT_EQ(4, devComm_.ginConnectionCount);
}

// Exclusive contexts are the conservative path: nothing is shared between
// devComms today, so the request is configured exactly like a shared one.
TEST_F(GinHostDevCommSetupMicrotest, ExclusiveContextsAreConfiguredLikeSharedOnes) {
  auto reqs = proxyReqs();
  reqs.ginContextCount = 1;
  reqs.ginExclusiveContexts = true;

  ASSERT_EQ(ncclSuccess, setupWithBackend(reqs, &devComm_));
  EXPECT_EQ(1u, devComm_.ginContextCount);
  EXPECT_EQ(1, lastConfig().nContexts);
}

// The traffic class comes from the requirements when set, and from the comm's
// config otherwise.
TEST_F(GinHostDevCommSetupMicrotest, TrafficClassFallsBackToCommConfig) {
  comm_->config.trafficClass = 7;
  auto reqs = proxyReqs();

  reqs.ginTrafficClass = 3;
  ASSERT_EQ(ncclSuccess, setupAndFree(reqs, NCCL_VERSION_CODE));
  EXPECT_EQ(3, lastConfig().trafficClass);

  reqs.ginTrafficClass = NCCL_CONFIG_UNDEF_INT;
  ASSERT_EQ(ncclSuccess, setupAndFree(reqs, NCCL_VERSION_CODE));
  EXPECT_EQ(7, lastConfig().trafficClass);
}

// Signal and counter counts are passed through, and legacy signals default to
// the strength the requirements asked for.
TEST_F(GinHostDevCommSetupMicrotest, SignalRequirementsReachTheDevCommAndThePlugin) {
  auto reqs = proxyReqs();
  reqs.ginSignalCount = 5;
  reqs.ginCounterCount = 6;
  reqs.ginQueueDepth = 64;
  reqs.ginStrongSignalsRequired = true;

  ASSERT_EQ(ncclSuccess, setupWithBackend(reqs, &devComm_));

  EXPECT_EQ(5, devComm_.ginSignalCount);
  EXPECT_EQ(6, devComm_.ginCounterCount);
  EXPECT_TRUE(devComm_.ginStrongLegacySignals);
  EXPECT_EQ(5, lastConfig().nSignals);
  EXPECT_EQ(6, lastConfig().nCounters);
  EXPECT_EQ(64, lastConfig().queueDepth);
}

// A rail request takes its stride from the rail team.
TEST_F(GinHostDevCommSetupMicrotest, RailRequestUsesTheRailStride) {
  g_railStride = 2;
  auto reqs = proxyReqs();
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;

  ASSERT_EQ(ncclSuccess, setupWithBackend(reqs, &devComm_));

  EXPECT_EQ(2, devComm_.ginContextStride);
  EXPECT_EQ(1, devComm_.ginConnectionStride);  // the comm itself is FULL-connected
  EXPECT_EQ(2, lastConfig().rankStride);
}

// A rail-connected comm's connections already span whole hosts, so the config
// stride is expressed relative to that.
TEST_F(GinHostDevCommSetupMicrotest, StridesAreRelativeToTheConnectedStride) {
  gin()->ginConnectionType = NCCL_GIN_CONNECTION_RAIL;  // connectedStride = ranks per host
  comm_->contiguousRanksPerHost = 2;
  g_railStride = 4;

  auto reqs = proxyReqs();
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_CUSTOM_STRIDE;
  reqs.ginCustomStride = 4;
  ASSERT_EQ(ncclSuccess, setupWithBackend(reqs, &devComm_));

  EXPECT_EQ(2, devComm_.ginConnectionStride);
  EXPECT_EQ(4, devComm_.ginContextStride);
  EXPECT_EQ(2, lastConfig().rankStride);  // 4 world ranks / 2 ranks per connection
}

// A zero stride would make every rank its own peer; it is rejected with the
// hint to disable GIN instead.
TEST_F(GinHostDevCommSetupMicrotest, ZeroCustomStrideRejected) {
  auto reqs = proxyReqs();
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_CUSTOM_STRIDE;
  reqs.ginCustomStride = 0;

  EXPECT_EQ(ncclInvalidUsage, setupWithBackend(reqs, &devComm_));
  EXPECT_EQ(0, fake_.createContextCalls);
}

// Hierarchical barriers assume GIN reaches at least the rail team, so a wider
// stride than the rail's is rejected.
TEST_F(GinHostDevCommSetupMicrotest, StrideWiderThanTheRailTeamRejected) {
  g_railStride = 2;
  auto reqs = proxyReqs();
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_CUSTOM_STRIDE;
  reqs.ginCustomStride = 4;

  EXPECT_EQ(ncclInvalidUsage, setupWithBackend(reqs, &devComm_));
  EXPECT_EQ(0, fake_.createContextCalls);
}

// The requested stride has to be reachable by stepping whole connections.
TEST_F(GinHostDevCommSetupMicrotest, StrideThatIsNotAMultipleOfTheConnectedStrideRejected) {
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
TEST_F(GinHostDevCommSetupMicrotest, FailedSetupClearsTheDevComm) {
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
TEST_F(GinHostDevCommSetupMicrotest, IncompleteContextFromThePluginRejected) {
  const BadContext cases[] = {BadContext::NullGinCtx, BadContext::NullDevHandle, BadContext::NullHandle};
  auto reqs = proxyReqs();
  for (BadContext bad : cases) {
    SCOPED_TRACE(static_cast<int>(bad));
    fake_.badContext = bad;
    ncclDevComm devComm{};
    EXPECT_EQ(ncclInternalError, setupWithBackend(reqs, &devComm));
    EXPECT_EQ(0, devComm.ginConnectionCount);
  }
}

// A context that was created before the failure is destroyed again rather than
// left behind in the plugin.
TEST_F(GinHostDevCommSetupMicrotest, CreateContextFailureDestroysEarlierContexts) {
  gin()->backends[0].ginCommCount = 2;
  gin()->backends[0].ginComms[1] = reinterpret_cast<void*>(0x45);
  fake_.failCreateContext = {ncclSystemError, 2};

  auto reqs = proxyReqs();
  reqs.ginContextCount = 2;
  EXPECT_EQ(ncclSystemError, setupWithBackend(reqs, &devComm_));
  EXPECT_EQ(1, fake_.destroyCalls.load());
  EXPECT_EQ(0, devComm_.ginConnectionCount);
}

// Progress threads are started once; a later devComm joins the existing list
// under the writer lock instead of spawning a second set.
TEST_F(GinHostDevCommSetupMicrotest, LaterDevCommsReuseTheExistingProgressThreads) {
  auto reqs = proxyReqs();
  ncclDevComm first{};
  ncclDevComm second{};
  ASSERT_EQ(ncclSuccess, setupWithBackend(reqs, &first));
  ASSERT_TRUE(gin()->proxyThreadsCreated);
  ASSERT_EQ(ncclSuccess, setupWithBackend(reqs, &second));
  // The third devComm has to walk past both of its predecessors to find the tail.
  ASSERT_EQ(ncclSuccess, setupWithBackend(reqs, &devComm_));

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

class GinHostDevCommFreeMicrotest : public GinHostTest {};

// Freeing against an empty list is an internal error, not a crash on a NULL
// list head.
TEST_F(GinHostDevCommFreeMicrotest, EmptyListReportsInternalError) {
  ncclDevComm devComm{};
  EXPECT_EQ(ncclInternalError, ncclGinDevCommFree(comm(), &devComm));
}

// The devComm is looked up by its GIN handle; an unknown handle walks the whole
// list and then fails.
TEST_F(GinHostDevCommFreeMicrotest, UnknownHandleReportsInternalError) {
  attachProgressList(1, 1, {1});
  appendDevComm(1, {1});

  ncclDevComm devComm{};
  devComm.ginHandles[0] = reinterpret_cast<void*>(0x999);  // belongs to no devComm
  EXPECT_EQ(ncclInternalError, ncclGinDevCommFree(comm(), &devComm));
  EXPECT_EQ(0, fake_.destroyCalls.load());
  freeProgressList();
}

// A plugin that fails to destroy a context surfaces that status to the caller.
TEST_F(GinHostDevCommFreeMicrotest, DestroyContextFailurePropagates) {
  attachProgressList(1, 1, {1});
  fake_.failDestroyContext = {ncclSystemError, 1};

  ncclDevComm devComm{};
  devComm.ginHandles[0] = gin()->devComms->devHandles[0]->handle;
  EXPECT_EQ(ncclSystemError, ncclGinDevCommFree(comm(), &devComm));
  // The devComm was already unlinked, so the list -- not this test -- owns nothing.
  EXPECT_EQ(nullptr, gin()->devComms);
}

////////////////////////////////////////////////////////////////////////////////
// ncclGinHostFinalize

class GinHostFinalizeMicrotest : public GinHostTest {};

// Finalizing a comm that never connected touches no plugin state.
TEST_F(GinHostFinalizeMicrotest, UnconnectedCommIsANoOp) {
  ASSERT_FALSE(gin()->connected);
  EXPECT_EQ(ncclSuccess, ncclGinHostFinalize(comm()));
  EXPECT_EQ(0, fake_.closeCollCalls);
}

// Finalize closes every connection and wipes the GIN state, so a later
// ncclGinConnectOnce starts from scratch.
TEST_F(GinHostFinalizeMicrotest, ClosesEveryConnectionAndClearsState) {
  nthreadsParam_ = 2;
  ASSERT_EQ(ncclSuccess, connectOnce());
  ASSERT_EQ(2, gin()->backends[0].ginCommCount);

  ASSERT_EQ(ncclSuccess, ncclGinHostFinalize(comm()));

  EXPECT_EQ(2, fake_.closeCollCalls);
  EXPECT_FALSE(gin()->connected);
  EXPECT_EQ(0, gin()->numActiveBackends);
  // Finalize memsets the state; restore C++ lifetime for the fixture's teardown.
  new (gin()) ncclGinState{};
}

// A progress-thread slot that was never spawned is skipped rather than joined.
TEST_F(GinHostFinalizeMicrotest, SkipsProgressThreadSlotsThatWereNeverSpawned) {
  ASSERT_EQ(ncclSuccess, connectOnce());
  gin()->proxyThreadsCreated = true;  // ... but no std::thread was ever started
  ASSERT_FALSE(gin()->thread[0].joinable());

  EXPECT_EQ(ncclSuccess, ncclGinHostFinalize(comm()));
  new (gin()) ncclGinState{};
}

// A connection slot that is already empty is skipped instead of being closed a
// second time.
TEST_F(GinHostFinalizeMicrotest, SkipsConnectionSlotsThatAreAlreadyClosed) {
  nthreadsParam_ = 2;
  ASSERT_EQ(ncclSuccess, connectOnce());
  gin()->backends[0].ginComms[1] = nullptr;

  ASSERT_EQ(ncclSuccess, ncclGinHostFinalize(comm()));
  EXPECT_EQ(1, fake_.closeCollCalls);
  new (gin()) ncclGinState{};
}

// A close that fails surfaces to the caller instead of being swallowed by the
// teardown path.
TEST_F(GinHostFinalizeMicrotest, CloseCollFailurePropagates) {
  ASSERT_EQ(ncclSuccess, connectOnce());
  fake_.failCloseColl = {ncclSystemError, 1};
  EXPECT_EQ(ncclSystemError, ncclGinHostFinalize(comm()));
}

////////////////////////////////////////////////////////////////////////////////
// ncclGinRegister / ncclGinDeregister

class GinHostRegisterMicrotest : public GinHostTest {
 protected:
  void* hostWins_[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS] = {};
  ncclGinWindow_t devWins_[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS] = {};
  char buffer_[64] = {};

  void SetUp() override {
    GinHostTest::SetUp();
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

  static void* collCommFor(int backendIdx, int commIdx) {
    return reinterpret_cast<void*>(0x100 + 0x10 * (uintptr_t)backendIdx + (uintptr_t)commIdx);
  }

  ncclResult_t registerWindow(int winFlags = 0, bool multiSegment = false, int memType = NCCL_PTR_CUDA) {
    return ncclGinRegister(comm(), buffer_, sizeof(buffer_), hostWins_, devWins_, winFlags, multiSegment, memType);
  }
};

// Every connection of every active backend gets its own registration, and the
// handles land in that connection's slot.
TEST_F(GinHostRegisterMicrotest, RegistersOnEveryConnectionOfEveryBackend) {
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
TEST_F(GinHostRegisterMicrotest, PassesMemoryTypeThrough) {
  ASSERT_EQ(ncclSuccess, registerWindow(/*winFlags=*/0, /*multiSegment=*/false, NCCL_PTR_HOST));
  ASSERT_FALSE(fake_.regMrCalls.empty());
  EXPECT_EQ(NCCL_PTR_HOST, fake_.regMrCalls[0].memType);
}

// A strictly-ordered window must be registered with strong ordering forced on
// the NIC.
TEST_F(GinHostRegisterMicrotest, StrictOrderingWindowForcesStrongOrdering) {
  ASSERT_EQ(ncclSuccess, registerWindow(NCCL_WIN_STRICT_ORDERING));
  ASSERT_FALSE(fake_.regMrCalls.empty());
  for (const auto& call : fake_.regMrCalls) EXPECT_EQ(NCCL_NET_MR_FLAG_FORCE_SO, call.mrFlags);
}

// Without that flag no ordering constraint is imposed.
TEST_F(GinHostRegisterMicrotest, OrdinaryWindowRegistersWithoutFlags) {
  ASSERT_EQ(ncclSuccess, registerWindow());
  ASSERT_FALSE(fake_.regMrCalls.empty());
  for (const auto& call : fake_.regMrCalls) EXPECT_EQ(0u, call.mrFlags);
}

// A multi-segment buffer needs DMABUF on every connection; one connection
// without it rejects the whole registration before anything is registered.
TEST_F(GinHostRegisterMicrotest, MultiSegmentWithoutDmabufOnEveryConnectionRejected) {
  gin()->backends[0].ginProps[0].ptrSupport |= NCCL_PTR_DMABUF;  // only the first connection

  EXPECT_EQ(ncclInvalidArgument, registerWindow(/*winFlags=*/0, /*multiSegment=*/true));
  EXPECT_TRUE(fake_.regMrCalls.empty());
}

// With DMABUF everywhere the multi-segment registration proceeds normally.
TEST_F(GinHostRegisterMicrotest, MultiSegmentWithDmabufEverywhereRegisters) {
  for (int i = 0; i < 2; i++) gin()->backends[0].ginProps[i].ptrSupport |= NCCL_PTR_DMABUF;

  EXPECT_EQ(ncclSuccess, registerWindow(/*winFlags=*/0, /*multiSegment=*/true));
  EXPECT_EQ(2u, fake_.regMrCalls.size());
}

// A plugin that reports success but hands back no window is treated as a
// failure rather than storing a NULL window.
TEST_F(GinHostRegisterMicrotest, NullWindowFromThePluginIsASystemError) {
  fake_.regMrSymReturnsNullWindow = true;
  EXPECT_EQ(ncclSystemError, registerWindow());
  EXPECT_EQ(1u, fake_.regMrCalls.size());  // stops at the first bad connection
}

// A failing registration surfaces the plugin's status.
TEST_F(GinHostRegisterMicrotest, RegistrationFailurePropagates) {
  fake_.failRegMrSym = {ncclInternalError, 1};
  EXPECT_EQ(ncclInternalError, registerWindow());
}

// Deregistration mirrors registration: one call per populated slot.
TEST_F(GinHostRegisterMicrotest, DeregistersEveryPopulatedSlot) {
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
TEST_F(GinHostRegisterMicrotest, SkipsEmptySlots) {
  ASSERT_EQ(ncclSuccess, registerWindow());
  void* const win1 = hostWins_[1];
  hostWins_[0] = nullptr;

  ASSERT_EQ(ncclSuccess, ncclGinDeregister(comm(), hostWins_));

  ASSERT_EQ(1u, fake_.deregMrCalls.size());
  EXPECT_EQ(win1, fake_.deregMrCalls[0].second);
}

// A failing deregistration surfaces the plugin's status.
TEST_F(GinHostRegisterMicrotest, DeregistrationFailurePropagates) {
  ASSERT_EQ(ncclSuccess, registerWindow());
  fake_.failDeregMrSym = {ncclSystemError, 1};
  EXPECT_EQ(ncclSystemError, ncclGinDeregister(comm(), hostWins_));
}

////////////////////////////////////////////////////////////////////////////////
// ncclGinQueryLastError

class GinHostQueryLastErrorMicrotest : public GinHostTest {
 protected:
  void TearDown() override {
    freeProgressList();
    GinHostTest::TearDown();
  }
};

// With nothing registered there is nothing to ask, and the answer is "no error".
TEST_F(GinHostQueryLastErrorMicrotest, NoDevCommsReportsNoError) {
  bool hasError = true;
  EXPECT_EQ(ncclSuccess, ncclGinQueryLastError(gin(), &hasError));
  EXPECT_FALSE(hasError);
  EXPECT_EQ(0, fake_.queryCalls);
}

// Every context of every devComm is asked, including the ones that do not need
// proxy progress -- that is the only way a device-initiated backend reports.
TEST_F(GinHostQueryLastErrorMicrotest, AsksEveryContextOfEveryDevComm) {
  attachProgressList(2, 1, {1, 0});
  appendDevComm(2, {0, 0});

  bool hasError = true;
  EXPECT_EQ(ncclSuccess, ncclGinQueryLastError(gin(), &hasError));
  EXPECT_FALSE(hasError);
  EXPECT_EQ(4, fake_.queryCalls);
}

// The first context reporting an error ends the walk: the caller only needs to
// know that something failed.
TEST_F(GinHostQueryLastErrorMicrotest, StopsAtTheFirstContextReportingAnError) {
  attachProgressList(2, 1, {1, 1});
  appendDevComm(2, {1, 1});
  fake_.queryErrorOnCall = 2;

  bool hasError = false;
  EXPECT_EQ(ncclSuccess, ncclGinQueryLastError(gin(), &hasError));
  EXPECT_TRUE(hasError);
  EXPECT_EQ(2, fake_.queryCalls);
}

// A query that fails outright is a different thing from a query that reports an
// error, and surfaces the plugin's status.
TEST_F(GinHostQueryLastErrorMicrotest, QueryFailurePropagates) {
  attachProgressList(1, 1, {1});
  fake_.failQueryLastError = {ncclSystemError, 1};

  bool hasError = false;
  EXPECT_EQ(ncclSystemError, ncclGinQueryLastError(gin(), &hasError));
}

}  // namespace
