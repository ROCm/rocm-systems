/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/gin/gin_host.cc GIN_PROXY_NTHREADS (NVIDIA/nccl#2279,
// AICOMRCCL-2017): per-thread endpoint assignment, ginCommCount bump, writer-priority
// list lock, and progress-thread lifecycle, plus proxy-thread CPU affinity
// (AICOMRCCL-1859): the comm->cpuAffinity stash in ginDevCommSetupWithBackend and
// the pin applied by each progress thread.
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

int g_nLocalGinDevs = 1;
int g_peerGinCommCount = -1;  // -1: AllGather is a no-op copy; else fill other ranks
int64_t g_paramGinType = -1;

}  // namespace

int64_t ncclParamGinType() { return g_paramGinType; }

ncclResult_t ncclTopoGetLocalGinDevs(struct ncclComm*, int* localGinDevs, int* localGinCount) {
  if (localGinCount) *localGinCount = g_nLocalGinDevs;
  if (localGinDevs) {
    for (int i = 0; i < g_nLocalGinDevs; i++) localGinDevs[i] = i;
  }
  return ncclSuccess;
}

ncclResult_t bootstrapAllGather(void*, void* allData, int size) {
  if (g_peerGinCommCount >= 0 && size == static_cast<int>(sizeof(int))) {
    int* counts = static_cast<int*>(allData);
    // Rank 0 already wrote counts[0]; other ranks report a smaller ginCommCount.
    counts[1] = g_peerGinCommCount;
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
  t.nRanks = 1;
  t.rank = 0;
  t.stride = 1;
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

struct FakeGin {
  int ndev = 1;
  bool needsProxyProgress = true;
  std::atomic<int> totalProgressCalls{0};
  std::atomic<int> destroyCalls{0};
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
    if (ndev) *ndev = current().ndev;
    return ncclSuccess;
  }
  static ncclResult_t Listen(void*, int, void*, void** listenComm) {
    *listenComm = reinterpret_cast<void*>(0x11);
    return ncclSuccess;
  }
  static ncclResult_t GetProperties(int, ncclNetProperties_t* props) {
    if (props) std::memset(props, 0, sizeof(*props));
    return ncclSuccess;
  }
  static ncclResult_t Connect(void*, void**, int, int, void*, void** collComm) {
    *collComm = reinterpret_cast<void*>(0x22);
    return ncclSuccess;
  }
  static ncclResult_t CloseListen(void*) { return ncclSuccess; }
  static ncclResult_t CloseColl(void*) { return ncclSuccess; }

  static ncclResult_t CreateContext(void*, ncclGinConfig_t*, void** ginCtx, ncclNetDeviceHandle_t** devHandle) {
    FakeGin& self = current();
    auto slot = std::make_unique<FakeSlot>();
    slot->idx = static_cast<int>(self.slots.size());
    slot->handle.netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
    slot->handle.handle = slot.get();
    slot->handle.needsProxyProgress = self.needsProxyProgress ? 1 : 0;
    *devHandle = &slot->handle;
    *ginCtx = slot.get();
    self.slots.push_back(std::move(slot));
    return ncclSuccess;
  }

  static ncclResult_t DestroyContext(void* ginCtx) {
    FakeGin& self = current();
    ++self.destroyCalls;
    self.destroyed.push_back(ginCtx);
    return ncclSuccess;
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
    if (hasError) *hasError = false;
    return ncclSuccess;
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
    g_nLocalGinDevs = 1;
    g_peerGinCommCount = -1;
    g_paramGinType = -1;
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
    auto* dc = static_cast<ncclGinStateDevComm*>(std::calloc(1, sizeof(ncclGinStateDevComm)));
    if (dc == nullptr) {
      ADD_FAILURE() << "calloc ncclGinStateDevComm";
      return;
    }
    dc->backendIndex = 0;
    fake_.slots.clear();
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

}  // namespace
