/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/proxy.cc. Sockets are socketpair-backed fakes (fakes/proxy_micro_fakes.cc);
// only the allocation seams around the #include live in this TU.

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "../common/LogCapture.hpp"
#include "ScopedHook.h"
#include "fakes/env_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/os_fakes.h"
#include "fakes/proxy_micro_fakes.h"
#include "alloc.h"
#include "comm.h"
#include "info.h"
#include "collectives.h"
#include "socket.h"
#include "shmutils.h"
#include "profiler.h"
#include "transport.h"
#include "cpuset.h"
#include "compiler.h"
#include "os.h"

namespace {
template <typename Pred>
bool WaitFor(Pred pred) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!pred()) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::yield();
  }
  return true;
}

// A g_loadParam hook that sets one proxy.cc param (named as in its NCCL_PARAM) and leaves the rest at their defaults.
auto Param(const char* name, int64_t value) {
  return [name, value](const char* env, int64_t deftVal) { return std::strcmp(env, name) == 0 ? value : deftVal; };
}

// HIP calls the proxy threads make, recorded through hip_fakes.cc's hooks (installed by ProxyFixture).
std::mutex g_hipMutex;
hipError_t g_hipSetDeviceResult, g_hipExportResult, g_hipIpcCloseResult;
std::vector<int> g_hipSetDevices;
std::vector<void*> g_hipIpcClosed;
}  // namespace

// Allocation seams over the unit only: fail the Nth ncclCalloc / ncclRealloc / malloc it makes (0-based).
namespace {
int g_callocCalls, g_callocFailAt = -1, g_reallocCalls, g_reallocFailAt = -1, g_mallocCalls, g_mallocFailAt = -1;
}  // namespace
template <typename... Args>
static ncclResult_t ProxyTestCalloc(const char* file, int line, const char* fn, Args&&... args) {
  if (g_callocCalls++ == g_callocFailAt) return ncclSystemError;
  return ncclCallocDebug(std::forward<Args>(args)..., file, line, fn, true);
}
template <typename T>
static ncclResult_t ProxyTestRealloc(T** ptr, size_t oldNelem, size_t nelem, const char* file, int line,
                                     const char* fn) {
  if (g_reallocCalls++ == g_reallocFailAt) return ncclSystemError;
  return ncclReallocDebug(ptr, oldNelem, nelem, file, line, fn, true);
}
// Fail-path tests free what the unit left behind only if the unit did not free it itself.
void* g_watchedFree;
std::atomic<bool> g_watchedFreed;
// Response buffers and async ops the unit has not freed yet; FreeUnitLeftovers() releases them.
std::mutex g_leftoverMutex;
std::set<void*> g_leftovers;
static void Track(void* p) {
  std::lock_guard<std::mutex> lock(g_leftoverMutex);
  if (p != nullptr) g_leftovers.insert(p);
}
static void ProxyTestFree(void* p) {
  if (p != nullptr && p == g_watchedFree) g_watchedFreed = true;
  {
    std::lock_guard<std::mutex> lock(g_leftoverMutex);
    g_leftovers.erase(p);
  }
  std::free(p);
}
static void* ProxyTestMalloc(size_t n) {
  void* p = g_mallocCalls++ == g_mallocFailAt ? nullptr : std::malloc(n);
  Track(p);
  return p;
}
static ncclResult_t ProxyTestCalloc(const char* file, int line, const char* fn, ncclProxyAsyncOp** op, int n) {
  ncclResult_t res = ProxyTestCalloc<ncclProxyAsyncOp**, int>(file, line, fn, std::move(op), std::move(n));
  if (res == ncclSuccess) Track(*op);
  return res;
}
static size_t LeftoverCount() {
  std::lock_guard<std::mutex> lock(g_leftoverMutex);
  return g_leftovers.size();
}
static void FreeUnitLeftovers() {
  std::lock_guard<std::mutex> lock(g_leftoverMutex);
  for (void* p : g_leftovers) std::free(p);
  g_leftovers.clear();
}
// Counts the service teardown's sync; hip_fakes.cc's hipDeviceSynchronize has no hook to count it.
std::atomic<int> g_hipSyncs;
static hipError_t ProxyTestHipDeviceSynchronize() {
  g_hipSyncs++;
  return hipDeviceSynchronize();
}
#undef ncclCalloc
#define ncclCalloc(...) ProxyTestCalloc(__FILE__, __LINE__, __func__, __VA_ARGS__)
#undef ncclRealloc
#define ncclRealloc(ptr, o, n) ProxyTestRealloc(ptr, o, n, __FILE__, __LINE__, __func__)
#define malloc(n) ProxyTestMalloc(n)
#define free(p) ProxyTestFree(p)
#define hipDeviceSynchronize ProxyTestHipDeviceSynchronize

#include "fakes/param_redirect.h"
#include PROXY_CC_PATH

#undef malloc
#undef free
#undef hipDeviceSynchronize

namespace {

class ProxyFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetProxyMicroFakes();
    ResetEnvFakes();
    ResetOsFakes();
    g_watchedFree = nullptr;
    g_watchedFreed = false;
    g_leftovers.clear();
    g_hipSyncs = 0;
    g_hipSetDeviceResult = g_hipExportResult = g_hipIpcCloseResult = hipSuccess;
    g_hipSetDevices.clear();
    g_hipIpcClosed.clear();
    g_callocCalls = g_reallocCalls = g_mallocCalls = 0;
    g_callocFailAt = g_reallocFailAt = g_mallocFailAt = -1;
  }
  void TearDown() override {
    for (int fd : fds_) close(fd);
  }

  // Returns {ours, theirs}: the unit gets `ours`, the test reads/writes `theirs`.
  std::pair<int, int> SockPair() {
    int sv[2];
    EXPECT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
    fds_.push_back(sv[1]);
    return {sv[0], sv[1]};
  }
  template <typename T>
  static void Put(int fd, const T& v) {
    ASSERT_EQ(send(fd, &v, sizeof(v), MSG_NOSIGNAL), (ssize_t)sizeof(v));
  }
  template <typename T>
  static T Get(int fd) {
    T v{};
    EXPECT_TRUE(WaitReadable(fd));
    EXPECT_EQ(read(fd, &v, sizeof(v)), (ssize_t)sizeof(v));
    return v;
  }

  std::vector<int> fds_;
  std::unique_ptr<ncclComm> comm_ = std::make_unique<ncclComm>();
  ScopedHook<hipError_t(int)> setDevice_{g_hipSetDevice, [](int dev) {
    std::lock_guard<std::mutex> lock(g_hipMutex);
    g_hipSetDevices.push_back(dev);
    return g_hipSetDeviceResult;
  }};
  ScopedHook<hipError_t(void*)> ipcClose_{g_hipIpcCloseMemHandle, [](void* p) {
    g_hipIpcClosed.push_back(p);
    return g_hipIpcCloseResult;
  }};
  ScopedHook<hipError_t(void*, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType, unsigned long long)>
      export_{g_hipMemExportToShareableHandle,
              [](void* shareable, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType, unsigned long long) {
                *(int*)shareable = 77;
                return g_hipExportResult;
              }};
};

// Connection pool

class ProxyConnPoolMicrotest : public ProxyFixture {
 protected:
  void TearDown() override {
    for (int b = 0; b < pool_.banks; b++) free(pool_.pools[b]);
    free(pool_.pools);
    ProxyFixture::TearDown();
  }
  // As ncclProxyService initializes it: the first NewConnection allocates bank 0.
  ncclProxyConnectionPool pool_ = {nullptr, 0, NCCL_PROXY_CONN_POOL_SIZE};
};

TEST_F(ProxyConnPoolMicrotest, NewConnection_AllocatesBanksOnDemandAndEncodesBankInId) {
  int id = -1;
  ASSERT_EQ(ncclProxyNewConnection(&pool_, &id), ncclSuccess);
  EXPECT_EQ(id, 0);
  EXPECT_EQ(pool_.banks, 1);
  for (int i = 1; i < NCCL_PROXY_CONN_POOL_SIZE; i++) ASSERT_EQ(ncclProxyNewConnection(&pool_, &id), ncclSuccess);
  EXPECT_EQ(id, NCCL_PROXY_CONN_POOL_SIZE - 1);
  EXPECT_EQ(pool_.banks, 1);
  ncclProxyConnection* bank0 = pool_.pools[0];
  ASSERT_EQ(ncclProxyNewConnection(&pool_, &id), ncclSuccess);
  EXPECT_EQ(id, 1 << NCCL_PROXY_CONN_POOL_SIZE_POW2);
  EXPECT_EQ(pool_.banks, 2);
  EXPECT_EQ(pool_.offset, 1);
  EXPECT_EQ(pool_.pools[0], bank0);
  EXPECT_EQ(pool_.pools[1][0].state, connUninitialized);
}

TEST_F(ProxyConnPoolMicrotest, NewConnection_AllocationFailuresPropagate) {
  int id = -1;
  g_callocFailAt = 0;
  EXPECT_EQ(ncclProxyNewConnection(&pool_, &id), ncclSystemError);
  EXPECT_EQ(pool_.banks, 0);
  g_reallocFailAt = g_reallocCalls;
  EXPECT_EQ(ncclProxyNewConnection(&pool_, &id), ncclSystemError);
  EXPECT_EQ(pool_.banks, 0);
}

TEST_F(ProxyConnPoolMicrotest, GetConnection_RejectsIdsOutsideTheIssuedRange) {
  ncclProxyConnection* conn = nullptr;
  EXPECT_EQ(ncclProxyGetConnection(&pool_, 0, &conn), ncclInvalidArgument);
  int id;
  for (int i = 0; i < NCCL_PROXY_CONN_POOL_SIZE + 2; i++) ASSERT_EQ(ncclProxyNewConnection(&pool_, &id), ncclSuccess);
  EXPECT_EQ(ncclProxyGetConnection(&pool_, -1, &conn), ncclInvalidArgument);
  EXPECT_EQ(ncclProxyGetConnection(&pool_, 2 << NCCL_PROXY_CONN_POOL_SIZE_POW2, &conn), ncclInvalidArgument);
  EXPECT_EQ(ncclProxyGetConnection(&pool_, id + 1, &conn), ncclInvalidArgument);  // past the last bank's mark
  EXPECT_EQ(conn, nullptr);
  ASSERT_EQ(ncclProxyGetConnection(&pool_, id, &conn), ncclSuccess);
  EXPECT_EQ(conn, pool_.pools[1] + 1);
  // Full earlier banks accept every offset, even past the last bank's high-water mark.
  ASSERT_EQ(ncclProxyGetConnection(&pool_, NCCL_PROXY_CONN_POOL_SIZE - 1, &conn), ncclSuccess);
  EXPECT_EQ(conn, pool_.pools[0] + NCCL_PROXY_CONN_POOL_SIZE - 1);
  free(pool_.pools[1]);
  pool_.pools[1] = nullptr;
  EXPECT_EQ(ncclProxyGetConnection(&pool_, id, &conn), ncclInvalidArgument);
}

TEST_F(ProxyConnPoolMicrotest, ValidateConnection_ChecksBoundsAlignmentAndOwner) {
  int id;
  for (int i = 0; i < NCCL_PROXY_CONN_POOL_SIZE + 2; i++) ASSERT_EQ(ncclProxyNewConnection(&pool_, &id), ncclSuccess);
  pool_.pools[0][5].peerId = 7;
  pool_.pools[1][1].peerId = 9;
  EXPECT_EQ(ncclProxyValidateConnection(&pool_, &pool_.pools[0][5], 7), ncclSuccess);
  EXPECT_EQ(ncclProxyValidateConnection(&pool_, &pool_.pools[0][NCCL_PROXY_CONN_POOL_SIZE - 1], 0), ncclSuccess);
  EXPECT_EQ(ncclProxyValidateConnection(&pool_, &pool_.pools[1][1], 9), ncclSuccess);
  EXPECT_EQ(ncclProxyValidateConnection(&pool_, &pool_.pools[0][5], 8), ncclInvalidArgument);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Invalid proxy connection peer: 7 != 8"));
  EXPECT_EQ(ncclProxyValidateConnection(&pool_, (ncclProxyConnection*)((char*)&pool_.pools[0][5] + 8), 7),
            ncclInvalidArgument);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "wrong alignment"));
  EXPECT_EQ(ncclProxyValidateConnection(&pool_, &pool_.pools[1][2], 0), ncclInvalidArgument);  // past offset
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "not found"));
  ncclProxyConnection outside = {};
  EXPECT_EQ(ncclProxyValidateConnection(&pool_, &outside, 0), ncclInvalidArgument);
  free(pool_.pools[0]);
  pool_.pools[0] = nullptr;  // a NULL bank is skipped, not dereferenced
  EXPECT_EQ(ncclProxyValidateConnection(&pool_, &pool_.pools[1][1], 9), ncclSuccess);
}

TEST_F(ProxyConnPoolMicrotest, FreeConnections_FreesInitializedOnesThroughTheirTransportSide) {
  int id;
  for (int i = 0; i < NCCL_PROXY_CONN_POOL_SIZE + 3; i++) ASSERT_EQ(ncclProxyNewConnection(&pool_, &id), ncclSuccess);
  ncclProxyConnection* c = pool_.pools[0];
  c[1] = {1, 0};  // send, transport 0
  c[1].state = connInitialized;
  c[2].state = connConnected;  // recv, transport 0
  c[3] = {1, 1};  // send, transport 1: no proxyFree hook
  c[3].state = connConnected;
  c[4] = {0, 1};
  c[4].state = connSetupDone;
  pool_.pools[1][2].state = connConnected;  // last bank, within offset
  ncclProxyState state = {};
  ASSERT_EQ(ncclProxyFreeConnections(&pool_, &state), ncclSuccess);
  pool_ = {nullptr, 0, 0};
  ASSERT_EQ(g_tp.frees.size(), 3u);
  EXPECT_EQ(g_tp.frees[0].second, 1);
  EXPECT_EQ(g_tp.frees[1].second, 0);
  EXPECT_EQ(g_tp.frees[2].second, 0);
}

// The connections after the failing one are never freed. Pins AICOMRCCL-2964; a fix flips this pin.
TEST_F(ProxyConnPoolMicrotest, FreeConnections_KnownBug_StopsAtTheFirstTransportError) {
  int id;
  for (int i = 0; i < 3; i++) ASSERT_EQ(ncclProxyNewConnection(&pool_, &id), ncclSuccess);
  pool_.pools[0][0].state = connConnected;
  pool_.pools[0][1].state = connConnected;
  g_tp.freeResult = ncclRemoteError;
  ncclProxyState state = {};
  EXPECT_EQ(ncclProxyFreeConnections(&pool_, &state), ncclRemoteError);
  EXPECT_EQ(g_tp.frees.size(), 1u);  // the pool_ is left allocated; TearDown frees it
}

// Expected-response queue (client side of async RPCs)

class ProxyResponseQueueMicrotest : public ProxyFixture {
 protected:
  void TearDown() override {
    expectedProxyResponseFree(&state_);
    ProxyFixture::TearDown();
  }
  std::vector<void*> Ids() {
    std::vector<void*> ids;
    for (auto* e = state_.expectedResponses; e; e = e->next) ids.push_back(e->opId);
    return ids;
  }
  static void* Heap(int v) {
    int* p = (int*)std::malloc(sizeof(int));
    *p = v;
    return p;
  }
  ncclProxyState state_ = {};
  char a_, b_, c_;
};

TEST_F(ProxyResponseQueueMicrotest, Enqueue_AppendsInOrderWithAPendingInternalError) {
  ASSERT_EQ(expectedProxyResponseEnqueue(&state_, &a_, 4), ncclSuccess);
  ASSERT_EQ(expectedProxyResponseEnqueue(&state_, &b_, 0), ncclSuccess);
  ASSERT_EQ(expectedProxyResponseEnqueue(&state_, &c_, 8), ncclSuccess);
  EXPECT_EQ(Ids(), (std::vector<void*>{&a_, &b_, &c_}));
  auto* e = state_.expectedResponses;
  EXPECT_EQ(e->respSize, 4);
  EXPECT_FALSE(e->done);
  EXPECT_EQ(e->res, ncclInternalError);
  EXPECT_NE(e->respBuff, nullptr);
}

TEST_F(ProxyResponseQueueMicrotest, Enqueue_AllocationFailures) {
  g_callocFailAt = 0;
  EXPECT_EQ(expectedProxyResponseEnqueue(&state_, &a_, 4), ncclSystemError);
  g_mallocFailAt = 0;
  EXPECT_EQ(expectedProxyResponseEnqueue(&state_, &a_, 4), ncclSystemError);
  EXPECT_EQ(state_.expectedResponses, nullptr);
  g_mallocFailAt = 1;  // malloc(0) may return NULL: fine for an empty response
  EXPECT_EQ(expectedProxyResponseEnqueue(&state_, &a_, 0), ncclSuccess);
  EXPECT_EQ(Ids(), (std::vector<void*>{&a_}));
}

TEST_F(ProxyResponseQueueMicrotest, Store_CopiesAndFreesTheBufferAndMarksDone) {
  ASSERT_EQ(expectedProxyResponseEnqueue(&state_, &a_, 0), ncclSuccess);
  ASSERT_EQ(expectedProxyResponseEnqueue(&state_, &b_, sizeof(int)), ncclSuccess);
  ASSERT_EQ(expectedProxyResponseStore(&state_, &b_, Heap(42), sizeof(int), ncclRemoteError), ncclSuccess);
  auto* e = state_.expectedResponses->next;
  EXPECT_TRUE(e->done);
  EXPECT_EQ(e->res, ncclRemoteError);
  EXPECT_EQ(*(int*)e->respBuff, 42);
  ASSERT_EQ(expectedProxyResponseStore(&state_, &a_, nullptr, 0, ncclSuccess), ncclSuccess);
  EXPECT_TRUE(state_.expectedResponses->done);
}

TEST_F(ProxyResponseQueueMicrotest, Store_RejectsSizeMismatchRepeatsAndUnknownIds) {
  ASSERT_EQ(expectedProxyResponseEnqueue(&state_, &a_, sizeof(int)), ncclSuccess);
  void* buf = Heap(1);
  EXPECT_EQ(expectedProxyResponseStore(&state_, &a_, buf, 2, ncclSuccess), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "got 2, expected 4"));
  EXPECT_FALSE(state_.expectedResponses->done);
  ASSERT_EQ(expectedProxyResponseStore(&state_, &a_, buf, sizeof(int), ncclSuccess), ncclSuccess);
  buf = Heap(2);
  EXPECT_EQ(expectedProxyResponseStore(&state_, &a_, buf, sizeof(int), ncclSuccess), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "already completed"));
  EXPECT_EQ(expectedProxyResponseStore(&state_, &b_, buf, sizeof(int), ncclSuccess), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "doesn't match any expected response"));
  free(buf);
}

TEST_F(ProxyResponseQueueMicrotest, Dequeue_ReturnsOnlyCompletedResponsesAndUnlinksThem) {
  for (void* id : {(void*)&a_, (void*)&b_, (void*)&c_})
    ASSERT_EQ(expectedProxyResponseEnqueue(&state_, id, sizeof(int)), ncclSuccess);
  int out = 0, found = 7;
  EXPECT_EQ(expectedProxyResponseDequeue(&state_, &b_, &out, &found), ncclSuccess);
  EXPECT_EQ(found, 0);
  ASSERT_EQ(expectedProxyResponseStore(&state_, &b_, Heap(5), sizeof(int), ncclRemoteError), ncclSuccess);
  ASSERT_EQ(expectedProxyResponseStore(&state_, &a_, Heap(6), sizeof(int), ncclSuccess), ncclSuccess);
  EXPECT_EQ(expectedProxyResponseDequeue(&state_, &b_, &out, &found), ncclRemoteError);
  EXPECT_EQ(found, 1);
  EXPECT_EQ(out, 5);
  EXPECT_EQ(Ids(), (std::vector<void*>{&a_, &c_}));
  EXPECT_EQ(expectedProxyResponseDequeue(&state_, &a_, &out, &found), ncclSuccess);
  EXPECT_EQ(out, 6);
  EXPECT_EQ(Ids(), (std::vector<void*>{&c_}));
}

TEST_F(ProxyResponseQueueMicrotest, Remove_UnlinksHeadOrMiddleAndWarnsOnUnknown) {
  for (void* id : {(void*)&a_, (void*)&b_, (void*)&c_})
    ASSERT_EQ(expectedProxyResponseEnqueue(&state_, id, 1), ncclSuccess);
  EXPECT_EQ(expectedProxyResponseRemove(&state_, &b_), ncclSuccess);
  EXPECT_EQ(Ids(), (std::vector<void*>{&a_, &c_}));
  EXPECT_EQ(expectedProxyResponseRemove(&state_, &a_), ncclSuccess);
  EXPECT_EQ(Ids(), (std::vector<void*>{&c_}));
  EXPECT_EQ(expectedProxyResponseRemove(&state_, &a_), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Couldn't find opId"));
}

// Service-side async op queue

class ProxyAsyncOpQueueMicrotest : public ProxyFixture {
 protected:
  void TearDown() override {
    while (peer_.asyncOps) asyncProxyOpDequeue(&peer_, peer_.asyncOps);
    ProxyFixture::TearDown();
  }
  ncclProxyAsyncOp* Op(void* opId, bool bufs) {
    auto* op = (ncclProxyAsyncOp*)calloc(1, sizeof(ncclProxyAsyncOp));
    op->opId = opId;
    if (bufs) {
      op->reqBuff = (char*)std::malloc(4);
      op->respBuff = (char*)std::malloc(4);
    }
    return op;
  }
  std::vector<void*> Ids() {
    std::vector<void*> ids;
    for (auto* e = peer_.asyncOps; e; e = e->next) ids.push_back(e->opId);
    return ids;
  }
  ncclProxyLocalPeer peer_ = {};
  char a_, b_, c_;
};

TEST_F(ProxyAsyncOpQueueMicrotest, EnqueueDequeue_KeepsOrderAndFreesBuffers) {
  ASSERT_EQ(asyncProxyOpEnqueue(&peer_, Op(&a_, true)), ncclSuccess);
  ASSERT_EQ(asyncProxyOpEnqueue(&peer_, Op(&b_, false)), ncclSuccess);
  ASSERT_EQ(asyncProxyOpEnqueue(&peer_, Op(&c_, true)), ncclSuccess);
  EXPECT_EQ(Ids(), (std::vector<void*>{&a_, &b_, &c_}));
  ncclProxyAsyncOp key = {};
  key.opId = &b_;
  EXPECT_EQ(asyncProxyOpDequeue(&peer_, &key), ncclSuccess);
  EXPECT_EQ(Ids(), (std::vector<void*>{&a_, &c_}));
  EXPECT_EQ(asyncProxyOpDequeue(&peer_, peer_.asyncOps), ncclSuccess);
  EXPECT_EQ(Ids(), (std::vector<void*>{&c_}));
  key.opId = &a_;
  EXPECT_EQ(asyncProxyOpDequeue(&peer_, &key), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "dequeue nonexistent async opId"));
}

TEST_F(ProxyAsyncOpQueueMicrotest, Dequeue_NullOpOnAnEmptyQueueWarns) {
  EXPECT_EQ(asyncProxyOpDequeue(&peer_, nullptr), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "dequeue null operation"));
}

// Proxy state teardown

class ProxyDestroyMicrotest : public ProxyFixture {};

TEST_F(ProxyDestroyMicrotest, Destroy_FreesSharedArraysAndPendingResponses) {
  ncclSharedResources res = {};
  ncclComm& comm = *comm_;
  comm.sharedRes = &res;
  EXPECT_EQ(ncclProxyDestroy(&comm), ncclSuccess);
  res.proxyState = new ncclProxyState{};
  res.proxyState->refCount = 1;
  EXPECT_EQ(ncclProxyDestroy(&comm), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "refCount is 1, expected 0"));
  res.proxyState->refCount = 0;
  res.proxyState->peerAddresses = (ncclSocketAddress*)calloc(1, sizeof(ncclSocketAddress));
  res.proxyState->peerAddressesUDS = (uint64_t*)calloc(1, 8);
  res.proxyState->peerSocks = (ncclSocket*)calloc(1, sizeof(ncclSocket));
  res.proxyState->proxyOps = (ncclProxyOps*)calloc(1, sizeof(ncclProxyOps));
  res.proxyState->sharedDevMems = (void**)calloc(1, sizeof(void*));
  ASSERT_EQ(expectedProxyResponseEnqueue(res.proxyState, &res, 16), ncclSuccess);
  EXPECT_EQ(ncclProxyDestroy(&comm), ncclSuccess);
}

// Proxy-op queue (enqueue side): ncclLocalOpAppend / ncclProxyPost / ncclProxyStart / ncclProxySaveOp

class TestRing : public RingAlgorithm {
 public:
  void getNextSendAddr(int, uint8_t**, size_t*, void**) override {}
  void getNextRecvAddr(int, uint8_t**, size_t*, void**) override {}
  int refs() const { return refCount; }
};

// Threads each local rank's MAX_OPS_PER_PEER slice of the pool into its own free list, as proxyProgressInit does.
void InitFreeLists(ncclProxyOpsPool* pool, int nLocal) {
  pool->nextOps = pool->nextOpsEnd = -1;
  for (int r = 0; r < nLocal; r++) {
    pool->freeOps[r] = r * MAX_OPS_PER_PEER;
    for (int i = 0; i < MAX_OPS_PER_PEER - 1; i++) pool->ops[r * MAX_OPS_PER_PEER + i].next = r * MAX_OPS_PER_PEER + i + 1;
    pool->ops[(r + 1) * MAX_OPS_PER_PEER - 1].next = -1;
  }
}

class ProxyOpsFixture : public ProxyFixture {
 protected:
  static constexpr int kRanks = 8;
  struct Saved {
    int peer, send, connIndex, opPeer, nsteps, prevRank, nextRank;
    bool operator==(const Saved& o) const {
      return peer == o.peer && send == o.send && connIndex == o.connIndex && opPeer == o.opPeer &&
             nsteps == o.nsteps;
    }
  };

  void SetUp() override {
    ProxyFixture::SetUp();
    pool_ = (ncclProxyOpsPool*)calloc(1, sizeof(ncclProxyOpsPool));
    InitFreeLists(pool_, 2);
    for (auto& o : ops_) o = {pool_, nullptr, 0, -1, -1, -1};
    ps_.proxyOps = ops_;
    ps_.peerArraySize = 2;
    comm_->proxyState = &ps_;
    comm_->topParentLocalRanks = tpLocal_;
    comm_->nRanks = kRanks;
    comm_->rank = 0;
    comm_->localRanks = 1;
    for (int i = 0; i < kRanks; i++) userRanks_[i] = i;
    ncclChannel& ch = comm_->channels[0];
    ch.peers = peerPtrs_;
    ch.ring.userRanks = userRanks_;
    ch.ring.prev = kRanks - 1;
    ch.ring.next = 1;
    for (int p = 0; p < kRanks; p++) {
      peerPtrs_[p] = &peers_[p];
      for (int c = 0; c < NCCL_MAX_CONNS; c++) {
        for (int send = 0; send < 2; send++) {
          ncclConnector& conn = send ? peers_[p].send[c] : peers_[p].recv[c];
          conn.transportComm = send ? &g_transports[0].send : &g_transports[0].recv;
          conn.proxyConn.proxyProgress = FakeProxyProgress;
          conn.proxyConn.connection = &conns_[p][send][c];
        }
      }
    }
  }
  void TearDown() override {
    free(pool_);
    ProxyFixture::TearDown();
  }

  ncclProxyConnector* Conn(int peer, int send, int c = 0) {
    return send ? &peers_[peer].send[c].proxyConn : &peers_[peer].recv[c].proxyConn;
  }
  // The ops queued (not yet posted) for local proxy `r`, identified by the connection each was saved on.
  std::vector<Saved> Queued(int r = 0) {
    std::vector<Saved> out;
    for (int i = ops_[r].nextOps; i != -1; i = pool_->ops[i].next) {
      const ncclProxyOp& op = pool_->ops[i];
      int idx = (ncclProxyConnection*)op.connection - &conns_[0][0][0];
      out.push_back({idx / (2 * NCCL_MAX_CONNS), idx / NCCL_MAX_CONNS % 2, idx % NCCL_MAX_CONNS, op.peer, op.nsteps,
                     op.prevRank, op.nextRank});
      if (i == ops_[r].nextOpsEnd) break;
    }
    return out;
  }
  static Saved S(int peer, int send, int connIndex, int nsteps = 0) {
    return {peer, send, connIndex, peer, nsteps, 0, 0};
  }
  ncclProxyOp Op(ncclPattern_t pattern, int root = 0) {
    ncclProxyOp op = {};
    op.pattern = pattern;
    op.root = root;
    return op;
  }
  ncclResult_t Save(ncclProxyOp op) { return ncclProxySaveOp(comm_.get(), &op, nullptr); }
  bool Inquire(ncclProxyOp op) {
    bool need = true;
    EXPECT_EQ(ncclProxySaveOp(comm_.get(), &op, &need), ncclSuccess);
    return need;
  }

  ncclProxyOpsPool* pool_ = nullptr;
  ncclProxyOps ops_[2];
  ncclProxyState ps_ = {};
  int tpLocal_[2] = {0, 1};
  int userRanks_[kRanks];
  ncclChannelPeer peers_[kRanks] = {};
  ncclChannelPeer* peerPtrs_[kRanks];
  ncclProxyConnection conns_[kRanks][2][NCCL_MAX_CONNS] = {};
};

class ProxyLocalOpAppendMicrotest : public ProxyOpsFixture {};

TEST_F(ProxyLocalOpAppendMicrotest, LocalOpAppend_NoProxyOpsIsAnInternalError) {
  ps_.proxyOps = nullptr;
  ncclProxyOp op = {};
  EXPECT_EQ(ncclLocalOpAppend(comm_.get(), Conn(1, 1), &op), ncclInternalError);
}

TEST_F(ProxyLocalOpAppendMicrotest, LocalOpAppend_ClaimsOwnFreeListOnceThenChainsOpsInOrder) {
  comm_->localRank = 1;  // our ops live in the target pool's rank-1 slice
  Conn(2, 1)->tpLocalRank = 0;
  TestRing ring;
  ncclProxyOp op = {};
  op.opCount = 5;
  op.ringAlgo = &ring;
  op.next = 1234;
  ASSERT_EQ(ncclLocalOpAppend(comm_.get(), Conn(2, 1), &op), ncclSuccess);
  const int first = MAX_OPS_PER_PEER;
  EXPECT_EQ(pool_->freeOps[1], -1);  // the whole list moved to the client side
  EXPECT_EQ(pool_->freeOps[0], 0);
  EXPECT_EQ(ops_[0].freeOp, first + 1);
  EXPECT_EQ(ops_[0].nextOps, first);
  EXPECT_EQ(ops_[0].nextOpsEnd, first);
  EXPECT_EQ(ops_[0].count, 1);
  EXPECT_EQ(pool_->ops[first].opCount, 5u);
  EXPECT_EQ(pool_->ops[first].next, -1);
  EXPECT_EQ(pool_->ops[first].connection, Conn(2, 1)->connection);
  EXPECT_EQ(ring.refs(), 1);
  op.ringAlgo = nullptr;
  ASSERT_EQ(ncclLocalOpAppend(comm_.get(), Conn(2, 1), &op), ncclSuccess);
  EXPECT_EQ(ops_[0].freeOp, first + 2);
  EXPECT_EQ(ops_[0].nextOps, first);
  EXPECT_EQ(ops_[0].nextOpsEnd, first + 1);
  EXPECT_EQ(pool_->ops[first].next, first + 1);
  EXPECT_EQ(ops_[0].count, 2);
  EXPECT_EQ(ops_[1].nextOps, -1);
}

TEST_F(ProxyLocalOpAppendMicrotest, LocalOpAppend_WaitsUntilTheProxyReturnsOps) {
  pool_->freeOps[0] = -1;
  std::thread proxy([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    COMPILER_ATOMIC_STORE(&pool_->freeOps[0], 7, std::memory_order_release);
  });
  ncclProxyOp op = {};
  ASSERT_EQ(ncclLocalOpAppend(comm_.get(), Conn(1, 1), &op), ncclSuccess);
  proxy.join();
  EXPECT_EQ(ops_[0].nextOps, 7);
  EXPECT_EQ(ops_[0].freeOp, 8);
}

TEST_F(ProxyLocalOpAppendMicrotest, LocalOpAppend_FullQueuePostsAllButTheTrailingOpCount) {
  ncclProxyOp op = {};
  for (int i = 0; i < MAX_OPS_PER_PEER; i++) {
    op.opCount = i < MAX_OPS_PER_PEER - 3 ? i / 2 : 1000000;  // the last 3 ops share one opCount
    ASSERT_EQ(ncclLocalOpAppend(comm_.get(), Conn(1, 1), &op), ncclSuccess);
  }
  EXPECT_EQ(pool_->nextOps, 0);
  EXPECT_EQ(pool_->nextOpsEnd, MAX_OPS_PER_PEER - 4);
  EXPECT_EQ(pool_->ops[MAX_OPS_PER_PEER - 4].next, -1);
  EXPECT_EQ(ops_[0].count, 3);
  EXPECT_EQ(ops_[0].nextOps, MAX_OPS_PER_PEER - 3);
  EXPECT_EQ(ops_[0].nextOpsEnd, MAX_OPS_PER_PEER - 1);
}

TEST_F(ProxyLocalOpAppendMicrotest, LocalOpAppend_FullQueueOfASingleOpCountCannotBePosted) {
  ncclProxyOp op = {};
  op.opCount = 9;
  for (int i = 0; i < MAX_OPS_PER_PEER - 1; i++) ASSERT_EQ(ncclLocalOpAppend(comm_.get(), Conn(1, 1), &op), ncclSuccess);
  EXPECT_EQ(ncclLocalOpAppend(comm_.get(), Conn(1, 1), &op), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Unable to post incomplete proxy op chain"));
  EXPECT_EQ(pool_->nextOps, -1);
}

class ProxyPostMicrotest : public ProxyOpsFixture {};

TEST_F(ProxyPostMicrotest, Post_StartsOrExtendsThePoolChain) {
  ASSERT_EQ(ncclProxyPost(pool_, 3, 4), ncclSuccess);
  EXPECT_EQ(pool_->nextOps, 3);
  EXPECT_EQ(pool_->nextOpsEnd, 4);
  ASSERT_EQ(ncclProxyPost(pool_, 10, 12), ncclSuccess);
  EXPECT_EQ(pool_->nextOps, 3);
  EXPECT_EQ(pool_->ops[4].next, 10);
  EXPECT_EQ(pool_->nextOpsEnd, 12);
}

TEST_F(ProxyPostMicrotest, Start_PostsEveryPendingPeerQueueAndResetsIt) {
  ps_.proxyOps = nullptr;
  EXPECT_EQ(ncclProxyStart(comm_.get()), ncclSuccess);
  ps_.proxyOps = ops_;
  ncclProxyOp op = {};
  ASSERT_EQ(ncclLocalOpAppend(comm_.get(), Conn(1, 1), &op), ncclSuccess);
  ASSERT_EQ(ncclLocalOpAppend(comm_.get(), Conn(1, 1), &op), ncclSuccess);
  ops_[1].pool = nullptr;
  ops_[1].nextOps = 3;  // no pool: skipped
  ASSERT_EQ(ncclProxyStart(comm_.get()), ncclSuccess);
  EXPECT_EQ(pool_->nextOps, 0);
  EXPECT_EQ(pool_->nextOpsEnd, 1);
  EXPECT_EQ(ops_[0].nextOps, -1);
  EXPECT_EQ(ops_[0].nextOpsEnd, -1);
  EXPECT_EQ(ops_[0].count, 0);
  EXPECT_EQ(ops_[1].nextOps, 3);
  ASSERT_EQ(ncclProxyStart(comm_.get()), ncclSuccess);  // nothing pending: the pool chain is untouched
  EXPECT_EQ(pool_->nextOpsEnd, 1);
}

class ProxySaveOpMicrotest : public ProxyOpsFixture {};

TEST_F(ProxySaveOpMicrotest, SaveProxy_SkipsAbsentPeersAndConnectionsWithoutProgress) {
  ncclProxyOp op = {};
  bool need = false;
  EXPECT_EQ(SaveProxy(comm_.get(), &comm_->channels[0], proxySend, -1, &op, 0, &need), ncclSuccess);
  Conn(3, 0)->proxyProgress = nullptr;
  EXPECT_EQ(SaveProxy(comm_.get(), &comm_->channels[0], proxyRecv, 3, &op, 0, &need), ncclSuccess);
  EXPECT_FALSE(need);
  EXPECT_EQ(SaveProxy(comm_.get(), &comm_->channels[0], proxySend, 3, &op, 0, &need), ncclSuccess);
  EXPECT_TRUE(need);
  EXPECT_TRUE(Queued().empty());
  peers_[4].send[1].transportComm = nullptr;
  comm_->channels[0].id = 0;
  EXPECT_EQ(SaveProxy(comm_.get(), &comm_->channels[0], proxySend, 4, &op, 1, nullptr), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Rank 0 has no transport for send peer 4 on channel 0/1"));
  peers_[4].recv[0].transportComm = nullptr;
  EXPECT_EQ(SaveProxy(comm_.get(), &comm_->channels[0], proxyRecv, 4, &op, 0, nullptr), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "no transport for recv peer 4"));
}

TEST_F(ProxySaveOpMicrotest, Ring_SavesRecvFromPrevThenSendToNext) {
  comm_->commHash = 0xabc;
  ncclProxyOp op = Op(ncclPatternRingTwice);
  op.connIndex = 2;
  op.nsteps = 6;
  ASSERT_EQ(ncclProxySaveOp(comm_.get(), &op, nullptr), ncclSuccess);
  EXPECT_EQ(op.commHash, 0xabcu);
  auto q = Queued();
  ASSERT_EQ(q, (std::vector<Saved>{S(kRanks - 1, 0, 2, 6), S(1, 1, 2, 6)}));
  EXPECT_EQ(q[0].prevRank, kRanks - 1);
  EXPECT_EQ(q[0].nextRank, 1);
  EXPECT_TRUE(Inquire(Op(ncclPatternRing)));
}

TEST_F(ProxySaveOpMicrotest, Pipelines_SkipTheSideFacingTheRoot) {
  // Broadcast: the root never receives; the rank before the root never sends.
  ASSERT_EQ(Save(Op(ncclPatternPipelineFrom, /*root=*/0)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(1, 1, 0)}));
  ops_[0].nextOps = -1;
  ASSERT_EQ(Save(Op(ncclPatternPipelineFrom, /*root=*/1)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(kRanks - 1, 0, 0)}));
  ops_[0].nextOps = -1;
  // Reduce: the root never sends; the rank after the root never receives.
  ASSERT_EQ(Save(Op(ncclPatternPipelineTo, /*root=*/0)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(kRanks - 1, 0, 0)}));
  ops_[0].nextOps = -1;
  ASSERT_EQ(Save(Op(ncclPatternPipelineTo, /*root=*/kRanks - 1)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(1, 1, 0)}));
  ops_[0].nextOps = -1;
  ASSERT_EQ(Save(Op(ncclPatternPipelineTo, /*root=*/3)), ncclSuccess);
  EXPECT_EQ(Queued().size(), 2u);
}

TEST_F(ProxySaveOpMicrotest, AllGatherVRing_TakesStepsFromSlicesAndSkipsEmptySides) {
  ncclProxyOp op = Op(ncclPatternRing);
  op.coll = ncclFuncAllGatherV;
  op.specifics.bcast.recvSlices = 3;
  op.specifics.bcast.sendSlices = 0;
  ASSERT_EQ(Save(op), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(kRanks - 1, 0, 0, 3)}));
  ops_[0].nextOps = -1;
  op.specifics.bcast.recvSlices = 0;
  op.specifics.bcast.sendSlices = 4;
  ASSERT_EQ(Save(op), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(1, 1, 0, 4)}));
  op.specifics.bcast.sendSlices = 0;
  EXPECT_FALSE(Inquire(op));
  op.pattern = ncclPatternRingTwice;  // only plain Ring takes the slice counts
  EXPECT_TRUE(Inquire(op));
}

TEST_F(ProxySaveOpMicrotest, Trees_SaveUpAndDownPeersPerDirection) {
  ncclTree& t = comm_->channels[0].tree;
  t.up = 2;
  t.down[0] = 3;
  t.down[1] = -1;
  t.down[2] = 5;
  ASSERT_EQ(Save(Op(ncclPatternTreeUp)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(3, 0, 0), S(5, 0, 0), S(2, 1, 0)}));
  ops_[0].nextOps = -1;
  ASSERT_EQ(Save(Op(ncclPatternTreeDown)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(3, 1, 0), S(5, 1, 0), S(2, 0, 0)}));
  ops_[0].nextOps = -1;
  ASSERT_EQ(Save(Op(ncclPatternTreeUpDown)), ncclSuccess);
  EXPECT_EQ(Queued().size(), 6u);
  t.up = -1;
  t.down[0] = t.down[2] = -1;
  EXPECT_FALSE(Inquire(Op(ncclPatternTreeUpDown)));
}

TEST_F(ProxySaveOpMicrotest, CollnetAndNvls_SaveSendOnConn1AndRecvOnConn0) {
  comm_->channels[0].collnetChain.up = 2;
  comm_->channels[0].collnetDirect.out = 3;
  comm_->channels[0].nvls.out = 4;
  ASSERT_EQ(Save(Op(ncclPatternCollnetChain)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(2, 1, 1), S(2, 0, 0)}));
  ops_[0].nextOps = -1;
  ASSERT_EQ(Save(Op(ncclPatternCollnetDirect)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(3, 1, 1), S(3, 0, 0)}));
  ops_[0].nextOps = -1;
  ASSERT_EQ(Save(Op(ncclPatternNvls)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(4, 1, 1), S(4, 0, 0)}));
}

TEST_F(ProxySaveOpMicrotest, NvlsTree_SavesBothTreeDownPeersAndTheTreeUpPeer) {
  ncclNvls& n = comm_->channels[0].nvls;
  n.treeDown[0] = 7;  // index 0 is never a proxy peer
  n.treeDown[1] = 2;
  n.treeDown[2] = 3;
  n.treeUp = 4;
  ASSERT_EQ(Save(Op(ncclPatternNvlsTree)), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(2, 0, 0), S(3, 0, 0), S(4, 1, 0), S(2, 1, 0), S(3, 1, 0), S(4, 0, 0)}));
}

TEST_F(ProxySaveOpMicrotest, SendRecv_SaveTheRootPeerUnlessItIsSelf) {
  ncclProxyOp op = Op(ncclPatternSend, 3);
  op.connIndex = 1;
  ASSERT_EQ(Save(op), ncclSuccess);
  op = Op(ncclPatternRecv, 5);
  ASSERT_EQ(Save(op), ncclSuccess);
  EXPECT_EQ(Queued(), (std::vector<Saved>{S(3, 1, 1), S(5, 0, 0)}));
  EXPECT_FALSE(Inquire(Op(ncclPatternSend, 0)));
  EXPECT_FALSE(Inquire(Op((ncclPattern_t)99)));  // unknown patterns need no proxy
}

TEST_F(ProxySaveOpMicrotest, Errors_FromSaveProxyPropagate) {
  peers_[kRanks - 1].recv[0].transportComm = nullptr;
  EXPECT_EQ(Save(Op(ncclPatternRing)), ncclInternalError);
  peers_[kRanks - 1].recv[0].transportComm = &g_transports[0].recv;
  peers_[1].send[0].transportComm = nullptr;
  EXPECT_EQ(Save(Op(ncclPatternRing)), ncclInternalError);
  comm_->channels[0].tree = {0, 1, {-1, -1, -1}};
  EXPECT_EQ(Save(Op(ncclPatternTreeUp)), ncclInternalError);
  EXPECT_EQ(Save(Op(ncclPatternTreeDown)), ncclSuccess);  // recv from up=1 is fine; no down peers
  comm_->channels[0].tree = {0, -1, {1, -1, -1}};
  EXPECT_EQ(Save(Op(ncclPatternTreeDown)), ncclInternalError);
  peers_[2].recv[0].transportComm = nullptr;
  comm_->channels[0].tree = {0, 3, {2, -1, -1}};
  EXPECT_EQ(Save(Op(ncclPatternTreeUp)), ncclInternalError);
  comm_->channels[0].tree = {0, 2, {-1, -1, -1}};
  EXPECT_EQ(Save(Op(ncclPatternTreeDown)), ncclInternalError);
  peers_[1].send[1].transportComm = nullptr;  // fails the send save; peer 2 then fails the recv save
  comm_->channels[0].collnetChain.up = comm_->channels[0].collnetDirect.out = comm_->channels[0].nvls.out = 1;
  EXPECT_EQ(Save(Op(ncclPatternCollnetChain)), ncclInternalError);
  EXPECT_EQ(Save(Op(ncclPatternCollnetDirect)), ncclInternalError);
  EXPECT_EQ(Save(Op(ncclPatternNvls)), ncclInternalError);
  comm_->channels[0].collnetChain.up = comm_->channels[0].collnetDirect.out = comm_->channels[0].nvls.out = 2;
  EXPECT_EQ(Save(Op(ncclPatternCollnetChain)), ncclInternalError);
  EXPECT_EQ(Save(Op(ncclPatternCollnetDirect)), ncclInternalError);
  EXPECT_EQ(Save(Op(ncclPatternNvls)), ncclInternalError);
  EXPECT_EQ(Save(Op(ncclPatternSend, 1)), ncclInternalError);
}

TEST_F(ProxySaveOpMicrotest, NvlsTree_ErrorsFromEachSavePropagate) {
  ncclNvls& n = comm_->channels[0].nvls;
  n.treeDown[1] = n.treeDown[2] = n.treeUp = -1;
  for (int which = 0; which < 6; which++) {
    int* slot[] = {&n.treeDown[1], &n.treeDown[2], &n.treeUp};
    int dir[] = {0, 0, 1, 1, 1, 0};  // send side of each SaveProxy in order
    *slot[which % 3] = 1;
    (dir[which] ? peers_[1].send[0] : peers_[1].recv[0]).transportComm = nullptr;
    EXPECT_EQ(Save(Op(ncclPatternNvlsTree)), ncclInternalError) << which;
    peers_[1].send[0].transportComm = &g_transports[0].send;
    peers_[1].recv[0].transportComm = &g_transports[0].recv;
    *slot[which % 3] = -1;
  }
}

class ProxySavePatMicrotest : public ProxyOpsFixture {
 protected:
  void SetUp() override {
    ProxyOpsFixture::SetUp();
    comm_->nRanks = 4;
    comm_->nNodes = 4;
    comm_->node = 0;
    for (int i = 0; i < kRanks; i++) dense_[i] = kRanks - 1 - i;
  }
  std::vector<std::pair<int, int>> Peers() {  // (send, peer)
    std::vector<std::pair<int, int>> out;
    for (const Saved& s : Queued()) {
      EXPECT_GT(s.nsteps, 0);
      EXPECT_EQ(s.connIndex, 0);
      out.push_back({s.send, s.peer});
    }
    return out;
  }
  ncclProxyOp PatOp(ncclPattern_t p) {
    ncclProxyOp op = Op(p);
    op.nbytes = 4 << 20;
    op.chunkSize = 1 << 16;
    return op;
  }
  int dense_[kRanks];
};

TEST_F(ProxySavePatMicrotest, PatUp_SendsToHigherNodesAndReceivesFromLowerOnes) {
  ASSERT_EQ(Save(PatOp(ncclPatternPatUp)), ncclSuccess);
  // Dimension i pairs node+2^i (send) with node-2^i (recv).
  EXPECT_EQ(Peers(), (std::vector<std::pair<int, int>>{{1, 1}, {0, 3}, {1, 2}, {0, 2}}));
}

TEST_F(ProxySavePatMicrotest, PatDown_MirrorsPatUpUnlessQpsAreShared) {
  ASSERT_EQ(Save(PatOp(ncclPatternPatDown)), ncclSuccess);
  auto own = Peers();
  EXPECT_EQ(own, (std::vector<std::pair<int, int>>{{1, 3}, {0, 1}, {1, 2}, {0, 2}}));
  ops_[0].nextOps = -1;
  comm_->patSharedQps = true;
  ASSERT_EQ(Save(PatOp(ncclPatternPatDown)), ncclSuccess);
  EXPECT_EQ(Peers(), (std::vector<std::pair<int, int>>{{1, 1}, {0, 3}, {1, 2}, {0, 2}}));
}

TEST_F(ProxySavePatMicrotest, Pat_MapsDensePeersThroughHeadRankAndDenseToUserRank) {
  comm_->nRanks = 8;
  comm_->nNodes = 2;
  comm_->localRanks = 4;
  comm_->channels[0].nvls.headRank = 1;
  comm_->denseToUserRank = dense_;
  ASSERT_EQ(Save(PatOp(ncclPatternPatUp)), ncclSuccess);
  // dense peer = node 1 * 4 local ranks + head rank 1 = 5, mapped to user rank 8-1-5 = 2.
  EXPECT_EQ(Peers(), (std::vector<std::pair<int, int>>{{1, 2}, {0, 2}}));
  comm_->isOneRPN = true;
  for (ncclPattern_t p : {ncclPatternPatUp, ncclPatternPatDown}) {
    ops_[0].nextOps = -1;
    ASSERT_EQ(Save(PatOp(p)), ncclSuccess);
    EXPECT_EQ(Peers(), (std::vector<std::pair<int, int>>{{1, 2}, {0, 2}}));
  }
}

TEST_F(ProxySavePatMicrotest, Pat_AllocationAndSaveErrorsPropagate) {
  for (ncclPattern_t p : {ncclPatternPatUp, ncclPatternPatDown}) {
    for (int fail = 0; fail < 2; fail++) {
      g_callocCalls = 0;
      g_callocFailAt = fail;
      EXPECT_EQ(Save(PatOp(p)), ncclSystemError);
    }
    g_callocFailAt = -1;
    peers_[1].send[0].transportComm = nullptr;
    peers_[3].send[0].transportComm = nullptr;
    EXPECT_EQ(Save(PatOp(p)), ncclInternalError);
    peers_[1].send[0].transportComm = peers_[3].send[0].transportComm = &g_transports[0].send;
    peers_[1].recv[0].transportComm = peers_[3].recv[0].transportComm = nullptr;
    EXPECT_EQ(Save(PatOp(p)), ncclInternalError);
    peers_[1].recv[0].transportComm = peers_[3].recv[0].transportComm = &g_transports[0].recv;
  }
}

// Progress side: args pool, ProxyAppend, removeOp, progressOps, ncclProxyGetPostedOps, the progress thread

class ProxyProgressFixture : public ProxyFixture {
 protected:
  static constexpr int kArgs = 8;
  void SetUp() override {
    ProxyFixture::SetUp();
    // A small pre-chained free list: allocateArgs' real pool is ~280 MB, which only its own test pays for.
    args_ = (ncclProxyArgs*)calloc(kArgs, sizeof(ncclProxyArgs));
    for (int i = 0; i < kArgs - 1; i++) args_[i].next = &args_[i + 1];
    st().pool = args_;
    st().nextOps = -1;
    ps_.abortFlag = &abort_;
    ps_.tpLocalnRanks = 2;
    for (int i = 0; i < 4; i++) {
      conns_[i].send = i % 2;
      conns_[i].tcomm = &g_transports[0].send;
      conns_[i].proxyAppendPtr = &conns_[i].proxyAppend;
    }
  }
  void TearDown() override {
    if (st().thread.joinable()) {
      std::unique_lock<std::mutex> lock(opsPool_->mutex);  // under the lock, so the wakeup cannot be lost
      COMPILER_ATOMIC_STORE(&abort_, 1, std::memory_order_release);
      st().stop.store(2);
      opsPool_->cond.notify_all();
      lock.unlock();
      st().thread.join();
    }
    free(args_);
    free(opsPool_);
    ProxyFixture::TearDown();
  }
  ncclProxyProgressState& st() { return ps_.progressState; }
  ncclProxyOp Op(int conn, uint64_t opCount) {
    ncclProxyOp op = {};
    op.connection = &conns_[conn];
    op.opCount = opCount;
    op.nsteps = 4;
    op.sliceSteps = 2;
    return op;
  }
  std::vector<ncclProxyArgs*> Active() {
    std::vector<ncclProxyArgs*> out;
    for (ncclProxyArgs* a = st().active; a; a = a->next) out.push_back(a);
    return out;
  }
  int FreeCount() {
    int n = 0;
    for (ncclProxyArgs* a = st().pool; a; a = a->next) n++;
    return n;
  }
  void MakeOpsPool() {
    opsPool_ = (ncclProxyOpsPool*)calloc(1, sizeof(ncclProxyOpsPool));
    InitFreeLists(opsPool_, 2);
    st().opsPool = opsPool_;
  }
  // Fills pool slots the way clients do and returns {first, last}; slot = localRank * MAX_OPS_PER_PEER + i.
  std::pair<int, int> Chain(std::vector<std::tuple<int, int, uint64_t>> ops) {  // (localRank, conn, opCount)
    int first = -1, prev = -1;
    std::map<int, int> used;
    for (auto& [r, c, opCount] : ops) {
      int idx = r * MAX_OPS_PER_PEER + 100 + used[r]++;
      opsPool_->ops[idx] = Op(c, opCount);
      opsPool_->ops[idx].next = -1;
      if (prev == -1) first = idx;
      else opsPool_->ops[prev].next = idx;
      prev = idx;
    }
    return {first, prev};
  }

  ncclProxyState ps_ = {};
  ncclProxyArgs* args_ = nullptr;
  ncclProxyOpsPool* opsPool_ = nullptr;
  ncclProxyConnection conns_[4] = {};
  uint32_t abort_ = 0;
};

class ProxyArgsPoolMicrotest : public ProxyProgressFixture {
 protected:
  // Pools built here are calloc'd (lazily zeroed), not ncclCalloc'd, so they stay cheap.
  ncclProxyPool* NewPool(ncclProxyPool* next) {
    auto* p = (ncclProxyPool*)calloc(1, sizeof(ncclProxyPool));
    p->next = next;
    return p;
  }
};

TEST_F(ProxyArgsPoolMicrotest, AllocateArgs_PopsTheFreeListAndGrowsByAWholePool) {
  ncclProxyArgs* a = nullptr;
  args_[0].nextPeer = &args_[3];
  ASSERT_EQ(allocateArgs(&st(), &a), ncclSuccess);
  EXPECT_EQ(a, &args_[0]);
  EXPECT_EQ(a->next, nullptr);
  EXPECT_EQ(a->nextPeer, nullptr);
  EXPECT_EQ(st().pool, &args_[1]);
  st().pool = nullptr;
  g_callocFailAt = 0;
  EXPECT_EQ(allocateArgs(&st(), &a), ncclSystemError);
  g_callocFailAt = -1;
  ASSERT_EQ(allocateArgs(&st(), &a), ncclSuccess);
  ASSERT_NE(st().pools, nullptr);
  EXPECT_EQ(a, st().pools->elems);
  EXPECT_EQ(st().pool, st().pools->elems + 1);
  EXPECT_EQ(st().pools->elems[PROXYARGS_ALLOCATE_SIZE - 2].next, st().pools->elems + PROXYARGS_ALLOCATE_SIZE - 1);
  EXPECT_EQ(st().pools->elems[PROXYARGS_ALLOCATE_SIZE - 1].next, nullptr);
  ASSERT_EQ(ncclProxyProgressDestroy(&ps_), ncclSuccess);  // no ops pool: just frees the args pools
  EXPECT_EQ(st().pools, nullptr);
}

TEST_F(ProxyArgsPoolMicrotest, GetOpIndex_LocatesAnOpInAnyPool) {
  st().pools = NewPool(NewPool(nullptr));
  int pool = -1, op = -1;
  ASSERT_EQ(getOpIndex(st().pools->next->elems + 9, &st(), &pool, &op), ncclSuccess);
  EXPECT_EQ(pool, 1);
  EXPECT_EQ(op, 9);
  ASSERT_EQ(getOpIndex(st().pools->elems + PROXYARGS_ALLOCATE_SIZE - 1, &st(), &pool, &op), ncclSuccess);
  EXPECT_EQ(pool, 0);
  EXPECT_EQ(op, PROXYARGS_ALLOCATE_SIZE - 1);
  EXPECT_EQ(getOpIndex(args_, &st(), &pool, &op), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Could not find pool of op"));
  st().pool = args_;
  ASSERT_EQ(ncclProxyProgressDestroy(&ps_), ncclSuccess);
}

TEST_F(ProxyArgsPoolMicrotest, DumpProxyState_WalksActiveAndFreeListsAndClearsMarks) {
  st().pools = NewPool(nullptr);
  ncclProxyArgs* e = st().pools->elems;
  auto freeFrom = [&](int i) {  // e[i..] is the free list
    st().pool = &e[i];
    for (; i < PROXYARGS_ALLOCATE_SIZE - 1; i++) e[i].next = &e[i + 1];
  };
  // active: e0 (nextPeer e1 -> e2), e3; free: e6..; e4 and e5 are in no list.
  st().active = &e[0];
  e[0].next = &e[3];
  e[0].nextPeer = &e[1];
  e[1].nextPeer = &e[2];
  e[2].next = &e[3];
  e[0].state = ncclProxyOpProgress;
  e[0].nsubs = 1;
  e[3].state = ncclProxyOpReady;
  freeFrom(6);
  ncclLastProxyState = &st();
  ncclDumpProxyState(SIGUSR2);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_ATTN, "Inactive op has next set!"));
  EXPECT_FALSE(ProxyLogged(NCCL_LOG_ATTN, "List loop"));
  for (int i = 0; i < PROXYARGS_ALLOCATE_SIZE; i++) ASSERT_EQ(e[i].state & OP_SEEN, 0) << i;
  // Duplicates across lists are reported: e1 is both e0's nextPeer and active, then e3's nextPeer;
  // e3 is both active and free.
  e[1] = e[2] = ncclProxyArgs{};
  e[0].next = &e[1];
  e[1].next = &e[3];
  e[3].nextPeer = &e[1];
  freeFrom(4);
  st().pool = &e[2];
  e[2].next = &e[4];
  e[PROXYARGS_ALLOCATE_SIZE - 1].next = &e[3];
  ASSERT_EQ(dumpProxyState(&st()), ncclSuccess);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_ATTN, "List loop at element 0-1"));
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_ATTN, "List loop at element 0-3"));
  // Ops outside every pool abort the dump, whichever list they are on.
  e[PROXYARGS_ALLOCATE_SIZE - 1].next = args_;
  EXPECT_EQ(dumpProxyState(&st()), ncclInternalError);
  e[3].nextPeer = args_;
  EXPECT_EQ(dumpProxyState(&st()), ncclInternalError);
  st().active = args_;
  EXPECT_EQ(dumpProxyState(&st()), ncclInternalError);
  st().active = nullptr;
  st().pool = args_;
  ASSERT_EQ(ncclProxyProgressDestroy(&ps_), ncclSuccess);
}

TEST_F(ProxyArgsPoolMicrotest, PrintProxyOp_ReportsEveryStatusLetter) {
  ncclProxyArgs& a = args_[0];
  a.nsubs = 1;
  ncclProxySubArgs& s = a.subs[0];
  a.state = ncclProxyOpProgress;
  a.retry_total = 3;
  struct Case {
    uint8_t pattern;
    uint64_t posted, transmitted, received, done;
    int nsteps;
    const char* expected;
  };
  for (const Case& c : std::vector<Case>{{ncclPatternRecv, 0, 0, 0, 0, 4, "RecvI peer:"},
                                         {ncclPatternRecv, 8, 8, 8, 0, 16, "RecvG peer:"},
                                         {ncclPatternSend, 8, 8, 0, 0, 16, "SendS peer:"},
                                         {ncclPatternRecv, 4, 0, 2, 0, 4, "RecvR peer:"},
                                         {ncclPatternRecv, 4, 4, 4, 0, 4, "RecvG peer:"},
                                         {ncclPatternRecv, 4, 5, 4, 0, 4, "RecvR peer:"},
                                         {ncclPatternRecv, 4, 3, 4, 3, 4, "RecvF peer:"},
                                         {ncclPatternRecv, 4, 4, 4, 4, 4, "RecvD peer:"},
                                         {ncclPatternSend, 0, 0, 0, 0, 4, "SendI peer:"},
                                         {ncclPatternSend, 4, 2, 0, 0, 4, "SendG peer:"},
                                         {ncclPatternSend, 4, 4, 0, 2, 4, "SendS peer:"},
                                         {ncclPatternSend, 4, 4, 0, 4, 4, "SendD peer:"},
                                         {ncclPatternRing, 1, 1, 1, 1, 4,
                                          "Coll->Recv nb:0 ns:4 p:1 t:1 r:1, d:1   peer:"}}) {
    a.pattern = c.pattern;
    a.send = c.pattern == ncclPatternSend;
    s.posted = c.posted;
    s.transmitted = c.transmitted;
    s.received = c.received;
    s.done = c.done;
    s.nsteps = c.nsteps;
    ncclResult_t res = ncclInternalError;
    std::string out = RcclUnitTesting::CaptureLog([&] { res = printProxyOp(&a, 0, 0); });
    EXPECT_EQ(res, ncclSuccess);
    EXPECT_TRUE(RcclUnitTesting::LogHas(out, c.expected)) << out;
    EXPECT_TRUE(RcclUnitTesting::LogHas(out, "(retries:3)]\n")) << out;
  }
  a.pattern = ncclPatternRecv;
  a.send = 0;
  for (auto [state, expected] : std::vector<std::pair<int, const char*>>{
         {ncclProxyOpNone, "Recv\t[] peer:"}, {ncclProxyOpReady, "Recv\t[R] peer:"}, {7, "Recv\t[UNDEFINED] peer:"}}) {
    a.state = state;
    std::string out = RcclUnitTesting::CaptureLog([&] { EXPECT_EQ(printProxyOp(&a, 0, 0), ncclSuccess); });
    EXPECT_TRUE(RcclUnitTesting::LogHas(out, expected)) << out;
  }
}

class ProxyAppendMicrotest : public ProxyProgressFixture {};

TEST_F(ProxyAppendMicrotest, OpToArgs_FirstSubCopiesTheOpAndLaterSubsMustMatch) {
  ncclProxyArgs& a = args_[0];
  ncclProxyOp op = Op(1, 7);
  op.nChannels = 4;
  op.nPeers = 3;
  op.prevRank = 5;
  op.nextRank = 6;
  op.commHash = 99;
  op.channelId = 2;
  ASSERT_EQ(ncclProxyOpToArgs(&op, &a, 0), ncclSuccess);
  EXPECT_EQ(a.nsubs, 1);
  EXPECT_EQ(a.opCount, 7u);
  EXPECT_EQ(a.state, ncclProxyOpReady);
  EXPECT_EQ(a.progress, FakeProxyProgress);
  EXPECT_EQ(a.proxyAppendPtr, &conns_[1].proxyAppend);
  EXPECT_EQ(a.send, 1);
  EXPECT_EQ(a.prevRank, 5);
  EXPECT_EQ(a.nextRank, 6);
  EXPECT_EQ(a.subs[0].channelId, 2);
  EXPECT_EQ(a.subs[0].nsteps, 4);
  EXPECT_EQ(a.subs[0].commHash, 0u);
  EXPECT_EQ(g_profOpStarts, 1);
  g_proxyDiagEnabled = true;
  op.nChannels = 2;
  op.nPeers = 9;
  op.totalBytes = 1234;
  ASSERT_EQ(ncclProxyOpToArgs(&op, &a, 1), ncclSuccess);
  EXPECT_EQ(a.nsubs, 2);
  EXPECT_EQ(a.nChannels, 2);
  EXPECT_EQ(a.nPeers, 3);
  EXPECT_EQ(a.subs[1].commHash, 99u);
  EXPECT_EQ(a.subs[1].profExtras.totalBytes, 1234u);
  for (int field = 0; field < 6; field++) {
    ncclProxyOp bad = op;
    uint8_t* f[] = {&bad.sliceSteps, &bad.chunkSteps, &bad.protocol, &bad.dtype, &bad.redOp, &bad.coll};
    (*f[field])++;
    EXPECT_EQ(ncclProxyOpToArgs(&bad, &a, 2), ncclInternalError) << field;
  }
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Proxy append mismatch"));
  a.state = ncclProxyOpProgress;
  EXPECT_EQ(ncclProxyOpToArgs(&op, &a, 2), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Proxy append on running operation"));
  EXPECT_EQ(ncclProxyOpToArgs(&op, &a, NCCL_PROXY_MAX_SUBS), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Proxy append out of bounds"));
}

TEST_F(ProxyAppendMicrotest, Append_NewPeersJoinTheActiveListInOrder) {
  ncclProxyOp op0 = Op(0, 1), op1 = Op(1, 1), op2 = Op(2, 1);
  ASSERT_EQ(ProxyAppend(&st(), &op0), ncclSuccess);
  ASSERT_EQ(ProxyAppend(&st(), &op1), ncclSuccess);
  ASSERT_EQ(ProxyAppend(&st(), &op2), ncclSuccess);
  EXPECT_EQ(Active(), (std::vector<ncclProxyArgs*>{&args_[0], &args_[1], &args_[2]}));
  EXPECT_EQ(conns_[1].proxyAppend, &args_[1]);
}

TEST_F(ProxyAppendMicrotest, Append_SharedSameOpCountBecomesASubElseANextPeer) {
  conns_[0].shared = 1;
  ncclProxyOp op = Op(0, 3);
  ASSERT_EQ(ProxyAppend(&st(), &op), ncclSuccess);
  ASSERT_EQ(ProxyAppend(&st(), &op), ncclSuccess);
  EXPECT_EQ(args_[0].nsubs, 2);
  op.opCount = 4;
  ASSERT_EQ(ProxyAppend(&st(), &op), ncclSuccess);
  EXPECT_EQ(args_[0].nextPeer, &args_[1]);
  EXPECT_EQ(conns_[0].proxyAppend, &args_[1]);
  conns_[0].shared = 0;  // unshared connections never merge, even on equal opCount
  ASSERT_EQ(ProxyAppend(&st(), &op), ncclSuccess);
  EXPECT_EQ(args_[1].nextPeer, &args_[2]);
  EXPECT_EQ(Active(), (std::vector<ncclProxyArgs*>{&args_[0]}));
}

TEST_F(ProxyAppendMicrotest, Append_ErrorsPropagate) {
  conns_[0].shared = 1;
  ncclProxyOp op = Op(0, 3);
  ASSERT_EQ(ProxyAppend(&st(), &op), ncclSuccess);
  op.coll++;
  EXPECT_EQ(ProxyAppend(&st(), &op), ncclInternalError);  // mismatched sub
  st().pool = nullptr;
  g_callocFailAt = g_callocCalls;
  op.opCount = 5;
  EXPECT_EQ(ProxyAppend(&st(), &op), ncclSystemError);  // nextPeer allocation
  EXPECT_EQ(args_[0].nextPeer, nullptr);
  g_callocFailAt = g_callocCalls;
  ncclProxyOp other = Op(1, 1);
  EXPECT_EQ(ProxyAppend(&st(), &other), ncclSystemError);  // new list entry allocation
  EXPECT_EQ(Active(), (std::vector<ncclProxyArgs*>{&args_[0]}));
}

// ncclProxyOpToArgs bumps nsubs before the mismatch check rejects the sub.
TEST_F(ProxyAppendMicrotest, Append_CountsAMismatchedSubBeforeRejectingIt) {
  conns_[0].shared = 1;
  ncclProxyOp op = Op(0, 3);
  ASSERT_EQ(ProxyAppend(&st(), &op), ncclSuccess);
  op.coll++;
  EXPECT_EQ(ProxyAppend(&st(), &op), ncclInternalError);
  EXPECT_EQ(args_[0].nsubs, 2);
}

class ProxyProgressOpsMicrotest : public ProxyProgressFixture {};

TEST_F(ProxyProgressOpsMicrotest, RemoveOp_PromotesNextPeerOrUnlinksAndRecycles) {
  ncclProxyOp a = Op(0, 1), b = Op(1, 1), b2 = Op(1, 2), c = Op(2, 1);
  for (ncclProxyOp* op : {&a, &b, &b2, &c}) ASSERT_EQ(ProxyAppend(&st(), op), ncclSuccess);
  // active: A0 -> B1(nextPeer B2) -> C3
  ncclProxyArgs *prev = &args_[0], *op = &args_[1];
  ASSERT_EQ(removeOp(&st(), &op, &prev), ncclSuccess);
  EXPECT_EQ(Active(), (std::vector<ncclProxyArgs*>{&args_[0], &args_[2], &args_[3]}));
  EXPECT_EQ(prev, &args_[2]);
  EXPECT_EQ(op, &args_[3]);
  EXPECT_EQ(st().pool, &args_[1]);
  EXPECT_EQ(conns_[1].proxyAppend, &args_[2]);
  prev = nullptr;
  op = &args_[0];
  ASSERT_EQ(removeOp(&st(), &op, &prev), ncclSuccess);  // head without nextPeer
  EXPECT_EQ(Active(), (std::vector<ncclProxyArgs*>{&args_[2], &args_[3]}));
  EXPECT_EQ(conns_[0].proxyAppend, nullptr);
  EXPECT_EQ(prev, nullptr);
  prev = &args_[2];
  op = &args_[3];
  ASSERT_EQ(removeOp(&st(), &op, &prev), ncclSuccess);  // middle without nextPeer
  EXPECT_EQ(Active(), (std::vector<ncclProxyArgs*>{&args_[2]}));
  ncclProxyOp b3 = Op(1, 3);
  ASSERT_EQ(ProxyAppend(&st(), &b3), ncclSuccess);
  prev = nullptr;
  op = &args_[2];
  ASSERT_EQ(removeOp(&st(), &op, &prev), ncclSuccess);  // head with nextPeer
  EXPECT_EQ(st().active, conns_[1].proxyAppend);
  EXPECT_EQ(FreeCount(), kArgs - 1);
}

TEST_F(ProxyProgressOpsMicrotest, ProgressOps_RemovesFinishedOrFailedOpsAndReportsTheFirstError) {
  for (int c = 0; c < 4; c++) {
    ncclProxyOp op = Op(c, 1);
    ASSERT_EQ(ProxyAppend(&st(), &op), ncclSuccess);
  }
  ScopedHook progress(g_progressFn, [&](ncclProxyArgs* a) {
    int i = a - args_;
    a->idle = i != 1;
    if (i == 0) a->state = ncclProxyOpNone;
    return i == 2 ? ncclRemoteError : i == 3 ? ncclSystemError : ncclSuccess;
  });
  int idle = 1;
  EXPECT_EQ(progressOps(&ps_, &st(), st().active, &idle), ncclRemoteError);
  EXPECT_EQ(idle, 0);
  EXPECT_EQ(Active(), (std::vector<ncclProxyArgs*>{&args_[1]}));
  EXPECT_EQ(args_[1].retry_total, 1);
  idle = 1;
  ScopedHook allIdle(g_progressFn, [](ncclProxyArgs* a) {
    a->idle = 1;
    return ncclSuccess;
  });
  EXPECT_EQ(progressOps(&ps_, &st(), st().active, &idle), ncclSuccess);
  EXPECT_EQ(idle, 1);
  EXPECT_EQ(args_[1].retry_total, 2);
  args_[1].state = ncclProxyOpNone;
  EXPECT_EQ(progressOps(&ps_, &st(), st().active, &idle), ncclInternalError);
}

class ProxyGetPostedOpsMicrotest : public ProxyProgressFixture {};

TEST_F(ProxyGetPostedOpsMicrotest, GetPostedOps_NoOpsPoolIsAnInternalError) {
  int added = 0;
  EXPECT_EQ(ncclProxyGetPostedOps(&ps_, &added), ncclInternalError);
}

TEST_F(ProxyGetPostedOpsMicrotest, GetPostedOps_AppendsPostedOpsAndReturnsSlotsToEachPeerFreeList) {
  MakeOpsPool();
  auto [first, last] = Chain({{0, 0, 1}, {0, 1, 1}, {1, 2, 1}});
  opsPool_->nextOps = first;
  opsPool_->nextOpsEnd = last;
  opsPool_->freeOps[0] = 7;  // the client still holds 7; returned slots go in front of it
  opsPool_->freeOps[1] = -1;
  int added = 0;
  st().stop.store(1);  // must not block even with nothing active
  ASSERT_EQ(ncclProxyGetPostedOps(&ps_, &added), ncclSuccess);
  EXPECT_EQ(added, 3);
  EXPECT_EQ(Active().size(), 3u);
  EXPECT_EQ(opsPool_->nextOps, -1);
  EXPECT_EQ(opsPool_->nextOpsEnd, -1);
  EXPECT_EQ(st().nextOps, -1);
  EXPECT_EQ(opsPool_->freeOps[0], 101);
  EXPECT_EQ(opsPool_->ops[101].next, 100);
  EXPECT_EQ(opsPool_->ops[100].next, 7);
  EXPECT_EQ(opsPool_->freeOps[1], MAX_OPS_PER_PEER + 100);
  EXPECT_EQ(opsPool_->ops[MAX_OPS_PER_PEER + 100].next, -1);
  EXPECT_EQ(g_profCtrlStates, (std::vector<int>{ncclProfilerProxyCtrlAppend, ncclProfilerProxyCtrlAppendEnd}));
}

TEST_F(ProxyGetPostedOpsMicrotest, GetPostedOps_StopsAtTheBatchSizeAndResumesWithoutTheLock) {
  MakeOpsPool();
  ScopedHook param(g_loadParam, Param("PROXY_APPEND_BATCH_SIZE", 1));
  auto [first, last] = Chain({{0, 0, 1}, {0, 0, 1}, {0, 1, 2}, {0, 2, 3}, {1, 3, 3}, {1, 3, 4}});
  opsPool_->nextOps = first;
  opsPool_->nextOpsEnd = last;
  int added = 0;
  ASSERT_EQ(ncclProxyGetPostedOps(&ps_, &added), ncclSuccess);
  EXPECT_EQ(added, 3);  // two groups (opCount 1, opCount 2); opCount 3 would start a third
  EXPECT_EQ(st().nextOps, 103);
  opsPool_->nextOps = 5;  // a fresh post must wait: the leftover chain is drained first, without the lock
  std::unique_lock<std::mutex> held(opsPool_->mutex);
  added = 0;
  ASSERT_EQ(ncclProxyGetPostedOps(&ps_, &added), ncclSuccess);
  EXPECT_EQ(added, 2);  // a change of peer also starts a group
  EXPECT_EQ(st().nextOps, MAX_OPS_PER_PEER + 101);
  EXPECT_EQ(opsPool_->nextOps, 5);
}

TEST_F(ProxyGetPostedOpsMicrotest, GetPostedOps_WithActiveOpsNeverWaitsForNewPostsOrTheLock) {
  MakeOpsPool();
  ncclProxyOp op = Op(0, 1);
  ASSERT_EQ(ProxyAppend(&st(), &op), ncclSuccess);
  int added = 0;
  ASSERT_EQ(ncclProxyGetPostedOps(&ps_, &added), ncclSuccess);
  auto [first, last] = Chain({{0, 1, 1}});
  opsPool_->nextOps = first;
  opsPool_->nextOpsEnd = last;
  {
    std::unique_lock<std::mutex> held(opsPool_->mutex);
    std::thread t([&] { EXPECT_EQ(ncclProxyGetPostedOps(&ps_, &added), ncclSuccess); });
    t.join();
  }
  EXPECT_EQ(added, 0);
  EXPECT_EQ(opsPool_->nextOps, first);
  ASSERT_EQ(ncclProxyGetPostedOps(&ps_, &added), ncclSuccess);
  EXPECT_EQ(added, 1);
}

TEST_F(ProxyGetPostedOpsMicrotest, GetPostedOps_IdleProxySleepsUntilAPost) {
  MakeOpsPool();
  auto [first, last] = Chain({{1, 1, 4}});
  std::thread client([&, first = first, last = last] {
    EXPECT_TRUE(WaitFor([] { return g_profSleeps > 0; }));  // recorded before the proxy waits, under the lock
    ncclProxyPost(opsPool_, first, last);
  });
  int added = 0;
  ASSERT_EQ(ncclProxyGetPostedOps(&ps_, &added), ncclSuccess);
  client.join();
  EXPECT_EQ(added, 1);
  EXPECT_EQ(g_profCtrlStates, (std::vector<int>{ncclProfilerProxyCtrlSleep, ncclProfilerProxyCtrlWakeup,
                                                ncclProfilerProxyCtrlAppend, ncclProfilerProxyCtrlAppendEnd}));
}

TEST_F(ProxyGetPostedOpsMicrotest, GetPostedOps_NullConnectionOrAppendErrorsFail) {
  MakeOpsPool();
  auto [first, last] = Chain({{0, 0, 1}});
  opsPool_->ops[first].connection = nullptr;
  opsPool_->nextOps = first;
  opsPool_->nextOpsEnd = last;
  int added = 0;
  EXPECT_EQ(ncclProxyGetPostedOps(&ps_, &added), ncclInternalError);
  opsPool_->ops[first].connection = &conns_[0];
  st().pool = nullptr;
  g_callocFailAt = 0;
  EXPECT_EQ(ncclProxyGetPostedOps(&ps_, &added), ncclSystemError);
  EXPECT_EQ(added, 0);
}

class ProxyProgressThreadMicrotest : public ProxyProgressFixture {
 protected:
  void Start() { st().thread = std::thread(ncclProxyProgress, &ps_); }
  void Stop() { ASSERT_EQ(ncclProxyProgressDestroy(&ps_), ncclSuccess); }
};

TEST_F(ProxyProgressThreadMicrotest, ProgressThread_RunsPostedOpsToCompletionThenStops) {
  MakeOpsPool();
  ps_.cudaDev = 3;
  g_hipSetDeviceResult = hipErrorInvalidDevice;
  std::atomic<int> finished{0};
  ScopedHook progress(g_progressFn, [&](ncclProxyArgs* a) {
    a->idle = 0;
    if (a->retry_total >= 3) {
      a->state = ncclProxyOpNone;
      finished++;
    }
    return ncclSuccess;
  });
  Start();
  auto [first, last] = Chain({{0, 0, 1}, {0, 1, 1}});
  ASSERT_EQ(ncclProxyPost(opsPool_, first, last), ncclSuccess);
  ASSERT_TRUE(WaitFor([&] { return finished == 2; }));
  Stop();
  EXPECT_EQ(g_hipSetDevices, std::vector<int>{3});
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_ATTN, "[Proxy Progress] Failed to set CUDA device 3"));
  EXPECT_EQ(ps_.asyncResult, ncclSuccess);
  EXPECT_EQ(st().active, nullptr);
  EXPECT_EQ(opsPool_->freeOps[0], 101);
  auto& s = g_profCtrlStates;
  EXPECT_NE(std::find(s.begin(), s.end(), ncclProfilerProxyCtrlActive), s.end());
  EXPECT_NE(std::find(s.begin(), s.end(), ncclProfilerProxyCtrlIdle), s.end());
}

TEST_F(ProxyProgressThreadMicrotest, ProgressThread_StopWaitsForActiveOpsButAbortDoesNot) {
  MakeOpsPool();
  ScopedHook param(g_loadParam, Param("PROGRESS_APPENDOP_FREQ", 2));
  std::atomic<bool> release{false};
  ScopedHook progress(g_progressFn, [&](ncclProxyArgs* a) {
    if (release) a->state = ncclProxyOpNone;
    return ncclSuccess;
  });
  Start();
  auto [first, last] = Chain({{0, 0, 1}});
  ASSERT_EQ(ncclProxyPost(opsPool_, first, last), ncclSuccess);
  ASSERT_TRUE(WaitFor([] { return g_progressCalls >= 10; }));
  {
    std::lock_guard<std::mutex> lock(opsPool_->mutex);
    st().stop.store(1);
  }
  int callsAtStop = g_progressCalls;
  EXPECT_TRUE(WaitFor([&] { return g_progressCalls > callsAtStop + 10; }));  // still progressing after stop
  release = true;
  st().thread.join();
  EXPECT_EQ(st().active, nullptr);
  st().stop.store(0);
  release = false;
  Start();
  g_progressCalls = 0;
  auto [f2, l2] = Chain({{0, 1, 1}});
  ASSERT_EQ(ncclProxyPost(opsPool_, f2, l2), ncclSuccess);
  ASSERT_TRUE(WaitFor([] { return g_progressCalls > 0; }));
  COMPILER_ATOMIC_STORE(&abort_, 1, std::memory_order_release);
  st().thread.join();
  EXPECT_NE(st().active, nullptr);
}

TEST_F(ProxyProgressThreadMicrotest, ProgressThread_ProgressErrorsAreRecordedAndEndTheThread) {
  MakeOpsPool();
  ScopedHook progress(g_progressFn, [](ncclProxyArgs*) { return ncclRemoteError; });
  Start();
  auto [first, last] = Chain({{0, 0, 1}});
  ASSERT_EQ(ncclProxyPost(opsPool_, first, last), ncclSuccess);
  st().thread.join();
  EXPECT_EQ(ps_.asyncResult, ncclRemoteError);
}

TEST_F(ProxyProgressThreadMicrotest, ProgressThread_AppendErrorsAreRecordedButTheThreadKeepsRunning) {
  MakeOpsPool();
  Start();
  auto [first, last] = Chain({{0, 0, 1}});
  opsPool_->ops[first].connection = nullptr;
  ASSERT_EQ(ncclProxyPost(opsPool_, first, last), ncclSuccess);
  ASSERT_TRUE(WaitFor([&] { return COMPILER_ATOMIC_LOAD(&ps_.asyncResult, std::memory_order_acquire) != ncclSuccess; }));
  EXPECT_EQ(COMPILER_ATOMIC_LOAD(&ps_.asyncResult, std::memory_order_acquire), ncclInternalError);
  COMPILER_ATOMIC_STORE(&ps_.asyncResult, ncclSuccess, std::memory_order_release);
  EXPECT_TRUE(WaitFor([&] {  // the thread is still looping: it retries the bad op and reports again
    return COMPILER_ATOMIC_LOAD(&ps_.asyncResult, std::memory_order_acquire) == ncclInternalError;
  }));
  COMPILER_ATOMIC_STORE(&abort_, 1, std::memory_order_release);  // the bad op is retried forever; abort ends it
  Stop();
}

TEST_F(ProxyProgressThreadMicrotest, ProgressThread_DumpSignalInstallsTheStateDumper) {
  MakeOpsPool();
  ScopedHook param(g_loadParam, Param("PROXY_DUMP_SIGNAL", SIGUSR2));
  Start();
  Stop();
  EXPECT_EQ(signal(SIGUSR2, SIG_DFL), ncclDumpProxyState);
  EXPECT_EQ(ncclLastProxyState, &st());
}

TEST_F(ProxyProgressThreadMicrotest, ProgressCreate_StartsOneNamedThread) {
  MakeOpsPool();
  ASSERT_EQ(ncclProxyProgressCreate(&ps_), ncclSuccess);
  ASSERT_EQ(ncclProxyProgressCreate(&ps_), ncclSuccess);
  EXPECT_EQ(g_threadNames, 1);
  Stop();
  EXPECT_FALSE(st().thread.joinable());
}

// Shared-memory ops pool and connection init (service side)

class ProxyShmPoolMicrotest : public ProxyFixture {
 protected:
  void SetUp() override {
    ProxyFixture::SetUp();
    ps_.abortFlag = &abort_;
    ps_.tpLocalnRanks = 3;
    ps_.tpnRanks = 8;
  }
  void TearDown() override {
    if (st().opsPool) {
      EXPECT_EQ(ncclProxyProgressDestroy(&ps_), ncclSuccess);
      proxyOpsFree(&ps_);
    }
    for (int b = 0; b < pool_.banks; b++) free(pool_.pools[b]);
    free(pool_.pools);
    ProxyFixture::TearDown();
  }
  ncclProxyProgressState& st() { return ps_.progressState; }

  ncclProxyState ps_ = {};
  uint32_t abort_ = 0;
  ncclProxyConnectionPool pool_ = {nullptr, 0, NCCL_PROXY_CONN_POOL_SIZE};
};

TEST_F(ProxyShmPoolMicrotest, ProgressInit_MapsThePoolOnceThreadsFreeListsAndStartsTheThread) {
  ASSERT_EQ(proxyProgressInit(&ps_), ncclSuccess);
  ncclProxyOpsPool* pool = st().opsPool;
  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(g_shm.opens, 1);
  EXPECT_EQ(g_shm.lastRefcount, 3);
  EXPECT_STREQ(st().opsPoolShmSuffix, "AbC123");
  EXPECT_EQ(pool->nextOps, -1);
  for (int r = 0; r < 3; r++) {
    EXPECT_EQ(pool->freeOps[r], r * MAX_OPS_PER_PEER);
    EXPECT_EQ(pool->ops[r * MAX_OPS_PER_PEER].next, r * MAX_OPS_PER_PEER + 1);
    EXPECT_EQ(pool->ops[(r + 1) * MAX_OPS_PER_PEER - 1].next, -1);
  }
  EXPECT_EQ(pool->syncObjectsInitialized, 1);
  EXPECT_TRUE(st().thread.joinable());
  ASSERT_EQ(proxyProgressInit(&ps_), ncclSuccess);
  EXPECT_EQ(g_shm.opens, 1);
}

TEST_F(ProxyShmPoolMicrotest, ProgressInit_ShmOpenFailurePropagates) {
  g_shm.openResult = ncclSystemError;
  EXPECT_EQ(proxyProgressInit(&ps_), ncclSystemError);
  EXPECT_EQ(st().opsPool, nullptr);
  EXPECT_FALSE(st().thread.joinable());
}

TEST_F(ProxyShmPoolMicrotest, OpsFree_UnsharesClosesAndForgetsThePool) {
  ASSERT_EQ(proxyProgressInit(&ps_), ncclSuccess);
  ASSERT_EQ(ncclProxyProgressDestroy(&ps_), ncclSuccess);
  g_shm.closeResult = ncclSystemError;
  proxyOpsFree(&ps_);
  EXPECT_EQ(g_ncclOsMutexCondUnsharedCalls, 1);
  EXPECT_EQ(g_shm.closes, 1);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_ATTN, "shm close failed"));
  EXPECT_EQ(st().opsPool, nullptr);
  EXPECT_EQ(st().handle, nullptr);
  proxyOpsFree(&ps_);  // nothing mapped: only the (null) handle close
  EXPECT_EQ(g_ncclOsMutexCondUnsharedCalls, 1);
  EXPECT_EQ(g_shm.closes, 2);
}

TEST_F(ProxyShmPoolMicrotest, ShmUnlink_OnlyWhenMappedAndFailureIsNotFatal) {
  ncclComm& comm = *comm_;
  comm.proxyState = &ps_;
  EXPECT_EQ(ncclProxyShmUnlink(&comm), ncclSuccess);
  EXPECT_EQ(g_shm.unlinks, 0);
  ASSERT_EQ(proxyProgressInit(&ps_), ncclSuccess);
  EXPECT_EQ(ncclProxyShmUnlink(&comm), ncclSuccess);
  g_shm.unlinkResult = ncclSystemError;
  EXPECT_EQ(ncclProxyShmUnlink(&comm), ncclSuccess);
  EXPECT_EQ(g_shm.unlinks, 2);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_ATTN, "proxy ops shm unlink failed"));
}

TEST_F(ProxyShmPoolMicrotest, ConnInit_RejectsOutOfRangeLocalRanks) {
  ncclProxyLocalPeer peer = {};
  ncclProxyInitReq req = {0, 1, 3, 0, 0};
  ncclProxyInitResp resp = {};
  ncclProxyConnection* conn = nullptr;
  EXPECT_EQ(proxyConnInit(&peer, &pool_, &ps_, &req, &resp, &conn), ncclInvalidArgument);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "tpLocalRank 3 out of range [0,3)"));
  req.tpLocalRank = -1;
  EXPECT_EQ(proxyConnInit(&peer, &pool_, &ps_, &req, &resp, &conn), ncclInvalidArgument);
  EXPECT_EQ(pool_.banks, 0);
}

TEST_F(ProxyShmPoolMicrotest, ConnInit_RegistersTheConnectionAndMapsOpsForProgressTransports) {
  ncclProxyLocalPeer peer = {};
  peer.id = 42;
  ncclProxyInitReq req = {/*transport*/ 0, /*send*/ 1, /*tpLocalRank*/ 2, /*tpRank*/ 6, /*sameProcess*/ 1};
  ncclProxyInitResp resp = {};
  ncclProxyConnection* conn = nullptr;
  ASSERT_EQ(proxyConnInit(&peer, &pool_, &ps_, &req, &resp, &conn), ncclSuccess);
  ASSERT_EQ(conn, pool_.pools[0]);
  EXPECT_EQ(resp.connection, conn);
  EXPECT_EQ(resp.connId, 0);
  EXPECT_EQ(conn->sock, &peer.sock);
  EXPECT_EQ(conn->peerId, 42u);
  EXPECT_EQ(conn->send, 1);
  EXPECT_EQ(conn->tpLocalRank, 2);
  EXPECT_EQ(conn->sameProcess, 1);
  EXPECT_EQ(conn->tcomm, &g_transports[0].send);
  EXPECT_EQ(conn->state, connInitialized);
  EXPECT_EQ(peer.tpLocalRank, 2);
  EXPECT_EQ(peer.tpRank, 6);
  EXPECT_EQ(std::string(resp.devShmPath, 6), "AbC123");
  req = {/*transport*/ 1, /*send*/ 0, 1, 1, 0};  // no proxyProgress: no shm path
  ncclProxyInitResp resp2 = {};
  ASSERT_EQ(proxyConnInit(&peer, &pool_, &ps_, &req, &resp2, &conn), ncclSuccess);
  EXPECT_EQ(resp2.connId, 1);
  EXPECT_EQ(conn->tcomm, &g_transports[1].recv);
  EXPECT_EQ(resp2.devShmPath[0], '\0');
  EXPECT_EQ(g_shm.opens, 1);
}

TEST_F(ProxyShmPoolMicrotest, ConnInit_PoolAndShmFailuresPropagate) {
  ncclProxyLocalPeer peer = {};
  ncclProxyInitReq req = {0, 0, 0, 0, 0};
  ncclProxyInitResp resp = {};
  ncclProxyConnection* conn = nullptr;
  g_callocFailAt = 0;
  EXPECT_EQ(proxyConnInit(&peer, &pool_, &ps_, &req, &resp, &conn), ncclSystemError);
  g_callocFailAt = -1;
  g_shm.openResult = ncclSystemError;
  EXPECT_EQ(proxyConnInit(&peer, &pool_, &ps_, &req, &resp, &conn), ncclSystemError);
  EXPECT_NE(conn->state, connInitialized);
}

// Client RPCs: ncclProxyConnect, ncclProxyCallAsync / ncclPollProxyResponse / ncclProxyCallBlocking, UDS calls

struct WireRequest {
  int type, connId, reqSize, respSize;
  std::vector<char> req;
  void* opId;
};

class ProxyClientFixture : public ProxyFixture {
 protected:
  static constexpr int kRanks = 4;
  void SetUp() override {
    ProxyFixture::SetUp();
    res_.tpNRanks = kRanks;
    res_.magic = 0x5eed;
    comm_->sharedRes = &res_;
    comm_->proxyState = &ps_;
    comm_->abortFlag = &abort_;
    comm_->nRanks = kRanks;
    comm_->rank = 0;
    comm_->localRank = 0;
    comm_->topParentRanks = tpRanks_;
    comm_->topParentLocalRanks = tpLocal_;
    comm_->peerInfo = info_;
    comm_->gproxyConn = gconns_;
    for (int r = 0; r < kRanks; r++) {
      info_[r].hostHash = r < 2 ? 11 : 22;
      info_[r].pidHash = r == 1 ? 33 : 44;  // rank 1 shares our host but not our process
      uds_[r] = 0x100 + r;
    }
    ps_.peerAddresses = addrs_;
    ps_.peerAddressesUDS = uds_;
  }
  void TearDown() override {
    for (int i = 0; ps_.peerSocks && i < ps_.peerArraySize; i++) {
      if (ps_.peerSocks[i].socketDescriptor >= 0) close(ps_.peerSocks[i].socketDescriptor);
      if (ps_.proxyOps[i].pool) free(ps_.proxyOps[i].handle);
    }
    free(ps_.peerSocks);
    free(ps_.proxyOps);
    free(ps_.sharedDevMems);
    expectedProxyResponseFree(&ps_);
    ProxyFixture::TearDown();
  }

  // Gives the unit a connected peer socket for `tpRank` and returns the test's end of it.
  int Peer(int tpRank) {
    if (ps_.peerSocks == nullptr) {
      ps_.peerArraySize = kRanks;
      ps_.peerSocks = (ncclSocket*)calloc(kRanks, sizeof(ncclSocket));
      ps_.proxyOps = (ncclProxyOps*)calloc(kRanks, sizeof(ncclProxyOps));
      ps_.sharedDevMems = (void**)calloc(kRanks, sizeof(void*));
      for (int i = 0; i < kRanks; i++) ps_.peerSocks[i].socketDescriptor = -1;
    }
    auto [ours, theirs] = SockPair();
    if (ps_.peerSocks[tpRank].socketDescriptor >= 0) close(ps_.peerSocks[tpRank].socketDescriptor);
    ps_.peerSocks[tpRank].socketDescriptor = ours;
    conn_.tpLocalRank = conn_.tpRank = tpRank;
    return theirs;
  }
  static WireRequest Read(int fd) {
    WireRequest r;
    r.type = Get<int>(fd);
    r.connId = Get<int>(fd);
    r.reqSize = Get<int>(fd);
    r.respSize = Get<int>(fd);
    r.req.resize(r.reqSize);
    if (r.reqSize) EXPECT_EQ(read(fd, r.req.data(), r.reqSize), r.reqSize);
    r.opId = Get<void*>(fd);
    return r;
  }
  static void Reply(int fd, void* opId, ncclResult_t res, const void* resp, int size) {
    ncclProxyRpcResponseHeader h = {opId, res, size};
    Put(fd, h);
    if (size) ASSERT_EQ(write(fd, resp, size), size);
  }
  // Answers each request arriving on `fd` with `resp` once the unit starts polling.
  template <typename T>
  void Serve(int fd, const T& resp, ncclResult_t res = ncclSuccess, int idlePolls = 0) {
    g_sock.onProgress = [this, fd, resp, res, idlePolls]() mutable {
      struct pollfd p = {fd, POLLIN, 0};
      if (idlePolls-- > 0 || poll(&p, 1, 0) <= 0) return;
      WireRequest r = Read(fd);
      requests_.push_back(r);
      Reply(fd, r.opId, res, &resp, sizeof(resp));
    };
  }

  ncclSharedResources res_ = {};
  ncclProxyState ps_ = {};
  uint32_t abort_ = 0;
  int tpRanks_[kRanks] = {0, 1, 2, 3};
  int tpLocal_[kRanks] = {0, 1, 2, 3};
  ncclPeerInfo info_[kRanks] = {};
  ncclProxyConnector gconns_[kRanks] = {};
  union ncclSocketAddress addrs_[kRanks] = {};
  uint64_t uds_[kRanks];
  ncclProxyConnector conn_ = {};
  std::vector<WireRequest> requests_;
};

class ProxyCallMicrotest : public ProxyClientFixture {};

TEST_F(ProxyCallMicrotest, CallAsync_SendsTheFramedRequestAndExpectsAResponse) {
  EXPECT_EQ(ncclProxyCallAsync(comm_.get(), &conn_, ncclProxyMsgSetup, nullptr, 0, 0, this), ncclInternalError);
  int theirs = Peer(2);
  conn_.connId = 17;
  int req = 0xfeed;
  ASSERT_EQ(ncclProxyCallAsync(comm_.get(), &conn_, ncclProxyMsgSetup, &req, sizeof(req), 8, this), ncclSuccess);
  WireRequest r = Read(theirs);
  EXPECT_EQ(r.type, ncclProxyMsgSetup);
  EXPECT_EQ(r.connId, 17);
  EXPECT_EQ(r.reqSize, 4);
  EXPECT_EQ(r.respSize, 8);
  EXPECT_EQ(*(int*)r.req.data(), 0xfeed);
  EXPECT_EQ(r.opId, this);
  ASSERT_NE(ps_.expectedResponses, nullptr);
  EXPECT_EQ(ps_.expectedResponses->opId, this);
  EXPECT_EQ(ps_.expectedResponses->respSize, 8);
  ASSERT_EQ(ncclProxyCallAsync(comm_.get(), &conn_, ncclProxyMsgConnect, nullptr, 0, 0, &req), ncclSuccess);
  EXPECT_EQ(Read(theirs).reqSize, 0);
}

TEST_F(ProxyCallMicrotest, CallAsync_SendAndEnqueueFailuresPropagate) {
  Peer(1);
  int req = 1;
  for (int i = 0; i < 6; i++) {
    g_sock.sends = 0;
    g_sock.sendFailAt = i;
    EXPECT_EQ(ncclProxyCallAsync(comm_.get(), &conn_, ncclProxyMsgSetup, &req, sizeof(req), 0, this), ncclRemoteError) << i;
  }
  g_sock.sendFailAt = -1;
  g_callocFailAt = 0;
  EXPECT_EQ(ncclProxyCallAsync(comm_.get(), &conn_, ncclProxyMsgSetup, &req, sizeof(req), 0, this), ncclSystemError);
  EXPECT_EQ(ps_.expectedResponses, nullptr);
}

TEST_F(ProxyCallMicrotest, Poll_AbortAndMissingSocketsFail) {
  int out;
  EXPECT_EQ(ncclPollProxyResponse(comm_.get(), &conn_, &out, this), ncclInternalError);  // no peer sockets
  abort_ = 1;
  EXPECT_EQ(ncclPollProxyResponse(comm_.get(), &conn_, &out, this), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "is in abort state"));
}

TEST_F(ProxyCallMicrotest, Poll_ReturnsInProgressUntilOurResponseArrives) {
  int theirs = Peer(1);
  ASSERT_EQ(expectedProxyResponseEnqueue(&ps_, this, sizeof(int)), ncclSuccess);
  int out = 0;
  EXPECT_EQ(ncclPollProxyResponse(comm_.get(), &conn_, &out, this), ncclInProgress);
  int value = 99;
  Reply(theirs, this, ncclRemoteError, &value, sizeof(value));
  g_sock.progressChunk = 5;  // a partial header is completed before parsing
  EXPECT_EQ(ncclPollProxyResponse(comm_.get(), &conn_, &out, this), ncclRemoteError);
  EXPECT_EQ(out, 99);
  EXPECT_EQ(ps_.expectedResponses, nullptr);
}

TEST_F(ProxyCallMicrotest, Poll_BuffersOtherResponsesAndServesThemFromTheQueue) {
  int theirs = Peer(1);
  char other;
  ASSERT_EQ(expectedProxyResponseEnqueue(&ps_, &other, sizeof(int)), ncclSuccess);
  ASSERT_EQ(expectedProxyResponseEnqueue(&ps_, this, 0), ncclSuccess);
  int value = 7;
  Reply(theirs, &other, ncclSuccess, &value, sizeof(value));
  Reply(theirs, this, ncclSuccess, nullptr, 0);
  int out = 0;
  EXPECT_EQ(ncclPollProxyResponse(comm_.get(), &conn_, nullptr, this), ncclInProgress);  // stored other's response
  EXPECT_EQ(ncclPollProxyResponse(comm_.get(), &conn_, nullptr, this), ncclSuccess);
  close(ps_.peerSocks[1].socketDescriptor);  // the cached response needs no socket
  ps_.peerSocks[1].socketDescriptor = -1;
  EXPECT_EQ(ncclPollProxyResponse(comm_.get(), &conn_, &out, &other), ncclSuccess);
  EXPECT_EQ(out, 7);
  EXPECT_EQ(ps_.expectedResponses, nullptr);
}

TEST_F(ProxyCallMicrotest, Poll_SocketFailuresBeforeOrInsideTheHeader) {
  int theirs = Peer(1);
  int out;
  g_sock.progressFailAt = 0;
  EXPECT_EQ(ncclPollProxyResponse(comm_.get(), &conn_, &out, this), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Socket recv failed while polling"));
  Reply(theirs, this, ncclSuccess, nullptr, 0);
  g_sock.progressFailAt = 2;  // mid-header
  g_sock.progressChunk = 3;
  EXPECT_EQ(ncclPollProxyResponse(comm_.get(), &conn_, &out, this), ncclRemoteError);
}

TEST_F(ProxyCallMicrotest, Poll_OurResponseWithoutABufferFails) {
  int theirs = Peer(1);
  int value = 1;
  Reply(theirs, this, ncclSuccess, &value, sizeof(value));
  EXPECT_EQ(ncclPollProxyResponse(comm_.get(), &conn_, nullptr, this), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Proxy response buffer is NULL"));
}

TEST_F(ProxyCallMicrotest, Poll_OtherResponseAllocationOrRecvFailures) {
  int theirs = Peer(1);
  char other;
  int value = 1, out;
  Reply(theirs, &other, ncclSuccess, &value, sizeof(value));
  g_mallocFailAt = 0;
  EXPECT_EQ(ncclPollProxyResponse(comm_.get(), &conn_, &out, this), ncclSystemError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Could not allocate proxy response buffer"));
}

// The failed recv returns without freeing the response buffer it allocated. Pins a leak; a fix flips this pin.
TEST_F(ProxyCallMicrotest, Poll_KnownBug_RecvFailureLeaksTheResponseBuffer) {
  int theirs = Peer(1);
  char other;
  int value = 1, out;
  Reply(theirs, &other, ncclSuccess, &value, sizeof(value));
  g_sock.recvFailAt = 0;
  EXPECT_EQ(ncclPollProxyResponse(comm_.get(), &conn_, &out, this), ncclRemoteError);
  EXPECT_EQ(LeftoverCount(), 1u);
  FreeUnitLeftovers();
}

TEST_F(ProxyCallMicrotest, Poll_UnexpectedOpIdsFail) {
  int theirs = Peer(1);
  char other;
  int out;
  Reply(theirs, &other, ncclSuccess, nullptr, 0);
  EXPECT_EQ(ncclPollProxyResponse(comm_.get(), &conn_, &out, this), ncclInternalError);  // nobody expects `other`
  Reply(theirs, this, ncclSuccess, nullptr, 0);
  EXPECT_EQ(ncclPollProxyResponse(comm_.get(), &conn_, &out, this), ncclInternalError);  // nor `this`
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Couldn't find opId"));
}

TEST_F(ProxyCallMicrotest, CallBlocking_PollsUntilTheResponseForItsOwnOpId) {
  int theirs = Peer(3);
  conn_.connId = 5;
  int value = 1234;
  Serve(theirs, value, ncclSuccess, /*idlePolls*/ 2);
  int req = 3, out = 0;
  ASSERT_EQ(ncclProxyCallBlocking(comm_.get(), &conn_, ncclProxyMsgRegister, &req, sizeof(req), &out, sizeof(out)),
            ncclSuccess);
  EXPECT_EQ(out, 1234);
  ASSERT_EQ(requests_.size(), 1u);
  EXPECT_EQ(requests_[0].type, ncclProxyMsgRegister);
  EXPECT_EQ(requests_[0].connId, 5);
  EXPECT_EQ(g_sock.progresses, 3);
  Serve(theirs, value, ncclRemoteError);
  EXPECT_EQ(ncclProxyCallBlocking(comm_.get(), &conn_, ncclProxyMsgRegister, &req, sizeof(req), &out, sizeof(out)),
            ncclRemoteError);
}

TEST_F(ProxyCallMicrotest, CallBlocking_AllocationAndSendFailures) {
  Peer(3);
  int out;
  g_mallocFailAt = 0;
  EXPECT_EQ(ncclProxyCallBlocking(comm_.get(), &conn_, ncclProxyMsgSetup, nullptr, 0, &out, sizeof(out)), ncclSystemError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Failed to allocate proxy operation ID"));
  g_sock.sendFailAt = 0;
  EXPECT_EQ(ncclProxyCallBlocking(comm_.get(), &conn_, ncclProxyMsgSetup, nullptr, 0, &out, sizeof(out)), ncclRemoteError);
}

class ProxyConnectMicrotest : public ProxyClientFixture {
 protected:
  // Hands the unit a socket for its next connect and answers the Init RPC on it.
  int ServeInit(ncclProxyConnection* connection, int connId, const char* shm = "QwErTy") {
    auto [ours, theirs] = SockPair();
    g_sock.connectFds.push_back(ours);
    ncclProxyInitResp resp = {connection, connId, {}};
    std::memcpy(resp.devShmPath, shm, 6);
    Serve(theirs, resp);
    return theirs;
  }
  ncclProxyConnection remote_ = {};
};

TEST_F(ProxyConnectMicrotest, Connect_RejectsRanksOutsideTheCommAndTheTopParent) {
  ncclProxyConnector pc = {};
  EXPECT_EQ(ncclProxyConnect(comm_.get(), 0, 1, -1, &pc), ncclInternalError);
  EXPECT_EQ(ncclProxyConnect(comm_.get(), 0, 1, kRanks, &pc), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "proxyRank 4 out of bounds [0, 4)"));
  tpRanks_[2] = kRanks;
  EXPECT_EQ(ncclProxyConnect(comm_.get(), 0, 1, 2, &pc), ncclInternalError);
  tpRanks_[2] = -1;
  EXPECT_EQ(ncclProxyConnect(comm_.get(), 0, 1, 2, &pc), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "tpRank -1 out of bounds [0, 4) for rank 2"));
  tpRanks_[2] = 2;
  ps_.peerSocks = (ncclSocket*)calloc(1, sizeof(ncclSocket));  // sized for fewer ranks by an earlier comm
  ps_.peerArraySize = 1;
  ps_.proxyOps = (ncclProxyOps*)calloc(1, sizeof(ncclProxyOps));
  ps_.sharedDevMems = (void**)calloc(1, sizeof(void*));
  ps_.peerSocks[0].socketDescriptor = -1;
  EXPECT_EQ(ncclProxyConnect(comm_.get(), 0, 1, 2, &pc), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "peerIndex 2 out of bounds [0, 1)"));
}

TEST_F(ProxyConnectMicrotest, Connect_ConnectsInitsAndMapsThePeerOpsPoolOnce) {
  ServeInit(&remote_, 9);
  ncclProxyConnector pc = {};
  ASSERT_EQ(ncclProxyConnect(comm_.get(), /*transport*/ 0, /*send*/ 1, /*proxyRank*/ 2, &pc), ncclSuccess);
  EXPECT_TRUE(pc.initialized);
  EXPECT_EQ(pc.connection, &remote_);
  EXPECT_EQ(pc.connId, 9);
  EXPECT_EQ(pc.rank, 2);
  EXPECT_EQ(pc.tpRank, 2);
  EXPECT_EQ(pc.tpLocalRank, 2);
  EXPECT_EQ(pc.sameProcess, 0);
  EXPECT_EQ(ps_.peerArraySize, kRanks);
  EXPECT_EQ(ps_.peerSocks[0].socketDescriptor, -1);
  EXPECT_EQ(g_sock.inits, 1);
  EXPECT_EQ(g_sock.lastInitAddr, &addrs_[2]);
  EXPECT_EQ(g_sock.lastInitMagic, 0x5eedu);
  EXPECT_EQ(g_sock.lastInitType, ncclSocketTypeProxy);
  ASSERT_EQ(requests_.size(), 1u);
  EXPECT_EQ(requests_[0].type, ncclProxyMsgInit);
  ncclProxyInitReq req;
  std::memcpy(&req, requests_[0].req.data(), sizeof(req));
  EXPECT_EQ(req.transport, 0);
  EXPECT_EQ(req.send, 1);
  EXPECT_EQ(req.tpLocalRank, 0);
  EXPECT_EQ(req.tpRank, 0);
  EXPECT_EQ(req.sameProcess, 0);
  EXPECT_EQ(g_shm.lastPath, "/dev/shm/nccl-QwErTy");
  ncclProxyOps& ops = ps_.proxyOps[2];
  ASSERT_NE(ops.pool, nullptr);
  EXPECT_EQ(ops.nextOps, -1);
  EXPECT_EQ(ops.nextOpsEnd, -1);
  EXPECT_EQ(ops.freeOp, -1);
  // A second connection to the same peer reuses its socket and its mapped pool.
  ncclProxyConnector pc2 = {};
  ASSERT_EQ(ncclProxyConnect(comm_.get(), 0, 0, 2, &pc2), ncclSuccess);
  EXPECT_EQ(g_sock.inits, 1);
  EXPECT_EQ(g_shm.opens, 1);
}

TEST_F(ProxyConnectMicrotest, Connect_SameProcessPeersAndTransportsWithoutProgress) {
  int rank1 = ServeInit(&remote_, 1);
  info_[1] = info_[0];
  ncclProxyConnector pc = {};
  ASSERT_EQ(ncclProxyConnect(comm_.get(), /*transport without hooks*/ 1, 0, 1, &pc), ncclSuccess);
  EXPECT_EQ(pc.sameProcess, 1);
  EXPECT_EQ(g_shm.opens, 0);
  info_[1].pidHash = 1;
  ServeInit(&remote_, 1);
  ASSERT_EQ(ncclProxyConnect(comm_.get(), 1, 0, 3, &pc), ncclSuccess);
  EXPECT_EQ(pc.sameProcess, 0);
  ncclProxyConnector pc1 = {};
  Serve(rank1, ncclProxyInitResp{&remote_, 2, {}});
  ASSERT_EQ(ncclProxyConnect(comm_.get(), 1, 0, 1, &pc1), ncclSuccess);
  EXPECT_EQ(pc1.sameProcess, 0);
}

TEST_F(ProxyConnectMicrotest, Connect_SetupFailuresPropagate) {
  ncclProxyConnector pc = {};
  for (int i = 0; i < 3; i++) {
    g_callocCalls = 0;
    g_callocFailAt = i;
    EXPECT_EQ(ncclProxyConnect(comm_.get(), 0, 1, 2, &pc), ncclSystemError) << i;
    free(ps_.peerSocks);
    free(ps_.proxyOps);
    ps_.peerSocks = nullptr;
    ps_.proxyOps = nullptr;
  }
  g_callocFailAt = -1;
  g_sock.initResult = ncclSystemError;
  EXPECT_EQ(ncclProxyConnect(comm_.get(), 0, 1, 2, &pc), ncclSystemError);
  g_sock.initResult = ncclSuccess;
  g_sock.readyResult = ncclInternalError;
  EXPECT_EQ(ncclProxyConnect(comm_.get(), 0, 1, 2, &pc), ncclInternalError);
  g_sock.readyResult = ncclSuccess;
  g_sock.connectResult = ncclRemoteError;
  EXPECT_EQ(ncclProxyConnect(comm_.get(), 0, 1, 2, &pc), ncclRemoteError);
  g_sock.connectResult = ncclSuccess;
  ServeInit(&remote_, 1);
  g_sock.sendFailAt = g_sock.sends;
  EXPECT_EQ(ncclProxyConnect(comm_.get(), 0, 1, 2, &pc), ncclRemoteError);
  g_sock.sendFailAt = -1;
  int theirs = Peer(2);
  ncclProxyInitResp resp = {&remote_, 1, {'a', 'b', 'c', 'd', 'e', 'f'}};
  Serve(theirs, resp);
  g_shm.openResult = ncclSystemError;
  EXPECT_EQ(ncclProxyConnect(comm_.get(), 0, 1, 2, &pc), ncclSystemError);
  EXPECT_FALSE(pc.initialized);
}

class ProxyUdsClientMicrotest : public ProxyClientFixture {
 protected:
  ncclIpcHdr Sent(size_t i = 0) {
    ncclIpcHdr h = {};
    EXPECT_LT(i, g_ipc.sent.size());
    if (i < g_ipc.sent.size()) std::memcpy(&h, g_ipc.sent[i].data.data(), sizeof(h));
    return h;
  }
};

TEST_F(ProxyUdsClientMicrotest, CallUDS_SendsAHeaderAndReceivesTheResponseOnAPerCallSocket) {
  comm_->localRank = 1;
  conn_.tpRank = 3;
  int req = 0x77;
  int resp = 0;
  g_ipc.recvBytes = {5, 0, 0, 0};
  g_ipc.recvFd = 61;
  ncclIpcFd respFd = -1;
  ASSERT_EQ(ncclProxyCallBlockingUDS(comm_.get(), &conn_, ncclProxyMsgGetFd, &req, sizeof(req), &resp, sizeof(resp), nullptr,
                                     &respFd),
            ncclSuccess);
  EXPECT_EQ(resp, 5);
  EXPECT_EQ(respFd, 61);
  EXPECT_EQ(g_ipc.lastInitRank, 1);
  ncclIpcHdr h = Sent();
  EXPECT_EQ(h.type, ncclProxyMsgGetFd);
  EXPECT_EQ(h.rank, 1);
  EXPECT_EQ(h.reqSize, 4);
  EXPECT_EQ(h.respSize, 4);
  EXPECT_EQ((uint64_t)h.opId, g_ipc.lastInitHash);  // the reply socket is named after the op
  EXPECT_EQ((int)h.data[0], 0x77);
  EXPECT_EQ(g_ipc.sent[0].fd, 1001);  // no request fd: our own socket's fd as a placeholder
  EXPECT_EQ(g_ipc.sent[0].rank, 3);
  EXPECT_EQ(g_ipc.sent[0].hash, 0x103u);
  EXPECT_EQ(g_ipc.closes, 1);
  ncclIpcFd reqFd = 55;
  ASSERT_EQ(ncclProxyCallBlockingUDS(comm_.get(), &conn_, ncclProxyMsgQueryFd, nullptr, 0, &resp, sizeof(resp), &reqFd,
                                     nullptr),
            ncclSuccess);
  EXPECT_EQ(g_ipc.sent[1].fd, 55);
}

TEST_F(ProxyUdsClientMicrotest, CallUDS_FailuresCloseTheSocketAndReport) {
  int resp;
  g_ipc.initResult = ncclSystemError;
  EXPECT_EQ(ncclProxyCallBlockingUDS(comm_.get(), &conn_, 1, nullptr, 0, &resp, 4, nullptr, nullptr), ncclSystemError);
  g_ipc.initResult = ncclSuccess;
  g_ipc.getFdResult = ncclInternalError;
  EXPECT_EQ(ncclProxyCallBlockingUDS(comm_.get(), &conn_, 1, nullptr, 0, &resp, 4, nullptr, nullptr), ncclInternalError);
  g_ipc.getFdResult = ncclSuccess;
  std::vector<char> big(sizeof(ncclIpcHdr::data) + 1);
  EXPECT_EQ(ncclProxyCallBlockingUDS(comm_.get(), &conn_, 1, big.data(), big.size(), &resp, 4, nullptr, nullptr),
            ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, ("Invalid UDS proxy request size " + std::to_string(big.size())).c_str()));
  EXPECT_TRUE(g_ipc.sent.empty());
  for (ncclResult_t* r : {&g_ipc.sendResult, &g_ipc.recvResult}) {
    g_ipc.closes = 0;
    *r = ncclRemoteError;
    EXPECT_EQ(ncclProxyCallBlockingUDS(comm_.get(), &conn_, 1, nullptr, 0, &resp, 4, nullptr, nullptr), ncclRemoteError);
    *r = ncclSuccess;
    EXPECT_EQ(g_ipc.closes, 1);
  }
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "ncclProxyCallBlockingUDS call to tpRank 0(100) failed"));
}

// A failed close is retried on the already-closed socket; a fix flips this pin.
TEST_F(ProxyUdsClientMicrotest, CallUDS_KnownBug_ClosesTheSocketTwiceWhenCloseFails) {
  int resp;
  g_ipc.closes = 0;
  g_ipc.closeResult = ncclRemoteError;
  EXPECT_EQ(ncclProxyCallBlockingUDS(comm_.get(), &conn_, 1, nullptr, 0, &resp, 4, nullptr, nullptr), ncclRemoteError);
  EXPECT_EQ(g_ipc.closes, 2);
}

TEST_F(ProxyUdsClientMicrotest, GetFd_ConnectsOnFirstUseAndSendsTheHandle) {
  auto [ours, theirs] = SockPair();
  g_sock.connectFds.push_back(ours);
  ncclProxyConnection remote = {};
  ncclProxyInitResp resp = {&remote, 4, {}};
  Serve(theirs, resp);
  g_ipc.recvFd = 88;
  uint64_t handle = 0xabcdef;
  ncclIpcFd fd = -1;
  ASSERT_EQ(ncclProxyClientGetFdBlocking(comm_.get(), 2, &handle, &fd), ncclSuccess);
  EXPECT_EQ(fd, 88);
  EXPECT_TRUE(gconns_[2].initialized);
  EXPECT_EQ(requests_.size(), 1u);
  ncclIpcHdr h = Sent();
  EXPECT_EQ(h.type, ncclProxyMsgGetFd);
  EXPECT_EQ(h.reqSize, 8);
  EXPECT_EQ(h.respSize, 0);
  EXPECT_EQ(h.data[0], 0xabcdefu);
  ASSERT_EQ(ncclProxyClientGetFdBlocking(comm_.get(), 2, &handle, &fd), ncclSuccess);
  EXPECT_EQ(requests_.size(), 1u);
  g_ipc.sendResult = ncclRemoteError;
  EXPECT_EQ(ncclProxyClientGetFdBlocking(comm_.get(), 2, &handle, &fd), ncclRemoteError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "ncclProxyClientGetFd call to tpRank 2 handle 0xabcdef failed"));
  g_sock.connectResult = ncclRemoteError;
  EXPECT_EQ(ncclProxyClientGetFdBlocking(comm_.get(), 3, &handle, &fd), ncclRemoteError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "ncclProxyClientGetFd call to tpRank 3"));
}

TEST_F(ProxyUdsClientMicrotest, QueryFd_SendsOneRequestPerSegmentAndStopsAtTheFirstFailure) {
  conn_.tpRank = 1;
  g_ipc.recvBytes = {9, 0, 0, 0};
  ncclIpcFd local[3] = {21, 22, 23}, remote[3] = {-1, -1, -1};
  ASSERT_EQ(ncclProxyClientBatchQueryFdBlocking(comm_.get(), &conn_, local, remote, 3), ncclSuccess);
  ASSERT_EQ(g_ipc.sent.size(), 3u);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(Sent(i).type, ncclProxyMsgQueryFd);
    EXPECT_EQ(Sent(i).respSize, (int)sizeof(ncclIpcFd));
    EXPECT_EQ(g_ipc.sent[i].fd, local[i]);
    EXPECT_EQ(remote[i], 9);
  }
  g_ipc.sent.clear();
  g_ipc.sendResult = ncclRemoteError;
  EXPECT_EQ(ncclProxyClientBatchQueryFdBlocking(comm_.get(), &conn_, local, remote, 3), ncclRemoteError);
  EXPECT_EQ(g_ipc.sent.size(), 1u);
  ncclIpcFd one = -1;
  EXPECT_EQ(ncclProxyClientQueryFdBlocking(comm_.get(), &conn_, 31, &one), ncclRemoteError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "ncclProxyClientQueryFdBlocking call to tpRank 1 localFd 31 failed"));
  g_ipc.sendResult = ncclSuccess;
  ASSERT_EQ(ncclProxyClientQueryFdBlocking(comm_.get(), &conn_, 31, &one), ncclSuccess);
  EXPECT_EQ(one, 9);
  EXPECT_EQ(g_ipc.sent.back().fd, 31);
}

// Service side: proxyProgressAsync, proxyServiceInitOp, the ncclProxyService loop, UDS service

class ProxyServiceFixture : public ProxyFixture {
 protected:
  void SetUp() override {
    ProxyFixture::SetUp();
    std::call_once(proxyCpusetOnceFlag, [] {});  // ProxyCpusetMicrotest drives proxyCpusetOnceFunc directly
    CPU_ZERO(&proxyCpuset);
    int p[2];
    ASSERT_EQ(pipe(p), 0);
    listenW_ = p[1];
    fds_.push_back(p[1]);
    ps_.listenSock = (ncclSocket*)calloc(1, sizeof(ncclSocket));
    ps_.listenSock->socketDescriptor = p[0];
    g_watchedFree = ps_.listenSock;
    ps_.abortFlag = &abort_;
    ps_.tpnRanks = 4;
    ps_.tpLocalnRanks = 2;
    ps_.tpRank = 1;
  }
  void TearDown() override {
    FreeListenSock();
    if (st().opsPool) {
      EXPECT_EQ(ncclProxyProgressDestroy(&ps_), ncclSuccess);
      proxyOpsFree(&ps_);
    }
    for (int b = 0; b < pool_.banks; b++) free(pool_.pools[b]);
    free(pool_.pools);
    while (peer_.asyncOps) asyncProxyOpDequeue(&peer_, peer_.asyncOps);
    if (peer_.sock.socketDescriptor > 0) close(peer_.sock.socketDescriptor);
    ProxyFixture::TearDown();
  }
  ncclProxyProgressState& st() { return ps_.progressState; }
  // Releases the listen socket unless the unit did (only the service's normal exit frees it).
  void FreeListenSock() {
    if (ps_.listenSock == nullptr || g_watchedFreed) return;
    if (ps_.listenSock->socketDescriptor >= 0) close(ps_.listenSock->socketDescriptor);
    free(ps_.listenSock);
    ps_.listenSock = nullptr;
  }
  static void CloseUnlessClosed(int fd) {
    if (std::find(g_closedFds.begin(), g_closedFds.end(), fd) == g_closedFds.end()) close(fd);
  }

  // A client the service will accept on its next poll; returns the client's end.
  int Connect() {
    auto [ours, theirs] = SockPair();
    g_sock.acceptFds.push_back(ours);
    EXPECT_EQ(write(listenW_, "c", 1), 1);
    return theirs;
  }
  static void Request(int fd, int type, int connId, const void* req, int reqSize, int respSize, void* opId) {
    Put(fd, type);
    Put(fd, connId);
    Put(fd, reqSize);
    Put(fd, respSize);
    if (reqSize) ASSERT_EQ(write(fd, req, reqSize), reqSize);
    Put(fd, opId);
  }
  static void Init(int fd, ncclProxyInitReq req, void* opId) {
    Request(fd, ncclProxyMsgInit, -1, &req, sizeof(req), sizeof(ncclProxyInitResp), opId);
  }
  template <typename T>
  static T Response(int fd, void* opId, ncclResult_t res = ncclSuccess) {
    auto h = Get<ncclProxyRpcResponseHeader>(fd);
    EXPECT_EQ(h.opId, opId);
    EXPECT_EQ(h.res, res);
    EXPECT_EQ(h.respSize, (int)sizeof(T));
    return Get<T>(fd);
  }
  // EOF, or ECONNRESET when the service closed with our bytes still unread.
  static bool Closed(int fd) {
    char c;
    return WaitReadable(fd) && read(fd, &c, 1) <= 0;
  }

  ncclProxyState ps_ = {};
  uint32_t abort_ = 0;
  int listenW_ = -1;
  ncclProxyConnectionPool pool_ = {nullptr, 0, NCCL_PROXY_CONN_POOL_SIZE};
  ncclProxyLocalPeer peer_ = {};
  int asyncOps_ = 0;
  char op1_, op2_, op3_;
};

class ProxyServiceOpMicrotest : public ProxyServiceFixture {
 protected:
  void SetUp() override {
    ProxyServiceFixture::SetUp();
    auto [ours, theirs] = SockPair();
    peer_.sock.socketDescriptor = ours;
    peer_.id = 3;
    client_ = theirs;
  }
  // Runs one Init through proxyServiceInitOp and returns the new connection's id.
  int NewConn(int transport = 0, int send = 1) {
    Init(client_, {transport, send, 1, 2, 0}, &op1_);
    EXPECT_EQ(InitOp(ncclProxyMsgInit), ncclSuccess);
    return Response<ncclProxyInitResp>(client_, &op1_).connId;
  }
  // The service loop has already consumed the type word when it calls proxyServiceInitOp.
  ncclResult_t InitOp(int type) {
    EXPECT_EQ(Get<int>(peer_.sock.socketDescriptor), type);
    return proxyServiceInitOp(type, &peer_, &pool_, &ps_, &asyncOps_);
  }
  void Drain() {
    char buf[256];
    while (recv(peer_.sock.socketDescriptor, buf, sizeof(buf), MSG_DONTWAIT) > 0) {
    }
  }
  ncclProxyConnection* Conn(int id) {
    ncclProxyConnection* c = nullptr;
    EXPECT_EQ(ncclProxyGetConnection(&pool_, id, &c), ncclSuccess);
    return c;
  }
  int client_ = -1;
};

TEST_F(ProxyServiceOpMicrotest, Init_CreatesAConnectionOwnedByThePeerAndAnswersIt) {
  Init(client_, {0, 1, 1, 2, 1}, &op1_);
  ASSERT_EQ(InitOp(ncclProxyMsgInit), ncclSuccess);
  auto resp = Response<ncclProxyInitResp>(client_, &op1_);
  EXPECT_EQ(resp.connId, 0);
  EXPECT_EQ(resp.connection, pool_.pools[0]);
  EXPECT_EQ(std::string(resp.devShmPath, 6), "AbC123");
  EXPECT_EQ(pool_.pools[0]->peerId, 3u);
  EXPECT_EQ(pool_.pools[0]->state, connInitialized);
  EXPECT_EQ(asyncOps_, 0);
  EXPECT_EQ(peer_.asyncOps, nullptr);
}

TEST_F(ProxyServiceOpMicrotest, Init_RejectsMalformedRequests) {
  ncclProxyInitReq bad[] = {{-1, 0, 0, 0, 0}, {NTRANSPORTS, 0, 0, 0, 0}, {0, 0, -1, 0, 0},
                            {0, 0, 2, 0, 0},  {0, 0, 0, -1, 0},          {0, 0, 0, 4, 0}};
  for (const auto& req : bad) {
    Init(client_, req, &op1_);
    EXPECT_EQ(InitOp(ncclProxyMsgInit), ncclInvalidArgument);
    Drain();
  }
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "rejecting Init from localRank 0: transport=-1"));
  int req = 0;
  Request(client_, ncclProxyMsgInit, -1, &req, sizeof(req), sizeof(ncclProxyInitResp), &op1_);
  EXPECT_EQ(InitOp(ncclProxyMsgInit), ncclInvalidArgument);
  Drain();
  ncclProxyInitReq ok = {};
  Request(client_, ncclProxyMsgInit, -1, &ok, sizeof(ok), 1, &op1_);
  EXPECT_EQ(InitOp(ncclProxyMsgInit), ncclInvalidArgument);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "respSize=1 (expected"));
  EXPECT_EQ(pool_.banks, 0);
  EXPECT_EQ(asyncOps_, 0);
}

TEST_F(ProxyServiceOpMicrotest, Ops_RejectUnknownOrForeignConnections) {
  int id = NewConn();
  Request(client_, ncclProxyMsgSetup, id + 1, nullptr, 0, 0, &op2_);
  EXPECT_EQ(InitOp(ncclProxyMsgSetup), ncclInvalidArgument);
  Drain();
  peer_.id = 4;  // another peer presenting this peer's connection id
  Request(client_, ncclProxyMsgSetup, id, nullptr, 0, 0, &op2_);
  EXPECT_EQ(InitOp(ncclProxyMsgSetup), ncclInvalidArgument);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Invalid proxy connection peer: 3 != 4"));
  EXPECT_EQ(g_tp.setups, 0);
}

TEST_F(ProxyServiceOpMicrotest, Ops_DispatchToTheTransportAndAdvanceTheConnectionState) {
  int id = NewConn();
  int req = 0x1234;
  Request(client_, ncclProxyMsgSetup, id, &req, sizeof(req), sizeof(int), &op2_);
  ASSERT_EQ(InitOp(ncclProxyMsgSetup), ncclSuccess);
  EXPECT_EQ(Response<int>(client_, &op2_), 0x1234);
  EXPECT_EQ(Conn(id)->state, connSetupDone);
  int nChannels = 6;
  Request(client_, ncclProxyMsgSharedInit, id, &nChannels, sizeof(nChannels), 0, &op2_);
  ASSERT_EQ(InitOp(ncclProxyMsgSharedInit), ncclSuccess);
  EXPECT_EQ(Get<ncclProxyRpcResponseHeader>(client_).respSize, 0);
  EXPECT_EQ(g_tp.lastSharedInitChannels, 6);
  EXPECT_EQ(Conn(id)->state, connSharedInitialized);
  g_tp.connectResult = ncclRemoteError;  // reported to the client, not to the service
  Request(client_, ncclProxyMsgConnect, id, nullptr, 0, 0, &op2_);
  ASSERT_EQ(InitOp(ncclProxyMsgConnect), ncclSuccess);
  EXPECT_EQ(Get<ncclProxyRpcResponseHeader>(client_).res, ncclRemoteError);
  EXPECT_EQ(Conn(id)->state, connConnected);
  Request(client_, ncclProxyMsgRegister, id, &req, sizeof(req), 0, &op2_);
  ASSERT_EQ(InitOp(ncclProxyMsgRegister), ncclSuccess);
  Request(client_, ncclProxyMsgDeregister, id, &req, sizeof(req), 0, &op2_);
  ASSERT_EQ(InitOp(ncclProxyMsgDeregister), ncclSuccess);
  EXPECT_EQ(g_tp.registers, 1);
  EXPECT_EQ(g_tp.deregisters, 1);
  EXPECT_EQ(Conn(id)->state, connConnected);
  EXPECT_EQ(asyncOps_, 0);
}

// A transport without proxySharedInit replies an error but still marks the connection.
// Pins AICOMRCCL-2965; a fix flips this pin.
TEST_F(ProxyServiceOpMicrotest, SharedInit_KnownBug_MarksTheConnectionWithoutAHook) {
  int id = NewConn(/*transport*/ 1);
  int nChannels = 2;
  Request(client_, ncclProxyMsgSharedInit, id, &nChannels, sizeof(nChannels), 0, &op2_);
  ASSERT_EQ(InitOp(ncclProxyMsgSharedInit), ncclSuccess);
  EXPECT_EQ(Get<ncclProxyRpcResponseHeader>(client_).res, ncclInternalError);
  EXPECT_EQ(Conn(id)->state, connSharedInitialized);
}

TEST_F(ProxyServiceOpMicrotest, ProgressAsync_PendingOpsStayQueuedUntilDoneOrAbort) {
  int id = NewConn();
  g_tp.setupDoneAfter = 2;
  Request(client_, ncclProxyMsgSetup, id, nullptr, 0, 0, &op2_);
  ASSERT_EQ(InitOp(ncclProxyMsgSetup), ncclSuccess);
  EXPECT_EQ(asyncOps_, 1);
  ncclProxyAsyncOp* op = peer_.asyncOps;
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(proxyProgressAsync(op, &ps_, &asyncOps_, &peer_, &pool_), ncclInProgress);
  EXPECT_EQ(Conn(id)->state, connInitialized);
  EXPECT_EQ(proxyProgressAsync(op, &ps_, &asyncOps_, &peer_, &pool_), ncclSuccess);
  EXPECT_EQ(asyncOps_, 0);
  EXPECT_EQ(Get<ncclProxyRpcResponseHeader>(client_).opId, &op2_);
  g_tp.setups = 0;
  Request(client_, ncclProxyMsgSetup, id, nullptr, 0, 0, &op3_);
  ASSERT_EQ(InitOp(ncclProxyMsgSetup), ncclSuccess);
  abort_ = 1;
  EXPECT_EQ(proxyProgressAsync(peer_.asyncOps, &ps_, &asyncOps_, &peer_, &pool_), ncclInternalError);
  EXPECT_EQ(asyncOps_, 1);
}

TEST_F(ProxyServiceOpMicrotest, ProgressAsync_UnknownTypesAndResponseSendFailures) {
  int id = NewConn();
  ncclProxyAsyncOp op = {};
  op.type = ncclProxyMsgStart;
  EXPECT_EQ(proxyProgressAsync(&op, &ps_, &asyncOps_, &peer_, &pool_), ncclInternalError);
  for (int failAt : {0, 1}) {
    g_sock.sends = 0;
    g_sock.sendFailAt = failAt;
    int req = 1;
    Request(client_, ncclProxyMsgSetup, id, &req, sizeof(req), sizeof(int), &op2_);
    EXPECT_EQ(InitOp(ncclProxyMsgSetup), ncclRemoteError) << failAt;
  }
  EXPECT_EQ(asyncOps_, 2);  // the failed ops stay queued for the service loop to drain
}

TEST_F(ProxyServiceOpMicrotest, InitOp_RecvAndAllocationFailures) {
  int id = NewConn();
  int req = 1;
  for (int failAt = 0; failAt < 5; failAt++) {
    Request(client_, ncclProxyMsgSetup, id, &req, sizeof(req), sizeof(int), &op2_);
    g_sock.recvs = 0;
    g_sock.recvFailAt = failAt;
    EXPECT_EQ(InitOp(ncclProxyMsgSetup), ncclRemoteError) << failAt;
    Drain();
  }
  g_sock.recvFailAt = -1;
  for (int failAt = 0; failAt < 3; failAt++) {
    Request(client_, ncclProxyMsgSetup, id, &req, sizeof(req), sizeof(int), &op2_);
    g_callocCalls = 0;
    g_callocFailAt = failAt;
    EXPECT_EQ(InitOp(ncclProxyMsgSetup), ncclSystemError) << failAt;
    Drain();
  }
  EXPECT_EQ(asyncOps_, 0);
}

TEST_F(ProxyServiceOpMicrotest, MatchOpType_AcceptsOnlyRequestTypes) {
  std::vector<int> accepted;
  for (int t = 0; t <= ncclProxyMsgDeregister + 1; t++)
    if (proxyMatchOpType(t)) accepted.push_back(t);
  EXPECT_EQ(accepted, (std::vector<int>{ncclProxyMsgInit, ncclProxyMsgSharedInit, ncclProxyMsgSetup, ncclProxyMsgConnect,
                                        ncclProxyMsgGetFd, ncclProxyMsgRegister, ncclProxyMsgDeregister}));
}

class ProxyServiceLoopMicrotest : public ProxyServiceFixture {};

TEST_F(ProxyServiceLoopMicrotest, Service_ServesAPeerUntilStopThenTearsEverythingDown) {
  ps_.cudaDev = 2;
  g_hipSetDeviceResult = hipErrorInvalidDevice;
  int c = Connect();
  Init(c, {0, 1, 0, 0, 0}, &op1_);
  g_tp.setupDoneAfter = 1;  // completes on the next loop iteration
  Request(c, ncclProxyMsgSetup, 0, nullptr, 0, 0, &op2_);
  Put(c, (int)ncclProxyMsgStop);
  ncclProxyService(&ps_);
  EXPECT_EQ(Response<ncclProxyInitResp>(c, &op1_).connId, 0);
  EXPECT_EQ(Get<ncclProxyRpcResponseHeader>(c).opId, &op2_);
  EXPECT_TRUE(Closed(c));
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_ATTN, "[Proxy Service] Failed to set CUDA device 2"));
  EXPECT_EQ(g_hipSyncs.load(), 1);  // the teardown ran
  EXPECT_EQ(g_tp.frees.size(), 1u);
  EXPECT_EQ(st().opsPool, nullptr);  // progress thread joined and ops pool unmapped
  EXPECT_FALSE(st().thread.joinable());
  EXPECT_TRUE(g_ncclOsSetAffinityMasks.empty());
}

TEST_F(ProxyServiceLoopMicrotest, Service_ClosedOrFailingPeersAreDroppedWhileOthersAreServed) {
  int quiet = Connect();             // forced closed once the service stops
  int closer = Connect();            // sends Close
  int unknown = Connect();           // sends an unknown command
  int gone = Connect();              // hangs up
  int badConn = Connect();           // presents a connection id it never got
  int pending = Connect();           // queues a setup that never completes
  int stopper = Connect();
  g_sock.acceptFds.push_back(-1);    // an accepted-but-invalid fd is skipped
  ASSERT_EQ(write(listenW_, "c", 1), 1);
  int hup[2];
  ASSERT_EQ(pipe(hup), 0);
  close(hup[1]);
  g_sock.acceptFds.push_back(hup[0]);  // POLLHUP without POLLIN
  ASSERT_EQ(write(listenW_, "c", 1), 1);
  Put(closer, (int)ncclProxyMsgClose);
  Put(unknown, 0);  // ncclProxyMsgTypeStr's own "Unknown"; past its end the INFO line would read out of bounds
  close(gone);
  fds_.erase(std::find(fds_.begin(), fds_.end(), gone));
  Request(badConn, ncclProxyMsgSetup, 5, nullptr, 0, 0, &op1_);
  Init(pending, {0, 1, 0, 0, 0}, &op1_);
  g_tp.setupDoneAfter = 1000;
  Request(pending, ncclProxyMsgSetup, 0, nullptr, 0, 0, &op2_);
  ScopedHook onSetup(g_tp.onSetup, [&] {  // once every peer is in and the setup is being re-progressed
    if (g_tp.setups == 3 && g_sock.acceptFds.empty()) Put(stopper, (int)ncclProxyMsgStop);
  });
  ncclProxyService(&ps_);
  EXPECT_EQ(Response<ncclProxyInitResp>(pending, &op1_).connId, 0);
  for (int fd : {quiet, closer, unknown, badConn, pending, stopper}) EXPECT_TRUE(Closed(fd)) << fd;
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_ATTN, "Unknown command 0"));
  EXPECT_FALSE(ProxyLogged(NCCL_LOG_ATTN, "Unknown command 6"));  // Close is a command, not an unknown one
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_INFO, "Connection closed by localRank"));
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Failed to execute operation Setup"));
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_INFO, "Force closing peer=0"));
  FreeUnitLeftovers();  // the queued setup, see Service_KnownBug_StopLeaksAForceClosedPeersQueuedSetup
}

// Force-closing a peer at stop does not drain its queued setup. Pins a leak; a fix flips this pin.
TEST_F(ProxyServiceLoopMicrotest, Service_KnownBug_StopLeaksAForceClosedPeersQueuedSetup) {
  int pending = Connect();
  int stopper = Connect();
  Init(pending, {0, 1, 0, 0, 0}, &op1_);
  g_tp.setupDoneAfter = 1000;
  Request(pending, ncclProxyMsgSetup, 0, nullptr, 0, 0, &op2_);
  ScopedHook onSetup(g_tp.onSetup, [&] {
    if (g_tp.setups == 3) Put(stopper, (int)ncclProxyMsgStop);
  });
  ncclProxyService(&ps_);
  EXPECT_EQ(Response<ncclProxyInitResp>(pending, &op1_).connId, 0);
  EXPECT_EQ(LeftoverCount(), 1u);
  FreeUnitLeftovers();
}

TEST_F(ProxyServiceLoopMicrotest, Service_ClosingAPeerDrainsAllItsPendingOps) {
  int a = Connect();
  Init(a, {0, 1, 0, 0, 0}, &op1_);
  g_tp.setupDoneAfter = 1000;
  Request(a, ncclProxyMsgSetup, 0, nullptr, 0, 0, &op2_);
  Request(a, ncclProxyMsgSetup, 0, nullptr, 0, 0, &op3_);
  Put(a, (int)ncclProxyMsgClose);
  int setupsAtClose = -1;
  ScopedHook onClose(g_sock.onClose, [&](int) {  // the next client reuses the closed peer's slot
    if (setupsAtClose != -1) return;
    setupsAtClose = g_tp.setups;
    Put(Connect(), (int)ncclProxyMsgStop);
  });
  ncclProxyService(&ps_);
  EXPECT_GT(setupsAtClose, 2);
  EXPECT_EQ(g_tp.setups, setupsAtClose);  // no op of the closed peer is progressed for its successor
}

TEST_F(ProxyServiceLoopMicrotest, Service_AbortDropsPeersWithPendingOps) {
  int pending = Connect();
  int other = Connect();
  Init(pending, {0, 1, 0, 0, 0}, &op1_);
  g_tp.setupDoneAfter = 1000;
  Request(pending, ncclProxyMsgSetup, 0, nullptr, 0, 0, &op2_);
  ScopedHook onSetup(g_tp.onSetup, [&] {
    if (g_tp.setups != 3) return;
    abort_ = 1;
    EXPECT_EQ(write(listenW_, "c", 1), 1);  // wake the final poll at once
  });
  ncclProxyService(&ps_);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Error encountered progressing operation=Setup"));
  EXPECT_EQ(Response<ncclProxyInitResp>(pending, &op1_).connId, 0);
  EXPECT_TRUE(Closed(pending));
  EXPECT_TRUE(Closed(other));
}

TEST_F(ProxyServiceLoopMicrotest, Service_RecvErrorsCloseThePeerAndAreQuietOnceAborted) {
  Put(Connect(), 1);
  Put(Connect(), 1);
  ScopedHook<ncclResult_t()> tryRecv(g_sock.onTryRecv, [&] {
    if (tryRecv.calls == 2) {
      abort_ = 1;
      EXPECT_EQ(write(listenW_, "c", 1), 1);  // wake the final poll at once
    }
    return ncclRemoteError;
  });
  ncclProxyService(&ps_);
  EXPECT_EQ(tryRecv.calls, 2);
  EXPECT_EQ(ProxyLogCount("Could not receive type"), 1);
}

TEST_F(ProxyServiceLoopMicrotest, Service_AbortWithCuMemHostClosesPeersWhenStopped) {
  int a = Connect();
  g_cuMemEnabled = g_cuMemHostEnabled = 1;
  ps_.stop = 1;
  abort_ = 1;
  ncclProxyService(&ps_);
  EXPECT_TRUE(Closed(a));
  EXPECT_FALSE(ProxyLogged(NCCL_LOG_INFO, "Force closing"));  // closed by the abort check, not the stop sweep
  for (int variant = 0; variant < 2; variant++) {  // direct mode, or the proxy not stopped yet: swept instead
    ps_.directMode = variant == 0;
    ps_.stop = variant == 0;
    ps_.listenSock = (ncclSocket*)calloc(1, sizeof(ncclSocket));
    g_watchedFree = ps_.listenSock;
    int p[2];
    ASSERT_EQ(pipe(p), 0);
    fds_.push_back(p[1]);
    ps_.listenSock->socketDescriptor = p[0];
    listenW_ = p[1];
    ResetProxyLog();
    Connect();
    ncclProxyService(&ps_);
    EXPECT_TRUE(ProxyLogged(NCCL_LOG_INFO, "Force closing")) << variant;
  }
}

TEST_F(ProxyServiceLoopMicrotest, Service_AcceptFailuresAreLoggedAndAffinityIsApplied) {
  CPU_SET(0, &proxyCpuset);
  g_ncclOsCpuCountValue = 1;
  ASSERT_EQ(write(listenW_, "c", 1), 1);  // nothing queued to accept
  abort_ = 1;
  ncclProxyService(&ps_);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_ATTN, "Accept failed"));
  ASSERT_EQ(g_ncclOsSetAffinityMasks.size(), 1u);
  EXPECT_TRUE(CPU_EQUAL(&g_ncclOsSetAffinityMasks[0], &proxyCpuset));
}

// "Too many connections" jumps past the teardown, so the listen socket is never freed.
// Pins AICOMRCCL-2963; a fix flips this pin.
TEST_F(ProxyServiceLoopMicrotest, Service_KnownBug_TooManyConnectionsSkipsTeardown) {
  ps_.tpnRanks = 1;  // NCCL_MAX_PROXY_CONNECTIONS slots
  for (int i = 0; i < NCCL_MAX_PROXY_CONNECTIONS + 1; i++) Connect();
  std::vector<int> ours(g_sock.acceptFds.begin(), g_sock.acceptFds.end());
  ncclProxyService(&ps_);
  ASSERT_EQ(g_sock.acceptFds.size(), 1u);  // every slot was filled; the extra client was never accepted
  EXPECT_EQ(g_sock.acceptFds.front(), ours.back());
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, ("Too many connections (" + std::to_string(NCCL_MAX_PROXY_CONNECTIONS) +
                                     " max)").c_str()));
  EXPECT_FALSE(g_watchedFreed);
  EXPECT_EQ(g_hipSyncs.load(), 0);
  for (int fd : ours) CloseUnlessClosed(fd);
}

TEST_F(ProxyServiceLoopMicrotest, Service_StartupFailuresReturnWithoutServing) {
  for (int i = 0; i < 4; i++) {
    g_callocCalls = 0;
    g_callocFailAt = i;
    ncclProxyService(&ps_);
  }
  g_callocFailAt = -1;
  g_sock.getFdFailAt = 0;
  ncclProxyService(&ps_);
  EXPECT_EQ(g_sock.getFds, 1);
  EXPECT_EQ(g_hipSyncs.load(), 0);
}

// A poll failure jumps past the teardown, so the listen socket is never freed.
// Pins AICOMRCCL-2963; a fix flips this pin.
TEST_F(ProxyServiceLoopMicrotest, Service_KnownBug_PollFailureSkipsTeardown) {
  Connect();
  int ours = g_sock.acceptFds.front();
  struct rlimit old;
  ASSERT_EQ(getrlimit(RLIMIT_NOFILE, &old), 0);
  struct rlimit none = {0, old.rlim_max};
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &none), 0);  // poll() rejects nfds above the fd limit
  ncclProxyService(&ps_);
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &old), 0);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "[Proxy Service] Poll failed"));
  EXPECT_EQ(g_sock.acceptFds.size(), 1u);  // the first poll failed, so the pending client was never accepted
  EXPECT_FALSE(g_watchedFreed);
  EXPECT_EQ(g_hipSyncs.load(), 0);
  CloseUnlessClosed(ours);
}

class ProxyCpusetMicrotest : public ProxyFixture {};

TEST_F(ProxyCpusetMicrotest, CpusetOnce_ParsesNcclProxyCpusetOrClearsIt) {
  ScopedHook current(g_ncclOsGetAffinity, [](ncclAffinity* a) {
    CPU_ZERO(a);
    CPU_SET(0, a);
    return ncclSuccess;
  });
  CPU_SET(5, &proxyCpuset);
  proxyCpusetOnceFunc();
  EXPECT_EQ(CPU_COUNT(&proxyCpuset), 0);
  SetMicroEnv("NCCL_PROXY_CPUSET", "1,3");
  proxyCpusetOnceFunc();
  EXPECT_EQ(CPU_COUNT(&proxyCpuset), 2);
  EXPECT_TRUE(CPU_ISSET(1, &proxyCpuset));
  EXPECT_TRUE(CPU_ISSET(3, &proxyCpuset));
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_INFO, "NCCL_PROXY_CPUSET = 1,3: 0 changed to 1,3"));
  CPU_ZERO(&proxyCpuset);
}

class ProxyUdsServiceMicrotest : public ProxyServiceFixture {
 protected:
  void SetUp() override {
    ProxyServiceFixture::SetUp();
    int p[2];
    ASSERT_EQ(pipe(p), 0);
    fds_.push_back(p[0]);
    fds_.push_back(p[1]);
    udsW_ = p[1];
    ps_.ipcSock.fd = p[0];
  }
  void Hdr(int type, uint64_t data, void* opId, int rank = 2) {
    ncclIpcHdr h = {};
    h.type = type;
    h.rank = rank;
    h.opId = opId;
    h.data[0] = data;
    g_ipc.recvBytes.assign((char*)&h, (char*)&h + sizeof(h));
  }
  int udsW_ = -1;
};

// proxy.cc rejects GetFd/QueryFd requests before ROCm 7.
class ProxyUdsFdServiceMicrotest : public ProxyUdsServiceMicrotest {
 protected:
  void SetUp() override {
    ProxyUdsServiceMicrotest::SetUp();  // first: TearDown runs even after a skip
    if (ROCM_VERSION < 70000) GTEST_SKIP() << "proxyGetFd/proxyQueryFd need ROCm 7";
  }
};

TEST_F(ProxyUdsFdServiceMicrotest, GetFd_ExportsTheHandleAndSendsTheFdBack) {
  Hdr(ncclProxyMsgGetFd, 0xbeef, (void*)0x40);
  g_ipc.recvFd = 12;
  ASSERT_EQ(proxyUDSRecvReq(&ps_, 0), ncclSuccess);
  EXPECT_EQ(g_ipc.closedFds, (std::vector<ncclIpcFd>{12, 77}));  // the dummy request fd, then the exported fd
  EXPECT_EQ(g_ipc.lastInitRank, 1);
  EXPECT_EQ(g_ipc.lastInitHash, 0x41u);
  ASSERT_EQ(g_ipc.sent.size(), 1u);
  EXPECT_EQ(g_ipc.sent[0].fd, 77);
  EXPECT_EQ(g_ipc.sent[0].rank, 2);
  EXPECT_EQ(g_ipc.sent[0].hash, 0x40u);
  EXPECT_EQ(g_ipc.closes, 1);
}

TEST_F(ProxyUdsFdServiceMicrotest, GetFd_ExportInitSendCloseAndFdCloseFailures) {
  Hdr(ncclProxyMsgGetFd, 1, (void*)0x40);
  g_hipExportResult = hipErrorInvalidValue;
  EXPECT_EQ(proxyUDSRecvReq(&ps_, 0), ncclUnhandledCudaError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "HIP failure '[hip_fake] stub error'"));
  g_hipExportResult = hipSuccess;
  g_ipc.initResult = ncclSystemError;
  EXPECT_EQ(proxyUDSRecvReq(&ps_, 0), ncclSystemError);
  g_ipc.initResult = ncclSuccess;
  g_ipc.sendFdResult = ncclRemoteError;
  EXPECT_EQ(proxyUDSRecvReq(&ps_, 0), ncclRemoteError);
  g_ipc.sendFdResult = ncclSuccess;
  g_ipc.closeResult = ncclInternalError;
  EXPECT_EQ(proxyUDSRecvReq(&ps_, 0), ncclInternalError);
  g_ipc.closeResult = ncclSuccess;
  g_ipc.fdCloseResult = -1;
  EXPECT_EQ(proxyUDSRecvReq(&ps_, 0), ncclSystemError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Call to close failed"));
}

TEST_F(ProxyUdsFdServiceMicrotest, QueryFd_EchoesTheRemoteFd) {
  Hdr(ncclProxyMsgQueryFd, 0, (void*)0x80, 3);
  g_ipc.recvFd = 44;
  ASSERT_EQ(proxyUDSRecvReq(&ps_, 0), ncclSuccess);
  ASSERT_EQ(g_ipc.sent.size(), 1u);
  EXPECT_EQ(*(ncclIpcFd*)g_ipc.sent[0].data.data(), 44);
  EXPECT_EQ(g_ipc.sent[0].fd, NCCL_INVALID_IPC_FD);
  EXPECT_EQ(g_ipc.sent[0].rank, 3);
  EXPECT_EQ(g_ipc.lastInitHash, 0x81u);
  g_ipc.closeResult = ncclInternalError;
  EXPECT_EQ(proxyUDSRecvReq(&ps_, 0), ncclInternalError);
}

// proxyQueryFd returns success when the reply socket's init or send fails; a fix flips this pin.
TEST_F(ProxyUdsFdServiceMicrotest, QueryFd_KnownBug_SwallowsInitAndSendErrors) {
  Hdr(ncclProxyMsgQueryFd, 0, (void*)0x80, 3);
  g_ipc.recvFd = 44;
  g_ipc.sendResult = ncclRemoteError;
  EXPECT_EQ(proxyUDSRecvReq(&ps_, 0), ncclSuccess);
  g_ipc.sendResult = ncclSuccess;
  g_ipc.initResult = ncclSystemError;
  EXPECT_EQ(proxyUDSRecvReq(&ps_, 0), ncclSuccess);
}

TEST_F(ProxyUdsFdServiceMicrotest, ServiceUDS_HandlesRequestsUntilStopOrAbort) {
  ps_.cudaDev = 4;
  g_hipSetDeviceResult = hipErrorInvalidDevice;
  CPU_SET(0, &proxyCpuset);
  g_ncclOsCpuCountValue = 1;
  Hdr(ncclProxyMsgQueryFd, 0, (void*)0x80);
  ASSERT_EQ(write(udsW_, "r", 1), 1);  // stays readable: the fake recv does not drain it
  ScopedHook onRecv(g_ipc.onRecv, [&] { ps_.stop = 1; });
  ncclProxyServiceUDS(&ps_);
  EXPECT_EQ(g_ipc.sent.size(), 1u);
  EXPECT_EQ(g_ipc.closes, 2);  // the reply socket, then the service socket
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_ATTN, "[Proxy Service UDS] Failed to set CUDA device 4"));
  ASSERT_EQ(g_ncclOsSetAffinityMasks.size(), 1u);
  EXPECT_TRUE(CPU_EQUAL(&g_ncclOsSetAffinityMasks[0], &proxyCpuset));
  ps_.stop = 0;
  abort_ = 1;
  ncclProxyServiceUDS(&ps_);
  EXPECT_EQ(g_ipc.sent.size(), 1u);
}

TEST_F(ProxyUdsServiceMicrotest, RecvReq_UnknownTypeOrRecvFailure) {
  Hdr(ncclProxyMsgSetup, 0, nullptr);
  EXPECT_EQ(proxyUDSRecvReq(&ps_, 0), ncclInternalError);
  g_ipc.recvResult = ncclRemoteError;
  EXPECT_EQ(proxyUDSRecvReq(&ps_, 0), ncclRemoteError);
}

TEST_F(ProxyUdsServiceMicrotest, ServiceUDS_GetFdOrPollFailuresEndIt) {
  g_ipc.getFdResult = ncclInternalError;
  ncclProxyServiceUDS(&ps_);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Get listenSock fd fails"));
  EXPECT_TRUE(g_ncclOsSetAffinityMasks.empty());  // no proxy cpuset: the thread keeps its affinity
  g_ipc.getFdResult = ncclSuccess;
  struct rlimit old;
  ASSERT_EQ(getrlimit(RLIMIT_NOFILE, &old), 0);
  struct rlimit none = {0, old.rlim_max};
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &none), 0);
  ncclProxyServiceUDS(&ps_);
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &old), 0);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "[Proxy Service UDS] Poll failed"));
  EXPECT_EQ(g_ipc.closes, 0);
}

// Proxy lifecycle: ncclProxyInit / ncclProxyCreate / ncclProxyStop

class ProxyLifecycleMicrotest : public ProxyFixture {
 protected:
  void SetUp() override {
    ProxyFixture::SetUp();
    std::call_once(proxyCpusetOnceFlag, [] {});
    comm_->sharedRes = &res_;
    comm_->abortFlag = &abort_;
    comm_->rank = 1;
    comm_->nRanks = 2;
    comm_->topParentRanks = tpRanks_;
    res_.magic = 0xabc;
  }
  void TearDown() override {
    if (res_.proxyState) {
      res_.proxyState->refCount = 0;
      res_.proxyState->peerAddresses = nullptr;
      res_.proxyState->peerAddressesUDS = nullptr;
      ncclProxyDestroy(comm_.get());
    }
    ProxyFixture::TearDown();
  }
  ncclSharedResources res_ = {};
  uint32_t abort_ = 0;
  int tpRanks_[2] = {0, 1};
  ncclSocket listen_ = {};
  union ncclSocketAddress addrs_[2] = {};
  uint64_t uds_[2] = {0x10, 0x11};
  char devMem_;
};

TEST_F(ProxyLifecycleMicrotest, Init_CreatesTheSharedStateOnceAndOpensTheUdsSocket) {
  ASSERT_EQ(ncclProxyInit(comm_.get(), &listen_, addrs_, uds_), ncclSuccess);
  ncclProxyState* ps = res_.proxyState;
  ASSERT_NE(ps, nullptr);
  EXPECT_EQ(comm_->proxyState, ps);
  EXPECT_EQ(ps->refCount, 1);
  EXPECT_EQ(ps->listenSock, &listen_);
  EXPECT_EQ(ps->peerAddresses, addrs_);
  EXPECT_EQ(ps->peerAddressesUDS, uds_);
  EXPECT_EQ(g_ipc.lastInitRank, 1);
  EXPECT_EQ(g_ipc.lastInitHash, 0x11u);
  EXPECT_EQ(ncclProxyInit(comm_.get(), &listen_, addrs_, uds_), ncclInternalError);
  EXPECT_TRUE(ProxyLogged(NCCL_LOG_WARN, "Proxy state is already initialized"));
  delete res_.proxyState;
  res_.proxyState = nullptr;
  g_ipc.initResult = ncclSystemError;
  EXPECT_EQ(ncclProxyInit(comm_.get(), &listen_, addrs_, uds_), ncclSystemError);
}

TEST_F(ProxyLifecycleMicrotest, Create_CopiesCommSettingsAndStartsBothServiceThreadsOnce) {
  auto* listen = (ncclSocket*)calloc(1, sizeof(ncclSocket));
  int lp[2], up[2];
  ASSERT_EQ(pipe(lp), 0);
  ASSERT_EQ(pipe(up), 0);
  fds_.insert(fds_.end(), {lp[1], up[0], up[1]});
  listen->socketDescriptor = lp[0];
  ASSERT_EQ(ncclProxyInit(comm_.get(), listen, addrs_, uds_), ncclSuccess);
  ncclProxyState* ps = comm_->proxyState;
  ps->ipcSock.fd = up[0];
  ASSERT_EQ(write(lp[1], "c", 1), 1);  // both threads wake at once and see the abort
  ASSERT_EQ(write(up[1], "c", 1), 1);
  comm_->cudaDev = 5;
  comm_->localRanks = 2;
  comm_->p2pChunkSize = 1 << 19;
  comm_->buffSizes[1] = 777;
  comm_->ll128LineElems = 9;
  abort_ = 1;
  ps->refCount = 2;  // a split comm sharing an already running proxy
  ASSERT_EQ(ncclProxyCreate(comm_.get()), ncclSuccess);
  EXPECT_FALSE(ps->thread.joinable());
  ps->refCount = 1;
  ASSERT_EQ(ncclProxyCreate(comm_.get()), ncclSuccess);
  ps->thread.join();
  ps->threadUDS.join();
  EXPECT_EQ(ps->comm, comm_.get());
  EXPECT_EQ(ps->tpRank, 1);
  EXPECT_EQ(ps->tpnRanks, 2);
  EXPECT_EQ(ps->tpLocalnRanks, 2);
  EXPECT_EQ(ps->cudaDev, 5);
  EXPECT_EQ(ps->abortFlag, &abort_);
  EXPECT_EQ(ps->p2pChunkSize, 1 << 19);
  EXPECT_EQ(ps->buffSizes[1], 777);
  EXPECT_EQ(ps->ll128LineElems, 9);
  EXPECT_EQ(ps->ginState, &res_.ginState);
  EXPECT_EQ(g_threadNames, 2);
  EXPECT_EQ(g_hipSetDevices, (std::vector<int>{5, 5}));
}

TEST_F(ProxyLifecycleMicrotest, Stop_LastReferenceStopsOwnProxyAndClosesPeers) {
  comm_->proxyState = nullptr;
  EXPECT_EQ(ncclProxyStop(comm_.get()), ncclSuccess);
  ASSERT_EQ(ncclProxyInit(comm_.get(), &listen_, addrs_, uds_), ncclSuccess);
  ncclProxyState* ps = comm_->proxyState;
  ps->refCount = 2;
  ASSERT_EQ(ncclProxyStop(comm_.get()), ncclSuccess);
  EXPECT_EQ(comm_->proxyRefCountOld, 1);
  EXPECT_EQ(g_sock.inits, 0);
  EXPECT_EQ(ps->stop, 0);
  auto [self, selfTheirs] = SockPair();
  auto [peer, peerTheirs] = SockPair();
  g_sock.connectFds.push_back(self);
  ps->peerArraySize = 3;
  ps->peerSocks = (ncclSocket*)calloc(3, sizeof(ncclSocket));
  ps->proxyOps = (ncclProxyOps*)calloc(3, sizeof(ncclProxyOps));
  ps->sharedDevMems = (void**)calloc(3, sizeof(void*));
  auto [peer2, peer2Theirs] = SockPair();
  ps->peerSocks[0].socketDescriptor = -1;
  ps->peerSocks[1].socketDescriptor = peer;
  ps->peerSocks[2].socketDescriptor = peer2;  // nothing mapped or imported
  ps->proxyOps[1].pool = (ncclProxyOpsPool*)this;
  ps->proxyOps[1].handle = malloc(1);
  ps->sharedDevMems[1] = &devMem_;
  ps->sharedDevMems[0] = &devMem_;  // no socket: left alone
  ASSERT_EQ(ncclProxyStop(comm_.get()), ncclSuccess);
  EXPECT_EQ(comm_->proxyRefCountOld, 0);
  EXPECT_EQ(g_sock.lastInitAddr, &addrs_[1]);
  EXPECT_EQ(g_sock.lastInitMagic, 0xabcu);
  EXPECT_EQ(Get<int>(selfTheirs), ncclProxyMsgStop);
  EXPECT_EQ(Get<int>(peerTheirs), ncclProxyMsgClose);
  EXPECT_EQ(Get<int>(peer2Theirs), ncclProxyMsgClose);
  EXPECT_EQ(g_shm.closes, 1);
  EXPECT_EQ(g_hipIpcClosed, std::vector<void*>{&devMem_});
  EXPECT_EQ(ps->peerSocks[1].socketDescriptor, -1);
  EXPECT_EQ(ps->stop, 1);
  free(ps->peerSocks);
  free(ps->proxyOps);
  free(ps->sharedDevMems);
  ps->peerSocks = nullptr;
  ps->proxyOps = nullptr;
  ps->sharedDevMems = nullptr;
}

TEST_F(ProxyLifecycleMicrotest, Stop_SkipsTheStopMessageWhenAbortedOrUnreachable) {
  ASSERT_EQ(ncclProxyInit(comm_.get(), &listen_, addrs_, uds_), ncclSuccess);
  ncclProxyState* ps = comm_->proxyState;
  abort_ = 1;
  ASSERT_EQ(ncclProxyStop(comm_.get()), ncclSuccess);
  EXPECT_EQ(g_sock.inits, 0);
  EXPECT_EQ(ps->stop, 1);
  abort_ = 0;
  ps->refCount = 1;
  g_sock.connectResult = ncclRemoteError;
  ASSERT_EQ(ncclProxyStop(comm_.get()), ncclSuccess);
  EXPECT_EQ(g_sock.sends, 0);
  EXPECT_EQ(g_sock.closes, 1);
  ps->refCount = 1;
  g_sock.initResult = ncclSystemError;
  EXPECT_EQ(ncclProxyStop(comm_.get()), ncclSystemError);
  ps->refCount = 1;
  ps->peerAddresses = nullptr;
  ASSERT_EQ(ncclProxyStop(comm_.get()), ncclSuccess);
}

TEST_F(ProxyLifecycleMicrotest, Stop_PeerCleanupFailuresPropagate) {
  ASSERT_EQ(ncclProxyInit(comm_.get(), &listen_, addrs_, uds_), ncclSuccess);
  ncclProxyState* ps = comm_->proxyState;
  abort_ = 1;
  ps->peerArraySize = 1;
  ps->peerSocks = (ncclSocket*)calloc(1, sizeof(ncclSocket));
  ps->proxyOps = (ncclProxyOps*)calloc(1, sizeof(ncclProxyOps));
  ps->sharedDevMems = (void**)calloc(1, sizeof(void*));
  auto reset = [&] {
    ps->refCount = 1;
    ps->peerSocks[0].socketDescriptor = SockPair().first;
  };
  reset();
  g_sock.getFdFailAt = g_sock.getFds;
  EXPECT_EQ(ncclProxyStop(comm_.get()), ncclInternalError);
  reset();
  ps->proxyOps[0].pool = (ncclProxyOpsPool*)this;
  g_shm.closeResult = ncclSystemError;
  EXPECT_EQ(ncclProxyStop(comm_.get()), ncclSystemError);
  ps->proxyOps[0].pool = nullptr;
  reset();
  ps->sharedDevMems[0] = &devMem_;
  g_hipIpcCloseResult = hipErrorInvalidValue;
  EXPECT_NE(ncclProxyStop(comm_.get()), ncclSuccess);
  reset();
  g_cuMemEnabled = 1;  // cuMem imports are not IPC handles
  g_hipIpcClosed.clear();
  ASSERT_EQ(ncclProxyStop(comm_.get()), ncclSuccess);
  EXPECT_TRUE(g_hipIpcClosed.empty());
  free(ps->peerSocks);
  free(ps->proxyOps);
  free(ps->sharedDevMems);
  ps->peerSocks = nullptr;
  ps->proxyOps = nullptr;
  ps->sharedDevMems = nullptr;
}

}  // namespace
