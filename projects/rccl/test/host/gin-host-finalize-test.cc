/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/gin/gin_host.cc's ncclGinHostFinalize, the
// register / deregister / DevCommSetup walks that must not call through NULLed
// ginComms[] after host finalize, and ncclGinConnectOnce's failure and
// finalized paths (AICOMRCCL-2739).
//
// Own binary: rccl-UnitTestsMicro already defines the GIN host entry points in
// fakes/dev_runtime_micro_fakes.cc for dev-runtime-test.cc. Compiling gin_host.cc
// into that binary is a duplicate-symbol error. GinFinalizeTest in
// gin-plugin-init-test.cc covers ncclGinFinalize against a hand-built ginState;
// this suite covers the host half that preserves those records and marks
// backends closed.

#include <gtest/gtest.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "nccl.h"
#include "comm.h"
#include "nccl_gin.h"
#include "gin/gin_host.h"
#include "debug.h"

// Lean seams for this binary only -- do not pull fakes/nccl_fakes.cc (its
// std::function seams keep large unrelated graphs live against --gc-sections).
uint32_t ncclDebugLevelMask = 0;
uint64_t ncclDebugMask = 0;
FILE* ncclDebugFile = nullptr;
thread_local int ncclDebugNoWarn = 0;
char ncclLastError[1024] = {};

void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int, const char* fmt, ...) {
  if (!fmt) return;
  std::va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fputc('\n', stderr);
}

static int64_t g_loadParam(const char*, int64_t deftVal) { return deftVal; }

#include "fakes/param_redirect.h"

#include GIN_HOST_CC_PATH

// Link seams for ncclGinConnectOnce / ncclGinDevCommSetup. A single-rank comm
// makes every AllGather a no-op.
static int g_localGinDevCount = 2;

int64_t ncclParamGinType() { return NCCL_GIN_TYPE_NONE; }

ncclResult_t ncclTopoGetLocalGinDevs(struct ncclComm*, int* localGinDevs, int* localGinCount) {
  for (int i = 0; i < g_localGinDevCount; i++) localGinDevs[i] = i;
  *localGinCount = g_localGinDevCount;
  return ncclSuccess;
}

ncclResult_t bootstrapAllGather(void*, void*, int) { return ncclSuccess; }

ncclTeam_t ncclTeamWorld(ncclComm_t comm) { return {comm->nRanks, comm->rank, 1}; }
ncclTeam_t ncclTeamRail(ncclComm_t comm) { return {comm->nRanks, comm->rank, 1}; }
int ncclTeamRankToWorld(ncclComm_t, ncclTeam_t team, int rank) { return rank * team.stride; }

void ncclSetThreadName(std::thread&, const char*, ...) {}

namespace {

struct FakeGinHost {
  int closeCollCalls = 0;
  int regMrSymCalls = 0;
  int deregMrSymCalls = 0;
  int listenCalls = 0;
  int connectCalls = 0;
  int closeListenCalls = 0;
  int createContextCalls = 0;
  std::vector<void*> closedColls;
  std::vector<void*> deregColls;
  // backend->closed as seen from inside each closeColl call.
  std::vector<bool> closedFlagAtClose;
  struct ncclGinBackendState* watchedBackend = nullptr;
  void* closeCollFailOn = nullptr;
  int connectFailOnCall = 0;  // 1-based; 0 never fails
  uintptr_t nextCollComm = 0x5000;

  static FakeGinHost*& currentPtr() {
    static FakeGinHost* p = nullptr;
    return p;
  }
  static FakeGinHost& current() { return *currentPtr(); }
  static void setCurrent(FakeGinHost* p) { currentPtr() = p; }

  static ncclResult_t CloseColl(void* collComm) {
    FakeGinHost& self = current();
    ++self.closeCollCalls;
    self.closedColls.push_back(collComm);
    if (self.watchedBackend) self.closedFlagAtClose.push_back(self.watchedBackend->closed);
    return collComm == self.closeCollFailOn ? ncclSystemError : ncclSuccess;
  }

  static ncclResult_t RegMrSym(void* /*collComm*/, void* /*data*/, size_t /*size*/, int /*type*/,
                               uint64_t /*mrFlags*/, void** mhandle, void** ginHandle) {
    FakeGinHost& self = current();
    ++self.regMrSymCalls;
    if (mhandle) *mhandle = &self;
    if (ginHandle) *ginHandle = &self;
    return ncclSuccess;
  }

  static ncclResult_t DeregMrSym(void* collComm, void* /*mhandle*/) {
    FakeGinHost& self = current();
    ++self.deregMrSymCalls;
    self.deregColls.push_back(collComm);
    return ncclSuccess;
  }

  static ncclResult_t Devices(int* ndev) {
    *ndev = 1;
    return ncclSuccess;
  }

  static ncclResult_t GetProperties(int /*dev*/, ncclNetProperties_t* props) {
    std::memset(props, 0, sizeof(*props));
    return ncclSuccess;
  }

  static ncclResult_t Listen(void* /*ctx*/, int /*dev*/, void* /*handle*/, void** listenComm) {
    FakeGinHost& self = current();
    ++self.listenCalls;
    *listenComm = &self;
    return ncclSuccess;
  }

  static ncclResult_t Connect(void* /*ctx*/, void* /*handles*/[], int /*nranks*/, int /*rank*/,
                              void* /*listenComm*/, void** collComm) {
    FakeGinHost& self = current();
    if (++self.connectCalls == self.connectFailOnCall) return ncclSystemError;
    *collComm = reinterpret_cast<void*>(self.nextCollComm++);
    return ncclSuccess;
  }

  static ncclResult_t CloseListen(void* /*listenComm*/) {
    ++current().closeListenCalls;
    return ncclSuccess;
  }

  static ncclResult_t CreateContext(void* /*collComm*/, ncclGinConfig_t* /*config*/, void** /*ginCtx*/,
                                    ncclNetDeviceHandle_t** /*devHandle*/) {
    ++current().createContextCalls;
    return ncclInternalError;
  }

  static ncclResult_t DestroyContext(void* /*ginCtx*/) { return ncclSuccess; }

  ncclGin_t vtable() {
    ncclGin_t gin{};
    gin.name = "GinHostFinalizeStub";
    gin.devices = &Devices;
    gin.getProperties = &GetProperties;
    gin.listen = &Listen;
    gin.connect = &Connect;
    gin.createContext = &CreateContext;
    gin.regMrSym = &RegMrSym;
    gin.deregMrSym = &DeregMrSym;
    gin.destroyContext = &DestroyContext;
    gin.closeColl = &CloseColl;
    gin.closeListen = &CloseListen;
    return gin;
  }
};

class GinHostFinalizeTest : public ::testing::Test {
 protected:
  FakeGinHost fake_;
  ncclGin_t gin_{};
  std::unique_ptr<ncclSharedResources> sharedRes_ = std::make_unique<ncclSharedResources>();
  ncclComm comm_{};
  void* ginInstance_ = reinterpret_cast<void*>(0x1111);
  void* ginComm0_ = reinterpret_cast<void*>(0x2222);
  void* ginComm1_ = reinterpret_cast<void*>(0x2223);
  void* hostWins_[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS] = {};
  ncclGinWindow_t devWins_[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS] = {};
  uint32_t winGens_[NCCL_GIN_MAX_ACTIVE_BACKENDS] = {};

  void SetUp() override {
    FakeGinHost::setCurrent(&fake_);
    gin_ = fake_.vtable();
    g_localGinDevCount = 2;
    // Both are already value-initialized (make_unique, and `ncclComm comm_{}`);
    // do not memset either. ginState embeds std::thread / mutex / atomic, and
    // ncclComm embeds ncclRmaState, which holds a thread, mutex and condvar.
    comm_.sharedRes = sharedRes_.get();
    comm_.rank = 0;
    comm_.nRanks = 1;
    comm_.contiguousRanksPerHost = 1;
    comm_.symmetricSupport = true;
    comm_.globalGinSupport = NCCL_GIN_CONNECTION_FULL;

    struct ncclGinState* ginState = &sharedRes_->ginState;
    ginState->connected = true;
    ginState->supported = true;
    ginState->numActiveBackends = 1;
    struct ncclGinBackendState* backend = &ginState->backends[0];
    backend->ncclGin = &gin_;
    backend->ginInstance = ginInstance_;
    backend->ginType = NCCL_GIN_TYPE_PROXY;
    backend->pluginIndex = 0;
    backend->ginCommCount = 2;
    backend->ginComms[0] = ginComm0_;
    backend->ginComms[1] = ginComm1_;
    backend->closed = false;
    backend->connectGeneration = 1;
    fake_.watchedBackend = backend;
  }

  void TearDown() override { FakeGinHost::setCurrent(nullptr); }

  struct ncclGinState* ginState() { return &sharedRes_->ginState; }
  struct ncclGinBackendState* backend() { return &ginState()->backends[0]; }

  ncclResult_t registerWindow() {
    std::memset(hostWins_, 0, sizeof(hostWins_));
    return ncclGinRegister(&comm_, reinterpret_cast<void*>(0x1), 8, hostWins_, devWins_, winGens_, 0);
  }
};

TEST_F(GinHostFinalizeTest, PreservesBackendRecordsAndMarksClosed) {
  EXPECT_EQ(ncclGinHostFinalize(&comm_), ncclSuccess);

  EXPECT_EQ(fake_.closedColls, (std::vector<void*>{ginComm0_, ginComm1_}));
  EXPECT_EQ(backend()->ginComms[0], nullptr);
  EXPECT_EQ(backend()->ginComms[1], nullptr);
  EXPECT_TRUE(backend()->closed);
  EXPECT_TRUE(ginState()->finalized);

  // ncclGinFinalize reads these; clearing them here was the leak.
  EXPECT_EQ(ginState()->numActiveBackends, 1);
  EXPECT_EQ(backend()->ginInstance, ginInstance_);
  EXPECT_EQ(backend()->ginCommCount, 2);
  EXPECT_FALSE(ginState()->connected);
  EXPECT_FALSE(ginState()->supported);
}

// A failure part-way must never leave NULL ginComms[] with closed still false,
// or register / deregister would call through them.
TEST_F(GinHostFinalizeTest, MarksClosedBeforeClosingAnyHandle) {
  EXPECT_EQ(ncclGinHostFinalize(&comm_), ncclSuccess);
  EXPECT_EQ(fake_.closedFlagAtClose, (std::vector<bool>{true, true}));
}

TEST_F(GinHostFinalizeTest, ResetsHostStateButKeepsBackendRecords) {
  struct ncclGinState* gs = ginState();
  auto* devCommsSentinel = reinterpret_cast<struct ncclGinStateDevComm*>(0x4444);
  gs->proxyNthreads = 3;
  gs->proxyThreadsCreated = true;  // no joinable threads: the join loop skips them
  gs->devComms = devCommsSentinel;
  gs->asyncResult = ncclSystemError;
  gs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  gs->writePending.store(true);

  EXPECT_EQ(ncclGinHostFinalize(&comm_), ncclSuccess);

  EXPECT_EQ(gs->proxyNthreads, 0);
  EXPECT_FALSE(gs->proxyThreadsCreated);
  EXPECT_FALSE(gs->proxyThreadStopSignal.load());
  EXPECT_FALSE(gs->writePending.load());
  EXPECT_EQ(gs->devComms, nullptr);
  EXPECT_EQ(gs->asyncResult, ncclSuccess);
  EXPECT_EQ(gs->ginConnectionType, NCCL_GIN_CONNECTION_NONE);
  EXPECT_TRUE(gs->finalized);
}

// commFree ignores this result until ncclGinFinalize and the frees have run, so
// one failing handle must not leak the others or skip the reset.
TEST_F(GinHostFinalizeTest, CloseCollFailure_ClosesEveryHandleResetsAndReturnsError) {
  fake_.closeCollFailOn = ginComm0_;
  ginState()->proxyNthreads = 2;

  EXPECT_EQ(ncclGinHostFinalize(&comm_), ncclSystemError);

  EXPECT_EQ(fake_.closedColls, (std::vector<void*>{ginComm0_, ginComm1_}));
  EXPECT_EQ(backend()->ginComms[0], nullptr);
  EXPECT_EQ(backend()->ginComms[1], nullptr);
  EXPECT_TRUE(backend()->closed);
  EXPECT_TRUE(ginState()->finalized);
  EXPECT_FALSE(ginState()->connected);
  EXPECT_FALSE(ginState()->supported);
  EXPECT_EQ(ginState()->proxyNthreads, 0);
  EXPECT_EQ(ginState()->numActiveBackends, 1);
  EXPECT_EQ(backend()->ginInstance, ginInstance_);
}

TEST_F(GinHostFinalizeTest, RegisterAndDeregisterSkipClosedBackend) {
  ASSERT_EQ(ncclGinHostFinalize(&comm_), ncclSuccess);

  hostWins_[0] = reinterpret_cast<void*>(0x3333);
  EXPECT_EQ(ncclGinRegister(&comm_, reinterpret_cast<void*>(0x1), 8, hostWins_, devWins_, winGens_, 0),
            ncclSuccess);
  EXPECT_EQ(ncclGinDeregister(&comm_, hostWins_, winGens_), ncclSuccess);

  // Surviving splitShare siblings must not call through the NULLed ginComms[].
  EXPECT_EQ(fake_.regMrSymCalls, 0);
  EXPECT_EQ(fake_.deregMrSymCalls, 0);
}

TEST_F(GinHostFinalizeTest, DevCommSetupSkipsClosedBackend) {
  ASSERT_EQ(ncclGinHostFinalize(&comm_), ncclSuccess);

  ncclDevCommRequirements reqs{};
  ncclDevComm devComm{};
  EXPECT_EQ(ncclGinDevCommSetup(&comm_, &reqs, &devComm, NCCL_VERSION_CODE), ncclInternalError);
  EXPECT_EQ(fake_.createContextCalls, 0);
}

// Even if supported were re-armed, a finalized sharedRes must not reconnect:
// ginInstance still points at the comm that ran ncclGinInit, which may be freed.
TEST_F(GinHostFinalizeTest, ConnectOnceAfterFinalizeIsRejectedWithoutListening) {
  ASSERT_EQ(ncclGinHostFinalize(&comm_), ncclSuccess);
  ginState()->supported = true;

  EXPECT_EQ(ncclGinConnectOnce(&comm_), ncclInvalidUsage);
  EXPECT_EQ(fake_.listenCalls, 0);
  EXPECT_FALSE(ginState()->connected);
  EXPECT_TRUE(backend()->closed);
}

// The end-to-end shape of the original bug: a window registered before a
// sibling's teardown is never deregistered against a handle it was not
// registered on, and the survivor cannot reconnect underneath it.
TEST_F(GinHostFinalizeTest, WindowFromBeforeFinalize_IsNeverDeregisteredAgainstAnotherHandle) {
  ASSERT_EQ(registerWindow(), ncclSuccess);
  ASSERT_EQ(fake_.regMrSymCalls, 2);
  EXPECT_EQ(winGens_[0], 1u);

  ASSERT_EQ(ncclGinHostFinalize(&comm_), ncclSuccess);
  ginState()->supported = true;
  EXPECT_EQ(ncclGinConnectOnce(&comm_), ncclInvalidUsage);

  EXPECT_EQ(ncclGinDeregister(&comm_, hostWins_, winGens_), ncclSuccess);
  EXPECT_EQ(fake_.deregMrSymCalls, 0);
}

// A connect failure part-way closes what was opened, marks the backend closed so
// windows skip it, and a retry repopulates under a new generation.
TEST_F(GinHostFinalizeTest, ConnectOnceFailureThenRetry_ReopensUnderNewGeneration) {
  ginState()->connected = false;
  backend()->ginComms[0] = nullptr;
  backend()->ginComms[1] = nullptr;
  fake_.connectFailOnCall = 2;

  EXPECT_EQ(ncclGinConnectOnce(&comm_), ncclSystemError);
  void* const firstAttemptComm = reinterpret_cast<void*>(0x5000);
  EXPECT_EQ(fake_.closedColls, (std::vector<void*>{firstAttemptComm}));
  EXPECT_EQ(fake_.closeListenCalls, 2);  // the successful pass, then the fail path
  EXPECT_EQ(backend()->ginComms[0], nullptr);
  EXPECT_EQ(backend()->ginComms[1], nullptr);
  EXPECT_TRUE(backend()->closed);
  EXPECT_FALSE(ginState()->connected);
  EXPECT_FALSE(ginState()->finalized);
  EXPECT_EQ(backend()->connectGeneration, 1u);

  EXPECT_EQ(registerWindow(), ncclSuccess);
  EXPECT_EQ(fake_.regMrSymCalls, 0);

  fake_.connectFailOnCall = 0;
  ASSERT_EQ(ncclGinConnectOnce(&comm_), ncclSuccess);
  EXPECT_TRUE(ginState()->connected);
  EXPECT_FALSE(backend()->closed);
  EXPECT_EQ(backend()->connectGeneration, 2u);
  void* const retryComm0 = backend()->ginComms[0];
  void* const retryComm1 = backend()->ginComms[1];
  EXPECT_NE(retryComm0, nullptr);
  EXPECT_NE(retryComm1, nullptr);

  ASSERT_EQ(registerWindow(), ncclSuccess);
  EXPECT_EQ(winGens_[0], 2u);
  EXPECT_EQ(ncclGinDeregister(&comm_, hostWins_, winGens_), ncclSuccess);
  EXPECT_EQ(fake_.deregColls, (std::vector<void*>{retryComm0, retryComm1}));
}

TEST_F(GinHostFinalizeTest, DeregisterSkipsWindowStampedByAnotherGeneration) {
  ASSERT_EQ(registerWindow(), ncclSuccess);
  backend()->connectGeneration++;

  EXPECT_EQ(ncclGinDeregister(&comm_, hostWins_, winGens_), ncclSuccess);
  EXPECT_EQ(fake_.deregMrSymCalls, 0);
}

TEST_F(GinHostFinalizeTest, AlreadyDisconnectedIsNoOp) {
  ginState()->connected = false;

  EXPECT_EQ(ncclGinHostFinalize(&comm_), ncclSuccess);
  EXPECT_EQ(fake_.closeCollCalls, 0);
  EXPECT_FALSE(backend()->closed);
  EXPECT_FALSE(ginState()->finalized);
  EXPECT_EQ(backend()->ginComms[0], ginComm0_);
}

}  // namespace
