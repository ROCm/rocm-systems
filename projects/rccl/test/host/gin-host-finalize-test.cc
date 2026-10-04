/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/gin/gin_host.cc's ncclGinHostFinalize and the
// register / deregister walks that must not call through NULLed ginComms[]
// after host finalize (AICOMRCCL-2739).
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

#include "nccl.h"
#include "comm.h"
#include "nccl_gin.h"
#include "gin/gin_host.h"
#include "debug.h"

// Lean seams for this binary only -- do not pull fakes/nccl_fakes.cc (its
// std::function seams keep large unrelated graphs live against --gc-sections).
int ncclDebugLevel = 0;
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

namespace {

struct FakeGinHost {
  int closeCollCalls = 0;
  int regMrSymCalls = 0;
  int deregMrSymCalls = 0;
  void* lastCloseColl = nullptr;

  static FakeGinHost*& currentPtr() {
    static FakeGinHost* p = nullptr;
    return p;
  }
  static FakeGinHost& current() { return *currentPtr(); }
  static void setCurrent(FakeGinHost* p) { currentPtr() = p; }

  static ncclResult_t CloseColl(void* collComm) {
    FakeGinHost& self = current();
    ++self.closeCollCalls;
    self.lastCloseColl = collComm;
    return ncclSuccess;
  }

  static ncclResult_t RegMrSym(void* /*collComm*/, void* /*data*/, size_t /*size*/, int /*type*/,
                               uint64_t /*mrFlags*/, void** mhandle, void** ginHandle) {
    FakeGinHost& self = current();
    ++self.regMrSymCalls;
    if (mhandle) *mhandle = &self;
    if (ginHandle) *ginHandle = &self;
    return ncclSuccess;
  }

  static ncclResult_t DeregMrSym(void* /*collComm*/, void* /*mhandle*/) {
    ++current().deregMrSymCalls;
    return ncclSuccess;
  }

  ncclGin_t vtable() {
    ncclGin_t gin{};
    gin.name = "GinHostFinalizeStub";
    gin.closeColl = &CloseColl;
    gin.regMrSym = &RegMrSym;
    gin.deregMrSym = &DeregMrSym;
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

  void SetUp() override {
    FakeGinHost::setCurrent(&fake_);
    gin_ = fake_.vtable();
    // ncclSharedResources is value-initialized by make_unique; do not memset it —
    // ginState embeds std::thread / mutex / atomic.
    std::memset(&comm_, 0, sizeof(comm_));
    comm_.sharedRes = sharedRes_.get();

    struct ncclGinState* ginState = &sharedRes_->ginState;
    ginState->connected = true;
    ginState->supported = true;
    ginState->numActiveBackends = 1;
    struct ncclGinBackendState* backend = &ginState->backends[0];
    backend->ncclGin = &gin_;
    backend->ginInstance = ginInstance_;
    backend->pluginIndex = 0;
    backend->ginCommCount = 1;
    backend->ginComms[0] = ginComm0_;
    backend->closed = false;
  }

  void TearDown() override { FakeGinHost::setCurrent(nullptr); }

  struct ncclGinState* ginState() { return &sharedRes_->ginState; }
  struct ncclGinBackendState* backend() { return &ginState()->backends[0]; }
};

TEST_F(GinHostFinalizeTest, PreservesBackendRecordsAndMarksClosed) {
  EXPECT_EQ(ncclGinHostFinalize(&comm_), ncclSuccess);

  EXPECT_EQ(fake_.closeCollCalls, 1);
  EXPECT_EQ(fake_.lastCloseColl, ginComm0_);
  EXPECT_EQ(backend()->ginComms[0], nullptr);
  EXPECT_TRUE(backend()->closed);

  // ncclGinFinalize reads these; clearing them here was the leak.
  EXPECT_EQ(ginState()->numActiveBackends, 1);
  EXPECT_EQ(backend()->ginInstance, ginInstance_);
  EXPECT_EQ(backend()->ginCommCount, 1);
  EXPECT_FALSE(ginState()->connected);
  EXPECT_FALSE(ginState()->supported);
}

TEST_F(GinHostFinalizeTest, RegisterAndDeregisterSkipClosedBackend) {
  ASSERT_EQ(ncclGinHostFinalize(&comm_), ncclSuccess);

  void* hostWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS] = {};
  ncclGinWindow_t devWins[NCCL_GIN_MAX_CONNECTIONS * NCCL_GIN_MAX_ACTIVE_BACKENDS] = {};
  hostWins[0] = reinterpret_cast<void*>(0x3333);

  EXPECT_EQ(ncclGinRegister(&comm_, reinterpret_cast<void*>(0x1), 8, hostWins, devWins, 0), ncclSuccess);
  EXPECT_EQ(ncclGinDeregister(&comm_, hostWins), ncclSuccess);

  // Surviving splitShare siblings must not call through the NULLed ginComms[].
  EXPECT_EQ(fake_.regMrSymCalls, 0);
  EXPECT_EQ(fake_.deregMrSymCalls, 0);
}

TEST_F(GinHostFinalizeTest, AlreadyDisconnectedIsNoOp) {
  ginState()->connected = false;

  EXPECT_EQ(ncclGinHostFinalize(&comm_), ncclSuccess);
  EXPECT_EQ(fake_.closeCollCalls, 0);
  EXPECT_FALSE(backend()->closed);
  EXPECT_EQ(backend()->ginComms[0], ginComm0_);
}

}  // namespace
