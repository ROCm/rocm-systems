/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the GIN context count reported to applications.
//
// NCCL 2.31.2 / NVIDIA/nccl#2301: devComm->ginContextCount was assigned from
// the requested count before that count was rounded up to a multiple of the
// GIN connection count. Contexts were allocated at the rounded size, but
// applications only observed the smaller request, so they could not use the
// extra contexts (and an all-context fence under-counted them).
//
// ncclGinDevCommSetup is exercised here against src/gin/gin_host.cc with a
// scriptable plugin vtable. No GPU, no librccl.so, no network.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "comm.h"
#include "gin/gin_host.h"
#include "nccl_device/net_device.h"

int ncclDebugLevel = 0;
uint64_t ncclDebugMask = 0;
thread_local int ncclDebugNoWarn = 0;

void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}

void ncclSetThreadName(std::thread&, const char*, ...) {}
void ncclSetThreadName(pthread_t, const char*, ...) {}

int64_t ncclParamGinType() { return -1; }

int ncclOsCpuCount(const ncclAffinity&) { return 0; }
ncclResult_t ncclOsSetAffinity(const ncclAffinity&) { return ncclSuccess; }

extern "C" ncclTeam_t ncclTeamRail(ncclComm_t) {
  ncclTeam_t team{};
  team.nRanks = 1;
  team.rank = 0;
  team.stride = 1;
  return team;
}

#include GIN_HOST_CC_PATH

namespace {

struct FakePlugin {
  int createCalls = 0;
  int destroyCalls = 0;
  int nContextsSeen = -1;
  bool contextCountMismatch = false;
  ncclResult_t createResult = ncclSuccess;
  ncclNetDeviceHandle_t handles[NCCL_GIN_MAX_CONNECTIONS]{};
  int tokens[NCCL_GIN_MAX_CONNECTIONS]{};

  static FakePlugin*& current() {
    static FakePlugin* plugin = nullptr;
    return plugin;
  }

  static ncclResult_t CreateContext(void*, ncclGinConfig_t* config, void** ginCtx,
                                    ncclNetDeviceHandle_t** devHandle) {
    FakePlugin& self = *current();
    if (self.createResult != ncclSuccess) return self.createResult;
    const int slot = self.createCalls++;
    if (self.nContextsSeen < 0) {
      self.nContextsSeen = config->nContexts;
    } else if (self.nContextsSeen != config->nContexts) {
      self.contextCountMismatch = true;
    }
    self.tokens[slot] = slot + 1;
    self.handles[slot] = {};
    self.handles[slot].handle = &self.tokens[slot];
    self.handles[slot].netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
    self.handles[slot].needsProxyProgress = 0;
    *ginCtx = &self.tokens[slot];
    *devHandle = &self.handles[slot];
    return ncclSuccess;
  }

  static ncclResult_t DestroyContext(void*) {
    current()->destroyCalls++;
    return ncclSuccess;
  }

  ncclGin_t vtable() const {
    ncclGin_t gin{};
    gin.name = "GinContextCountStub";
    gin.createContext = &CreateContext;
    gin.destroyContext = &DestroyContext;
    return gin;
  }
};

struct CommOwner {
  ncclComm* comm = nullptr;
  ncclSharedResources* shared = nullptr;

  CommOwner() {
    comm = new ncclComm();
    shared = new ncclSharedResources();
    comm->sharedRes = shared;
  }
  ~CommOwner() {
    delete comm;
    delete shared;
  }
  CommOwner(const CommOwner&) = delete;
  CommOwner& operator=(const CommOwner&) = delete;
};

struct SetupResult {
  ncclResult_t rc = ncclInternalError;
  uint32_t reported = 0;
  int perConnection = -1;
  int createCalls = 0;
  bool contextCountMismatch = false;
};

SetupResult setupContexts(int requested, int connections, FakePlugin* plugin) {
  CommOwner owner;
  ncclGin_t gin = plugin->vtable();
  FakePlugin::current() = plugin;

  ncclGinState* ginState = &owner.shared->ginState;
  ginState->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  ginState->numActiveBackends = 1;
  ginState->proxyThreadsCreated = false;
  ncclGinBackendState* backend = &ginState->backends[0];
  backend->ginType = NCCL_GIN_TYPE_PROXY;
  backend->ncclGin = &gin;
  backend->ginCommCount = connections;
  backend->supportsStrongSignals = true;
  backend->supportsVASignals = true;

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs.ginContextCount = requested;
  reqs.ginType = NCCL_GIN_TYPE_PROXY;

  ncclDevComm devComm{};
  SetupResult result;
  result.rc = ncclGinDevCommSetup(owner.comm, &reqs, &devComm, /*deviceCodeVersion=*/0);
  result.reported = devComm.ginContextCount;
  result.perConnection = plugin->nContextsSeen;
  result.createCalls = plugin->createCalls;
  result.contextCountMismatch = plugin->contextCountMismatch;
  if (result.rc == ncclSuccess) {
    EXPECT_EQ(ncclGinDevCommFree(owner.comm, &devComm), ncclSuccess);
  }
  FakePlugin::current() = nullptr;
  return result;
}

struct ContextCountCase {
  int requested;
  int connections;
  int allocated;
};

// The count returned on ncclDevComm is the rounded-up allocation, not the
// request. per-connection contexts * connection count must equal that report,
// which is what NVIDIA/nccl#2301 failed to do.
TEST(GinContextCount, ReportsAllocatedCount) {
  const ContextCountCase cases[] = {
      // Already a multiple: the report matches the request.
      {1, 1, 1},
      {4, 1, 4},
      {3, 3, 3},
      {6, 3, 6},
      // Not a multiple: allocation rounds up, and that larger count is reported.
      // The pre-#2301 assignment reported `requested` instead.
      {1, 4, 4},
      {4, 3, 6},
      {5, 2, 6},
      {3, 2, 4},
  };

  for (const ContextCountCase& spec : cases) {
    SCOPED_TRACE("requested=" + std::to_string(spec.requested) +
                 " connections=" + std::to_string(spec.connections));
    FakePlugin plugin;
    const SetupResult result = setupContexts(spec.requested, spec.connections, &plugin);

    ASSERT_EQ(result.rc, ncclSuccess);
    EXPECT_EQ(result.reported, static_cast<uint32_t>(spec.allocated));
    EXPECT_GE(result.reported, static_cast<uint32_t>(spec.requested));
    EXPECT_EQ(result.reported % static_cast<uint32_t>(spec.connections), 0u);
    EXPECT_EQ(result.createCalls, spec.connections);
    EXPECT_FALSE(result.contextCountMismatch);
    ASSERT_GT(result.perConnection, 0);
    EXPECT_EQ(static_cast<uint32_t>(result.perConnection * spec.connections), result.reported);
  }
}

TEST(GinContextCount, CreateFailureClearsReportedCount) {
  FakePlugin plugin;
  plugin.createResult = ncclSystemError;
  const SetupResult result = setupContexts(/*requested=*/4, /*connections=*/3, &plugin);

  EXPECT_NE(result.rc, ncclSuccess);
  EXPECT_EQ(result.reported, 0u);
  EXPECT_EQ(result.createCalls, 0);
}

}  // namespace
