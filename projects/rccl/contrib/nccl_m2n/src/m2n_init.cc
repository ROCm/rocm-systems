/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>

#include "nccl.h"
#include "nccl_m2n.h"
#include "m2n_checks.h"
#include "reshard_types.h"
#include "m2n_log.h"
#include "reshard_internal.h"

static std::mutex gM2nApiMutex;
static std::mutex gM2nRuntimeMutex;
static std::weak_ptr<ncclM2nRuntime> gM2nRuntime;
static std::mutex gM2nDefaultHandleMutex;
static std::shared_ptr<ncclM2nHandleState> gM2nDefaultHandle;
static std::mutex gM2nHandleTableMutex;
static uintptr_t gNextM2nHandleId = 1;
static std::unordered_map<ncclM2nHandle_t, std::shared_ptr<ncclM2nHandleState>> gM2nHandleTable;
static thread_local char gM2nLastError[M2N_LAST_ERROR_BYTES] = {};
static std::atomic<bool> gM2nProcessExiting{false};
static thread_local std::unique_lock<std::mutex>* gCurrentM2nApiLock = nullptr;
static int gM2nApiInFlightCalls = 0;

void m2nClearLastError() {
  gM2nLastError[0] = '\0';
}

void m2nSetLastError(const char* message) {
  (void)snprintf(gM2nLastError, sizeof(gM2nLastError), "%s", message);
}

struct M2nExitGuard {
  ~M2nExitGuard() {
    gM2nProcessExiting.store(true, std::memory_order_relaxed);
  }
};
static M2nExitGuard gM2nExitGuard;

static ncclResult_t copyHandleConfig(ncclM2nHandleState* handle, const ncclM2nConfig_t* config) {
  if (config == nullptr) {
    handle->config = NCCL_M2N_CONFIG_INITIALIZER;
    return ncclSuccess;
  }

  NCCL_M2N_CHECK(validateReshardConfigHeader(config));
  handle->config = *config;
  return ncclSuccess;
}

static void deleteHandleState(ncclM2nHandleState* handle) {
  // The OS reclaims abandoned process-lifetime state; CUDA teardown is unsafe
  // after main when the CUDA runtime may already be unloading.
  if (gM2nProcessExiting.load(std::memory_order_relaxed)) return;
  std::lock_guard<std::mutex> lock(gM2nRuntimeMutex);
  // Releasing the last state also releases the shared runtime and tears down
  // its process-global resources before a new runtime epoch can be created.
  delete handle;
}

static ncclResult_t createHandle(std::shared_ptr<ncclM2nHandleState>* handle) {
  try {
    *handle = std::shared_ptr<ncclM2nHandleState>(new ncclM2nHandleState(), deleteHandleState);
  } catch (const std::bad_alloc&) {
    NCCL_M2N_FAIL(ncclSystemError, -1, "ncclM2nInit: failed to allocate handle state");
  }
  return ncclSuccess;
}

static ncclResult_t createRuntime(std::shared_ptr<ncclM2nRuntime>* runtime) {
  try {
    *runtime = std::make_shared<ncclM2nRuntime>();
  } catch (const std::bad_alloc&) {
    NCCL_M2N_FAIL(ncclSystemError, -1, "ncclM2nInit: failed to allocate runtime state");
  }
  return ncclSuccess;
}

static void resolveNumCtas() {
  if (gReshardNumCtasOverride > 0) {
    gReshardNumCtas = gReshardNumCtasOverride;
    return;
  }
  const bool capNumCtas = gReshardMaxCta > 0 && gReshardMaxCta < DEFAULT_NUM_CTAS;
  gReshardNumCtas = capNumCtas ? gReshardMaxCta : DEFAULT_NUM_CTAS;
}

static ncclResult_t initializeRuntimeForFirstHandle(const ncclM2nHandleState* handle) {
  /* A quarantine only outlives the epoch that raised it. The previous epoch's
   * caches are gone by now, so a fresh one starts clean. */
  reshardClearResourceQuarantine();
  resetReshardRuntimeConfig();
  applyReshardConfig(&handle->config);
  applyReshardEnv();
  resolveNumCtas();
  return ncclSuccess;
}

static bool configMatches(const ncclM2nConfig_t& a, const ncclM2nConfig_t& b) {
  // Compare every public config field; update this when ncclM2nConfig_t grows.
  return a.size == b.size && a.magic == b.magic && a.version == b.version && a.maxCta == b.maxCta;
}

static ncclResult_t attachRuntime(ncclM2nHandleState* handle, bool warnOnConfigMismatch) {
  std::lock_guard<std::mutex> lock(gM2nRuntimeMutex);
  std::shared_ptr<ncclM2nRuntime> runtime = gM2nRuntime.lock();
  if (runtime == nullptr) {
    NCCL_M2N_CHECK(createRuntime(&runtime));
    runtime->firstConfig = handle->config;
    NCCL_M2N_CHECK(initializeRuntimeForFirstHandle(handle));
    gM2nRuntime = runtime;
  } else if (warnOnConfigMismatch && !configMatches(runtime->firstConfig, handle->config)) {
    RESHARD_WARN(-1, "ncclM2nInit: config ignored because runtime was already "
                     "configured by an earlier ncclM2nInit; finalize all handles "
                     "before reconfiguring.");
  }

  handle->runtime = runtime;
  return ncclSuccess;
}

static ncclResult_t createInitializedHandle(std::shared_ptr<ncclM2nHandleState>* handle,
                                            const ncclM2nConfig_t* config, bool warnOnConfigMismatch) {
  std::shared_ptr<ncclM2nHandleState> newHandle;
  NCCL_M2N_CHECK(createHandle(&newHandle));
  NCCL_M2N_CHECK(copyHandleConfig(newHandle.get(), config));
  NCCL_M2N_CHECK(attachRuntime(newHandle.get(), warnOnConfigMismatch));
  *handle = std::move(newHandle);
  return ncclSuccess;
}

static ncclResult_t registerExplicitHandle(const std::shared_ptr<ncclM2nHandleState>& handle,
                                           ncclM2nHandle_t* token) {
  ncclM2nHandle_t newToken = nullptr;
  try {
    std::lock_guard<std::mutex> lock(gM2nHandleTableMutex);
    if (gNextM2nHandleId == 0) {
      NCCL_M2N_FAIL(ncclSystemError, -1, "ncclM2nInit: handle identifier space is exhausted");
    }
    newToken = reinterpret_cast<ncclM2nHandle_t>(gNextM2nHandleId++);
    if (!gM2nHandleTable.emplace(newToken, handle).second) {
      NCCL_M2N_FAIL(ncclSystemError, -1, "ncclM2nInit: failed to register handle %p", (void*)newToken);
    }
  } catch (const std::bad_alloc&) {
    NCCL_M2N_FAIL(ncclSystemError, -1, "ncclM2nInit: failed to allocate handle-table entry");
  }
  *token = newToken;
  return ncclSuccess;
}

static ncclResult_t destroyExplicitHandle(ncclM2nHandle_t handle) {
  std::shared_ptr<ncclM2nHandleState> ownedHandle;
  {
    std::lock_guard<std::mutex> lock(gM2nHandleTableMutex);
    auto it = gM2nHandleTable.find(handle);
    NCCL_M2N_CHECK_ARG(it != gM2nHandleTable.end(), -1,
                       "ncclM2nFinalize: handle %p is unknown or already finalized", (void*)handle);
    ownedHandle = std::move(it->second);
    gM2nHandleTable.erase(it);
  }
  ownedHandle.reset();
  return ncclSuccess;
}

ncclM2nRuntime::~ncclM2nRuntime() {
  packStagingSynchronize();
  reshardSplitSynchronizeCommBLanes();
  cacheFinalize();
  stagingPipeLaunchCompletionFinalize();
  reshardSplitCommFinalize();
  packStagingFinalize();
  resetReshardRuntimeConfig();
}

std::mutex& getM2nApiMutex() {
  return gM2nApiMutex;
}

bool m2nApiHasConcurrentCalls() {
  return gM2nApiInFlightCalls > 1;
}

extern "C" {
M2nApiLock::M2nApiLock() : lock_(gM2nApiMutex) {
  gM2nApiInFlightCalls++;
  gCurrentM2nApiLock = &lock_;
}

M2nApiLock::~M2nApiLock() {
  gCurrentM2nApiLock = nullptr;
  gM2nApiInFlightCalls--;
}

M2nApiUnlock::M2nApiUnlock() {
  if (gCurrentM2nApiLock != nullptr && gCurrentM2nApiLock->owns_lock()) {
    gCurrentM2nApiLock->unlock();
    unlocked_ = true;
  }
}

M2nApiUnlock::~M2nApiUnlock() {
  if (unlocked_) {
    gCurrentM2nApiLock->lock();
  }
}

ncclResult_t ncclM2nInit(ncclM2nHandle_t* handle, const ncclM2nConfig_t* config) {
  M2nApiLock apiLock;
  m2nClearLastError();
  NCCL_M2N_CHECK_ARG(handle != nullptr, -1, "ncclM2nInit: handle output must be non-null");
  *handle = nullptr;
  std::shared_ptr<ncclM2nHandleState> newHandle;
  NCCL_M2N_CHECK(createInitializedHandle(&newHandle, config, true));
  ncclResult_t result = registerExplicitHandle(newHandle, handle);
  if (result != ncclSuccess) {
    newHandle.reset();
  }
  return result;
}

ncclResult_t ncclM2nFinalize(ncclM2nHandle_t handle) {
  M2nApiLock apiLock;
  m2nClearLastError();
  NCCL_M2N_CHECK_ARG(!m2nGroupIsActive(), -1,
                     "ncclM2nFinalize: cannot finalize while a group is active on this host thread");
  if (handle != nullptr) {
    return destroyExplicitHandle(handle);
  }

  std::shared_ptr<ncclM2nHandleState> defaultHandle;
  {
    std::lock_guard<std::mutex> lock(gM2nDefaultHandleMutex);
    defaultHandle = std::move(gM2nDefaultHandle);
  }
  defaultHandle.reset();
  return ncclSuccess;
}

const char* ncclM2nGetLastError(void) {
  return gM2nLastError;
}

} // extern "C"

ncclResult_t acquireM2nHandle(ncclM2nHandle_t token, std::shared_ptr<ncclM2nHandleState>* handle) {
  NCCL_M2N_CHECK_ARG(handle != nullptr, -1, "acquireM2nHandle: handle state output must be non-null");
  *handle = nullptr;

  if (token != nullptr) {
    std::lock_guard<std::mutex> lock(gM2nHandleTableMutex);
    auto it = gM2nHandleTable.find(token);
    NCCL_M2N_CHECK_ARG(it != gM2nHandleTable.end(), -1,
                       "NCCL M2N handle %p is unknown or already finalized", (void*)token);
    *handle = it->second;
    return ncclSuccess;
  }

  std::lock_guard<std::mutex> lock(gM2nDefaultHandleMutex);
  if (gM2nDefaultHandle == nullptr) {
    ncclM2nConfig_t config = NCCL_M2N_CONFIG_INITIALIZER;
    NCCL_M2N_CHECK(createInitializedHandle(&gM2nDefaultHandle, &config, false));
  }
  *handle = gM2nDefaultHandle;
  return ncclSuccess;
}
