/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hip/hip_runtime.h"
#include "hip_internal.hpp"

#include <atomic>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include <vector>

namespace hip {

typedef struct DevResourceDesc {
  std::vector<hipDevResource> resources;
  int deviceId{-1};
} DevResourceDesc;

class ExecutionCtx {
public:
  ExecutionCtx(int deviceId, DevResourceDesc* desc, uint32_t flags);
  ~ExecutionCtx();

  hipError_t Create();

  int deviceId() const { return deviceId_; }
  hip::Device* device() const { return g_devices[deviceId_]; }
  const std::vector<uint32_t>& cuMask() const { return cuMask_; }
  uint32_t cuCount() const { return cuCount_; }
  uint32_t flags() const { return flags_; }
  uint64_t ctxId() const { return ctxId_; }
  std::recursive_mutex& lock() { return lock_; }

  void addStream(hip::Stream* stream);

  hipError_t synchronize();
  hipError_t recordEvent(hipEvent_t event);
  hipError_t waitEvent(hipEvent_t event);
  hipError_t getDevResource(hipDevResource* resource, hipDevResourceType type);

  static hipError_t deviceGetDevResource(int device, hipDevResource* resource,
                                         hipDevResourceType type);
  static hipError_t devSmResourceSplitByCount(hipDevResource* result, uint32_t* nbGroups,
                                              const hipDevResource* input,
                                              hipDevResource* remainder,
                                              uint32_t flags, uint32_t minCount);
  static hipError_t devSmResourceSplit(hipDevResource* result, uint32_t nbGroups,
                                       const hipDevResource* input, hipDevResource* remainder,
                                       uint32_t flags,
                                       hipDevSmResourceGroupParams* groupParams);
  static hipError_t devResourceGenerateDesc(hipDevResourceDesc_t* phDesc,
                                            hipDevResource* resources, uint32_t nbResources);

  static ExecutionCtx* createPrimaryCtx(int device);

private:
  ExecutionCtx(const ExecutionCtx&) = delete;
  ExecutionCtx& operator=(const ExecutionCtx&) = delete;
  ExecutionCtx(ExecutionCtx&&) = delete;
  ExecutionCtx& operator=(ExecutionCtx&&) = delete;

  int deviceId_;
  uint32_t flags_;
  uint32_t cuCount_;
  uint64_t ctxId_;
  std::vector<uint32_t> cuMask_;
  DevResourceDesc* resourceDesc_;

  std::recursive_mutex lock_;
  std::shared_mutex streamSetLock_;
  std::unordered_set<hip::Stream*> streams_;

  inline static std::atomic<uint64_t> nextCtxId_{1};
  inline static std::atomic<uint32_t> nextResourceId_{1};
  inline static std::atomic<uint32_t> nextFamilyId_{1};

  static void tagResource(hipDevResource* res, uint32_t resourceId, int deviceId);
  static uint32_t readResourceId(const hipDevResource* res);
  static int readDeviceId(const hipDevResource* res);
  static void registerResourceMeta(int deviceId, uint32_t resourceId,
                                   uint32_t familyId, uint32_t startCU);
  static const ResourceMeta* lookupResourceMeta(int deviceId, uint32_t resourceId);

  static void fillSmResult(hipDevResource* res, uint32_t smCount,
                           uint32_t alignment, uint32_t flags);
  static void fillRemainder(hipDevResource* remainder, uint32_t remainingCUs,
                            uint32_t alignment);
  static std::vector<uint32_t> buildCuMask(uint32_t startCU, uint32_t count,
                                           uint32_t totalCUs);
};

} // namespace hip
