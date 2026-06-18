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

/// Manages an isolated execution context with a subset of device compute resources.
class ExecutionCtx {
public:
  /// Construct a context for the given device using the provided resource descriptor.
  ExecutionCtx(int deviceId, DevResourceDesc* desc, uint32_t flags);
  ~ExecutionCtx();

  /// Build the CU mask from the resource descriptor; must be called after construction.
  hipError_t Create();

  /// Device ordinal this context is bound to.
  int deviceId() const { return deviceId_; }
  /// Underlying hip::Device pointer.
  hip::Device* device() const { return g_devices[deviceId_]; }
  /// Bitmask of compute units assigned to this context (one bit per WGP in WGP mode, one bit per CU otherwise).
  const std::vector<uint32_t>& cuMask() const { return cuMask_; }
  /// Number of compute units available in this context.
  uint32_t cuCount() const { return cuCount_; }
  /// Creation flags passed to hipGreenCtxCreate.
  uint32_t flags() const { return flags_; }
  /// Unique monotonic identifier for this context.
  uint64_t ctxId() const { return ctxId_; }
  /// Per-context mutex for callers that need exclusive access.
  std::recursive_mutex& lock() { return lock_; }

  /// Register a stream created under this context.
  void addStream(hip::Stream* stream);

  /// Block until all streams in this context have completed.
  hipError_t synchronize();
  /// Record an event across all streams in this context.
  hipError_t recordEvent(hipEvent_t event);
  /// Make all streams in this context wait on an event.
  hipError_t waitEvent(hipEvent_t event);
  /// Copy the matching device resource from the resource descriptor into the output.
  hipError_t getDevResource(hipDevResource* resource, hipDevResourceType type);

  /// Return SM co-scheduling alignment for the given device (2 in WGP mode, 1 otherwise).
  static uint32_t getSmAlignment(int deviceId);

  /// Return total CU count for device
  static uint32_t getTotalCuCount(int deviceId);

  /// Query the full device resource for a device (all CUs).
  static hipError_t deviceGetDevResource(int device, hipDevResource* resource,
                                         hipDevResourceType type);
  /// Split an SM resource into equal groups of at least minCount CUs each.
  static hipError_t devSmResourceSplitByCount(hipDevResource* result, uint32_t* nbGroups,
                                              const hipDevResource* input,
                                              hipDevResource* remainder,
                                              uint32_t flags, uint32_t minCount);
  /// Split an SM resource into groups defined by the provided group parameters.
  static hipError_t devSmResourceSplit(hipDevResource* result, uint32_t nbGroups,
                                       const hipDevResource* input, hipDevResource* remainder,
                                       uint32_t flags,
                                       hipDevSmResourceGroupParams* groupParams);
  /// Validate resources and create a resource descriptor suitable for context creation.
  static hipError_t devResourceGenerateDesc(hipDevResourceDesc_t* phDesc,
                                            hipDevResource* resources, uint32_t nbResources);

  /// Create the primary (whole-device) execution context for the given device.
  static ExecutionCtx* createPrimaryCtx(int device);

private:
  ExecutionCtx(const ExecutionCtx&) = delete;
  ExecutionCtx& operator=(const ExecutionCtx&) = delete;
  ExecutionCtx(ExecutionCtx&&) = delete;
  ExecutionCtx& operator=(ExecutionCtx&&) = delete;

  int deviceId_;                              ///< Device ordinal
  uint32_t flags_;                            ///< Context creation flags
  uint32_t cuCount_;                          ///< Number of CUs assigned to this context
  uint64_t ctxId_;                            ///< Unique context identifier
  std::vector<uint32_t> cuMask_;              ///< Bitmask of assigned WGPs (one bit per WGP)
  DevResourceDesc* resourceDesc_;             ///< Owning pointer to the resource descriptor

  std::recursive_mutex lock_;                 ///< Context-level lock
  std::shared_mutex streamSetLock_;           ///< Guards streams_ for concurrent reads
  std::unordered_set<hip::Stream*> streams_;  ///< Streams created under this context

  inline static std::atomic<uint64_t> nextCtxId_{1};       ///< Monotonic context ID generator
  inline static std::atomic<uint32_t> nextResourceId_{1};  ///< Monotonic resource ID generator
  inline static std::atomic<uint32_t> nextFamilyId_{1};    ///< Monotonic family ID generator

  /// Embed resource ID and device ID into a hipDevResource's internal padding.
  static void tagResource(hipDevResource* res, uint32_t resourceId, int deviceId);
  /// Read the resource ID previously stored by tagResource.
  static uint32_t readResourceId(const hipDevResource* res);
  /// Read the device ID previously stored by tagResource.
  static int readDeviceId(const hipDevResource* res);
  /// Register CU-range metadata for a split resource on the given device.
  static void registerResourceMeta(int deviceId, uint32_t resourceId,
                                   uint32_t familyId, uint32_t startCU);
  /// Look up CU-range metadata for a resource on the given device.
  static const ResourceMeta* lookupResourceMeta(int deviceId, uint32_t resourceId);

  /// Populate a hipDevResource as an SM-type result with the given counts and flags.
  static void fillSmResult(hipDevResource* res, uint32_t smCount,
                           uint32_t alignment, uint32_t flags);
  /// Populate a remainder resource with leftover CUs, or mark it invalid if none remain.
  static void fillRemainder(hipDevResource* remainder, uint32_t remainingCUs,
                            uint32_t alignment);
  /// Build a WGP bitmask covering CUs [startCU, startCU+count). Each bit represents
  /// one WGP (alignment CUs). startCU and count are in CU units.
  static std::vector<uint32_t> buildCuMask(uint32_t startCU, uint32_t count,
                                           uint32_t totalCUs, uint32_t alignment = 1);
};

} // namespace hip
