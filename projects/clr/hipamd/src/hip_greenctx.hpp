/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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

class GreenCtx {
public:
  GreenCtx(int deviceId, DevResourceDesc* desc, unsigned int flags);
  ~GreenCtx();

  hipError_t Create();

  int deviceId() const { return deviceId_; }
  hip::Device* device() const { return g_devices[deviceId_]; }
  const std::vector<uint32_t>& cuMask() const { return cuMask_; }
  unsigned int cuCount() const { return cuCount_; }
  unsigned int flags() const { return flags_; }
  unsigned long long ctxId() const { return ctxId_; }
  std::recursive_mutex& lock() { return lock_; }

  void addStream(hip::Stream* stream);
  void removeStream(hip::Stream* stream);

  hipError_t synchronize();
  hipError_t recordEvent(hipEvent_t event);
  hipError_t waitEvent(hipEvent_t event);
  hipError_t getDevResource(hipDevResource* resource, hipDevResourceType type);

  static hipError_t deviceGetDevResource(int device, hipDevResource* resource,
                                         hipDevResourceType type);
  static hipError_t devSmResourceSplitByCount(hipDevResource* result, unsigned int* nbGroups,
                                              const hipDevResource* input,
                                              hipDevResource* remainder,
                                              unsigned int flags, unsigned int minCount);
  static hipError_t devSmResourceSplit(hipDevResource* result, unsigned int nbGroups,
                                       const hipDevResource* input, hipDevResource* remainder,
                                       unsigned int flags,
                                       hipDevSmResourceGroupParams* groupParams);
  static hipError_t devResourceGenerateDesc(hipDevResourceDesc_t* phDesc,
                                            hipDevResource* resources, unsigned int nbResources);

  static GreenCtx* createPrimaryCtx(int device);

private:
  GreenCtx(const GreenCtx&) = delete;
  GreenCtx& operator=(const GreenCtx&) = delete;
  GreenCtx(GreenCtx&&) = delete;
  GreenCtx& operator=(GreenCtx&&) = delete;

  int deviceId_;
  unsigned int flags_;
  unsigned int cuCount_;
  unsigned long long ctxId_;
  std::vector<uint32_t> cuMask_;
  DevResourceDesc* resourceDesc_;

  std::recursive_mutex lock_;
  std::shared_mutex streamSetLock_;
  std::unordered_set<hip::Stream*> streams_;

  inline static std::atomic<unsigned long long> nextCtxId_{1};
  inline static std::atomic<uint32_t> nextResourceId_{1};
  inline static std::atomic<uint32_t> nextFamilyId_{1};

  static void tagResource(hipDevResource* res, uint32_t resourceId, int deviceId);
  static uint32_t readResourceId(const hipDevResource* res);
  static int readDeviceId(const hipDevResource* res);
  static void registerResourceMeta(int deviceId, uint32_t resourceId,
                                   uint32_t familyId, unsigned int startCU);
  static const ResourceMeta* lookupResourceMeta(int deviceId, uint32_t resourceId);

  static void fillSmResult(hipDevResource* res, unsigned int smCount,
                           unsigned int alignment, unsigned int flags);
  static void fillRemainder(hipDevResource* remainder, unsigned int remainingCUs,
                            unsigned int alignment);
  static std::vector<uint32_t> buildCuMask(unsigned int startCU, unsigned int count,
                                           unsigned int totalCUs);
};

} // namespace hip
