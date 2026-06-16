/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_executionctx.hpp"
#include "hip_event.hpp"
#include "hip_internal.hpp"
#include "utils/util.hpp"

#include <algorithm>
#include <cstring>

namespace hip {

// ---------------------------------------------------------------------------
// ExecutionCtx statics
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
ExecutionCtx::ExecutionCtx(int deviceId, DevResourceDesc* desc, uint32_t flags)
    : deviceId_(deviceId), flags_(flags), cuCount_(0), resourceDesc_(desc),
      ctxId_(nextCtxId_.fetch_add(1)) {}

hipError_t ExecutionCtx::Create() {
  hipDeviceProp_t prop{};
  hipError_t status = ihipGetDeviceProperties(&prop, deviceId_);
  if (status != hipSuccess) return status;

  uint32_t totalCUs = static_cast<uint32_t>(prop.multiProcessorCount);
  uint32_t maskWords = (totalCUs + 31) / 32;

  cuMask_.resize(maskWords, 0);
  for (const auto& res : resourceDesc_->resources) {
    if (res.type == hipDevResourceTypeSm) {
      cuCount_ += res.sm.smCount;
      uint32_t startCU = 0;
      uint32_t resId = readResourceId(&res);
      if (resId != 0) {
        const auto* meta = lookupResourceMeta(deviceId_, resId);
        if (meta != nullptr) startCU = meta->startCU;
      }
      auto partial = buildCuMask(startCU, res.sm.smCount, totalCUs);
      for (uint32_t w = 0; w < maskWords && w < partial.size(); w++)
        cuMask_[w] |= partial[w];
    }
  }
  return hipSuccess;
}

ExecutionCtx::~ExecutionCtx() {
  delete resourceDesc_;
  resourceDesc_ = nullptr;
  streams_.clear();
}

// ---------------------------------------------------------------------------
// Stream tracking
// ---------------------------------------------------------------------------
void ExecutionCtx::addStream(hip::Stream* stream) {
  std::unique_lock lk(streamSetLock_);
  streams_.insert(stream);
}
// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------
void ExecutionCtx::fillSmResult(hipDevResource* res, uint32_t smCount,
                             uint32_t alignment, uint32_t flags) {
  std::memset(res, 0, sizeof(hipDevResource));
  res->type = hipDevResourceTypeSm;
  res->sm.smCount = smCount;
  res->sm.smCoscheduledAlignment = alignment;
  res->sm.minSmPartitionSize = alignment;
  res->sm.flags = flags;
  res->nextResource = nullptr;
}

void ExecutionCtx::fillRemainder(hipDevResource* remainder, uint32_t remainingCUs,
                              uint32_t alignment) {
  if (remainder == nullptr) return;
  std::memset(remainder, 0, sizeof(hipDevResource));
  if (remainingCUs > 0) {
    remainder->type = hipDevResourceTypeSm;
    remainder->sm.smCount = remainingCUs;
    remainder->sm.smCoscheduledAlignment = alignment;
    remainder->sm.minSmPartitionSize = alignment;
    remainder->sm.flags = 0;
  } else {
    remainder->type = hipDevResourceTypeInvalid;
  }
  remainder->nextResource = nullptr;
}

std::vector<uint32_t> ExecutionCtx::buildCuMask(uint32_t startCU, uint32_t count,
                                             uint32_t totalCUs) {
  uint32_t maskWords = (totalCUs + 31) / 32;
  std::vector<uint32_t> mask(maskWords, 0);
  for (uint32_t i = startCU; i < startCU + count && i < totalCUs; i++) {
    mask[i / 32] |= (1u << (i % 32));
  }
  return mask;
}

void ExecutionCtx::tagResource(hipDevResource* res, uint32_t resourceId, int deviceId) {
  std::memcpy(&res->_internal_padding[0], &resourceId, sizeof(uint32_t));
  int32_t devId = static_cast<int32_t>(deviceId);
  std::memcpy(&res->_internal_padding[4], &devId, sizeof(int32_t));
}

uint32_t ExecutionCtx::readResourceId(const hipDevResource* res) {
  uint32_t id = 0;
  std::memcpy(&id, &res->_internal_padding[0], sizeof(uint32_t));
  return id;
}

int ExecutionCtx::readDeviceId(const hipDevResource* res) {
  int32_t devId = -1;
  std::memcpy(&devId, &res->_internal_padding[4], sizeof(int32_t));
  return static_cast<int>(devId);
}

void ExecutionCtx::registerResourceMeta(int deviceId, uint32_t resourceId,
                                    uint32_t familyId, uint32_t startCU) {
  g_devices[deviceId]->registerResource(resourceId, familyId, startCU);
}

const ResourceMeta* ExecutionCtx::lookupResourceMeta(int deviceId, uint32_t resourceId) {
  return g_devices[deviceId]->lookupResource(resourceId);
}

// ---------------------------------------------------------------------------
// Instance methods
// ---------------------------------------------------------------------------
hipError_t ExecutionCtx::getDevResource(hipDevResource* resource, hipDevResourceType type) {
  if (resourceDesc_ == nullptr)
    return hipErrorInvalidResourceConfiguration;

  uint32_t count = 0;
  for (const auto& res : resourceDesc_->resources) {
    if (res.type == type) {
      std::memcpy(&resource[count], &res, sizeof(hipDevResource));
      resource[count].nextResource = nullptr;
      if (count > 0) {
        resource[count - 1].nextResource = &resource[count];
      }
      count++;
    }
  }

  if (count == 0) return hipErrorInvalidResourceType;
  return hipSuccess;
}

hipError_t ExecutionCtx::synchronize() {
  std::vector<hip::Stream*> snapshot;
  {
    std::shared_lock lk(streamSetLock_);
    snapshot.reserve(streams_.size());
    for (auto* s : streams_) {
      s->retain();
      snapshot.push_back(s);
    }
  }

  for (auto* s : snapshot) {
    s->finish();
    s->release();
  }
  return hipSuccess;
}

hipError_t ExecutionCtx::recordEvent(hipEvent_t event) {
  std::vector<hip::Stream*> snapshot;
  {
    std::shared_lock lk(streamSetLock_);
    snapshot.reserve(streams_.size());
    for (auto* s : streams_) {
      s->retain();
      snapshot.push_back(s);
    }
  }

  if (snapshot.empty()) return hipSuccess;

  hip::Event* e = reinterpret_cast<hip::Event*>(event);

  // For single stream, use the existing addMarker path directly.
  if (snapshot.size() == 1) {
    hipError_t err = e->addMarker(snapshot[0], nullptr, true);
    snapshot[0]->release();
    return err;
  }
  
  std::scoped_lock lock(e->lock());

  amd::Command::EventWaitList waitList;

  MAKE_SCOPE_GUARD(cleanup, [&]() {
    for (auto* cmd : waitList) {
      cmd->release();
    }
    for (auto* s : snapshot) {
      s->release();
    }
  });

  // For multiple streams, enqueue independent markers on each stream.
  for (auto* s : snapshot) {
    amd::Command* cmd = nullptr;
    hipError_t err = e->recordCommand(cmd, s, 0, true);
    if (err != hipSuccess) {
      return err;
    }
    cmd->enqueue();
    waitList.push_back(&cmd->event());
  }
  
  // Create a single Marker to wait on all events in waitlist
  // Cache Flush not required for this fan-in marker
  amd::Command* fanIn = new amd::Marker(*snapshot[0], kMarkerDisableFlush, waitList);
  fanIn->setCommandEntryScope(amd::Device::kCacheStateIgnore);
  return e->enqueueRecordCommand(snapshot[0], fanIn);
}

hipError_t ExecutionCtx::waitEvent(hipEvent_t event) {
  hip::Event* e = reinterpret_cast<hip::Event*>(event);

  std::vector<hip::Stream*> snapshot;
  {
    std::shared_lock lk(streamSetLock_);
    snapshot.reserve(streams_.size());
    for (auto* s : streams_) {
      s->retain();
      snapshot.push_back(s);
    }
  }

  hipError_t result = hipSuccess;
  for (auto* s : snapshot) {
    hipError_t err = e->streamWait(s, 0);
    if (err != hipSuccess) result = err;
    s->release();
  }
  return result;
}

// ---------------------------------------------------------------------------
// Static: Resource Management APIs
// ---------------------------------------------------------------------------
hipError_t ExecutionCtx::deviceGetDevResource(int device, hipDevResource* resource,
                                           hipDevResourceType type) {
  if (resource == nullptr) return hipErrorInvalidValue;
  
  if (type != hipDevResourceTypeSm) {
    return hipErrorInvalidResourceType;
  }
  int count = 0;
  HIP_RETURN_ONFAIL(ihipDeviceGetCount(&count));
  if (device < 0 || device >= count) return hipErrorInvalidDevice;

  std::memset(resource, 0, sizeof(hipDevResource));
  resource->type = type;

  hipDeviceProp_t prop{};
  constexpr uint32_t kAlignment = 2;
  HIP_RETURN_ONFAIL(ihipGetDeviceProperties(&prop, device));
  uint32_t cuCount = static_cast<uint32_t>(prop.multiProcessorCount);

  resource->sm.smCount = cuCount;
  resource->sm.minSmPartitionSize = kAlignment;
  resource->sm.smCoscheduledAlignment =  kAlignment;
  resource->sm.flags = 0;

  resource->nextResource = nullptr;
  tagResource(resource, 0, device);

  return hipSuccess;
}

hipError_t ExecutionCtx::devSmResourceSplitByCount(
    hipDevResource* result, uint32_t* nbGroups,
    const hipDevResource* input, hipDevResource* remainder,
    uint32_t flags, uint32_t minCount) {

  if (nbGroups == nullptr || input == nullptr) return hipErrorInvalidValue;
  if (input->type != hipDevResourceTypeSm) return hipErrorInvalidResourceType;

  uint32_t totalCUs = input->sm.smCount;
  uint32_t alignment = input->sm.smCoscheduledAlignment;

  if (flags & hipDevSmResourceSplitIgnoreSmCoscheduling)
    alignment = 1;

  uint32_t alignedMin = ((minCount + alignment - 1) / alignment) * alignment;
  if (alignedMin == 0) alignedMin = alignment;

  uint32_t possibleGroups = totalCUs / alignedMin;

  if (result == nullptr) {
    *nbGroups = possibleGroups;
    return hipSuccess;
  }

  int inputDevId = readDeviceId(input);
  if (inputDevId < 0) return hipErrorInvalidDevice;

  uint32_t inputStartCU = 0;
  const ResourceMeta* inputMeta = lookupResourceMeta(inputDevId, readResourceId(input));
  if (inputMeta != nullptr) inputStartCU = inputMeta->startCU;
  uint32_t familyId = nextFamilyId_.fetch_add(1);

  uint32_t actualGroups = std::min(*nbGroups, possibleGroups);
  *nbGroups = actualGroups;

  uint32_t assignedCUs = 0;
  for (uint32_t i = 0; i < actualGroups; i++) {
    fillSmResult(&result[i], alignedMin, alignment, 0);
    uint32_t resId = nextResourceId_.fetch_add(1);
    tagResource(&result[i], resId, inputDevId);
    registerResourceMeta(inputDevId, resId, familyId, inputStartCU + assignedCUs);
    assignedCUs += alignedMin;
  }

  fillRemainder(remainder, totalCUs - assignedCUs, alignment);
  if (remainder != nullptr && remainder->type != hipDevResourceTypeInvalid) {
    uint32_t remId = nextResourceId_.fetch_add(1);
    tagResource(remainder, remId, inputDevId);
    registerResourceMeta(inputDevId, remId, familyId, inputStartCU + assignedCUs);
  }
  return hipSuccess;
}

hipError_t ExecutionCtx::devSmResourceSplit(
    hipDevResource* result, uint32_t nbGroups,
    const hipDevResource* input, hipDevResource* remainder,
    uint32_t flags, hipDevSmResourceGroupParams* groupParams) {

  if (input == nullptr || groupParams == nullptr) return hipErrorInvalidValue;
  if (input->type != hipDevResourceTypeSm) return hipErrorInvalidResourceType;
  if (flags != 0) return hipErrorInvalidValue;

  uint32_t totalCUs = input->sm.smCount;
  uint32_t defaultAlignment = input->sm.smCoscheduledAlignment;

  for (uint32_t i = 0; i < nbGroups; i++) {
    if (groupParams[i].coscheduledSmCount == 0)
      groupParams[i].coscheduledSmCount = defaultAlignment;
    if (groupParams[i].preferredCoscheduledSmCount == 0)
      groupParams[i].preferredCoscheduledSmCount = groupParams[i].coscheduledSmCount;

    if (groupParams[i].smCount != 0) {
      if (groupParams[i].smCount < 2 || groupParams[i].smCount > totalCUs)
        return hipErrorInvalidResourceConfiguration;
    }
  }

  uint32_t assignedCUs = 0;
  for (uint32_t i = 0; i < nbGroups; i++) {
    uint32_t cosched = groupParams[i].coscheduledSmCount;

    if (groupParams[i].smCount == 0) {
      uint32_t available = totalCUs - assignedCUs;
      if (groupParams[i].flags & hipDevSmResourceGroupBackfill) {
        groupParams[i].smCount = available;
      } else {
        groupParams[i].smCount = (available / cosched) * cosched;
      }
    }
    assignedCUs += groupParams[i].smCount;
  }

  if (assignedCUs > totalCUs) return hipErrorInvalidResourceConfiguration;

  int inputDevId = readDeviceId(input);
  if (inputDevId < 0) return hipErrorInvalidDevice;
  
  uint32_t inputStartCU = 0;
  const ResourceMeta* inputMeta = lookupResourceMeta(inputDevId, readResourceId(input));
  if (inputMeta != nullptr) inputStartCU = inputMeta->startCU;
  uint32_t familyId = nextFamilyId_.fetch_add(1);

  if (result != nullptr) {
    uint32_t cuOffset = 0;
    for (uint32_t i = 0; i < nbGroups; i++) {
      uint32_t cosched = groupParams[i].coscheduledSmCount;
      uint32_t count = groupParams[i].smCount;

      if (count == 0 || count < cosched)
        return hipErrorInvalidResourceConfiguration;
      if ((count % cosched != 0) &&
          !(groupParams[i].flags & hipDevSmResourceGroupBackfill))
        return hipErrorInvalidResourceConfiguration;

      fillSmResult(&result[i], count, cosched, groupParams[i].flags);
      uint32_t resId = nextResourceId_.fetch_add(1);
      tagResource(&result[i], resId, inputDevId);
      registerResourceMeta(inputDevId, resId, familyId, inputStartCU + cuOffset);
      cuOffset += count;
    }
  }

  fillRemainder(remainder, totalCUs - assignedCUs, defaultAlignment);
  if (remainder != nullptr && remainder->type != hipDevResourceTypeInvalid) {
    uint32_t remId = nextResourceId_.fetch_add(1);
    tagResource(remainder, remId, inputDevId);
    registerResourceMeta(inputDevId, remId, familyId, inputStartCU + assignedCUs);
  }
  return hipSuccess;
}

hipError_t ExecutionCtx::devResourceGenerateDesc(hipDevResourceDesc_t* phDesc,
                                              hipDevResource* resources,
                                              uint32_t nbResources) {
  if (phDesc == nullptr || resources == nullptr || nbResources == 0)
    return hipErrorInvalidValue;

  for (uint32_t i = 0; i < nbResources; i++) {
    if (resources[i].type != hipDevResourceTypeSm) {
      return hipErrorInvalidResourceType;
    }
  }

  uint32_t refSmAlignment = 0;
  uint32_t refFamilyId = 0;
  bool familyChecked = false;
  int refDeviceId = -1;
  for (uint32_t i = 0; i < nbResources; i++) {
    // Validate SM Resource Alignment
    if (resources[i].sm.smCount == 0)
      return hipErrorInvalidResourceConfiguration;
    if (refSmAlignment == 0) {
      refSmAlignment = resources[i].sm.smCoscheduledAlignment;
    } else if (resources[i].sm.smCoscheduledAlignment != refSmAlignment) {
      return hipErrorInvalidResourceConfiguration;
    }
    // Validate SM Resource Device ID
    int devId = readDeviceId(&resources[i]);
    if (refDeviceId == -1) {
      refDeviceId = devId;
    } else if (devId != refDeviceId ){
      return hipErrorInvalidResourceConfiguration;
    }
    uint32_t resId = readResourceId(&resources[i]);
    if (resId != 0) {
      const auto* meta = lookupResourceMeta(devId, resId);
      if (meta != nullptr) {
        if (!familyChecked) {
          refFamilyId = meta->familyId;
          familyChecked = true;
        } else if (meta->familyId != refFamilyId) {
          return hipErrorInvalidResourceConfiguration;
        }
      }
    }
  }

  auto* desc = new DevResourceDesc();
  desc->resources.assign(resources, resources + nbResources);
  desc->deviceId = refDeviceId;

  *phDesc = reinterpret_cast<hipDevResourceDesc_t>(desc);
  return hipSuccess;
}


// ---------------------------------------------------------------------------
// Static: Primary execution context factory
// ---------------------------------------------------------------------------
ExecutionCtx* ExecutionCtx::createPrimaryCtx(int device) {
  hipDevResource devRes{};
  hipError_t status = deviceGetDevResource(device, &devRes, hipDevResourceTypeSm);
  if (status != hipSuccess) return nullptr;

  auto* desc = new DevResourceDesc();
  desc->resources.push_back(devRes);
  desc->deviceId = device;

  auto* ctx = new ExecutionCtx(device, desc, 0);
  if (ctx->Create() != hipSuccess) {
    delete ctx;
    return nullptr;
  }
  return ctx;
}

} // namespace hip

// ===========================================================================
// HIP Runtime API entry points (in hip namespace for dispatch table)
// ===========================================================================
namespace hip {

// ---------------------------------------------------------------------------
// Resource Management
// ---------------------------------------------------------------------------
hipError_t hipDeviceGetDevResource(hipDevice_t device, hipDevResource* resource,
                                   hipDevResourceType type) {
  HIP_INIT_API(hipDeviceGetDevResource, device, resource, type);
  HIP_RETURN(ExecutionCtx::deviceGetDevResource(device, resource, type));
}

hipError_t hipDevSmResourceSplitByCount(hipDevResource* result, unsigned int* nbGroups,
                                        const hipDevResource* input, hipDevResource* remainder,
                                        unsigned int flags, unsigned int minCount) {
  HIP_INIT_API(hipDevSmResourceSplitByCount, result, nbGroups, input, remainder, flags, minCount);
  HIP_RETURN(ExecutionCtx::devSmResourceSplitByCount(result, nbGroups, input, remainder, flags,
                                                  minCount));
}

hipError_t hipDevSmResourceSplit(hipDevResource* result, unsigned int nbGroups,
                                 const hipDevResource* input, hipDevResource* remainder,
                                 unsigned int flags,
                                 hipDevSmResourceGroupParams* groupParams) {
  HIP_INIT_API(hipDevSmResourceSplit, result, nbGroups, input, remainder, flags, groupParams);
  HIP_RETURN(ExecutionCtx::devSmResourceSplit(result, nbGroups, input, remainder, flags, groupParams));
}

hipError_t hipDevResourceGenerateDesc(hipDevResourceDesc_t* phDesc, hipDevResource* resources,
                                       unsigned int nbResources) {
  HIP_INIT_API(hipDevResourceGenerateDesc, phDesc, resources, nbResources);
  HIP_RETURN(ExecutionCtx::devResourceGenerateDesc(phDesc, resources, nbResources));
}

// ---------------------------------------------------------------------------
// Context Lifecycle
// ---------------------------------------------------------------------------
hipError_t hipGreenCtxCreate(hipExecutionCtx_t* phCtx, hipDevResourceDesc_t desc,
                              int device, unsigned int flags) {
  HIP_INIT_API(hipGreenCtxCreate, phCtx, desc, device, flags);

  if (phCtx == nullptr || desc == nullptr || flags != 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  int count = 0;
  HIP_RETURN_ONFAIL(ihipDeviceGetCount(&count));
  if (device < 0 || device >= count) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  auto* resourceDesc = reinterpret_cast<DevResourceDesc*>(desc);

  if (resourceDesc->deviceId < 0) {
    HIP_RETURN(hipErrorInvalidDevice);
  }
  
  if (resourceDesc->deviceId != device) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (resourceDesc->resources.empty()) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  g_devices[device]->retain();

  auto* greenCtx = new ExecutionCtx(device, resourceDesc, flags);
  hipError_t err = greenCtx->Create();
  if (err != hipSuccess) {
    delete greenCtx;
    g_devices[device]->release();
    HIP_RETURN(err);
  }
  *phCtx = reinterpret_cast<hipExecutionCtx_t>(greenCtx);
  HIP_RETURN(hipSuccess);
}

hipError_t hipExecutionCtxDestroy(hipExecutionCtx_t ctx) {
  HIP_INIT_API(hipExecutionCtxDestroy, ctx);

  if (ctx == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto* greenCtx = reinterpret_cast<ExecutionCtx*>(ctx);
  int deviceId = greenCtx->deviceId();

  {
    std::scoped_lock lock(g_devices[deviceId]->getLock());
    if (greenCtx == g_devices[deviceId]->getPrimaryExecCtx()) {
      // The primary execution context is managed by the Device and must not
      // be destroyed via hipGreenCtxDestroy
      HIP_RETURN(hipErrorInvalidValue);
    }
  }

  delete greenCtx;

  g_devices[deviceId]->release();

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceGetExecutionCtx(hipExecutionCtx_t* ctx, int device) {
  HIP_INIT_API(hipDeviceGetExecutionCtx, ctx, device);

  if (ctx == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  int count = 0;
  HIP_RETURN_ONFAIL(ihipDeviceGetCount(&count));
  if (device < 0 || device >= count) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  std::scoped_lock lock(g_devices[device]->getLock());
  ExecutionCtx* primaryCtx = g_devices[device]->getPrimaryExecCtx();
  if (primaryCtx == nullptr) {
    primaryCtx = ExecutionCtx::createPrimaryCtx(device);
    if (primaryCtx == nullptr) {
      HIP_RETURN(hipErrorOutOfMemory);
    }
    g_devices[device]->setPrimaryExecCtx(primaryCtx);
  }

  *ctx = reinterpret_cast<hipExecutionCtx_t>(primaryCtx);
  HIP_RETURN(hipSuccess);
}

// ---------------------------------------------------------------------------
// Stream and Context Operations
// ---------------------------------------------------------------------------
hipError_t hipExecutionCtxStreamCreate(hipStream_t* stream, hipExecutionCtx_t greenctx,
                                        unsigned int flags, int priority) {
  HIP_INIT_API(hipExecutionCtxStreamCreate, stream, greenctx, flags, priority);

  if (stream == nullptr || greenctx == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (flags != hipStreamDefault && flags != hipStreamNonBlocking) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto* ctx = reinterpret_cast<ExecutionCtx*>(greenctx);

  Stream::Priority streamPriority;
  if (priority <= Stream::Priority::High)
    streamPriority = Stream::Priority::High;
  else if (priority >= Stream::Priority::Low)
    streamPriority = Stream::Priority::Low;
  else
    streamPriority = static_cast<Stream::Priority>(priority);

  const auto& cuMask = ctx->cuMask();
  Stream* hStream = new Stream(ctx->device(), streamPriority, flags, false, cuMask);
  if (hStream->vdev() == nullptr) {
    Stream::Destroy(hStream);
    HIP_RETURN(hipErrorOutOfMemory);
  }

  ctx->addStream(hStream);
  *stream = reinterpret_cast<hipStream_t>(hStream);
  HIP_RETURN(hipSuccess);
}

hipError_t hipExecutionCtxGetDevResource(hipExecutionCtx_t ctx, hipDevResource* resource,
                                          hipDevResourceType type) {
  HIP_INIT_API(hipExecutionCtxGetDevResource, ctx, resource, type);

  if (ctx == nullptr || resource == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto* greenCtx = reinterpret_cast<ExecutionCtx*>(ctx);
  std::scoped_lock lock(greenCtx->lock());
  HIP_RETURN(greenCtx->getDevResource(resource, type));
}

hipError_t hipExecutionCtxGetDevice(int* device, hipExecutionCtx_t ctx) {
  HIP_INIT_API(hipExecutionCtxGetDevice, device, ctx);
  if (device == nullptr || ctx == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *device = reinterpret_cast<ExecutionCtx*>(ctx)->deviceId();
  HIP_RETURN(hipSuccess);
}

hipError_t hipExecutionCtxGetId(hipExecutionCtx_t ctx, unsigned long long* ctxId) {
  HIP_INIT_API(hipExecutionCtxGetId, ctx, ctxId);
  if (ctx == nullptr || ctxId == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *ctxId = reinterpret_cast<ExecutionCtx*>(ctx)->ctxId();
  HIP_RETURN(hipSuccess);
}

hipError_t hipStreamGetDevResource(hipStream_t hStream, hipDevResource* resource,
                                    hipDevResourceType type) {
  HIP_INIT_API(hipStreamGetDevResource, hStream, resource, type);

  if (resource == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (type == hipDevResourceTypeWorkqueueConfig ||
      type == hipDevResourceTypeWorkqueue) {
    HIP_RETURN(hipErrorInvalidResourceType);
  }
  if (!isValid(hStream)) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  std::memset(resource, 0, sizeof(hipDevResource));

  if (hStream == nullptr) {
    int device = getCurrentDevice()->deviceId();
    HIP_RETURN(ExecutionCtx::deviceGetDevResource(device, resource, type));
  }

  auto* stream = reinterpret_cast<Stream*>(hStream);

  switch (type) {
    case hipDevResourceTypeSm: {
      resource->type = hipDevResourceTypeSm;
      const auto& cuMask = stream->GetCUMask();
      if (cuMask.empty()) {
        hipDeviceProp_t prop{};
        HIP_RETURN_ONFAIL(ihipGetDeviceProperties(&prop, stream->DeviceId()));
        resource->sm.smCount = prop.multiProcessorCount;
      } else {
        uint32_t cnt = 0;
        for (auto word : cuMask) cnt += amd::countBitsSet32(word);
        resource->sm.smCount = cnt;
      }
      resource->sm.smCoscheduledAlignment = 2;
      resource->sm.minSmPartitionSize = 2;
      resource->sm.flags = 0;
      resource->nextResource = nullptr;
      break;
    }
    default:
      HIP_RETURN(hipErrorInvalidResourceType);
  }
  HIP_RETURN(hipSuccess);
}

// ---------------------------------------------------------------------------
// Sync / Event Operations
// ---------------------------------------------------------------------------
hipError_t hipExecutionCtxRecordEvent(hipExecutionCtx_t ctx, hipEvent_t event) {
  HIP_INIT_API(hipExecutionCtxRecordEvent, ctx, event);
  if (ctx == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (!isValid(event)) {
    HIP_RETURN(hipErrorInvalidHandle);
  }
  HIP_RETURN(reinterpret_cast<ExecutionCtx*>(ctx)->recordEvent(event));
}

hipError_t hipExecutionCtxSynchronize(hipExecutionCtx_t ctx) {
  HIP_INIT_API(hipExecutionCtxSynchronize, ctx);
  if (ctx == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<ExecutionCtx*>(ctx)->synchronize());
}

hipError_t hipExecutionCtxWaitEvent(hipExecutionCtx_t ctx, hipEvent_t event) {
  HIP_INIT_API(hipExecutionCtxWaitEvent, ctx, event);
  if (ctx == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (!isValid(event)) {
    HIP_RETURN(hipErrorInvalidHandle);
  }
  HIP_RETURN(reinterpret_cast<ExecutionCtx*>(ctx)->waitEvent(event));
}

} // namespace hip
