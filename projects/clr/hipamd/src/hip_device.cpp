/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <hip/hip_deprecated.h>
#include <hip/amd_detail/hip_storage.h>

#include "hip_internal.hpp"
#include "hip_mempool_impl.hpp"
#include "hip_platform.hpp"

#undef hipGetDeviceProperties
#undef hipDeviceProp_t

namespace hip {

// ================================================================================================
hip::Stream* Device::NullStream(bool wait) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_WAIT, "NullStream %p, wait %d", null_stream_, wait);
  if (null_stream_ == nullptr) {
    std::scoped_lock lock(lock_);
    if (null_stream_ == nullptr) {
      null_stream_ = new Stream(this, Stream::Priority::Normal, 0, true);
      // Stream creation may fail in ROCclr, in which case vdev is null.
      if (null_stream_->vdev() == nullptr) {
        Stream::Destroy(null_stream_);
        null_stream_ = nullptr;
      }
    }
  }
  if (null_stream_ == nullptr) {
    LogError("Cannot create new Stream object");
    return nullptr;
  }
  if (wait) {
    // Wait for all active streams before executing commands on the default stream
    WaitActiveStreams(null_stream_);
  }
  return null_stream_;
}

// ================================================================================================
bool Device::Create() {
  // Create default memory pool
  default_mem_pool_ = new MemoryPool(this);
  if (default_mem_pool_ == nullptr) {
    return false;
  }

  // Create graph memory pool
  graph_mem_pool_ = new MemoryPool(this, nullptr, true);
  if (graph_mem_pool_ == nullptr) {
    return false;
  }

  if (!HIP_MEM_POOL_USE_VM) {
    // Without VM support, set max threshold so the pool never releases memory automatically.
    uint64_t max_size = std::numeric_limits<uint64_t>::max();
    [[maybe_unused]] const auto err =
        graph_mem_pool_->SetAttribute(hipMemPoolAttrReleaseThreshold, &max_size);
    assert(err == hipSuccess);
  }

  current_mem_pool_ = default_mem_pool_;

  // Create managed memory pool
  const hipMemPoolProps props = {.allocType = hipMemAllocationTypeManaged,
                                 .handleTypes = hipMemHandleTypeNone,
                                 .location = {.type = hipMemLocationTypeDevice,
                                              .id = deviceId_}};
  default_managed_mem_pool_ = new MemoryPool(this, &props);
  if (default_managed_mem_pool_ == nullptr) {
    return false;
  }
  current_managed_mem_pool_ = default_managed_mem_pool_;

  return true;
}

// ================================================================================================
bool Device::IsMemoryPoolValid(MemoryPool* pool) {
  std::scoped_lock lock(lock_);
  return mem_pools_.count(pool) != 0;
}

// ================================================================================================
void Device::AddMemoryPool(MemoryPool* pool) {
  std::scoped_lock lock(lock_);
  mem_pools_.insert(pool);
}

// ================================================================================================
void Device::RemoveMemoryPool(MemoryPool* pool) {
  std::scoped_lock lock(lock_);
  mem_pools_.erase(pool);
}

// ================================================================================================
bool Device::FreeMemory(amd::Memory* memory, Stream* stream, Event* event) {
  std::scoped_lock lock(lock_);
  for (auto* pool : mem_pools_) {
    if (pool->FreeMemory(memory, stream, event)) {
      return true;
    }
  }
  return false;
}

// ================================================================================================
void Device::ReleaseFreedMemory() {
  std::scoped_lock lock(lock_);
  for (auto* pool : mem_pools_) {
    pool->ReleaseFreedMemory();
  }
}

// ================================================================================================
void Device::RemoveStreamFromPools(Stream* stream) {
  std::scoped_lock lock(lock_);
  for (auto* pool : mem_pools_) {
    pool->RemoveStream(stream);
  }
}

// ================================================================================================
void Device::AddSafeStream(Stream* event_stream, Stream* wait_stream) {
  std::scoped_lock lock(lock_);
  for (auto* pool : mem_pools_) {
    pool->AddSafeStream(event_stream, wait_stream);
  }
}

// ================================================================================================
void Device::Reset() {
  {
    std::scoped_lock lock(lock_);
    auto pools_to_delete = std::exchange(mem_pools_, {});
    for (auto* pool : pools_to_delete) {
      pool->ReleaseAllMemory();
      delete pool;
    }
  }
  flags_ = hipDeviceScheduleSpin;
  destroyAllStreams();

  // Clear hostcall allocations to avoid ~Device() accessing freed Memory objects later.
  auto* dev = devices()[0];
  dev->ClearHostcallMemories();
  amd::MemObjMap::Purge(dev);
  Create();
}

// ================================================================================================
void Device::WaitActiveStreams(hip::Stream* blocking_stream, bool wait_null_stream) {
  amd::Command::EventWaitList eventWaitList(0);
  bool submitMarker = false;
  std::vector<amd::CommandQueue*> activeQueues;

  auto waitForStream = [&submitMarker, &eventWaitList](hip::Stream* stream) {
    if (amd::Command* command = stream->getLastQueuedCommand(true)) {
      amd::Event& event = command->event();
      // Check HW status of the ROCclr event.
      // Note: not all ROCclr modes support HW status
      bool ready = stream->device().IsHwEventReady(event);
      if (!ready) {
        ready = (command->status() == CL_COMPLETE);
      }
      submitMarker |= stream->vdev()->isFenceDirty();
      if (!ready) {
        command->notifyCmdQueue();
        eventWaitList.push_back(command);
      } else {
        command->release();
      }
    }
  };

  if (wait_null_stream) {
    if (null_stream_) {
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_WAIT, "Waiting on nullstream %p", null_stream_);
      waitForStream(null_stream_);
    }
  } else {
    activeQueues = blocking_stream->device().getActiveQueues();
    for (const auto& queue : activeQueues) {
      auto* active_stream = static_cast<hip::Stream*>(queue);
      // Only wait on blocking (non-nonblocking) streams other than the current one
      if ((active_stream->Flags() & hipStreamNonBlocking) == 0 &&
          active_stream != blocking_stream) {
        ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_WAIT, "Waiting on active stream %p", active_stream);
        waitForStream(active_stream);
      }
    }
  }

  if (!eventWaitList.empty() || submitMarker) {
    auto* marker = new amd::Marker(*blocking_stream, kMarkerDisableFlush, eventWaitList);
    marker->enqueue();
    marker->release();
  }

  // Release all active commands; safe after the marker was enqueued
  for (const auto& cmd : eventWaitList) {
    cmd->release();
  }

  // Release active queue references now that the marker has been fully enqueued
  // and no longer needs to access the queues via eventWaitList commands
  for (const auto& q : activeQueues) {
    q->release();
  }
}

// ================================================================================================
void Device::AddStream(Stream* stream) {
  std::unique_lock lock(streamSetLock_);
  streamSet_.insert(stream);
}

// ================================================================================================
void Device::RemoveStream(Stream* stream) {
  std::unique_lock lock(streamSetLock_);
  streamSet_.erase(stream);
}

// ================================================================================================
bool Device::StreamExists(const Stream* stream) {
  std::shared_lock lock(streamSetLock_);
  return streamSet_.count(const_cast<Stream*>(stream)) != 0;
}

// ================================================================================================
void Device::destroyAllStreams() {
  std::vector<Stream*> toBeDeleted;
  {
    std::shared_lock lock(streamSetLock_);
    toBeDeleted.reserve(streamSet_.size());
    for (auto* stream : streamSet_) {
      if (!stream->Null()) {
        toBeDeleted.push_back(stream);
      }
    }
  }
  for (auto* stream : toBeDeleted) {
    hip::Stream::Destroy(stream);
  }
  hip::tls.stream_per_thread_obj_.Clear();
}

// ================================================================================================
void Device::SyncAllStreams(bool cpu_wait, bool wait_blocking_streams_only) {
  // Make a local copy to avoid stalls for GPU finish with multiple threads
  std::vector<hip::Stream*> streams;
  {
    std::shared_lock lock(streamSetLock_);
    streams.reserve(streamSet_.size());
    if (wait_blocking_streams_only) {
      auto* null_stream = GetNullStream();
      for (auto* stream : streamSet_) {
        if (stream != null_stream && (stream->Flags() & hipStreamNonBlocking) == 0) {
          streams.push_back(stream);
          stream->retain();
        }
      }
      // Add null stream to the end so that wait happens after all blocking streams.
      if (null_stream != nullptr) {
        streams.push_back(null_stream);
        null_stream->retain();
      }
    } else {
      for (auto* stream : streamSet_) {
        streams.push_back(stream);
        stream->retain();
      }
    }
  }
  for (auto* stream : streams) {
    stream->finish(cpu_wait);
    stream->release();
  }
  // Release freed memory for all memory pools on the device
  ReleaseFreedMemory();
}

// ================================================================================================
bool Device::StreamCaptureBlocking() {
  std::shared_lock lock(streamSetLock_);
  for (auto* stream : streamSet_) {
    if (stream->GetCaptureStatus() == hipStreamCaptureStatusActive &&
        (stream->Flags() & hipStreamNonBlocking) == 0) {
      return true;
    }
  }
  return false;
}

// ================================================================================================
hipError_t Device::EnablePeerAccess(int peerDeviceId) {
  std::scoped_lock lock(lock_);
  if (std::find(userEnabledPeers_.begin(), userEnabledPeers_.end(), peerDeviceId)
      != userEnabledPeers_.end()) {
    return hipErrorPeerAccessAlreadyEnabled;
  }
  userEnabledPeers_.push_back(peerDeviceId);
  return hipSuccess;
}

// ================================================================================================
hipError_t Device::DisablePeerAccess(int peerDeviceId) {
  std::scoped_lock lock(lock_);
  auto it = std::find(userEnabledPeers_.begin(), userEnabledPeers_.end(), peerDeviceId);
  if (it == userEnabledPeers_.end()) {
    return hipErrorPeerAccessNotEnabled;
  }
  userEnabledPeers_.erase(it);
  return hipSuccess;
}

// ================================================================================================
bool Device::GetActiveStatus() {
  if (!isActive_.load(std::memory_order_acquire)) {
    std::shared_lock lock(streamSetLock_);
    for (const auto* stream : streamSet_) {
      if (stream->GetQueueStatus()) {
        isActive_.store(true, std::memory_order_release);
        break;
      }
    }
  }
  return isActive_.load(std::memory_order_relaxed);
}

// ================================================================================================
Device::~Device() {
  if ((IS_LINUX || !DEBUG_HIP_MEM_POOL_VMHEAP) && (default_mem_pool_ != nullptr)) {
    default_mem_pool_->release();
  }

  registeredGraphicsResources_.clear();
  mappedGraphicsResources_.clear();

  if (graph_mem_pool_ != nullptr) {
    graph_mem_pool_->release();
  }

  if (default_managed_mem_pool_ != nullptr) {
    default_managed_mem_pool_->release();
  }

  if (null_stream_ != nullptr) {
    hip::Stream::Destroy(null_stream_);
  }
}

// ================================================================================================
void ihipDestroyDevice() {
  for (auto deviceHandle : g_devices) {
    delete deviceHandle;
  }
}

// ================================================================================================
hipError_t ihipDeviceGet(hipDevice_t* device, int deviceId) {
  if (device == nullptr) {
    return hipErrorInvalidValue;
  }

  if (deviceId < 0 || static_cast<size_t>(deviceId) >= g_devices.size()) {
    return hipErrorInvalidDevice;
  }

  *device = deviceId;
  return hipSuccess;
}

// ================================================================================================
hipError_t hipDeviceGet(hipDevice_t* device, int deviceId) {
  HIP_INIT_API(hipDeviceGet, device, deviceId);

  HIP_RETURN(ihipDeviceGet(device, deviceId));
}

// ================================================================================================
hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device) {
  HIP_INIT_API(hipDeviceTotalMem, bytes, device);

  if (device < 0 || static_cast<size_t>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (bytes == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto* deviceHandle = g_devices[device]->devices()[0];
  const auto& info = deviceHandle->info();
  *bytes = info.globalMemSize_;

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device) {
  HIP_INIT_API(hipDeviceComputeCapability, major, minor, device);

  if (device < 0 || static_cast<size_t>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (major == nullptr || minor == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto* deviceHandle = g_devices[device]->devices()[0];
  const auto& isa = deviceHandle->isa();
  *major = isa.versionMajor();
  *minor = isa.versionMinor();

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipDeviceGetCount(int* count) {
  HIP_INIT_API(hipDeviceGetCount, count);

  HIP_RETURN(ihipDeviceGetCount(count));
}

// ================================================================================================
hipError_t ihipDeviceGetCount(int* count) {
  if (count == nullptr) {
    return hipErrorInvalidValue;
  }

  // Get all available devices
  *count = g_devices.size();

  if (*count < 1) {
    return hipErrorNoDevice;
  }

  return hipSuccess;
}

// ================================================================================================
hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device) {
  HIP_INIT_API(hipDeviceGetName, static_cast<void*>(name), len, device);

  if (device < 0 || static_cast<size_t>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (name == nullptr || len <= 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto* deviceHandle = g_devices[device]->devices()[0];
  const auto& info = deviceHandle->info();
  const auto nameLen = ::strlen(info.boardName_);

  // Only copy partial name if size of `dest` is smaller than size of `src` including
  // trailing zero byte
  const auto memcpySize = (len <= (nameLen + 1) ? (len - 1) : nameLen);
  ::memcpy(name, info.boardName_, memcpySize);
  name[memcpySize] = '\0';

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t device) {
  HIP_INIT_API(hipDeviceGetUuid, reinterpret_cast<void*>(uuid), device);

  if (device < 0 || static_cast<size_t>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (uuid == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto* deviceHandle = g_devices[device]->devices()[0];
  const auto& info = deviceHandle->info();
  memcpy(uuid->bytes, info.uuid_, sizeof(info.uuid_));

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t ihipGetDeviceProperties(hipDeviceProp_tR0600* props, int device) {
  if (props == nullptr) {
    return hipErrorInvalidValue;
  }

  if (static_cast<unsigned>(device) >= g_devices.size()) {
    return hipErrorInvalidDevice;
  }
  auto* deviceHandle = g_devices[device]->devices()[0];

  constexpr auto kPixelSizeMax = 16;
  constexpr auto kInt32Max = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
  constexpr auto kUint16Max = static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1;
  hipDeviceProp_tR0600 deviceProps = {};

  const auto& info = deviceHandle->info();
  const auto& isa = deviceHandle->isa();
  ::strncpy(deviceProps.name, info.boardName_, sizeof(info.boardName_));
  memcpy(deviceProps.uuid.bytes, info.uuid_, sizeof(info.uuid_));
  deviceProps.totalGlobalMem = info.globalMemSize_;
  deviceProps.sharedMemPerBlock = info.localMemSizePerCU_;
  deviceProps.sharedMemPerMultiprocessor = info.localMemSizePerCU_;
  deviceProps.regsPerBlock = info.availableRegistersPerCU_;
  deviceProps.warpSize = info.wavefrontWidth_;
  deviceProps.maxThreadsPerBlock = info.maxWorkGroupSize_;
  deviceProps.maxThreadsDim[0] = info.maxWorkItemSizes_[0];
  deviceProps.maxThreadsDim[1] = info.maxWorkItemSizes_[1];
  deviceProps.maxThreadsDim[2] = info.maxWorkItemSizes_[2];
  deviceProps.maxGridSize[0] = kInt32Max;
  deviceProps.maxGridSize[1] = kUint16Max;
  deviceProps.maxGridSize[2] = kUint16Max;
  deviceProps.clockRate = info.maxEngineClockFrequency_ * 1000;
  deviceProps.memoryClockRate = info.maxMemoryClockFrequency_ * 1000;
  deviceProps.memoryBusWidth = info.vramBusBitWidth_;
  deviceProps.totalConstMem = std::min(info.maxConstantBufferSize_, kInt32Max);
  deviceProps.major = isa.versionMajor();
  deviceProps.minor = isa.versionMinor();
  deviceProps.multiProcessorCount = info.maxComputeUnits_;
  deviceProps.l2CacheSize = info.l2CacheSize_;
  deviceProps.maxThreadsPerMultiProcessor = info.maxThreadsPerCU_;
  deviceProps.maxBlocksPerMultiProcessor = static_cast<int>(info.maxThreadsPerCU_ / info.wavefrontWidth_);
  deviceProps.computeMode = 0;
  deviceProps.clockInstructionRate = info.timeStampFrequency_;
  deviceProps.arch.hasGlobalInt32Atomics = 1;
  deviceProps.arch.hasGlobalFloatAtomicExch = 1;
  deviceProps.arch.hasSharedInt32Atomics = 1;
  deviceProps.arch.hasSharedFloatAtomicExch = 1;
  deviceProps.arch.hasFloatAtomicAdd = 1;
  deviceProps.arch.hasGlobalInt64Atomics = 1;
  deviceProps.arch.hasSharedInt64Atomics = 1;
  deviceProps.hostNativeAtomicSupported = info.pcie_atomics_ ? 1 : 0;
  deviceProps.arch.hasDoubles = 1;
  deviceProps.arch.hasWarpVote = 1;
  deviceProps.arch.hasWarpBallot = 1;
  deviceProps.arch.hasWarpShuffle = 1;
  deviceProps.arch.hasFunnelShift = 0;
  deviceProps.arch.hasThreadFenceSystem = 1;
  deviceProps.arch.hasSyncThreadsExt = 0;
  deviceProps.arch.hasSurfaceFuncs = 0;
  deviceProps.arch.has3dGrid = 1;
  deviceProps.arch.hasDynamicParallelism = 0;
  deviceProps.concurrentKernels = 1;
  deviceProps.pciDomainID = info.pciDomainID;
  deviceProps.pciBusID = info.deviceTopology_.pcie.bus;
  deviceProps.pciDeviceID = info.deviceTopology_.pcie.device;
  deviceProps.maxSharedMemoryPerMultiProcessor = info.localMemSizePerCU_;
  deviceProps.canMapHostMemory = 1;
  deviceProps.regsPerMultiprocessor = info.availableRegistersPerCU_;
  snprintf(deviceProps.gcnArchName, sizeof(deviceProps.gcnArchName), "%s", isa.targetId());
  deviceProps.cooperativeLaunch = info.cooperativeGroups_;
  deviceProps.cooperativeMultiDeviceLaunch = info.cooperativeMultiDeviceGroups_;

  deviceProps.cooperativeMultiDeviceUnmatchedFunc = info.cooperativeMultiDeviceGroups_;
  deviceProps.cooperativeMultiDeviceUnmatchedGridDim = info.cooperativeMultiDeviceGroups_;
  deviceProps.cooperativeMultiDeviceUnmatchedBlockDim = info.cooperativeMultiDeviceGroups_;
  deviceProps.cooperativeMultiDeviceUnmatchedSharedMem = info.cooperativeMultiDeviceGroups_;

  deviceProps.maxTexture1DLinear = std::min(kPixelSizeMax * info.imageMaxBufferSize_, kInt32Max);
  deviceProps.maxTexture1DMipmap = std::min(kPixelSizeMax * info.imageMaxBufferSize_, kInt32Max);
  deviceProps.maxTexture1D = deviceProps.maxSurface1D = std::min(info.image1DMaxWidth_, kInt32Max);
  deviceProps.maxTexture2D[0] = deviceProps.maxSurface2D[0] =
      std::min(info.image2DMaxWidth_, kInt32Max);
  deviceProps.maxTexture2D[1] = deviceProps.maxSurface2D[1] =
      std::min(info.image2DMaxHeight_, kInt32Max);
  deviceProps.maxTexture3D[0] = deviceProps.maxSurface3D[0] =
      std::min(info.image3DMaxWidth_, kInt32Max);
  deviceProps.maxTexture3D[1] = deviceProps.maxSurface3D[1] =
      std::min(info.image3DMaxHeight_, kInt32Max);
  deviceProps.maxTexture3D[2] = deviceProps.maxSurface3D[2] =
      std::min(info.image3DMaxDepth_, kInt32Max);
  deviceProps.maxTexture1DLayered[0] = deviceProps.maxSurface1DLayered[0] =
      std::min(info.image1DAMaxWidth_, kInt32Max);
  deviceProps.maxTexture1DLayered[1] = deviceProps.maxSurface1DLayered[1] =
      std::min(info.imageMaxArraySize_, kInt32Max);
  deviceProps.maxTexture2DLayered[0] = deviceProps.maxSurface2DLayered[0] =
      std::min(info.image2DAMaxWidth_[0], kInt32Max);
  deviceProps.maxTexture2DLayered[1] = deviceProps.maxSurface2DLayered[1] =
      std::min(info.image2DAMaxWidth_[1], kInt32Max);
  deviceProps.maxTexture2DLayered[2] = deviceProps.maxSurface2DLayered[2] =
      std::min(info.imageMaxArraySize_, kInt32Max);
  deviceProps.hdpMemFlushCntl = info.hdpMemFlushCntl;
  deviceProps.hdpRegFlushCntl = info.hdpRegFlushCntl;

  deviceProps.memPitch = std::min(info.maxMemAllocSize_, kInt32Max);
  deviceProps.textureAlignment = deviceProps.surfaceAlignment = info.imageBaseAddressAlignment_;
  deviceProps.texturePitchAlignment = info.imagePitchAlignment_;
  deviceProps.kernelExecTimeoutEnabled = 0;
  deviceProps.ECCEnabled = info.errorCorrectionSupport_ ? 1 : 0;
  deviceProps.isLargeBar = info.largeBar_ ? 1 : 0;
  deviceProps.asicRevision = info.asicRevision_;
  deviceProps.ipcEventSupported = 1;
  deviceProps.streamPrioritiesSupported = 1;
  deviceProps.multiGpuBoardGroupID = info.deviceTopology_.pcie.device;

  // HMM capabilities
  deviceProps.asyncEngineCount = info.numAsyncQueues_;
  deviceProps.deviceOverlap = (info.numAsyncQueues_ > 0) ? 1 : 0;
  deviceProps.unifiedAddressing = info.hmmDirectHostAccess_;
  deviceProps.managedMemory = info.hmmSupported_;
  deviceProps.concurrentManagedAccess = info.hmmSupported_;
  deviceProps.directManagedMemAccessFromHost = info.hmmDirectHostAccess_;
  deviceProps.canUseHostPointerForRegisteredMem = info.hostUnifiedMemory_;
  deviceProps.pageableMemoryAccess = info.hmmCpuMemoryAccessible_;
  deviceProps.hostRegisterSupported = true;
  deviceProps.pageableMemoryAccessUsesHostPageTables = info.iommuv2_;

  // Mem pool
  deviceProps.memoryPoolsSupported = HIP_MEM_POOL_SUPPORT;
  unsigned int memPoolHandleType = 0;
  if (HIP_MEM_POOL_SUPPORT) {
#if defined(__linux__)
    memPoolHandleType |= hipMemHandleTypePosixFileDescriptor;
#elif defined(_WIN32)
    memPoolHandleType |= hipMemHandleTypeWin32;
    memPoolHandleType |= hipMemHandleTypeWin32Kmt;
#endif
  }
  deviceProps.memoryPoolSupportedHandleTypes = memPoolHandleType;

  // Caching behavior
  deviceProps.globalL1CacheSupported = 1;
  deviceProps.localL1CacheSupported = 1;
  deviceProps.persistingL2CacheMaxSize = info.l2CacheSize_;
  deviceProps.reservedSharedMemPerBlock = 0;
  deviceProps.sharedMemPerBlockOptin = info.localMemSizePerCU_;

  // Unsupported features
  // Single to double precision perf ratio
  deviceProps.singleToDoublePrecisionPerfRatio = 0;
  // Flag hipHostRegisterReadOnly
  deviceProps.hostRegisterReadOnlySupported = 0;
  // Compute preemption
  deviceProps.computePreemptionSupported = 0;
  // Cubemaps
  deviceProps.maxTextureCubemap = 0;
  deviceProps.maxTextureCubemapLayered[0] = 0;
  deviceProps.maxTextureCubemapLayered[1] = 0;
  deviceProps.maxSurfaceCubemap = 0;
  deviceProps.maxSurfaceCubemapLayered[0] = 0;
  deviceProps.maxSurfaceCubemapLayered[1] = 0;
  // Texture gather ops
  deviceProps.maxTexture2DGather[0] = 0;
  deviceProps.maxTexture2DGather[1] = 0;
  // Textures bound to pitch memory
  deviceProps.maxTexture2DLinear[0] = std::min(info.image2DMaxWidth_, kInt32Max);
  deviceProps.maxTexture2DLinear[1] = std::min(info.image2DMaxHeight_, kInt32Max);
  deviceProps.maxTexture2DLinear[2] = std::min(kPixelSizeMax * info.image2DMaxWidth_, kInt32Max);
  // Alternate 3D texture
  deviceProps.maxTexture3DAlt[0] = 0;
  deviceProps.maxTexture3DAlt[1] = 0;
  deviceProps.maxTexture3DAlt[2] = 0;
  // access policy
  deviceProps.accessPolicyMaxWindowSize = 0;
  // cluster launch
  deviceProps.clusterLaunch = 0;
  // Mapping HIP array
  deviceProps.deferredMappingHipArraySupported = 0;
  // RDMA options
  deviceProps.gpuDirectRDMASupported = 0;
  deviceProps.gpuDirectRDMAFlushWritesOptions = 0;
  deviceProps.gpuDirectRDMAWritesOrdering = 0;
  *reinterpret_cast<uint32_t*>(&deviceProps.luid[0]) = info.luidLowPart_;
  *reinterpret_cast<uint32_t*>(&deviceProps.luid[sizeof(uint32_t)]) = info.luidHighPart_;
  deviceProps.luidDeviceNodeMask = info.luidDeviceNodeMask_;

  deviceProps.sparseHipArraySupported = 0;
  deviceProps.timelineSemaphoreInteropSupported = 0;
  deviceProps.unifiedFunctionPointers = 0;

  deviceProps.integrated = info.hostUnifiedMemory_;

  *props = deviceProps;
  return hipSuccess;
}

// ================================================================================================
hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_tR0600* prop, int device) {
  HIP_INIT_API(hipGetDevicePropertiesR0600, prop, device);

  HIP_RETURN(ihipGetDeviceProperties(prop, device));
}

// ================================================================================================
hipError_t hipGetDevicePropertiesR0000(hipDeviceProp_tR0000* prop, int device) {
  HIP_INIT_API(hipGetDevicePropertiesR0000, prop, device);

  if (prop == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (static_cast<unsigned>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }
  auto* deviceHandle = g_devices[device]->devices()[0];

  constexpr auto kPixelSizeMax = 16;
  constexpr auto kInt32Max = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
  constexpr auto kUint16Max = static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1;
  hipDeviceProp_tR0000 deviceProps = {};

  const auto& info = deviceHandle->info();
  const auto& isa = deviceHandle->isa();
  ::strncpy(deviceProps.name, info.boardName_, sizeof(deviceProps.name));
  deviceProps.totalGlobalMem = info.globalMemSize_;
  deviceProps.sharedMemPerBlock = info.localMemSizePerCU_;
  deviceProps.regsPerBlock = info.availableRegistersPerCU_;
  deviceProps.warpSize = info.wavefrontWidth_;
  deviceProps.maxThreadsPerBlock = info.maxWorkGroupSize_;
  deviceProps.maxThreadsDim[0] = info.maxWorkItemSizes_[0];
  deviceProps.maxThreadsDim[1] = info.maxWorkItemSizes_[1];
  deviceProps.maxThreadsDim[2] = info.maxWorkItemSizes_[2];
  deviceProps.maxGridSize[0] = kInt32Max;
  deviceProps.maxGridSize[1] = kUint16Max;
  deviceProps.maxGridSize[2] = kUint16Max;
  deviceProps.clockRate = info.maxEngineClockFrequency_ * 1000;
  deviceProps.memoryClockRate = info.maxMemoryClockFrequency_ * 1000;
  deviceProps.memoryBusWidth = info.vramBusBitWidth_;
  deviceProps.totalConstMem = std::min(info.maxConstantBufferSize_, kInt32Max);
  deviceProps.major = isa.versionMajor();
  deviceProps.minor = isa.versionMinor();
  deviceProps.multiProcessorCount = info.maxComputeUnits_;
  deviceProps.l2CacheSize = info.l2CacheSize_;
  deviceProps.maxThreadsPerMultiProcessor = info.maxThreadsPerCU_;
  deviceProps.computeMode = 0;
  deviceProps.clockInstructionRate = info.timeStampFrequency_;
  deviceProps.arch.hasGlobalInt32Atomics = 1;
  deviceProps.arch.hasGlobalFloatAtomicExch = 1;
  deviceProps.arch.hasSharedInt32Atomics = 1;
  deviceProps.arch.hasSharedFloatAtomicExch = 1;
  deviceProps.arch.hasFloatAtomicAdd = 1;
  deviceProps.arch.hasGlobalInt64Atomics = 1;
  deviceProps.arch.hasSharedInt64Atomics = 1;
  deviceProps.arch.hasDoubles = 1;
  deviceProps.arch.hasWarpVote = 1;
  deviceProps.arch.hasWarpBallot = 1;
  deviceProps.arch.hasWarpShuffle = 1;
  deviceProps.arch.hasFunnelShift = 0;
  deviceProps.arch.hasThreadFenceSystem = 1;
  deviceProps.arch.hasSyncThreadsExt = 0;
  deviceProps.arch.hasSurfaceFuncs = 0;
  deviceProps.arch.has3dGrid = 1;
  deviceProps.arch.hasDynamicParallelism = 0;
  deviceProps.concurrentKernels = 1;
  deviceProps.pciDomainID = info.pciDomainID;
  deviceProps.pciBusID = info.deviceTopology_.pcie.bus;
  deviceProps.pciDeviceID = info.deviceTopology_.pcie.device;
  deviceProps.maxSharedMemoryPerMultiProcessor = info.localMemSizePerCU_;
  deviceProps.canMapHostMemory = 1;
  // FIXME: This should be removed, targets can have character names as well.
  deviceProps.gcnArch = isa.versionMajor() * 100 + isa.versionMinor() * 10 + isa.versionStepping();
  snprintf(deviceProps.gcnArchName, sizeof(deviceProps.gcnArchName), "%s", isa.targetId());
  deviceProps.cooperativeLaunch = info.cooperativeGroups_;
  deviceProps.cooperativeMultiDeviceLaunch = info.cooperativeMultiDeviceGroups_;

  deviceProps.cooperativeMultiDeviceUnmatchedFunc = info.cooperativeMultiDeviceGroups_;
  deviceProps.cooperativeMultiDeviceUnmatchedGridDim = info.cooperativeMultiDeviceGroups_;
  deviceProps.cooperativeMultiDeviceUnmatchedBlockDim = info.cooperativeMultiDeviceGroups_;
  deviceProps.cooperativeMultiDeviceUnmatchedSharedMem = info.cooperativeMultiDeviceGroups_;

  deviceProps.maxTexture1DLinear =
      std::min(kPixelSizeMax * info.imageMaxBufferSize_, kInt32Max);
  deviceProps.maxTexture1D = std::min(info.image1DMaxWidth_, kInt32Max);
  deviceProps.maxTexture2D[0] = std::min(info.image2DMaxWidth_, kInt32Max);
  deviceProps.maxTexture2D[1] = std::min(info.image2DMaxHeight_, kInt32Max);
  deviceProps.maxTexture3D[0] = std::min(info.image3DMaxWidth_, kInt32Max);
  deviceProps.maxTexture3D[1] = std::min(info.image3DMaxHeight_, kInt32Max);
  deviceProps.maxTexture3D[2] = std::min(info.image3DMaxDepth_, kInt32Max);
  deviceProps.hdpMemFlushCntl = info.hdpMemFlushCntl;
  deviceProps.hdpRegFlushCntl = info.hdpRegFlushCntl;

  deviceProps.memPitch = std::min(info.maxMemAllocSize_, kInt32Max);
  deviceProps.textureAlignment = info.imageBaseAddressAlignment_;
  deviceProps.texturePitchAlignment = info.imagePitchAlignment_;
  deviceProps.kernelExecTimeoutEnabled = 0;
  deviceProps.ECCEnabled = info.errorCorrectionSupport_ ? 1 : 0;
  deviceProps.isLargeBar = info.largeBar_ ? 1 : 0;
  deviceProps.asicRevision = info.asicRevision_;

  // HMM capabilities
  deviceProps.managedMemory = info.hmmSupported_;
  deviceProps.concurrentManagedAccess = info.hmmSupported_;
  deviceProps.directManagedMemAccessFromHost = info.hmmDirectHostAccess_;
  deviceProps.pageableMemoryAccess = info.hmmCpuMemoryAccessible_;
  deviceProps.pageableMemoryAccessUsesHostPageTables = info.hostUnifiedMemory_;

  *prop = deviceProps;
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipGetProcAddress_common(const char* symbol, void** pfn, int hipVersion, uint64_t flags,
                             hipDriverProcAddressQueryResult* symbolStatus) {
  if (symbol == nullptr || *symbol == '\0' || pfn == nullptr) {
    return hipErrorInvalidValue;
  }
  std::string symbolString = symbol;

  if (flags != HIP_GET_PROC_ADDRESS_DEFAULT && flags != HIP_GET_PROC_ADDRESS_LEGACY_STREAM &&
      flags != HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) {
    return hipErrorInvalidValue;
  }

  bool checkSpt = (flags == HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM);
  if (symbolString == "hipGetDeviceProperties") {
    if (hipVersion >= 600) {
      symbolString = "hipGetDevicePropertiesR0600";
    }
    checkSpt = false;
  } else if (symbolString == "hipChooseDevice") {
    if (hipVersion >= 600) {
      symbolString = "hipChooseDeviceR0600";
    }
    checkSpt = false;
  } else if (symbolString == "hipAmdFileRead") {
    *pfn = reinterpret_cast<void*>(&hipAmdFileRead);
    if (symbolStatus != nullptr) {
      *symbolStatus = HIP_GET_PROC_ADDRESS_SUCCESS;
    }
    return hipSuccess;
  } else if (symbolString == "hipAmdFileWrite") {
    *pfn = reinterpret_cast<void*>(&hipAmdFileWrite);
    if (symbolStatus != nullptr) {
      *symbolStatus = HIP_GET_PROC_ADDRESS_SUCCESS;
    }
    return hipSuccess;
  }

  void* handle = hip::PlatformState::Instance().GetDynamicLibraryHandle();
  if (handle == nullptr) {
    return hipErrorInvalidValue;
  }

  if (checkSpt) {
    symbolString += "_spt";
  }

  *pfn = amd::Os::getSymbol(handle, symbolString.c_str());
  if (*pfn == nullptr) {
    if (checkSpt) {
      *pfn = amd::Os::getSymbol(handle, symbol);
    }
    if (*pfn == nullptr) {
      if (symbolStatus != nullptr) {
        *symbolStatus = HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
      }
      return hipErrorInvalidValue;
    }
  }

  if (symbolStatus != nullptr) {
    *symbolStatus = HIP_GET_PROC_ADDRESS_SUCCESS;
  }

  return hipSuccess;
}

// ================================================================================================
hipError_t hipGetProcAddress(const char* symbol, void** pfn, int hipVersion, uint64_t flags,
                             hipDriverProcAddressQueryResult* symbolStatus) {
  HIP_INIT_API(hipGetProcAddress, symbol, pfn, hipVersion, flags, symbolStatus);
  HIP_RETURN(hipGetProcAddress_common(symbol, pfn, hipVersion, flags, symbolStatus));
}

// ================================================================================================
hipError_t hipGetProcAddress_spt(const char* symbol, void** pfn, int hipVersion, uint64_t flags,
                                 hipDriverProcAddressQueryResult* symbolStatus) {
  HIP_INIT_API(hipGetProcAddress, symbol, pfn, hipVersion, flags, symbolStatus);
  flags = (flags == HIP_GET_PROC_ADDRESS_DEFAULT) ? HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
                                                  : flags;
  HIP_RETURN(hipGetProcAddress_common(symbol, pfn, hipVersion, flags, symbolStatus));
}

}  // namespace hip

// ================================================================================================
extern "C" hipError_t hipGetDeviceProperties(hipDeviceProp_tR0000* props, hipDevice_t device) {
  return hip::hipGetDevicePropertiesR0000(props, device);
}
