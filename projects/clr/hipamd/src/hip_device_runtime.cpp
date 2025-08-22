/* Copyright (c) 2018 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <hip/hip_runtime.h>

#include "hip_internal.hpp"
#include "hip_platform.hpp"

#undef hipChooseDevice
#undef hipDeviceProp_t

namespace hip {

hipError_t hipGetDevicePropertiesR0000(hipDeviceProp_tR0000* prop, int device);

template <typename DeviceProp>
hipError_t ihipChooseDevice(int* device, const DeviceProp* properties) {
  if (device == nullptr || properties == nullptr) {
    return hipErrorInvalidValue;
  }

  *device = 0;
  cl_uint maxMatchedCount = 0;
  int count = 0;
  IHIP_RETURN_ONFAIL(ihipDeviceGetCount(&count));

  for (cl_int i = 0; i < count; ++i) {
    DeviceProp currentProp = {0};
    cl_uint validPropCount = 0;
    cl_uint matchedCount = 0;
    hipError_t err = hipSuccess;

    if constexpr (std::is_same_v<DeviceProp, hipDeviceProp_tR0600>) {
      err = ihipGetDeviceProperties(&currentProp, i);
    } else {
      err = hip::hipGetDevicePropertiesR0000(&currentProp, i);
    }

    if (properties->major != 0) {
      validPropCount++;
      if (currentProp.major >= properties->major) {
        matchedCount++;
      }
    }
    if (properties->minor != 0) {
      validPropCount++;
      if (currentProp.minor >= properties->minor) {
        matchedCount++;
      }
    }
    if (properties->totalGlobalMem != 0) {
      validPropCount++;
      if (currentProp.totalGlobalMem >= properties->totalGlobalMem) {
        matchedCount++;
      }
    }
    if (properties->sharedMemPerBlock != 0) {
      validPropCount++;
      if (currentProp.sharedMemPerBlock >= properties->sharedMemPerBlock) {
        matchedCount++;
      }
    }
    if (properties->maxThreadsPerBlock != 0) {
      validPropCount++;
      if (currentProp.maxThreadsPerBlock >= properties->maxThreadsPerBlock) {
        matchedCount++;
      }
    }
    if (properties->totalConstMem != 0) {
      validPropCount++;
      if (currentProp.totalConstMem >= properties->totalConstMem) {
        matchedCount++;
      }
    }
    if (properties->multiProcessorCount != 0) {
      validPropCount++;
      if (currentProp.multiProcessorCount >= properties->multiProcessorCount) {
        matchedCount++;
      }
    }
    if (properties->maxThreadsPerMultiProcessor != 0) {
      validPropCount++;
      if (currentProp.maxThreadsPerMultiProcessor >= properties->maxThreadsPerMultiProcessor) {
        matchedCount++;
      }
    }
    if (properties->memoryClockRate != 0) {
      validPropCount++;
      if (currentProp.memoryClockRate >= properties->memoryClockRate) {
        matchedCount++;
      }
    }
    if (properties->memoryBusWidth != 0) {
      validPropCount++;
      if (currentProp.memoryBusWidth >= properties->memoryBusWidth) {
        matchedCount++;
      }
    }
    if (properties->l2CacheSize != 0) {
      validPropCount++;
      if (currentProp.l2CacheSize >= properties->l2CacheSize) {
        matchedCount++;
      }
    }
    if (properties->regsPerBlock != 0) {
      validPropCount++;
      if (currentProp.regsPerBlock >= properties->regsPerBlock) {
        matchedCount++;
      }
    }
    if (properties->maxSharedMemoryPerMultiProcessor != 0) {
      validPropCount++;
      if (currentProp.maxSharedMemoryPerMultiProcessor >=
          properties->maxSharedMemoryPerMultiProcessor) {
        matchedCount++;
      }
    }
    if (properties->warpSize != 0) {
      validPropCount++;
      if (currentProp.warpSize >= properties->warpSize) {
        matchedCount++;
      }
    }
    if (validPropCount == matchedCount) {
      *device = matchedCount > maxMatchedCount ? i : *device;
      maxMatchedCount = std::max(matchedCount, maxMatchedCount);
    }
  }

  return hipSuccess;
}

hipError_t hipChooseDeviceR0600(int* device, const hipDeviceProp_tR0600* properties) {
  HIP_INIT_API(hipChooseDeviceR0600, device, properties);
  HIP_RETURN(ihipChooseDevice(device, properties));
}

hipError_t hipChooseDeviceR0000(int* device, const hipDeviceProp_tR0000* properties) {
  HIP_INIT_API(hipChooseDeviceR0000, device, properties);
  HIP_RETURN(ihipChooseDevice(device, properties));
}

hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int device) {
  HIP_INIT_API(hipDeviceGetAttribute, pi, attr, device);

  if (pi == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  int count = 0;
  HIP_RETURN_ONFAIL(ihipDeviceGetCount(&count));

  if (device < 0 || device >= count) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  // FIXME: should we cache the props, or just select from deviceHandle->info_?
  hipDeviceProp_tR0600 prop = {0};
  HIP_RETURN_ONFAIL(ihipGetDeviceProperties(&prop, device));

  constexpr auto int32_max = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());

  switch (attr) {
    case hipDeviceAttributeMaxThreadsPerBlock:
      *pi = prop.maxThreadsPerBlock;
      break;
    case hipDeviceAttributeAsyncEngineCount:
      *pi = prop.asyncEngineCount;
      break;
    case hipDeviceAttributeMaxBlockDimX:
      *pi = prop.maxThreadsDim[0];
      break;
    case hipDeviceAttributeMaxBlockDimY:
      *pi = prop.maxThreadsDim[1];
      break;
    case hipDeviceAttributeMaxBlockDimZ:
      *pi = prop.maxThreadsDim[2];
      break;
    case hipDeviceAttributeMaxGridDimX:
      *pi = prop.maxGridSize[0];
      break;
    case hipDeviceAttributeMaxGridDimY:
      *pi = prop.maxGridSize[1];
      break;
    case hipDeviceAttributeMaxGridDimZ:
      *pi = prop.maxGridSize[2];
      break;
    case hipDeviceAttributeMaxSurface1D:
      *pi = prop.maxSurface1D;
      break;
    case hipDeviceAttributeMaxSharedMemoryPerBlock:
      *pi = prop.sharedMemPerBlock;
      break;
    case hipDeviceAttributeSharedMemPerBlockOptin:
      *pi = prop.sharedMemPerBlockOptin;
      break;
    case hipDeviceAttributeSharedMemPerMultiprocessor:
      *pi = prop.sharedMemPerMultiprocessor;
      break;
    case hipDeviceAttributeStreamPrioritiesSupported:
      *pi = prop.streamPrioritiesSupported;
      break;
    case hipDeviceAttributeSurfaceAlignment:
      *pi = prop.surfaceAlignment;
      break;
    case hipDeviceAttributeTotalConstantMemory:
      // size_t to int casting
      *pi = std::min(prop.totalConstMem, int32_max);
      break;
    case hipDeviceAttributeTotalGlobalMem:
      *pi = std::min(prop.totalGlobalMem, int32_max);
      break;
    case hipDeviceAttributeWarpSize:
      *pi = prop.warpSize;
      break;
    case hipDeviceAttributeMaxRegistersPerBlock:
      *pi = prop.regsPerBlock;
      break;
    case hipDeviceAttributeClockRate:
      *pi = prop.clockRate;
      break;
    case hipDeviceAttributeWallClockRate:
      *pi = g_devices[device]->devices()[0]->info().wallClockFrequency_;
      break;
    case hipDeviceAttributeMemoryClockRate:
      *pi = prop.memoryClockRate;
      break;
    case hipDeviceAttributeMemoryBusWidth:
      *pi = prop.memoryBusWidth;
      break;
    case hipDeviceAttributeMultiprocessorCount:
      *pi = prop.multiProcessorCount;
      break;
    case hipDeviceAttributeComputeMode:
      *pi = prop.computeMode;
      break;
    case hipDeviceAttributeComputePreemptionSupported:
      *pi = prop.computePreemptionSupported;
      break;
    case hipDeviceAttributeL2CacheSize:
      *pi = prop.l2CacheSize;
      break;
    case hipDeviceAttributeLocalL1CacheSupported:
      *pi = prop.localL1CacheSupported;
      break;
    case hipDeviceAttributeLuidDeviceNodeMask:
      *pi = prop.luidDeviceNodeMask;
      break;
    case hipDeviceAttributeMaxThreadsPerMultiProcessor:
      *pi = prop.maxThreadsPerMultiProcessor;
      break;
    case hipDeviceAttributeComputeCapabilityMajor:
      *pi = prop.major;
      break;
    case hipDeviceAttributeComputeCapabilityMinor:
      *pi = prop.minor;
      break;
    case hipDeviceAttributeMultiGpuBoardGroupID:
      *pi = prop.multiGpuBoardGroupID;
      break;
    case hipDeviceAttributePciBusId:
      *pi = prop.pciBusID;
      break;
    case hipDeviceAttributeConcurrentKernels:
      *pi = prop.concurrentKernels;
      break;
    case hipDeviceAttributePciDeviceId:
      *pi = prop.pciDeviceID;
      break;
    case hipDeviceAttributePciDomainId:
      *pi = prop.pciDomainID;
      break;
    case hipDeviceAttributePciChipId:
      *pi = static_cast<int>(g_devices[device]->devices()[0]->info().pcieDeviceId_);
      break;
    case hipDeviceAttributePersistingL2CacheMaxSize:
      *pi = prop.persistingL2CacheMaxSize;
      break;
    case hipDeviceAttributeMaxRegistersPerMultiprocessor:
      *pi = prop.regsPerMultiprocessor;
      break;
    case hipDeviceAttributeReservedSharedMemPerBlock:
      *pi = prop.reservedSharedMemPerBlock;
      break;
    case hipDeviceAttributeMaxSharedMemoryPerMultiprocessor:
      *pi = prop.maxSharedMemoryPerMultiProcessor;
      break;
    case hipDeviceAttributeIsMultiGpuBoard:
      *pi = prop.isMultiGpuBoard;
      break;
    case hipDeviceAttributeCooperativeLaunch:
      *pi = prop.cooperativeLaunch;
      break;
    case hipDeviceAttributeHostRegisterSupported:
      *pi = 1;  // AMD GPUs allow you to register host memory regardless of the GPU
      break;
    case hipDeviceAttributeDeviceOverlap:
      *pi = prop.asyncEngineCount > 0 ? 1 : 0;
      break;
    case hipDeviceAttributeCooperativeMultiDeviceLaunch:
      *pi = prop.cooperativeMultiDeviceLaunch;
      break;
    case hipDeviceAttributeIntegrated:
      *pi = prop.integrated;
      break;
    case hipDeviceAttributeMaxTexture1DWidth:
      *pi = prop.maxTexture1D;
      break;
    case hipDeviceAttributeMaxTexture1DLinear:
      *pi = prop.maxTexture1DLinear;
      break;
    case hipDeviceAttributeMaxTexture1DMipmap:
      *pi = prop.maxTexture1DMipmap;
      break;
    case hipDeviceAttributeMaxTextureCubemap:
      *pi = prop.maxTextureCubemap;
      break;
    case hipDeviceAttributeMaxTexture2DWidth:
      *pi = prop.maxTexture2D[0];
      break;
    case hipDeviceAttributeMaxTexture2DHeight:
      *pi = prop.maxTexture2D[1];
      break;
    case hipDeviceAttributeMaxTexture3DWidth:
      *pi = prop.maxTexture3D[0];
      break;
    case hipDeviceAttributeMaxTexture3DHeight:
      *pi = prop.maxTexture3D[1];
      break;
    case hipDeviceAttributeMaxTexture3DDepth:
      *pi = prop.maxTexture3D[2];
      break;
    case hipDeviceAttributeHdpMemFlushCntl:
      *reinterpret_cast<unsigned int**>(pi) = prop.hdpMemFlushCntl;
      break;
    case hipDeviceAttributeHdpRegFlushCntl:
      *reinterpret_cast<unsigned int**>(pi) = prop.hdpRegFlushCntl;
      break;
    case hipDeviceAttributeMaxPitch:
      // size_t to int casting
      *pi = std::min(prop.memPitch, int32_max);
      break;
    case hipDeviceAttributeTextureAlignment:
      *pi = prop.textureAlignment;
      break;
    case hipDeviceAttributeTexturePitchAlignment:
      *pi = prop.texturePitchAlignment;
      break;
    case hipDeviceAttributeKernelExecTimeout:
      *pi = prop.kernelExecTimeoutEnabled;
      break;
    case hipDeviceAttributeCanMapHostMemory:
      *pi = prop.canMapHostMemory;
      break;
    case hipDeviceAttributeCanUseHostPointerForRegisteredMem:
      *pi = prop.canUseHostPointerForRegisteredMem;
      break;
    case hipDeviceAttributeEccEnabled:
      *pi = prop.ECCEnabled;
      break;
    case hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc:
      *pi = prop.cooperativeMultiDeviceUnmatchedFunc;
      break;
    case hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim:
      *pi = prop.cooperativeMultiDeviceUnmatchedGridDim;
      break;
    case hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim:
      *pi = prop.cooperativeMultiDeviceUnmatchedBlockDim;
      break;
    case hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem:
      *pi = prop.cooperativeMultiDeviceUnmatchedSharedMem;
      break;
    case hipDeviceAttributeAsicRevision:
      *pi = prop.asicRevision;
      break;
    case hipDeviceAttributeManagedMemory:
      *pi = prop.managedMemory;
      break;
    case hipDeviceAttributeMaxBlocksPerMultiProcessor:
      *pi = prop.maxBlocksPerMultiProcessor;
      break;
    case hipDeviceAttributeDirectManagedMemAccessFromHost:
      *pi = prop.directManagedMemAccessFromHost;
      break;
    case hipDeviceAttributeGlobalL1CacheSupported:
      *pi = prop.globalL1CacheSupported;
      break;
    case hipDeviceAttributeHostNativeAtomicSupported:
      *pi = prop.hostNativeAtomicSupported;
      break;
    case hipDeviceAttributeConcurrentManagedAccess:
      *pi = prop.concurrentManagedAccess;
      break;
    case hipDeviceAttributePageableMemoryAccess:
      *pi = prop.pageableMemoryAccess;
      break;
    case hipDeviceAttributePageableMemoryAccessUsesHostPageTables:
      *pi = prop.pageableMemoryAccessUsesHostPageTables;
      break;
    case hipDeviceAttributeIsLargeBar:
      *pi = prop.isLargeBar;
      break;
    case hipDeviceAttributeUnifiedAddressing:
      // HIP runtime always uses SVM for host memory allocations.
      // Note: Host registered memory isn't covered by this feature
      // and still requires hipMemHostGetDevicePointer() call
      *pi = true;
      break;
    case hipDeviceAttributeCanUseStreamWaitValue:
      // hipStreamWaitValue64() and hipStreamWaitValue32() support
      *pi = g_devices[device]->devices()[0]->info().aqlBarrierValue_;
      break;
    case hipDeviceAttributeImageSupport:
      *pi = static_cast<int>(g_devices[device]->devices()[0]->info().imageSupport_);
      break;
    case hipDeviceAttributePhysicalMultiProcessorCount:
      *pi = g_devices[device]->devices()[0]->info().maxPhysicalComputeUnits_;
      break;
    case hipDeviceAttributeFineGrainSupport:
      *pi = static_cast<int>(g_devices[device]->devices()[0]->isFineGrainSupported());
      break;
    case hipDeviceAttributeMemoryPoolsSupported:
      *pi = HIP_MEM_POOL_SUPPORT;
      break;
    case hipDeviceAttributeMemoryPoolSupportedHandleTypes:
      *pi = prop.memoryPoolSupportedHandleTypes;
      break;
    case hipDeviceAttributeVirtualMemoryManagementSupported:
      *pi = static_cast<int>(g_devices[device]->devices()[0]->info().virtualMemoryManagement_);
      break;
    case hipDeviceAttributeAccessPolicyMaxWindowSize:
      *pi = prop.accessPolicyMaxWindowSize;
      break;
    case hipDeviceAttributeNumberOfXccs:
      *pi = static_cast<int>(g_devices[device]->devices()[0]->info().numberOfXccs_);
      break;
    case hipDeviceAttributeMaxAvailableVgprsPerThread:
      *pi = static_cast<int>(g_devices[device]->devices()[0]->info().availableVGPRs_);
      break;
    default:
      HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusIdstr) {
  HIP_INIT_API(hipDeviceGetByPCIBusId, device, pciBusIdstr);

  if (device == nullptr || pciBusIdstr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  int pciBusID = -1;
  int pciDeviceID = -1;
  int pciDomainID = -1;
  int pciFunction = -1;
  bool found = false;
  if (sscanf(pciBusIdstr, "%04x:%02x:%02x.%01x", reinterpret_cast<unsigned int*>(&pciDomainID),
             reinterpret_cast<unsigned int*>(&pciBusID),
             reinterpret_cast<unsigned int*>(&pciDeviceID),
             reinterpret_cast<unsigned int*>(&pciFunction)) == 0x4) {
    int count = 0;
    HIP_RETURN_ONFAIL(ihipDeviceGetCount(&count));
    for (cl_int i = 0; i < count; i++) {
      hipDevice_t dev;
      hipDeviceProp_tR0600 prop;
      HIP_RETURN_ONFAIL(ihipDeviceGet(&dev, i));
      HIP_RETURN_ONFAIL(ihipGetDeviceProperties(&prop, dev));
      auto* deviceHandle = g_devices[dev]->devices()[0];

      if ((pciBusID == prop.pciBusID) && (pciDomainID == prop.pciDomainID) &&
          (pciDeviceID == prop.pciDeviceID) &&
          (pciFunction == deviceHandle->info().deviceTopology_.pcie.function)) {
        *device = i;
        found = true;
        break;
      }
    }
  }
  if (!found) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* cacheConfig) {
  HIP_INIT_API(hipDeviceGetCacheConfig, cacheConfig);

  if (cacheConfig == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *cacheConfig = hipFuncCache_t();

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceGetLimit(size_t* pValue, hipLimit_t limit) {
  HIP_INIT_API(hipDeviceGetLimit, pValue, limit);

  if (pValue == nullptr || limit >= hipLimitRange) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  switch (limit) {
    case hipLimitMallocHeapSize:
      *pValue = hip::getCurrentDevice()->devices()[0]->InitialHeapSize();
      break;
    case hipLimitStackSize:
      *pValue = hip::getCurrentDevice()->devices()[0]->StackSize();
      break;
    case hipExtLimitScratchMin:
      *pValue = hip::getCurrentDevice()->devices()[0]->info().scratchLimitMin;
      break;
    case hipExtLimitScratchMax:
      *pValue = hip::getCurrentDevice()->devices()[0]->info().scratchLimitMax;
      ;
      break;
    case hipExtLimitScratchCurrent:
      *pValue = hip::getCurrentDevice()->devices()[0]->ScratchLimitCurrent();
      break;
    default:
      LogPrintfError("UnsupportedLimit = %d is passed", limit);
      HIP_RETURN(hipErrorUnsupportedLimit);
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device) {
  HIP_INIT_API(hipDeviceGetPCIBusId, (void*)pciBusId, len, device);

  int count;
  HIP_RETURN_ONFAIL(ihipDeviceGetCount(&count));

  if (device < 0 || device >= count) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  // pciBusId should be large enough to store 13 characters including the NULL-terminator.
  if (pciBusId == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipDeviceProp_tR0600 prop;
  HIP_RETURN_ONFAIL(ihipGetDeviceProperties(&prop, device));
  auto* deviceHandle = g_devices[device]->devices()[0];
  snprintf(pciBusId, len, "%04x:%02x:%02x.%01x", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID,
           deviceHandle->info().deviceTopology_.pcie.function);

  HIP_RETURN(len <= 12 ? hipErrorInvalidValue : hipSuccess);
}

hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* pConfig) {
  HIP_INIT_API(hipDeviceGetSharedMemConfig, pConfig);
  if (pConfig == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *pConfig = hipSharedMemBankSizeFourByte;

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceReset(void) {
  HIP_INIT_API(hipDeviceReset);

  hip::getCurrentDevice()->Reset();

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) {
  HIP_INIT_API(hipDeviceSetCacheConfig, cacheConfig);

  if (cacheConfig != hipFuncCachePreferNone && cacheConfig != hipFuncCachePreferShared &&
      cacheConfig != hipFuncCachePreferL1 && cacheConfig != hipFuncCachePreferEqual) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (!hip::tls.capture_streams_.empty() || !g_captureStreams.empty()) {
    HIP_RETURN(hipErrorStreamCaptureUnsupported);
  }

  // No way to set cache config yet.

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceSetLimit(hipLimit_t limit, size_t value) {
  HIP_INIT_API(hipDeviceSetLimit, limit, value);
  if (limit >= hipLimitRange) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  switch (limit) {
    case hipLimitStackSize:
      // need to query device size and take action
      if (!hip::getCurrentDevice()->devices()[0]->UpdateStackSize(value)) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      break;
    case hipLimitMallocHeapSize:
      if (!hip::getCurrentDevice()->devices()[0]->UpdateInitialHeapSize(value)) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      break;
    case hipExtLimitScratchCurrent:
      if (!hip::getCurrentDevice()->devices()[0]->UpdateScratchLimitCurrent(value)) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      break;
    default:
      LogPrintfError("UnsupportedLimit = %d is passed", limit);
      HIP_RETURN(hipErrorUnsupportedLimit);
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config) {
  HIP_INIT_API(hipDeviceSetSharedMemConfig, config);
  if (config != hipSharedMemBankSizeDefault && config != hipSharedMemBankSizeFourByte &&
      config != hipSharedMemBankSizeEightByte) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (!hip::tls.capture_streams_.empty() || !g_captureStreams.empty()) {
    HIP_RETURN(hipErrorStreamCaptureUnsupported);
  }

  // No way to set cache config yet.

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements,
                                               const hipChannelFormatDesc* fmtDesc, int device) {
  HIP_INIT_API(hipDeviceGetTexture1DLinearMaxWidth, maxWidthInElements, fmtDesc, device);
  if (maxWidthInElements == nullptr || fmtDesc == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipDeviceProp_tR0600 prop = {0};
  HIP_RETURN_ONFAIL(ihipGetDeviceProperties(&prop, device));
  // Calculate element size according to fmtDesc
  size_t elementSize =
      (fmtDesc->x + fmtDesc->y + fmtDesc->z + fmtDesc->w) / 8;  // Convert from bits to bytes
  if (elementSize == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *maxWidthInElements = prop.maxTexture1DLinear / elementSize;
  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceSynchronize() {
  HIP_INIT_API(hipDeviceSynchronize);
  CHECK_SUPPORTED_DURING_CAPTURE();
  constexpr bool kDoWaitForCpu = false;
  hip::getCurrentDevice()->SyncAllStreams(kDoWaitForCpu);
  HIP_RETURN_DURATION(hipSuccess);
}

int ihipGetDevice() {
  hip::Device* device = hip::getCurrentDevice();
  if (device == nullptr) {
    return -1;
  }
  return device->deviceId();
}

hipError_t hipGetDevice(int* deviceId) {
  HIP_INIT_API(hipGetDevice, deviceId);

  if (deviceId == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  Device* device = hip::getCurrentDevice();
  if (device == nullptr) {
    HIP_RETURN(hipErrorNoDevice);
  }

  *deviceId = device->deviceId();
  HIP_RETURN(hipSuccess, *deviceId);
}

hipError_t hipGetDeviceCount(int* count) {
  HIP_INIT_API_NO_RETURN(hipGetDeviceCount, count);

  HIP_RETURN(ihipDeviceGetCount(count));
}

hipError_t hipGetDeviceFlags(unsigned int* flags) {
  HIP_INIT_API(hipGetDeviceFlags, flags);
  if (flags == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *flags = hip::getCurrentDevice()->getFlags();
  HIP_RETURN(hipSuccess);
}

hipError_t hipGetDriverEntryPoint_common(const char* symbol, void** funcPtr,
                                         unsigned long long flags,
                                         hipDriverEntryPointQueryResult* status) {
  std::string symbolString = symbol;
  if (symbol == nullptr || symbolString == "" || funcPtr == nullptr) {
    return hipErrorInvalidValue;
  }

  if (flags != hipEnableDefault && flags != hipEnableLegacyStream &&
      flags != hipEnablePerThreadDefaultStream) {
    return hipErrorInvalidValue;
  }

  void* handle = hip::PlatformState::instance().getDynamicLibraryHandle();
  if (handle == nullptr) {
    return hipErrorInvalidValue;
  }

  if (flags == hipEnablePerThreadDefaultStream) {
    symbolString += "_spt";
  }

  *funcPtr = amd::Os::getSymbol(handle, symbolString.c_str());
  if (*funcPtr == nullptr) {
    if (flags == hipEnablePerThreadDefaultStream) {
      *funcPtr = amd::Os::getSymbol(handle, symbol);
    }
    if (*funcPtr == nullptr) {
      if (status != nullptr) {
        *status = hipDriverEntryPointSymbolNotFound;
      }
      return hipErrorInvalidValue;
    }
  }

  if (status != nullptr) {
    *status = hipDriverEntryPointSuccess;
  }

  return hipSuccess;
}

hipError_t hipGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags,
                                  hipDriverEntryPointQueryResult* status) {
  HIP_INIT_API(hipGetDriverEntryPoint, symbol, funcPtr, flags, status);
  HIP_RETURN(hipGetDriverEntryPoint_common(symbol, funcPtr, flags, status));
}

hipError_t hipGetDriverEntryPoint_spt(const char* symbol, void** funcPtr, unsigned long long flags,
                                      hipDriverEntryPointQueryResult* status) {
  HIP_INIT_API(hipGetDriverEntryPoint, symbol, funcPtr, flags, status);
  flags = (flags == hipEnableDefault) ? hipEnablePerThreadDefaultStream : flags;
  HIP_RETURN(hipGetDriverEntryPoint_common(symbol, funcPtr, flags, status));
}

hipError_t hipSetDevice(int device) {
  HIP_INIT_API_NO_RETURN(hipSetDevice, device);

  hip::tls.isSetDeviceCalled = true;
  // Check if the device is already set
  if (hip::tls.device_ != nullptr && hip::tls.device_->deviceId() == device) {
    HIP_RETURN(hipSuccess);
  }

  if (static_cast<unsigned int>(device) < g_devices.size()) {
    hip::setCurrentDevice(device);

    HIP_RETURN(hipSuccess);
  } else if (g_devices.empty()) {
    HIP_RETURN(hipErrorNoDevice);
  }
  HIP_RETURN(hipErrorInvalidDevice);
}

hipError_t hipSetDeviceFlags(unsigned int flags) {
  HIP_INIT_API(hipSetDeviceFlags, flags);
  if (g_devices.empty()) {
    HIP_RETURN(hipErrorNoDevice);
  }
  constexpr uint32_t supportedFlags =
      hipDeviceScheduleMask | hipDeviceMapHost | hipDeviceLmemResizeToMax;
  constexpr uint32_t mutualExclusiveFlags =
      hipDeviceScheduleSpin | hipDeviceScheduleYield | hipDeviceScheduleBlockingSync;
  // Only one scheduling flag allowed a time
  uint32_t scheduleFlag = flags & hipDeviceScheduleMask;

  if (((scheduleFlag & mutualExclusiveFlags) != hipDeviceScheduleSpin) &&
      ((scheduleFlag & mutualExclusiveFlags) != hipDeviceScheduleYield) &&
      ((scheduleFlag & mutualExclusiveFlags) != hipDeviceScheduleBlockingSync) &&
      ((scheduleFlag & mutualExclusiveFlags) != hipDeviceScheduleAuto)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (flags & ~supportedFlags) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Device* device = hip::getCurrentDevice()->devices()[0];
  switch (scheduleFlag) {
    case hipDeviceScheduleAuto:
      // Current behavior is different from the spec, due to MT usage in runtime
      if (hip::host_context->devices().size() >= std::thread::hardware_concurrency()) {
        device->SetActiveWait(false);
        break;
      }
      // Fall through for active wait...
    case hipDeviceScheduleSpin:
    case hipDeviceScheduleYield:
      // The both options falls into yield, because MT usage in runtime
      device->SetActiveWait(true);
      break;
    case hipDeviceScheduleBlockingSync:
      device->SetActiveWait(false);
      break;
    default:
      break;
  }
  hip::getCurrentDevice()->setFlags(flags & hipDeviceScheduleMask);

  HIP_RETURN(hipSuccess);
}

hipError_t hipSetValidDevices(int* device_arr, int len) {
  HIP_INIT_API(hipSetValidDevices, device_arr, len);
  // HIP runtime will go ahead with the default behavior of trying devices
  // from a default list sequentially, if the len passed is 0
  if (len == 0) {
    HIP_RETURN(hipSuccess);
  }
  int count = 0;
  HIP_RETURN_ONFAIL(ihipDeviceGetCount(&count));

  if (device_arr == nullptr || len < 0 || len > count) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  for (int i = 0; i < len; ++i) {
    if (device_arr[i] < 0 || device_arr[i] >= count) {
      HIP_RETURN(hipErrorInvalidDevice);
    }
  }

  if (tls.isSetDeviceCalled) {
    HIP_RETURN(hipSuccess);
  }
  tls.device_ = g_devices[device_arr[0]];
  uint32_t preferredNumaNode = (tls.device_)->devices()[0]->getPreferredNumaNode();
  amd::Os::setPreferredNumaNode(preferredNumaNode);
  HIP_RETURN(hipSuccess);
}
}  // namespace hip

extern "C" hipError_t hipChooseDevice(int* device, const hipDeviceProp_tR0000* properties) {
  return hip::hipChooseDeviceR0000(device, properties);
}
