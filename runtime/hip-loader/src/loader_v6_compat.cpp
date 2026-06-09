/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#define __HIP_PLATFORM_AMD__ 1
#define __HIP_DISABLE_CPP_FUNCTIONS__ 1
typedef unsigned int GLenum;
typedef unsigned int GLuint;
#include <hip/hip_runtime.h>
#include <hip/hip_gl_interop.h>
#include <hip/hip_deprecated.h>
#include <hip/amd_detail/hip_profiler_ext.h>
#include <hip/amd_detail/hip_api_trace.hpp>

#include "loader_v6_compat.h"

#include <cstring>

namespace {

void copy_name(char (&dst)[256], const char* src) {
  std::memset(dst, 0, sizeof(dst));
  if (src != nullptr) {
    std::strncpy(dst, src, sizeof(dst) - 1);
  }
}

hipMemsetParams to_current_memset_params(const HIP_MEMSET_NODE_PARAMS* old_params) {
  hipMemsetParams params = {};
  params.dst = old_params->dst;
  params.elementSize = old_params->elementSize;
  params.height = old_params->height;
  params.pitch = old_params->pitch;
  params.value = old_params->value;
  params.width = old_params->width;
  return params;
}

}  // namespace

hipError_t hip_loader_v6_compat_GetDevicePropertiesR0000(
    hip_loader_backend_GetDevicePropertiesR0600_fn backend_fn, hipDeviceProp_tR0000* prop,
    int device) {
  if (backend_fn == nullptr || prop == nullptr) {
    return hipErrorInvalidValue;
  }

  hipDeviceProp_tR0600 current = {};
  hipError_t status = backend_fn(&current, device);
  if (status != hipSuccess) {
    return status;
  }

  std::memset(prop, 0, sizeof(*prop));
  copy_name(prop->name, current.name);
  prop->totalGlobalMem = current.totalGlobalMem;
  prop->sharedMemPerBlock = current.sharedMemPerBlock;
  prop->regsPerBlock = current.regsPerBlock;
  prop->warpSize = current.warpSize;
  prop->maxThreadsPerBlock = current.maxThreadsPerBlock;
  prop->maxThreadsDim[0] = current.maxThreadsDim[0];
  prop->maxThreadsDim[1] = current.maxThreadsDim[1];
  prop->maxThreadsDim[2] = current.maxThreadsDim[2];
  prop->maxGridSize[0] = current.maxGridSize[0];
  prop->maxGridSize[1] = current.maxGridSize[1];
  prop->maxGridSize[2] = current.maxGridSize[2];
  prop->clockRate = current.clockRate;
  prop->memoryClockRate = current.memoryClockRate;
  prop->memoryBusWidth = current.memoryBusWidth;
  prop->totalConstMem = current.totalConstMem;
  prop->major = current.major;
  prop->minor = current.minor;
  prop->multiProcessorCount = current.multiProcessorCount;
  prop->l2CacheSize = current.l2CacheSize;
  prop->maxThreadsPerMultiProcessor = current.maxThreadsPerMultiProcessor;
  prop->computeMode = current.computeMode;
  prop->arch = current.arch;
  prop->concurrentKernels = current.concurrentKernels;
  prop->pciDomainID = current.pciDomainID;
  prop->pciBusID = current.pciBusID;
  prop->pciDeviceID = current.pciDeviceID;
  prop->maxSharedMemoryPerMultiProcessor = current.maxSharedMemoryPerMultiProcessor;
  prop->isMultiGpuBoard = current.isMultiGpuBoard;
  prop->canMapHostMemory = current.canMapHostMemory;
  copy_name(prop->gcnArchName, current.gcnArchName);
  prop->integrated = current.integrated;
  prop->cooperativeLaunch = current.cooperativeLaunch;
  prop->cooperativeMultiDeviceLaunch = current.cooperativeMultiDeviceLaunch;
  prop->maxTexture1DLinear = current.maxTexture1DLinear;
  prop->maxTexture1D = current.maxTexture1D;
  prop->maxTexture2D[0] = current.maxTexture2D[0];
  prop->maxTexture2D[1] = current.maxTexture2D[1];
  prop->maxTexture3D[0] = current.maxTexture3D[0];
  prop->maxTexture3D[1] = current.maxTexture3D[1];
  prop->maxTexture3D[2] = current.maxTexture3D[2];
  prop->hdpMemFlushCntl = current.hdpMemFlushCntl;
  prop->hdpRegFlushCntl = current.hdpRegFlushCntl;
  prop->memPitch = current.memPitch;
  prop->textureAlignment = current.textureAlignment;
  prop->texturePitchAlignment = current.texturePitchAlignment;
  prop->kernelExecTimeoutEnabled = current.kernelExecTimeoutEnabled;
  prop->ECCEnabled = current.ECCEnabled;
  prop->tccDriver = current.tccDriver;
  prop->cooperativeMultiDeviceUnmatchedFunc = current.cooperativeMultiDeviceUnmatchedFunc;
  prop->cooperativeMultiDeviceUnmatchedGridDim = current.cooperativeMultiDeviceUnmatchedGridDim;
  prop->cooperativeMultiDeviceUnmatchedBlockDim = current.cooperativeMultiDeviceUnmatchedBlockDim;
  prop->cooperativeMultiDeviceUnmatchedSharedMem = current.cooperativeMultiDeviceUnmatchedSharedMem;
  prop->isLargeBar = current.isLargeBar;
  prop->asicRevision = current.asicRevision;
  prop->managedMemory = current.managedMemory;
  prop->directManagedMemAccessFromHost = current.directManagedMemAccessFromHost;
  prop->concurrentManagedAccess = current.concurrentManagedAccess;
  prop->pageableMemoryAccess = current.pageableMemoryAccess;
  prop->pageableMemoryAccessUsesHostPageTables = current.pageableMemoryAccessUsesHostPageTables;
  return hipSuccess;
}

hipError_t hip_loader_v6_compat_ChooseDeviceR0000(
    hip_loader_backend_ChooseDeviceR0600_fn backend_fn, int* device,
    const hipDeviceProp_tR0000* prop) {
  if (backend_fn == nullptr || device == nullptr || prop == nullptr) {
    return hipErrorInvalidValue;
  }

  hipDeviceProp_tR0600 current = {};
  copy_name(current.name, prop->name);
  current.totalGlobalMem = prop->totalGlobalMem;
  current.sharedMemPerBlock = prop->sharedMemPerBlock;
  current.regsPerBlock = prop->regsPerBlock;
  current.warpSize = prop->warpSize;
  current.maxThreadsPerBlock = prop->maxThreadsPerBlock;
  current.maxThreadsDim[0] = prop->maxThreadsDim[0];
  current.maxThreadsDim[1] = prop->maxThreadsDim[1];
  current.maxThreadsDim[2] = prop->maxThreadsDim[2];
  current.maxGridSize[0] = prop->maxGridSize[0];
  current.maxGridSize[1] = prop->maxGridSize[1];
  current.maxGridSize[2] = prop->maxGridSize[2];
  current.clockRate = prop->clockRate;
  current.memoryClockRate = prop->memoryClockRate;
  current.memoryBusWidth = prop->memoryBusWidth;
  current.totalConstMem = prop->totalConstMem;
  current.major = prop->major;
  current.minor = prop->minor;
  current.multiProcessorCount = prop->multiProcessorCount;
  current.l2CacheSize = prop->l2CacheSize;
  current.maxThreadsPerMultiProcessor = prop->maxThreadsPerMultiProcessor;
  current.computeMode = prop->computeMode;
  current.arch = prop->arch;
  current.concurrentKernels = prop->concurrentKernels;
  current.pciDomainID = prop->pciDomainID;
  current.pciBusID = prop->pciBusID;
  current.pciDeviceID = prop->pciDeviceID;
  current.maxSharedMemoryPerMultiProcessor = prop->maxSharedMemoryPerMultiProcessor;
  current.isMultiGpuBoard = prop->isMultiGpuBoard;
  current.canMapHostMemory = prop->canMapHostMemory;
  copy_name(current.gcnArchName, prop->gcnArchName);
  current.integrated = prop->integrated;
  current.cooperativeLaunch = prop->cooperativeLaunch;
  current.cooperativeMultiDeviceLaunch = prop->cooperativeMultiDeviceLaunch;
  current.maxTexture1DLinear = prop->maxTexture1DLinear;
  current.maxTexture1D = prop->maxTexture1D;
  current.maxTexture2D[0] = prop->maxTexture2D[0];
  current.maxTexture2D[1] = prop->maxTexture2D[1];
  current.maxTexture3D[0] = prop->maxTexture3D[0];
  current.maxTexture3D[1] = prop->maxTexture3D[1];
  current.maxTexture3D[2] = prop->maxTexture3D[2];
  current.hdpMemFlushCntl = prop->hdpMemFlushCntl;
  current.hdpRegFlushCntl = prop->hdpRegFlushCntl;
  current.memPitch = prop->memPitch;
  current.textureAlignment = prop->textureAlignment;
  current.texturePitchAlignment = prop->texturePitchAlignment;
  current.kernelExecTimeoutEnabled = prop->kernelExecTimeoutEnabled;
  current.ECCEnabled = prop->ECCEnabled;
  current.tccDriver = prop->tccDriver;
  current.cooperativeMultiDeviceUnmatchedFunc = prop->cooperativeMultiDeviceUnmatchedFunc;
  current.cooperativeMultiDeviceUnmatchedGridDim = prop->cooperativeMultiDeviceUnmatchedGridDim;
  current.cooperativeMultiDeviceUnmatchedBlockDim = prop->cooperativeMultiDeviceUnmatchedBlockDim;
  current.cooperativeMultiDeviceUnmatchedSharedMem = prop->cooperativeMultiDeviceUnmatchedSharedMem;
  current.isLargeBar = prop->isLargeBar;
  current.asicRevision = prop->asicRevision;
  current.managedMemory = prop->managedMemory;
  current.directManagedMemAccessFromHost = prop->directManagedMemAccessFromHost;
  current.concurrentManagedAccess = prop->concurrentManagedAccess;
  current.pageableMemoryAccess = prop->pageableMemoryAccess;
  current.pageableMemoryAccessUsesHostPageTables = prop->pageableMemoryAccessUsesHostPageTables;
  return backend_fn(device, &current);
}

hipError_t hip_loader_v6_compat_DrvGraphAddMemsetNode(
    hip_loader_backend_DrvGraphAddMemsetNode_fn backend_fn, hipGraphNode_t* phGraphNode,
    hipGraph_t hGraph, const hipGraphNode_t* dependencies, size_t numDependencies,
    const HIP_MEMSET_NODE_PARAMS* memsetParams, hipCtx_t ctx) {
  if (backend_fn == nullptr || memsetParams == nullptr) {
    return hipErrorInvalidValue;
  }
  hipMemsetParams current = to_current_memset_params(memsetParams);
  return backend_fn(phGraphNode, hGraph, dependencies, numDependencies, &current, ctx);
}

hipError_t hip_loader_v6_compat_DrvGraphExecMemsetNodeSetParams(
    hip_loader_backend_DrvGraphExecMemsetNodeSetParams_fn backend_fn, hipGraphExec_t hGraphExec,
    hipGraphNode_t hNode, const HIP_MEMSET_NODE_PARAMS* memsetParams, hipCtx_t ctx) {
  if (backend_fn == nullptr || memsetParams == nullptr) {
    return hipErrorInvalidValue;
  }
  hipMemsetParams current = to_current_memset_params(memsetParams);
  return backend_fn(hGraphExec, hNode, &current, ctx);
}
