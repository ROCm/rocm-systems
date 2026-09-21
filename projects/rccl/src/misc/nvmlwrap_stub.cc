/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nvmlwrap.h"

ncclResult_t ncclNvmlSymbols(void) {
  return ncclSuccess;
}

ncclResult_t ncclNvmlInit(void) {
  return ncclSuccess;
}

ncclResult_t ncclNvmlShutdown(void) {
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device) {
  return ncclSystemError;
}

ncclResult_t ncclNvmlDeviceGetIndex(nvmlDevice_t device, unsigned* index) {
  *index = 0;
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t* pci) {
  return ncclSystemError;
}

// NCCL 2.31 RAS GPU diagnostics query NVML directly; there is no NVML on ROCm,
// so report no devices and let the callers skip those checks.
int ncclNvmlDeviceCount = 0;

ncclResult_t ncclNvmlDeviceGetCount(unsigned int* deviceCount) {
  *deviceCount = 0;
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) {
  return ncclSystemError;
}

ncclResult_t ncclNvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length) {
  if (length > 0) name[0] = '\0';
  return ncclSystemError;
}

ncclResult_t ncclNvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType,
                                                 nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType,
                                                 unsigned long long* count) {
  *count = 0;
  return ncclSystemError;
}

ncclResult_t ncclNvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int* minorNumber) {
  *minorNumber = 0;
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t* isActive) {
  return ncclSystemError;
}

ncclResult_t ncclNvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci) {
  return ncclSystemError;
}

ncclResult_t ncclNvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link,
                                               nvmlNvLinkCapability_t capability, unsigned int* capResult) {
  return ncclSystemError;
}

ncclResult_t ncclNvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor) {
  *major = *minor = 1;
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int* currLinkGen) {
  return ncclSystemError;
}

ncclResult_t ncclNvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int* currLinkWidth) {
  return ncclSystemError;
}
