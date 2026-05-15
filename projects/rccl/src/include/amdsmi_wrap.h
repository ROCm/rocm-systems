// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#ifndef AMDSMI_WRAP_H_
#define AMDSMI_WRAP_H_

#include <cstdint>
#include "amd_smi/amdsmi.h"
#include "nccl.h"

ncclResult_t amd_smi_init();
ncclResult_t amd_smi_shutdown();
ncclResult_t amd_smi_getNumDevice(uint32_t* num_devs);
ncclResult_t amd_smi_getDevicePciBusIdString(uint32_t deviceIndex, char* pciBusId, size_t len);
ncclResult_t amd_smi_getDeviceIndexByPciBusId(const char* pciBusId, uint32_t* deviceIndex);
ncclResult_t amd_smi_getLinkInfo(int srcDev, int dstDev, amdsmi_link_type_t* type, int *hops, int *count);

#endif
