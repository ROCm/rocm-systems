/*
Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef ROCM_SMI_WRAP_H_
#define ROCM_SMI_WRAP_H_

#include <cstddef>
#include <cstdint>
#include "nccl.h"

#if defined(RCCL_SMI_ENABLED) && !defined(USE_AMDSMI)
#include "rocm_smi/rocm_smi.h"
#ifdef HAVE_ROCM_SMI64CONFIG
#include "rocm_smi/rocm_smi64Config.h"
#endif

ncclResult_t rocm_smi_init();
ncclResult_t rocm_smi_getNumDevice(uint32_t* num_devs);
ncclResult_t rocm_smi_getDevicePciBusIdString(uint32_t deviceIndex, char* pciBusId, size_t len);
ncclResult_t rocm_smi_getDeviceIndexByPciBusId(const char* pciBusId, uint32_t* deviceIndex);
ncclResult_t rocm_smi_getLinkInfo(int srcDev, int dstDev, RSMI_IO_LINK_TYPE* rsmi_type, int *hops, int *count);
#else
typedef int RSMI_IO_LINK_TYPE;
enum {
  RSMI_IOLINK_TYPE_PCIEXPRESS = 0,
  RSMI_IOLINK_TYPE_XGMI = 1,
};

inline ncclResult_t rocm_smi_init() { return ncclSuccess; }
inline ncclResult_t rocm_smi_getNumDevice(uint32_t* num_devs) { *num_devs = 0; return ncclSuccess; }
inline ncclResult_t rocm_smi_getDevicePciBusIdString(uint32_t deviceIndex, char* pciBusId, size_t len) { (void)deviceIndex; if (len > 0) pciBusId[0] = '\0'; return ncclSuccess; }
inline ncclResult_t rocm_smi_getDeviceIndexByPciBusId(const char* pciBusId, uint32_t* deviceIndex) { (void)pciBusId; *deviceIndex = 0; return ncclSuccess; }
inline ncclResult_t rocm_smi_getLinkInfo(int srcDev, int dstDev, RSMI_IO_LINK_TYPE* rsmi_type, int *hops, int *count) { (void)srcDev; (void)dstDev; (void)rsmi_type; *hops = 1; *count = 1; return ncclSuccess; }
#endif

#endif
