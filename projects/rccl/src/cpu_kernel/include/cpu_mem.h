/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#ifndef RCCL_CPU_MEM_H_
#define RCCL_CPU_MEM_H_

#include "checks.h"

#include <cstddef>
#include <cstdint>

ncclResult_t rcclCpuCopyBytes(int cudaDev, void* dst, void const* src, size_t bytes);
ncclResult_t rcclCpuLoadDevU64(int cudaDev, uint64_t const* devPtr, uint64_t* out);
ncclResult_t rcclCpuStoreDevU64(int cudaDev, uint64_t* devPtr, uint64_t val);
ncclResult_t rcclCpuStoreDevU32(int cudaDev, uint32_t* devPtr, uint32_t val);

#endif
