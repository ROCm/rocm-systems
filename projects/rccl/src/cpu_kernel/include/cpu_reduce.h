/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#ifndef RCCL_CPU_REDUCE_H_
#define RCCL_CPU_REDUCE_H_

#include "device.h"

#include <cstddef>
#include <cstdint>

void rcclCpuReduceCopy(
    int tid, int tn,
    ncclDataType_t dtype, ncclDevRedOp_t redop,
    void const* const* srcs, int nSrcs,
    void* const* dsts, int nDsts,
    size_t nElts, uint64_t redOpArg, bool postOp);

void rcclCpuMemcpy(
    int tid, int tn,
    void const* src, void* dst, size_t bytes);

#endif
