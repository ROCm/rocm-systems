/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host entry points for the GIN-SDMA all-gather path launched from ncclAllGather.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef GIN_ALL_GATHER_H_
#define GIN_ALL_GATHER_H_

#include "nccl.h"
#include "nccl_device.h"

struct ncclComm;

// Upper bound on CTAs, and so on the LSA barriers and GIN signals reserved here.
constexpr int kGinAllGatherMaxCtas = 64;

// Lazily created on the first eligible allgather and torn down with the comm.
// Declared unconditionally, or rcclras and librccl disagree on ncclComm layout.
struct ncclGinAllGatherState {
  bool initialized;
  struct ncclDevComm devComm;
};

#if defined(ENABLE_ROCSHMEM_GIN)

bool ncclAllGatherGinSdmaEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                  ncclDataType_t datatype);

ncclResult_t ncclAllGatherGinSdma(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                  ncclComm* comm, cudaStream_t stream);

ncclResult_t ncclGinAllGatherFinalize(ncclComm* comm);

#else

inline ncclResult_t ncclGinAllGatherFinalize(ncclComm*) { return ncclSuccess; }

#endif

#endif
