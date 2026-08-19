/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host entry points for the GIN-SDMA all-to-all path launched from ncclAlltoAll.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef GIN_ALLTOALL_H_
#define GIN_ALLTOALL_H_

#include "nccl.h"
#include "nccl_device.h"

struct ncclComm;

// Upper bound on CTAs, and therefore on both the LSA barrier count and the
// number of GIN signals reserved for this path.
constexpr int kGinA2AMaxCtas = 16;
constexpr int kGinA2AThreadsPerCta = 256;

// Lazily created on the first eligible alltoall and torn down with the comm. The
// devComm is private so it does not disturb the symmetric kernels. Declared
// unconditionally, or rcclras and librccl disagree on the layout of ncclComm.
struct ncclGinA2AState {
  bool initialized;
  struct ncclDevComm devComm;
};

#if defined(ENABLE_ROCSHMEM_GIN)

bool ncclAllToAllGinSdmaEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                 ncclDataType_t datatype);

ncclResult_t ncclAllToAllGinSdma(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                 ncclComm* comm, cudaStream_t stream);

ncclResult_t ncclGinA2AFinalize(ncclComm* comm);

#else

// gin_alltoall_sdma.cu is only compiled with the GIN backend, so teardown has
// nothing to release here.
inline ncclResult_t ncclGinA2AFinalize(ncclComm*) { return ncclSuccess; }

#endif

#endif
