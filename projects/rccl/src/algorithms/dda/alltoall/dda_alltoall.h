/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host entry points for the DDA all-to-all paths launched from ncclAllToAll.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_ALLTOALL_H_
#define DDA_ALLTOALL_H_

#include "nccl.h"

struct ncclComm;

/**
 * Check if DDA alltoall is eligible for the given parameters
 */
bool ncclAllToAllDdaIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                ncclDataType_t datatype);

/**
 * Execute DDA alltoall operation using IPC
 */
ncclResult_t ncclAllToAllDdaIpc(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                ncclComm* comm, cudaStream_t stream);

// Functor form of the entry point above. rcclAddonLaunch() takes a callable and
// brackets it with the launch contract, so every DDA all-to-all entry point that
// runs on the user's stream has one; see enqueue.h.
struct ncclAllToAllDdaIpcFn {
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclComm* comm;
  cudaStream_t stream;
  ncclResult_t operator()() const {
    return ncclAllToAllDdaIpc(sendbuff, recvbuff, count, datatype, comm, stream);
  }
};

/**
 * Check if DDA alltoall is eligible for the fabric/VMM path (runtime nRanks
 * up to kDdaMaxNranks, single- or multi-node within an MNNVL clique).
 */
bool ncclAllToAllDdaFabricEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                   ncclDataType_t datatype);

/**
 * Execute DDA alltoall operation using the fabric/VMM path
 */
ncclResult_t ncclAllToAllDdaFabric(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclComm* comm, cudaStream_t stream);

struct ncclAllToAllDdaFabricFn {
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclComm* comm;
  cudaStream_t stream;
  ncclResult_t operator()() const {
    return ncclAllToAllDdaFabric(sendbuff, recvbuff, count, datatype, comm, stream);
  }
};

// LL-protocol fabric path (small-chunk fast lane, 16B lines, no barrier).
bool ncclAllToAllDdaFabricLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                     ncclDataType_t datatype);

ncclResult_t ncclAllToAllDdaFabricLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                     ncclComm* comm, cudaStream_t stream);

struct ncclAllToAllDdaFabricLLFn {
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclComm* comm;
  cudaStream_t stream;
  ncclResult_t operator()() const {
    return ncclAllToAllDdaFabricLL(sendbuff, recvbuff, count, datatype, comm, stream);
  }
};

// LL128-protocol fabric path (mid-chunk fast lane, 128B lines, no barrier).
bool ncclAllToAllDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                        ncclDataType_t datatype);

ncclResult_t ncclAllToAllDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                        ncclComm* comm, cudaStream_t stream);

struct ncclAllToAllDdaFabricLL128Fn {
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclComm* comm;
  cudaStream_t stream;
  ncclResult_t operator()() const {
    return ncclAllToAllDdaFabricLL128(sendbuff, recvbuff, count, datatype, comm, stream);
  }
};

#endif
