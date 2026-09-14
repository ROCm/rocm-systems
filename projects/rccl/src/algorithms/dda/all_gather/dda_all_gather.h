/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host entry points for the DDA all-gather paths launched from ncclAllGather.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_ALL_GATHER_H_
#define DDA_ALL_GATHER_H_

#include "nccl.h"

#include <cstdint>

struct ncclComm;

/**
 * Check if DDA allgather is eligible for the given parameters
 */
bool ncclAllGatherDdaIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                 ncclDataType_t datatype);

/**
 * Execute DDA allgather operation using IPC
 */
ncclResult_t ncclAllGatherDdaIpc(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                 ncclComm* comm, cudaStream_t stream);

// Functor form of the entry point above. rcclAddonLaunch() takes a callable and
// brackets it with the launch contract, so every DDA all-gather entry point that
// runs on the user's stream has one; see enqueue.h.
struct ncclAllGatherDdaIpcFn {
  const void* sendbuff;
  void* recvbuff;
  size_t sendcount;
  ncclDataType_t datatype;
  ncclComm* comm;
  cudaStream_t stream;
  ncclResult_t operator()() const {
    return ncclAllGatherDdaIpc(sendbuff, recvbuff, sendcount, datatype, comm, stream);
  }
};

// Total CTAs (grid blocks) each DDA allgather launcher would use for the given
// operands. Mirrors the launch grid math so reporting reflects real occupancy.
uint32_t ncclAllGatherDdaIpcBlocks(ncclComm* comm, size_t sendcount, ncclDataType_t datatype);
uint32_t ncclAllGatherDdaFabricBlocks(ncclComm* comm, size_t sendcount, ncclDataType_t datatype);
uint32_t ncclAllGatherDdaFabricLLBlocks(ncclComm* comm, size_t sendcount, ncclDataType_t datatype);
uint32_t ncclAllGatherDdaFabricLL128Blocks(ncclComm* comm, size_t sendcount, ncclDataType_t datatype);

/**
 * Check if DDA allgather is eligible for the fabric/VMM path (runtime nRanks
 * up to kDdaMaxNranks, single- or multi-node within an MNNVL clique).
 */
bool ncclAllGatherDdaFabricEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                    ncclDataType_t datatype);

/**
 * Execute DDA allgather operation using the fabric/VMM path
 */
ncclResult_t ncclAllGatherDdaFabric(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                    ncclComm* comm, cudaStream_t stream);

struct ncclAllGatherDdaFabricFn {
  const void* sendbuff;
  void* recvbuff;
  size_t sendcount;
  ncclDataType_t datatype;
  ncclComm* comm;
  cudaStream_t stream;
  ncclResult_t operator()() const {
    return ncclAllGatherDdaFabric(sendbuff, recvbuff, sendcount, datatype, comm, stream);
  }
};

/**
 * Check if the LL-protocol DDA allgather is eligible for the fabric/VMM path.
 */
bool ncclAllGatherDdaFabricLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                      ncclDataType_t datatype);

/**
 * Execute the LL-protocol DDA allgather using the fabric/VMM path.
 */
ncclResult_t ncclAllGatherDdaFabricLL(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                      ncclComm* comm, cudaStream_t stream);

struct ncclAllGatherDdaFabricLLFn {
  const void* sendbuff;
  void* recvbuff;
  size_t sendcount;
  ncclDataType_t datatype;
  ncclComm* comm;
  cudaStream_t stream;
  ncclResult_t operator()() const {
    return ncclAllGatherDdaFabricLL(sendbuff, recvbuff, sendcount, datatype, comm, stream);
  }
};

/**
 * Check if the LL128-protocol DDA allgather is eligible for the fabric/VMM path.
 */
bool ncclAllGatherDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                         ncclDataType_t datatype);

/**
 * Execute the LL128-protocol DDA allgather using the fabric/VMM path.
 */
ncclResult_t ncclAllGatherDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t sendcount,
                                         ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream);

struct ncclAllGatherDdaFabricLL128Fn {
  const void* sendbuff;
  void* recvbuff;
  size_t sendcount;
  ncclDataType_t datatype;
  ncclComm* comm;
  cudaStream_t stream;
  ncclResult_t operator()() const {
    return ncclAllGatherDdaFabricLL128(sendbuff, recvbuff, sendcount, datatype, comm, stream);
  }
};

#endif
