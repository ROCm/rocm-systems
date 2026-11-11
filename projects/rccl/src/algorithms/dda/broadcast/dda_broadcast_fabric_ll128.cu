/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA broadcast.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/broadcast/dda_broadcast.h"

#include "debug.h"

// Not implemented yet: never eligible.
bool ncclBroadcastDdaFabricLL128Eligible(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  (void)comm;
  (void)count;
  (void)datatype;
  return false;
}

uint32_t ncclBroadcastDdaFabricLL128Blocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  (void)comm;
  (void)count;
  (void)datatype;
  return 0;
}

ncclResult_t ncclBroadcastDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                         int root, ncclComm* comm, cudaStream_t stream) {
  (void)sendbuff;
  (void)recvbuff;
  (void)count;
  (void)datatype;
  (void)root;
  (void)comm;
  (void)stream;
  WARN("DDA broadcast LL128: not implemented");
  return ncclInvalidUsage;
}
