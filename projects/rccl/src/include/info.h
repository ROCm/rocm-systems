/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include "nccl.h"
#include "collectives.h"
#include "core.h"
#include "utils.h"
#include "rccl_decision.h"

// Used to pass NCCL call information between functions
struct ncclInfo {
  ncclFunc_t coll;
  const char* opName;
  // NCCL Coll Args
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int root; // peer for p2p operations
  ncclComm_t comm;
  cudaStream_t stream;
  // Algorithm details
  int chunkSteps;
  int sliceSteps;
  const void* acc;

  // Optional per-operation metadata (e.g., rocSHMEM collectives, CE AlltoAllv).
  size_t* sizes;

  bool useDirect;
  // One-sided ops
  size_t peerWinOffset;
  ncclWindow_t peerWin;
  int sigIdx;
  int ctx;
  unsigned int flags;
  int nDesc;
  ncclWaitSignalDesc_t* signalDescs;
  // Implementation decision precomputed by ncclAllReduce_impl() /
  // ncclAllGather_impl() via rcclSelectAllReduce() / rcclSelectAllGather() and
  // consumed by taskAppend() so it does not recompute CE-vs-kernel or
  // graph-capture state. Valid only when decisionValid is true (false for
  // collectives that have not been wired yet, and the AllReduce WithBias path).
  struct rcclCollDecision decision;
  bool decisionValid;
};

#endif
