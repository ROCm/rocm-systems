/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_GRAPH_ROME_TOPO_CONSENSUS_H_
#define RCCL_GRAPH_ROME_TOPO_CONSENSUS_H_

#include "nccl.h"
#include <functional>

/* Plurality vote on romeTopoModelIdx across ranks; all ranks must match the winner.
 * Callers supply getters so this stays independent of ncclComm / allGatherInfo. */
ncclResult_t rcclCheckRomeTopoModelIdxConsensus(
  int nranks,
  std::function<int(int /*rank*/)> getRomeTopoModelIdx,
  std::function<const char*(int /*rank*/)> getHostname,
  std::function<uint64_t(int /*rank*/)> getHostHash);

#endif
