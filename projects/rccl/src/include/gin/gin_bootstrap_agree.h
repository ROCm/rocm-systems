/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// One bootstrap all-gather + caller-supplied reduction, shared by the GIN
// Anvil-SDMA setup paths that have to agree a decision across every rank.
// Internal header: not installed, not part of any public surface.

#ifndef NCCL_GIN_BOOTSTRAP_AGREE_H_
#define NCCL_GIN_BOOTSTRAP_AGREE_H_

#ifdef __cplusplus

#include <cstdlib>

#include "alloc.h"
#include "bootstrap.h"
#include "checks.h"
#include "comm.h"
#include "nccl.h"

// Gathers one T per rank and hands the whole array to reduce().
//
// The buffer is always sized to comm->nRanks and the local slot is comm->rank:
// bootstrapAllGather indexes by the communicator rank, so a GIN rail team size
// (cctx->nranks) must never be substituted here.
//
// Returns the allgather's result. reduce() runs only when the allgather
// succeeded, so a caller that reduces into an out-param must treat a non-
// ncclSuccess return as "no decision", not as a negative vote.
template <typename T, typename ReduceFn>
static inline ncclResult_t ncclGinBootstrapAgree(struct ncclComm* comm, T const& local, ReduceFn reduce) {
  if (comm == nullptr || comm->nRanks < 1) return ncclInternalError;
  const int nRanks = comm->nRanks;
  const int rank = comm->rank;
  if (rank < 0 || rank >= nRanks) return ncclInternalError;

  T* slots = nullptr;
  NCCLCHECK(ncclCalloc(&slots, nRanks));
  slots[rank] = local;
  ncclResult_t ret = bootstrapAllGather(comm->bootstrap, slots, sizeof(T));
  if (ret == ncclSuccess) reduce(static_cast<T const*>(slots), nRanks);
  free(slots);
  return ret;
}

// AND-reduces one int flag per rank. *allOk is written only on ncclSuccess.
static inline ncclResult_t ncclGinBootstrapAgreeAll(struct ncclComm* comm, int localOk, int* allOk) {
  int agreed = 1;
  ncclResult_t ret = ncclGinBootstrapAgree(comm, localOk ? 1 : 0, [&agreed](int const* votes, int n) {
    for (int i = 0; i < n; i++) {
      if (!votes[i]) agreed = 0;
    }
  });
  if (ret != ncclSuccess) return ret;
  if (allOk) *allOk = agreed;
  return ncclSuccess;
}

#endif // __cplusplus

#endif // NCCL_GIN_BOOTSTRAP_AGREE_H_
