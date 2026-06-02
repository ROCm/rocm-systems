/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_BOOTSTRAP_H_
#define NCCL_BOOTSTRAP_H_

#include "nccl.h"
#include "comm.h"

struct ncclBootstrapHandle {
  uint64_t magic;
  union ncclSocketAddress addr;
  int nRanks; // number of existing ranks
};
static_assert(sizeof(struct ncclBootstrapHandle) <= sizeof(ncclUniqueId), "Bootstrap handle is too large to fit inside NCCL unique ID");

ncclResult_t bootstrapNetInit();
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv);
ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle, struct ncclComm* comm);
ncclResult_t bcastGrowHandle(struct ncclBootstrapHandle* handle, struct ncclComm* parent, bool isRoot);
ncclResult_t bootstrapInit(int nHandles, void* handle, struct ncclComm* comm, struct ncclComm* parent);
ncclResult_t bootstrapSplit(uint64_t magic, struct ncclComm* comm, struct ncclComm* parent, int color, int key, int* parentRanks);
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size);
ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size);
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size);
ncclResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag);
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size);
ncclResult_t bootstrapIntraNodeBarrier(void* commState, int *ranks, int rank, int nranks, int tag);
ncclResult_t bootstrapIntraNodeAllGather(void* commState, int *ranks, int rank, int nranks, void* allData, int size);
ncclResult_t bootstrapIntraNodeBroadcast(void* commState, int *ranks, int rank, int nranks, int root, void* bcastData, int size);
ncclResult_t bootstrapClose(void* commState);
ncclResult_t bootstrapAbort(void* commState);

// Bootstrap bidirectional AllGather gating. Exposed for unit tests; production
// callers go through bootstrapInit / bootstrapAllGather. Reads (and depends on
// the process-global cache of) NCCL_BOOTSTRAP_BIDIR_ALLGATHER,
// NCCL_BOOTSTRAP_BIDIR_NET, NCCL_BOOTSTRAP_BIDIR_THRESHOLD and
// NCCL_OOB_NET_ENABLE — same inputs as the dispatcher, so the test can verify
// the exact contract production uses.
//   nranks: total ranks in the comm (returns false unconditionally for < 3).
//   kind:   0 = socket OOB path, 1 = net (IB/OFI) OOB path.
// Marked visibility("hidden") so the symbol is linkable inside librccl.so for
// the test TU but never exported in the final shared library; see the matching
// attribute on the definition in src/bootstrap.cc.
__attribute__((visibility("hidden")))
bool bootstrapBidirEnabled(int nranks, int kind);

#endif
