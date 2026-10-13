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
static_assert(sizeof(struct ncclBootstrapHandle) <= sizeof(ncclUniqueId),
              "Bootstrap handle is too large to fit inside NCCL unique ID");

ncclResult_t bootstrapNetInit();
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv);
ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle, struct ncclComm* comm);
ncclResult_t bcastGrowHandle(struct ncclBootstrapHandle* handle, struct ncclComm* parent, bool isRoot);
ncclResult_t bootstrapInit(int nHandles, void* handle, struct ncclComm* comm, struct ncclComm* parent);
ncclResult_t bootstrapSplit(uint64_t magic, struct ncclComm* comm, struct ncclComm* parent, int color, int key,
                            int* parentRanks);
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size);
ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size);
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size);
// [RCCL] As bootstrapRecv, but gives up with ncclRemoteError if no peer connects by deadlineNs (clockNano clock).
// 0 waits indefinitely.
ncclResult_t bootstrapRecvDeadline(void* commState, int peer, int tag, void* data, int size, int64_t deadlineNs);
ncclResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag);
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size);
ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag);
// [RCCL] As bootstrapIntraNodeBarrier, but returns ncclRemoteError instead of blocking forever when a host-local
// peer never arrives. timeoutMsec <= 0 restores the unbounded wait.
ncclResult_t bootstrapIntraNodeBarrierTimeout(void* commState, int* ranks, int rank, int nranks, int tag,
                                              int64_t timeoutMsec);
ncclResult_t bootstrapIntraNodeAllGather(void* commState, int* ranks, int rank, int nranks, void* allData, int size);
ncclResult_t bootstrapIntraNodeBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData,
                                         int size);
ncclResult_t bootstrapClose(void* commState);

// Bootstrap bidirectional AllGather gating. Exposed for unit tests; production
// callers go through bootstrapInit / bootstrapAllGather. Reads (and depends on
// the process-global cache of) NCCL_BOOTSTRAP_BIDIR_ALLGATHER,
// NCCL_BOOTSTRAP_BIDIR_NET, NCCL_BOOTSTRAP_BIDIR_THRESHOLD and
// NCCL_OOB_NET_ENABLE — same inputs as the dispatcher, so the test can verify
// the exact contract production uses.
//   nranks: total ranks in the comm (returns false unconditionally for < 3).
//   kind:   0 = socket OOB path, 1 = net (IB/OFI) OOB path.
// Visibility is left to the global -fvisibility flag: hidden (not exported) in
// production and Release builds, default (exported, so rccl-UnitTestsFixturesDebug
// can link it) in BUILD_TESTS + Debug builds. See the note on the definition in
// src/bootstrap.cc — an explicit visibility("hidden") attribute here would override
// that flag and break the Debug test link.
bool bootstrapBidirEnabled(int nranks, int kind);

#endif
