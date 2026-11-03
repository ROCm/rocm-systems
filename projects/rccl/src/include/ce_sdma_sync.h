/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * SDMA-gated synchronization for the hierarchical CE all-gather: the collective's waits, copies and flag
 * updates run as SDMA packet chains armed by a host function and released by one CP wait.
 * Gates are 64-bit counters polled on their low 32 bits, so call numbers stay below 2^32.
 ************************************************************************/
#ifndef NCCL_CE_SDMA_SYNC_H_
#define NCCL_CE_SDMA_SYNC_H_

#include <stddef.h>
#include <stdint.h>

#include "nccl.h"

struct ncclComm;
struct ncclDevrWindow;
struct ncclCeSdmaPlan;  // the chains of one all-gather

#include <hip/hip_runtime_api.h>

// Fails with a WARN unless GPU_STREAMOPS_CP_WAIT=1, the GPU is gfx90a/gfx942/gfx950 and every LSA peer is reachable.
ncclResult_t ncclCeSdmaInit(struct ncclComm* comm);
void ncclCeSdmaDestroy(struct ncclComm* comm);  // idempotent
bool ncclCeSdmaActive(struct ncclComm* comm);   // false once the 32-bit gate arithmetic has no headroom
size_t ncclCeSdmaGateBytes(int lsaSize);        // gate counters to reserve in the CE sync window

ncclResult_t ncclCeSdmaPlanNew(struct ncclComm* comm, struct ncclCeSdmaPlan** out);
void ncclCeSdmaPlanFree(struct ncclCeSdmaPlan* plan);  // for a plan that was not passed to Finish

// Records a stream memory op: a wait on the control queue (with fanOut, on every peer queue) or a write on the control
// queue.
void ncclCeSdmaPlanWait(struct ncclCeSdmaPlan* plan, const void* counter64, uint64_t expected, bool fanOut);
void ncclCeSdmaPlanWrite(struct ncclCeSdmaPlan* plan, void* dst, uint64_t value);

ncclResult_t ncclCeSdmaAnnounceReady(struct ncclCeSdmaPlan* plan); // tell the LSA peers this output buffer is free
// Copies src into this rank's slot at 'offset' of recvWin and, once each peer is ready, into the peer's.
ncclResult_t ncclCeSdmaBroadcast(struct ncclCeSdmaPlan* plan, struct ncclDevrWindow* recvWin, size_t offset,
                                 const void* src, size_t bytes);
// Copies src into every peer's recvWin at 'offset'.
ncclResult_t ncclCeSdmaScatter(struct ncclCeSdmaPlan* plan, struct ncclDevrWindow* recvWin, size_t offset,
                               const void* src, size_t bytes);
// Waits for all copies, then arms the chains from a host function and waits for the release on the stream. Takes
// ownership of plan.
ncclResult_t ncclCeSdmaFinish(struct ncclCeSdmaPlan* plan, hipStream_t stream);

#endif // NCCL_CE_SDMA_SYNC_H_
