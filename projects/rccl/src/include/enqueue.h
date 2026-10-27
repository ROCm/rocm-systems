/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_ENQUEUE_H_
#define NCCL_ENQUEUE_H_

#include "comm.h"
#include "info.h"
#include "group.h"
#include "collectives.h"
#include "utils.h"
#include "enqueue/raw_task.h"
#include "enqueue/task_pretuning.h"
#include "enqueue/task_classify.h"
#include "enqueue/task_posttuning.h"
#include "enqueue/task_sched.h"
#include "enqueue/mgmt_task_enq.h"

#define NCCL_LL_ALIGNMENT_PER_THREAD sizeof(uint64_t)
#define NCCL_LL128_ALIGNMENT_PER_WARP 480
#define NCCL_SIMPLE_ALIGNMENT (WARP_SIZE * 8LL * 16LL)
#define NCCL_BYTES_ALIGNMENT 16

int64_t ncclParamGraphStreamOrdering();
int64_t ncclParamEnqueueRearchEnable();
int64_t ncclParamAllgathervEnable();
int64_t ncclParamP2pLLThreshold();
int64_t ncclParamChunkSize();
int64_t ncclParamLaunchOrderImplicit();

ncclResult_t ncclGroupJobLaunch(struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next>* asyncJobsMain,
                                volatile bool* groupAbortFlag);

ncclResult_t ncclTaskPreTuning(struct ncclComm* comm, struct ncclRawTaskQueue* rtq,
                               struct ncclTaskTuningInfoQueue* tiq);
ncclResult_t ncclTaskPrepare(struct ncclComm* comm, ncclSimInfo_t* simInfo);

ncclResult_t ncclInitKernelsForDevice(int cudaArch, int maxSharedMem, size_t* maxStackSize);
ncclResult_t ncclEnqueueCheck(struct ncclInfo* info);
ncclResult_t ncclPlannerSetCapturingGraph(struct ncclComm* comm, struct ncclInfo* info);
ncclResult_t ncclLaunchPrepare(struct ncclComm* comm);
ncclResult_t ncclLaunchKernelBefore_NoUncapturedCuda(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchKernelAfter_NoCuda(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchFinish(struct ncclComm* comm);
ncclResult_t ncclValidateCollConfigLaunchCompletionEvents(struct ncclComm* comm);

// Addon backends launch onto the user's stream themselves instead of going through doLaunches, which is what
// makes comm->cudaDev current and installs the cross-stream dependency. Bracket such a launch with these.
//
// The prologue also offers comm->doneEvent on the communicator, for the launch's last kernel to carry as its
// stopEvent via rcclTakeAddonStopEvent() instead of paying for a standalone record. A launch that never takes
// it, because its last stream operation is not a kernel or because it returned early, gets the record from the
// epilogue.
struct rcclAddonLaunchState {
  int savedDev;
  // Whether the prologue offered the stop event, which is what lets the epilogue tell a taken event from one
  // that was never on offer. False while capturing, where a fused stop event is not bound.
  bool eventOffered;
  bool capturing;
  struct ncclCudaGraph graph;
  // Set once sharedRes->deviceStream is acquired; the epilogue must release it on every path.
  bool deviceStreamAcquired;
  cudaStream_t deviceStream;
};

ncclResult_t rcclAddonLaunchBegin(struct ncclComm* comm, cudaStream_t stream, struct rcclAddonLaunchState* state);
ncclResult_t rcclAddonLaunchEnd(struct ncclComm* comm, cudaStream_t stream,
                                const struct rcclAddonLaunchState& state, ncclResult_t launchRes);

template <typename LaunchFn>
inline ncclResult_t rcclAddonLaunch(struct ncclComm* comm, cudaStream_t stream, LaunchFn&& launch) {
  struct rcclAddonLaunchState state = {-1, false};
  ncclResult_t result = rcclAddonLaunchBegin(comm, stream, &state);
  if (result != ncclSuccess) return rcclAddonLaunchEnd(comm, stream, state, result);
  return rcclAddonLaunchEnd(comm, stream, state, launch());
}

// An addon collective that returns before ncclEnqueueCheck must still print its per-collective COLL line and
// advance comm->opCount on the user's comm. Hierarchical addons use this too: their sub-collectives run on the
// separate intra/inter comms (and log there under their own names), so the parent comm is counted here exactly
// once, matching the native path.
void rcclLogCollInfo(const struct ncclInfo& info);

template <typename LaunchFn>
inline ncclResult_t rcclAddonLaunch(const struct ncclInfo& info, LaunchFn&& launch) {
  rcclLogCollInfo(info);
  ncclResult_t result = rcclAddonLaunch(info.comm, info.stream, std::forward<LaunchFn>(launch));
  if (result == ncclSuccess) info.comm->opCount++;
  return result;
}

ncclResult_t ncclPrepareTasks(struct ncclComm* comm, bool* algoNeedConnect, bool* needConnect, ncclSimInfo_t* simInfo);
ncclResult_t ncclTasksRegAndEnqueue(struct ncclComm* comm);

// Defined via NCCL_PARAM in enqueue.cc.
int64_t ncclParamLaunchOrderImplicit();

static inline size_t ncclFuncSendCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncReduceScatter ? nRanks * count : count;
}
static inline size_t ncclFuncRecvCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncAllGather ? nRanks * count : count;
}
rccl_static inline size_t ncclFuncMaxSendRecvCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncAllGather || func == ncclFuncReduceScatter ? nRanks * count : count;
}
// Match the public AllGather in-place convention: each rank's input aliases
// its own slice of the aggregate receive buffer.
static inline bool ncclAllGatherIsInPlace(const void* sendbuff, const void* recvbuff, int rank, size_t perRankBytes) {
  return (uintptr_t)sendbuff == (uintptr_t)recvbuff + size_t(rank) * perRankBytes;
}

// [RCCL] Whether the Tma symmetric kernels may run: input and output must sit at the same offset
// modulo 16 within their windows. The answer must be the same on every rank, or ranks pick
// different symmetric kernels and launch grids that never meet at the LSA barrier. Both windows
// must therefore be symmetric: an unregistered buffer's raw address differs by rank. In-place
// AllGather and ReduceScatter shift one side by rank * perRankBytes, so there the per-rank slice
// must itself be 16 B aligned.
static inline bool ncclSymAligned16B(ncclFunc_t func, int rank, const void* sendbuff, const void* recvbuff,
                                     struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin,
                                     size_t perRankBytes) {
  if (sendWin == nullptr || recvWin == nullptr || !(sendWin->winFlags & NCCL_WIN_COLL_SYMMETRIC) ||
      !(recvWin->winFlags & NCCL_WIN_COLL_SYMMETRIC)) {
    return false;
  }
  size_t inputOff = (uintptr_t)sendbuff - (uintptr_t)sendWin->userPtr;
  size_t outputOff = (uintptr_t)recvbuff - (uintptr_t)recvWin->userPtr;
  if (uint32_t(inputOff - outputOff) % 16 != 0) return false;
  bool inPlace = (func == ncclFuncAllGather && ncclAllGatherIsInPlace(sendbuff, recvbuff, rank, perRankBytes)) ||
                 (func == ncclFuncReduceScatter && ncclAllGatherIsInPlace(recvbuff, sendbuff, rank, perRankBytes));
  return !inPlace || perRankBytes % 16 == 0;
}

ncclResult_t ncclGetCollNetSupport(struct ncclComm* comm, struct ncclTaskColl* task, int* collNetSupport);
ncclResult_t ncclGetAlgoInfo(struct ncclComm* comm, struct ncclTaskColl* task, int collNetSupport, int nvlsSupport,
                             int numPipeOps, ncclSimInfo_t* simInfo = NULL);
bool ncclTestBudget(struct ncclKernelPlanBudget* budget, int nWorkBatches, ssize_t nWorkBytes);

void ncclAddWorkBatchToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan, int channelId,
                            enum ncclDevWorkType workType, int devFuncId, int progressSlot, uint32_t workOffset,
                            int p2pEpoch = -1, int p2pRound = -1, bool newBatch = false);

// RCCL-only shim: sets plan->kernelFn / kernelSpecialized using the file-local
// ncclKerns table in enqueue.cc, for upstream-style schedulers (e.g.
// scheduler/allgatherv_sched.cc) that would otherwise reference the absent
// ncclDevKernelForFunc[] / ncclDevKernelForFuncIsSpecialized[] arrays.
void ncclPlanSetDefaultKernel(struct ncclComm* comm, struct ncclKernelPlan* plan);

ncclResult_t ncclAddProxyOpIfNeeded(struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclProxyOp* op);

ncclResult_t ncclGetRegBuff(struct ncclComm* comm, struct ncclTaskColl* info, int* regBuff);

// Reports the Send/Recv protocol and channel count used by Direct AllToAll's
// per-peer P2P tasks. This is intended for post-launch reporting: it consults
// the connected P2P path to the rank's first non-self peer.
ncclResult_t rcclGetAlltoAllP2pInfo(struct ncclComm* comm, size_t count, ncclDataType_t datatype, int* protocol,
                                    int* nChannels);

#endif // End include guard
