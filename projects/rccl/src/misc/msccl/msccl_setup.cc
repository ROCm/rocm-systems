/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include "channel.h"
#include "checks.h"
#include "collectives.h"
#include "proxy.h"
#include "transport.h"

#include "msccl/msccl_lifecycle.h"
#ifdef COMPILE_MSCCL_KERNEL
#include "msccl/msccl_kernel.h"
#endif
#include "msccl/msccl_setup.h"
#include "msccl/msccl_status.h"

RCCL_PARAM(MscclWorkFifoDepth, "MSCCL_WORK_FIFO_DEPTH", 256<<10);

static inline size_t computeSizeNeeded(size_t nBytes, int nScratchChunks, int nChunksPerLoop) {
  return (nBytes * (size_t)nScratchChunks) / (size_t)nChunksPerLoop;
}

ncclResult_t mscclGetCaptureStatus(const ncclComm_t comm, hipStream_t stream) {
  mscclStatus& status = mscclGetStatus(comm);
  mscclThreadLocalStatus& threadLocalStatus = mscclGetThreadLocalStatus();
  mscclSavedProxyArgs& savedProxyArgs = mscclGetSavedProxyArgs(comm);
  cudaStreamCaptureStatus captureStatus;
  unsigned long long captureId;
  CUDACHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus, &captureId, &threadLocalStatus.graph, nullptr, nullptr));
  if (captureStatus == cudaStreamCaptureStatusActive) {
    if (savedProxyArgs.count(captureId) == 0) {
      threadLocalStatus.captureStatus = mscclNewCapture;
      savedProxyArgs[captureId] = std::vector<struct mscclProxyArg>();
      NCCLCHECK(mscclInitWorkFifoStatus(&(status.graphWorkFifoStatus[captureId])));
    } else {
      INFO(NCCL_NET,"mscclGetCaptureStatus: captureId %llu is same with the previous one\n", captureId);
      threadLocalStatus.captureStatus = mscclExistingCapture;
    }
    threadLocalStatus.captureId = captureId;
  } else {
    threadLocalStatus.captureStatus = mscclNoCapture;
  }
  INFO(NCCL_NET,"mscclGetCaptureStatus: %d, captureId: %llu, size: %lu\n", threadLocalStatus.captureStatus, threadLocalStatus.captureId, savedProxyArgs[captureId].size());
  return ncclSuccess;
}

ncclResult_t mscclSetupCount(struct mscclAlgo* hostAlgo, ncclComm_t comm, size_t count, ncclDataType_t dataType) {
  mscclStatus& status = mscclGetStatus(comm);
  status.stepSize = comm->buffSizes[hostAlgo->protocol] / NCCL_STEPS;
  status.chunkSteps = hostAlgo->protocol == NCCL_PROTO_SIMPLE ? hostAlgo->chunkSteps : 1;
  status.sliceSteps = hostAlgo->protocol == NCCL_PROTO_SIMPLE ? hostAlgo->sliceSteps : 1;
  status.chunkSize  = status.stepSize * status.chunkSteps;
  status.chunkEffectiveSize = status.chunkSize;
  if (hostAlgo->protocol == NCCL_PROTO_LL) status.chunkEffectiveSize /= 2;
  if (hostAlgo->protocol == NCCL_PROTO_LL128) status.chunkEffectiveSize = (status.chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
  status.dataType = dataType;
  status.nBytes = count * ncclTypeSize(status.dataType) * hostAlgo->sizeMultiplier;
  status.maxAllowedCount = std::max((uint32_t)1, (uint32_t)(status.chunkEffectiveSize / DIVUP(status.nBytes, (size_t)(hostAlgo->nChunksPerLoop))));
  if (status.maxAllowedCount == 0){
    WARN("MSCCL: something went wrong. Max allowed count is 0\n");
    return ncclInternalError;
  }
  if (status.maxAllowedCount >= MSCCL_MAX_COUNT) {
    status.maxAllowedCount = MSCCL_MAX_COUNT - 1;
  }
  return ncclSuccess;
}

ncclResult_t mscclSetupScratch(struct mscclAlgo* hostAlgo, hipStream_t stream) {
  return ncclSuccess;
}

ncclResult_t mscclSetupSyncFlags(const ncclComm_t comm, hipStream_t stream) {
  mscclStatus& status = mscclGetStatus(comm);
  mscclThreadLocalStatus& threadLocalStatus = mscclGetThreadLocalStatus();
  if (threadLocalStatus.captureStatus == mscclNewCapture ||
      status.workIndex > (1ULL << (8*sizeof(status.workIndex))) - 2 * NCCL_MAX_OPS - 1) {
    CUDACHECK(hipMemsetAsync(status.syncFlags, 0, sizeof(struct mscclFlag) * MSCCL_MAX_NUM_THREAD_BLOCKS, stream));
    status.workIndex = 1; // setting the workIndex back to 1 for next iterations
    status.graphFirstKernel = false;
  }
  return ncclSuccess;
}

ncclResult_t mscclSetupConnections(struct mscclAlgo* hostAlgo, ncclComm_t comm) {
  mscclStatus& status = mscclGetStatus(comm);

  // Check whether there are enough channels
  if (hostAlgo->nChannels > comm->nChannels) {
    WARN("MSCCL: number of channels available (%d) less than required (%d)", comm->nChannels, hostAlgo->nChannels);
    return ncclInvalidUsage;
  }

  // Flag MSCCL connections
  for (int i = 0; i < hostAlgo->nChannels; i++) {
    struct mscclChannelInfo* mCh = hostAlgo->mscclChannels + i;

    int sendPeers[MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
    for (int p = 0; p < mCh->nSendPeers; p++) {
      sendPeers[p] = mCh->sendPeerInfo[p].peer;
    }

    int recvPeers[MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
    for (int p = 0; p < mCh->nRecvPeers; p++) {
      recvPeers[p] = mCh->recvPeerInfo[p].peer;
    }

    NCCLCHECK(ncclTransportP2pConnect(comm, i, mCh->nRecvPeers, recvPeers, mCh->nSendPeers, sendPeers, 0 /*connIndex*/));
  }

  // Connect MSCCL connections
  mscclSetIsCallerFlag();
  bool needsProxy = false;
  NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 0, &needsProxy));
  status.needsProxy |= needsProxy;
  mscclClearIsCallerFlag();

  INFO(NCCL_INIT, "MSCCL: Setup connections finished, used %ld", allocTracker[comm->cudaDev].totalAllocSize);
  return ncclSuccess;
}

static ncclResult_t mscclSetupProxyImpl(struct mscclAlgo* hostAlgo, ncclComm_t comm) {
  mscclStatus& status = mscclGetStatus(comm);
  struct ncclProxyOp proxyOp = {};
  proxyOp.connIndex = 0;
  proxyOp.sliceSteps = status.sliceSteps;
  proxyOp.chunkSteps = status.chunkSteps;
  proxyOp.chunkSize = status.chunkSize;
  proxyOp.protocol = hostAlgo->protocol;
  proxyOp.dtype = status.dataType;
  proxyOp.redOp = 0;
  proxyOp.pattern = 0;
  proxyOp.root = 0;
  proxyOp.nbytes = status.stepSize*proxyOp.sliceSteps;
  proxyOp.opCount = comm->sharedRes->collOpCount;
  int nLoops = (int)(DIVUP(status.nBytes, (size_t)((size_t)hostAlgo->nChunksPerLoop*(size_t)status.chunkEffectiveSize)));
  int nLoopsChunkSteps = nLoops * status.chunkSteps;
  for (int ch = 0; ch < hostAlgo->nChannels; ch++) {
    proxyOp.channelId = ch;
    struct mscclChannelInfo* mscclChannel = hostAlgo->mscclChannels + ch;
    struct ncclChannel* ncclChannel = comm->channels + ch;
    for (int i = 0; i < mscclChannel->nRecvPeers; i++){
      struct mscclChannelPeerInfo* recvPeer = mscclChannel->recvPeerInfo + i;
      int nRecvs = 0;
      for (int j = 0; j < recvPeer->nExistingCounts; j++){
        int c = recvPeer->existingCounts[j];
        int nStepsInCount = DIVUP(c, status.maxAllowedCount);
        nRecvs += recvPeer->nTransmissionsOfCount[c] * nStepsInCount;
      }
      proxyOp.nsteps = nLoopsChunkSteps * nRecvs;
      if (proxyOp.nsteps > 0) {
        NCCLCHECK(mscclSaveProxy(comm, ncclChannel, proxyRecv, recvPeer->peer, &proxyOp, 0));
      }
    }
    for (int i=0; i<mscclChannel->nSendPeers; i++){
      struct mscclChannelPeerInfo* sendPeer = &mscclChannel->sendPeerInfo[i];
      int nSends = 0;
      for (int j = 0; j < sendPeer->nExistingCounts; j++){
        int c = sendPeer->existingCounts[j];
        int nStepsInCount = DIVUP(c, status.maxAllowedCount);
        nSends += sendPeer->nTransmissionsOfCount[c] * nStepsInCount;
      }
      proxyOp.nsteps = nLoopsChunkSteps * nSends;
      if (proxyOp.nsteps > 0) {
        NCCLCHECK(mscclSaveProxy(comm, ncclChannel, proxySend, sendPeer->peer, &proxyOp, 0));
      }
    }
  }
  NCCLCHECK(ncclProxyStart(comm));
  comm->sharedRes->collOpCount++;
  return ncclSuccess;
}

static void HIPRT_CB mscclSetupProxyCallback(void *args) {
  std::vector<struct mscclProxyArg>* params = (std::vector<struct mscclProxyArg>*)args;
  INFO(NCCL_NET,"mscclSetupProxyCallback: proxy args size: %ld\n", params->size());
  for (auto &p : *params) {
    mscclSetupProxyImpl(p.hostAlgo, p.comm);
  }
}

ncclResult_t mscclSetupProxy(struct mscclAlgo* hostAlgo, ncclComm_t comm, hipStream_t stream) {
  mscclStatus& status = mscclGetStatus(comm);
  mscclThreadLocalStatus& threadLocalStatus = mscclGetThreadLocalStatus();
  mscclSavedProxyArgs& savedProxyArgs = mscclGetSavedProxyArgs(comm);
  if (threadLocalStatus.captureStatus == mscclUnknownCaptureStatus) {
    INFO(NCCL_NET, "mscclSetupProxy: reading capture status");
    NCCLCHECK(mscclGetCaptureStatus(comm, stream));
  }
  if (threadLocalStatus.captureStatus == mscclNoCapture) {
    INFO(NCCL_NET,"mscclSetupProxy: no capture\n");
    NCCLCHECK(mscclSetupProxyImpl(hostAlgo, comm));
  } else if (status.needsProxy) {
    INFO(NCCL_NET,"mscclSetupProxy: capture\n");
    if (savedProxyArgs[threadLocalStatus.captureId].size() == 0) {
      INFO(NCCL_NET,"mscclSetupProxy: adding callback\n");

      hipGraphNode_t callbackNode;
      hipHostNodeParams p;
      p.fn = mscclSetupProxyCallback;
      auto params = &savedProxyArgs[threadLocalStatus.captureId];
      p.userData = params;
      CUDACHECK(hipGraphAddHostNode(&callbackNode, threadLocalStatus.graph, nullptr, 0, &p));
    }
    savedProxyArgs[threadLocalStatus.captureId].emplace_back(hostAlgo, comm);
  }
  return ncclSuccess;
}

static ncclResult_t hostToDevRedOp(
    ncclDevRedOpFull *opFull, ncclRedOp_t op, ncclDataType_t datatype, ncclComm *comm
  ) {
  union {
    int8_t i8;
    uint8_t u8;
    int32_t i32;
    uint32_t u32;
    int64_t i64;
    uint64_t u64;
    half f16;
    #if defined(RCCL_BFLOAT16)
      hip_bfloat16 bf16;
    #endif
    #if defined(RCCL_FLOAT8)
      rccl_float8 fp8_e4m3;
      rccl_bfloat8 fp8_e5m2;
    #endif
    float f32;
    double f64;
    void *ptr;
  };
  u64 = 0;
  opFull->scalarArgIsPtr = false;
  opFull->proxyOp = op;

  int nbits = 8*ncclTypeSize(datatype);
  uint64_t allBits = uint64_t(-1)>>(64-nbits);
  uint64_t signBit = allBits^(allBits>>1);

  switch (int(op)) {
  case ncclSum:  opFull->op = ncclDevSum;  break;
  case ncclProd: opFull->op = ncclDevProd; break;
  case ncclMin:
  case ncclMax:
    opFull->op = ncclDevMinMax;
    opFull->scalarArg = 0;
    // The xormask used by ncclFuncMinMax<[u]int> is the XOR of the sign bit
    // for signed (opposed to unsigned) types and all the bits for max (opposed to min).
    if (datatype==ncclInt8 || datatype==ncclInt32 || datatype==ncclInt64) {
      opFull->scalarArg ^= signBit;
    }
    opFull->scalarArg ^= (op == ncclMax) ? allBits : 0;
    break;
  case ncclAvg:
    switch ((int)datatype) {
    case ncclInt8:  case ncclInt32:  case ncclInt64:
    case ncclUint8: case ncclUint32: case ncclUint64:
      opFull->op = ncclDevSumPostDiv;
      u64 = comm->nRanks;
      break;
    case ncclFloat16:
      opFull->op = ncclDevPreMulSum;
      f16 = __float2half(float(1.0/comm->nRanks)); // __double2half not supported pre CUDA 11.x
      break;
    #if defined(RCCL_BFLOAT16)
    case ncclBfloat16:
      opFull->op = ncclDevPreMulSum;
      bf16 = (hip_bfloat16)(float(1.0/comm->nRanks));
      break;
    #endif
    #if defined(RCCL_FLOAT8)
    case ncclFloat8e4m3:
      opFull->op = ncclDevPreMulSum;
      fp8_e4m3 = (rccl_float8)(float(1.0/comm->nRanks));
      break;
    case ncclFloat8e5m2:
      opFull->op = ncclDevPreMulSum;
      fp8_e5m2 = (rccl_bfloat8)(float(1.0/comm->nRanks));
      break;
    #endif
    case ncclFloat32:
      opFull->op = ncclDevPreMulSum;
      f32 = float(1.0/comm->nRanks);
      break;
    case ncclFloat64:
      opFull->op = ncclDevPreMulSum;
      f64 = 1.0/comm->nRanks;
      break;
    }
    opFull->scalarArgIsPtr = false;
    opFull->scalarArg = u64;
    break;
  default: // user created
    int ix = int(ncclUserRedOpMangle(comm, op)) - int(ncclNumOps);
    ncclUserRedOp *user = &comm->userRedOps[ix];
    if (datatype != user->datatype) {
      WARN("Data type supplied to user-created ncclRedOp_t does not match type "
           "given to reduction operation");
      return ncclInvalidArgument;
    }
    *opFull = user->opFull;
    break;
  }
  return ncclSuccess;
}

#define MSCCL_KERNEL_ENTRY_DEVREDOP_NULL() \
  nullptr, \
  nullptr, \
  nullptr

#define MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, type, fullOps) \
  (void *)MSCCL_KERNEL_ENTRY_NAME(devredop, type, LL, fullOps), \
  (void *)MSCCL_KERNEL_ENTRY_NAME(devredop, type, LL128, fullOps), \
  (void *)MSCCL_KERNEL_ENTRY_NAME(devredop, type, Simple, fullOps)

#define MSCCL_KERNEL_ENTRY_DEVREDOP(devredop, fullOps) \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int8_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint8_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int32_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint32_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int64_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint64_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, half, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, float, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, double, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, hip_bfloat16, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, rccl_float8, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, rccl_bfloat8, fullOps)

#define MSCCL_KERNEL_ENTRY_DEVREDOP_NOFLOAT(devredop, fullOps) \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int8_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint8_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int32_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint32_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int64_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint64_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL()

#define MSCCL_KERNEL_ENTRY() \
  MSCCL_KERNEL_ENTRY_DEVREDOP(Sum, false), \
  MSCCL_KERNEL_ENTRY_DEVREDOP(Prod, false), \
  MSCCL_KERNEL_ENTRY_DEVREDOP(MinMax, false)

// Except for ncclDevPreMulSum and ncclDevSumPostDiv required by ncclAvg
void* mscclKernelEntries[(ncclNumDevRedOps-2) * ncclNumTypes * NCCL_NUM_PROTOCOLS] = {
#ifdef COMPILE_MSCCL_KERNEL
  MSCCL_KERNEL_ENTRY()
#endif
};

// Comparison of monotonic rolling counters.
static inline bool rollingLess32(uint32_t a, uint32_t b) {
  constexpr uint32_t PositiveMax = uint32_t(-1)>>1;
  return a-b > PositiveMax;
}

static inline uint32_t rollingMin32(uint32_t a, uint32_t b) {
  constexpr uint32_t PositiveMax = uint32_t(-1)>>1;
  return (b-a <= PositiveMax) ? a : b;
}

static void mscclWaitWorkFifoAvailable(uint32_t desiredSent, mscclWorkFifoStatus* status) {
  if (__builtin_expect(rollingLess32(status->workFifoAckdMin + status->workFifoDepth, desiredSent), false)) {
    while (1) {
      // We have to poll for notifications from device.
      uint32_t* doneLive = status->workFifoDone;
      uint32_t ackd[MSCCL_MAX_NUM_THREAD_BLOCKS];
      for (int c=0; c < MSCCL_MAX_NUM_THREAD_BLOCKS; c++) {
        ackd[c] = __atomic_load_n(&doneLive[c], __ATOMIC_RELAXED);
      }
      // Compiler-only fence to prevent fusion of loops to encourage dense loads.
      __atomic_signal_fence(__ATOMIC_SEQ_CST);

      uint32_t ackdAll = status->workFifoSent;
      for (int c=0; c < MSCCL_MAX_NUM_THREAD_BLOCKS; c++) {
        // ackdAll is min over all non-quiesced channels
        if (ackd[c] != status->workFifoSentPerThreadBlock[c])
          ackdAll = rollingMin32(ackdAll, ackd[c]);
      }

      // Compiler only fence to prevent fusion of loops to encourage dense stores.
      __atomic_signal_fence(__ATOMIC_SEQ_CST);

      for (int c=0; c < MSCCL_MAX_NUM_THREAD_BLOCKS; c++) {
        // Advance counter on quiesced channels so they don't lag behind
        // too far where they could get lost in 32-bit wraparound.
        if (ackd[c] == status->workFifoSentPerThreadBlock[c]) {
          status->workFifoSentPerThreadBlock[c] = ackdAll;
          __atomic_store_n(&doneLive[c], ackdAll, __ATOMIC_RELAXED);
        }
      }
      status->workFifoAckdMin = ackdAll;

      // See if that was enough.
      if (!rollingLess32(status->workFifoAckdMin + status->workFifoDepth, desiredSent)) break;
      sched_yield();
    }
  }
}

RCCL_PARAM(MscclForceFullOps, "MSCCL_FORCE_FULLOPS", 0);

ncclResult_t mscclSetupKernel(const void* sendBuff, void* recvBuff, size_t count,
    ncclDataType_t dataType, ncclRedOp_t op, struct mscclAlgo* hostAlgo, struct mscclAlgo* devAlgo,
    ncclComm_t comm, hipStream_t stream) {
  mscclStatus& status = mscclGetStatus(comm);
  mscclThreadLocalStatus& threadLocalStatus = mscclGetThreadLocalStatus();

  if (status.lastStream != stream && status.lastStream != nullptr) {
    CUDACHECK(hipStreamWaitEvent(stream, comm->doneEvent, 0));
  }

  uint32_t numBlocks = (uint32_t)hostAlgo->nBlocks;
  dim3 grid = {numBlocks, 1, 1};
  dim3 block = {MSCCL_MAX_NTHREADS, 1, 1};
  ncclDevRedOpFull opFull = {};
  NCCLCHECK(hostToDevRedOp(&opFull, op, dataType, comm));

  uint32_t fnIndex = (opFull.op * ncclNumTypes + dataType) * NCCL_NUM_PROTOCOLS + hostAlgo->protocol;
  uint8_t fullOpMask = (1<<MSCCL_RECV_COPY_SEND) |
                        (1<<MSCCL_RECV_REDUCE_SEND) |
                        (1<<MSCCL_RECV_REDUCE_COPY_SEND) |
                        (1<<MSCCL_RECV_REDUCE_COPY);
  //check if need full ops msccl kernel
  if ((hostAlgo->typeMask & fullOpMask) || rcclParamMscclForceFullOps()) {
    WARN("MSCCL: this version of MSCCL build doesn't support full Ops");
    return ncclInternalError;
  }

  mscclWork work;
  work.syncFlags = status.syncFlags;
  size_t sizeNeeded = computeSizeNeeded(status.nBytes, hostAlgo->nScratchChunks, hostAlgo->nChunksPerLoop);
  if (sizeNeeded > 0) {
    auto itr = status.scratchBuffers.lower_bound(sizeNeeded);
    if (itr == status.scratchBuffers.end()) {
      void *scratchBuffer = nullptr;
      size_t sizeRounded = 1;
      if (status.scratchBuffers.size() > 0) {
        sizeRounded = status.scratchBuffers.rbegin()->first;
      }
      while (sizeRounded < sizeNeeded) {
        if (sizeRounded >= sizeRounded * 2) {
          WARN("MSCCL: Size of allocation for scratch buffer (%lu * 2) will wrap around", sizeRounded);
          return ncclInvalidUsage;
        }
        sizeRounded *= 2;
      }
#if defined(HIP_UNCACHED_MEMORY)
      NCCLCHECK(ncclCudaMalloc((char**)&scratchBuffer, sizeRounded, hipDeviceMallocUncached));
#else
      NCCLCHECK(ncclCudaMalloc((char**)&scratchBuffer, sizeRounded, hipDeviceMallocFinegrained));
#endif
      work.scratchBuffer = status.scratchBuffers[sizeRounded] = scratchBuffer;
      TRACE(NCCL_INIT, "MSCCL: Allocated scratch buffer of size %lu on request (%lu)", sizeRounded, sizeNeeded);
    } else {
      work.scratchBuffer = itr->second;
    }
  } else {
    work.scratchBuffer = nullptr;
  }
  work.sendBuff = sendBuff;
  work.recvBuff = recvBuff;
  work.sizePerMscclChunk = count * hostAlgo->sizeMultiplier / hostAlgo->nChunksPerLoop; // count is sum of all ranks in MSCCL kernel
  work.redOpArg = opFull.scalarArg;
  work.workIndex = status.workIndex;
  work.nChunksPerLoop = hostAlgo->nChunksPerLoop;
  work.maxAllowedCount = status.maxAllowedCount;
  work.hasReduce = hostAlgo->hasReduce;
  work.redOpArgIsPtr = opFull.scalarArgIsPtr;
  work.fnIndex = fnIndex;
  INFO(NCCL_COLL, "MSCCL: typeMask %x fnIndex %d Setup Kernel finished", hostAlgo->typeMask, fnIndex);

  if (threadLocalStatus.captureStatus == mscclUnknownCaptureStatus) {
    INFO(NCCL_NET, "MSCCL: reading capture status");
    NCCLCHECK(mscclGetCaptureStatus(comm, stream));
  }
  mscclWorkFifoStatus* workFifoStatus = nullptr;
  if (threadLocalStatus.captureStatus == mscclNoCapture) {
    workFifoStatus = &(status.defaultWorkFifoStatus);
  } else {
    workFifoStatus = &(status.graphWorkFifoStatus[threadLocalStatus.captureId]);
  }

  uint32_t workFifoIdxMask = workFifoStatus->workFifoDepth - 1;
  uint32_t workFifoSent = workFifoStatus->workFifoSent;

  if (threadLocalStatus.captureStatus != mscclNoCapture && workFifoSent + numBlocks > workFifoStatus->workFifoDepth) {
    WARN("MSCCL: number of captured works (%u) > max limit (%lu)", workFifoSent + numBlocks, workFifoStatus->workFifoDepth);
    return ncclInternalError;
  }

  // First work for a channel has to be at workHeap+blockIdx.x which means
  // we cannot tolerate fifo wraparound. So round up to the wrap boundary
  // if not doing so would incur crossing it.
  if (((workFifoSent + numBlocks - 1) & workFifoIdxMask) < (workFifoSent & workFifoIdxMask)) {
    workFifoSent = (workFifoSent + workFifoIdxMask) & ~workFifoIdxMask;
    // Need to update workFifoSent so waitWorkFifoAvailable() knows we've
    // skipped those elements. Consider if all the channels report quiesced,
    // this way the skipped slots will be considered consumed as well.
    workFifoStatus->workFifoSent = workFifoSent;
  }
  mscclWaitWorkFifoAvailable(workFifoSent + numBlocks, workFifoStatus);
  for (int i = 0; i < numBlocks; i++) {
    work.workFifoDoneAck = workFifoSent + i;
    work.workFifoDone = workFifoStatus->workFifoDone + i;
    workFifoStatus->workFifoSentPerThreadBlock[i] = workFifoSent + i;
    workFifoStatus->workFifo[(workFifoSent + i) & workFifoIdxMask] = work;
  }

  struct mscclWork *workPtr = workFifoStatus->workFifo + (workFifoSent & workFifoIdxMask);
  workFifoStatus->workFifoSent = workFifoSent + numBlocks;

  void *args[3] = {&comm->devComm, &devAlgo, &workPtr};
  void *func = mscclKernelEntries[fnIndex];
  CUDACHECK(hipExtLaunchKernel(func, grid, block, args, 0, stream, NULL, comm->doneEvent, 0));
  status.workIndex++;
  status.lastStream = stream;
  return ncclSuccess;
}

// Determine the maximum kernel stack size of all MSCCL kernels
size_t mscclKernMaxLocalSize() {
  ncclResult_t res = ncclSuccess;
  int numMscclKerns = sizeof(mscclKernelEntries)/sizeof(void *);
  hipFuncAttributes attr = {0};
  size_t max = 0;
  for (int i = 0; i < numMscclKerns; i++) {
    if (mscclKernelEntries[i] != nullptr) {
      CUDACHECKGOTO(hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(mscclKernelEntries[i])), res, error);
      if (attr.localSizeBytes > max) max = attr.localSizeBytes;
    }
  }

error:
  return (res != ncclSuccess) ? 0 : max;
}

ncclResult_t mscclInitWorkFifoStatus(mscclWorkFifoStatus* workFifoStatus) {
  workFifoStatus->workFifoDepth = rcclParamMscclWorkFifoDepth();
#if defined(HIP_UNCACHED_MEMORY)
  NCCLCHECK(ncclCudaMalloc(&(workFifoStatus->workFifo), workFifoStatus->workFifoDepth, hipDeviceMallocUncached));
#else
  NCCLCHECK(ncclCudaMalloc(&(workFifoStatus->workFifo), workFifoStatus->workFifoDepth, hipDeviceMallocFinegrained));
#endif
  NCCLCHECK(ncclCudaHostCalloc(&(workFifoStatus->workFifoDone), MSCCL_MAX_NUM_THREAD_BLOCKS));
  workFifoStatus->workFifoSent = 0;
  for (int i = 0; i < MSCCL_MAX_NUM_THREAD_BLOCKS; i++) {
    workFifoStatus->workFifoSentPerThreadBlock[i] = 0;
  }
  workFifoStatus->workFifoAckdMin = 0;
  return ncclSuccess;
}

ncclResult_t mscclDestroyWorkFifoStatus(mscclWorkFifoStatus* workFifoStatus) {
  NCCLCHECK(ncclCudaFree(workFifoStatus->workFifo));
  NCCLCHECK(ncclCudaHostFree(workFifoStatus->workFifoDone));
  return ncclSuccess;
}
