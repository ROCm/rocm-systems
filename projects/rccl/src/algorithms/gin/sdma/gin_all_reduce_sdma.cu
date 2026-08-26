/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * GIN-SDMA AllReduce for single-node (scaleup-only) symmetric windows.
 *   <= 16 MiB  — LSA one-shot
 *   > 16 MiB   — LSA two-shot
 *   >= 128 MiB — GIN two-shot (LSA reduce-scatter + GIN all-gather)
 *
 * Compiled with NCCL_GIN_ANVIL_SDMA_ENABLE=1 and NCCL_GIN_PROXY_ENABLE=0 so
 * ncclGinCallImpl resolves the SDMA backend at compile time.
 * See LICENSE.txt for license information.
 ******************************************************************************/

#include "algorithms/gin/gin_all_reduce.h"

#include "algorithms/gin/sdma/all_reduce_gin_sdma.h"
#include "algorithms/dda/device/CollCommon.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "dev_runtime.h"
#include "param.h"

#include <cuda_runtime.h>

NCCL_PARAM(GinAllReduceEnable, "GIN_ALLREDUCE_ENABLE", 1);
// LSA two-shot tuning. CTAs default to kGinAllReduceLsaTwoShotCtasPerPeer * nRanks; the DDA IPC
// kernels top out at DDA_IPC_MAXBLOCKS (24), so this range is worth sweeping. OVERLAP=1 selects the
// pipelined kernel that pushes reduced columns to peers instead of the non-overlapped one.
NCCL_PARAM(GinAllReduceLsaTwoShotCtas, "GIN_ALLREDUCE_LSA_TWOSHOT_CTAS", 0);
NCCL_PARAM(GinAllReduceLsaTwoShotOverlap, "GIN_ALLREDUCE_LSA_TWOSHOT_OVERLAP", 0);

namespace {

constexpr bool kSdmaDeviceBackendCompiled = (NCCL_GIN_ANVIL_SDMA_ENABLE != 0);
constexpr int kGinAllReduceMaxRanks = 8;

static ncclResult_t ncclGinAllReduceInitOnce(ncclComm* comm) {
  NCCLCHECK(ncclDevrInitOnce(comm));
  struct ncclGinAllReduceState* state = &comm->ginAllReduceState;
  if (!state->initialized) {
    struct ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = kGinAllReduceLsaTwoShotMaxCtas;
    // GinAlltoAllKernel: one world barrier + GIN signal per CTA.
    reqs.barrierCount = kGinAllReduceLsaCtas;
    reqs.ginSignalCount = kGinAllReduceLsaCtas;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    NCCLCHECK(ncclDevrCommCreateInternal(comm, &reqs, &state->devComm, /*isInternal=*/true));
    gin::sdma::ginAllReduceResetSignalsKernel<<<kGinAllReduceLsaCtas, 1>>>(state->devComm);
    CUDACHECK(cudaDeviceSynchronize());
    state->initialized = true;
  }
  return ncclSuccess;
}

static ncclResult_t ncclGinAllReduceEnsureTwoShotSync(ncclComm* comm) {
  struct ncclGinAllReduceState* state = &comm->ginAllReduceState;
  if (state->twoShotSync != nullptr) {
    return ncclSuccess;
  }
  CUDACHECK(cudaMalloc(&state->twoShotSync, kGinAllReduceTwoShotSyncBytes));
  CUDACHECK(cudaMemset(state->twoShotSync, 0, kGinAllReduceTwoShotSyncBytes));
  state->twoShotReduceEpoch = 0;
  state->twoShotAgEpoch = 0;
  return ncclSuccess;
}

static uint64_t ginAllReduceNextReduceTarget(ncclComm* comm) {
  struct ncclGinAllReduceState* state = &comm->ginAllReduceState;
  state->twoShotReduceEpoch += static_cast<uint64_t>(kGinAllReduceLsaCtas);
  return state->twoShotReduceEpoch;
}

static uint64_t ginAllReduceNextAgTarget(ncclComm* comm) {
  struct ncclGinAllReduceState* state = &comm->ginAllReduceState;
  state->twoShotAgEpoch += static_cast<uint64_t>(kGinAllReduceLsaCtas);
  return state->twoShotAgEpoch;
}

template <typename T>
static ncclResult_t ncclAllReduceGinSdmaOneShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                     ncclComm* comm, cudaStream_t stream,
                                                     struct ncclDevrWindow* sendWin,
                                                     struct ncclDevrWindow* recvWin) {
  NCCLCHECK(ncclGinAllReduceInitOnce(comm));

  const size_t sendOff =
    static_cast<size_t>(static_cast<const char*>(sendbuff) - static_cast<const char*>(sendWin->userPtr));
  const size_t recvOff =
    static_cast<size_t>(static_cast<char*>(recvbuff) - static_cast<const char*>(recvWin->userPtr));

  gin::sdma::allReduceLsaOneShotKernel<T><<<kGinAllReduceLsaCtas, kGinAllReduceLsaThreadsPerCta, 0, stream>>>(
    sendWin->vidmem, sendOff, recvWin->vidmem, recvOff, count, comm->ginAllReduceState.devComm);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

// Grid must stay within the lsaBarrierCount reserved in ncclGinAllReduceInitOnce, since each CTA
// syncs on the barrier at its own blockIdx.
static int ginAllReduceLsaTwoShotCtas(int nRanks) {
  const int64_t requested = ncclParamGinAllReduceLsaTwoShotCtas();
  int ctas = requested > 0 ? static_cast<int>(requested) : kGinAllReduceLsaTwoShotCtasPerPeer * nRanks;
  if (ctas > kGinAllReduceLsaTwoShotMaxCtas) {
    ctas = kGinAllReduceLsaTwoShotMaxCtas;
  }
  return ctas < 1 ? 1 : ctas;
}

// NRANKS_CT folds the clique size into the kernel so the peer loops unroll fully; 0 is the
// runtime fallback for clique sizes without a specialization.
template <typename T, int NRANKS_CT>
static void ginAllReduceLaunchLsaTwoShot(ncclComm* comm, cudaStream_t stream, struct ncclDevrWindow* sendWin,
                                         size_t sendOff, struct ncclDevrWindow* recvWin, size_t recvOff,
                                         size_t countPerRank, int gridCtas) {
  size_t msgSize = countPerRank * sizeof(T) * comm->nRanks;  

  gin::sdma::lsaAllReduceTwoShotKernel<T, NRANKS_CT><<<gridCtas, kGinAllReduceLsaThreadsPerCta, 0, stream>>>(
      comm->ginAllReduceState.devComm, sendWin->vidmem, sendOff, recvWin->vidmem, recvOff, countPerRank, comm->nRanks);
}

template <typename T>
static ncclResult_t ncclAllReduceGinSdmaLsaTwoShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                        ncclComm* comm, cudaStream_t stream,
                                                        struct ncclDevrWindow* sendWin,
                                                        struct ncclDevrWindow* recvWin) {
  NCCLCHECK(ncclGinAllReduceInitOnce(comm));

  const size_t sendOff =
    static_cast<size_t>(static_cast<const char*>(sendbuff) - static_cast<const char*>(sendWin->userPtr));
  const size_t recvOff =
    static_cast<size_t>(static_cast<char*>(recvbuff) - static_cast<const char*>(recvWin->userPtr));
  const size_t countPerRank = count / static_cast<size_t>(comm->nRanks);
  const int gridCtas = ginAllReduceLsaTwoShotCtas(comm->nRanks);

  switch (comm->nRanks) {
  case 2:
    ginAllReduceLaunchLsaTwoShot<T, 2>(comm, stream, sendWin, sendOff, recvWin, recvOff, countPerRank, gridCtas);
    break;
  case 4:
    ginAllReduceLaunchLsaTwoShot<T, 4>(comm, stream, sendWin, sendOff, recvWin, recvOff, countPerRank, gridCtas);
    break;
  case 8:
    ginAllReduceLaunchLsaTwoShot<T, 8>(comm, stream, sendWin, sendOff, recvWin, recvOff, countPerRank, gridCtas);
    break;
  default:
    ginAllReduceLaunchLsaTwoShot<T, 0>(comm, stream, sendWin, sendOff, recvWin, recvOff, countPerRank, gridCtas);
    break;
  }
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclAllReduceGinSdmaGinTwoShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                        ncclComm* comm, cudaStream_t stream,
                                                        struct ncclDevrWindow* sendWin,
                                                        struct ncclDevrWindow* recvWin) {
  NCCLCHECK(ncclGinAllReduceInitOnce(comm));
  NCCLCHECK(ncclGinAllReduceEnsureTwoShotSync(comm));

  const size_t sendOff =
    static_cast<size_t>(static_cast<const char*>(sendbuff) - static_cast<const char*>(sendWin->userPtr));
  const size_t recvOff =
    static_cast<size_t>(static_cast<char*>(recvbuff) - static_cast<const char*>(recvWin->userPtr));
  const size_t countPerRank = count / static_cast<size_t>(comm->nRanks);
  const uint64_t reduceTarget = ginAllReduceNextReduceTarget(comm);
  const uint64_t agTarget = ginAllReduceNextAgTarget(comm);

  gin::sdma::ginAllReduceTwoShotKernel<T><<<kGinAllReduceLsaCtas, kGinAllReduceLsaThreadsPerCta, 0, stream>>>(
    	comm->ginAllReduceState.devComm, sendWin->vidmem, sendOff, recvWin->vidmem, recvOff, countPerRank,
    	comm->ginAllReduceState.twoShotSync, reduceTarget, comm->ginAllReduceState.twoShotSync + 1, agTarget,
    	comm->nRanks);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclAllReduceGinSdmaTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                              cudaStream_t stream, struct ncclDevrWindow* sendWin,
                                              struct ncclDevrWindow* recvWin) {
  const size_t bytes = count * sizeof(T);
  if (bytes < kGinAllReduceLsaOneShotMaxBytes) {
    return ncclAllReduceGinSdmaOneShotTyped<T>(sendbuff, recvbuff, count, comm, stream, sendWin, recvWin);
  }
  if (bytes >= kGinAllReduceGinTwoShotMinBytes) {
    return ncclAllReduceGinSdmaGinTwoShotTyped<T>(sendbuff, recvbuff, count, comm, stream, sendWin, recvWin);
  }
  return ncclAllReduceGinSdmaLsaTwoShotTyped<T>(sendbuff, recvbuff, count, comm, stream, sendWin, recvWin);
}

static bool ginAllReduceTwoShotEligible(size_t count, ncclDataType_t datatype, int nRanks) {
  if (count % static_cast<size_t>(nRanks) != 0) {
    return false;
  }
  const size_t countPerRank = count / static_cast<size_t>(nRanks);
  const size_t typeSize = ncclTypeSize(datatype);
  if ((countPerRank * typeSize) % 16 != 0) {
    return false;
  }
  return true;
}

static bool ginAllReduceGinTwoShotEligible(size_t count, ncclDataType_t datatype, int nRanks) {
  if (!ginAllReduceTwoShotEligible(count, datatype, nRanks)) {
    return false;
  }
  const size_t chunkBytes = (count / static_cast<size_t>(nRanks)) * ncclTypeSize(datatype);
  return chunkBytes >= kGinAllReduceMinPutBytes;
}

} // namespace


bool ncclAllReduceGinSdmaEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                  ncclDataType_t datatype, ncclRedOp_t op) {
  if (!kSdmaDeviceBackendCompiled) return false;
  if (!ncclParamGinAllReduceEnable()) return false;
  
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr) return false;
  if (count == 0) return false;
  if (op != ncclSum) return false;
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) return false;
  if (!comm->symmetricSupport) return false;

  bool symEligible = (op == ncclSum) && isSymmetricKernelRequested(comm, ncclFuncAllReduce, (int)ncclDevSum, datatype,
                                                                   count, sendbuff, recvbuff);
  if (!symEligible) return false;

  if (comm->globalGinSupport != NCCL_GIN_CONNECTION_FULL) return false;
  if (comm->nNodes != 1) return false;
  if (ncclTeamLsa(comm).nRanks != comm->nRanks) return false;
  if (comm->nRanks > kGinAllReduceMaxRanks) return false;
  if (comm->sharedRes->ginState.ginType != (ncclGinType_t)NCCL_NET_DEVICE_GIN_ANVIL_SDMA) return false;


  const size_t bytes = count * ncclTypeSize(datatype);
  if (bytes < kGinAllReduceMinBytes) {
    return false;
  } 
  if (bytes <= kGinAllReduceLsaOneShotMaxBytes) {
    return true;
  }
  if (bytes >= kGinAllReduceGinTwoShotMinBytes) {
    return ginAllReduceGinTwoShotEligible(count, datatype, comm->nRanks);
  }
  return ginAllReduceTwoShotEligible(count, datatype, comm->nRanks);
}

ncclResult_t ncclAllReduceGinSdma(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                  ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct ncclDevrWindow* sendWin = nullptr;
  struct ncclDevrWindow* recvWin = nullptr;
  NCCLCHECK(ncclDevrFindWindow(comm, sendbuff, &sendWin));
  NCCLCHECK(ncclDevrFindWindow(comm, recvbuff, &recvWin));
  if (sendWin == nullptr || recvWin == nullptr) return ncclInvalidUsage;

  switch (datatype) {
  case ncclFloat32:
    return ncclAllReduceGinSdmaTyped<float>(sendbuff, recvbuff, count, comm, stream, sendWin, recvWin);
  case ncclFloat16:
    return ncclAllReduceGinSdmaTyped<half>(sendbuff, recvbuff, count, comm, stream, sendWin, recvWin);
  case ncclBfloat16:
    return ncclAllReduceGinSdmaTyped<bf16>(sendbuff, recvbuff, count, comm, stream, sendWin, recvWin);
  default:
    return ncclInvalidArgument;
  }
}

ncclResult_t ncclGinAllReduceFinalize(ncclComm* comm) {
  struct ncclGinAllReduceState* state = &comm->ginAllReduceState;
  if (state->initialized) {
    NCCLCHECK(ncclDevCommDestroy(comm, &state->devComm));
    state->initialized = false;
  }
  if (state->twoShotSync != nullptr) {
    CUDACHECK(cudaFree(state->twoShotSync));
    state->twoShotSync = nullptr;
    state->twoShotReduceEpoch = 0;
    state->twoShotAgEpoch = 0;
  }
  return ncclSuccess;
}
