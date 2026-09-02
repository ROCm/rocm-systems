/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * GIN-SDMA AllReduce for single-node (scaleup-only) symmetric windows.
 * Default: only messages >= 256 MiB take this path (GIN two-shot); smaller
 * messages fall through to DDA AllReduce. RCCL_GIN_ALLREDUCE_FORCE_ENABLE=1
 * also enables the LSA bands:
 *   <= 4 MiB   — LSA one-shot
 *   (4, 256) MiB — LSA two-shot
 *   >= 256 MiB — GIN two-shot (LSA reduce-scatter + GIN all-gather)
 *
 * Compiled with NCCL_GIN_ANVIL_SDMA_ENABLE=1 and NCCL_GIN_PROXY_ENABLE=0 so
 * ncclGinCallImpl resolves the SDMA backend at compile time.
 * See LICENSE.txt for license information.
 ******************************************************************************/

#include "algorithms/gin/gin_all_reduce.h"

#include "algorithms/gin/sdma/all_reduce_gin_sdma.h"
#include "algorithms/dda/device/CollCommon.h"
#include "alloc.h"
#include "archinfo.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "dev_runtime.h"
#include "param.h"

#include <cuda_runtime.h>

NCCL_PARAM(GinAllReduceEnable, "GIN_ALLREDUCE_ENABLE", 1);
// When 0 (default), GIN AllReduce is eligible only for messages >= 256 MiB.
// Set to 1 to also take LSA one-shot / LSA two-shot for smaller sizes.
RCCL_PARAM(GinAllReduceForceEnable, "GIN_ALLREDUCE_FORCE_ENABLE", 0);
// LSA two-shot tuning. CTAs default to kGinAllReduceLsaTwoShotCtasPerPeer * nRanks; the DDA IPC
// kernels top out at DDA_IPC_MAXBLOCKS (24), so this range is worth sweeping. OVERLAP=1 selects the
// pipelined kernel that pushes reduced columns to peers instead of the non-overlapped one.
NCCL_PARAM(GinAllReduceLsaTwoShotCtas, "GIN_ALLREDUCE_LSA_TWOSHOT_CTAS", 0);

namespace {

constexpr bool kSdmaDeviceBackendCompiled = (NCCL_GIN_ANVIL_SDMA_ENABLE != 0);

// Runs on the first eligible AllReduce, which may itself be inside a graph capture. Everything
// here has to stay out of the captured graph and must not disturb the capture in progress:
//
//   - ncclDevrCommCreateInternal already handles this itself; it does its allocations and
//     bootstrap collectives on a private stream with the capture mode set to relaxed.
//   - The signal reset is ours to place. Launching it on the NULL stream would pull it into the
//     user's capture (cudaErrorStreamCaptureImplicit), and cudaDeviceSynchronize is illegal
//     while capturing, so it runs on a private stream under relaxed capture mode. Being outside
//     the graph is also required for correctness: a captured reset would re-zero signals on
//     every replay while peers were incrementing them.
//
// Resetting before returning is safe against peers because the kernel does not issue any put
// until after the world barrier that follows the reduce-scatter, and no rank reaches that
// barrier before every rank has finished initializing.
static ncclResult_t ncclGinAllReduceInitOnce(ncclComm* comm) {
  NCCLCHECK(ncclDevrInitOnce(comm));
  struct ncclGinAllReduceState* state = &comm->ginAllReduceState;
  if (state->initialized) {
    return ncclSuccess;
  }

  struct ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = kGinAllReduceLsaTwoShotMaxCtas;
  // GinAlltoAllKernel: one world barrier + GIN signal per CTA.
  reqs.barrierCount = kGinAllReduceLsaCtas;
  reqs.ginSignalCount = kGinAllReduceLsaCtas;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  NCCLCHECK(ncclDevrCommCreateInternal(comm, &reqs, &state->devComm, /*isInternal=*/true));

  // Two words (arrived, sense) for ginIntraGpuCtaBarrier. ncclCudaCalloc is capture-safe
  // (relaxed + side stream). Sense-reversing needs no per-launch host reset.
  if (state->intraGpuCtaBar == nullptr) {
    NCCLCHECK(ncclCudaCalloc(&state->intraGpuCtaBar, 2, comm->memManager));
  }

  cudaStreamCaptureMode captureMode = cudaStreamCaptureModeRelaxed;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&captureMode));

  cudaStream_t initStream = nullptr;
  // Not named `err`: CUDACHECK declares a local `err`, which this would shadow.
  cudaError_t resetErr = cudaStreamCreateWithFlags(&initStream, cudaStreamNonBlocking);
  if (resetErr == cudaSuccess) {
    gin::sdma::ginAllReduceResetSignalsKernel<<<kGinAllReduceLsaCtas, 1, 0, initStream>>>(state->devComm);
    resetErr = cudaGetLastError();
    if (resetErr == cudaSuccess) {
      resetErr = cudaStreamSynchronize(initStream);
    }
    const cudaError_t destroyErr = cudaStreamDestroy(initStream);
    if (resetErr == cudaSuccess) {
      resetErr = destroyErr;
    }
  }

  // Restore the caller's capture mode before reporting, so a failure above cannot leave this
  // thread in relaxed mode.
  CUDACHECKIGNORE(cudaThreadExchangeStreamCaptureMode(&captureMode));
  CUDACHECK(resetErr);

  state->initialized = true;
  return ncclSuccess;
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

  const size_t sendOff =
    static_cast<size_t>(static_cast<const char*>(sendbuff) - static_cast<const char*>(sendWin->userPtr));
  const size_t recvOff =
    static_cast<size_t>(static_cast<char*>(recvbuff) - static_cast<const char*>(recvWin->userPtr));
  const size_t countPerRank = count / static_cast<size_t>(comm->nRanks);

  gin::sdma::ginAllReduceTwoShotKernel<T><<<kGinAllReduceLsaCtas, kGinAllReduceLsaThreadsPerCta, 0, stream>>>(
    comm->ginAllReduceState.devComm, sendWin->vidmem, sendOff, recvWin->vidmem, recvOff, countPerRank, comm->nRanks,
    comm->ginAllReduceState.intraGpuCtaBar);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclAllReduceGinSdmaTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                              cudaStream_t stream, struct ncclDevrWindow* sendWin,
                                              struct ncclDevrWindow* recvWin) {
  const size_t bytes = count * sizeof(T);
  // Inclusive of 4 MiB so dispatch matches ginAllReduceSizePolicyEligible() and the
  // documented LSA one-shot band. Two-shot starts strictly above this threshold.
  if (bytes <= kGinAllReduceLsaOneShotMaxBytes) {
    return ncclAllReduceGinSdmaOneShotTyped<T>(sendbuff, recvbuff, count, comm, stream, sendWin, recvWin);
  }
  if (bytes >= kGinAllReduceGinTwoShotMinBytes) {
    return ncclAllReduceGinSdmaGinTwoShotTyped<T>(sendbuff, recvbuff, count, comm, stream, sendWin, recvWin);
  }
  return ncclAllReduceGinSdmaLsaTwoShotTyped<T>(sendbuff, recvbuff, count, comm, stream, sendWin, recvWin);
}

// Comm / buffer / datatype gates shared by eligibility and the DDA fallback.
// Size policy lives in gin_all_reduce_policy.h so unit tests cover the same gates.
static bool ginAllReduceBaseEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                     ncclDataType_t datatype, ncclRedOp_t op) {
  if (!kSdmaDeviceBackendCompiled) return false;
  if (!ncclParamGinAllReduceEnable()) return false;

  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr) return false;
  if (count == 0) return false;
  if (!IsArchMatch(comm->archName, "gfx950")) return false;
  if (op != ncclSum) return false;
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) return false;
  if (!comm->symmetricSupport) return false;

  bool symEligible = (op == ncclSum) && isSymmetricKernelRequested(comm, ncclFuncAllReduce, (int)ncclDevSum, datatype,
                                                                   count, sendbuff, recvbuff);
  if (!symEligible) return false;

  if (comm->globalGinSupport != NCCL_GIN_CONNECTION_FULL) return false;
  if (comm->nNodes != 1) return false;
  if (comm->nRanks > kGinAllReduceMaxRanks) return false;
  if (comm->sharedRes == nullptr) return false;
  if (comm->sharedRes->ginState.ginType != (ncclGinType_t)NCCL_NET_DEVICE_GIN_ANVIL_SDMA) return false;
  // After cheaper ginType: ncclTeamLsa() can initialize the device runtime.
  if (ncclTeamLsa(comm).nRanks != comm->nRanks) return false;
  return true;
}

} // namespace


bool ncclAllReduceGinSdmaEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                  ncclDataType_t datatype, ncclRedOp_t op) {
  if (!ginAllReduceBaseEligible(comm, sendbuff, recvbuff, count, datatype, op)) return false;
  return ginAllReduceSizePolicyEligible(count, ncclTypeSize(datatype), comm->nRanks,
                                       rcclParamGinAllReduceForceEnable() == 1);
}

bool ncclAllReduceGinSdmaYieldToDda(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                    ncclDataType_t datatype, ncclRedOp_t op) {
  if (!ginAllReduceBaseEligible(comm, sendbuff, recvbuff, count, datatype, op)) return false;
  return ginAllReduceYieldToDdaBySize(count, ncclTypeSize(datatype), rcclParamGinAllReduceForceEnable() == 1);
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
  if (state->intraGpuCtaBar != nullptr) {
    NCCLCHECK(ncclCudaFree(state->intraGpuCtaBar, comm->memManager));
    state->intraGpuCtaBar = nullptr;
  }
  if (state->initialized) {
    NCCLCHECK(ncclDevCommDestroy(comm, &state->devComm));
    state->initialized = false;
  }
  return ncclSuccess;
}
