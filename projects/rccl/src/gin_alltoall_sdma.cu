/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * GIN-SDMA all-to-all for single-node (scaleup-only) communicators.
 *
 * This translation unit is compiled with NCCL_GIN_ANVIL_SDMA_ENABLE=1 and
 * NCCL_GIN_PROXY_ENABLE=0 so that NCCL_GIN_BACKEND_MASK_ALL has a single bit
 * set and ncclGinCallImpl resolves the backend at compile time.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "gin_alltoall.h"

#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "dev_runtime.h"
#include "param.h"

#include <cuda_runtime.h>

#include <cstdint>

NCCL_PARAM(GinA2AEnable, "GIN_A2A_ENABLE", 1)
NCCL_PARAM(GinA2AMinBytes, "GIN_A2A_MIN_BYTES", 0)

namespace {

// True only when the SDMA device templates were compiled into this TU.
constexpr bool kSdmaDeviceBackendCompiled = (NCCL_GIN_ANVIL_SDMA_ENABLE != 0);

// markSdmaDirty packs (peer, channel) into a 64-bit mask and drops anything past
// bit 64. The scaleup path uses one channel, so this bound keeps us in range.
constexpr int kGinA2AMaxRanks = 16;

// This kernel only queues one put per peer and the SDMA engines do the copying.
// That is little enough work for a single CTA, and more would not copy faster.
constexpr int kGinA2ACtas = 1;

__global__ void ncclGinA2ASdmaKernel(ncclWindow_t sendWin, size_t sendOff, ncclWindow_t recvWin, size_t recvOff,
                                     size_t bytesPerPeer, struct ncclDevComm devComm) {
  constexpr int ginContext = 0;
  unsigned int signalIndex = blockIdx.x;
  ncclGin gin{devComm, ginContext};
  uint64_t signalValue = gin.readSignal(signalIndex);

  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x};
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  for (int r = tid; r < devComm.nRanks; r += nthreads) {
    gin.put(ncclTeamWorld(devComm), r, recvWin, recvOff + devComm.rank * bytesPerPeer, sendWin,
            sendOff + r * bytesPerPeer, bytesPerPeer, ncclGin_SignalInc{signalIndex});
  }

  int receivingCta = (devComm.rank % nthreads) / blockDim.x;
  if (blockIdx.x == receivingCta) {
    gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  }
  gin.flush(ncclCoopCta());

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

ncclResult_t ncclGinA2AInitOnce(ncclComm* comm) {
  NCCLCHECK(ncclDevrInitOnce(comm));
  struct ncclGinA2AState* state = &comm->ginA2AState;
  if (!state->initialized) {
    struct ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = kGinA2AMaxCtas;
    reqs.ginSignalCount = kGinA2AMaxCtas;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    NCCLCHECK(ncclDevrCommCreateInternal(comm, &reqs, &state->devComm, /*isInternal=*/true));
    state->initialized = true;
  }
  return ncclSuccess;
}

} // namespace

bool ncclAllToAllGinSdmaEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                 ncclDataType_t datatype) {
  if (!kSdmaDeviceBackendCompiled) return false;
  if (!ncclParamGinA2AEnable()) return false;
  if (comm == nullptr || comm->bootstrap == nullptr) return false;
  if (count == 0) return false;

  // Scaleup only: every rank must be reachable over LSA.
  if (comm->nNodes != 1) return false;
  if (ncclTeamLsa(comm).nRanks != comm->nRanks) return false;
  if (comm->nRanks > kGinA2AMaxRanks) return false;

  if (!comm->symmetricSupport) return false;
  if (comm->globalGinSupport != NCCL_GIN_CONNECTION_FULL) return false;

  // This path runs on the comm's shared GIN backend, so it has to be SDMA.
  if (comm->sharedRes->ginState.ginType != (ncclGinType_t)NCCL_NET_DEVICE_GIN_ANVIL_SDMA) return false;

  // sendbuff == recvbuff is accepted. ncclAlltoAll defines no in-place semantics
  // (unlike ncclAlltoAllv), and the reference GIN kernel likewise runs the same
  // puts with the send and receive windows aliased.
  size_t bytesPerPeer = count * ncclTypeSize(datatype);
  if (bytesPerPeer < (size_t)ncclParamGinA2AMinBytes()) return false;

  // Both buffers must already be symmetrically registered.
  struct ncclDevrWindow* sendWin = nullptr;
  struct ncclDevrWindow* recvWin = nullptr;
  if (ncclDevrFindWindow(comm, sendbuff, &sendWin) != ncclSuccess || sendWin == nullptr) return false;
  if (ncclDevrFindWindow(comm, recvbuff, &recvWin) != ncclSuccess || recvWin == nullptr) return false;

  return true;
}

ncclResult_t ncclAllToAllGinSdma(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                 ncclComm* comm, cudaStream_t stream) {
  struct ncclDevrWindow* sendWin = nullptr;
  struct ncclDevrWindow* recvWin = nullptr;
  NCCLCHECK(ncclDevrFindWindow(comm, sendbuff, &sendWin));
  NCCLCHECK(ncclDevrFindWindow(comm, recvbuff, &recvWin));
  if (sendWin == nullptr || recvWin == nullptr) return ncclInvalidUsage;

  NCCLCHECK(ncclGinA2AInitOnce(comm));

  size_t sendOff = (uint8_t*)sendbuff - (uint8_t*)sendWin->userPtr;
  size_t recvOff = (uint8_t*)recvbuff - (uint8_t*)recvWin->userPtr;
  size_t bytesPerPeer = count * ncclTypeSize(datatype);

  ncclGinA2ASdmaKernel<<<kGinA2ACtas, kGinA2AThreadsPerCta, 0, stream>>>(
    sendWin->vidmem, sendOff, recvWin->vidmem, recvOff, bytesPerPeer, comm->ginA2AState.devComm);
  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

ncclResult_t ncclGinA2AFinalize(ncclComm* comm) {
  struct ncclGinA2AState* state = &comm->ginA2AState;
  if (state->initialized) {
    NCCLCHECK(ncclDevCommDestroy(comm, &state->devComm));
    state->initialized = false;
  }
  return ncclSuccess;
}
