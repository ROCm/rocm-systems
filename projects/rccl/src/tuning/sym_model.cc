/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "cost_model.h"
#include "sym_kernels.h"
#include "sym_model/model.h"

#include "comm.h"
#include "core.h"

#include <cfloat>
#include <cmath>
#include <algorithm>

NCCL_PARAM(SymCTAs, "SYM_CTAS", 0)

static constexpr float disableTime = 1.e30f;

int ncclSymkModelCtasEnvOverride() {
  int64_t nUserCTAs = ncclParamSymCTAs();
  if (nUserCTAs < 1) return 0;
  if (nUserCTAs > ncclSymkMaxBlocks) return ncclSymkMaxBlocks;
  return static_cast<int>(nUserCTAs);
}

static ncclResult_t queryModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes,
                               float* timeUs, float* selectionTimeUs, int* nBlocks) {
  if (ncclSymkGinKernelMask() >> kernelId & 1) {
    NCCLCHECK(ncclSymkGinModel(input, kernelId, nBytes, timeUs, nBlocks));
    *selectionTimeUs = *timeUs;
  } else {
    NCCLCHECK(ncclSymkLsaModel(input, kernelId, nBytes, timeUs, selectionTimeUs, nBlocks));
  }
  return ncclSuccess;
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
// On gfx950 AllGather's store kernel overtakes LL well before the shared cost model switches, so LL
// gives way wherever the store kernel can run, unless the user forced or narrowed the choice.
static bool rcclSymkAllGatherLLYieldsToST(struct ncclTuningInput_t* const inputs, enum ncclSymkKernelId kernelId,
                                          uint32_t valid_kmask) {
  struct ncclComm* comm = inputs->comm;
  return kernelId == ncclSymkKernelId_AllGather_LL && ncclSymkIsGfx950(comm) &&
         !comm->tuningContext.forced[inputs->func] && (valid_kmask >> ncclSymkKernelId_AllGather_ST & 1) &&
         (inputs->tuningMask >> (NCCL_TUNING_SYM_KERNEL_ID_OFFSET + ncclSymkKernelId_AllGather_ST) & 1) &&
         (inputs->winRegType == ncclSymSendRegRecvReg || inputs->winRegType == ncclSymSendNonregRecvReg) &&
         ncclSymkGfx950AllGatherPrefersStore(comm->nRanks, inputs->nBytes);
}

static int rcclSymkCalculateWarps(struct ncclTuningInput_t* const inputs, enum ncclSymkKernelId kernelId) {
  struct ncclComm* comm = inputs->comm;
  bool isLL = ncclSymkLLKernelMask() >> kernelId & 1;
  bool isLsa = ncclSymkLsaKernelMask() >> kernelId & 1;
  // GIN carves its pipeline roles out of blockDim.x and symCheckTmaLaunch() requires the full launch
  // for Tma, so both keep it.
  bool fullWidth = (ncclSymkGinKernelMask() | ncclSymkTmaKernelMask()) >> kernelId & 1;
  int nThreads = ncclSymkMaxThreads;
  if (fullWidth) {
    nThreads = ncclSymkWarpsPerBlock * comm->WarpSize;
  } else if (ncclSymkIsGfx950(comm) && isLsa) {
    // The width tuning is fitted to the gfx950 LSA kernels.
    nThreads = ncclSymkGfx950BlockThreads(inputs->func, isLL, comm->nRanks, inputs->nBytes);
  }
  return std::max(1, nThreads / comm->WarpSize);
}
#endif

ncclResult_t ncclTuningSymkModelSim(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning) {
  ncclResult_t ret = ncclSuccess;
  tuning->selectionTimeUs = NCCL_TUNING_IGNORE;

  if (tuning->symKernelId == ncclSymkKernelId_Count) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ncclSuccess;
  }

  if (!ncclSymkAvailable(inputs->comm, inputs->func, inputs->devRedOp, inputs->datatype, inputs->count)) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ncclSuccess;
  }

  uint32_t tuning_kmask = (1 << tuning->symKernelId);
  uint32_t valid_kmask = ncclSymkMask(inputs->comm, inputs->func, inputs->devRedOp, inputs->datatype, inputs->countMax,
                                      inputs->symAligned16B);
  if ((tuning_kmask & valid_kmask) == 0) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ncclSuccess;
  }

  if ((inputs->nWorks > 1 &&
       ((tuning_kmask & ncclSymkLLKernelMask()) != 0)) // We currently don't support grouping for LL kernels.
      || (inputs->func == ncclFuncAllReduce && inputs->winRegType != ncclSymSendRegRecvReg &&
          (tuning_kmask & ncclSymkLLKernelMask()) == 0) ||
      (inputs->func == ncclFuncAllGather && inputs->winRegType != ncclSymSendRegRecvReg &&
       inputs->winRegType != ncclSymSendNonregRecvReg && (tuning_kmask & ncclSymkLLKernelMask()) == 0) ||
      (inputs->func == ncclFuncReduceScatter && inputs->winRegType != ncclSymSendRegRecvReg &&
       inputs->winRegType != ncclSymSendRegRecvNonreg && (tuning_kmask & ncclSymkLLKernelMask()) == 0) ||
      (inputs->func == ncclFuncAllGather && inputs->winRegType != ncclSymSendRegRecvReg && inputs->comm->nNodes > 1 &&
       (tuning_kmask & ncclSymkGinKernelMask()) != 0)) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ncclSuccess;
  }

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (rcclSymkAllGatherLLYieldsToST(inputs, (enum ncclSymkKernelId)tuning->symKernelId, valid_kmask)) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ncclSuccess;
  }
#endif

  float kTime = FLT_MAX;
  float kSelectionTime = FLT_MAX;
  int kBlocks = 0;
  NCCLCHECK(queryModel(inputs, (enum ncclSymkKernelId)tuning->symKernelId, inputs->nBytes, &kTime, &kSelectionTime,
                       &kBlocks));
  if (kBlocks <= 0 || !std::isfinite(kTime) || kTime >= disableTime) {
    tuning->valid = 0;
    tuning->timeUs = -1.0f;
    tuning->nChannels = 0;
    return ncclSuccess;
  }

  tuning->timeUs = kTime;
  tuning->selectionTimeUs = kSelectionTime;
  tuning->nChannels = kBlocks;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  tuning->maxChannels = kBlocks;
  tuning->nWarps = rcclSymkCalculateWarps(inputs, (enum ncclSymkKernelId)tuning->symKernelId);
#else
  tuning->nWarps = 16;
#endif
  return ret;
}
