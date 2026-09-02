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

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
// Thresholds bounding the block width of the gfx950 LD reduce kernels, measured on 8 ranks.
// ReduceScatter's are bus bytes, AllReduce's are message bytes since nBytes counts all ranks.
static constexpr size_t ncclSymkRsWideBlockMinBusBytes = 1 << 20;
static constexpr size_t ncclSymkRsNarrowBlockBusBytes = 16 << 20;
static constexpr size_t ncclSymkArTailSaturatedBytes = 512 << 10;
static constexpr size_t ncclSymkArDeepTierBytes = 2 << 20;
static constexpr size_t ncclSymkArOccupancyBoundBytes = 1 << 30;
// Below this message size AllReduce's LL packs fit few enough epochs that a wider block only adds
// threads to the epoch barrier without removing an epoch.
static constexpr size_t ncclSymkArLLWideBytes = 64 << 10;
// Block widths those thresholds select between. 1024 is the widest workgroup gfx950 will launch.
static constexpr int ncclSymkGfx950NarrowThreads = 256;
static constexpr int ncclSymkGfx950WideThreads = 512;
static constexpr int ncclSymkGfx950WidestThreads = 1024;
#endif

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
  // Only the gfx950 reduce kernels are tuned. GIN carves its pipeline roles out of blockDim.x and
  // symCheckTmaLaunch() requires the full launch for Tma, so both keep it.
  struct ncclComm* comm = inputs->comm;
  ncclFunc_t coll = inputs->func;
  size_t nBytes = inputs->nBytes;
  int nThreads = ncclSymkMaxThreads;
  bool isLL = (tuning_kmask & ncclSymkLLKernelMask()) != 0;
  bool isReduceColl = coll == ncclFuncReduceScatter || coll == ncclFuncAllReduce;
  bool fullWidth = (ncclSymkGinKernelMask() | ncclSymkTmaKernelMask()) >> tuning->symKernelId & 1;
  if (fullWidth) {
    nThreads = ncclSymkWarpsPerBlock * comm->WarpSize;
  } else if (ncclSymkIsGfx950(comm) && isReduceColl) {
    if (isLL) {
      // AllReduce narrows below the threshold and picks that width up from blockDim. ReduceScatter
      // always stays at the full width, which its device code hardcodes as the loop stride.
      bool narrowLL = coll == ncclFuncAllReduce && nBytes < ncclSymkArLLWideBytes;
      nThreads = narrowLL ? ncclSymkGfx950NarrowThreads : ncclSymkGfx950LLThreads;
    } else if (coll == ncclFuncReduceScatter) {
      // Small sizes are latency bound on per-peer loads and want every thread. Large ones are
      // bandwidth bound, where a narrower block keeps iterations per globally strided warp high.
      size_t busBytes = size_t(comm->nRanks) * nBytes;
      if (busBytes >= ncclSymkRsNarrowBlockBusBytes) {
        nThreads = ncclSymkGfx950NarrowThreads;
      } else if (busBytes >= ncclSymkRsWideBlockMinBusBytes) {
        nThreads = ncclSymkGfx950WidestThreads;
      } else {
        nThreads = ncclSymkGfx950WideThreads;
      }
    } else {
      // AllReduce folds rank into its thread index, so across the deep tiers a wider block halves
      // iterations per warp rather than covering more GPU. Outside them the wider block wins.
      bool narrowBlock = nBytes < ncclSymkArTailSaturatedBytes ||
                         (ncclSymkArDeepTierBytes <= nBytes && nBytes < ncclSymkArOccupancyBoundBytes);
      nThreads = narrowBlock ? ncclSymkGfx950NarrowThreads : ncclSymkGfx950WideThreads;
    }
  }
  tuning->nWarps = std::max(1, nThreads / comm->WarpSize);
#else
  tuning->nWarps = 16;
#endif
  return ret;
}
