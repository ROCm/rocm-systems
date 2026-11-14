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
#include "group.h"

#include <cfloat>
#include <cmath>
#include <algorithm>

NCCL_PARAM(SymCTAs, "SYM_CTAS", 0)

static constexpr float disableTime = 1.e30f;

int ncclSymkModelCtasEnvOverride(enum ncclSymkKernelId kernelId) {
  int maxBlocks = ncclSymkMaxBlocks;
  if (kernelId == ncclSymkKernelId_AllGather_HierLsaST) {
    maxBlocks = ncclSymkAgHierMaxBlocks;
  }
  if (kernelId == ncclSymkKernelId_AllGather_HierLsaTmaST) {
    maxBlocks = ncclSymkAgHierTmaMaxBlocks;
  }
  int64_t nUserCTAs = ncclParamSymCTAs();
  if (nUserCTAs < 1) return 0;
  if (nUserCTAs > maxBlocks) return maxBlocks;
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
static bool rcclSymkFuncForced(struct ncclComm* comm, ncclFunc_t func) {
  int tuningFunc = ncclTuningFuncIndex(func);
  return tuningFunc >= 0 && comm->tuningContext.forced[tuningFunc];
}

// On gfx950 AllGather's store kernel overtakes LL well before the shared cost model switches, so LL
// gives way wherever the store kernel can run, unless the user forced or narrowed the choice.
static bool rcclSymkAllGatherLLYieldsToST(struct ncclTuningInput_t* const inputs, enum ncclSymkKernelId kernelId,
                                          ncclSymkKernelMask valid_kmask) {
  struct ncclComm* comm = inputs->comm;
  return kernelId == ncclSymkKernelId_AllGather_LL && ncclSymkIsGfx950(comm) &&
         !rcclSymkFuncForced(comm, inputs->func) && (valid_kmask >> ncclSymkKernelId_AllGather_ST & 1) &&
         (inputs->tuningMask >> (NCCL_TUNING_SYM_KERNEL_ID_OFFSET + ncclSymkKernelId_AllGather_ST) & 1) &&
         (inputs->winRegType == ncclSymSendRegRecvReg || inputs->winRegType == ncclSymSendNonregRecvReg) &&
         ncclSymkGfx950AllGatherPrefersStore(comm->nRanks, inputs->nBytes);
}

// In gfx1250 partition mode AllGather runs LL, then the store kernel, then the hierarchical kernel, each
// from where it overtakes the one before, unless the user forced the choice. If either of the latter two
// can't run, the other covers its range. Below the store kernel's threshold the cost model picks LL, and
// the next kernel stays in for the launches LL cannot take, such as grouped ones.
static bool rcclSymkAllGatherYieldsInPartitionMode(struct ncclTuningInput_t* const inputs,
                                                   enum ncclSymkKernelId kernelId, ncclSymkKernelMask valid_kmask) {
  constexpr int vecSt = ncclSymkKernelId_AllGather_ST;
  constexpr int tmaSt = ncclSymkKernelId_AllGather_TmaST;
  constexpr int vecHier = ncclSymkKernelId_AllGather_HierLsaST;
  constexpr int tmaHier = ncclSymkKernelId_AllGather_HierLsaTmaST;
  struct ncclComm* comm = inputs->comm;
  if (inputs->func != ncclFuncAllGather || !ncclSymkIsGfx1250(comm) || !comm->hasMloPart ||
      rcclSymkFuncForced(comm, inputs->func) ||
      (inputs->winRegType != ncclSymSendRegRecvReg && inputs->winRegType != ncclSymSendNonregRecvReg)) {
    return false;
  }
  struct AllowMask {
    ncclSymkKernelMask bits;
    bool has(int id) const { return bits >> id & 1; }
  };
  const AllowMask allowed{valid_kmask & ncclSymkKernelMask(inputs->tuningMask >> NCCL_TUNING_SYM_KERNEL_ID_OFFSET)};
  int st = allowed.has(vecSt) ? vecSt : tmaSt;
  int hier = allowed.has(vecHier) ? vecHier : tmaHier;
  bool hasSt = allowed.has(st);
  bool hasHier = allowed.has(hier);
  if (!hasSt && !hasHier) return false;
  int afterLL = hasSt ? st : hier;
  if (hasHier && ncclSymkGfx1250AllGatherPrefersHier(comm->nRanks, inputs->nBytes)) return kernelId != hier;
  if (ncclSymkGfx1250AllGatherPrefersStore(comm->nRanks, inputs->nBytes)) return kernelId != afterLL;
  return kernelId != afterLL && kernelId != ncclSymkKernelId_AllGather_LL;
}

static int rcclSymkCalculateWarps(struct ncclTuningInput_t* const inputs, enum ncclSymkKernelId kernelId) {
  struct ncclComm* comm = inputs->comm;
  bool isLL = ncclSymkLLKernelMask() >> kernelId & 1;
  bool isLsa = ncclSymkLsaKernelMask() >> kernelId & 1;
  // GIN RS carves its pipeline roles out of blockDim.x and symCheckTmaLaunch() requires the full launch
  // for Tma, so both keep it. The GIN+LSA AG kernel sizes itself from ncclSymkMaxThreads.
  bool fullWidth = ((ncclSymkGinKernelMask() & ~ncclSymkAGKernelMask()) | ncclSymkTmaKernelMask()) >> kernelId & 1;
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

ncclResult_t ncclTuningSymkModelSim(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning,
                                    struct ncclTuningModelState* /*internal*/) {
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

  ncclSymkKernelMask tuning_kmask = 1ull << tuning->symKernelId;
  ncclSymkKernelMask valid_kmask = ncclSymkMask(inputs->comm, inputs->func, inputs->devRedOp, inputs->datatype,
                                                inputs->countMax, inputs->symAligned16B);
  ncclSymkKernelMask window_optional_kmask = ncclSymkLLKernelMask() | ncclGenkKernelMask();
  // Kernels that don't support grouping -- these are symmetric LL kernels.
  ncclSymkKernelMask ungrouped_kmask = ncclSymkLLKernelMask() & ~ncclGenkKernelMask();
  // Symmetric subset of window_optional_kmask.  These kernels may not be available with multiple GPUs per thread,
  // unless the symk devComm has already been initialized.
  ncclSymkKernelMask window_optional_symk_kmask = ungrouped_kmask;
  // Kernels that don't require symmetric user buffers but that have this capability disabled by default.
  ncclSymkKernelMask nowin_kmask = (ncclParamSymNoWinEnable() ? 0 : ungrouped_kmask);
  if ((tuning_kmask & valid_kmask) == 0) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ncclSuccess;
  }

  // The specialized LL kernels do not support grouping; Flow protocols do.
  // While the specialized LL kernels don't require symmetric windows, we currently disable them by default if neither
  // input nor output is symmetric, to avoid performance regressions in common cases (only until the new symmetric cost
  // model is complete).  They can be enabled via NCCL_SYM_NOWIN_ENABLE=1, with the exception of the case of multiple
  // GPUs per thread if the internal symk devComm is not yet initialized (restriction to be removed when the new enqueue
  // becomes the default).
  bool isOneThreadMultiGpus = (inputs->comm->intraRanks > 1 && !ncclParamSingleProcMemRegEnable());
  if ((inputs->nWorks > 1 && (tuning_kmask & ungrouped_kmask) != 0) ||
      (inputs->winRegType == ncclSymSendNonregRecvNonreg &&
       ((tuning_kmask & nowin_kmask) != 0 || ((tuning_kmask & window_optional_symk_kmask) != 0 &&
                                              isOneThreadMultiGpus && !inputs->comm->symkState.initialized)))) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ncclSuccess;
  }

  if ((inputs->func == ncclFuncAllReduce && inputs->winRegType != ncclSymSendRegRecvReg &&
       (tuning_kmask & window_optional_kmask) == 0) ||
      ((inputs->func == ncclFuncBroadcast || inputs->func == ncclFuncAllGather) &&
       inputs->winRegType != ncclSymSendRegRecvReg && inputs->winRegType != ncclSymSendNonregRecvReg &&
       (tuning_kmask & window_optional_kmask) == 0) ||
      ((inputs->func == ncclFuncReduce || inputs->func == ncclFuncReduceScatter) &&
       inputs->winRegType != ncclSymSendRegRecvReg && inputs->winRegType != ncclSymSendRegRecvNonreg &&
       (tuning_kmask & window_optional_kmask) == 0) ||
      (inputs->func == ncclFuncAllGather && inputs->winRegType != ncclSymSendRegRecvReg && inputs->comm->nNodes > 1 &&
       (tuning_kmask & ncclSymkGinKernelMask()) != 0)) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ncclSuccess;
  }

  if ((tuning_kmask & ncclGenkKernelMask()) != 0) return ncclTuningGenkModelSim(inputs, tuning);

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (rcclSymkAllGatherLLYieldsToST(inputs, (enum ncclSymkKernelId)tuning->symKernelId, valid_kmask) ||
      rcclSymkAllGatherYieldsInPartitionMode(inputs, (enum ncclSymkKernelId)tuning->symKernelId, valid_kmask)) {
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
  tuning->nWarps = tuning->symKernelId == ncclSymkKernelId_AlltoAll_LL ?
                     ncclSymkKernelMaxWarps(ncclSymkKernelId_AlltoAll_LL, inputs->devRedOp, inputs->datatype) :
                     rcclSymkCalculateWarps(inputs, (enum ncclSymkKernelId)tuning->symKernelId);
#else
  tuning->nWarps = tuning->symKernelId == ncclSymkKernelId_AlltoAll_LL ?
                     ncclSymkKernelMaxWarps(ncclSymkKernelId_AlltoAll_LL, inputs->devRedOp, inputs->datatype) :
                     ncclSymkMaxThreads / WARP_SIZE;
#endif
  return ret;
}
