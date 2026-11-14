/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "model.h"

#include "comm.h"
#include "core.h"
#include "cudawrap.h" // RUBIN_AND_LATER

#include <algorithm>
#include <cfloat>
#include <cmath>

// RCCL caps a few LSA kernels below ncclSymkMaxBlocks (or, for the hierarchical AllGather, above it).
static int maxBlocksLsa(struct ncclComm* comm, enum ncclSymkKernelId kernelId) {
  switch (kernelId) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  case ncclSymkKernelId_AllGather_ST:
    return ncclSymkIsGfx950(comm) ? std::min(ncclSymkMaxBlocks, 24) : ncclSymkMaxBlocks;
#endif
  case ncclSymkKernelId_AllGather_HierLsaST:
    return ncclSymkAgHierMaxBlocks;
  case ncclSymkKernelId_AllGather_HierLsaTmaST:
    return ncclSymkAgHierTmaMaxBlocks;
  default:
    (void)comm;
    return ncclSymkMaxBlocks;
  }
}

// Kernels that stage tiles through the Tensor Data Mover.
static bool usesTdm(struct ncclComm* comm, enum ncclSymkKernelId kernelId) {
#if defined(__HIP_PLATFORM_AMD__)
  return (ncclSymkTmaKernelMask() >> kernelId & 1) && ncclSymkTmaAvailable(comm);
#else
  (void)comm, (void)kernelId;
  return false;
#endif
}

static ncclResult_t evaluateLsaEstimate(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes,
                                        int nBlocks, struct ncclSymkLsaEstimate* estimate, bool* modeled) {
  float selectionCostPercent = 0.0250f;
  NCCLCHECK(ncclSymkLsaA2AModel(input, kernelId, nBlocks, &estimate->timeUs, modeled));
  if (*modeled) {
    estimate->ctaSelectionTimeUs = estimate->timeUs;
#if defined(__HIP_PLATFORM_AMD__)
    // [RCCL] No NVLink VR reduction on AMD; AMD compCap values (e.g. gfx1150 = 115) overlap RUBIN_AND_LATER.
    bool vrReduction = false;
#else
    bool vrReduction = RUBIN_AND_LATER(input->comm->minCompCap) &&
                       (input->func == ncclFuncReduceScatter || input->func == ncclFuncAllReduce);
#endif
    selectionCostPercent = vrReduction ? 0.0100f : 0.0200f;
  } else {
    NCCLCHECK(ncclSymkLsaBaseModel(input, kernelId, nBytes, nBlocks, estimate, modeled));
    if (!*modeled) return ncclSuccess;
  }
  estimate->selectionTimeUs = estimate->timeUs * (1.0f + selectionCostPercent * nBlocks);
  return ncclSuccess;
}

// Select the CTA count with the shared policy using either the A2A or base model.
ncclResult_t ncclSymkLsaModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes,
                              float* timeUs, float* selectionTimeUs, int* nBlocks) {
  struct ncclComm* comm = input->comm;
  int nMaxBlocks = std::min<int>(maxBlocksLsa(comm, kernelId), input->maxCTAs);
  int nMinBlocks = std::min(input->minCTAs, nMaxBlocks);

  *timeUs = FLT_MAX;
  *nBlocks = 0;

  // minCTAs/maxCTAs are resolved (env > per-call > comm) at task-append time.
  // NCCL_SYM_CTAS is an explicit override of the resolved bounds.
  int nUserCTAs = ncclSymkModelCtasEnvOverride(kernelId);
  if (nUserCTAs > 0) nMinBlocks = nMaxBlocks = nUserCTAs;

  // Even CTA counts are preferred for optimal performance, except for when CTAs==1.
  if (nMinBlocks != nMaxBlocks) {
    if (nMinBlocks != 1) nMinBlocks = roundUp(nMinBlocks, 2);
    if (nMaxBlocks != 1) nMaxBlocks = roundDown(nMaxBlocks, 2);
  }

  if (ncclSymkTmaKernelMask() >> kernelId & 1) {
    size_t maxWorkBytes = input->countMax * ncclTypeSize(input->datatype);
    if (!ncclSymkTmaDeepEligible(comm, kernelId, maxWorkBytes, nMinBlocks)) {
      const char* symKernelIdEnv = ncclGetEnv("NCCL_SYM_KERNEL");
      if (symKernelIdEnv) {
        INFO(NCCL_TUNING,
             "NCCL_SYM_KERNEL set to %s. At largest grouped work size %zu Bytes, kernel will not exercise TMA paths.",
             symKernelIdEnv, maxWorkBytes);
      } else if (ncclSymkTmaForced(comm)) {
        INFO(NCCL_TUNING,
             "NCCL_SYM_TMA_ENABLE=2 forces %s. At largest grouped work size %zu Bytes, kernel will not exercise TMA "
             "paths.",
             ncclSymkKernelIdToString(kernelId), maxWorkBytes);
      } else {
        return ncclSuccess;
      }
    } else {
      while (nMaxBlocks > nMinBlocks && !ncclSymkTmaDeepEligible(comm, kernelId, maxWorkBytes, nMaxBlocks)) {
        nMaxBlocks -= (nMaxBlocks == 2 ? 1 : 2);
      }
    }
  }

  double withinPeakFactor = 1.025;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  withinPeakFactor = rcclSymkLsaWithinPeakFactor(kernelId);
#endif

  // maxBlocksLsa() lets the hierarchical AllGather exceed ncclSymkMaxBlocks.
  struct ncclSymkLsaEstimate candidates[std::max(ncclSymkMaxBlocks, ncclSymkAgHierMaxBlocks) + 1];
  bool modeled = false;
  NCCLCHECK(evaluateLsaEstimate(input, kernelId, nBytes, nMaxBlocks, &candidates[nMaxBlocks], &modeled));
  if (!modeled) return ncclSuccess;
  *nBlocks = nMaxBlocks;
  // Evaluate every candidate before choosing the fewest CTAs within withinPeakFactor of the best estimate
  // (2.5% upstream; per-kernel via rcclSymkLsaWithinPeakFactor on AMD).
  double bestCtaSelectionTimeUs = candidates[nMaxBlocks].ctaSelectionTimeUs;
  for (int candidate = nMinBlocks; candidate < nMaxBlocks; candidate += candidate == 1 ? 1 : 2) {
    struct ncclSymkLsaEstimate& candidateEstimate = candidates[candidate];
    NCCLCHECK(evaluateLsaEstimate(input, kernelId, nBytes, candidate, &candidateEstimate, &modeled));
    if (!modeled) {
      candidateEstimate.ctaSelectionTimeUs = INFINITY;
      continue;
    }
    bestCtaSelectionTimeUs = std::min(bestCtaSelectionTimeUs, candidateEstimate.ctaSelectionTimeUs);
  }
  for (int candidate = nMinBlocks; candidate < nMaxBlocks; candidate += candidate == 1 ? 1 : 2) {
    if (candidates[candidate].ctaSelectionTimeUs <= withinPeakFactor * bestCtaSelectionTimeUs) {
      *nBlocks = candidate;
      break;
    }
  }
  *timeUs = candidates[*nBlocks].timeUs;
  *selectionTimeUs = candidates[*nBlocks].selectionTimeUs;

  // TDM throughput keeps scaling with resident blocks, so launch it at the widest allowed count.
  if (usesTdm(comm, kernelId)) *nBlocks = nMaxBlocks;
  if (kernelId == ncclSymkKernelId_AllGather_HierLsaST || kernelId == ncclSymkKernelId_AllGather_HierLsaTmaST) {
    *nBlocks = ncclSymkAgHierLaunchBlocks(comm, *nBlocks);
  }
  return ncclSuccess;
}
