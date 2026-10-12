/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "model.h"

#include "comm.h"
#include "core.h"

#include <cmath>

static double softmin(double x, double ceiling, double softness) {
  // looks like a smooth version of: min(x, ceiling)
  return ceiling - softness * std::log1p((std::exp(ceiling / softness) - 1) * std::exp(-x / softness));
}

static double softplus(double x, double softness) {
  // looks like a smooth version of: max(0, x)
  double z = x / softness;
  return 100.0 <= z ? x : softness * std::log1p(std::exp(z));
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
RCCL_PARAM(SymModel, "SYM_MODEL", 0)

enum struct rcclSymkColl : int {
  AllReduce = 0,
  AllGather = 1,
  ReduceScatter = 2,
  Count = 3
};
enum struct rcclSymkProto : int {
  LL = 0,
  Simple = 1,
  Count = 2
};

constexpr int rcclSymkCollCount = static_cast<int>(rcclSymkColl::Count);
constexpr int rcclSymkProtoCount = static_cast<int>(rcclSymkProto::Count);

struct rcclSymkTuningModel {
  double baseLat[rcclSymkCollCount][rcclSymkProtoCount];
  double smBw[rcclSymkCollCount][rcclSymkProtoCount];
  double peakBw[rcclSymkCollCount];
  double llBusFactor[rcclSymkCollCount];
  double withinPeakFactor[rcclSymkCollCount][rcclSymkProtoCount];
  // The NaN-flag kernels: latency (us), then bandwidth (GB/s of pushes per rank, the local one
  // included) that each block adds up to a peak the blocks share, blended harmonically: a block's
  // pushes are limited by the stores it keeps in flight, and every block's by the links. The
  // two-shot AllReduce has its own. Pushing through the Tensor Data Mover speeds up blocks that
  // take more than an epoch, and costs a little on the rest. Only a block per CU runs at once, so
  // a grid past the CU count adds a wave.
  double nanLat;
  double nanBlockBw;
  double nanPeakBw;
  double nanTwoShotLat;
  double nanTwoShotBlockBw;
  double nanTwoShotPeakBw;
  double nanTdmBwGain;
  double nanTdmLat;
  double nanEpochLat;
  double nanExtraWaveLat;
  double nanWithinPeakFactor;
};

// Default tuning model.
static constexpr struct rcclSymkTuningModel rcclSymkTuningModel_0 = {
  .baseLat = {
             //         LL     Simple
    /* AR */ {11.0, 19.5},
    /* AG */ {8.5, 13.0},
    /* RS */ {11.0, 15.0},
  },
  .smBw = {{25.0, 5.0}, {22.0, 5.0}, {10.0, 20.0}},
  .peakBw = {800.0, 1200.0, 1200.0},
  // The higher, the more conservative the model (less LL usage, more ST usage)
  .llBusFactor = {12.0, 4.0, 3.0},
  // The higher, the more conservative the model (less CTAs)
  .withinPeakFactor = {{1.100, 1.005}, {1.015, 1.015}, {1.025, 1.005}},
  .nanLat = 7.0,
  .nanBlockBw = 8.0,
  .nanPeakBw = 350.0,
  .nanTwoShotLat = 10.3,
  .nanTwoShotBlockBw = 8.0,
  .nanTwoShotPeakBw = 260.0,
  .nanTdmBwGain = 1.15,
  .nanTdmLat = 0.4,
  .nanEpochLat = 8.0,
  .nanExtraWaveLat = 3.5,
  .nanWithinPeakFactor = 1.01,
};

// Alternate ReduceScatter tuning, selected with SYM_MODEL=1.
static constexpr struct rcclSymkTuningModel rcclSymkTuningModel_1 = {
  .baseLat = {
             //         LL     Simple
    /* AR */ {11.0, 19.5},
    /* AG */ {8.5, 13.0},
    /* RS */ {11.0, 13.0},
  },
  .smBw = {{25.0, 5.0}, {22.0, 5.0}, {25.0, 20.0}},
  .peakBw = {800.0, 1200.0, 1200.0},
  // The higher, the more conservative the model (less LL usage, more ST usage)
  .llBusFactor = {12.0, 4.0, 9.0},
  // The higher, the more conservative the model (less CTAs)
  .withinPeakFactor = {{1.100, 1.005}, {1.015, 1.015}, {1.025, 1.025}},
  .nanLat = 7.0,
  .nanBlockBw = 8.0,
  .nanPeakBw = 350.0,
  .nanTwoShotLat = 10.3,
  .nanTwoShotBlockBw = 8.0,
  .nanTwoShotPeakBw = 260.0,
  .nanTdmBwGain = 1.15,
  .nanTdmLat = 0.4,
  .nanEpochLat = 8.0,
  .nanExtraWaveLat = 3.5,
  .nanWithinPeakFactor = 1.01,
};

static constexpr struct rcclSymkTuningModel rcclSymkTuningModels[] = {rcclSymkTuningModel_0, rcclSymkTuningModel_1};
static constexpr int rcclSymkTuningModelCount = int(sizeof(rcclSymkTuningModels) / sizeof(rcclSymkTuningModels[0]));

static int rcclSymkTuningModelIndex() {
  static int s_cache = -1;
  if (s_cache < 0) {
    int64_t env = rcclParamSymModel();
    if (env < 0 || env >= rcclSymkTuningModelCount) {
      INFO(NCCL_ENV, "RCCL_SYM_MODEL %ld is out of range [0, %d); using RCCL_SYM_MODEL 0", (long)env,
           rcclSymkTuningModelCount);
      s_cache = 0;
    } else {
      s_cache = (int)env;
    }
  }
  return s_cache;
}

static const struct rcclSymkTuningModel& rcclSymkModelFor(enum ncclSymkKernelId kernelId, int* coll, int* proto) {
  bool isAR = ncclSymkARKernelMask() >> kernelId & 1;
  bool isAG = ncclSymkAGKernelMask() >> kernelId & 1;
  bool isLL = (ncclSymkLLKernelMask() | ncclSymkNanKernelMask()) >> kernelId & 1;
  *coll = static_cast<int>(isAR ? rcclSymkColl::AllReduce : isAG ? rcclSymkColl::AllGather : rcclSymkColl::ReduceScatter);
  *proto = static_cast<int>(isLL ? rcclSymkProto::LL : rcclSymkProto::Simple);
  return rcclSymkTuningModels[rcclSymkTuningModelIndex()];
}

double rcclSymkLsaWithinPeakFactor(enum ncclSymkKernelId kernelId) {
  int c, p;
  const struct rcclSymkTuningModel& m = rcclSymkModelFor(kernelId, &c, &p);
  if (ncclSymkNanKernelMask() >> kernelId & 1) return m.nanWithinPeakFactor;
  return m.withinPeakFactor[c][p];
}

static bool rcclSymkNanModel(struct ncclComm* comm, enum ncclSymkKernelId kernelId, size_t nBytes, int nBlocks,
                             struct ncclSymkLsaEstimate* estimate) {
  int c, p;
  const struct rcclSymkTuningModel& m = rcclSymkModelFor(kernelId, &c, &p);
  constexpr double GBps = 1.e9 / 1.e6;
  // nBytes is the whole buffer for AllReduce, a rank's part for AllGather and ReduceScatter.
  double pushBytes = double(comm->nRanks) * nBytes;
  double lat = m.nanLat, blockBw = m.nanBlockBw, peakBw = m.nanPeakBw;
  if (rcclSymkNanTwoShot(kernelId, nBytes)) {
    // Pushed with stores in either kernel.
    pushBytes = 2.0 * nBytes;
    lat = m.nanTwoShotLat;
    blockBw = m.nanTwoShotBlockBw;
    peakBw = m.nanTwoShotPeakBw;
    if (rcclSymkNanTdm(kernelId)) lat += m.nanTdmLat;
  } else {
    // A block takes its part per source an epoch at a time, each waiting out the last, and past
    // one epoch the partial last one costs about a full one.
    constexpr double epochBytes = ncclSymkNanEpochUnits * 16.0;
    size_t cells = divUp(nBytes, size_t(NCCL_SYM_KERNEL_CELL_SIZE));
    double part = double(divUp(cells, size_t(nBlocks)) * NCCL_SYM_KERNEL_CELL_SIZE);
    double epochs = std::ceil(part / epochBytes);
    if (epochs > 1) {
      pushBytes *= epochs * epochBytes * nBlocks / nBytes;
      lat += (epochs - 1) * m.nanEpochLat;
    }
    if (rcclSymkNanTdm(kernelId)) {
      if (epochs > 1) {
        blockBw *= m.nanTdmBwGain;
        peakBw *= m.nanTdmBwGain;
      } else {
        lat += m.nanTdmLat;
      }
    }
  }
  if (comm->cuCount > 0 && nBlocks > comm->cuCount) lat += m.nanExtraWaveLat;
  double bw = GBps / (1.0 / (nBlocks * blockBw) + 1.0 / peakBw);
  estimate->ctaSelectionTimeUs = lat + pushBytes / bw;
  estimate->timeUs = static_cast<float>(estimate->ctaSelectionTimeUs);
  estimate->selectionTimeUs = estimate->timeUs;
  return std::isfinite(estimate->timeUs) && estimate->timeUs > 0.0f;
}
#endif

bool ncclSymkLsaBaseModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes, int nBlocks,
                          struct ncclSymkLsaEstimate* estimate) {
  constexpr double LL_BusFactor = 9; // 2X the bytes, plus some processing, plus no unrolling

  struct ncclComm* comm = input->comm;
  int nRanks = comm->nRanks;
  size_t busBytes; // max(bytes sent, bytes received)
  double busMultiplier = 1;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (ncclSymkNanKernelMask() >> kernelId & 1) return rcclSymkNanModel(comm, kernelId, nBytes, nBlocks, estimate);
#endif

  switch (kernelId) {
  default:
    busBytes = size_t(1) << 50;
    break;

  case ncclSymkKernelId_AllReduce_AGxLL_R:
    busBytes = nRanks * nBytes * LL_BusFactor;
    break;
  case ncclSymkKernelId_AllReduce_AGxLLMC_R:
    busBytes = nRanks * nBytes * LL_BusFactor;
    busMultiplier = 1.1; // To beat non-MC LL
    break;
  case ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST:
  case ncclSymkKernelId_AllReduce_RSxLD_AGxST:
    busBytes = 2 * nBytes * (nRanks - 1) / nRanks;
    break;
  case ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC:
    busBytes = nBytes / nRanks + nBytes;
    busMultiplier = nRanks;
    break;

  case ncclSymkKernelId_AllGather_LL:
    busBytes = nRanks * nBytes * LL_BusFactor;
    break;
  case ncclSymkKernelId_AllGather_LLMC:
    busBytes = nRanks * nBytes * LL_BusFactor;
    busMultiplier = 1.1; // To beat non-MC LL
    break;
  case ncclSymkKernelId_AllGather_TmaST:
  case ncclSymkKernelId_AllGather_ST:
    busBytes = (nRanks - 1) * nBytes;
    break;
  case ncclSymkKernelId_AllGather_TmaSTMC:
    busMultiplier = 0.99;
    // fall through
  case ncclSymkKernelId_AllGather_STMC:
    busBytes = (nRanks - 1) * nBytes; // Wrong. Should be nRanks*nBytes but we want to beat non-MC.
    busMultiplier *= 0.55 * nRanks;
    break;

  case ncclSymkKernelId_ReduceScatter_LL:
    busBytes = nRanks * nBytes * LL_BusFactor;
    break;
  case ncclSymkKernelId_ReduceScatter_TmaLD:
  case ncclSymkKernelId_ReduceScatter_LD:
    busBytes = (nRanks - 1) * nBytes;
    break;
  case ncclSymkKernelId_ReduceScatter_LDMC:
    busBytes = (nRanks - 1) * nBytes; // Wrong. Should be nRanks*nBytes but we want to beat non-MC.
    busMultiplier = 0.55 * nRanks;
    break;
  }

  bool isTma = ncclSymkTmaKernelMask() >> kernelId & 1;
  bool isLL = ncclSymkLLKernelMask() >> kernelId & 1;
  bool isAG = ncclSymkAGKernelMask() >> kernelId & 1;
  bool isAR = ncclSymkARKernelMask() >> kernelId & 1;
  [[maybe_unused]] bool isRS = ncclSymkRSKernelMask() >> kernelId & 1;
  constexpr double GBps = (1 << 30) / 1.e6;
  double baseLat, smBw, peakBw;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  {
    int c, p;
    const struct rcclSymkTuningModel& m = rcclSymkModelFor(kernelId, &c, &p);
    baseLat = m.baseLat[c][p];
    smBw = m.smBw[c][p] * GBps;
    peakBw = m.peakBw[c] * GBps;
    if (isLL) busBytes *= m.llBusFactor[c] / LL_BusFactor;
    // TDM kernels carry extra setup latency and higher bandwidth.
    if (isTma) {
      double bwGain = 1.0;
      baseLat += 4.0;
      if (kernelId == ncclSymkKernelId_AllGather_TmaST) {
        bwGain = 700.0 / 600.0;
        baseLat += 19.0;
      }
      if (kernelId != ncclSymkKernelId_AllGather_TmaSTMC) bwGain *= 1.07;
      smBw *= bwGain;
      peakBw *= bwGain;
    }
  }
#else
  if (comm->cudaArch < 1000) {
    baseLat = isLL ? 4.5 : 7.8;
    smBw = isAR ? 65 * GBps : 44 * GBps;
    peakBw = kernelId == ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC ? 480 * GBps : 320 * GBps;
  } else {
    baseLat = isLL ? (isAG ? 8.5 : (isRS ? 10.5 : 11.0)) : (isAR ? 19.5 : 13.0);
    smBw = 55 * GBps;
    peakBw = kernelId == ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC ? 1000 * GBps : 600 * GBps;
    if (isRS) peakBw = 650 * GBps;
    if (isTma) {
      baseLat += 4.0;
      if (kernelId == ncclSymkKernelId_AllGather_TmaST) {
        peakBw = 700 * GBps;
        baseLat += 19.0; // This is for optimal CTA and kernel selection.
      }
      if (kernelId != ncclSymkKernelId_AllGather_TmaSTMC) peakBw *= 1.07;
    }
  }
#endif

  double bw = softmin(nBlocks * smBw * busMultiplier, peakBw, smBw);
  estimate->ctaSelectionTimeUs = baseLat + softplus(busBytes / bw - 1, 1);
  estimate->timeUs = static_cast<float>(estimate->ctaSelectionTimeUs);
  estimate->selectionTimeUs = estimate->timeUs;
  return std::isfinite(estimate->timeUs) && estimate->timeUs > 0.0f;
}
