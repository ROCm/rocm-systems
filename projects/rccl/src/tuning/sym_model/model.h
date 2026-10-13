/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_SYM_MODEL_MODEL_H_
#define NCCL_INT_SYM_MODEL_MODEL_H_

#include "sym_kernels.h"
#include "tuning.h"

#include <cstddef>

int ncclSymkModelCtasEnvOverride();

struct ncclSymkLsaEstimate {
  double ctaSelectionTimeUs;
  float timeUs;
  float selectionTimeUs;
};

ncclResult_t ncclSymkGinModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes,
                              float* timeUs, int* nBlocks);
ncclResult_t ncclSymkLsaModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes,
                              float* timeUs, float* selectionTimeUs, int* nBlocks);
bool ncclSymkLsaBaseModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes, int nBlocks,
                          struct ncclSymkLsaEstimate* estimate);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
// CTA search tolerance for this kernel under the selected tuning model.
double rcclSymkLsaWithinPeakFactor(enum ncclSymkKernelId kernelId);

// The NaN AllReduce kernels' choice of shot, made the same way in nan.cuh.
inline bool rcclSymkNanTwoShot(enum ncclSymkKernelId kernelId, size_t nBytes) {
  return (kernelId == ncclSymkKernelId_AllReduce_AGxNaN_R || kernelId == ncclSymkKernelId_AllReduce_AGxTmaNaN_R) &&
         nBytes > ncclSymkNanTwoShotMinBytes;
}
inline bool rcclSymkNanTdm(enum ncclSymkKernelId kernelId) {
  return kernelId == ncclSymkKernelId_AllGather_TmaNaN || kernelId == ncclSymkKernelId_AllReduce_AGxTmaNaN_R ||
         kernelId == ncclSymkKernelId_ReduceScatter_TmaNaN;
}
#endif

#endif // NCCL_INT_SYM_MODEL_MODEL_H_
