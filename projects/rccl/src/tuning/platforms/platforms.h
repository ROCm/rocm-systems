/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0 and BSD-3
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_TUNING_PLATFORMS_H_
#define NCCL_TUNING_PLATFORMS_H_

#include "cudawrap.h"
#include "tuning.h"
#include "../cost_model.h"
#include "rubin.h"

// [RCCL] RUBIN_AND_LATER() is an NVIDIA compute-capability test that also matches AMD gfx11xx/gfx1250 values
// (e.g. gfx1250 -> 125); there is no Rubin platform tuning on AMD.
static inline bool ncclTuningIsRubinPlatform(int compCap) {
#if defined(__HIP_PLATFORM_AMD__)
  (void)compCap;
  return false;
#else
  return RUBIN_AND_LATER(compCap);
#endif
}

static ncclResult_t ncclTuningGetModelPlatformEntry(int compCap, int id, struct ncclTuningModelEntry_t** entry) {
  if (ncclTuningIsRubinPlatform(compCap)) {
    NCCLCHECK(ncclTuningRubinGetModelEntry(compCap, id, entry));
  }
  return ncclSuccess;
}

static ncclResult_t ncclTuningPlatformPreInit(struct ncclComm* comm) {
  if (ncclTuningIsRubinPlatform(comm->minCompCap)) {
    ncclTuningRubinPreInit(comm);
  }
  return ncclSuccess;
}

static ncclResult_t ncclTuningPlatformChannelTuning(struct ncclTuningInput_t* const input,
                                                    const struct ncclTuningResult_t* result, int* maxChannels) {
  if (ncclTuningIsRubinPlatform(input->comm->minCompCap)) {
    NCCLCHECK(ncclTuningRubinChannelTuning(input, result, maxChannels));
  }
  return ncclSuccess;
}

#endif // NCCL_TUNING_PLATFORMS_H_
