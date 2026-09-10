/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams strongstream_stubs.cc OWNS (src/misc/strongstream.cc), declared once so a
// signature change is a compile error rather than a link mismatch.

#ifndef RCCL_TEST_HOST_STRONGSTREAM_STUBS_H_
#define RCCL_TEST_HOST_STRONGSTREAM_STUBS_H_

#include "nccl.h"

extern ncclResult_t g_ncclStrongStreamResult;

// init.cc:778, gated on NCCL_LAUNCH_ORDER_IMPLICIT.
extern ncclResult_t g_ncclCudaContextTrackResult;
extern int g_ncclCudaContextTrackCalls;

void ResetStrongStreamStubs();

#endif  // RCCL_TEST_HOST_STRONGSTREAM_STUBS_H_
