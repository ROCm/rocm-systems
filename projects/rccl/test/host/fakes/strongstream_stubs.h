/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams strongstream_stubs.cc OWNS (src/misc/strongstream.cc), declared once so a
// signature change is a compile error rather than a link mismatch.

#ifndef RCCL_TEST_HOST_STRONGSTREAM_STUBS_H_
#define RCCL_TEST_HOST_STRONGSTREAM_STUBS_H_

#include <functional>

#include "nccl.h"
#include "strongstream.h"

extern std::function<ncclResult_t(struct ncclCudaGraph*, hipStream_t, int)> g_cudaGetCapturingGraph;

// Fail loud by default like the rest of this floor; the addon capture cases in
// enqueue-test.cc install hooks.
extern std::function<ncclResult_t(struct ncclCudaGraph, hipStream_t, hipEvent_t)> g_ncclStreamAdvanceToEvent;
extern std::function<ncclResult_t(struct ncclCudaGraph, hipEvent_t, hipStream_t)> g_ncclCudaGraphRecordEvent;

extern ncclResult_t g_ncclStrongStreamResult;

// init.cc:778, gated on NCCL_LAUNCH_ORDER_IMPLICIT.
extern ncclResult_t g_ncclCudaContextTrackResult;
extern int g_ncclCudaContextTrackCalls;

void ResetStrongStreamStubs();

#endif  // RCCL_TEST_HOST_STRONGSTREAM_STUBS_H_
